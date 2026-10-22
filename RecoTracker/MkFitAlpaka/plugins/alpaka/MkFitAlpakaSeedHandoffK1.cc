// K1: the device seed hand-off. The LST track candidates stay on the device and become the mkFit seed rows of the
// build module (deviceSeeds) without the TC copy to the host. The seeds equal the light LSTOutputConverter +
// MkFitSeedConverter ones, hit order as DEVIATIONS DEV-8 (MkFitAlpakaSeedHandoffK1Kernels.h).
// acquire() runs K1 and copies back one small row record per TC plus the host requests (the dropOTHitsPurePLS creator
// refits, placeholder states: a few per event); produce() makes those states with the CMSSW objects of
// LSTOutputConverter, numbers the kept rows in TC order and compacts them on the device.
// Products: the seed rows (TrackSoADeviceCollection, label = row) and pixelTrackOfSeed (host, one entry per seed).
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2TrackerRecHitOnDemand.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/TrackProduct.h"
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MkFitAlpakaSeedHandoffK1Kernels.h"

namespace mkfitdev::k1 {
  // the det table by (mkFit layer, short id): built once per geometry for the job (the streams copy it to their device)
  struct HostTable {
    std::vector<DetInfo> dets;
    std::vector<int32_t> base;  // per mkFit layer: first row; base[nLayers] = rows
    std::vector<int8_t> isPixel;
    int32_t nLayers = 0;
  };
  struct HostTableCache {
    std::shared_ptr<const HostTable> get(TrackerGeometry const& geom,
                                         TrackerTopology const& topo,
                                         MkFitGeometry const& mkg) const {
      std::lock_guard<std::mutex> lk(mutex);
      if (key == &mkg && table)
        return table;
      auto t = std::make_shared<HostTable>();
      auto const& ti = mkg.trackerInfo();
      const int nL = ti.n_layers();
      t->nLayers = nL;
      t->base.assign(nL + 1, 0);
      for (int l = 0; l < nL; ++l)
        t->base[l + 1] = t->base[l] + ti[l].n_modules();
      t->dets.assign(std::max(t->base[nL], 1), DetInfo{0.f, -1, -1, 0, 0, 0, 0, 0});
      t->isPixel.assign(std::max(nL, 1), 0);
      std::unordered_map<uint32_t, GeomDet const*> detById;
      for (auto const* d : geom.dets())
        detById.emplace(d->geographicalId().rawId(), d);
      // the module type of every sensor from its GeometricDet name (TrackerGeometry::moduleType), as fillTestMap stores it.
      // getDetectorType scans fillTestMap's run-length list, which is built over the DetId-sorted sensors, so for a
      // sensor it returns exactly this type; the scan cost 76 ms per job for 41.6k modules
      std::unordered_map<uint32_t, int8_t> typeById;
      if (auto const* root = geom.trackerDet()) {
        for (auto const* g : root->deepComponents()) {
          std::string const& full = g->name();
          const auto mt = geom.moduleType(full.substr(full.find(':') + 1));
          typeById.emplace(
              g->geographicalId().rawId(),
              mt == TrackerGeometry::ModuleType::Ph2PSP ? 1 : (mt == TrackerGeometry::ModuleType::Ph2PSS ? 2 : 0));
        }
      }
      for (int l = 0; l < nL; ++l) {
        t->isPixel[l] = ti[l].is_pixel();
        for (int s = 0; s < ti[l].n_modules(); ++s) {
          const DetId id(ti[l].module_info(s).detid);
          auto const di = detById.find(id.rawId());  // mkFit layers may list ids without a det: no hit sits there
          if (di == detById.end())
            continue;
          GeomDet const* d = di->second;
          DetInfo& e = t->dets[t->base[l] + s];
          const auto sub = d->subDetector();
          e.sub = int16_t(sub);
          e.isOT = GeomDetEnumerators::isOuterTracker(sub);
          auto const& sf = d->surface();
          e.sortKey = GeomDetEnumerators::isBarrel(sub) ? sf.rSpan().first : std::abs(sf.zSpan().first);
          e.topoLayer = int32_t(topo.layer(id));
          e.barrel = GeomDetEnumerators::isBarrel(sub);
          if (e.isOT)  // the LST logical layer: barrel layers 1-6, endcap disks 7-11
            e.slot = int8_t(e.barrel ? e.topoLayer : 6 + e.topoLayer);
          if (auto const ti2 = typeById.find(id.rawId()); ti2 != typeById.end()) {
            e.type = ti2->second;
          } else if (d->isLeaf()) {  // a sensor that is not a GeometricDet leaf: TrackerGeometry's scan
            const auto mt = geom.getDetectorType(id);
            e.type =
                mt == TrackerGeometry::ModuleType::Ph2PSP ? 1 : (mt == TrackerGeometry::ModuleType::Ph2PSS ? 2 : 0);
          }  // else a composite (stack) det: no hit sits on it, its type is never read
        }
      }
      table = std::move(t);
      key = &mkg;
      return table;
    }
    mutable std::mutex mutex;
    mutable void const* key = nullptr;
    mutable std::shared_ptr<const HostTable> table;
  };
}  // namespace mkfitdev::k1

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::mkfitdev::k1;

  class MkFitAlpakaSeedHandoffK1
      : public stream::SynchronizingEDProducer<edm::GlobalCache<::mkfitdev::k1::HostTableCache>> {
  public:
    static std::unique_ptr<::mkfitdev::k1::HostTableCache> initializeGlobalCache(edm::ParameterSet const&) {
      return std::make_unique<::mkfitdev::k1::HostTableCache>();
    }
    static void globalEndJob(::mkfitdev::k1::HostTableCache const*) {}

    MkFitAlpakaSeedHandoffK1(edm::ParameterSet const& cfg, ::mkfitdev::k1::HostTableCache const*)
        : SynchronizingEDProducer<edm::GlobalCache<::mkfitdev::k1::HostTableCache>>(cfg),
          lstToken_(consumes(cfg.getParameter<edm::InputTag>("lstOutput"))),
          statesToken_(consumes(cfg.getParameter<edm::InputTag>("pixelSeedStates"))),
          eohToken_(consumes(cfg.getParameter<edm::InputTag>("eventOfHits"))),
          pixelTracksToken_(consumes(cfg.getParameter<edm::InputTag>("pixelTracks"))),
          otClustersToken_(consumes(cfg.getParameter<edm::InputTag>("otClusters"))),
          otCpeToken_(esConsumes(cfg.getParameter<edm::ESInputTag>("Phase2StripCPE"))),
          mfToken_(esConsumes()),
          tGeomToken_(esConsumes()),
          tTopoToken_(esConsumes()),
          mkFitGeomToken_(esConsumes()),
          includeFourthHit_(cfg.getParameter<bool>("includeFourthHit")),
          dropOTHitsPurePLS_(cfg.getParameter<bool>("dropOTHitsPurePLS")),
          maxITHitsToDrop_(cfg.getParameter<int>("maxITHitsToDropOTHitsPurePLS")),
          deviceFitStates_(cfg.getParameter<bool>("deviceFitStates")),
          seedCreator_(SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator",
                                                         cfg.getParameter<edm::ParameterSet>("SeedCreatorPSet"),
                                                         consumesCollector())),
          seedsPutToken_(produces()),
          pixOfSeedPutToken_(produces("pixelTrackOfSeed")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("lstOutput", edm::InputTag("hltLST"));
      desc.add<edm::InputTag>("pixelSeedStates", edm::InputTag("hltInputLSTDevice", "pixelSeedStates"))
          ->setComment("device pLS states + pixel seed hit lists per pixel track (hltInputLSTDevice)");
      desc.add<edm::InputTag>("eventOfHits", edm::InputTag("hltMkFitEventOfHits"));
      desc.add<edm::InputTag>("pixelTracks", edm::InputTag("hltPhase2PixelTracks"));
      desc.add<edm::InputTag>("otClusters", edm::InputTag("hltSiPhase2Clusters"));
      desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"));
      desc.add<bool>("includeFourthHit", true);
      desc.add<bool>("dropOTHitsPurePLS", true);
      desc.add<int>("maxITHitsToDropOTHitsPurePLS", 3);
      desc.add<bool>("deviceFitStates", false);
      edm::ParameterSetDescription psd0;
      psd0.add<std::string>("ComponentName", std::string("SeedFromConsecutiveHitsCreator"));
      psd0.add<std::string>("propagator", std::string("PropagatorWithMaterial"));
      psd0.add<double>("SeedMomentumForBOFF", 5.0);
      psd0.add<double>("OriginTransverseErrorMultiplier", 1.0);
      psd0.add<double>("MinOneOverPtError", 1.0);
      psd0.add<std::string>("magneticField", std::string(""));
      psd0.add<std::string>("TTRHBuilder", std::string("WithTrackAngle"));
      psd0.add<bool>("forceKinematicWithRegionDirection", false);
      desc.add<edm::ParameterSetDescription>("SeedCreatorPSet", psd0);
      descriptions.addWithDefaultLabel(desc);
    }

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      auto& queue = iEvent.queue();
      auto const& tcs = iEvent.get(lstToken_);
      auto const& states = iEvent.get(statesToken_);
      auto const& eoh = iEvent.get(eohToken_);
      cap_ = uint32_t(tcs.const_view().metadata().size());
      DevTables const& T = tables(queue, iSetup);
      K1Config cfg{};
      cfg.nPixTracks = int32_t(iEvent.get(pixelTracksToken_).size());
      cfg.nStates = int32_t(states.const_view().metadata().size());
      cfg.nLayers = T.nLayers;
      cfg.includeFourthHit = includeFourthHit_;
      cfg.dropOTHitsPurePLS = dropOTHitsPurePLS_;
      cfg.maxITHitsToDrop = maxITHitsToDrop_;
      cfg.deviceFitStates = deviceFitStates_;
      const uint32_t c = std::max<uint32_t>(cap_, 1);
      rows_.emplace(queue, int32_t(c));
      dInfo_.emplace(cms::alpakatools::make_device_buffer<RowInfo[]>(queue, c));
      dReq_.emplace(cms::alpakatools::make_device_buffer<HostReq[]>(queue, kMaxHostReq));
      dCnt_.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, kNCounters + 2));
      alpaka::memset(queue, *dCnt_, 0);
      ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::k1::launchK1Rows(queue,
                                                               tcs.const_view(),
                                                               cap_,
                                                               states.const_view(),
                                                               eoh.const_view().hits(),
                                                               T.dets.data(),
                                                               T.layerBase.data(),
                                                               T.isPixel.data(),
                                                               cfg,
                                                               rows_->view(),
                                                               dInfo_->data(),
                                                               dReq_->data(),
                                                               dCnt_->data() + kNCounters,
                                                               dCnt_->data(),
                                                               dCnt_->data() + kNCounters + 1);
      hInfo_.emplace(cms::alpakatools::make_host_buffer<RowInfo[]>(queue, c));
      hReq_.emplace(cms::alpakatools::make_host_buffer<HostReq[]>(queue, kMaxHostReq));
      hCnt_.emplace(cms::alpakatools::make_host_buffer<int32_t[]>(queue, kNCounters + 2));
      alpaka::memcpy(queue, *hInfo_, *dInfo_);
      alpaka::memcpy(queue, *hReq_, *dReq_);
      alpaka::memcpy(queue, *hCnt_, *dCnt_);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto& queue = iEvent.queue();
      const int32_t* cnt = hCnt_->data();
      const uint32_t nTC = uint32_t(cnt[kNCounters + 1]);
      const int32_t nReqAll = cnt[kNCounters];
      if (cnt[0] > 0 || cnt[1] > 0)
        edm::LogWarning("MkFitAlpakaSeedHandoffK1")
            << "event " << iEvent.id().event() << ": " << cnt[0] << " host-request overflows, " << cnt[1]
            << " error rows (no seed: no pixel seed list / a hit without an mkFit row)";
      nReqOvf_ += cnt[0];
      nErr_ += cnt[1];
      RowInfo const* info = hInfo_->data();
      // host states for the requests (indexed by TC row)
      std::vector<int32_t> reqOfRow(nTC, -1);
      const int32_t nReq = std::min(nReqAll, kMaxHostReq);
      for (int32_t q = 0; q < nReq; ++q)
        if (uint32_t(hReq_->data()[q].tc) < nTC)
          reqOfRow[hReq_->data()[q].tc] = q;
      std::vector<HostState> made;
      std::vector<int32_t> madeOfReq(std::max(nReq, 1), -1);
      if (nReq > 0) {
        mkFitGeom_ = &iSetup.getData(mkFitGeomToken_);
        auto const& mf = iSetup.getData(mfToken_);
        auto const& geom = iSetup.getData(tGeomToken_);
        auto const& pixelTracks = iEvent.get(pixelTracksToken_);
        std::optional<Phase2TrackerRecHitOnDemand> otOnDemand;
        for (int32_t q = 0; q < nReq; ++q) {
          HostReq const& r = hReq_->data()[q];
          HostState hs{};
          bool ok = false;
          if (r.kind == kHostOTState) {
            if (!otOnDemand)
              otOnDemand.emplace(static_cast<edm::Event const&>(iEvent).getHandle(otClustersToken_),
                                 geom,
                                 iSetup.getData(otCpeToken_));
            const auto hit = otOnDemand->make(uint32_t(r.hot[0].index));
            ok = convert(placeholder(hit.localPosition(), hit.geographicalId().rawId()), &hit.det()->surface(), mf, hs);
            ++nOTState_;
          } else if (r.e >= 0 && size_t(r.e) < pixelTracks.size()) {
            auto const hits = trackHits(pixelTracks[r.e], r, geom);
            if (int(hits.size()) == r.nHot) {
              if (r.kind == kHostPixState) {
                auto const* h = hits.back();
                ok =
                    convert(placeholder(h->localPosition(), h->geographicalId().rawId()), &h->det()->surface(), mf, hs);
                ++nPixState_;
              } else if (r.kind == kHostRefit) {
                std::vector<SeedingHitSet::ConstRecHitPointer> hitsForRefit;
                hitsForRefit.reserve(hits.size());
                for (auto const* h : hits)
                  hitsForRefit.emplace_back(dynamic_cast<SeedingHitSet::ConstRecHitPointer>(h));
                GlobalTrackingRegion region;
                seedCreator_->init(region, static_cast<edm::EventSetup const&>(iSetup), nullptr);
                TrajectorySeedCollection seeds;
                seedCreator_->makeSeed(seeds, hitsForRefit);
                ++nRefit_;
                if (!seeds.empty())
                  ok = convert(seeds[0].startingState(), &hits.back()->det()->surface(), mf, hs);
                else
                  ++nRefitFailed_;  // the seed is skipped (LSTOutputConverter read seeds[0] of an empty vector)
              }
            } else {
              ++nHostMiss_;
            }
          }
          if (ok) {
            madeOfReq[q] = int32_t(made.size());
            made.push_back(hs);
          }
        }
      }
      // kept rows in TC order: label = output row
      auto hOut = cms::alpakatools::make_host_buffer<int32_t[]>(queue, std::max<uint32_t>(nTC, 1));
      std::vector<int> pixOfSeed;
      pixOfSeed.reserve(nTC);
      int32_t n = 0, ovf = 0;
      std::vector<HostState> placed;
      for (uint32_t r = 0; r < nTC; ++r) {
        int32_t o = -1;
        const int kind = info[r].kind;
        if (kind == kDevice) {
          o = n++;
        } else if (kind == kHostRefit || kind == kHostPixState || kind == kHostOTState) {
          const int32_t q = reqOfRow[r];
          if (q >= 0 && madeOfReq[q] >= 0) {
            o = n++;
            HostState hs = made[madeOfReq[q]];
            hs.row = o;
            placed.push_back(hs);
          }
        }
        hOut.data()[r] = o;
        if (o >= 0) {
          pixOfSeed.push_back(info[r].pixTrack);
          ovf += info[r].nTot > ::mkfitdev::kMaxSeedHits;
        }
      }
      auto dOut = cms::alpakatools::make_device_buffer<int32_t[]>(queue, std::max<uint32_t>(nTC, 1));
      alpaka::memcpy(queue, dOut, hOut);
      const int32_t np = int32_t(placed.size());
      auto hStates = cms::alpakatools::make_host_buffer<HostState[]>(queue, std::max(np, 1));
      std::copy(placed.begin(), placed.end(), hStates.data());
      auto dStates = cms::alpakatools::make_device_buffer<HostState[]>(queue, std::max(np, 1));
      alpaka::memcpy(queue, dStates, hStates);
      mkfitdev::TrackSoADeviceCollection out(queue, std::max(n, 1));
      ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::k1::launchK1Compact(
          queue, rows_->const_view(), dOut.data(), nTC, dStates.data(), np, ovf, n, out.view());
      ++nEv_;
      nSeeds_ += n;
      nHostRows_ += np;
      iEvent.emplace(seedsPutToken_, std::move(out));
      iEvent.emplace(pixOfSeedPutToken_, std::move(pixOfSeed));
      rows_.reset();
      dInfo_.reset();
      dReq_.reset();
      dCnt_.reset();
    }

    void endStream() override {
      if (nEv_ == 0)
        return;
      edm::LogInfo("MkFitAlpakaSeedHandoffK1")
          << "events " << nEv_ << " seeds " << nSeeds_ << " host rows " << nHostRows_ << " (refits " << nRefit_
          << " failed " << nRefitFailed_ << ", pixel placeholders " << nPixState_ << ", OT placeholders " << nOTState_
          << ") host misses " << nHostMiss_ << " error rows " << nErr_ << " request overflows " << nReqOvf_;
    }

  private:
    struct DevTables {
      cms::alpakatools::device_buffer<Device, DetInfo[]> dets;
      cms::alpakatools::device_buffer<Device, int32_t[]> layerBase;
      cms::alpakatools::device_buffer<Device, int8_t[]> isPixel;
      int32_t nLayers;
      void const* key;
    };

    // the device copy of the job's det table, per device of this stream (rebuilt when the geometry changes)
    DevTables const& tables(Queue& queue, device::EventSetup const& iSetup) {
      auto const& mkg = iSetup.getData(mkFitGeomToken_);
      const auto devKey = alpaka::getNativeHandle(alpaka::getDev(queue));
      auto it = tables_.find(devKey);
      if (it != tables_.end() && it->second.key == &mkg)
        return it->second;
      auto const host = globalCache()->get(iSetup.getData(tGeomToken_), iSetup.getData(tTopoToken_), mkg);
      const int nL = host->nLayers;
      const int nd = int(host->dets.size());
      auto hDets = cms::alpakatools::make_host_buffer<DetInfo[]>(queue, nd);
      auto hBase = cms::alpakatools::make_host_buffer<int32_t[]>(queue, nL + 1);
      auto hPix = cms::alpakatools::make_host_buffer<int8_t[]>(queue, std::max(nL, 1));
      std::copy(host->dets.begin(), host->dets.end(), hDets.data());
      std::copy(host->base.begin(), host->base.end(), hBase.data());
      std::copy(host->isPixel.begin(), host->isPixel.end(), hPix.data());
      DevTables t{cms::alpakatools::make_device_buffer<DetInfo[]>(queue, nd),
                  cms::alpakatools::make_device_buffer<int32_t[]>(queue, nL + 1),
                  cms::alpakatools::make_device_buffer<int8_t[]>(queue, std::max(nL, 1)),
                  nL,
                  &mkg};
      alpaka::memcpy(queue, t.dets, hDets);
      alpaka::memcpy(queue, t.layerBase, hBase);
      alpaka::memcpy(queue, t.isPixel, hPix);
      tables_.erase(devKey);
      return tables_.emplace(devKey, std::move(t)).first->second;
    }

    // the pixel track's rechits for the request's hits (mkFit layer, cluster key), in the request's order
    std::vector<TrackingRecHit const*> trackHits(reco::Track const& trk,
                                                 HostReq const& r,
                                                 TrackerGeometry const& geom) const {
      std::vector<TrackingRecHit const*> out;
      for (int k = 0; k < r.nHot; ++k) {
        for (auto const* h : trk.recHits()) {
          if (!h->isValid())
            continue;
          auto const& b = static_cast<BaseTrackerRecHit const&>(*h);
          if (b.isMatched() || b.firstClusterRef().index() != uint32_t(r.hot[k].index))
            continue;
          const DetId id = h->geographicalId();
          if (mkFitLayer(id) != r.hot[k].layer)
            continue;
          out.push_back(h);
          break;
        }
      }
      return out;
    }
    int mkFitLayer(DetId id) const { return mkFitGeom_ ? mkFitGeom_->mkFitLayerNumber(id) : -1; }

    // LSTOutputConverter's placeholder state (unit momentum along the module normal at the hit, unit errors)
    static PTrajectoryStateOnDet placeholder(LocalPoint const& lp, uint32_t detId) {
      const LocalTrajectoryParameters ltp(1.f, 0.f, 0.f, lp.x(), lp.y(), 1.f);
      float errs[15] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f};
      return PTrajectoryStateOnDet(ltp, 1.f, errs, detId, 0);
    }
    // MkFitSeedConverter::convertSeeds for one state
    static bool convert(PTrajectoryStateOnDet const& pst,
                        Surface const* surface,
                        MagneticField const& mf,
                        HostState& hs) {
      const auto tsos = trajectoryStateTransform::transientState(pst, surface, &mf);
      const auto& g = tsos.globalParameters();
      mkfit::SVector3 pos(g.position().x(), g.position().y(), g.position().z());
      mkfit::SVector3 mom(g.momentum().x(), g.momentum().y(), g.momentum().z());
      const auto& cov = tsos.curvilinearError().matrix();
      mkfit::SMatrixSym66 err;
      for (int i = 0; i < 5; ++i)
        for (int j = i; j < 5; ++j)
          err.At(i, j) = cov[i][j];
      mkfit::TrackState state(tsos.charge(), pos, mom, err);
      state.convertFromGlbCurvilinearToCCS();
      for (int k = 0; k < 6; ++k)
        hs.par[k] = state.parameters[k];
      std::memcpy(hs.err, state.errors.Array(), sizeof(float) * 21);
      hs.charge = int16_t(state.charge);
      return true;
    }

  private:
    const device::EDGetToken<lst::TrackCandidatesBaseDeviceCollection> lstToken_;
    const device::EDGetToken<mkfitdev::TrackSoADeviceCollection> statesToken_;
    const device::EDGetToken<mkfitdev::EventOfHitsDeviceCollection> eohToken_;
    const edm::EDGetTokenT<reco::TrackCollection> pixelTracksToken_;
    const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otClustersToken_;
    const edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> otCpeToken_;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
    const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
    const bool includeFourthHit_;
    const bool dropOTHitsPurePLS_;
    const int maxITHitsToDrop_;
    const bool deviceFitStates_;
    std::unique_ptr<SeedCreator> seedCreator_;
    const device::EDPutToken<mkfitdev::TrackSoADeviceCollection> seedsPutToken_;
    const edm::EDPutTokenT<std::vector<int>> pixOfSeedPutToken_;

    std::map<decltype(alpaka::getNativeHandle(std::declval<Device>())), DevTables> tables_;
    MkFitGeometry const* mkFitGeom_ = nullptr;
    // per event (acquire -> produce)
    uint32_t cap_ = 0;
    std::optional<mkfitdev::TrackSoADeviceCollection> rows_;
    std::optional<cms::alpakatools::device_buffer<Device, RowInfo[]>> dInfo_;
    std::optional<cms::alpakatools::device_buffer<Device, HostReq[]>> dReq_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> dCnt_;
    std::optional<cms::alpakatools::host_buffer<RowInfo[]>> hInfo_;
    std::optional<cms::alpakatools::host_buffer<HostReq[]>> hReq_;
    std::optional<cms::alpakatools::host_buffer<int32_t[]>> hCnt_;
    // counters (per stream, summarised at the end)
    long nEv_ = 0, nSeeds_ = 0, nHostRows_ = 0, nRefit_ = 0, nRefitFailed_ = 0, nPixState_ = 0, nOTState_ = 0,
         nHostMiss_ = 0, nErr_ = 0, nReqOvf_ = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaSeedHandoffK1);
