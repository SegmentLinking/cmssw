// MkFitAlpakaFitDeviceProducer: the mkFit final fit on the device, in place of MkFitFitProducer. It applies the
// candCutSel preselection to the TrackSoA of MkFitAlpakaBuildProducer and fits the selected rows; produce() only
// enqueues device work (no host synchronization, no CMSSW thread held), as the build module.
// - Inputs: the device TrackSoA of the building, the device EventOfHits, the pixel hit map (its size only) and, with
//   the CPE, the pixel clusters (plus the pixel digi and cluster SoA when the cluster quantities come from the device).
// - Output: the fitted TrackSoA (same capacity; rows [0, nTracks) in MkFitCore order, outliers marked with
//   Hit::fitOutlierIdx, failed fits with chi2 = -1, the state at the outermost fitted hit in `outer`) and the status.
// - CPE cluster quantities (clusterSource): "device" computes them from the pixel digi and cluster SoA, the host only
//   writes per mkFit pixel row the SoA module and cluster id (clusters without originalId make the event fall back to
//   "host"); "tracks" (CPU backends) computes on the host only those of the hits on the selected tracks; "host"
//   computes all of them on the host. "auto" is "device" on GPU backends and "tracks" on CPU backends.

#include <algorithm>
#include <optional>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/Backend.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusCollect.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/tracks/TrackSoADeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeESData.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsDeviceCollections.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/FitHandoff.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/fit/FitTracks.h"

#include "MkFitAlpakaReleaseOrder.h"

namespace mkfitdev::cpe {

  // PixelCPEGenericBase::collect_edge_charges without truncation (menu TruncatePixelCharge = False) + charge.
  inline ClusterCpe clusterCpe(SiPixelCluster const& cl, int module) {
    ClusterCpe cpe{};
    cpe.module = module;
    cpe.edges.minRow = cl.minPixelRow();
    cpe.edges.maxRow = cl.maxPixelRow();
    cpe.edges.minCol = cl.minPixelCol();
    cpe.edges.maxCol = cl.maxPixelCol();
    for (int k = 0; k < cl.size(); ++k) {
      const auto px = cl.pixel(k);
      const int row = px.x, col = px.y, adc = px.adc;
      if (row == cpe.edges.minRow)
        cpe.edges.qfX += adc;
      if (row == cpe.edges.maxRow)
        cpe.edges.qlX += adc;
      if (col == cpe.edges.minCol)
        cpe.edges.qfY += adc;
      if (col == cpe.edges.maxCol)
        cpe.edges.qlY += adc;
    }
    cpe.charge = cl.charge();
    return cpe;
  }

  // clusterCpe() per mkFit pixel row (= legacy cluster key, as convertHits) with need[row] != 0 (every row when need is
  // empty); needed rows without a cluster get module -1 (no CPE), the other rows are untouched
  inline void fillClusterCpe(edmNew::DetSetVector<SiPixelCluster> const& dsv,
                             CpeTablesHost const& tables,
                             uint32_t nPix,
                             ClusterCpe* out,
                             std::vector<uint8_t> const& need = {}) {
    const bool all = need.empty();
    auto const& data = dsv.data();
    const SiPixelCluster* base = data.data();
    for (auto const& ds : dsv) {
      int mod = -2;  // looked up on the first filled cluster of the module
      for (auto const& cl : ds) {
        const uint32_t key = &cl - base;
        if (key >= nPix || !(all || need[key]))
          continue;
        if (mod == -2) {
          auto it = tables.rawToModule.find(ds.detId());
          mod = it == tables.rawToModule.end() ? -1 : it->second;
        }
        out[key] = clusterCpe(cl, mod);
      }
    }
    for (uint32_t i = data.size(); i < nPix; ++i)
      if (all || need[i]) {
        out[i] = ClusterCpe{};
        out[i].module = -1;
      }
  }

  // Device mode: per mkFit pixel row the SoA module index and the SoA cluster id (SiPixelCluster::originalId).
  // Returns false if a cluster has no originalId (persisted clusters): the caller falls back to the host computation.
  inline bool fillClusterRefs(edmNew::DetSetVector<SiPixelCluster> const& dsv,
                              CpeTablesHost const& tables,
                              uint32_t nPix,
                              ::mkfitdev::handoff::ClusterRef* out) {
    auto const& data = dsv.data();
    const SiPixelCluster* base = data.data();
    for (auto const& ds : dsv) {
      auto it = tables.rawToModule.find(ds.detId());
      const int mod = it == tables.rawToModule.end() ? -1 : it->second;
      for (auto const& cl : ds) {
        const uint32_t key = &cl - base;
        if (cl.originalId() == SiPixelCluster::invalidClusterId)
          return false;
        if (key < nPix)
          out[key] = ::mkfitdev::handoff::ClusterRef{mod, int32_t(cl.originalId())};
      }
    }
    for (uint32_t i = data.size(); i < nPix; ++i)
      out[i] = ::mkfitdev::handoff::ClusterRef{-1, 0};
    return true;
  }

}  // namespace mkfitdev::cpe

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkFitAlpakaFitDeviceProducer : public global::EDProducer<> {
  public:
    explicit MkFitAlpakaFitDeviceProducer(edm::ParameterSet const& iConfig)
        : EDProducer<>(iConfig),
          tracksToken_{consumes(iConfig.getParameter<edm::InputTag>("tracks"))},
          eohToken_{consumes(iConfig.getParameter<edm::InputTag>("eventOfHits"))},
          pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
          esToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("esData"))},
          putToken_{produces()},
          statusToken_{produces()},
          useCpe_{iConfig.getParameter<bool>("cpe")} {
      if (auto const& queueTag = iConfig.getParameter<edm::InputTag>("eventOfHitsQueue"); !queueTag.label().empty())
        eohQueueToken_ = consumes(queueTag);
      if (auto const& queueTag = iConfig.getParameter<edm::InputTag>("tracksQueue"); !queueTag.label().empty())
        tracksQueueToken_ = consumes(queueTag);
      sel_.enabled = iConfig.getParameter<bool>("candCutSel");
      sel_.minPt = float(iConfig.getParameter<double>("candMinPtCut"));
      sel_.minNHits = iConfig.getParameter<int>("candMinNHitsCut");
      sel_.minPtRelaxed = float(iConfig.getParameter<double>("candMinPtRelaxedCut"));
      sel_.minAbsEtaRelaxed = float(iConfig.getParameter<double>("candMinAbsEtaForRelaxedCut"));
      if (useCpe_) {
        clusterCollToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelClusters"));
        const auto src = iConfig.getParameter<std::string>("clusterSource");
        if (src != "auto" && src != "host" && src != "device" && src != "tracks")
          throw cms::Exception("Configuration")
              << "MkFitAlpakaFitDeviceProducer: clusterSource must be auto, device, tracks or host";
        if (src == "tracks" && !mkfitdev::kCpuBackend)
          throw cms::Exception("Configuration")
              << "MkFitAlpakaFitDeviceProducer: clusterSource tracks is for CPU backends";
        clusTracks_ = src == "tracks" || (src == "auto" && mkfitdev::kCpuBackend);
        clusDevice_ = src == "device" || (src == "auto" && !mkfitdev::kCpuBackend);
        if (clusDevice_) {
          digisToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelClustersSoA"));
          clustersSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelClustersSoA"));
        }
        cpeToken_ = esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("cpeTables")));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracks", edm::InputTag("hltInitialStepTrackCandidatesMkFitDevice"))
          ->setComment("device TrackSoA of the building (MkFitAlpakaBuildProducer)");
      desc.add<edm::InputTag>("eventOfHits", edm::InputTag("hltMkFitEventOfHits"))
          ->setComment("device EventOfHits product (hits block, same rows as the hit wrappers)");
      desc.add<edm::InputTag>("eventOfHitsQueue", edm::InputTag(""))
          ->setComment("the EventOfHits allocation queue (instance 'queue'), set when it is deleted early");
      desc.add<edm::InputTag>("tracksQueue", edm::InputTag(""))
          ->setComment("the building TrackSoA allocation queue (instance 'queue'), set when it is deleted early");
      desc.add<edm::InputTag>("pixelHits", edm::InputTag("hltMkFitSiPixelHits"))
          ->setComment("MkFitClusterIndexToHit of the pixel hits: only its size (strip hit row base)");
      desc.add<edm::ESInputTag>("esData", edm::ESInputTag("", ""));
      desc.add<bool>("candCutSel", false);
      desc.add<double>("candMinPtCut", 0);
      desc.add<int>("candMinNHitsCut", 0);
      desc.add<double>("candMinPtRelaxedCut", 0);
      desc.add<double>("candMinAbsEtaForRelaxedCut", 0);
      desc.add<bool>("cpe", true)
          ->setComment("device PixelCPEGeneric with track angles (as MkFitFitProducer); false = no-CPE");
      desc.add<std::string>("cpeTables", "MkFitAlpakaFitCpe");
      desc.add<edm::InputTag>("pixelClusters", edm::InputTag("hltSiPixelClusters"))
          ->setComment("the clusters of the pixel rechits (mkFit pixel hit row = cluster key)");
      desc.add<std::string>("clusterSource", "auto")
          ->setComment(
              "CPE cluster quantities: auto (device on GPU, tracks on CPU) | device (from the pixel digi/cluster "
              "SoA) | tracks (CPU backends: host, hits on the selected tracks only) | host (all hits)");
      desc.add<edm::InputTag>("pixelClustersSoA", edm::InputTag("hltPhase2SiPixelClustersSoA"))
          ->setComment(
              "producer of the pixel digi + cluster SoA (SiPixelDigisSoACollection, SiPixelClustersSoACollection)");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      auto const& inD = iEvent.get(tracksToken_);  // first: take over the building's queue
      auto& queue = iEvent.queue();
      const int capacity = inD.const_view().metadata().size();
      mkfitdev::TrackSoADeviceCollection out(queue, capacity);
      // per-event status (fit overflow / hit-count mismatch, CPE cluster consistency)
      mkfitdev::MkFitStatusDeviceObject status(queue);
      mkfitdev::zeroStatus(queue, status);
      if (capacity == 0) {  // skipped or seedless event; zero-filled scalars
        auto buf = out.buffer();
        alpaka::memset(queue, buf, 0x00);
        iEvent.emplace(putToken_, std::move(out));
        iEvent.emplace(statusToken_, std::move(status));
        return;
      }
      auto const& es = iSetup.getData(esToken_);
      ::mkfitdev::checkSupportedFitConfig(es.hostConfigValue());
      const uint32_t nPix = iEvent.get(pixelHitsToken_).hits().size();

      // 1. candCutSel on the device, order-preserving; out.nTracks() = kept count (device)
      mkfitdev::handoff::selectFitInput(queue, inD.const_view(), capacity, sel_, out.view());

      // 2. CPE cluster quantities (clusterSource): device kernels from the pixel digi/cluster SoA (GPU), host values for
      //    the hits on the selected tracks (CPU backends), or host values for every pixel hit (host / fallback)
      std::optional<cms::alpakatools::host_buffer<::mkfitdev::cpe::ClusterCpe[]>> hClus;
      std::optional<cms::alpakatools::device_buffer<Device, ::mkfitdev::cpe::ClusterCpe[]>> dClus;
      const ::mkfitdev::cpe::ClusterCpe* clusPtr = nullptr;
      ::mkfitdev::cpe::CpeTables dCpe{};
      std::optional<cms::alpakatools::device_buffer<Device, ::mkfitdev::handoff::ClusterCpeCounters>> dCnt;
      if (useCpe_) {
        auto const& cpeES = iSetup.getData(cpeToken_);
        auto const& tables = *cpeES.host;
        dCpe = cpeES.view();
        auto const& dsv = iEvent.get(clusterCollToken_);
        const uint32_t nAlloc = std::max<uint32_t>(nPix, 1);
        bool onDevice = clusDevice_;
        if (clusTracks_) {  // CPU backends: the selection above has completed (blocking queue); hits on its tracks only
          alpaka::wait(queue);  // no-op for the blocking serial queue, keeps the contract explicit
          hClus.emplace(cms::alpakatools::make_host_buffer<::mkfitdev::cpe::ClusterCpe[]>(queue, nAlloc));
          fillClusterCpeOnTracks(dsv, tables, es.view(), out.const_view(), nPix, hClus->data());
        }
        std::optional<cms::alpakatools::host_buffer<::mkfitdev::handoff::ClusterRef[]>> hRefs;
        if (onDevice) {
          hRefs.emplace(cms::alpakatools::make_host_buffer<::mkfitdev::handoff::ClusterRef[]>(queue, nAlloc));
          onDevice = ::mkfitdev::cpe::fillClusterRefs(dsv, tables, nPix, hRefs->data());
        }
        if (!onDevice && !clusTracks_) {
          hClus.emplace(cms::alpakatools::make_host_buffer<::mkfitdev::cpe::ClusterCpe[]>(queue, nAlloc));
          ::mkfitdev::cpe::fillClusterCpe(dsv, tables, nPix, hClus->data());
        }
        if (onDevice) {
          auto const& digis = iEvent.get(digisToken_);
          auto const& clus = iEvent.get(clustersSoAToken_);
          dClus.emplace(cms::alpakatools::make_device_buffer<::mkfitdev::cpe::ClusterCpe[]>(queue, nAlloc));
          auto dRefs = cms::alpakatools::make_device_buffer<::mkfitdev::handoff::ClusterRef[]>(queue, nAlloc);
          alpaka::memcpy(queue, dRefs, *hRefs);
          dCnt.emplace(cms::alpakatools::make_device_buffer<::mkfitdev::handoff::ClusterCpeCounters>(queue));
          alpaka::memset(queue, *dCnt, 0);
          mkfitdev::handoff::buildClusterCpe(queue,
                                             digis.const_view(),
                                             digis.nDigis(),
                                             clus.const_view(),
                                             clus.nClusters(),
                                             dRefs.data(),
                                             nPix,
                                             dClus->data(),
                                             dCnt->data());
          clusPtr = dClus->data();
        } else if constexpr (mkfitdev::kCpuBackend) {
          clusPtr = hClus->data();
        } else {
          dClus.emplace(cms::alpakatools::make_device_buffer<::mkfitdev::cpe::ClusterCpe[]>(queue, nAlloc));
          alpaka::memcpy(queue, *dClus, *hClus);
          clusPtr = dClus->data();
        }
      }

      // 3. the fit, in place on the selected rows; row count from the device
      auto dCounters = cms::alpakatools::make_device_buffer<mkfitdev::fit::FitCounters>(queue);
      alpaka::memset(queue, dCounters, 0);
      const auto eohHits = iEvent.get(eohToken_).const_view().hits();
      // The EventOfHits hit rows are the pixel rows of the pixel hit map, then the strip rows: nPix is the strip row
      // base of the track hit indices.
      if (uint32_t(eohHits.metadata().size()) < nPix)
        throw cms::Exception("Configuration") << "MkFitAlpakaFitDeviceProducer: the EventOfHits has "
                                              << eohHits.metadata().size() << " hit rows, fewer than the " << nPix
                                              << " pixel rows of pixelHits; both must come from the same pixel rechits";
      mkfitdev::fit::runFinalFit(queue,
                                 es.view(),
                                 eohHits,
                                 nPix,
                                 out.view(),
                                 capacity,
                                 dCounters.data(),
                                 useCpe_,
                                 dCpe,
                                 clusPtr,
                                 &out.view().nTracks());
      {
        mkfitdev::StatusSources src;
        src.add(&dCounters.data()->nOverflow, ::mkfitdev::kFitOverflow);
        src.add(&dCounters.data()->nHitCountMismatch, ::mkfitdev::kFitHitCountMismatch);
        if (dCnt) {
          src.add(&dCnt->data()->nClusterOverflow, ::mkfitdev::kCpeClusterOverflow);
          src.add(&dCnt->data()->nRefOutOfRange, ::mkfitdev::kCpeClusterRefErrors);
        }
        mkfitdev::collectStatus(queue, status, src);
      }
      if (!eohQueueToken_.isUninitialized())
        mkfitdev::orderReleaseAfterReads(queue, iEvent.get(eohQueueToken_));
      if (!tracksQueueToken_.isUninitialized())
        mkfitdev::orderReleaseAfterReads(queue, iEvent.get(tracksQueueToken_));
      iEvent.emplace(putToken_, std::move(out));
      iEvent.emplace(statusToken_, std::move(status));
    }

  private:
    // CPU backends: clusterCpe for the pixel hits on the selected tracks, other rows untouched
    static void fillClusterCpeOnTracks(edmNew::DetSetVector<SiPixelCluster> const& dsv,
                                       ::mkfitdev::cpe::CpeTablesHost const& tables,
                                       ::mkfitdev::ESView const& es,
                                       ::mkfitdev::TrackSoAConstView trk,
                                       uint32_t nPix,
                                       ::mkfitdev::cpe::ClusterCpe* out) {
      std::vector<uint8_t> need(nPix, 0);
      for (int itrack = 0; itrack < trk.nTracks(); ++itrack) {
        const int nh = std::min<int>(trk[itrack].nTotalHits(), ::mkfitdev::kMaxTrkHits);
        for (int ihit = 0; ihit < nh; ++ihit) {
          const auto hot = trk[itrack].hits().hot[ihit];
          if (hot.index >= 0 && uint32_t(hot.index) < nPix && es.layers[hot.layer].is_pixel())
            need[hot.index] = 1;
        }
      }
      ::mkfitdev::cpe::fillClusterCpe(dsv, tables, nPix, out, need);
    }

    const device::EDGetToken<mkfitdev::TrackSoADeviceCollection> tracksToken_;
    const device::EDGetToken<mkfitdev::EventOfHitsDeviceCollection> eohToken_;
    edm::EDGetTokenT<unsigned long long> eohQueueToken_, tracksQueueToken_;
    const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelHitsToken_;
    const device::ESGetToken<::mkfitdev::ESData<Device>, TrackerRecoGeometryRecord> esToken_;
    const device::EDPutToken<mkfitdev::TrackSoADeviceCollection> putToken_;
    const device::EDPutToken<mkfitdev::MkFitStatusDeviceObject> statusToken_;
    const bool useCpe_;
    ::mkfitdev::handoff::CandCutSel sel_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> clusterCollToken_;
    device::ESGetToken<::mkfitdev::cpe::CpeESData<Device>, PixelCPEFastParamsRecord> cpeToken_;
    bool clusDevice_ = false, clusTracks_ = false;
    device::EDGetToken<SiPixelDigisSoACollection> digisToken_;
    device::EDGetToken<SiPixelClustersSoACollection> clustersSoAToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaFitDeviceProducer);
