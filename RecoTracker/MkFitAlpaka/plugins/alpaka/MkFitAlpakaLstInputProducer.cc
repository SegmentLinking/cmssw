// LST's input device collection made on the device. Replaces hltInputLST (host packing of hltSiPhase2RecHits +
// the host KF-refit seeds hltInitialStepSeeds) for the LST producer in customizeHLTforMkFitAlpaka.
//   OT hits : straight from the device OT rechit SoA (MkFitAlpakaOTRecHitsProducer, columns gx/gy/gz/detId/clustSize;
//             row = cluster key = legacy rechit index); the CA-row -> key map from the producer's per-module first
//             key (otCAKeyStart). otSoAQueue: the SoA is deleted early (canDeleteEarly), this module orders its
//             allocation queue after its reads (MkFitAlpakaReleaseOrder.h).
//   pLS     : Patatrack's device pixel tracks; seedIdx = the edm pixel track index (pixelTracks' SoA->edm map).
// The pLS fields come from the device KF on the pixel-track hits (the host creator of hltInitialStepSeeds emulated +
// TSCBL; MkFitAlpaka fitPixelSeeds) and the device creator-failure rule (no seed -> no pLS); the device seed states
// and seed hits go to 'pixelSeedStates' for the build module and the seed hand-off. The KF capacity / lookup failures
// go to a per-event status product "status" (MkFitStatus: pixSeedTooManyHits / NoRow / NoLayer / NoEOH) and a
// LogWarning when non-zero.
#include <Eigen/Core>  // before any SoA header (Eigen columns of TracksSoA)
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusCollect.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/LstSeedFit.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/tracks/TrackSoADeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/math/TkBfield.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/alpaka/OTRecHitDeviceCollection.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "MkFitAlpakaLstInputKernels.h"
#include "MkFitAlpakaReleaseOrder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkFitAlpakaLstInputProducer : public stream::SynchronizingEDProducer<> {
    using HMS = std::vector<uint32_t>;
    using PLS = ::mkfitdev::lstin::PLS;
    static constexpr uint32_t kNoKey = ::mkfitdev::lstin::kNoKey;
    using LstInputDevice = ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTInputDeviceCollection;

  public:
    explicit MkFitAlpakaLstInputProducer(edm::ParameterSet const& cfg)
        : SynchronizingEDProducer<>(cfg),
          ptCut_(cfg.getParameter<double>("ptCut")),
          pseudoLastHit_(cfg.getParameter<int>("pseudoLastHit")),
          pcaAnchor_(cfg.getParameter<int>("pcaAnchor")),
          ptFieldCorrection_(cfg.getParameter<bool>("ptFieldCorrection")),
          pixToken_(consumes(cfg.getParameter<edm::InputTag>("pixelRecHits"))),
          otCluToken_(consumes(cfg.getParameter<edm::InputTag>("otClusters"))),
          pixCluToken_(consumes(cfg.getParameter<edm::InputTag>("pixelClusters"))),
          pixHMSToken_(consumes(cfg.getParameter<edm::InputTag>("pixelRecHits"))),
          otHMSToken_(consumes(cfg.getParameter<edm::InputTag>("otRecHitsSoA"))),
          pixelHitsSoAToken_(consumes(cfg.getParameter<edm::InputTag>("pixelRecHitSrc"))),
          otHitsSoAToken_(consumes(cfg.getParameter<edm::InputTag>("trackerRecHitsSoA"))),
          bsToken_(consumes(cfg.getParameter<edm::InputTag>("beamSpot"))),
          mfToken_(esConsumes()),
          outToken_(produces()) {
      otSoAToken_ = consumes(cfg.getParameter<edm::InputTag>("otSoA"));
      if (auto const q = cfg.getParameter<edm::InputTag>("otSoAQueue"); !q.label().empty())
        otSoAQueueToken_ = consumes(q);
      eohToken_ = consumes(cfg.getParameter<edm::InputTag>("eventOfHits"));
      esDataToken_ = esConsumes(cfg.getParameter<edm::ESInputTag>("esData"));
      kfCfg_.originPrior = ::mkfitdev::lstseeds::kHostCreator;
      kfCfg_.passes = cfg.getParameter<int>("pixKFPasses");
      kfCfg_.hlBackwardTol = cfg.getParameter<double>("pixKFBackwardTol");
      statesPutToken_ = produces("pixelSeedStates");
      statusPutToken_ = produces("status");
      tracksDevToken_ = consumes(cfg.getParameter<edm::InputTag>("pixelTracksSoA"));
      indToEdmToken_ = consumes(cfg.getParameter<edm::InputTag>("pixelTracks"));
      recoTracksToken_ = consumes(cfg.getParameter<edm::InputTag>("pixelTracks"));
      if (auto const t = cfg.getParameter<edm::InputTag>("pixelSrcRows"); !t.label().empty())
        srcRowsToken_ = consumes(t);
      keyStartToken_ = consumes(cfg.getParameter<edm::InputTag>("otCAKeyStart"));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<double>("ptCut", 0.8);
      desc.add<bool>("ptFieldCorrection", false)
          ->setComment("pLS momentum x <Bz>_hits / Bz(0) with the closed-form tracker field");
      desc.add<edm::InputTag>("pixelTracks", edm::InputTag("hltPhase2PixelTracks"))
          ->setComment("seedIdx = index in these edm pixel tracks (their SoA->edm map)");
      desc.add<int>("pseudoLastHit", 0)
          ->setComment(
              "the pLS last-hit pseudo-hit r3LH; 0 = the outermost hit, 1 = the"
              "Patatrack helix point at the outermost hit's transverse radius (KF-state-like, as the host seed)");
      desc.add<int>("pcaAnchor", 0)
          ->setComment(
              "the pLS PCA quantities (PCA point and momentum, dxy, dz, superbin); 0 = Patatrack's"
              "PCA, 1 = the PCA of the Patatrack helix re-anchored on the outermost hit (same curvature, "
              "tangent there), as the host takes the TSCBL of the KF seed state on the last hit");
      desc.add<edm::InputTag>("otSoA", edm::InputTag("hltMkFitAlpakaOTRecHits"))
          ->setComment("OT hits from this device OT rechit SoA (MkFitAlpakaOTRecHitsProducer)");
      desc.add<edm::InputTag>("otSoAQueue", edm::InputTag(""))
          ->setComment(
              "early deletion of otSoA: its producer's 'queue' product; on another queue this module orders that "
              "queue after its reads of the SoA (empty = off)");
      desc.add<int>("pixKFPasses", 1)
          ->setComment("field model of the device creator fit: 1 = B at the step start (host)");
      desc.add<double>("pixKFBackwardTol", 0.01)
          ->setComment("cm, 'against the momentum' tolerance (as lstSeedFitBackwardTol)");
      desc.add<edm::InputTag>("eventOfHits", edm::InputTag("hltMkFitEventOfHits"))
          ->setComment("device EventOfHits (hits)");
      desc.add<edm::ESInputTag>("esData", edm::ESInputTag("", "hltMkFitAlpakaES"))
          ->setComment("MkFitAlpaka ES product");
      desc.add<edm::InputTag>("pixelRecHits", edm::InputTag("hltSiPixelRecHits"));
      desc.add<edm::InputTag>("otClusters", edm::InputTag("hltSiPhase2Clusters"))
          ->setComment("the clusters of the OT rechits");
      desc.add<edm::InputTag>("pixelClusters", edm::InputTag("hltSiPixelClusters"))
          ->setComment("the clusters of the pixel rechits");
      desc.add<edm::InputTag>("otCAKeyStart", edm::InputTag("hltMkFitAlpakaOTRecHits", "caKeyStart"))
          ->setComment(
              "the first OT cluster key per P module from the producer of otRecHitsSoA: the CA-row -> key map");
      desc.add<edm::InputTag>("pixelSrcRows", edm::InputTag(""))
          ->setComment(
              "the device EventOfHits' pixel key -> SoA row map (producePixelSrcRows), inverted here"
              "instead of the host pass over the pixel rechits; empty tag or empty map: the host pass");
      desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("hltPhase2OtRecHitsSoA"));
      desc.add<edm::InputTag>("pixelTracksSoA", edm::InputTag("hltPhase2PixelTrackTorchHighPuritySelector"));
      desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("hltPhase2SiPixelRecHitsSoA"))
          ->setComment("the pixel rechits of the pixel tracks (the CA's pixelRecHitSrc)");
      desc.add<edm::InputTag>("trackerRecHitsSoA", edm::InputTag("hltPhase2OtRecHitsSoA"))
          ->setComment("the OT rechits of the pixel tracks (the CA's trackerRecHitsSoA)");
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    // per-IOV geometry: module table (per GeomDet index), detId -> OT-SoA module id (P sensors of TOB PS modules,
    // in detUnits order, as Phase2OTRecHitsSoAConverter / PixelTrackProducerFromSoAAlpaka)
    // acquire: host staging into pinned buffers, H2D copies, the per-track pLS kernels and the counts D2H; produce
    // (after the framework's synchronisation, no blocking wait on the host thread) allocates the exact-size collection
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      auto& queue = iEvent.queue();
      auto const& pixHits = iEvent.get(pixToken_);
      auto const& pixHMS = iEvent.get(pixHMSToken_);
      auto const& otHMS = iEvent.get(otHMSToken_);
      auto const& bs = iEvent.get(bsToken_);
      auto const& mf = iSetup.getData(mfToken_);
      // the hits the pixel-track hit indices refer to: the pixel rechits, then the OT rechits (the CA's indexing)
      ::mkfitdev::lstin::PixelTrackHits pixelTrackHits;
      for (auto const* hitsSoA : {&iEvent.get(pixelHitsSoAToken_), &iEvent.get(otHitsSoAToken_)}) {
        auto const view = hitsSoA->const_view().trackingHits();
        pixelTrackHits.addView(view, view.metadata().size());
      }
      auto const& otClu = iEvent.get(otCluToken_);
      auto const& pixClu = iEvent.get(pixCluToken_);

      // ---- host staging: OT-SoA row -> legacy OT index, pixel SoA row -> legacy cluster key (as
      //      PixelTrackProducerFromSoAAlpaka's hitmap, inverted)
      const uint32_t nOT = otClu.dataSize();
      otSoAView_ = iEvent.get(otSoAToken_).const_view();
      if (uint32_t(otSoAView_.metadata().size()) != nOT)
        throw cms::Exception("MkFitAlpakaLstInput")
            << "OT rechit SoA rows " << otSoAView_.metadata().size() << " != OT clusters " << nOT;
      const uint32_t nPixelSoA = otHMS.empty() ? 0 : otHMS[0];
      const uint32_t nOTSoA = otHMS.empty() ? 0 : otHMS.back() - nPixelSoA;
      if (uint32_t(pixelTrackHits.view(0).metadata().size()) != nPixelSoA ||
          uint32_t(pixelTrackHits.view(1).metadata().size()) != nOTSoA)
        throw cms::Exception("MkFitAlpakaLstInput")
            << "pixel / OT rechit SoA rows " << pixelTrackHits.view(0).metadata().size() << " / "
            << pixelTrackHits.view(1).metadata().size() << " != the OT module starts' " << nPixelSoA << " / " << nOTSoA;
      const uint32_t nOTSoAb = std::max<uint32_t>(nOTSoA, 1);
      const uint32_t nPixb = std::max<uint32_t>(nPixelSoA, 1);
      hOtKey_.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nOTSoAb));
      hPixKey_.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nPixb));
      uint32_t* otKey = hOtKey_->data();
      uint32_t* pixKey = hPixKey_->data();
      std::fill(otKey, otKey + nOTSoAb, kNoKey);
      std::fill(pixKey, pixKey + nPixb, kNoKey);
      {
        // the producer's per-P-module first key: rows hms[i] .. hms[i+1] - 1 <-> keys keyStart[i] + j
        auto const& keyStart = iEvent.get(keyStartToken_);
        if (keyStart.size() + 1 != otHMS.size())
          throw cms::Exception("MkFitAlpakaLstInput")
              << "caKeyStart " << keyStart.size() << " != CA OT modules " << otHMS.size() - 1;
        for (uint32_t i = 0; i < keyStart.size(); ++i) {
          const uint32_t row = otHMS[i] - nPixelSoA, size = otHMS[i + 1] - otHMS[i];
          for (uint32_t j = 0; j < size && row + j < nOTSoA; ++j)
            otKey[row + j] = keyStart[i] + j;
        }
      }
      // pixel SoA row -> cluster key: the inverse of the device EventOfHits' key -> row map when it is exact (the same
      // moduleStart[det] + originalId of the same rechits and clusters), else the host pass over the pixel rechits
      std::vector<uint32_t> const* srcRows = srcRowsToken_.isUninitialized() ? nullptr : &iEvent.get(srcRowsToken_);
      if (srcRows && !srcRows->empty()) {
        const uint32_t nKeys = srcRows->size();
        for (uint32_t k = 0; k < nKeys; ++k)
          if (const uint32_t soa = (*srcRows)[k]; soa < nPixelSoA)
            pixKey[soa] = k;
      } else {
        for (auto const& ds : pixHits) {
          if (ds.empty())
            continue;
          const uint32_t start = pixHMS[ds.begin()->det()->index()];
          for (auto const& h : ds) {
            const uint32_t k = h.firstClusterRef().index();
            const uint32_t soa = start + pixClu.data()[k].originalId();
            if (soa < nPixelSoA)
              pixKey[soa] = k;
          }
        }
      }

      auto& p = p_;
      p.ptCut = ptCut_;
      p.bsx = bs.x0();
      p.bsy = bs.y0();
      p.bsz = bs.z0();
      p.k = mf.inInverseGeV(GlobalPoint(0, 0, 0)).z();  // PixelRecoUtilities::fieldInInvGev (Patatrack's fit field)
      p.nPixelSoA = nPixelSoA;
      p.nOTSoA = nOTSoA;
      p.nOT = nOT;
      p.nPixKeys = nPixelSoA;
      p.minQuality = static_cast<int>(pixelTrack::Quality::tight);
      p.ptFieldCorrection = ptFieldCorrection_;
      p.bz0 = ::mkfitdev::field::tkBz(0.f, 0.f, 0.f);
      p.pseudoLH = pseudoLastHit_;
      p.pcaAnchor = pcaAnchor_;

      // ---- device copies
      dOtKey_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nOTSoAb));
      dPixKey_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nPixb));
      alpaka::memcpy(queue, *dOtKey_, *hOtKey_);
      alpaka::memcpy(queue, *dPixKey_, *hPixKey_);

      // ---- pass 1: pLS per track + deterministic scan; counts to the host
      const ::reco::TrackBlocksConstView tracksView = iEvent.get(tracksDevToken_).const_view();
      maxTracks_ = tracksView.tracks().metadata().size();
      {
        // SoA track -> edm track index (PixelTrackProducerFromSoAAlpaka's map); no host seed collection is read
        auto const& i2e = iEvent.get(indToEdmToken_);
        const uint32_t nTrk = iEvent.get(recoTracksToken_).size();
        const uint32_t n = std::max<uint32_t>(i2e.size(), 1);
        hSeedOf_.emplace(cms::alpakatools::make_host_buffer<int32_t[]>(queue, n));
        for (uint32_t t = 0; t < n; ++t)
          hSeedOf_->data()[t] = (t < i2e.size() && i2e[t] < nTrk) ? int32_t(i2e[t]) : -1;
        dSeedOf_.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, n));
        alpaka::memcpy(queue, *dSeedOf_, *hSeedOf_);
        p.nSeedMap = std::max<uint32_t>(i2e.size(), 1);
      }
      // with a seed map the kernels skip t >= nSeedMap (= the SoA's nTracks), so buffers and grids stop
      // there instead of at the SoA capacity (122,880 on Phase-2); same outputs
      if (p.nSeedMap > 0)
        maxTracks_ = std::min<uint32_t>(maxTracks_, p.nSeedMap);
      const uint32_t nChunks = (maxTracks_ + 255) / 256;
      mt_ = std::max<uint32_t>(maxTracks_, 1);
      dScratch_.emplace(cms::alpakatools::make_device_buffer<PLS[]>(queue, mt_));
      dPIdx_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, mt_));
      dHOff_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, mt_));
      const uint32_t nCounts = ::mkfitdev::lstin::kNCounts + 2 * nChunks;
      dCounts_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nCounts));
      alpaka::memset(queue, *dCounts_, 0);
      ::mkfitdev::lstseeds::PixSeedOut const* kfOut = nullptr;
      p.pixKF = 0;
      {
        auto const ev = iEvent.get(eohToken_).const_view();
        status_.emplace(queue);
        mkfitdev::zeroStatus(queue, *status_);
        if (ev.hits().metadata().size() == 0) {
          ++nKFNoEOH_;  // empty EventOfHits: this event keeps option (i)
          // the hits-only pLS seeds of this event keep the placeholder state; never silent
          edm::LogError("MkFitAlpakaLstInput") << "pixKF: empty device EventOfHits in event" << iEvent.id().event()
                                               << ": option (i) pLS fields, pLS seeds with a placeholder state";
          mkfitdev::addStatus(queue, *status_, ::mkfitdev::kPixSeedNoEOH, 1);
        } else {
          p.pixKF = 1;
          auto const& esD = iSetup.getData(esDataToken_);
          dPixIn_.emplace(cms::alpakatools::make_device_buffer<::mkfitdev::lstseeds::PixSeedIn[]>(queue, mt_));
          dPixOut_.emplace(cms::alpakatools::make_device_buffer<::mkfitdev::lstseeds::PixSeedOut[]>(queue, mt_));
          ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin::launchPixSeedIn(queue,
                                                                         tracksView,
                                                                         maxTracks_,
                                                                         pixelTrackHits,
                                                                         dPixKey_->data(),
                                                                         dOtKey_->data(),
                                                                         dSeedOf_->data(),
                                                                         p,
                                                                         ev.hits(),
                                                                         dPixIn_->data(),
                                                                         dCounts_->data());
          mkfitdev::StatusSources src;  // KernelPixSeedIn's counters (MkFitAlpakaLstInputKernels.h kNCounts)
          src.add(dCounts_->data() + 8, ::mkfitdev::kPixSeedTooManyHits);
          src.add(dCounts_->data() + 11, ::mkfitdev::kPixSeedNoRow);
          src.add(dCounts_->data() + 12, ::mkfitdev::kPixSeedNoLayer);
          mkfitdev::collectStatus(queue, *status_, src);
          const ::mkfitdev::lstseeds::PixSeedBeam beam{
              float(bs.x0()), float(bs.y0()), float(bs.z0()), float(bs.dxdz()), float(bs.dydz())};
          ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds::fitPixelSeeds(
              queue, esD.view(), ev.hits(), dPixIn_->data(), dPixOut_->data(), int32_t(maxTracks_), kfCfg_, beam);
          kfOut = dPixOut_->data();
          nStates_ = int32_t(iEvent.get(recoTracksToken_).size());
        }
      }
      ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin::launchPLS(queue,
                                                               tracksView,
                                                               maxTracks_,
                                                               pixelTrackHits,
                                                               dPixKey_->data(),
                                                               dOtKey_->data(),
                                                               otSoAView_.metadata().addressOf_detId(),
                                                               otSoAView_.metadata().addressOf_clustSize(),
                                                               dSeedOf_->data(),
                                                               p,
                                                               dScratch_->data(),
                                                               dPIdx_->data(),
                                                               dHOff_->data(),
                                                               dCounts_->data(),
                                                               kfOut);
      hCounts_.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, ::mkfitdev::lstin::kNCounts));
      alpaka::memcpy(
          queue, *hCounts_, cms::alpakatools::make_device_view(queue, dCounts_->data(), ::mkfitdev::lstin::kNCounts));
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto& queue = iEvent.queue();
      uint32_t const* hCounts = hCounts_->data();
      const uint32_t nPLSAll = hCounts[0];
      const uint32_t nHitsIT = hCounts[1];
      const uint32_t nPLS = std::min<uint32_t>(nPLSAll, ::lst::n_max_pixel_segments_per_module);
      if (p_.pixKF) {
        for (int k = 6; k <= 12; ++k)
          kfTot_[k - 6] += hCounts[k];
      }
      if (p_.pixKF && (hCounts[8] || hCounts[11] || hCounts[12]))
        edm::LogWarning("MkFitAlpakaLstInput")
            << "pixKF: pixel tracks without a pLS: > kMaxPixSeedHits hits" << hCounts[8]
            << ", hits without an EventOfHits row " << hCounts[11] << ", hits without an mkFit layer " << hCounts[12];
      if (hCounts[3] || hCounts[4] || hCounts[5])
        edm::LogWarning("MkFitAlpakaLstInput")
            << "tracks with > kMaxTrackHits or < 3 hits " << hCounts[3] << ", hits without a legacy key " << hCounts[4]
            << ", non-finite tracks " << hCounts[5];

      // ---- pass 2: the LST input collection
      LstInputDevice out(queue, int(p_.nOT + nHitsIT), int(nPLS), int(nHitsIT));
      ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin::launchFillSoA(
          queue, out.view(), maxTracks_, nPLS, otSoAView_, p_, dScratch_->data(), dPIdx_->data(), dHOff_->data());
      // the last read of the OT rechit SoA: order its allocation queue after it (early deletion)
      if (!otSoAQueueToken_.isUninitialized()) {
        if (mkfitdev::orderReleaseAfterReads(queue, iEvent.get(otSoAQueueToken_), "MkFitAlpakaLstInput"))
          ++nOtOrdered_;
        else
          ++nOtSameQueue_;
      }
      iEvent.emplace(outToken_, std::move(out));
      {
        // per EDM pixel track the device creator's state (charge 0 = none: failed, not tight, or option (i) event)
        mkfitdev::TrackSoADeviceCollection states(queue, std::max(nStates_, 1));
        auto statesBuf = states.buffer();
        alpaka::memset(queue, statesBuf, 0x00);
        if (p_.pixKF)
          ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin::launchPixSeedStates(queue,
                                                                             states.view(),
                                                                             nStates_,
                                                                             maxTracks_,
                                                                             dSeedOf_->data(),
                                                                             p_.nSeedMap,
                                                                             dPixOut_->data(),
                                                                             dPixIn_->data());
        iEvent.emplace(statesPutToken_, std::move(states));
      }
      iEvent.emplace(statusPutToken_, std::move(*status_));
      status_.reset();
      // per-event buffers: the caching allocators keep them alive until the queue has used them
      hOtKey_.reset();
      hPixKey_.reset();
      dOtKey_.reset();
      dPixKey_.reset();
      dScratch_.reset();
      dPIdx_.reset();
      dHOff_.reset();
      dCounts_.reset();
      hCounts_.reset();
      hSeedOf_.reset();
      dSeedOf_.reset();
      dPixIn_.reset();
      dPixOut_.reset();
    }

    void endStream() override {
      edm::LogInfo("MkFitAlpakaLstInput")
          << "LSTIN_PIXKF totals (stream): devNoPLS " << kfTot_[0] << " pcaInvalid " << kfTot_[1] << " tooManyHits "
          << kfTot_[2] << " rowMismatch " << kfTot_[3] << " pcaOutsideField " << kfTot_[4] << " noRow " << kfTot_[5]
          << " noLayer " << kfTot_[6] << " noEOH " << nKFNoEOH_;
      if (!otSoAQueueToken_.isUninitialized())
        edm::LogInfo("MkFitAlpakaLstInput")
            << "OT rechit SoA release (stream): same queue " << nOtSameQueue_ << ", ordered " << nOtOrdered_;
    }

    const float ptCut_;
    const int pseudoLastHit_;
    const int pcaAnchor_;
    const bool ptFieldCorrection_;
    edm::EDGetTokenT<std::vector<uint32_t>> indToEdmToken_;
    edm::EDGetTokenT<::reco::TrackCollection> recoTracksToken_;
    device::EDGetToken<mkfitdev::OTRecHitDeviceCollection> otSoAToken_;
    edm::EDGetTokenT<unsigned long long> otSoAQueueToken_;
    unsigned long nOtSameQueue_ = 0, nOtOrdered_ = 0;
    ::mkfitdev::OTRecHitSoA::ConstView otSoAView_;
    const edm::EDGetTokenT<SiPixelRecHitCollection> pixToken_;
    const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otCluToken_;
    const edm::EDGetTokenT<SiPixelClusterCollectionNew> pixCluToken_;
    const edm::EDGetTokenT<HMS> pixHMSToken_;
    const edm::EDGetTokenT<HMS> otHMSToken_;
    device::EDGetToken<reco::TracksSoACollection> tracksDevToken_;
    // the pixel and OT rechit SoAs of the pixel tracks
    const device::EDGetToken<reco::TrackingRecHitsSoACollection> pixelHitsSoAToken_;
    const device::EDGetToken<reco::TrackingRecHitsSoACollection> otHitsSoAToken_;
    const edm::EDGetTokenT<::reco::BeamSpot> bsToken_;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
    const device::EDPutToken<LstInputDevice> outToken_;

    // per-event buffers (acquire -> produce)
    template <typename T>
    using HBuf = std::optional<cms::alpakatools::host_buffer<T>>;
    template <typename T>
    using DBuf = std::optional<cms::alpakatools::device_buffer<Device, T>>;
    HBuf<uint32_t[]> hOtKey_, hPixKey_, hCounts_;
    DBuf<uint32_t[]> dOtKey_, dPixKey_, dPIdx_, dHOff_, dCounts_;
    DBuf<PLS[]> dScratch_;
    ::mkfitdev::lstin::Params p_{};
    HBuf<int32_t[]> hSeedOf_;
    DBuf<int32_t[]> dSeedOf_;
    uint32_t maxTracks_ = 0, mt_ = 1;
    device::EDGetToken<mkfitdev::EventOfHitsDeviceCollection> eohToken_;
    device::EDPutToken<mkfitdev::TrackSoADeviceCollection> statesPutToken_;
    device::EDPutToken<mkfitdev::MkFitStatusDeviceObject> statusPutToken_;
    edm::EDGetTokenT<std::vector<uint32_t>> srcRowsToken_;
    edm::EDGetTokenT<std::vector<uint32_t>> keyStartToken_;
    std::optional<mkfitdev::MkFitStatusDeviceObject> status_;
    int32_t nStates_ = 0;
    device::ESGetToken<::mkfitdev::ESData<Device>, TrackerRecoGeometryRecord> esDataToken_;
    ::mkfitdev::lstseeds::LstSeedFitConfig kfCfg_;
    DBuf<::mkfitdev::lstseeds::PixSeedIn[]> dPixIn_;
    DBuf<::mkfitdev::lstseeds::PixSeedOut[]> dPixOut_;
    unsigned long kfTot_[7] = {0, 0, 0, 0, 0, 0, 0};
    unsigned long nKFNoEOH_ = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaLstInputProducer);
