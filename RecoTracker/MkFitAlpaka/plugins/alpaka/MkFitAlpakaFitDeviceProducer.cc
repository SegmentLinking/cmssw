// MkFitAlpakaFitDeviceProducer: the device mkFit final fit as an ASYNCHRONOUS Alpaka producer,
// fed directly by the device building output, in place of MkFitFitProducer in the trackingMkFitFit menu:
//   build TrackSoA (device) -> [this module: device candCutSel, device fit] -> fitted TrackSoA (device)
//   -> framework device->host copy -> MkFitAlpakaOutputTrackConverter (reco::Track).
// - global::EDProducer: produce() only enqueues (no alpaka::wait, no ExternalWork): no CMSSW thread is held while the
//   device works, as the build module.
// - The row count comes from the device (candCutSel output), the grid from the capacity.
// - CPE cluster quantities (ClusterCpe) for EVERY pixel hit, no dependency on the host tracks:
//   clusterSource = "auto" (default): "device" on GPU backends, "tracks" on CPU backends.
//   "tracks" (CPU backends only): the selection runs synchronously, so the host computes clusterCpe() only for the
//   pixel hits on the selected tracks, straight from the cluster collection (no Ref resolution).
//   "device": computed on the device from the pixel digi + cluster SoA (handoff::
//   buildClusterCpe); the host only writes, per mkFit pixel row (= legacy cluster key, as convertHits), the SoA
//   module index and SiPixelCluster::originalId (4 + 4 bytes, no pixel loop). Clusters without originalId (persisted
//   replay clusters) make the event fall back to "host".
//   "host": the host computation (clusterCpe over every cluster, pinned buffer + copy).
// - Output: TrackSoA (capacity = the building's), rows [0, nTracks) = fitted tracks in MkFitCore order; status fields,
//   label, score, charge copied; removed outliers have index -1 and nFoundHits decremented (Track::removeHit).
// - the building's TrackSoA is read first, so the module takes over the building's queue (=
//   the EventOfHits queue). eventOfHitsQueue set (the menu deletes the device EventOfHits right after this
//   module, canDeleteEarly): if this module's queue is not the one the EventOfHits was allocated on, it waits for its
//   own work before returning, so the caching allocator cannot hand the block to another queue while the fit reads it
//   (counted; EOH_RELEASE at endJob).

#include <algorithm>
#include <cstring>
#include <atomic>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusCollect.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/tracks/TrackSoADeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeESData.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsDeviceCollections.h"
#include "RecoTracker/MkFitAlpaka/plugins/alpaka/MkFitAlpakaClusterCpeFill.h"
#include "RecoTracker/MkFitAlpaka/plugins/alpaka/MkFitAlpakaFitCpeTables.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/FitHandoff.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/fit/FitTracks.h"

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
      if (auto const q = iConfig.getParameter<edm::InputTag>("eventOfHitsQueue"); !q.label().empty()) {
        eohQueueToken_ = consumes(q);
        eohQueueCheck_ = true;
      }
      fitOpt_.edgeOutliers = iConfig.getParameter<bool>("edgeOutliers");
      fitOpt_.edgeChi2Cut = float(iConfig.getParameter<double>("edgeChi2Cut"));
      fitOpt_.outlierRounds = iConfig.getParameter<int>("outlierRounds");
      if (fitOpt_.outlierRounds < 0)
        throw cms::Exception("Configuration")
            << "outlierRounds must be >= 0 (1 = MkFitCore, 0 = until a round removes no hit)";
      fitOpt_.firstHitProp = iConfig.getParameter<int>("firstHitProp");            // DEVIATION DEV-5
      fitOpt_.nearEdgeHitsInner = iConfig.getParameter<int>("nearEdgeHitsInner");  // DEVIATION DEV-10
      fitOpt_.nearEdgeHitsOuter = iConfig.getParameter<int>("nearEdgeHitsOuter");
      fitOpt_.outliersPerRound = iConfig.getParameter<int>("outliersPerRound");
      if (fitOpt_.nearEdgeHitsInner < 1 || fitOpt_.nearEdgeHitsOuter < 1 || fitOpt_.outliersPerRound < 0)
        throw cms::Exception("Configuration")
            << "nearEdgeHitsInner / Outer must be >= 1 (1 = DEV-3), outliersPerRound >= 0";
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
        constexpr bool kCpuBackend = std::is_same_v<Device, alpaka_common::DevHost>;
        if (src == "tracks" && !kCpuBackend)
          throw cms::Exception("Configuration")
              << "MkFitAlpakaFitDeviceProducer: clusterSource tracks is for CPU backends";
        clusTracks_ = src == "tracks" || (src == "auto" && kCpuBackend);
        clusDevice_ = src == "device" || (src == "auto" && !kCpuBackend);
        if (clusDevice_) {
          digisToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelClustersSoA"));
          clustersSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelClustersSoA"));
        }
        cpeToken_ = esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("cpeTables")));
        const auto mode = iConfig.getParameter<std::string>("cpeCheck");
        if (mode != "throw" && mode != "off")
          throw cms::Exception("Configuration") << "MkFitAlpakaFitDeviceProducer: cpeCheck must be throw or off";
        cpeCheck_ = mode == "throw";
        if (cpeCheck_) {
          pixelCpeToken_ = esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("pixelCPE")));
          geomToken_ = esConsumes();
        }
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracks", edm::InputTag("hltInitialStepTrackCandidatesMkFitDevice"))
          ->setComment("device TrackSoA of the building (MkFitAlpakaBuildProducer)");
      desc.add<edm::InputTag>("eventOfHits", edm::InputTag("hltMkFitEventOfHits"))
          ->setComment("device EventOfHits product (hits block, same rows as the hit wrappers)");
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
      desc.add<std::string>("pixelCPE", "PixelCPEGeneric");
      desc.add<std::string>("cpeCheck", "throw")
          ->setComment("per-IOV synthetic-cluster cross-check vs pixelCPE: throw | off");
      desc.add<edm::InputTag>("pixelClusters", edm::InputTag("hltSiPixelClusters"))
          ->setComment("the clusters of the pixel rechits (mkFit pixel hit row = cluster key)");
      desc.add<std::string>("clusterSource", "auto")
          ->setComment(
              "CPE cluster quantities: auto (device on GPU, tracks on CPU) | device (from the pixel digi/cluster "
              "SoA) | tracks (CPU backends: host, hits on the selected tracks only) | host (all hits)");
      desc.add<edm::InputTag>("pixelClustersSoA", edm::InputTag("hltPhase2SiPixelClustersSoA"))
          ->setComment(
              "producer of the pixel digi + cluster SoA (SiPixelDigisSoACollection, SiPixelClustersSoACollection)");
      desc.add<bool>("edgeOutliers", false)
          ->setComment(
              "DEVIATION DEV-3 (switch, MkFitCore = false): the first/last fitted hit is an outlier when the pass that "
              "predicts it from all other hits has chi2 > edgeChi2Cut (as the KF EstimateCut)");
      desc.add<double>("edgeChi2Cut", 20.)->setComment("DEVIATION DEV-3: the edge-hit outlier cut");
      desc.add<int>("outlierRounds", 1)
          ->setComment(
              "DEVIATION DEV-3 (switch, MkFitCore = 1): refits after outlier removal; all but the last are checked "
              "again; 0 = until a round removes no hit (DEV-10, as the KF fit-smoother)");
      desc.add<int>("firstHitProp", 0)
          ->setComment(
              "DEVIATION DEV-5 (switch, MkFitCore = 0): propagate to the first hit of a fit pass instead of updating "
              "there without propagation; 1 = in the refits after outlier removal, 2 = every pass");
      desc.add<int>("nearEdgeHitsInner", 1)
          ->setComment(
              "DEVIATION DEV-10 (switch, DEV-3 = 1): with edgeOutliers, the hits this close to the inner end are "
              "outliers when the backward pass (predicted from the outer hits) has chi2 > edgeChi2Cut");
      desc.add<int>("nearEdgeHitsOuter", 1)
          ->setComment("DEVIATION DEV-10 (switch, DEV-3 = 1): the same for the outer end with the forward pass");
      desc.add<int>("outliersPerRound", 0)
          ->setComment(
              "DEVIATION DEV-10 (switch, MkFitCore = 0 = all): at most this many outliers (largest f + b first) "
              "removed per round, as the KF fit-smoother's one outlier per iteration");
      desc.add<edm::InputTag>("eventOfHitsQueue", edm::InputTag(""))
          ->setComment(
              "the EventOfHits producer's 'queue' product, set when the device EventOfHits is deleted"
              "early after this module; a different queue here makes the module wait for its work (counted)");
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
        releaseGuard(iEvent, queue);
        return;
      }
      auto const& es = iSetup.getData(esToken_);
      ::mkfitdev::checkSupportedFitConfig(es.hostConfigValue());  //
      const uint32_t nPix = iEvent.get(pixelHitsToken_).hits().size();

      // 1. candCutSel on the device, order-preserving; out.nTracks() = kept count (device)
      mkfitdev::handoff::selectFitInput(queue, inD.const_view(), capacity, sel_, out.view());

      // 2. CPE cluster quantities (clusterSource): device kernels from the pixel digi/cluster SoA (GPU), host values for
      //    the hits on the selected tracks (CPU backends), or host values for every pixel hit (host / fallback)
      constexpr bool kHostMem = std::is_same_v<Device, alpaka_common::DevHost>;
      std::optional<cms::alpakatools::host_buffer<::mkfitdev::cpe::ClusterCpe[]>> hClus;
      std::optional<cms::alpakatools::device_buffer<Device, ::mkfitdev::cpe::ClusterCpe[]>> dClus;
      const ::mkfitdev::cpe::ClusterCpe* clusPtr = nullptr;
      ::mkfitdev::cpe::CpeTables dCpe{};
      std::optional<cms::alpakatools::device_buffer<Device, ::mkfitdev::handoff::ClusterCpeCounters>> dCnt;
      if (useCpe_) {
        auto const& cpeES = iSetup.getData(cpeToken_);
        auto const& tables = *cpeES.host;
        dCpe = cpeES.view();
        if (cpeCheck_)
          checkCpePerIOV(iSetup, tables);
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
          if (!onDevice)
            ++nHostFallback_;
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
        } else if constexpr (kHostMem) {
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
      mkfitdev::fit::runFinalFit(queue,
                                 es.view(),
                                 iEvent.get(eohToken_).const_view().hits(),
                                 nPix,
                                 out.view(),
                                 capacity,
                                 dCounters.data(),
                                 useCpe_,
                                 dCpe,
                                 clusPtr,
                                 nullptr,  // hitStates: storeHitStates is not offered on the device-handoff path
                                 &out.view().nTracks(),
                                 fitOpt_);
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
      iEvent.emplace(putToken_, std::move(out));
      iEvent.emplace(statusToken_, std::move(status));
      releaseGuard(iEvent, queue);
    }

    void endJob() override {
      if (eohQueueCheck_)
        edm::LogInfo("MkFitAlpakaFitDeviceProducer")
            << "EventOfHits release: same queue " << nSameQueue_ << ", waited " << nQueueWait_;
      if (clusDevice_ || nHostFallback_ > 0)
        edm::LogInfo("MkFitAlpakaFitDeviceProducer")
            << "cluster source: device, host fallback events (no originalId) " << nHostFallback_;
    }

  private:
    // the device EventOfHits is deleted right after this module (canDeleteEarly). Its buffer goes back to the
    // caching allocator with a marker on the queue it was allocated on; on any other queue, wait for this module's work
    // (which follows the building's) so no other queue can reuse the block while it is read.
    void releaseGuard(device::Event& iEvent, Queue& queue) const {
      if (!eohQueueCheck_)
        return;
#if !(defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED))
      {
        const auto eohQueue = iEvent.get(eohQueueToken_);
        if (eohQueue == reinterpret_cast<unsigned long long>(alpaka::getNativeHandle(queue))) {
          ++nSameQueue_;
          return;
        }
        alpaka::wait(queue);
        if (nQueueWait_++ == 0)
          edm::LogWarning("MkFitAlpakaFitDeviceProducer")
              << "event " << iEvent.id().event() << ": the fit does not run on the device EventOfHits queue; it waits "
              << "for its work before the early deletion (totals at endJob)";
      }
#else
      (void)iEvent;  // CPU backends: blocking queue, the work is done when produce() returns
      (void)queue;
#endif
    }

    // CPU backends: clusterCpe for the pixel hits on the selected tracks, other rows untouched
    static void fillClusterCpeOnTracks(edmNew::DetSetVector<SiPixelCluster> const& dsv,
                                       ::mkfitdev::cpe::CpeTablesHost const& tables,
                                       ::mkfitdev::ESView const& es,
                                       ::mkfitdev::TrackSoAConstView trk,
                                       uint32_t nPix,
                                       ::mkfitdev::cpe::ClusterCpe* out) {
      std::vector<uint8_t> need(nPix, 0);
      for (int t = 0; t < trk.nTracks(); ++t) {
        const int nh = std::min<int>(trk[t].nTotalHits(), ::mkfitdev::kMaxTrkHits);
        for (int h = 0; h < nh; ++h) {
          const auto hot = trk[t].hits().hot[h];
          if (hot.index >= 0 && uint32_t(hot.index) < nPix && es.layers[hot.layer].is_pixel())
            need[hot.index] = 1;
        }
      }
      ::mkfitdev::cpe::fillClusterCpe(dsv, tables, nPix, out, need);
    }

    // Once per IOV of the two CPE records, the device CPE tables against the host
    // CPE object of the menu fit, on synthetic clusters.
    void checkCpePerIOV(device::EventSetup const& iSetup, ::mkfitdev::cpe::CpeTablesHost const& t) const {
      edm::EventSetup const& es = iSetup;
      const unsigned long long ids[2] = {es.get<TkPixelCPERecord>().cacheIdentifier(),
                                         es.get<PixelCPEFastParamsRecord>().cacheIdentifier()};
      // lock-free fast path once the IOV is checked; the ids are stored only after the check has passed, so with a
      // failing check every stream checks and throws
      if (checkedIds_[0].load(std::memory_order_acquire) == ids[0] &&
          checkedIds_[1].load(std::memory_order_acquire) == ids[1])
        return;
      std::lock_guard<std::mutex> lock(checkMutex_);
      if (checkedIds_[0].load() == ids[0] && checkedIds_[1].load() == ids[1])
        return;
      const auto st = ::mkfitdev::cpe::crossCheckCpeSynthetic(
          iSetup.getData(pixelCpeToken_), iSetup.getData(geomToken_), t, kCheckModuleStride);
      edm::LogInfo("MkFitAlpakaFitDeviceProducer")
          << "CPE cross-check vs the pixelCPE object (synthetic clusters):" << st.summary();
      if (st.nBad > 0)
        throw cms::Exception("MkFitAlpakaFitCpe")
            << "the device PixelCPEGeneric differs from the pixelCPE object of the menu fit: " << st.summary();
      checkedIds_[1].store(ids[1], std::memory_order_release);
      checkedIds_[0].store(ids[0], std::memory_order_release);
    }

    static constexpr int kCheckModuleStride = 8;

    const device::EDGetToken<mkfitdev::TrackSoADeviceCollection> tracksToken_;
    const device::EDGetToken<mkfitdev::EventOfHitsDeviceCollection> eohToken_;
    const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelHitsToken_;
    const device::ESGetToken<::mkfitdev::ESData<Device>, TrackerRecoGeometryRecord> esToken_;
    const device::EDPutToken<mkfitdev::TrackSoADeviceCollection> putToken_;
    const device::EDPutToken<mkfitdev::MkFitStatusDeviceObject> statusToken_;
    const bool useCpe_;
    mkfitdev::fit::FitOptions fitOpt_;
    ::mkfitdev::handoff::CandCutSel sel_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> clusterCollToken_;
    device::ESGetToken<::mkfitdev::cpe::CpeESData<Device>, PixelCPEFastParamsRecord> cpeToken_;
    edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> pixelCpeToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    bool cpeCheck_ = false;
    bool clusDevice_ = false, clusTracks_ = false;
    device::EDGetToken<SiPixelDigisSoACollection> digisToken_;
    device::EDGetToken<SiPixelClustersSoACollection> clustersSoAToken_;
    mutable std::atomic<long> nHostFallback_{0};
    mutable std::mutex checkMutex_;
    mutable std::atomic<unsigned long long> checkedIds_[2] = {0, 0};
    edm::EDGetTokenT<unsigned long long> eohQueueToken_;
    bool eohQueueCheck_ = false;
    mutable std::atomic<long> nSameQueue_{0}, nQueueWait_{0};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaFitDeviceProducer);
