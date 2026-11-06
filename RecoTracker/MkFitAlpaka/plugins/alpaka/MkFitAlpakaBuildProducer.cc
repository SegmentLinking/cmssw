// MkFitAlpakaBuildProducer: the device equivalent of the MkFitProducer of the LST step
// (hltInitialStepTrackCandidatesMkFit): seed import -> forward search -> pre-filter -> backward fit -> backward search
// -> post-filter -> export + duplicate cleaner -> TrackSoA.
// - Inputs: host seeds (MkFitSeedWrapper), the device EventOfHits product, the MkFitAlpaka ES product, MkFitGeometry.
// - Candidate storage is module-internal scratch: two EngineBuffers sets + the K2 list/props/sels.
//   On CPU backends per-stream scratch: allocated once per EDM stream at a grow-only seed capacity and reused
//   every event (no page faults from uncached buffers); GPU backends allocate per event (caching allocator). Rows
//   past the event's seed count are never read (grids and loops are bounded by the seed count and the device row
//   counts).
// - Engine tables (layer plans, layer/module tables), iteration parameters and propagation flags are built from the ES
//   product ONCE per IOV and device, not per event.
// - No host synchronization on the event path: the seed count after import and the survivor counts of both filters
//   stay on the device (EngineBuffers::nRowsDev); grids are sized from the seed capacity.
// - The overflow / repack counters go to the "status" output (MkFitStatus), read by the output converter.
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusCollect.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/SeedsPackDevice.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/LstSeedFit.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/tracks/TrackSoADeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/EngineFromES.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsEngine.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedsHostPack.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/SeedsAlgo.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/SeedsDeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/PropagationFlagsAdapter.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/bkfit/BkFitLaunch.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineSelectBridge.h"
#include "MkFitAlpakaSeedHandoffK1Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    // Everything the engine needs from the ES, built once per (device, IOV).
    struct BuildTables {
      BuildTables(Queue& queue,
                  ::mkfitdev::ESDataHost const& esH,
                  ::mkfitdev::ESView const& esView,
                  ::mkfitdev::SeedPartitionLimits lim)
          : plan(::mkfitdev::makeEnginePlan(esH.hostConfigValue())),
            fwd(queue, ::mkfitdev::makeEngineStepTables(plan, true)),
            bkw(queue, ::mkfitdev::makeEngineStepTables(plan, false)),
            layers(cms::alpakatools::make_device_buffer<::mkfitdev::EngineLayerParams[]>(
                queue, esH.sizes.nLayers > 0 ? esH.sizes.nLayers : 1)),
            modules(cms::alpakatools::make_device_buffer<::mkfitdev::EngineModule[]>(
                queue, esH.sizes.nModules > 0 ? esH.sizes.nModules : 1)),
            limits(lim) {
        const ::mkfitdev::ESConfig& cfg = esH.hostConfigValue();
        const auto layersV = ::mkfitdev::makeEngineLayerParams(esH.layers->const_view(), esH.sizes.nLayers);
        const auto modulesV = ::mkfitdev::makeEngineModules(esH.modules->const_view(), esH.sizes.nModules);
        alpaka::memcpy(queue, layers, cms::alpakatools::make_host_view(layersV.data(), layersV.size()));
        alpaka::memcpy(queue, modules, cms::alpakatools::make_host_view(modulesV.data(), modulesV.size()));
        alpaka::wait(queue);  // once per IOV: the host vectors go away, other events' queues use the tables
        pc = mkfitdev::EnginePropConfig{cfg.prop_config.finding_inter_layer_pflags,
                                        cfg.prop_config.finding_intra_layer_pflags,
                                        esView.material,
                                        cfg.prop_config.finding_requires_propagation_to_hit_pos};
        bkfitPF = ::mkfitdev::prop::makePropagationFlags(cfg.prop_config.backward_fit_pflags, esView.material);
        ipFwd = ::mkfitdev::makeEngineIterParams(cfg.params);
        ipBkw = ::mkfitdev::makeEngineIterParams(cfg.backward_params);
        minHitsQFPre = cfg.params.minHitsQF;
        minHitsQFPost = cfg.backward_params.minHitsQF;
        backwardFitMinHits = cfg.backward_fit_min_hits;
        dc[0] = cfg.dc_fracSharedHits;
        dc[1] = cfg.dc_drth_central;
        dc[2] = cfg.dc_drth_obarrel;
        dc[3] = cfg.dc_drth_forward;
        // MkFitCore duplicate cleaner choice
        pixelPriority = cfg.duplicate_cleaner == ::mkfitdev::DuplicateCleaner::SharedHitsPixelPriority;
        for (int w = 0; w < 4; ++w)
          pixelLayers[w] = cfg.pixel_layer_mask[w];
        bkfitOutliers = ::mkfitdev::bkfit::OutlierParams{
            cfg.backward_fit_outlier_chi2, cfg.backward_fit_max_outliers, cfg.backward_fit_outlier_min_pt};
        bkwGate = ::mkfitdev::BkwSearchGate{
            cfg.backward_search_min_pixel_layers,
            cfg.backward_search_prompt_max_d0,
            {cfg.pixel_layer_mask[0], cfg.pixel_layer_mask[1], cfg.pixel_layer_mask[2], cfg.pixel_layer_mask[3]},
            0.f,
            0.f};
      }
      ::mkfitdev::EnginePlan plan;
      mkfitdev::EngineStepTablesDevice fwd, bkw;
      cms::alpakatools::device_buffer<Device, ::mkfitdev::EngineLayerParams[]> layers;
      cms::alpakatools::device_buffer<Device, ::mkfitdev::EngineModule[]> modules;
      ::mkfitdev::SeedPartitionLimits limits;
      mkfitdev::EnginePropConfig pc;
      ::mkfitdev::prop::PropagationFlags bkfitPF;
      ::mkfitdev::EngineIterParams ipFwd, ipBkw;
      int minHitsQFPre = 0, minHitsQFPost = 0, backwardFitMinHits = 0;
      float dc[4] = {0.f, 0.f, 0.f, 0.f};
      bool pixelPriority = false;
      uint64_t pixelLayers[4] = {0, 0, 0, 0};
      ::mkfitdev::bkfit::OutlierParams bkfitOutliers{};
      ::mkfitdev::BkwSearchGate bkwGate{};
    };

    // clone-engine scratch of one EDM stream. The EDM stream starts the next event only after the device work
    // of this one has completed (EDMetadata synchronizes when the event's products go away), so reuse is safe on
    // every backend.
    struct BuildScratch {
      int cap = 0, hps = 0;
      uint64_t nAllocs = 0;
      std::optional<mkfitdev::EngineBuffers> b, work;
      std::optional<PortableCollection<::mkfitdev::SelListSoA>> list;
      std::optional<PortableCollection<::mkfitdev::PropStateSoA>> props;
      std::optional<PortableCollection<::mkfitdev::SelHitsSoA>> sels;

      // Capacity for n seeds: exact without reuse (the per-event path), else grow-only with 25% headroom.
      void prepare(Queue& queue, int n, int hotsPerSeed, bool reuse) {
        if (!reuse || n > cap || hotsPerSeed != hps || !b) {
          cap = reuse ? ((n + n / 4 + 1023) / 1024) * 1024 : n;
          hps = hotsPerSeed;
          b.reset();  // free before allocating the larger set
          work.reset();
          list.reset();
          props.reset();
          sels.reset();
          b.emplace(queue, cap, hps);              // the constructor zeroes the seed rows
          work.emplace(queue, cap, hps, b->step);  // the searches of b and work run one after the other
          const int nList = cap * ::mkfitdev::kMaxCandsPerSeed;
          list.emplace(queue, nList);
          props.emplace(queue, nList);
          sels.emplace(queue, nList);
          ++nAllocs;
          return;
        }
        // reuse: the state of freshly constructed EngineBuffers (seed rows zeroed, no device row count)
        for (mkfitdev::EngineBuffers* e : {&*b, &*work}) {
          auto seedsBuf = e->seeds.buffer();
          alpaka::memset(queue, seedsBuf, 0);
          e->nRowsDev = nullptr;
        }
      }
    };
  }  // namespace

  class MkFitAlpakaBuildProducer : public global::EDProducer<edm::StreamCache<BuildScratch>> {
  public:
    explicit MkFitAlpakaBuildProducer(edm::ParameterSet const& iConfig)
        : EDProducer<edm::StreamCache<BuildScratch>>(iConfig),
          pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
          eohToken_{consumes(iConfig.getParameter<edm::InputTag>("eventOfHits"))},
          beamSpotToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
          mkFitGeomToken_{esConsumes()},
          esToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("esData"))},
          esHostToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("esData"))},
          tracksToken_{produces()},
          hotsPerSeed_{iConfig.getParameter<int>("hotsPerSeed")},
          removeDuplicates_{iConfig.getParameter<bool>("removeDuplicates")},
          groupScan_{iConfig.getParameter<bool>("k2GroupScan")},
          statusToken_{produces()},
          lstSeedFit_{iConfig.getParameter<bool>("lstSeedFit")} {
      if (auto const t = iConfig.getParameter<edm::InputTag>("deviceSeeds"); !t.label().empty()) {
        deviceSeedsToken_ = consumes(t);  // K1: the seed rows of MkFitAlpakaSeedHandoffK1 (device)
        k1Seeds_ = true;
      } else
        seedsToken_ = consumes(iConfig.getParameter<edm::InputTag>("seeds"));
      if (auto const t = iConfig.getParameter<edm::InputTag>("pixelSeedStates"); !t.label().empty()) {
        // seeds with a pixel track get the device creator's state of it (pT3 / pT5: also its hits when their fit fails)
        pixStatesToken_ = consumes(t);
        pixStates_ = true;
        pixOfSeedToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelTrackOfSeed"));
      }
      lstSeedCfg_.passes = iConfig.getParameter<int>("lstSeedFitPasses");
      lstSeedCfg_.errScale = iConfig.getParameter<double>("lstSeedFitErrScale");
      lstSeedCfg_.dropFailed = iConfig.getParameter<bool>("lstSeedFitDropFailed");
      lstSeedCfg_.originPrior = iConfig.getParameter<int>("lstSeedFitOriginPrior");
      lstSeedCfg_.hlBackwardTol = iConfig.getParameter<double>("lstSeedFitBackwardTol");
      lstSeedCfg_.hlPixelFallback = iConfig.getParameter<bool>("lstSeedFitPixelFallback");
      // MkFitProducer parameters the device build depends on
      moduleCfg_.clustersToSkip = iConfig.getParameter<edm::InputTag>("clustersToSkip").label();
      moduleCfg_.buildingRoutine = iConfig.getParameter<std::string>("buildingRoutine");
      moduleCfg_.seedCleaning = iConfig.getParameter<bool>("seedCleaning");
      moduleCfg_.removeDuplicates = removeDuplicates_;
      moduleCfg_.backwardFitInCMSSW = iConfig.getParameter<bool>("backwardFitInCMSSW");
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("seeds", edm::InputTag{"hltInitialStepMkFitSeeds"});
      desc.add("deviceSeeds", edm::InputTag{""})
          ->setComment(
              "K1: the device seed rows of MkFitAlpakaSeedHandoffK1 instead of 'seeds' (the seed count ="
              "the size of pixelTrackOfSeed, which must then come from the same module)");
      desc.add("pixelHits", edm::InputTag{"hltMkFitSiPixelHits"})
          ->setComment("MkFitClusterIndexToHit, only for nPixel (strip hit row base)");
      desc.add("eventOfHits", edm::InputTag{"hltMkFitEventOfHitsAlpaka"})->setComment("device EventOfHits product");
      desc.add("beamSpot", edm::InputTag{"hltOnlineBeamSpot"})
          ->setComment("beam spot of the MkFitEventOfHitsProducer (backward-search gate)");
      desc.add("esData", edm::ESInputTag{"", ""})->setComment("MkFitAlpakaESProducer ComponentName");
      desc.add("hotsPerSeed", 256)->setComment("HoT pool rows per seed (overflow: seed failed and counted)");
      desc.add("removeDuplicates", true);
      // MkFitProducer parameters, accepted only inside the validated envelope (interface/SupportedConfig.h)
      desc.add("clustersToSkip", edm::InputTag())->setComment("must be empty: no hit mask on the device");
      desc.add<std::string>("buildingRoutine", "cloneEngine");
      desc.add("seedCleaning", true);
      desc.add("backwardFitInCMSSW", false);
      desc.add("k2GroupScan", true)
          ->setComment("GPU: K2 hit scan with several lanes per candidate; false = thread per candidate");
      desc.add("lstSeedFit", false)
          ->setComment(
              "option (b), DEVIATION candidate, OFF by default: the state of every seed with an"
              "outer-tracker hit (LST T5/T4/pT3/pT5) from a device mkFit Kalman fit of its hits; pLS keep "
              "the copied pixel state (interface/seeds/alpaka/LstSeedFit.h)");
      desc.add("pixelSeedStates", edm::InputTag{""})
          ->setComment("hltInputLSTDevice:pixelSeedStates (device pixel seed hits and states per pixel track)");
      desc.add("pixelTrackOfSeed", edm::InputTag{""})
          ->setComment("LSTOutputConverter:pixelTrackOfSeed (pixel track index per seed, -1 = none)");
      desc.add("lstSeedFitPasses", 3)->setComment("3 = forward, backward, forward; 1 = one forward pass");
      desc.add("lstSeedFitErrScale", 1.0)->setComment("final seed errors scaled by this factor");
      desc.add("lstSeedFitOriginPrior", 0)
          ->setComment("beam-line prior as the host seed creator: 0 none, 1 OT-only seeds (T5/T4), 2 all fitted seeds");
      desc.add("lstSeedFitBackwardTol", 0.01)
          ->setComment("originPrior 3: a step is 'against the momentum' only below -tol cm");
      desc.add("lstSeedFitPixelFallback", true)
          ->setComment("originPrior 3 + dropFailed: failed pT3/pT5 seeds take their pixel seed (pixelSeedStates)");
      desc.add("lstSeedFitDropFailed", false)
          ->setComment("failed device seed fits removed (true; host seeds with placeholder states) or kept (false)");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<BuildScratch> beginStream(edm::StreamID) const override { return std::make_unique<BuildScratch>(); }

    void produce(edm::StreamID sid, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      // the device EventOfHits first, so this module takes over its queue
      const auto& eohD = iEvent.get(eohToken_);
      auto& queue = iEvent.queue();
      const bool k1 = k1Seeds_;
      static const mkfit::TrackVec kNoSeeds;
      const auto& seedsIn = k1 ? kNoSeeds : iEvent.get(seedsToken_).seeds();
      const uint32_t nPixel = iEvent.get(pixelHitsToken_).hits().size();  // = HitVec size (convertHits)
      const auto& esD = iSetup.getData(esToken_);
      const ::mkfitdev::ESView esView = esD.view();
      if (k1 && !pixStates_)
        throw cms::Exception("Configuration")
            << "MkFitAlpakaBuildProducer: deviceSeeds needs pixelTrackOfSeed (the K1 seed count)";
      const int n = k1 ? int(iEvent.get(pixOfSeedToken_).size()) : int(seedsIn.size());
      const int hps = hotsPerSeed_;
      ::mkfitdev::checkSupportedBuildConfig(iSetup.getData(esHostToken_).hostConfigValue(),
                                            moduleCfg_);  // (no allocation when it passes)
      // per-event status product, zeroed here, counters collected after the tail
      mkfitdev::MkFitStatusDeviceObject statusProduct(queue);
      mkfitdev::zeroStatus(queue, statusProduct);
      // an EventOfHits beyond the device build limits arrives empty (nLayers == 0; host-readable metadata)
      const bool eohSkipped = eohD.const_view().layers().metadata().size() == 0;
      if (eohSkipped)
        mkfitdev::addStatus(queue, statusProduct, ::mkfitdev::kEventSkipped, 1);
      if (n == 0 || eohSkipped) {
        iEvent.emplace(statusToken_, std::move(statusProduct));
        mkfitdev::TrackSoADeviceCollection empty(queue, 0);
        auto buf = empty.buffer();
        alpaka::memset(queue, buf, 0x00);  // defined scalars (nTracks = 0, overflow counters 0)
        iEvent.emplace(tracksToken_, std::move(empty));
        return;
      }
      const BuildTables& T = tables(queue, iSetup, esView);

      // 1. seed import into the engine buffers; the kept-seed count stays on the device
      // io's packer: CPU backends pack straight into the device collection (no staging copy)
      mkfitdev::SeedsDeviceCollection seedsD =
          k1 ? mkfitdev::SeedsDeviceCollection(queue, n) : mkfitdev::seeds::packSeedsToDevice(queue, seedsIn);
      if (k1)  // K1: the device seed rows -> the seed table (as packSeeds; the status is MkFitCore Track's default)
        ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::k1::launchSeedsFromK1(
            queue, iEvent.get(deviceSeedsToken_).const_view(), n, k1SeedStatus_, seedsD.view());
      const auto ev = eohD.const_view();
      ::mkfitdev::lstseeds::PixelSeedRows pix;  // the pixel seed of each seed row (pLS, pT3, pT5)
      std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> mapD;
      if (pixStates_) {
        // before the LST seed fit (pixel+OT seeds are refitted there, pixel-only rows keep this)
        auto const& map = iEvent.get(pixOfSeedToken_);
        if (int(map.size()) != n)
          throw cms::Exception("LogicError") << "pixelTrackOfSeed size " << map.size() << " != seeds " << n;
        auto const& states = iEvent.get(pixStatesToken_);
        auto mapH = cms::alpakatools::make_host_buffer<int32_t[]>(queue, n);
        std::copy(map.begin(), map.end(), mapH.data());
        mapD.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, n));
        alpaka::memcpy(queue, *mapD, mapH);
        pix = {mapD->data(), states.const_view(), states.const_view().metadata().size()};
        ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds::applyPixelSeedStates(
            queue, seedsD.view(), n, pix.pixIdx, pix.states, pix.nStates);
      }
      if (lstSeedFit_) {  // device seed state for the LST seeds with OT hits (switch-gated, default off)
        auto cnt = cms::alpakatools::make_device_buffer<mkfitdev::lstseeds::LstSeedFitCounters>(queue);
        alpaka::memset(queue, cnt, 0);
        mkfitdev::lstseeds::fitLstSeeds(
            queue, esView, ev.hits(), nPixel, seedsD.view(), n, lstSeedCfg_, pix, cnt.data());
      }
      mkfitdev::seeds::DeviceHitPositions hitPos;
      hitPos.x = ev.hits().metadata().addressOf_x();
      hitPos.y = ev.hits().metadata().addressOf_y();
      hitPos.z = ev.hits().metadata().addressOf_z();
      hitPos.layerHitBase = ev.layers().metadata().addressOf_hitBase();
      BuildScratch eventScratch;  // per-event path (GPU backends): freed at the end of produce
      const bool reuse = kCpuBackend;
      BuildScratch& S = reuse ? *streamCache(sid) : eventScratch;
      S.prepare(queue, n, hps, reuse);
      mkfitdev::EngineBuffers& b = *S.b;
      mkfitdev::EngineBuffers& work = *S.work;
      mkfitdev::seeds::importSeeds(
          queue, seedsD.view(), n, T.limits, b.seeds.view(), b.slots.view(), b.hots.view(), hps, hitPos);
      b.nRowsDev = &seedsD.view().nKept();

      // 2. event inputs of the engine kernels
      const auto hv = ev.hits();
      const ::mkfitdev::EngineHitInputs in{hv.metadata().addressOf_x(),
                                           hv.metadata().addressOf_y(),
                                           hv.metadata().addressOf_z(),
                                           hv.metadata().addressOf_e00(),
                                           hv.metadata().addressOf_e10(),
                                           hv.metadata().addressOf_e11(),
                                           hv.metadata().addressOf_e20(),
                                           hv.metadata().addressOf_e21(),
                                           hv.metadata().addressOf_e22(),
                                           hv.metadata().addressOf_packed(),
                                           nPixel,
                                           T.modules.data(),
                                           T.layers.data()};

      // 3. K2 (select) through the engine bridge; list capacity nSeeds * kMaxCandsPerSeed, count on the device
      auto& list = *S.list;
      auto& props = *S.props;
      auto& sels = *S.sels;
      const mkfitdev::EngineSelectK2 k2{list.view(),
                                        props.view(),
                                        sels.view(),
                                        esView,
                                        ev.layers(),
                                        ev.binnedHits(),
                                        ev.bins(),
                                        ev.hits(),
                                        n,
                                        groupScan_};
      const mkfitdev::EngineBackwardFitFn bkfit = mkfitdev::makeEngineBackwardFit(in, T.bkfitPF, T.bkfitOutliers);
      // backward-search gate: MkBuilder::beginBkwSearch reads the EventOfHits beam spot
      // (MkFitEventOfHitsProducer: mkfit::BeamSpot(bs.x0(), bs.y0(), ...), floats)
      ::mkfitdev::BkwSearchGate gate = T.bkwGate;
      if (gate.minPixelLayers > 0) {
        const auto& bs = iEvent.get(beamSpotToken_);
        gate.bsX = bs.x0();
        gate.bsY = bs.y0();
      }

      // 4. clone engine, sync-free
      mkfitdev::EngineBuffers& res = mkfitdev::engineRunChainAsync(queue,
                                                                   b,
                                                                   work,
                                                                   T.fwd,
                                                                   T.bkw,
                                                                   k2,
                                                                   k2,
                                                                   bkfit,
                                                                   in,
                                                                   in,
                                                                   T.pc,
                                                                   T.ipFwd,
                                                                   T.ipBkw,
                                                                   T.minHitsQFPre,
                                                                   T.minHitsQFPost,
                                                                   T.backwardFitMinHits,
                                                                   n,
                                                                   gate);

      // 5. tail: export + duplicate cleaner, NO filter; rows = the post-filter survivors
      auto status = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 1);
      alpaka::memset(queue, status, 0);
      mkfitdev::TrackSoADeviceCollection exportedD(queue, n);
      mkfitdev::TrackSoADeviceCollection finalD(queue, n);
      mkfitdev::seeds::exportAndClean(queue,
                                      res.seeds.const_view(),
                                      res.slots.const_view(),
                                      res.hots.const_view(),
                                      hps,
                                      res.nRowsDev,
                                      n,
                                      removeDuplicates_,
                                      T.dc,
                                      T.pixelPriority ? T.pixelLayers : nullptr,
                                      exportedD.view(),
                                      finalD.view(),
                                      status.data());
      {
        mkfitdev::StatusSources src;
        src.add(seedsD.view().metadata().addressOf_nOverflowHits(), ::mkfitdev::kSeedHitsTruncated);
        src.add(ev.layers().metadata().addressOf_nOverflowFirst(), ::mkfitdev::kEohOverflowFirst);
        src.add(ev.layers().metadata().addressOf_nOverflowCount(), ::mkfitdev::kEohOverflowCount);
        // the engine's compactions carry the seed-pool counters forward: the result buffer holds the totals
        src.add(res.seeds.view().metadata().addressOf_nOverflowHots(), ::mkfitdev::kHotOverflowSeeds);
        src.add(res.seeds.view().metadata().addressOf_nOverflowOpts(), ::mkfitdev::kOptsOverflow);
        src.add(res.seeds.view().metadata().addressOf_nOverflowExtras(), ::mkfitdev::kExtrasOverflow);
        // repack applied more than once (engine bit) or a broken HoT chain at export (exportAndClean)
        src.add(res.seeds.view().metadata().addressOf_nRepackRepeat(), ::mkfitdev::kRepackErrors);
        src.add(status.data(), ::mkfitdev::kRepackErrors);
        // the cleaner copies the export counters into the final TrackSoA
        src.add(finalD.view().metadata().addressOf_nOverflowTracks(), ::mkfitdev::kTrackOverflow);
        src.add(finalD.view().metadata().addressOf_nOverflowHits(), ::mkfitdev::kTrackHitsOverflow);
        mkfitdev::collectStatus(queue, statusProduct, src);
      }
      iEvent.emplace(statusToken_, std::move(statusProduct));
      iEvent.emplace(tracksToken_, std::move(finalD));
    }

  private:
    const BuildTables& tables(Queue& queue, device::EventSetup const& iSetup, ::mkfitdev::ESView const& esView) const {
      edm::EventSetup const& es = iSetup;
      const auto key = std::make_pair(static_cast<long>(alpaka::getNativeHandle(alpaka::getDev(queue))),
                                      es.get<TrackerRecoGeometryRecord>().cacheIdentifier());
      std::lock_guard<std::mutex> guard(mutex_);
      auto it = cache_.find(key);
      if (it == cache_.end()) {
        const auto& esH = iSetup.getData(esHostToken_);
        const auto& ti = iSetup.getData(mkFitGeomToken_).trackerInfo();
        // tables of earlier IOVs stay alive until the end of the job (events of other streams may still use them)
        it = cache_.emplace(key, std::make_unique<BuildTables>(queue, esH, esView, ::mkfitdev::seedPartitionLimits(ti)))
                 .first;
        edm::LogInfo("MkFitAlpakaBuild") << "build tables: device " << key.first << " iov " << key.second << " regions "
                                         << it->second->fwd.nRegions << " fwd steps " << it->second->fwd.nSteps
                                         << " bkw steps " << it->second->bkw.nSteps;
      }
      return *it->second;
    }

    edm::EDGetTokenT<MkFitSeedWrapper> seedsToken_;
    device::EDGetToken<mkfitdev::TrackSoADeviceCollection> deviceSeedsToken_;
    bool k1Seeds_ = false;
    const uint32_t k1SeedStatus_ = [] {
      const mkfit::Track t;
      const auto st = t.getStatus();
      uint32_t sb;
      std::memcpy(&sb, &st, sizeof(sb));
      return sb;
    }();
    const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelHitsToken_;
    const device::EDGetToken<mkfitdev::EventOfHitsDeviceCollection> eohToken_;
    const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;  // backward-search gate (EOH beam spot)
    const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
    const device::ESGetToken<::mkfitdev::ESData<Device>, TrackerRecoGeometryRecord> esToken_;
    const edm::ESGetToken<::mkfitdev::ESDataHost, TrackerRecoGeometryRecord> esHostToken_;
    const device::EDPutToken<mkfitdev::TrackSoADeviceCollection> tracksToken_;
    const int hotsPerSeed_;
    const bool removeDuplicates_;
    const bool groupScan_;
    const device::EDPutToken<mkfitdev::MkFitStatusDeviceObject> statusToken_;
    const bool lstSeedFit_;
    static constexpr bool kCpuBackend = std::is_same_v<Device, alpaka::DevCpu>;
    ::mkfitdev::lstseeds::LstSeedFitConfig lstSeedCfg_;
    device::EDGetToken<mkfitdev::TrackSoADeviceCollection> pixStatesToken_;
    edm::EDGetTokenT<std::vector<int>> pixOfSeedToken_;
    bool pixStates_ = false;
    ::mkfitdev::BuildModuleConfig moduleCfg_;
    mutable std::mutex mutex_;
    mutable std::map<std::pair<long, unsigned long long>, std::unique_ptr<BuildTables>> cache_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaBuildProducer);
