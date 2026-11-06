// MkFitAlpakaBuildProducer: device track building of one mkFit iteration, in place of MkFitProducer
// (hltInitialStepTrackCandidatesMkFit): seed import, forward search, pre-filter, backward fit, backward search,
// post-filter, export and duplicate cleaning.
// - Inputs: host seeds (MkFitSeedWrapper), the device EventOfHits, the MkFitAlpaka ES product, MkFitGeometry.
// - Outputs: the TrackSoA of the built tracks and the per-event status product (overflow and repack counters).
// - Candidate scratch is module-internal. CPU backends reuse a pooled set per event in flight (grow-only seed
//   capacity, no page faults from fresh buffers); GPU backends allocate per event from the caching allocator. Rows
//   past the event's seed count are never read.
// - The engine tables are built once per IOV and device. The event path has no host synchronization: the seed count
//   and the survivor counts of both filters stay on the device, and grids are sized from the seed capacity.
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
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
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "MkFitAlpakaReleaseOrder.h"
#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/Backend.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusCollect.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/SeedsPackDevice.h"
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
        dupClean.fracSharedHits = cfg.dc_fracSharedHits;
        dupClean.drthCentral = cfg.dc_drth_central;
        dupClean.drthObarrel = cfg.dc_drth_obarrel;
        dupClean.drthForward = cfg.dc_drth_forward;
        // MkFitCore duplicate cleaner choice
        if (cfg.duplicate_cleaner == ::mkfitdev::DuplicateCleaner::SharedHitsPixelPriority) {
          dupClean.pixelPriority = 1;
          for (int word = 0; word < 4; ++word)
            dupClean.pixelLayers[word] = cfg.pixel_layer_mask[word];
        }
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
      int minHitsQFPre = 0, minHitsQFPost = 0;
      ::mkfitdev::clean::DupCleanParams dupClean{0.f, 0.f, 0.f, 0.f};
      ::mkfitdev::bkfit::OutlierParams bkfitOutliers{};
      ::mkfitdev::BkwSearchGate bkwGate{};
    };

    // Clone-engine scratch set. Reused only on CPU backends, whose blocking queues have finished the work when
    // produce() returns.
    struct BuildScratch {
      static constexpr uint32_t kCapacityGranularity = 1024;  // reused capacity: a multiple of this many seeds

      int cap = 0, hps = 0;
      std::optional<mkfitdev::EngineBuffers> buffers, work;
      std::optional<PortableCollection<::mkfitdev::SelListSoA>> list;
      std::optional<PortableCollection<::mkfitdev::PropStateSoA>> props;
      std::optional<PortableCollection<::mkfitdev::SelHitsSoA>> sels;

      // Capacity for nSeeds seeds: exact without reuse (the per-event path), else grow-only with 25% headroom.
      void prepare(Queue& queue, int nSeeds, int hotsPerSeed, bool reuse) {
        if (!reuse || nSeeds > cap || hotsPerSeed != hps || !buffers) {
          cap = reuse ? int(cms::alpakatools::divide_up_by(uint32_t(nSeeds + nSeeds / 4), kCapacityGranularity) *
                            kCapacityGranularity)
                      : nSeeds;
          hps = hotsPerSeed;
          buffers.reset();  // free before allocating the larger set
          work.reset();
          list.reset();
          props.reset();
          sels.reset();
          buffers.emplace(queue, cap, hps);              // the constructor zeroes the seed rows
          work.emplace(queue, cap, hps, buffers->step);  // the searches of both sets run one after the other
          const int nList = cap * ::mkfitdev::kMaxCandsPerSeed;
          list.emplace(queue, nList);
          props.emplace(queue, nList);
          sels.emplace(queue, nList);
          return;
        }
        // reuse: the state of freshly constructed EngineBuffers (seed rows zeroed, no device row count)
        for (mkfitdev::EngineBuffers* set : {&*buffers, &*work}) {
          auto seedsBuf = set->seeds.buffer();
          alpaka::memset(queue, seedsBuf, 0);
          set->nRowsDev = nullptr;
        }
      }
    };
  }  // namespace

  class MkFitAlpakaBuildProducer : public global::EDProducer<> {
  public:
    explicit MkFitAlpakaBuildProducer(edm::ParameterSet const& iConfig)
        : EDProducer<>(iConfig),
          seedsToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
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
          queueToken_{produces("queue")} {
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
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      // the device EventOfHits first, so this module takes over its queue
      const auto& eohD = iEvent.get(eohToken_);
      auto& queue = iEvent.queue();
      // the TrackSoA is deleted early: its reader orders this queue after its reads
      iEvent.emplace(queueToken_, mkfitdev::queueHandle(queue));
      const auto& seedsIn = iEvent.get(seedsToken_).seeds();
      const uint32_t nPixel = iEvent.get(pixelHitsToken_).hits().size();  // = HitVec size (convertHits)
      const auto& esD = iSetup.getData(esToken_);
      const ::mkfitdev::ESView esView = esD.view();
      const int nSeeds = int(seedsIn.size());
      const int hps = hotsPerSeed_;
      ::mkfitdev::checkSupportedBuildConfig(iSetup.getData(esHostToken_).hostConfigValue(),
                                            moduleCfg_);  // no allocation when it passes
      // per-event status product, zeroed here, counters collected after the tail
      mkfitdev::MkFitStatusDeviceObject statusProduct(queue);
      mkfitdev::zeroStatus(queue, statusProduct);
      // an EventOfHits beyond the device build limits arrives empty (nLayers == 0; host-readable metadata)
      const bool eohSkipped = eohD.const_view().layers().metadata().size() == 0;
      if (eohSkipped)
        mkfitdev::addStatus(queue, statusProduct, ::mkfitdev::kEventSkipped, 1);
      if (nSeeds == 0 || eohSkipped) {
        iEvent.emplace(statusToken_, std::move(statusProduct));
        mkfitdev::TrackSoADeviceCollection empty(queue, 0);
        auto buf = empty.buffer();
        alpaka::memset(queue, buf, 0x00);  // defined scalars (nTracks = 0, overflow counters 0)
        iEvent.emplace(tracksToken_, std::move(empty));
        return;
      }
      const BuildTables& engineTables = tables(queue, iSetup, esView);

      // 1. seed import into the engine buffers; the kept-seed count stays on the device
      // CPU backends pack straight into the device collection (no staging copy)
      mkfitdev::SeedsDeviceCollection seedsD = mkfitdev::seeds::packSeedsToDevice(queue, seedsIn);
      const auto ev = eohD.const_view();
      // The EventOfHits hit rows are the pixel rows of the pixel hit map, then the strip rows: nPixel is the strip row
      // base of every hit index.
      if (uint32_t(ev.hits().metadata().size()) < nPixel)
        throw cms::Exception("Configuration") << "MkFitAlpakaBuildProducer: the EventOfHits has "
                                              << ev.hits().metadata().size() << " hit rows, fewer than the " << nPixel
                                              << " pixel rows of pixelHits; both must come from the same pixel rechits";
      mkfitdev::seeds::DeviceHitPositions hitPos;
      hitPos.x = ev.hits().metadata().addressOf_x();
      hitPos.y = ev.hits().metadata().addressOf_y();
      hitPos.z = ev.hits().metadata().addressOf_z();
      hitPos.layerHitBase = ev.layers().metadata().addressOf_hitBase();
      BuildScratch eventScratch;  // per-event path (GPU backends): freed at the end of produce
      const bool reuse = mkfitdev::kCpuBackend;
      // CPU backends: a set of the module's pool, back to it at the end of produce (blocking queue: the work is done)
      const auto pooled = reuse ? scratch_.makeOrGet([] { return new BuildScratch(); }) : nullptr;
      BuildScratch& scratch = reuse ? *pooled : eventScratch;
      scratch.prepare(queue, nSeeds, hps, reuse);
      mkfitdev::EngineBuffers& buffers = *scratch.buffers;
      mkfitdev::EngineBuffers& work = *scratch.work;
      mkfitdev::seeds::importSeeds(queue,
                                   seedsD.view(),
                                   nSeeds,
                                   engineTables.limits,
                                   buffers.seeds.view(),
                                   buffers.slots.view(),
                                   buffers.hots.view(),
                                   hps,
                                   hitPos);
      buffers.nRowsDev = &seedsD.view().nKept();

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
                                           engineTables.modules.data(),
                                           engineTables.layers.data()};

      // 3. K2 (select) through the engine bridge; list capacity nSeeds * kMaxCandsPerSeed, count on the device
      auto& list = *scratch.list;
      auto& props = *scratch.props;
      auto& sels = *scratch.sels;
      const mkfitdev::EngineSelectK2 k2{list.view(),
                                        props.view(),
                                        sels.view(),
                                        esView,
                                        ev.layers(),
                                        ev.binnedHits(),
                                        ev.bins(),
                                        ev.hits(),
                                        nSeeds,
                                        groupScan_};
      const mkfitdev::EngineBackwardFitFn bkfit =
          mkfitdev::makeEngineBackwardFit(in, engineTables.bkfitPF, engineTables.bkfitOutliers);
      // backward-search gate: MkBuilder::beginBkwSearch reads the EventOfHits beam spot
      // (MkFitEventOfHitsProducer: mkfit::BeamSpot(bs.x0(), bs.y0(), ...), floats)
      ::mkfitdev::BkwSearchGate gate = engineTables.bkwGate;
      if (gate.minPixelLayers > 0) {
        const auto& bs = iEvent.get(beamSpotToken_);
        gate.bsX = bs.x0();
        gate.bsY = bs.y0();
      }

      // 4. clone engine, sync-free
      mkfitdev::EngineBuffers& res = mkfitdev::engineRunChainAsync(queue,
                                                                   buffers,
                                                                   work,
                                                                   engineTables.fwd,
                                                                   engineTables.bkw,
                                                                   k2,
                                                                   k2,
                                                                   bkfit,
                                                                   in,
                                                                   in,
                                                                   engineTables.pc,
                                                                   engineTables.ipFwd,
                                                                   engineTables.ipBkw,
                                                                   engineTables.minHitsQFPre,
                                                                   engineTables.minHitsQFPost,
                                                                   nSeeds,
                                                                   gate);

      // 5. tail: export + duplicate cleaner, NO filter; rows = the post-filter survivors
      auto brokenChains = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 1);
      alpaka::memset(queue, brokenChains, 0);
      mkfitdev::TrackSoADeviceCollection exportedD(queue, nSeeds);
      mkfitdev::TrackSoADeviceCollection finalD(queue, nSeeds);
      mkfitdev::seeds::exportAndClean(queue,
                                      res.seeds.const_view(),
                                      res.slots.const_view(),
                                      res.hots.const_view(),
                                      hps,
                                      res.nRowsDev,
                                      nSeeds,
                                      removeDuplicates_,
                                      engineTables.dupClean,
                                      exportedD.view(),
                                      finalD.view(),
                                      brokenChains.data());
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
        src.add(brokenChains.data(), ::mkfitdev::kRepackErrors);
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
      }
      return *it->second;
    }

    const edm::EDGetTokenT<MkFitSeedWrapper> seedsToken_;
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
    const edm::EDPutTokenT<unsigned long long> queueToken_;
    mutable edm::ReusableObjectHolder<BuildScratch> scratch_;
    ::mkfitdev::BuildModuleConfig moduleCfg_;
    mutable std::mutex mutex_;
    mutable std::map<std::pair<long, unsigned long long>, std::unique_ptr<BuildTables>> cache_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaBuildProducer);
