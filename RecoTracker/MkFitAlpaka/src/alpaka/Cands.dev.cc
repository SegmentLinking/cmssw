// Component "cands": the only translation unit that instantiates the candidate kernels. Exported entry points: the
// clone engine (interface/cands/alpaka/CandsEngine.h with the kernels of src/alpaka/engine/*.h).
// One translation unit for all candidate kernels: two sharing the inline device functions of interface/cands fail
// the CUDA device link (nvlink "unexpected reloc").
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/cands/CandSeedOpsKernels.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsEngine.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/PropagationFlagsAdapter.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineSelectBridge.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineSelectBridgeKernels.h"

// ===================================== clone engine =====================================

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  namespace {
    auto workDivFor(int nThreads) {
      const uint32_t blocks = cms::alpakatools::divide_up_by(nThreads > 0 ? nThreads : 1, int(kEngineBlock));
      return cms::alpakatools::make_workdiv<Acc1D>(blocks, kEngineBlock);
    }

    // Thread-per-seed kernels (K4 select, activate, merge, filter, gather, compactify, stage export): one serial,
    // latency-bound loop per thread; small blocks spread the few seed threads over more SMs (GPU backends only).
    constexpr uint32_t kEngineSeedBlock = (kNN == 1) ? 32 : kEngineBlock;
    auto workDivForSeeds(int nSeeds) {
      const uint32_t blocks = cms::alpakatools::divide_up_by(nSeeds > 0 ? nSeeds : 1, int(kEngineSeedBlock));
      return cms::alpakatools::make_workdiv<Acc1D>(blocks, kEngineSeedBlock);
    }

    // K6 and the bookkeeping ops before the backward search.
    class KernelEngineMerge {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    SeedCandsSoA::View seeds,
                                    CandSlotsSoA::View slots,
                                    CandHotsSoA::View hots,
                                    int maxCandsPerSeed,
                                    int nSeeds,
                                    const int32_t* nDev) const {
        nSeeds = engineRows(nSeeds, nDev);
        const int hps = seeds.hotsPerSeed();
        for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
          SeedCandsRef ref = makeSeedRef(seeds, slots, hots, s, hps);
          float pt[kMaxCandsPerSeed];
          for (int ic = 0; ic < *ref.nCands; ++ic) {
            const float invPt = 1.f / ref.states[ic].par[3];  // TrackState::pT()
            pt[ic] = invPt < 0.f ? -invPt : invPt;
          }
          float ptBest = 0.f;  // read by the merge only with a valid best-short candidate
          if (*ref.bestShortValid) {
            const float invPtBest = 1.f / ref.bestShortState->par[3];
            ptBest = invPtBest < 0.f ? -invPtBest : invPtBest;
          }
          mergeCandsAndBestShortOne(ref, maxCandsPerSeed, true, true, pt, ptBest);
        }
      }
    };

    class KernelEngineCompactifyBeginBkw {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    SeedCandsSoA::View seeds,
                                    CandSlotsSoA::View slots,
                                    CandHotsSoA::View hots,
                                    bool doCompactify,
                                    bool doBeginBkw,
                                    ::mkfitdev::BkwSearchGate gate,
                                    int nSeeds,
                                    const int32_t* nDev) const {
        nSeeds = engineRows(nSeeds, nDev);
        const int hps = seeds.hotsPerSeed();
        const bool doGate = doBeginBkw && gate.minPixelLayers > 0;
        if (doBeginBkw && cms::alpakatools::once_per_grid(acc))
          seeds.nRepackRepeat() = 0;
        for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
          if (doBeginBkw)
            seeds.bkwRepacked(s) = 0;  // candidates enter the backward representation, not yet repacked
          if (seeds.nCands(s) <= 0)
            continue;
          const uint32_t ovf0 = seeds.overflowBits(s);
          SeedCandsRef ref = makeSeedRef(seeds, slots, hots, s, hps);
          if (doCompactify)
            compactifyHitStorageForBestCand(ref);
          // EventOfCombCandidates::beginBkwSearch: gate threshold from |d0| of the front candidate,
          // then CombCandidate::beginBkwSearch, then the pre-search candidate is saved
          if (doGate) {
            const float d0 = d0BeamSpot(ref.states[0], gate.bsX, gate.bsY);
            seeds.bkwMinPixLayers(s) = int8_t(std::abs(d0) < gate.promptMaxD0 ? 1 : gate.minPixelLayers);
          }
          if (doBeginBkw)
            beginBkwSearch(ref);
          if (doGate) {
            seeds.preBkwBook(s) = ref.cands[0];
            seeds.preBkwState(s) = ref.states[0];
          }
          failSeedOnHotOverflow(acc, seeds, s, ovf0);
        }
      }
    };

    // EventOfCombCandidates::gateBkwSearch: after the backward search, a seed whose front candidate
    // gained found hits on fewer distinct pixel layers than required gets its pre-search candidate back (one
    // candidate). Runs before the post-filter (its repack sees the restored candidate: lastHitIdx = 0, as MkFitCore).
    class KernelEngineBkwGate {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    SeedCandsSoA::View seeds,
                                    CandSlotsSoA::View slots,
                                    CandHotsSoA::View hots,
                                    ::mkfitdev::BkwSearchGate gate,
                                    int nSeeds,
                                    const int32_t* nDev) const {
        nSeeds = engineRows(nSeeds, nDev);
        const int hps = seeds.hotsPerSeed();
        for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
          if (seeds.nCands(s) <= 0)
            continue;
          SeedCandsRef ref = makeSeedRef(seeds, slots, hots, s, hps);
          if (bkwSearchPixelLayers(ref, gate.pixelLayers) >= seeds.bkwMinPixLayers(s))
            continue;
          ref.cands[0] = seeds.preBkwBook(s);
          ref.states[0] = seeds.preBkwState(s);
          *ref.nCands = 1;
        }
      }
    };
  }  // namespace

  EngineStepScratch::EngineStepScratch(Queue& queue, int maxSeeds)
      : opts(queue, maxSeeds * kMaxOptsPerSeed),
        extras(queue, maxSeeds * kMaxExtrasPerSeed),
        upds(queue, maxSeeds * kMaxCandsPerSeed),
        sel(queue, maxSeeds * kMaxCandsPerSeed),
        c2(queue, maxSeeds * kMaxCandsPerSeed * kMaxHitsPerCand) {}

  EngineBuffers::EngineBuffers(Queue& queue,
                               int maxSeeds_,
                               int hotsPerSeed,
                               std::shared_ptr<EngineStepScratch> sharedStep)
      : maxSeeds(maxSeeds_),
        seeds(queue, maxSeeds_),
        slots(queue, maxSeeds_ * kSlotsPerSeed),
        hots(queue, maxSeeds_ * hotsPerSeed),
        step(sharedStep ? std::move(sharedStep) : std::make_shared<EngineStepScratch>(queue, maxSeeds_)),
        passed(cms::alpakatools::make_device_buffer<int8_t[]>(queue, maxSeeds_ > 0 ? maxSeeds_ : 1)),
        newIdx(cms::alpakatools::make_device_buffer<int32_t[]>(queue, maxSeeds_ > 0 ? maxSeeds_ : 1)),
        nSurvivors(cms::alpakatools::make_device_buffer<int32_t>(queue)) {
    auto seedsBuf = seeds.buffer();  // bkwRepacked = 0, nRepackRepeat = 0 for every producer of the rows
    alpaka::memset(queue, seedsBuf, 0);
  }

  EngineStepTablesDevice::EngineStepTablesDevice(Queue& queue, const ::mkfitdev::EngineStepTablesHost& host)
      : nSteps(host.nSteps),
        nRegions(host.nRegions),
        layer(cms::alpakatools::make_device_buffer<int16_t[]>(queue, host.layer.empty() ? 1 : host.layer.size())),
        prevLayer(cms::alpakatools::make_device_buffer<int16_t[]>(queue,
                                                                  host.prevLayer.empty() ? 1 : host.prevLayer.size())) {
    if (!host.layer.empty()) {
      auto hl = cms::alpakatools::make_host_view(host.layer.data(), host.layer.size());
      auto hp = cms::alpakatools::make_host_view(host.prevLayer.data(), host.prevLayer.size());
      alpaka::memcpy(queue, layer, hl);
      alpaka::memcpy(queue, prevLayer, hp);
      alpaka::wait(queue);  // the host vectors may go away
    }
  }

  void engineActivate(Queue& queue,
                      EngineBuffers& bufs,
                      const int16_t* dStepLayer,
                      const int16_t* dStepPrevLayer,
                      bool fwdSearch,
                      float minPtCut,
                      int nSeeds) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineActivate{},
                        bufs.seeds.view(),
                        bufs.slots.view(),
                        bufs.hots.view(),
                        bufs.step->sel.view(),
                        dStepLayer,
                        dStepPrevLayer,
                        fwdSearch,
                        minPtCut,
                        nSeeds,
                        bufs.nRowsDev);
  }

  void engineStep(Queue& queue,
                  EngineBuffers& bufs,
                  const ::mkfitdev::EngineHitInputs& in,
                  const EnginePropConfig& pc,
                  const ::mkfitdev::EngineIterParams& ip,
                  int nSeeds) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivFor(nSeeds * kMaxCandsPerSeed * kMaxHitsPerCand),
                        KernelEngineChi2{},
                        bufs.seeds.const_view(),
                        bufs.slots.const_view(),
                        bufs.step->sel.const_view(),
                        bufs.step->c2.view(),
                        in,
                        ::mkfitdev::prop::makePropagationFlags(pc.intraLayer, pc.material),
                        pc.propToHit,
                        nSeeds,
                        bufs.nRowsDev);
    alpaka::exec<Acc1D>(queue,
                        workDivFor(nSeeds * kMaxCandsPerSeed),
                        KernelEngineOptions{},
                        bufs.seeds.const_view(),
                        bufs.slots.view(),
                        bufs.hots.const_view(),
                        bufs.step->sel.const_view(),
                        bufs.step->c2.const_view(),
                        bufs.step->opts.view(),
                        bufs.step->extras.view(),
                        in,
                        ip,
                        nSeeds,
                        bufs.nRowsDev);
    SeedSelParams sp;
    sp.layer = -1;
    sp.maxCandsPerSeed = ip.maxCandsPerSeed;
    sp.pTCutOverlap = ip.pTCutOverlap;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineSelect{},
                        bufs.seeds.view(),
                        bufs.slots.view(),
                        bufs.hots.view(),
                        bufs.step->opts.const_view(),
                        bufs.step->extras.view(),
                        bufs.step->upds.view(),
                        bufs.step->sel.const_view(),
                        sp,
                        nSeeds,
                        bufs.nRowsDev);
    alpaka::exec<Acc1D>(queue,
                        workDivFor(nSeeds * kMaxCandsPerSeed),
                        KernelEngineUpdate{},
                        bufs.seeds.const_view(),
                        bufs.slots.view(),
                        bufs.step->upds.const_view(),
                        in,
                        ::mkfitdev::prop::makePropagationFlags(pc.interLayer, pc.material),
                        pc.propToHit,
                        nSeeds,
                        bufs.nRowsDev);
  }

  void engineMerge(Queue& queue, EngineBuffers& bufs, int maxCandsPerSeed, int nSeeds) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineMerge{},
                        bufs.seeds.view(),
                        bufs.slots.view(),
                        bufs.hots.view(),
                        maxCandsPerSeed,
                        nSeeds,
                        bufs.nRowsDev);
  }

  void engineSearch(Queue& queue,
                    EngineBuffers& bufs,
                    const EngineStepTablesDevice& tables,
                    bool fwdSearch,
                    const EngineSelectFn& select,
                    const ::mkfitdev::EngineHitInputs& in,
                    const EnginePropConfig& pc,
                    const ::mkfitdev::EngineIterParams& ip,
                    int nSeeds) {
    for (int step = 0; step < tables.nSteps; ++step) {
      engineActivate(queue, bufs, tables.layerAt(step), tables.prevLayerAt(step), fwdSearch, ip.minPtCut, nSeeds);
      if (select)
        select(queue, bufs, step, nSeeds);
      engineStep(queue, bufs, in, pc, ip, nSeeds);
    }
    engineMerge(queue, bufs, ip.maxCandsPerSeed, nSeeds);
  }

  void engineFilterCompactAsync(
      Queue& queue, EngineBuffers& bufs, EngineBuffers& out, bool bkwRep, int minHitsQF, int nSeeds) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineFilter{},
                        bufs.seeds.view(),
                        bufs.slots.view(),
                        bufs.hots.view(),
                        bufs.passed.data(),
                        bkwRep,
                        minHitsQF,
                        nSeeds,
                        bufs.nRowsDev);
    // one block: the scan carries across chunks of 1024 inside the block
    const auto wdScan = cms::alpakatools::make_workdiv<Acc1D>(1, kEngineBlock);
    alpaka::exec<Acc1D>(queue,
                        wdScan,
                        KernelEngineScanPass{},
                        bufs.passed.data(),
                        bufs.newIdx.data(),
                        bufs.nSurvivors.data(),
                        nSeeds,
                        bufs.nRowsDev);
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineGatherSeeds{},
                        bufs.seeds.const_view(),
                        bufs.slots.const_view(),
                        bufs.hots.const_view(),
                        out.seeds.view(),
                        out.slots.view(),
                        out.hots.view(),
                        bufs.newIdx.data(),
                        bufs.nSurvivors.data(),
                        nSeeds,
                        bufs.nRowsDev);
    out.nRowsDev = bufs.nSurvivors.data();
  }

  void engineCompactifyBeginBkw(Queue& queue,
                                EngineBuffers& bufs,
                                bool doCompactify,
                                bool doBeginBkw,
                                int nSeeds,
                                ::mkfitdev::BkwSearchGate const& gate) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineCompactifyBeginBkw{},
                        bufs.seeds.view(),
                        bufs.slots.view(),
                        bufs.hots.view(),
                        doCompactify,
                        doBeginBkw,
                        gate,
                        nSeeds,
                        bufs.nRowsDev);
  }

  void engineBkwSearchGate(Queue& queue, EngineBuffers& bufs, ::mkfitdev::BkwSearchGate const& gate, int nSeeds) {
    if (nSeeds <= 0 || gate.minPixelLayers <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineBkwGate{},
                        bufs.seeds.view(),
                        bufs.slots.view(),
                        bufs.hots.view(),
                        gate,
                        nSeeds,
                        bufs.nRowsDev);
  }

  EngineBuffers& engineRunChainAsync(Queue& queue,
                                     EngineBuffers& bufs,
                                     EngineBuffers& work,
                                     const EngineStepTablesDevice& fwdTables,
                                     const EngineStepTablesDevice& bkwTables,
                                     const EngineSelectFn& selectFwd,
                                     const EngineSelectFn& selectBkw,
                                     const EngineBackwardFitFn& backwardFit,
                                     const ::mkfitdev::EngineHitInputs& inFwd,
                                     const ::mkfitdev::EngineHitInputs& inBkw,
                                     const EnginePropConfig& pc,
                                     const ::mkfitdev::EngineIterParams& ipFwd,
                                     const ::mkfitdev::EngineIterParams& ipBkw,
                                     int minHitsQFPre,
                                     int minHitsQFPost,
                                     int capacity,
                                     const ::mkfitdev::BkwSearchGate& gate) {
    // forward search (bufs: rows [0, *bufs.nRowsDev))
    engineSearch(queue, bufs, fwdTables, true, selectFwd, inFwd, pc, ipFwd, capacity);
    // pre-filter (params_cur() = forward params: qfilter_n_hits_pixseed && nan_n_silly), compaction bufs -> work
    engineFilterCompactAsync(queue, bufs, work, false, minHitsQFPre, capacity);
    // compactify (backward_drop_seed_hits = false: SupportedConfig), backward fit (identity when empty), beginBkwSearch
    engineCompactifyBeginBkw(queue, work, true, false, capacity, gate);
    if (backwardFit)
      backwardFit(queue, work, capacity);  // rows past *work.nRowsDev have nCands = 0 (gather) and are skipped
    engineCompactifyBeginBkw(queue, work, false, true, capacity, gate);
    // backward search
    engineSearch(queue, work, bkwTables, false, selectBkw, inBkw, pc, ipBkw, capacity);
    engineBkwSearchGate(queue, work, gate, capacity);  // (no-op when off)
    // post-filter (backward params) with the ONE repack, compaction work -> bufs; bufs.nRowsDev = &work.nSurvivors
    engineFilterCompactAsync(queue, work, bufs, true, minHitsQFPost, capacity);
    // endBkwSearch: host-side flag only
    return bufs;
  }

  // ---- engine <-> select (K2) bridge (src/alpaka/engine/EngineSelectBridge.h) ----
  void engineBuildSelectList(Queue& queue, EngineBuffers& bufs, ::mkfitdev::SelListSoA::View list, int nSeeds) {
    if (nSeeds <= 0)
      return;
    // one block of 1024 threads (light kernel): the fill runs inside the single block (CPU: 1 thread)
    const auto wdList = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);
    alpaka::exec<Acc1D>(queue, wdList, KernelEngineBuildList{}, bufs.seeds.const_view(), list, nSeeds, bufs.nRowsDev);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev
