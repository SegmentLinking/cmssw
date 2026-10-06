// Component "cands": the only translation unit that instantiates the candidate kernels (src/alpaka/cands/*.h).
// Exported entry points: filterSeeds (interface/cands/alpaka/CandSeedOpsLaunch.h) and the clone engine
// (interface/cands/alpaka/CandsEngine.h with the kernels of src/alpaka/engine/*.h).
// ONE translation unit for all candidate kernels: two TUs sharing the inline device functions of interface/cands
// broke the CUDA device link (nvlink "unexpected reloc", sm_100).
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandSeedOpsLaunch.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/cands/CandSeedOpsKernels.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsEngine.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/PropagationFlagsAdapter.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineSelectBridge.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineSelectBridgeKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  void filterSeeds(Queue& queue,
                   ::mkfitdev::SeedCandsSoA::View seeds,
                   ::mkfitdev::CandSlotsSoA::View slots,
                   ::mkfitdev::CandHotsSoA::View hots,
                   int8_t* passed,
                   bool bkwRep,
                   bool attemptAllCands,
                   int minHitsQF,
                   int nSeeds) {
    filterSeedsImpl(queue, seeds, slots, hots, passed, bkwRep, attemptAllCands, minHitsQF, nSeeds);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

// ===================================== clone engine =====================================

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  namespace {
    auto workDivFor(int n) {
      const uint32_t blocks = cms::alpakatools::divide_up_by(n > 0 ? n : 1, int(kEngineBlock));
      return cms::alpakatools::make_workdiv<Acc1D>(blocks, kEngineBlock);
    }

    // Thread-per-seed kernels (K4 select, activate, merge, filter, gather, compactify, stage export): one serial,
    // latency-bound loop per thread over ~2.6k seeds. With 128-thread blocks the grid is ~21 blocks, i.e. a third of
    // the SMs of an L4 and 4 warps contending per SM; 32-thread blocks spread the same threads over 4x the SMs
    //. GPU backends only; no shared memory or block cooperation in these kernels.
    constexpr uint32_t kEngineSeedBlock = (kNN == 1) ? 32 : kEngineBlock;
    auto workDivForSeeds(int n) {
      const uint32_t blocks = cms::alpakatools::divide_up_by(n > 0 ? n : 1, int(kEngineSeedBlock));
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
          SeedCandsRef r = makeSeedRef(seeds, slots, hots, s, hps);
          float pt[kMaxCandsPerSeed];
          for (int ic = 0; ic < *r.nCands; ++ic) {
            const float v = 1.f / r.states[ic].par[3];  // TrackState::pT()
            pt[ic] = v < 0.f ? -v : v;
          }
          float ptBest = 0.f;  // read by the merge only with a valid best-short candidate
          if (*r.bestShortValid) {
            const float vb = 1.f / r.bestShortState->par[3];
            ptBest = vb < 0.f ? -vb : vb;
          }
          mergeCandsAndBestShortOne(r, maxCandsPerSeed, true, true, pt, ptBest);
        }
      }
    };

    class KernelEngineCompactifyBeginBkw {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    SeedCandsSoA::View seeds,
                                    CandSlotsSoA::View slots,
                                    CandHotsSoA::View hots,
                                    bool removeSeedHits,
                                    int backwardFitMinHits,
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
          SeedCandsRef r = makeSeedRef(seeds, slots, hots, s, hps);
          if (doCompactify)
            compactifyHitStorageForBestCand(r, removeSeedHits, backwardFitMinHits);
          // EventOfCombCandidates::beginBkwSearch: gate threshold from |d0| of the front candidate,
          // then CombCandidate::beginBkwSearch, then the pre-search candidate is saved
          if (doGate) {
            const float d0 = d0BeamSpot(r.states[0], gate.bsX, gate.bsY);
            seeds.bkwMinPixLayers(s) = int8_t(std::abs(d0) < gate.promptMaxD0 ? 1 : gate.minPixelLayers);
          }
          if (doBeginBkw)
            beginBkwSearch(r);
          if (doGate) {
            seeds.preBkwBook(s) = r.cands[0];
            seeds.preBkwState(s) = r.states[0];
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
          SeedCandsRef r = makeSeedRef(seeds, slots, hots, s, hps);
          if (bkwSearchPixelLayers(r, gate.pixelLayers) >= seeds.bkwMinPixLayers(s))
            continue;
          r.cands[0] = seeds.preBkwBook(s);
          r.states[0] = seeds.preBkwState(s);
          *r.nCands = 1;
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

  EngineStepTablesDevice::EngineStepTablesDevice(Queue& queue, const ::mkfitdev::EngineStepTablesHost& h)
      : nSteps(h.nSteps),
        nRegions(h.nRegions),
        layer(cms::alpakatools::make_device_buffer<int16_t[]>(queue, h.layer.empty() ? 1 : h.layer.size())),
        prevLayer(
            cms::alpakatools::make_device_buffer<int16_t[]>(queue, h.prevLayer.empty() ? 1 : h.prevLayer.size())) {
    if (!h.layer.empty()) {
      auto hl = cms::alpakatools::make_host_view(h.layer.data(), h.layer.size());
      auto hp = cms::alpakatools::make_host_view(h.prevLayer.data(), h.prevLayer.size());
      alpaka::memcpy(queue, layer, hl);
      alpaka::memcpy(queue, prevLayer, hp);
      alpaka::wait(queue);  // the host vectors may go away
    }
  }

  void engineActivate(Queue& queue,
                      EngineBuffers& b,
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
                        b.seeds.view(),
                        b.slots.view(),
                        b.hots.view(),
                        b.step->sel.view(),
                        dStepLayer,
                        dStepPrevLayer,
                        fwdSearch,
                        minPtCut,
                        nSeeds,
                        b.nRowsDev);
  }

  void engineStep(Queue& queue,
                  EngineBuffers& b,
                  const ::mkfitdev::EngineHitInputs& in,
                  const EnginePropConfig& pc,
                  const ::mkfitdev::EngineIterParams& ip,
                  int nSeeds) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivFor(nSeeds * kMaxCandsPerSeed * kMaxHitsPerCand),
                        KernelEngineChi2{},
                        b.seeds.const_view(),
                        b.slots.const_view(),
                        b.step->sel.const_view(),
                        b.step->c2.view(),
                        in,
                        ::mkfitdev::prop::makePropagationFlags(pc.intraLayer, pc.material),
                        pc.propToHit,
                        nSeeds,
                        b.nRowsDev);
    alpaka::exec<Acc1D>(queue,
                        workDivFor(nSeeds * kMaxCandsPerSeed),
                        KernelEngineOptions{},
                        b.seeds.const_view(),
                        b.slots.view(),
                        b.hots.const_view(),
                        b.step->sel.const_view(),
                        b.step->c2.const_view(),
                        b.step->opts.view(),
                        b.step->extras.view(),
                        in,
                        ip,
                        nSeeds,
                        b.nRowsDev);
    SeedSelParams sp;
    sp.layer = -1;
    sp.maxCandsPerSeed = ip.maxCandsPerSeed;
    sp.pTCutOverlap = ip.pTCutOverlap;
    sp.recheckOverlap = ip.recheckOverlap;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineSelect{},
                        b.seeds.view(),
                        b.slots.view(),
                        b.hots.view(),
                        b.step->opts.const_view(),
                        b.step->extras.view(),
                        b.step->upds.view(),
                        b.step->sel.const_view(),
                        sp,
                        nSeeds,
                        b.nRowsDev);
    alpaka::exec<Acc1D>(queue,
                        workDivFor(nSeeds * kMaxCandsPerSeed),
                        KernelEngineUpdate{},
                        b.seeds.const_view(),
                        b.slots.view(),
                        b.step->upds.const_view(),
                        in,
                        ::mkfitdev::prop::makePropagationFlags(pc.interLayer, pc.material),
                        pc.propToHit,
                        nSeeds,
                        b.nRowsDev);
  }

  void engineMerge(Queue& queue, EngineBuffers& b, int maxCandsPerSeed, int nSeeds) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineMerge{},
                        b.seeds.view(),
                        b.slots.view(),
                        b.hots.view(),
                        maxCandsPerSeed,
                        nSeeds,
                        b.nRowsDev);
  }

  void engineSearch(Queue& queue,
                    EngineBuffers& b,
                    const EngineStepTablesDevice& tables,
                    bool fwdSearch,
                    const EngineSelectFn& select,
                    const ::mkfitdev::EngineHitInputs& in,
                    const EnginePropConfig& pc,
                    const ::mkfitdev::EngineIterParams& ip,
                    int nSeeds) {
    for (int t = 0; t < tables.nSteps; ++t) {
      engineActivate(queue, b, tables.layerAt(t), tables.prevLayerAt(t), fwdSearch, ip.minPtCut, nSeeds);
      if (select)
        select(queue, b, t, nSeeds);
      engineStep(queue, b, in, pc, ip, nSeeds);
    }
    engineMerge(queue, b, ip.maxCandsPerSeed, nSeeds);
  }

  void engineFilterCompactAsync(
      Queue& queue, EngineBuffers& b, EngineBuffers& out, bool bkwRep, int minHitsQF, int nSeeds) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineFilter{},
                        b.seeds.view(),
                        b.slots.view(),
                        b.hots.view(),
                        b.passed.data(),
                        bkwRep,
                        minHitsQF,
                        nSeeds,
                        b.nRowsDev);
    // one block: the scan carries across chunks of 1024 inside the block
    const auto wdScan = cms::alpakatools::make_workdiv<Acc1D>(1, kEngineBlock);
    alpaka::exec<Acc1D>(queue,
                        wdScan,
                        KernelEngineScanPass{},
                        b.passed.data(),
                        b.newIdx.data(),
                        b.nSurvivors.data(),
                        nSeeds,
                        b.nRowsDev);
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineGatherSeeds{},
                        b.seeds.const_view(),
                        b.slots.const_view(),
                        b.hots.const_view(),
                        out.seeds.view(),
                        out.slots.view(),
                        out.hots.view(),
                        b.newIdx.data(),
                        b.nSurvivors.data(),
                        nSeeds,
                        b.nRowsDev);
    out.nRowsDev = b.nSurvivors.data();
  }

  int engineFilterCompact(Queue& queue, EngineBuffers& b, EngineBuffers& out, bool bkwRep, int minHitsQF, int nSeeds) {
    if (nSeeds <= 0)
      return 0;
    const int32_t* keep = out.nRowsDev;  // host-count API: the caller passes counts explicitly
    engineFilterCompactAsync(queue, b, out, bkwRep, minHitsQF, nSeeds);
    out.nRowsDev = keep;
    auto hN = cms::alpakatools::make_host_buffer<int32_t>(queue);
    alpaka::memcpy(queue, hN, b.nSurvivors);
    alpaka::wait(queue);
    return *hN.data();
  }

  void engineCompactifyBeginBkw(Queue& queue,
                                EngineBuffers& b,
                                bool removeSeedHits,
                                int backwardFitMinHits,
                                bool doCompactify,
                                bool doBeginBkw,
                                int nSeeds,
                                ::mkfitdev::BkwSearchGate const& gate) {
    if (nSeeds <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineCompactifyBeginBkw{},
                        b.seeds.view(),
                        b.slots.view(),
                        b.hots.view(),
                        removeSeedHits,
                        backwardFitMinHits,
                        doCompactify,
                        doBeginBkw,
                        gate,
                        nSeeds,
                        b.nRowsDev);
  }

  void engineBkwSearchGate(Queue& queue, EngineBuffers& b, ::mkfitdev::BkwSearchGate const& gate, int nSeeds) {
    if (nSeeds <= 0 || gate.minPixelLayers <= 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(nSeeds),
                        KernelEngineBkwGate{},
                        b.seeds.view(),
                        b.slots.view(),
                        b.hots.view(),
                        gate,
                        nSeeds,
                        b.nRowsDev);
  }

  EngineBuffers& engineRunChainAsync(Queue& queue,
                                     EngineBuffers& b,
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
                                     int backwardFitMinHits,
                                     int capacity,
                                     const ::mkfitdev::BkwSearchGate& gate) {
    // forward search (b: rows [0, *b.nRowsDev))
    engineSearch(queue, b, fwdTables, true, selectFwd, inFwd, pc, ipFwd, capacity);
    // pre-filter (params_cur() = forward params: qfilter_n_hits_pixseed && nan_n_silly), compaction b -> work
    engineFilterCompactAsync(queue, b, work, false, minHitsQFPre, capacity);
    // compactify (backward_drop_seed_hits = false for the LST step), backward fit (identity when
    // empty), beginBkwSearch
    engineCompactifyBeginBkw(queue, work, false, backwardFitMinHits, true, false, capacity, gate);
    if (backwardFit)
      backwardFit(queue, work, capacity);  // rows past *work.nRowsDev have nCands = 0 (gather) and are skipped
    engineCompactifyBeginBkw(queue, work, false, backwardFitMinHits, false, true, capacity, gate);
    // backward search
    engineSearch(queue, work, bkwTables, false, selectBkw, inBkw, pc, ipBkw, capacity);
    engineBkwSearchGate(queue, work, gate, capacity);  // (no-op when off)
    // post-filter (backward params) with the ONE repack, compaction work -> b; b.nRowsDev = &work.nSurvivors
    engineFilterCompactAsync(queue, work, b, true, minHitsQFPost, capacity);
    // endBkwSearch: host-side flag only
    return b;
  }

  EngineBuffers& engineRunChain(Queue& queue,
                                EngineBuffers& b,
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
                                int minHitsQF,
                                int backwardFitMinHits,
                                int nSeeds,
                                int& nOut) {
    // nSeeds rows, one sync at the end for nOut
    b.nRowsDev = nullptr;
    EngineBuffers& res = engineRunChainAsync(queue,
                                             b,
                                             work,
                                             fwdTables,
                                             bkwTables,
                                             selectFwd,
                                             selectBkw,
                                             backwardFit,
                                             inFwd,
                                             inBkw,
                                             pc,
                                             ipFwd,
                                             ipBkw,
                                             minHitsQF,
                                             minHitsQF,
                                             backwardFitMinHits,
                                             nSeeds);
    auto hN = cms::alpakatools::make_host_buffer<int32_t>(queue);
    alpaka::memcpy(queue, hN, work.nSurvivors);
    alpaka::wait(queue);
    nOut = *hN.data();
    return res;
  }

  // ---- engine <-> select (K2) bridge (src/alpaka/engine/EngineSelectBridge.h) ----
  void engineBuildSelectList(Queue& queue, EngineBuffers& b, ::mkfitdev::SelListSoA::View list, int nSeeds) {
    if (nSeeds <= 0)
      return;
    // one block of 1024 threads (light kernel, 34 registers): the fill runs inside the single block (CPU: 1 thread)
    const auto wdList = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);
    alpaka::exec<Acc1D>(queue, wdList, KernelEngineBuildList{}, b.seeds.const_view(), list, nSeeds, b.nRowsDev);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev
