// Component "v2p2": the only translation unit that instantiates the v2p2 search kernels (src/alpaka/v2p2/*.h).
// Exported entry points: interface/v2p2/alpaka/V2p2Search.h. It does not include src/alpaka/engine/*: the candidate
// helpers it shares with Cands.dev.cc are in src/alpaka/cands/CandSeedOpsKernels.h (one copy of each inline device
// function per device link).
#include <algorithm>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/alpaka/V2p2Search.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/PropagationFlagsAdapter.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/V2p2Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  namespace {
    auto workDivForSeeds(int nSeeds) {
      const uint32_t blocks = cms::alpakatools::divide_up_by(nSeeds, int(kV2p2SeedBlock));
      return cms::alpakatools::make_workdiv<Acc1D>(blocks, kV2p2SeedBlock);
    }

    // V2: blocks of kV2p2SelectRows entries of V0's list of CandIncomings (kV2p2SelectLanes threads; one thread on CPU
    // backends). GPU backends launch one block per seed row, which covers an entry's CandIncomings but for rare
    // entries, where blocks take several (grid-stride); CPU backends cover every row.
    auto workDivForCands(int nSeeds) {
      const int n = nSeeds * (kV2p2BlockThreads ? 1 : kMaxCandsPerSeed);
      const uint32_t blocks = cms::alpakatools::divide_up_by(n, int(kV2p2SelectRows));
      return cms::alpakatools::make_workdiv<Acc1D>(blocks, kV2p2SelectLanes);
    }

    // V4: a group of kV2p2SelectSeedLanes threads per seed, kV2p2SelectSeedsPerBlock groups per block, which runs
    // kV2p2SelectTilesPerBlock tiles of seeds (one thread per block on CPU backends)
    auto workDivForSeedGroups(int nSeeds) {
      const uint32_t blocks =
          cms::alpakatools::divide_up_by(nSeeds, int(kV2p2SelectSeedsPerBlock * kV2p2SelectTilesPerBlock));
      return cms::alpakatools::make_workdiv<Acc1D>(blocks, kV2p2SelectSeedsPerBlock * kV2p2SelectSeedLanes);
    }

    // V3: blocks of kV2p2InLayerRows CandIncoming rows (all rows on GPU backends, V0's list on CPU ones)
    auto workDivForItems(int nSeeds) {
      const uint32_t blocks = cms::alpakatools::divide_up_by(nSeeds * kMaxCandsPerSeed, int(kV2p2InLayerRows));
      return cms::alpakatools::make_workdiv<Acc1D>(blocks, kV2p2InLayerThreads);
    }

  }  // namespace

  V2p2Buffers::V2p2Buffers(Queue& queue, int maxSeeds, int preselStride_, int nodesPerSeed)
      : preselStride(preselStride_),
        nodeCapacity(maxSeeds * nodesPerSeed),
        seeds(queue, maxSeeds),
        cands(queue, maxSeeds * kMaxCandsPerSeed),
        layerHits(queue, maxSeeds * kMaxCandsPerSeed * preselStride_),
        nodes(queue, nodeCapacity) {
    auto seedsBuf = seeds.buffer();  // inLayer = 0 and zero counters before the first V0
    alpaka::memset(queue, seedsBuf, 0);
  }

  V2p2StepTablesDevice::V2p2StepTablesDevice(Queue& queue, const ::mkfitdev::V2p2StepTablesHost& host_)
      : host(host_), steps(queue, host_.steps) {}

  bool V2p2StepTablesDevice::awakenOnly(int step) const {
    for (int r = 0; r < host.steps.nRegions; ++r)
      if (host.steps.layer[step * host.steps.nRegions + r] >= 0)
        return false;
    return true;
  }

  void v2p2ResetCounters(Queue& queue, V2p2Buffers& v2) {
    auto counters =
        cms::alpakatools::make_device_view(alpaka::getDev(queue), v2.seeds.view().metadata().addressOf_counters(), 1);
    alpaka::memset(queue, counters, 0);
  }

  void v2p2Search(Queue& queue,
                  EngineBuffers& bufs,
                  V2p2Buffers& v2,
                  const V2p2StepTablesDevice& tables,
                  bool fwdSearch,
                  const V2p2EventInputs& in,
                  const V2p2SearchConfig& cfg,
                  int capacity) {
    if (capacity <= 0)
      return;
    if (!fwdSearch)
      engineSkipBkwSearchRegions(queue, bufs, tables.steps.plans, tables.steps.nRegions, capacity);
    // What MkFinderV2p2 reads from the MkJob and the steering iterator; rz is set per seed by begin_layer().
    V2p2LayerCtx ctx{};
    ctx.is_outward = fwdSearch;
    ctx.ip = fwdSearch ? cfg.ipFwd : cfg.ipBkw;  // params_cur()
    ctx.capacity = cfg.capacity;
    ctx.vp = cfg.vp;
    ctx.param_b = Config::V2p2::Window::mini_prop_param_b == 1 ||
                  (Config::V2p2::Window::mini_prop_param_b == 2 && cfg.pc.interLayer.use_param_b_field);
    ctx.bfield = cfg.pc.material.bField;
    const auto pfInter = ::mkfitdev::prop::makePropagationFlags(cfg.pc.interLayer, cfg.pc.material);
    const auto pfIntra = ::mkfitdev::prop::makePropagationFlags(cfg.pc.intraLayer, cfg.pc.material);
    const int P = v2.preselStride;

    // V0 resets the list count of the next step's parity; this search's first step starts from zero
    ::mkfitdev::V2p2Counters* counters = v2.seeds.view().metadata().addressOf_counters();
    auto nActiveCands = cms::alpakatools::make_device_view(alpaka::getDev(queue), counters->nActiveCands, 2);
    alpaka::memset(queue, nActiveCands, 0);
    // V4 of an entry runs V0 of the next one in the same launch
    bool pulledIn = false;  // V0 of this step ran in the previous V4 launch
    for (int step = 0; step < tables.host.steps.nSteps; ++step) {
      if (!pulledIn)
        alpaka::exec<Acc1D>(queue,
                            workDivForSeeds(capacity),
                            KernelV2p2PullIn{},
                            bufs.seeds.view(),
                            bufs.slots.view(),
                            bufs.hots.view(),
                            v2.seeds.view(),
                            v2.cands.view(),
                            tables.steps.layerAt(step),
                            tables.steps.prevLayerAt(step),
                            in.geo,
                            ctx,
                            step,
                            capacity,
                            bufs.nRowsDev);
      pulledIn = false;
      if (tables.awakenOnly(step))
        continue;
      alpaka::exec<Acc1D>(queue,
                          workDivForCands(capacity),
                          KernelV2p2SelectHits{},
                          bufs.seeds.const_view(),
                          bufs.slots.const_view(),
                          v2.seeds.const_view(),
                          v2.cands.view(),
                          v2.layerHits.view(),
                          tables.steps.layerAt(step),
                          in,
                          ctx,
                          P,
                          step);
      alpaka::exec<Acc1D>(queue,
                          workDivForItems(capacity),
                          KernelV2p2InLayer{},
                          bufs.seeds.const_view(),
                          bufs.slots.const_view(),
                          v2.seeds.view(),
                          v2.cands.const_view(),
                          v2.layerHits.const_view(),
                          v2.nodes.view(),
                          in.hits,
                          cfg.vp,
                          pfInter,
                          pfIntra,
                          P,
                          v2.nodeCapacity,
                          step,
                          capacity,
                          bufs.nRowsDev);
      pulledIn = step + 1 < tables.host.steps.nSteps;
      alpaka::exec<Acc1D>(queue,
                          workDivForSeedGroups(capacity),
                          KernelV2p2Select{},
                          bufs.seeds.view(),
                          bufs.slots.view(),
                          bufs.hots.view(),
                          v2.seeds.view(),
                          v2.cands.view(),
                          v2.nodes.const_view(),
                          tables.steps.layerAt(step),
                          pulledIn ? tables.steps.layerAt(step + 1) : nullptr,
                          pulledIn ? tables.steps.prevLayerAt(step + 1) : nullptr,
                          in.geo,
                          ctx,
                          step,
                          capacity,
                          bufs.nRowsDev);
    }

    // the final sorting; m_job->params() (forward) in both directions
    V2p2RegionScorers scorers{};
    const auto& st = tables.host.steps;
    for (int r = 0; r < cfg.nRegions && r < ::mkfitdev::es::kMaxRegions; ++r) {
      scorers.scorer[r] = fwdSearch ? cfg.finalPickScorer[r] : cfg.searchScorer[r];
      scorers.hasPlan[r] = st.plans.hasPlan[r];
    }
    alpaka::exec<Acc1D>(queue,
                        workDivForSeeds(capacity),
                        KernelV2p2FinalPick{},
                        bufs.seeds.view(),
                        bufs.slots.view(),
                        bufs.hots.view(),
                        scorers,
                        cfg.ipFwd.maxCandsPerSeed,
                        fwdSearch,
                        capacity,
                        bufs.nRowsDev);
  }

  EngineSearchFn makeV2p2SearchFn(V2p2Buffers& v2,
                                  const V2p2StepTablesDevice& fwdTables,
                                  const V2p2StepTablesDevice& bkwTables,
                                  const V2p2EventInputs& in,
                                  const V2p2SearchConfig& cfg) {
    return [&v2, &fwdTables, &bkwTables, &in, &cfg](Queue& queue, EngineBuffers& bufs, bool fwdSearch, int capacity) {
      v2p2Search(queue, bufs, v2, fwdSearch ? fwdTables : bkwTables, fwdSearch, in, cfg, capacity);
    };
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev
