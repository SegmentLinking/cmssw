#ifndef RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandsEngine_h
#define RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandsEngine_h

// Host entry points of the clone engine. Kernels: src/alpaka/engine/EngineKernels.h,
// compiled once in src/alpaka/Cands.dev.cc (one TU for all candidate kernels).
//
// Per plan step (regions in lock step, per-region layer tables from makeEngineStepTables):
//   engineActivate (K1) -> K2 (select callback) -> engineStep (K3a, K3b, K4, K5); after the last step engineMerge (K6).
// engineSearch runs the whole plan loop with K2 supplied as a callback; engineRunChainAsync runs forward search ->
// pre-filter -> compactify -> [backward fit callback] -> beginBkwSearch -> backward search -> post-filter with repack
// and the backward-search gate (endBkwSearch is a host flag only), on one set of seed rows, with the clone engine's
// searches or with the two searches supplied (EngineSearchFn: MkFinderV2p2, interface/v2p2/alpaka).

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSelection.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsDeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/MaterialView.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // Per-step scratch of the clone engine (K1 resets it at every step).
  struct EngineStepScratch {
    EngineStepScratch(Queue& queue, int maxSeeds);
    CandOptionsDeviceCollection opts;
    CandExtrasDeviceCollection extras;
    CandUpdatesDeviceCollection upds;
    CandSelHitsDeviceCollection sel;
    CandHitChi2DeviceCollection c2;
  };

  // Device storage of the clone engine for up to maxSeeds seeds, with the HoT pool of hotPool (CandsSoA.h).
  struct EngineBuffers {
    EngineBuffers(Queue& queue, int maxSeeds, ::mkfitdev::HotPoolLayout const& hotPool);
    // Candidate books only, without the clone engine's per-step scratch (step == nullptr): the v2p2 search.
    struct NoStepScratch {};
    EngineBuffers(Queue& queue, int maxSeeds, ::mkfitdev::HotPoolLayout const& hotPool, NoStepScratch);
    int maxSeeds;
    ::mkfitdev::HotPoolLayout hotPool;
    SeedCandsDeviceCollection seeds;
    CandSlotsDeviceCollection slots;
    CandHotsDeviceCollection hots;
    std::unique_ptr<EngineStepScratch> step;
    // Device count of the seed rows in use: kernels launched with a capacity-sized grid
    // process rows [0, *nRowsDev). nullptr: the host count passed to each call.
    const int32_t* nRowsDev = nullptr;
    // CPU backends: the rows the pre-filter kept, then -1 (rowList; nullptr before it), read by the kernels that batch
    // several seeds (K3a, K5, the backward fit), so that a batch holds the seeds MkFitCore batches.
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> keptRows;
    const int32_t* rowList = nullptr;
  };

  struct EngineStepTablesDevice {
    EngineStepTablesDevice(Queue& queue, const ::mkfitdev::EngineStepTablesHost& host);
    int nSteps, nRegions;
    cms::alpakatools::device_buffer<Device, int16_t[]> layer, prevLayer;
    ::mkfitdev::EngineRegionPlans plans;
    const int16_t* layerAt(int step) const { return layer.data() + step * nRegions; }
    const int16_t* prevLayerAt(int step) const { return prevLayer.data() + step * nRegions; }
  };

  // Propagation flags of the search: PropagationConfig finding_inter/intra_layer_pflags as the ES carries them, the
  // device material map and finding_requires_propagation_to_hit_pos (made into prop::PropagationFlags in Cands.dev.cc).
  struct EnginePropConfig {
    ::mkfitdev::PropFlags interLayer;
    ::mkfitdev::PropFlags intraLayer;
    ::mkfitdev::MaterialView material;
    bool propToHit;
  };

  // K1 for one step.
  void engineActivate(Queue& queue,
                      EngineBuffers& bufs,
                      const int16_t* dStepLayer,
                      const int16_t* dStepPrevLayer,
                      bool fwdSearch,
                      float minPtCut,
                      int nSeeds);

  // K3a, K3b, K4, K5 for one step (after K2 filled bufs.step->sel).
  void engineStep(Queue& queue,
                  EngineBuffers& bufs,
                  const ::mkfitdev::EngineHitInputs& in,
                  const EnginePropConfig& pc,
                  const ::mkfitdev::EngineIterParams& ip,
                  int nSeeds);

  // K6: the final sorting per seed of the regions the search runs on: addPreBkwSearchHoles (backward search only), then
  // mergeCandsAndBestShortOne(update_score = true, sort = true).
  void engineMerge(Queue& queue,
                   EngineBuffers& bufs,
                   const ::mkfitdev::EngineRegionPlans& plans,
                   int maxCandsPerSeed,
                   bool fwdSearch,
                   int nSeeds);

  // MkBuilder::skipBkwSearchRegion for every region of the backward search without a plan (nothing to launch when every
  // region has one): addPreBkwSearchHoles for its seeds.
  void engineSkipBkwSearchRegions(
      Queue& queue, EngineBuffers& bufs, const ::mkfitdev::EngineRegionPlans& plans, int nRegions, int nSeeds);

  // K2 callback: fills bufs.step->sel for the cands listed by K1 at the given step.
  // nSeeds = seed rows of bufs in this search.
  using EngineSelectFn = std::function<void(Queue&, EngineBuffers&, int step, int nSeeds)>;

  // The plan loop (find_tracks_in_layers for all regions in lock step) + K6.
  void engineSearch(Queue& queue,
                    EngineBuffers& bufs,
                    const EngineStepTablesDevice& tables,
                    bool fwdSearch,
                    const EngineSelectFn& select,
                    const ::mkfitdev::EngineHitInputs& in,
                    const EnginePropConfig& pc,
                    const ::mkfitdev::EngineIterParams& ip,
                    int nSeeds);

  // filter_comb_cands (attempt_all_cands = true) in place, without host synchronization: a failing seed's row is
  // left without candidates (Finished), which the later kernels and the export skip, so the survivors keep their
  // order. With bkwRep, the repack and the backward-search gate (on when gate.minPixelLayers > 0) run at most once
  // per seed.
  void engineFilterAsync(Queue& queue,
                         EngineBuffers& bufs,
                         bool bkwRep,
                         int minHitsQF,
                         int nSeeds,
                         ::mkfitdev::BkwSearchGate const& gate = ::mkfitdev::BkwSearchGate{});

  // compactifyHitStorageForBestCand + beginBkwSearch per seed (with the gate on: its threshold and the pre-search
  // candidate, taken before CombCandidate::beginBkwSearch).
  void engineCompactifyBeginBkw(Queue& queue,
                                EngineBuffers& bufs,
                                bool doCompactify,
                                bool doBeginBkw,
                                int nSeeds,
                                ::mkfitdev::BkwSearchGate const& gate = ::mkfitdev::BkwSearchGate{});

  // Backward-fit hook. Identity when empty.
  using EngineBackwardFitFn = std::function<void(Queue&, EngineBuffers&, int nSeeds)>;

  // A whole search of run_OneIteration (the builder's FindTracks for one direction, its final sorting included), on
  // the seed rows [0, *bufs.nRowsDev) of bufs: engineSearch with a K2 select for the clone engine, v2p2Search
  // (interface/v2p2/alpaka/V2p2Search.h) for MkFinderV2p2.
  using EngineSearchFn = std::function<void(Queue&, EngineBuffers&, bool fwdSearch, int capacity)>;

  // Sync-free chain: no host synchronization at all. bufs.nRowsDev = device count of the imported seed rows
  // (e.g. &SeedSoA nKept), capacity = rows of bufs. The result is in bufs, in place: rows [0, *bufs.nRowsDev), the
  // seeds that failed a filter without candidates. minHitsQFPre / Post = params / backward_params minHitsQF (MkFitCore
  // params_cur of the pre and post filter). The post-filter is the ONLY repack: export without a filter.
  void engineRunChainAsync(Queue& queue,
                           EngineBuffers& bufs,
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
                           const ::mkfitdev::BkwSearchGate& gate = ::mkfitdev::BkwSearchGate{});

  // The same chain with the searches supplied (searchFwd / searchBkw run the forward and the backward search, final
  // sorting included); the clone-engine overload above is this one with engineSearch and its K2 selects.
  void engineRunChainAsync(Queue& queue,
                           EngineBuffers& bufs,
                           const EngineSearchFn& searchFwd,
                           const EngineSearchFn& searchBkw,
                           const EngineBackwardFitFn& backwardFit,
                           int minHitsQFPre,
                           int minHitsQFPost,
                           int capacity,
                           const ::mkfitdev::BkwSearchGate& gate = ::mkfitdev::BkwSearchGate{});

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
