#ifndef RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandsEngine_h
#define RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandsEngine_h

// Host entry points of the clone engine. Kernels: src/alpaka/engine/EngineKernels.h,
// compiled once in src/alpaka/Cands.dev.cc (one TU for all candidate kernels).
//
// Per plan step t (regions in lock step, per-region layer tables from makeEngineStepTables):
//   engineActivate (K1)  ->  ->  engineStep (K3a, K3b,
//   K4, K5).  After the last step: engineMerge (K6).
// engineSearch runs the whole plan loop with K2 supplied as a callback; engineRunChain runs forward search ->
// pre-filter -> compaction -> compactify -> [backward fit callback] -> beginBkwSearch -> backward search ->
// post-filter with repack -> compaction (endBkwSearch is a host flag only).

#include <functional>
#include <memory>
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

  // Device storage of the clone engine for up to maxSeeds seeds. Buffer sets whose searches run one after the other
  // on the same queue can share their step scratch.
  struct EngineBuffers {
    EngineBuffers(Queue& queue, int maxSeeds, int hotsPerSeed, std::shared_ptr<EngineStepScratch> sharedStep = nullptr);
    int maxSeeds;
    SeedCandsDeviceCollection seeds;
    CandSlotsDeviceCollection slots;
    CandHotsDeviceCollection hots;
    std::shared_ptr<EngineStepScratch> step;
    cms::alpakatools::device_buffer<Device, int8_t[]> passed;
    cms::alpakatools::device_buffer<Device, int32_t[]> newIdx;
    cms::alpakatools::device_buffer<Device, int32_t> nSurvivors;
    // Device count of the seed rows in use: kernels launched with a capacity-sized grid
    // process rows [0, *nRowsDev). nullptr: the host count passed to each call.
    // engineFilterCompactAsync(b, out) sets out.nRowsDev = b.nSurvivors.data().
    const int32_t* nRowsDev = nullptr;
  };

  struct EngineStepTablesDevice {
    EngineStepTablesDevice(Queue& queue, const ::mkfitdev::EngineStepTablesHost& h);
    int nSteps, nRegions;
    cms::alpakatools::device_buffer<Device, int16_t[]> layer, prevLayer;
    const int16_t* layerAt(int t) const { return layer.data() + t * nRegions; }
    const int16_t* prevLayerAt(int t) const { return prevLayer.data() + t * nRegions; }
  };

  // Propagation flags of the search: PropagationConfig finding_inter/intra_layer_pflags as the ES carries them,
  // the device material map, finding_requires_propagation_to_hit_pos. Phase-2 LST step: both use_param_b_field |
  // apply_material, propToHit = true. Turned into prop::PropagationFlags in Cands.dev.cc by the shared adapter
  //.
  struct EnginePropConfig {
    ::mkfitdev::PropFlags interLayer;
    ::mkfitdev::PropFlags intraLayer;
    ::mkfitdev::MaterialView material;
    bool propToHit;
  };

  // K1 for one step.
  void engineActivate(Queue& queue,
                      EngineBuffers& b,
                      const int16_t* dStepLayer,
                      const int16_t* dStepPrevLayer,
                      bool fwdSearch,
                      float minPtCut,
                      int nSeeds);

  // K3a, K3b, K4, K5 for one step (after K2 filled b.sel).
  void engineStep(Queue& queue,
                  EngineBuffers& b,
                  const ::mkfitdev::EngineHitInputs& in,
                  const EnginePropConfig& pc,
                  const ::mkfitdev::EngineIterParams& ip,
                  int nSeeds);

  // K6: mergeCandsAndBestShortOne(update_score = true, sort = true) per seed.
  void engineMerge(Queue& queue, EngineBuffers& b, int maxCandsPerSeed, int nSeeds);

  // K2 callback: fills b.sel for the cands listed by K1 at step t.
  // nSeeds = seed rows of b in this search (the backward search runs on fewer rows than the forward one).
  using EngineSelectFn = std::function<void(Queue&, EngineBuffers&, int t, int nSeeds)>;

  // The plan loop (find_tracks_in_layers for all regions in lock step) + K6.
  void engineSearch(Queue& queue,
                    EngineBuffers& b,
                    const EngineStepTablesDevice& tables,
                    bool fwdSearch,
                    const EngineSelectFn& select,
                    const ::mkfitdev::EngineHitInputs& in,
                    const EnginePropConfig& pc,
                    const ::mkfitdev::EngineIterParams& ip,
                    int nSeeds);

  // filter_comb_cands (attempt_all_cands = true) + stable survivor compaction from b into out (seeds, slots, hots).
  // Returns the number of surviving seeds (synchronizes the queue).
  int engineFilterCompact(Queue& queue, EngineBuffers& b, EngineBuffers& out, bool bkwRep, int minHitsQF, int nSeeds);
  // The same without any host synchronization: the survivor count stays on the device (b.nSurvivors) and becomes
  // out.nRowsDev; rows [survivors, nSeeds) of out get nCands = 0. With bkwRep, the repack runs at most once per seed
  //.
  void engineFilterCompactAsync(
      Queue& queue, EngineBuffers& b, EngineBuffers& out, bool bkwRep, int minHitsQF, int nSeeds);

  // compactifyHitStorageForBestCand + beginBkwSearch per seed.
  void engineCompactifyBeginBkw(Queue& queue,
                                EngineBuffers& b,
                                bool removeSeedHits,
                                int backwardFitMinHits,
                                bool doCompactify,
                                bool doBeginBkw,
                                int nSeeds,
                                ::mkfitdev::BkwSearchGate const& gate = ::mkfitdev::BkwSearchGate{});

  // gateBkwSearch per seed (no-op when gate.minPixelLayers <= 0); after the backward search.
  void engineBkwSearchGate(Queue& queue, EngineBuffers& b, ::mkfitdev::BkwSearchGate const& gate, int nSeeds);

  // Backward-fit hook. Identity when empty.
  using EngineBackwardFitFn = std::function<void(Queue&, EngineBuffers&, int nSeeds)>;

  // Sync-free chain: no host synchronization at all. b.nRowsDev = device count of the imported seed rows
  // (e.g. &SeedSoA nKept), capacity = rows of b and work. The result is in the returned buffers (b), with
  // *b.nRowsDev (= work.nSurvivors) rows. minHitsQFPre / Post = params / backward_params minHitsQF (MkFitCore
  // params_cur of the pre and post filter). The post-filter is the ONLY repack: export without a filter.
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
                                     const ::mkfitdev::BkwSearchGate& gate = ::mkfitdev::BkwSearchGate{});

  // Whole clone-engine chain of the LST step. b holds the imported seeds (nSeeds rows); work is a second set of
  // buffers of the same capacity used for the compactions. Returns the buffers holding the result (b or work) and
  // the final number of seeds in nOut.
  EngineBuffers& engineRunChain(Queue& queue,
                                EngineBuffers& b,
                                EngineBuffers& work,
                                const EngineStepTablesDevice& fwdTables,
                                const EngineStepTablesDevice& bkwTables,
                                const EngineSelectFn& selectFwd,
                                const EngineSelectFn& selectBkw,
                                const EngineBackwardFitFn& backwardFit,
                                const ::mkfitdev::EngineHitInputs& inFwd,  // layer table with forward windows
                                const ::mkfitdev::EngineHitInputs& inBkw,  // ... backward windows (switch_to_backward)
                                const EnginePropConfig& pc,
                                const ::mkfitdev::EngineIterParams& ipFwd,
                                const ::mkfitdev::EngineIterParams& ipBkw,
                                int minHitsQF,
                                int backwardFitMinHits,
                                int nSeeds,
                                int& nOut);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
