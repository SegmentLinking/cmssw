#ifndef RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandsEngine_h
#define RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandsEngine_h

// Host entry points of the clone engine. Kernels: src/alpaka/engine/EngineKernels.h,
// compiled once in src/alpaka/Cands.dev.cc (one TU for all candidate kernels).
//
// Per plan step (regions in lock step, per-region layer tables from makeEngineStepTables):
//   engineActivate (K1) -> K2 (select callback) -> engineStep (K3a, K3b, K4, K5); after the last step engineMerge (K6).
// engineSearch runs the whole plan loop with K2 supplied as a callback; engineRunChainAsync runs forward search ->
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
    // engineFilterCompactAsync(bufs, out) sets out.nRowsDev = bufs.nSurvivors.data().
    const int32_t* nRowsDev = nullptr;
  };

  struct EngineStepTablesDevice {
    EngineStepTablesDevice(Queue& queue, const ::mkfitdev::EngineStepTablesHost& host);
    int nSteps, nRegions;
    cms::alpakatools::device_buffer<Device, int16_t[]> layer, prevLayer;
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

  // K6: mergeCandsAndBestShortOne(update_score = true, sort = true) per seed.
  void engineMerge(Queue& queue, EngineBuffers& bufs, int maxCandsPerSeed, int nSeeds);

  // K2 callback: fills bufs.step->sel for the cands listed by K1 at the given step.
  // nSeeds = seed rows of bufs in this search (the backward search runs on fewer rows than the forward one).
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

  // filter_comb_cands (attempt_all_cands = true) + stable survivor compaction from bufs into out (seeds, slots, hots),
  // without host synchronization: the survivor count stays on the device (bufs.nSurvivors) and becomes out.nRowsDev;
  // rows [survivors, nSeeds) of out get nCands = 0. With bkwRep, the repack runs at most once per seed.
  void engineFilterCompactAsync(
      Queue& queue, EngineBuffers& bufs, EngineBuffers& out, bool bkwRep, int minHitsQF, int nSeeds);

  // compactifyHitStorageForBestCand + beginBkwSearch per seed.
  void engineCompactifyBeginBkw(Queue& queue,
                                EngineBuffers& bufs,
                                bool doCompactify,
                                bool doBeginBkw,
                                int nSeeds,
                                ::mkfitdev::BkwSearchGate const& gate = ::mkfitdev::BkwSearchGate{});

  // gateBkwSearch per seed (no-op when gate.minPixelLayers <= 0); after the backward search.
  void engineBkwSearchGate(Queue& queue, EngineBuffers& bufs, ::mkfitdev::BkwSearchGate const& gate, int nSeeds);

  // Backward-fit hook. Identity when empty.
  using EngineBackwardFitFn = std::function<void(Queue&, EngineBuffers&, int nSeeds)>;

  // Sync-free chain: no host synchronization at all. bufs.nRowsDev = device count of the imported seed rows
  // (e.g. &SeedSoA nKept), capacity = rows of bufs and work. The result is in the returned buffers (bufs), with
  // *bufs.nRowsDev (= work.nSurvivors) rows. minHitsQFPre / Post = params / backward_params minHitsQF (MkFitCore
  // params_cur of the pre and post filter). The post-filter is the ONLY repack: export without a filter.
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
                                     const ::mkfitdev::BkwSearchGate& gate = ::mkfitdev::BkwSearchGate{});

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
