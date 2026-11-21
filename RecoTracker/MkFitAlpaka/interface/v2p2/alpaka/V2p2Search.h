#ifndef RecoTracker_MkFitAlpaka_interface_v2p2_alpaka_V2p2Search_h
#define RecoTracker_MkFitAlpaka_interface_v2p2_alpaka_V2p2Search_h

// Host entry points of the v2p2 search (MkBuilder::findTracksStandardv2p2 for all regions in lock step, then the
// final sorting). Kernels: src/alpaka/v2p2/V2p2Kernels.h, instantiated in src/alpaka/V2p2.dev.cc.
// Per step of the plan tables (interface/v2p2/V2p2Plan.h):
//   V0 KernelV2p2PullIn  -> V2 KernelV2p2SelectHits -> V3 KernelV2p2InLayer (depth 0 .. max_hits_per_layer - 1)
//   -> V4 KernelV2p2Select
// and after the last step V5 KernelV2p2FinalPick. No host synchronization: the kernels read the seed rows in use from
// bufs.nRowsDev, the node counts from V2p2Counters.

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsEngine.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Params.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Plan.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/alpaka/V2p2DeviceCollections.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // The lock-step plan of one search direction on the device, with the host copy the driver loop walks.
  struct V2p2StepTablesDevice {
    V2p2StepTablesDevice(Queue& queue, const ::mkfitdev::V2p2StepTablesHost& host);
    ::mkfitdev::V2p2StepTablesHost host;
    EngineStepTablesDevice steps;
    // true if no region has an entry to search at this step (an awaken-only step: only V0 runs)
    bool awakenOnly(int step) const;
  };

  // Event inputs of the v2p2 kernels (all device views / pointers)
  struct V2p2EventInputs {
    ::mkfitdev::EngineHitInputs hits;            // hit positions, errors, packed, module planes, layer table
    ::mkfitdev::LayerSoA::ConstView eohLayers;   // EventOfHits binning: layers
    ::mkfitdev::BinnedHitSoA::ConstView binned;  // EventOfHits binning: hits in internal order
    ::mkfitdev::BinSoA::ConstView bins;          // EventOfHits binning: bins
    ::mkfitdev::LayerInfoSoA::ConstView geo;     // ES layer table (LayerInfo)
  };

  // Iteration configuration of the v2p2 search (built once per IOV from the ES product)
  struct V2p2SearchConfig {
    ::mkfitdev::V2p2Params vp;
    EnginePropConfig pc;                 // finding_inter / intra_layer_pflags, material, field constants
    ::mkfitdev::EngineIterParams ipFwd;  // params
    ::mkfitdev::EngineIterParams ipBkw;  // backward_params
    int capacity;                        // MkJob::max_max_cands()
    int nRegions;
    ::mkfitdev::TrackScorer finalPickScorer[::mkfitdev::es::kMaxRegions];  // SteeringRegion::final_pick_track_scorer
    ::mkfitdev::TrackScorer searchScorer[::mkfitdev::es::kMaxRegions];     // SteeringRegion::track_scorer
  };

  // findTracksStandardv2p2(fwd ? IT_FwdSearch : IT_BkwSearch) on the seed rows [0, *bufs.nRowsDev) (capacity rows)
  // of bufs, scratch in v2: the plan loop and the final sorting (addPreBkwSearchHoles for the backward search, then
  // mergeCandsAndBestShortOne with the final-pick scorer forward, the search scorer backward).
  void v2p2Search(Queue& queue,
                  EngineBuffers& bufs,
                  V2p2Buffers& v2,
                  const V2p2StepTablesDevice& tables,
                  bool fwdSearch,
                  const V2p2EventInputs& in,
                  const V2p2SearchConfig& cfg,
                  int capacity);

  // The EngineSearchFn of engineRunChainAsync running v2p2Search (forward or backward tables by fwdSearch). The
  // referenced objects must outlive the chain's host calls.
  EngineSearchFn makeV2p2SearchFn(V2p2Buffers& v2,
                                  const V2p2StepTablesDevice& fwdTables,
                                  const V2p2StepTablesDevice& bkwTables,
                                  const V2p2EventInputs& in,
                                  const V2p2SearchConfig& cfg);

  // Zeroes the event counters of v2 (V2p2Counters) on the queue; once per event before the forward search.
  void v2p2ResetCounters(Queue& queue, V2p2Buffers& v2);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
