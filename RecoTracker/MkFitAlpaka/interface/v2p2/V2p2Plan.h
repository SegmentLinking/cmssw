#ifndef RecoTracker_MkFitAlpaka_interface_v2p2_V2p2Plan_h
#define RecoTracker_MkFitAlpaka_interface_v2p2_V2p2Plan_h

// The layer plans of MkBuilder::findTracksStandardv2p2 with the regions in lock step, as host tables (host code).
// Same layout as the clone engine's makeEngineStepTables (interface/cands/CandEngineTypes.h), with one step more:
// for region r, pickup index P (fwd: m_fwd_search_pickup, else m_bkw_search_pickup) and walk direction w (+1 / -1),
//   layer[s * nRegions + r]     = plan[P + w * (s + 1)]  (the entry searched at step s), -1 past the plan end
//   prevLayer[s * nRegions + r] = plan[P + w * s]        (the entry whose end_layer() awakens seeds before step s:
//                                                        awaken_candidates() at setup for s = 0), -1 past the end
// nSteps = 1 + max over regions of the number of entries after the pickup. The last step of every region has
// layer -1 and only awakens (MkFinderV2p2::end_layer of the last entry also awakens dormant CombCandidates).
// planIndex[s * nRegions + r] = P + w * (s + 1), -1 past the end (the plan index of the entry).

#include <vector>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"

namespace mkfitdev {

  struct V2p2StepTablesHost {
    EngineStepTablesHost steps;
    std::vector<int16_t> planIndex;
  };

  inline V2p2StepTablesHost makeV2p2StepTables(const EnginePlan& plan, bool fwd);

}  // namespace mkfitdev

#endif
