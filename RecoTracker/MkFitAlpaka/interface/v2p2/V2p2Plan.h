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

#include <cstdint>
#include <vector>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"

namespace mkfitdev {

  struct V2p2StepTablesHost {
    EngineStepTablesHost steps;
    std::vector<int16_t> planIndex;
  };

  // The walk of SteeringParams::iterator (make_iterator(IT_FwdSearch / IT_BkwSearch), operator++) for every region. A
  // region without a backward-search plan (pickup -1) gets no step: findTracksStandardv2p2 skips it.
  inline V2p2StepTablesHost makeV2p2StepTables(const EnginePlan& plan, bool fwd) {
    V2p2StepTablesHost tables;
    EngineStepTablesHost& st = tables.steps;
    st.nRegions = int(plan.layers.size());
    st.plans = makeEngineRegionPlans(plan, fwd);
    const int dir = fwd ? 1 : -1;
    for (int region = 0; region < st.nRegions; ++region) {
      const int nLayers = int(plan.layers[region].size());
      const int pickup = fwd ? plan.fwdPickup[region] : plan.bkwPickup[region];
      if (pickup < 0 || pickup >= nLayers)
        continue;
      const int entries = fwd ? nLayers - 1 - pickup : pickup;
      st.nSteps = entries + 1 > st.nSteps ? entries + 1 : st.nSteps;
    }
    st.layer.assign(st.nSteps * st.nRegions, -1);
    st.prevLayer.assign(st.nSteps * st.nRegions, -1);
    tables.planIndex.assign(st.nSteps * st.nRegions, -1);
    for (int region = 0; region < st.nRegions; ++region) {
      const auto& pl = plan.layers[region];
      const int nLayers = int(pl.size());
      const int pickup = fwd ? plan.fwdPickup[region] : plan.bkwPickup[region];
      for (int step = 0; step < st.nSteps; ++step) {
        const int prev = pickup + dir * step;
        if (prev < 0 || prev >= nLayers)
          break;
        const int row = step * st.nRegions + region;
        st.prevLayer[row] = int16_t(pl[prev]);
        const int idx = prev + dir;
        if (idx >= 0 && idx < nLayers) {
          st.layer[row] = int16_t(pl[idx]);
          tables.planIndex[row] = int16_t(idx);
        }
      }
    }
    return tables;
  }

}  // namespace mkfitdev

#endif
