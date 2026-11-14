#ifndef RecoTracker_MkFitAlpaka_interface_v2p2_V2p2Params_h
#define RecoTracker_MkFitAlpaka_interface_v2p2_V2p2Params_h

// Device copy of RecoTracker/MkFitCore/interface/V2p2Params.h: the MkFinderV2p2 tunings of one iteration
// (mkfit::IterationConfig::m_v2p2, set by MkFitIterationConfigESProducer from its PSet "v2p2"). Plain PODs without
// member initializers: ESConfig is zero-filled and src/esFill.cc copies every member from MkFitCore.

namespace mkfitdev {

  // mkfit::V2p2ScoreParams
  struct V2p2ScoreParams {
    static constexpr int n_hit_eff_grp = 5;

    float hit_bonus;
    float overlap_bonus;
    float chi2_weight;
    float miss_penalty;
    float edge_penalty;
    float gap_penalty;
    float stop_penalty;
    float hit_eff;
    float hit_eff_grp[n_hit_eff_grp];
  };

  // mkfit::V2p2Params
  struct V2p2Params {
    static constexpr int max_presel_hits_limit = 12;
    static constexpr int max_hits_per_layer_limit = 8;

    V2p2ScoreParams score_outward;
    V2p2ScoreParams score_inward;

    float chi2_trk_fac;
    float hit_chi2_cut;

    int max_presel_hits;
    int max_hits_per_layer;

    constexpr const V2p2ScoreParams& score(bool is_outward) const { return is_outward ? score_outward : score_inward; }
  };

}  // namespace mkfitdev

#endif
