#ifndef RecoTracker_MkFitAlpaka_interface_v2p2_V2p2Score_h
#define RecoTracker_MkFitAlpaka_interface_v2p2_V2p2Score_h

// Portable copy of RecoTracker/MkFitCore/src/V2p2Score.h: the MkFinderV2p2 layer-step score. Same operations in the
// same order, std::log per call. The track scorer "v2p2:llh" is with the other scorers in interface/cands/CandTypes.h.

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/math/HitConstants.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Params.h"

namespace mkfitdev {

  // V2p2HoleKind_e
  enum V2p2HoleKind_e : int16_t { V2P2_NoHole = 0, V2P2_Miss, V2P2_Edge, V2P2_Gap, V2P2_Stop, V2P2_HoleKind_Count };

  // LayerStepFeatures, same members and order
  struct LayerStepFeatures {
    float chi2_sum;
    float log_det_v_sum;
    float log_rho;
    int16_t n_hits;
    int16_t n_overlap;
    int16_t hole_kind;

    int16_t layer_to;
    bool is_outward;
    bool is_pixel;
    bool is_barrel;
  };

  // v2p2_hit_eff_group (V2p2Score.h)
  ALPAKA_FN_HOST_ACC inline int v2p2_hit_eff_group(const LayerStepFeatures& f);

  // v2p2_layer_step_loglh (V2p2Score.h)
  ALPAKA_FN_HOST_ACC inline float v2p2_layer_step_loglh(const LayerStepFeatures& f, const V2p2ScoreParams& p);

  // v2p2_layer_step_score (V2p2Score.h)
  ALPAKA_FN_HOST_ACC inline float v2p2_layer_step_score(const LayerStepFeatures& f, const V2p2Params& vp);

  // v2p2_hole_kind (V2p2Score.h)
  ALPAKA_FN_HOST_ACC inline V2p2HoleKind_e v2p2_hole_kind(int fake_hit_idx);

}  // namespace mkfitdev

#endif
