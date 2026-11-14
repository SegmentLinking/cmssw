#ifndef RecoTracker_MkFitAlpaka_interface_v2p2_V2p2Config_h
#define RecoTracker_MkFitAlpaka_interface_v2p2_V2p2Config_h

// Portable copy of RecoTracker/MkFitCore/src/V2p2Config.h: the switches of MkFinderV2p2 and of the v2p2 window in
// MkBins, constexpr as in the CMSSW build of MkFitCore. src/esFill.cc checks every value against MkFitCore.
// The per-iteration tunings are V2p2Params (interface/v2p2/V2p2Params.h), carried by the ES product.

#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Params.h"

namespace mkfitdev {

  namespace Config::V2p2 {

    // Pre-selection window, design notes "Search window".
    namespace Window {
      constexpr int mini_prop_param_b = 2;
      constexpr float dq_trk_fac = 1.5f;
      constexpr float dq_hit_fac = 1.2f;
      constexpr bool surface_q = true;
      constexpr int q_extra_bins = 1;

      constexpr float dphi_trk_fac = 2.0f;
      constexpr bool phi_per_hit = true;
      constexpr float dphi_hit_fac = 3.0f;
      constexpr float dphi_flat_rad = 2.0f * 0.0123f;
      constexpr int phi_extra_bins = 0;
    }  // namespace Window

    // Line pre-cut in select_hits(), design notes "Line pre-cut".
    namespace PreCut {
      constexpr bool q = true;
      constexpr bool phi = true;
      constexpr float dq_slack = 2.0f;
      constexpr float dphi_slack = 1.5f;
      constexpr float qbar_fac = 1.2f;
    }  // namespace PreCut

    // In-layer combinatorial search and end-of-layer selection.
    namespace InLayer {
      constexpr int max_presel_hits_limit = V2p2Params::max_presel_hits_limit;
      constexpr int max_hits_per_layer_limit = V2p2Params::max_hits_per_layer_limit;
      constexpr bool reserve_hole_slot = false;
      constexpr bool best_short = false;
      // MkFitCore's early selection only frees arena nodes sooner; the device selects at the end of every entry.
      constexpr bool early_select = true;
      constexpr int late_max_cands = 0;
      constexpr int late_max_cands_hits = 10;
      constexpr float late_max_cands_step_cm = 0.0f;
      constexpr bool sister_hole_drop = false;
      constexpr float sister_hole_chi2 = 0.0f;
    }  // namespace InLayer

    // Per-layer candidate policy and hit acceptance.
    namespace Policy {
      constexpr bool use_wsr = true;
      constexpr float wsr_n_sigma = 5.0f;
      constexpr bool use_hole_limits = true;
      constexpr bool use_stop_cuts = true;
      constexpr float final_pick_hole_penalty = 8.0f;
      constexpr float final_pick_tail_penalty = 3.0f;
      constexpr float looper_max_angle = Const::PIOver2 - 0.2f;
      constexpr float looper_max_pt = 1.2f;
      constexpr float looper_min_r = 25.0f;
    }  // namespace Policy

    // Layer-step score.
    namespace Score {
      constexpr int mode = 1;
      constexpr bool use_rho = true;
      constexpr float rho_const = 0.0f;
      constexpr bool use_detv = true;
      constexpr float detv_const = 0.0f;
    }  // namespace Score

  }  // namespace Config::V2p2

}  // namespace mkfitdev

#endif
