#ifndef RecoTracker_MkFitAlpaka_src_alpaka_select_SelectSoA_h
#define RecoTracker_MkFitAlpaka_src_alpaka_select_SelectSoA_h

// K2 (propagate to layer + selectHitIndicesV2 + find_tracks_handle_missed_layers) input and output SoAs.
// One row per listed candidate (the engine's dense (seed, ic) list, seed_cand_idx order).
//   SelListSoA   in : candidate slot row (CandSlotsSoA), layer
//   PropStateSoA out: state propagated to the layer (m_Par/m_Err[iP]) and the propagation fail flag
//   SelHitsSoA   out: selected hits (m_XHitArr order), raw WSR (the engine applies find_tracks_handle_missed_layers),
//                     in_gap, and the K2a -> K2b hand-off of the GPU group scan

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"

namespace mkfitdev {

  constexpr int kMaxSelHits = kMaxHitsPerCand;  // NEW_MAX_HIT of selectHitIndicesV2: every selected hit is an option

  struct PropState {
    float par[6];
    float err[21];  // packed lower triangle, MatriplexSym / CandState order
    int32_t fail;   // m_FailFlag after the inter-layer propagation
  };

  struct SelHits {
    int32_t hits[kMaxSelHits];  // original hit index within the layer (m_XHitArr), best ddphi first
    int8_t nHits;               // m_XHitSize
    int8_t wsrRaw;              // m_wsr as selectHitIndicesV2 left it (WSR_Failed kept)
    int8_t inGap;               // m_XWsrResult.m_in_gap
  };

  // K2a -> K2b hand-off: the MkFitCore Bins of the candidate and its preselection cuts
  struct SelScan {
    uint16_t q0, q1, q2, p1, p2;  // Bins q0/q1/q2/p1/p2 (bin_index_t)
    int8_t scan;                  // 1: the hit scan runs (not failed, not WSR_Outside)
    int8_t charge;
    float dqTrack;  // Bins dq_track
    float dphiCut;  // Bins dphi_track + DDPHI_PRESEL_FAC * 0.0123f
  };

  GENERATE_SOA_LAYOUT(SelListSoALayout,
                      SOA_COLUMN(int32_t, row),    // CandSlotsSoA row of the candidate
                      SOA_COLUMN(int16_t, layer),  // layer of the current plan step
                      SOA_SCALAR(int32_t, n))      // number of listed candidates

  GENERATE_SOA_LAYOUT(PropStateSoALayout, SOA_COLUMN(PropState, ps))
  GENERATE_SOA_LAYOUT(SelHitsSoALayout, SOA_COLUMN(SelHits, sel), SOA_COLUMN(SelScan, scan))

  using SelListSoA = SelListSoALayout<>;
  using PropStateSoA = PropStateSoALayout<>;
  using SelHitsSoA = SelHitsSoALayout<>;

}  // namespace mkfitdev

#endif
