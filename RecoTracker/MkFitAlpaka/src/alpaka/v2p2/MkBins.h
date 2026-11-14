#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkBins_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkBins_h

// Portable copy of RecoTracker/MkFitCore/src/MkBins.h / .cc: the layer crossings, the window and the binnor ranges
// of the v2p2 search. Templated on N candidates (the search uses N = 1 per candidate on every backend); a Matriplex
// expression of MkFitCore is a per-slot loop over [0, N) with the same per-element arithmetic. No member
// initializers: it lives in block shared memory.

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/LayerOfHitsAccess.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexBackend.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/MiniPropagators.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkRZLimits.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // MkBinTrackCovExtract: position block of the track covariance at MkBins::m_sp2
  template <idx_t N>
  struct MkBinTrackCovExtract {
    float m_cov_0_0[N];
    float m_cov_0_1[N];
    float m_cov_1_1[N];
    float m_cov_2_2[N];
    float m_cov_0_2[N];
    float m_cov_1_2[N];

    // calc_err_xy(x, y) for slot i
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float calc_err_xy(float x, float y, int i) const;
  };

  // MkBinLimits (MPlexQUH q1, q2, p1, p2)
  template <idx_t N>
  struct MkBinLimits {
    uint16_t q1[N], q2[N], p1[N], p2[N];
  };

  template <idx_t N>
  struct MkBins {
    mini_propagators::InitialStatePlex<N> m_isp;
    mini_propagators::StatePlex<N> m_sp1, m_sp2;

    float m_phi_min[N], m_phi_max[N], m_phi_center[N], m_phi_delta[N];
    float m_q_min[N], m_q_max[N];

    float m_dphi_track[N], m_dq_track[N];

    int m_n_proc;
    bool m_is_barrel;

    // prop_to_limits_in_order (MkBins.cc)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void prop_to_limits_in_order(const MkRZLimits& ls);

    // transport_position_cov (MkBins.cc): the caller fills slots [m_n_proc, N) with a valid state
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void transport_position_cov(const MPlexLV<N>& par0,
                                                                    const MPlexLS<N>& err0,
                                                                    MkBinTrackCovExtract<N>& tce) const;

    // determine_bin_windows (MkBins.cc)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_bin_windows(const MkBinTrackCovExtract<N>& cov_ex);

    // find_bin_ranges (MkBins.cc)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void find_bin_ranges(const ::mkfitdev::LayerOfHitsAccess& loh,
                                                             MkBinLimits<N>& bl) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
