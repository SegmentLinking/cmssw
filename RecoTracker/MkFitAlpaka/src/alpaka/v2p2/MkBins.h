#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkBins_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkBins_h

// Portable copy of RecoTracker/MkFitCore/src/MkBins.h / .cc: the layer crossings, the window and the binnor ranges
// of the v2p2 search. Templated on N candidates (a LayerBatch: kNN on CPU backends, one on GPU backends); a Matriplex
// expression of MkFitCore is a per-slot loop over [0, N) with the same per-element arithmetic. No member
// initializers: it lives in block shared memory.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/LayerOfHitsAccess.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexBackend.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/MiniPropagators.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkRZLimits.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using idx_t = ::mkfitdev::Matriplex::idx_t;

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
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float calc_err_xy(float x, float y, int i) const {
      return x * x * m_cov_0_0[i] + y * y * m_cov_1_1[i] + 2.0f * x * y * m_cov_0_1[i];
    }
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

    float m_phi1[N], m_phi2[N];  // phi at m_sp1 and m_sp2, from determine_bin_windows()
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

    // find_bin_ranges (MkBins.cc). dphi_trk_fac, dq_trk_fac: the track terms of the window, see V2p2Params; <= 0 takes
    // Window::dphi_trk_fac, Window::dq_trk_fac.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void find_bin_ranges(const ::mkfitdev::LayerOfHitsAccess& loh,
                                                             MkBinLimits<N>& bl,
                                                             float dphi_trk_fac = -1.0f,
                                                             float dq_trk_fac = -1.0f) const;
  };

  //==============================================================================

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MkBins<N>::prop_to_limits_in_order(const MkRZLimits& ls) {
    // m_isp is at the previous hit. m_sp1 is the crossing of the near bounding
    // surface of the layer, m_sp2 of the far one; m_isp is re-initialised at
    // m_sp1 on the way. dalpha accumulates, so both are measured from the
    // previous hit, which transport_position_cov() relies on.
    m_is_barrel = ls.m_is_barrel;
    float q[N];
    if (m_is_barrel) {
      float r;
      r = ls.m_is_outward ? ls.m_rin : ls.m_rout;
      for (int i = 0; i < N; ++i)
        q[i] = r;
      m_isp.propagate_to_r(q, m_sp1);
      m_isp = m_sp1;
      r = ls.m_is_outward ? ls.m_rout : ls.m_rin;
      for (int i = 0; i < N; ++i)
        q[i] = r;
      m_isp.propagate_to_r(q, m_sp2);
    } else {
      float z1 = ls.m_zmin, z2 = ls.m_zmax;
      bool is_pos = ls.m_layer_type == static_cast<int>(::mkfitdev::LayerType::EndCapPos);
      if ((is_pos && !ls.m_is_outward) || (!is_pos && ls.m_is_outward)) {
        const float t = z1;
        z1 = z2;
        z2 = t;
      }

      for (int i = 0; i < N; ++i)
        q[i] = z1;
      m_isp.propagate_to_z(q, m_sp1);
      m_isp = m_sp1;
      for (int i = 0; i < N; ++i)
        q[i] = z2;
      m_isp.propagate_to_z(q, m_sp2);
    }
  }

  //----------------------------------------------------------------------------
  // transport_position_cov() -- position block of the covariance at m_sp2,
  // J C0 J^T with J = [P_in | dx/d(ipt, phi, theta)] at fixed path length,
  // curvilinear at both ends. See doc/MkFinderV2p2-DesignNotes.md, "Track
  // covariance at the layer".
  //----------------------------------------------------------------------------

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MkBins<N>::transport_position_cov(const MPlexLV<N>& par0,
                                                                             const MPlexLS<N>& err0,
                                                                             MkBinTrackCovExtract<N>& tce) const {
    // All N slots; the caller fills unused ones with a valid state. Three slot loops: the exact f1, f2 of the larger
    // turning angles in a scalar loop between them, as MkFitCore, so that the other two vectorize.
    float iptinv[N], sp[N], cp[N], st[N], ct[N], cot[N], k[N], dx[N], dy[N], ak[N], f1[N], f2[N];
    MPLEX_SIMD
    for (int i = 0; i < N; ++i) {
      const float ipt = par0.constAt(i, 3, 0);
      iptinv[i] = 1.0f / ipt;
      vdt::fast_sincosf(par0.constAt(i, 4, 0), sp[i], cp[i]);
      vdt::fast_sincosf(par0.constAt(i, 5, 0), st[i], ct[i]);
      cot[i] = ct[i] / st[i];

      const float a = m_sp2.dalpha[i];
      k[i] = 1.0f / m_isp.inv_k[i];
      dx[i] = m_sp2.x[i] - par0.constAt(i, 0, 0);
      dy[i] = m_sp2.y[i] - par0.constAt(i, 1, 0);
      ak[i] = a * k[i];

      // f1 = a cos a - sin a, f2 = a sin a - (1 - cos a), by series below |a| = 0.25.
      const float a2 = a * a;
      f1[i] = a * a2 * (-1.0f / 3.0f + a2 * (1.0f / 30.0f - a2 * (1.0f / 840.0f)));
      f2[i] = a2 * (0.5f - a2 * (1.0f / 8.0f - a2 * (1.0f / 144.0f)));
    }
    for (int i = 0; i < N; ++i) {
      const float a = m_sp2.dalpha[i];
      if (i < m_n_proc) {
        if (std::abs(a) >= 0.25f) {
          const float sa = std::sin(a), ca = std::cos(a);
          f1[i] = a * ca - sa;
          f2[i] = a * sa - (1.0f - ca);
        }
      }
    }
    MPLEX_SIMD
    for (int i = 0; i < N; ++i) {
      const float kpx = k[i] * cp[i] * iptinv[i] * iptinv[i];
      const float kpy = k[i] * sp[i] * iptinv[i] * iptinv[i];

      // J = [P_in | D], 3 x 6, over (x, y, z, ipt, phi, theta). P_in = 1 - n0 n0^T.
      const float n0[3] = {cp[i] * st[i], sp[i] * st[i], ct[i]};
      float J[3][6];
      MPLEX_UNROLL
      for (int r = 0; r < 3; ++r) {
        MPLEX_UNROLL
        for (int c = 0; c < 3; ++c)
          J[r][c] = (r == c ? 1.0f : 0.0f) - n0[r] * n0[c];
      }
      J[0][3] = kpx * f1[i] - kpy * f2[i];
      J[0][4] = -dy[i];
      J[0][5] = ak[i] * m_sp2.px[i] * cot[i];
      J[1][3] = std::fma(kpy, f1[i], kpx * f2[i]);  // the product GCC and nvcc fuse in kpy * f1 + kpx * f2
      J[1][4] = dx[i];
      J[1][5] = ak[i] * m_sp2.py[i] * cot[i];
      J[2][3] = 0.0f;
      J[2][4] = 0.0f;
      J[2][5] = -ak[i] * iptinv[i];

      // M = J C0, then C = M J^T.
      float E[6][6];
      MPLEX_UNROLL
      for (int m = 0; m < 6; ++m) {
        MPLEX_UNROLL
        for (int c = m; c < 6; ++c)
          E[m][c] = E[c][m] = err0.constAt(i, m, c);
      }
      float M[3][6];
      MPLEX_UNROLL
      for (int r = 0; r < 3; ++r) {
        MPLEX_UNROLL
        for (int c = 0; c < 6; ++c) {
          M[r][c] = J[r][0] * E[0][c];
          MPLEX_UNROLL
          for (int m = 1; m < 6; ++m)
            M[r][c] += J[r][m] * E[m][c];
        }
      }
      float C[3][3];
      MPLEX_UNROLL
      for (int r = 0; r < 3; ++r) {
        MPLEX_UNROLL
        for (int c = r; c < 3; ++c) {
          C[r][c] = M[r][0] * J[c][0];
          MPLEX_UNROLL
          for (int m = 1; m < 6; ++m)
            C[r][c] += M[r][m] * J[c][m];
          C[c][r] = C[r][c];
        }
      }

      // Project onto the plane normal to the momentum at m_sp2: P C P.
      const float pinv =
          1.0f / std::sqrt(m_sp2.px[i] * m_sp2.px[i] + m_sp2.py[i] * m_sp2.py[i] + m_sp2.pz[i] * m_sp2.pz[i]);
      const float n1[3] = {m_sp2.px[i] * pinv, m_sp2.py[i] * pinv, m_sp2.pz[i] * pinv};
      float Cn[3];
      MPLEX_UNROLL
      for (int r = 0; r < 3; ++r)
        Cn[r] = C[r][0] * n1[0] + C[r][1] * n1[1] + C[r][2] * n1[2];
      const float nCn = n1[0] * Cn[0] + n1[1] * Cn[1] + n1[2] * Cn[2];
      auto pcp = [&](int r, int c) { return C[r][c] - n1[r] * Cn[c] - Cn[r] * n1[c] + n1[r] * n1[c] * nCn; };

      tce.m_cov_0_0[i] = pcp(0, 0);
      tce.m_cov_0_1[i] = pcp(0, 1);
      tce.m_cov_1_1[i] = pcp(1, 1);
      tce.m_cov_2_2[i] = pcp(2, 2);
      tce.m_cov_0_2[i] = pcp(0, 2);
      tce.m_cov_1_2[i] = pcp(1, 2);
    }
  }

  //----------------------------------------------------------------------------

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MkBins<N>::determine_bin_windows(const MkBinTrackCovExtract<N>& cov_ex) {
    // The wrap fix-up in a loop of its own, as MkFitCore, so that the window loop vectorizes.
    MPLEX_SIMD
    for (int ii = 0; ii < N; ++ii) {
      m_phi1[ii] = ::mkfitdev::Matriplex::internal::atan2_poly(m_sp1.y[ii], m_sp1.x[ii]);
      m_phi2[ii] = ::mkfitdev::Matriplex::internal::atan2_poly(m_sp2.y[ii], m_sp2.x[ii]);
      m_phi_min[ii] = std::min(m_phi1[ii], m_phi2[ii]);
      m_phi_max[ii] = std::max(m_phi1[ii], m_phi2[ii]);
      m_phi_delta[ii] = m_phi_max[ii] - m_phi_min[ii];
      m_phi_center[ii] = 0.5f * (m_phi_max[ii] + m_phi_min[ii]);
    }
    for (int ii = 0; ii < N; ++ii) {
      if (ii < m_n_proc) {
        if (m_phi_delta[ii] > Const::PI) {
          const float t = m_phi_max[ii];
          m_phi_max[ii] = m_phi_min[ii];
          m_phi_min[ii] = t;
          m_phi_delta[ii] = Const::TwoPI - m_phi_delta[ii];
          m_phi_center[ii] = squashPhiMinimal(m_phi_center[ii] + Const::PI);
        }
        m_phi_delta[ii] *= 0.5f;
      }
    }

    // Calculate dphi_track, dq_track differs for barrel/endcap
    for (int i = 0; i < N; ++i) {
      const float r2_c = m_isp.x[i] * m_isp.x[i] + m_isp.y[i] * m_isp.y[i];
      const float r2inv_c = 1.0f / r2_c;

      // sigma_phi = J sigma_xy J^T with J = (-y, x)/r^2, so |J| = 1/r. J is taken at
      // the layer crossing with the smaller radius, where sigma_phi is largest. See
      // doc/MkFinderV2p2-DesignNotes.md, "Search window".
      float jx = 0.0f, jy = 0.0f, r2inv_j = 0.0f;
      if (i < m_n_proc) {
        const float r2_1 = m_sp1.x[i] * m_sp1.x[i] + m_sp1.y[i] * m_sp1.y[i];
        const float r2_2 = m_sp2.x[i] * m_sp2.x[i] + m_sp2.y[i] * m_sp2.y[i];
        const bool first = r2_1 <= r2_2;
        jx = first ? m_sp1.x[i] : m_sp2.x[i];
        jy = first ? m_sp1.y[i] : m_sp2.y[i];
        r2inv_j = 1.0f / (first ? r2_1 : r2_2);
      }
      const float dphidx_c = -jy * r2inv_j;
      const float dphidy_c = jx * r2inv_j;
      m_dphi_track[i] = 3.0f * std::sqrt(std::abs(cov_ex.calc_err_xy(dphidx_c, dphidy_c, i)));

      if (m_is_barrel) {
        m_q_min[i] = std::min(m_sp1.z[i], m_sp2.z[i]);
        m_q_max[i] = std::max(m_sp1.z[i], m_sp2.z[i]);
        m_dq_track[i] = 3.0f * std::sqrt(std::abs(cov_ex.m_cov_2_2[i]));
      } else {
        // Matriplex::hypot
        const float h1 = std::sqrt(m_sp1.x[i] * m_sp1.x[i] + m_sp1.y[i] * m_sp1.y[i]);
        const float h2 = std::sqrt(m_sp2.x[i] * m_sp2.x[i] + m_sp2.y[i] * m_sp2.y[i]);
        m_q_min[i] = std::min(h1, h2);
        m_q_max[i] = std::max(h1, h2);
        m_dq_track[i] = 3.0f * std::sqrt(r2inv_c * std::abs(cov_ex.calc_err_xy(m_isp.x[i], m_isp.y[i], i)));
      }
    }
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MkBins<N>::find_bin_ranges(const ::mkfitdev::LayerOfHitsAccess& loh,
                                                                      MkBinLimits<N>& bl,
                                                                      float dphi_trk_fac,
                                                                      float dq_trk_fac) const {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    if (!(dphi_trk_fac > 0.0f))
      dphi_trk_fac = Window::dphi_trk_fac;
    if (!(dq_trk_fac > 0.0f))
      dq_trk_fac = Window::dq_trk_fac;
    for (int i = 0; i < N; ++i) {
      if (i < m_n_proc) {
        // Fetch what the cut can accept, using the layer's largest hit extent since
        // the hit is not known yet, then extend by whole bins on the bin index.
        // The margin is capped at Window::dphi_max_rad.
        // See doc/MkFinderV2p2-DesignNotes.md, "Search window".
        const float phi_hit_term =
            Window::phi_per_hit ? Window::dphi_hit_fac * loh.max_hit_phi_half_extent() : Window::dphi_flat_rad;
        const float cut_dphi = Window::capped_dphi(dphi_trk_fac * m_dphi_track[i] + phi_hit_term);
        auto pr = loh.phiRangeBins(m_phi_min[i] - cut_dphi, m_phi_max[i] + cut_dphi);
        bl.p1[i] = loh.phiMaskApply(pr.begin - Window::phi_extra_bins);
        bl.p2[i] = loh.phiMaskApply(pr.end + Window::phi_extra_bins);

        // Same for q. The q axis is bounded, so the extension clamps.
        const float cut_dq = dq_trk_fac * m_dq_track[i] + Window::dq_hit_fac * loh.max_hit_q_half_length();
        auto qr = loh.qRangeBins(m_q_min[i] - cut_dq, m_q_max[i] + cut_dq);
        const int nq = (int)loh.qNBins();
        int qb = (int)qr.begin - Window::q_extra_bins;
        int qe = (int)qr.end + Window::q_extra_bins;
        bl.q1[i] = (unsigned short)(qb < 0 ? 0 : qb);
        bl.q2[i] = (unsigned short)(qe > nq ? nq : qe);
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
