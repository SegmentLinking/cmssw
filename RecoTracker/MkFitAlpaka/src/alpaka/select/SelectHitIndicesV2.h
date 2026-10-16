#ifndef RecoTracker_MkFitAlpaka_src_alpaka_select_SelectHitIndicesV2_h
#define RecoTracker_MkFitAlpaka_src_alpaka_select_SelectHitIndicesV2_h

// Portable transliteration of MkFinder::selectHitIndicesV2 (CMSSW_20_1_0_pre2 RecoTracker/MkFitCore/src/
// MkFinder.cc:762-1150) for ONE candidate (MkFitCore processes NN slots; every quantity is per slot).
//   Bins::prop_to_limits / find_bin_ranges  -> BinsWindow<N>   (vectorized mini propagators; computeBins: one slot)
//   WSR from the layer limits (ES), in_gap from the dead-bin table of the central q bin
//   per-hit mini propagation (plane PA_Line / to_r / to_z), dq/dphi preselection
//   std::priority_queue<PQE> of 6 entries -> SelHeap: same libstdc++ push_heap / pop_heap element moves, so ties
//   come out in the MkFitCore order; drained as MkFitCore (best ddphi first).
// Dropped dead paths (doc/SIMPLIFICATIONS.txt): the phi_c = PI - phi_c wrap and dphi (debug-only values), the
// RNT_DUMP_MkF_SelHitIdcs instrumentation, dprintf.

#include <cmath>
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/LayerOfHitsAccess.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"
#include "RecoTracker/MkFitAlpaka/interface/math/vdtMath.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexCommon.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/MiniPropagators.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/SelectSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::select {

  namespace mp = mini_propagators;
  using bidx_t = ::mkfitdev::LayerOfHitsAccess::bin_index_t;

  constexpr int NEW_MAX_HIT = ::mkfitdev::kMaxSelHits;  // 4 - 6 give about the same # of tracks in quality-val
  constexpr float DDPHI_PRESEL_FAC = 2.0f;
  constexpr float DDQ_PRESEL_FAC = 1.2f;
  constexpr float PHI_BIN_EXTRA_FAC = 2.75f;
  constexpr float Q_BIN_EXTRA_FAC = 1.6f;

  //--------------------------------------------------------------------------------------------------------------
  // Fixed 6-entry max-heap with the element moves of libstdc++ std::priority_queue (vector storage, std::less-like
  // comparator a.score < b.score): push = push_back + __push_heap, pop = __pop_heap + pop_back.
  //--------------------------------------------------------------------------------------------------------------
  struct PQE {
    float score;
    unsigned int hit_index;
  };

  struct SelHeap {
    PQE a[NEW_MAX_HIT];
    int n = 0;

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static bool cmp(const PQE& x, const PQE& y) { return x.score < y.score; }

    // std::__push_heap(first, holeIndex, topIndex = 0, value)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void siftUp(int hole, int top, PQE v) {
      int parent = (hole - 1) / 2;
      while (hole > top && cmp(a[parent], v)) {
        a[hole] = a[parent];
        hole = parent;
        parent = (hole - 1) / 2;
      }
      a[hole] = v;
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void push(PQE v) {
      a[n] = v;
      ++n;
      siftUp(n - 1, 0, v);
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const PQE& top() const { return a[0]; }

    // std::pop_heap(first, last) + pop_back
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void pop() {
      if (n > 1) {
        // __pop_heap(first, last - 1, last - 1): value = *(last-1); *(last-1) = *first; __adjust_heap(first, 0, len, value)
        const int len = n - 1;
        PQE value = a[len];
        a[len] = a[0];
        // __adjust_heap
        int hole = 0;
        const int top = 0;
        int second = hole;
        while (second < (len - 1) / 2) {
          second = 2 * (second + 1);
          if (cmp(a[second], a[second - 1]))
            second--;
          a[hole] = a[second];
          hole = second;
        }
        if ((len & 1) == 0 && second == (len - 2) / 2) {
          second = 2 * (second + 1);
          a[hole] = a[second - 1];
          hole = second - 1;
        }
        siftUp(hole, top, value);
      }
      --n;
    }
  };

  //--------------------------------------------------------------------------------------------------------------
  // Bins (MkFitCore struct Bins in selectHitIndicesV2), one slot
  //--------------------------------------------------------------------------------------------------------------
  struct Bins {
    bidx_t q0, q1, q2, p1, p2;
    float dphi_track, dq_track;  // 3 sigma track errors at initial state
    float q_c;
  };

  // MkFitCore std::min / std::max as used by Matriplex::min_max
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void min_max(float a, float b, float& mn, float& mx) {
    mn = (b < a) ? b : a;
    mx = (a < b) ? b : a;
  }

  // Bins::prop_to_limits + Bins::find_bin_ranges over N slots, every operation per slot as in MkFitCore's Matriplex
  // expressions: the CPU backends compute the window of a batch of kNN candidates together (MPLEX_SIMD lane loops),
  // one candidate is N = 1 (computeBins).
  template <int N>
  struct BinsWindow {
    float pmin[N], pmax[N], qmin[N], qmax[N];
    float q_c[N], dphi_track[N], dq_track[N];

    // par / err: state at the layer (m_Par / m_Err[iP]) in Matriplex layout (element k of slot n at [k * N + n]);
    // lim1 / lim2: rin / rout (barrel) or zmin / zmax (endcap) of the slot's layer.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void compute(
        const float* par, const float* err, const int* chg, bool isBarrel, const float* lim1, const float* lim2) {
      const mp::InitialStatePlex<N> isp(par, chg);
      // prop_to_limits
      mp::StatePlex<N> sp1, sp2;
      if (isBarrel) {
        isp.propagate_to_r(lim1, sp1);
        isp.propagate_to_r(lim2, sp2);
      } else {
        isp.propagate_to_z(lim1, sp1);
        isp.propagate_to_z(lim2, sp2);
      }

      // find_bin_ranges (float part)
      float xp1[N], xp2[N];
      MPLEX_SIMD
      for (int n = 0; n < N; ++n)
        xp1[n] = ::mkfitdev::vdt::fast_atan2f(sp1.y[n], sp1.x[n]);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n)
        xp2[n] = ::mkfitdev::vdt::fast_atan2f(sp2.y[n], sp2.x[n]);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        float pl, ph;
        min_max(xp1[n], xp2[n], pl, ph);
        const float dp = ph - pl;
        // MkFitCore also sets dp = TwoPI - dp, dphi = 0.5 dp and phi_c = PI - phi_c: debug-only values (SIMPLIFICATIONS)
        pmin[n] = dp > ::mkfitdev::Const::PI ? ph : pl;
        pmax[n] = dp > ::mkfitdev::Const::PI ? pl : ph;
      }

      // err: packed lower triangle, (0,0)=0, (1,0)=1, (1,1)=2, (2,2)=5
      // Explicit fma: MkFitCore (Matriplex expressions, -Ofast, x86-64-v3) contracts a*b + c*d into fma(c, d, a*b). These
      // two forms reproduce q_c, dq_track and dphi_track bitwise on 106,484 dumped candidates (plain: 54-93% only).
      const auto calc_err_xy = [&](int n, float x, float y) {
        return std::fma(2.0f * x * y, err[N + n], std::fma(y * y, err[2 * N + n], x * x * err[n]));
      };
      float r2_c[N], r2inv_c[N];
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        const float x = isp.x[n], y = isp.y[n];
        r2_c[n] = std::fma(y, y, x * x);
        r2inv_c[n] = 1.0f / r2_c[n];
        const float dphidx_c = -y * r2inv_c[n];
        const float dphidy_c = x * r2inv_c[n];
        dphi_track[n] = 3.0f * std::sqrt(std::abs(calc_err_xy(n, dphidx_c, dphidy_c)));
      }

      if (isBarrel) {
        MPLEX_SIMD
        for (int n = 0; n < N; ++n) {
          min_max(sp1.z[n], sp2.z[n], qmin[n], qmax[n]);
          q_c[n] = isp.z[n];
          dq_track[n] = 3.0f * std::sqrt(std::abs(err[5 * N + n]));
        }
      } else {
        MPLEX_SIMD
        for (int n = 0; n < N; ++n) {
          min_max(::mkfitdev::hipo(sp1.x[n], sp1.y[n]), ::mkfitdev::hipo(sp2.x[n], sp2.y[n]), qmin[n], qmax[n]);
          q_c[n] = std::sqrt(r2_c[n]);
          dq_track[n] = 3.0f * std::sqrt(r2inv_c[n] * std::abs(calc_err_xy(n, isp.x[n], isp.y[n])));
        }
      }
    }

    // find_bin_ranges (bin part) of slot n in its layer
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void bins(int n,
                                                  float q_bin,
                                                  const ::mkfitdev::LayerOfHitsAccess& loh,
                                                  Bins& B) const {
      B.dphi_track = dphi_track[n];
      B.dq_track = dq_track[n];
      B.q_c = q_c[n];

      B.p1 = loh.phiBinChecked(pmin[n] - B.dphi_track - PHI_BIN_EXTRA_FAC * 0.0123f);
      B.p2 = loh.phiBinChecked(pmax[n] + B.dphi_track + PHI_BIN_EXTRA_FAC * 0.0123f);

      B.q0 = loh.qBinChecked(B.q_c);
      B.q1 = loh.qBinChecked(qmin[n] - B.dq_track - Q_BIN_EXTRA_FAC * 0.5f * q_bin);
      B.q2 = loh.qBinChecked(qmax[n] + B.dq_track + Q_BIN_EXTRA_FAC * 0.5f * q_bin) + 1;
    }
  };

  // Bins of one candidate. par/err: state at the layer (m_Par/m_Err[iP]).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void computeBins(const float* par,
                                                       const float* err,
                                                       int chg,
                                                       bool isBarrel,
                                                       float rin,
                                                       float rout,
                                                       float zmin,
                                                       float zmax,
                                                       float q_bin,
                                                       const ::mkfitdev::LayerOfHitsAccess& loh,
                                                       Bins& B) {
    const float lim1 = isBarrel ? rin : zmin;
    const float lim2 = isBarrel ? rout : zmax;
    BinsWindow<1> w;
    w.compute(par, err, &chg, isBarrel, &lim1, &lim2);
    w.bins(0, q_bin, loh, B);
  }

  struct SelResult {
    int32_t hits[NEW_MAX_HIT];
    int nHits;
    int wsr;
    bool inGap;
  };

  //--------------------------------------------------------------------------------------------------------------
  // selectHitIndicesV2, split in its per-candidate setup (selectPrepare) and its per-hit body (selectHitScore) so the
  // GPU K2 can scan one candidate's hits with several lanes; selectHitIndicesV2 = the serial composition.
  //--------------------------------------------------------------------------------------------------------------

  // WSR of a candidate with Bins B. Returns true when the hit scan runs (not failed, not WSR_Outside).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool selectWsr(
      int failFlag, int layer, bool isBarrel, const ::mkfitdev::ESView& es, const Bins& B, SelResult& out) {
    out.nHits = 0;
    out.inGap = false;
    if (failFlag) {
      out.wsr = ::mkfitdev::WSR_Failed;
      return false;
    }
    {
      // 0.5f * (q2 - q1): MkFitCore passes the bin-index span (MPlexQUH difference) as the q tolerance
      const float dq = 0.5f * (B.q2 - B.q1);
      const ::mkfitdev::WSRResult w =
          isBarrel ? es.isWithinZSensitiveRegion(layer, B.q_c, dq) : es.isWithinRSensitiveRegion(layer, B.q_c, dq);
      out.wsr = w.wsr;
      out.inGap = w.in_gap;
    }
    return out.wsr != ::mkfitdev::WSR_Outside;
  }

  // Bins + WSR of one candidate. Returns true when the hit scan runs (not failed, not WSR_Outside).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool selectPrepare(const float* par,
                                                         const float* err,
                                                         int chg,
                                                         int failFlag,
                                                         int layer,
                                                         const ::mkfitdev::ESView& es,
                                                         const ::mkfitdev::LayerOfHitsAccess& L,
                                                         Bins& B,
                                                         SelResult& out) {
    const auto li = es.layers[layer];
    const bool isBarrel = es.isBarrel(layer);

    computeBins(par, err, chg, isBarrel, li.rin(), li.rout(), li.zmin(), li.zmax(), li.q_bin(), L, B);
    return selectWsr(failFlag, layer, isBarrel, es, B, out);
  }

  // The per-hit body of the MkFitCore bin loop for binned hit hi: mini propagation, dq/dphi preselection. Returns true
  // when the hit enters the queue, with its score (ddphi) and original index.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool selectHitScore(const mp::InitialState& mp_is,
                                                          bool isBarrel,
                                                          int moduleBegin,
                                                          float dqTrack,
                                                          float dphiCut,
                                                          const ::mkfitdev::ESView& es,
                                                          const ::mkfitdev::LayerOfHitsAccess& L,
                                                          ::mkfitdev::HitSoA::ConstView hits,
                                                          uint32_t hi,
                                                          float& score,
                                                          unsigned int& hiOrig) {
    const unsigned int hi_orig = L.getOriginalHitIndex(hi);
    // LST step: no iteration hit mask (m_iteration_hit_mask == nullptr; checked by the MkFitCore dump)
    mp::State mp_s;

    float new_q, new_phi, new_ddphi, new_ddq;
    bool prop_fail;

    if (isBarrel) {
      const uint32_t mid = ::mkfitdev::hitpack::detIDinLayer(hits[L.hitRow(hi_orig)].packed());
      const auto mi = es.modules[moduleBegin + mid];

      // This could work well instead of prop-to-r, too. Limit to 0.05 rad, 2.85 deg.
      if (std::abs(mi.zdir_z()) > 0.05f) {
        const mp::ModulePlane mpl{{mi.pos_x(), mi.pos_y(), mi.pos_z()}, {mi.zdir_x(), mi.zdir_y(), mi.zdir_z()}};
        prop_fail = mp_is.propagate_to_plane(mpl, mp_s);
        new_q = mp_s.z;
        new_ddq = std::abs(new_q - L.hit_q(hi));
      } else {
        prop_fail = mp_is.propagate_to_r(L.hit_qbar(hi), mp_s);
        new_q = mp_s.z;
        new_ddq = std::abs(new_q - L.hit_q(hi));
      }
    } else {
      prop_fail = mp_is.propagate_to_z(L.hit_qbar(hi), mp_s);
      new_q = ::mkfitdev::hipo(mp_s.x, mp_s.y);
      new_ddq = std::abs(new_q - L.hit_q(hi));
    }

    new_phi = ::mkfitdev::vdt::fast_atan2f(mp_s.y, mp_s.x);
    new_ddphi = ::mkfitdev::cdist(std::abs(new_phi - L.hit_phi(hi)));
    const bool dqdphi_presel = new_ddq < dqTrack + DDQ_PRESEL_FAC * L.hit_q_half_length(hi) && new_ddphi < dphiCut;

    score = new_ddphi;
    hiOrig = hi_orig;
    return !(prop_fail || !dqdphi_presel);
  }

  // The MkFitCore bin loop with its std::priority_queue (SelHeap) and the best-first drain, for one candidate:
  // the serial selection (selectHitIndicesV2) and the GPU scan's exact-tie fallback (KernelSelectScan).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void selectScanHeap(const mp::InitialState& mp_is,
                                                          bool isBarrel,
                                                          int moduleBegin,
                                                          float dqTrack,
                                                          float dphiCut,
                                                          const ::mkfitdev::ESView& es,
                                                          const ::mkfitdev::LayerOfHitsAccess& L,
                                                          ::mkfitdev::HitSoA::ConstView hits,
                                                          bidx_t qb,
                                                          bidx_t qb1,
                                                          bidx_t qb2,
                                                          bidx_t pb1,
                                                          bidx_t pb2,
                                                          bool& inGap,
                                                          int32_t* outHits,
                                                          int& nHits) {
    SelHeap pq;

    for (bidx_t qi = qb1; qi != qb2; ++qi) {
      for (bidx_t pi = pb1; pi != pb2; pi = L.phiMaskApply(pi + 1)) {
        // Limit to central Q-bin
        if (qi == qb && L.isBinDead(pi, qi) == true)
          inGap = true;

        const uint32_t content = L.binContent(pi, qi);
        const uint32_t hbeg = ::mkfitdev::binFirst(content);
        const uint32_t hend = hbeg + ::mkfitdev::binCount(content);
        for (uint32_t hi = hbeg; hi < hend; ++hi) {
          float new_ddphi;
          unsigned int hi_orig;
          if (!selectHitScore(mp_is, isBarrel, moduleBegin, dqTrack, dphiCut, es, L, hits, hi, new_ddphi, hi_orig))
            continue;
          if (pq.n < NEW_MAX_HIT) {
            pq.push({new_ddphi, hi_orig});
          } else if (new_ddphi < pq.top().score) {
            pq.pop();
            pq.push({new_ddphi, hi_orig});
          }
        }  // hi
      }  // pi
    }  // qi

    // Reverse hits so best dphis/scores come first in the hit-index list.
    nHits = pq.n;
    int sz = pq.n;
    while (sz) {
      --sz;
      outHits[sz] = pq.top().hit_index;
      pq.pop();
    }
  }

  // The hit scan of selectHitIndicesV2 for one candidate whose Bins and WSR are known (selectPrepare / selectWsr
  // returned true).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void selectScan(const float* par,
                                                      int chg,
                                                      int layer,
                                                      const ::mkfitdev::ESView& es,
                                                      const ::mkfitdev::LayerOfHitsAccess& L,
                                                      ::mkfitdev::HitSoA::ConstView hits,
                                                      const Bins& B,
                                                      SelResult& out) {
    const mp::InitialState mp_is(par, chg);
    selectScanHeap(mp_is,
                   es.isBarrel(layer),
                   es.layers[layer].module_begin(),
                   B.dq_track,
                   B.dphi_track + DDPHI_PRESEL_FAC * 0.0123f,
                   es,
                   L,
                   hits,
                   B.q0,
                   B.q1,
                   B.q2,
                   B.p1,
                   B.p2,
                   out.inGap,
                   out.hits,
                   out.nHits);
  }

  //--------------------------------------------------------------------------------------------------------------
  // selectHitIndicesV2 for one candidate. par/err/chg: propagated state (iP), failFlag: inter-layer prop fail flag.
  //--------------------------------------------------------------------------------------------------------------
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void selectHitIndicesV2(const float* par,
                                                              const float* err,
                                                              int chg,
                                                              int failFlag,
                                                              int layer,
                                                              const ::mkfitdev::ESView& es,
                                                              const ::mkfitdev::LayerOfHitsAccess& L,
                                                              ::mkfitdev::HitSoA::ConstView hits,
                                                              Bins& B,
                                                              SelResult& out) {
    if (!selectPrepare(par, err, chg, failFlag, layer, es, L, B, out))
      return;
    selectScan(par, chg, layer, es, L, hits, B, out);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::select

#endif
