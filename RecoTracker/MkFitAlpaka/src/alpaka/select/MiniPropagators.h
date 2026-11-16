#ifndef RecoTracker_MkFitAlpaka_src_alpaka_select_MiniPropagators_h
#define RecoTracker_MkFitAlpaka_src_alpaka_select_MiniPropagators_h

// Portable transliteration of MkFitCore mini propagators (RecoTracker/MkFitCore/src/MiniPropagators.{h,cc}), as used by
// MkFinder::selectHitIndicesV2 of the clone engine and by MkFinderV2p2.
//   - State / InitialState: the scalar flavour (per-hit propagation). State(par) uses std::cos/sin/tan as MkFitCore.
//   - StatePlex / InitialStatePlex: the MkFitCore "Plex" flavour (bin limits, layer crossings) over N slots. MkFitCore
//     evaluates it element-wise on MPlexQF with vdt fast_sincos / fast_tan; every operation is per element, so any
//     N (CPU batches: kNN, one candidate: 1) is the same arithmetic per slot.
//   - Hermite3D / Hermite3DOnPlane: the cubic through two layer crossings and its solve onto a module plane.
// Only the algorithms the two finders reach are ported: propagate_to_r(PA_Exact), propagate_to_z(PA_Exact),
// propagate_to_plane(PA_Line); the Plex propagators always update the momentum. PA_Line/PA_Quadratic of to_r/to_z
// fall through to PA_Exact; the plane PA_Quadratic/PA_Exact throw in MkFitCore and are never called. Not ported: the
// Plex propagate_to_plane and Hermite3D's one-point mode and path_length (no caller outside diagnostics), and the
// projecting State / InitialState constructors (the v2p2 state on a hit is the POD V2p2InitialState).

#include <cmath>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"
#include "RecoTracker/MkFitAlpaka/interface/math/vdtMath.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexCommon.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/Matrix.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::mini_propagators {

  namespace mconst = ::mkfitdev::Const;
  namespace mconfig = ::mkfitdev::Config;

  enum PropAlgo_e { PA_Line, PA_Quadratic, PA_Exact };

  using ::mkfitdev::mini_propagators::kReachMargin;
  using ::mkfitdev::mini_propagators::kReachTolerance;

  struct State {
    float x, y, z;
    float px, py, pz;
    float dalpha;
    int fail_flag;
  };

  // State::State(const MPlexLV& par, int ti): par = {x, y, z, 1/pT, phi, theta}
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE State makeState(const float* par) {
    State s;
    s.x = par[0];
    s.y = par[1];
    s.z = par[2];
    const float pt = 1.0f / par[3];
    s.px = pt * std::cos(par[4]);
    s.py = pt * std::sin(par[4]);
    s.pz = pt / std::tan(par[5]);
    s.dalpha = 0.f;  // MkFitCore leaves dalpha and fail_flag indeterminate; dalpha only accumulates, never read here
    s.fail_flag = 0;
    return s;
  }

  struct InitialState : public State {
    float inv_pt, inv_k;
    float theta;

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE InitialState(const float* par, int charge) : State(makeState(par)) {
      inv_pt = par[3];
      theta = par[5];
      inv_k = ((charge < 0) ? 0.01f : -0.01f) * mconst::sol * mconfig::Bfield;
    }

    // InitialState::propagate_to_r (PA_Exact): scalar mirror of InitialStatePlex::propagate_to_r. Returns the fail
    // flag: 2 = R not on the trajectory (c clamped to the closest reachable radius), 1 = landed further than
    // kReachTolerance from it.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool propagate_to_r(float R, State& c) const {
      float R_eff = R;
      bool unreachable = false;

      const float k = 1.0f / inv_k;

      c = *this;

      const float cx = x - k * py;
      const float cy = y + k * px;
      const float d_c = ::mkfitdev::hipo(cx, cy);
      const float R_c = std::abs(k) * ::mkfitdev::hipo(px, py);
      const float dc2_m_Rc2 = x * x + y * y - 2.0f * k * (x * py - y * px);

      float r_lo = std::abs(d_c - R_c) + kReachMargin;
      float r_hi = d_c + R_c - kReachMargin;
      if (r_lo > r_hi)
        r_lo = r_hi = 0.5f * (std::abs(d_c - R_c) + d_c + R_c);
      if (R < r_lo) {
        R_eff = r_lo;
        unreachable = true;
      } else if (R > r_hi) {
        R_eff = r_hi;
        unreachable = true;
      }

      const float R2 = R_eff * R_eff;
      const float a = (R2 + dc2_m_Rc2) / (2.0f * d_c);
      const float b2 = R2 - a * a;
      const float b = b2 > 0.0f ? std::sqrt(b2) : 0.0f;

      const float inv_dc = 1.0f / d_c;
      const float hx = cx * inv_dc, hy = cy * inv_dc;  // Chat; Chat_perp = (-hy, hx)

      const float u0x = x - cx, u0y = y - cy;
      const float inv_Rc2 = 1.0f / (R_c * R_c);

      const float p1x = a * hx - b * hy, p1y = a * hy + b * hx;
      const float p2x = a * hx + b * hy, p2y = a * hy - b * hx;
      const float u1x = p1x - cx, u1y = p1y - cy;
      const float u2x = p2x - cx, u2y = p2y - cy;
      const float cos1 = (u0x * u1x + u0y * u1y) * inv_Rc2;
      const float sin1 = (u0x * u1y - u0y * u1x) * inv_Rc2;
      const float cos2 = (u0x * u2x + u0y * u2y) * inv_Rc2;
      const float sin2 = (u0x * u2y - u0y * u2x) * inv_Rc2;
      const float alpha1 = ::mkfitdev::vdt::fast_atan2f(sin1, cos1);
      const float alpha2 = ::mkfitdev::vdt::fast_atan2f(sin2, cos2);

      const bool first = std::abs(alpha1) <= std::abs(alpha2);
      const float alpha = first ? alpha1 : alpha2;
      const float sina = first ? sin1 : sin2;
      const float cosa = first ? cos1 : cos2;
      c.x = first ? p1x : p2x;
      c.y = first ? p1y : p2y;

      c.z = z + k * pz * alpha;
      c.dalpha = dalpha + alpha;

      const float o_px = px;
      c.px = px * cosa - py * sina;
      c.py = py * cosa + o_px * sina;
      c.pz = pz;

      if (unreachable)
        c.fail_flag = 2;
      else
        c.fail_flag = std::abs(R_eff - ::mkfitdev::hipo(c.x, c.y)) > kReachTolerance ? 1 : 0;
      return c.fail_flag;
    }

    // InitialState::propagate_to_z (PA_Exact, update_momentum = true); never fails
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool propagate_to_z(float Z, State& c) const {
      const float k = 1.0f / inv_k;

      const float dz = Z - z;
      const float alpha = dz * inv_k / pz;

      float sina, cosa;
      ::mkfitdev::vdt::fast_sincosf(alpha, sina, cosa);

      c.dalpha = dalpha + alpha;
      c.x = x + k * (px * sina - py * (1.0f - cosa));
      c.y = y + k * (py * sina + px * (1.0f - cosa));
      c.z = Z;

      c.px = px * cosa - py * sina;
      c.py = py * cosa + px * sina;
      c.pz = pz;

      c.fail_flag = 0;
      return c.fail_flag;
    }

    // InitialState::propagate_to_plane (PA_Line): straight step along the momentum onto the plane (pos, zdir), the
    // momentum is not changed; MkFitCore returns false (no failure) and does not touch c.fail_flag
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool propagate_to_plane(const float (&pos)[3],
                                                                const float (&zdir)[3],
                                                                State& c) const {
      // t * p_vec intersects the plane:
      float t = (pos[0] * zdir[0] + pos[1] * zdir[1] + pos[2] * zdir[2] - x * zdir[0] - y * zdir[1] - z * zdir[2]) /
                (px * zdir[0] + py * zdir[1] + pz * zdir[2]);

      c = *this;
      c.x += t * c.px;
      c.y += t * c.py;
      c.z += t * c.pz;

      // Signed helix turning angle over the same transverse path, t pT / (k pT).
      c.dalpha += t * inv_k;
      return false;
    }
  };

  //-----------------------------------------------------------
  // The MkFitCore vectorized flavour (StatePlex / InitialStatePlex) over N slots: every Matriplex expression of
  // MkFitCore is one slot loop (MPLEX_SIMD on the CPU backends); one candidate is N = 1.
  //-----------------------------------------------------------

  // Matriplex::hypot (a*a + b*b, then sqrt) as MkFitCore's -Ofast x86-64-v3 build contracts it (see InitialStatePlex).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float hypotV(float a, float b) { return std::sqrt(std::fma(a, a, b * b)); }

  template <int N>
  struct StatePlex {
    float x[N], y[N], z[N];
    float px[N], py[N], pz[N];
    float dalpha[N];
    int fail_flag[N];

    // StatePlex::copyIn(dst_slot, const StatePlex& src, src_slot)
    template <int M>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void copyIn(int dst_slot, const StatePlex<M>& src, int src_slot) {
      x[dst_slot] = src.x[src_slot];
      y[dst_slot] = src.y[src_slot];
      z[dst_slot] = src.z[src_slot];
      px[dst_slot] = src.px[src_slot];
      py[dst_slot] = src.py[src_slot];
      pz[dst_slot] = src.pz[src_slot];
      dalpha[dst_slot] = src.dalpha[src_slot];
      fail_flag[dst_slot] = src.fail_flag[src_slot];
    }
  };

  // The explicit fma forms below are the contractions of MkFitCore's -Ofast x86-64-v3 build of these Matriplex
  // expressions (an x86-64-v2 MkFitCore does not fuse); its vector divisions are reciprocal approximations, so the
  // results agree at the rounding level, not bitwise.
  template <int N>
  struct InitialStatePlex : public StatePlex<N> {
    float inv_pt[N], inv_k[N];
    float theta[N];

    // Trivial, so that MkBins can live in block shared memory; filled by copyIn_partial_track_state and
    // init_momentum_vec_and_k as in MkFinderV2p2::prop_to_layer_edges.
    InitialStatePlex() = default;

    // using StatePlex::operator= (MkBins::prop_to_limits_in_order: m_isp = m_sp1 keeps inv_pt, inv_k, theta)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE InitialStatePlex& operator=(const StatePlex<N>& sp) {
      StatePlex<N>::operator=(sp);
      return *this;
    }

    // InitialStatePlex::copyIn(dst_slot, const InitialStatePlex& src, src_slot)
    template <int M>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void copyIn(int dst_slot, const InitialStatePlex<M>& src, int src_slot) {
      StatePlex<N>::copyIn(dst_slot, src, src_slot);
      inv_pt[dst_slot] = src.inv_pt[src_slot];
      inv_k[dst_slot] = src.inv_k[src_slot];
      theta[dst_slot] = src.theta[src_slot];
    }

    // copyIn_partial_track_state: x, y, z, inv_pt and theta of par = {x, y, z, 1/pT, phi, theta}; to be followed by
    // init_momentum_vec_and_k. Also zeroes dalpha and fail_flag of the slot: MkFitCore's StatePlex member initializers
    // in its fresh LayerBatch, which this trivial type cannot have.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void copyIn_partial_track_state(int dst_slot, const float* par) {
      this->x[dst_slot] = par[0];
      this->y[dst_slot] = par[1];
      this->z[dst_slot] = par[2];
      inv_pt[dst_slot] = par[3];
      theta[dst_slot] = par[5];
      this->dalpha[dst_slot] = 0.0f;
      this->fail_flag[dst_slot] = 0;
    }

    // init_momentum_vec_and_k(phi, chg, float bf) (MiniPropagators.cc)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void init_momentum_vec_and_k(const float* phi, const int* chg, float bf);

    // init_momentum_vec_and_k(phi, chg, const MPF& bf): the uniform version with bf = 1, then inv_k *= bf[i]
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void init_momentum_vec_and_k(const float* phi, const int* chg, const float* bf);

    // StatePlex::StatePlex(const MPlexLV& par): fast_sincos(phi, py, px); px *= pt; py *= pt; pz = pt / fast_tan(theta).
    // par: Matriplex layout (element k of slot n at [k * N + n]).
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE InitialStatePlex(const float* par, const int* charge) {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        this->x[n] = par[n];
        this->y[n] = par[N + n];
        this->z[n] = par[2 * N + n];
        const float pt = 1.0f / par[3 * N + n];
        ::mkfitdev::vdt::fast_sincosf(par[4 * N + n], this->py[n], this->px[n]);
        this->px[n] *= pt;
        this->py[n] *= pt;
        this->pz[n] = pt / ::mkfitdev::vdt::fast_tanf(par[5 * N + n]);
        this->dalpha[n] = 0.f;
        this->fail_flag[n] = 0;
        inv_pt[n] = par[3 * N + n];
        theta[n] = par[5 * N + n];
        inv_k[n] = ((charge[n] < 0) ? 0.01f : -0.01f) * mconst::sol * mconfig::Bfield;
      }
    }

    // InitialStatePlex::propagate_to_r (PA_Exact) to R[n]: the closed-form crossing of the trajectory circle with the
    // cylinder; fail_flag 2 = R not on the trajectory (c clamped to the closest reachable radius), 1 = landed further
    // than kReachTolerance from it. Every slot is computed (MkFitCore: the first N_proc).
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void propagate_to_r(const float* R, StatePlex<N>& c) const {
      // The fail flags in a loop of their own, as MkFitCore, so that the propagation vectorizes.
      float R_effs[N];
      int unreachables[N];
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        float R_eff = R[n];
        int unreachable = 0;

        const float k = 1.0f / inv_k[n];

        const float x = this->x[n], y = this->y[n], z = this->z[n];
        const float px = this->px[n], py = this->py[n], pz = this->pz[n];

        // C = (x - k py, y + k px) is the centre of the trajectory circle, R_c its radius; d_c^2 - R_c^2 as
        // r0^2 - 2 k (x py - y px) keeps the float precision of stiff tracks.
        const float kpy = k * py;
        const float cx = x - kpy;
        const float cy = std::fma(k, px, y);
        const float d_c = hypotV(cx, cy);
        const float R_c = std::abs(k) * hypotV(px, py);
        const float dc2_m_Rc2 = std::fma(-(k * 2.0f), std::fma(-y, px, x * py), std::fma(y, y, x * x));

        // Reachability, see kReachMargin: an unreachable R is clamped to the band edge.
        float r_lo = std::abs(d_c - R_c) + kReachMargin;
        float r_hi = d_c + R_c - kReachMargin;
        if (r_lo > r_hi) {
          // (d_c + R_c) as r_hi: MkFitCore's build reuses that sum here
          r_lo = r_hi = 0.5f * (std::abs(d_c - R_c) + (d_c + R_c));
        }
        if (R[n] < r_lo) {
          R_eff = r_lo;
          unreachable = 1;
        } else if (R[n] > r_hi) {
          R_eff = r_hi;
          unreachable = 1;
        }

        const float a = std::fma(R_eff, R_eff, dc2_m_Rc2) / (2.0f * d_c);
        const float b2 = std::fma(R_eff, R_eff, -(a * a));
        const float b = std::sqrt((b2 < 0.0f) ? 0.0f : b2);  // Matriplex::max(R2 - a * a, 0)

        const float inv_dc = 1.0f / d_c;
        const float hx = cx * inv_dc, hy = cy * inv_dc;  // Chat; Chat_perp = (-hy, hx)

        // u0x = x - cx, which the MkFitCore build reassociates to k py
        const float u0x = kpy, u0y = y - cy;
        const float inv_Rc2 = 1.0f / (R_c * R_c);

        const float ahx = a * hx, ahy = a * hy;
        const float p1x = std::fma(-b, hy, ahx), p1y = std::fma(b, hx, ahy);
        const float p2x = std::fma(b, hy, ahx), p2y = std::fma(-b, hx, ahy);
        const float u1x = p1x - cx, u1y = p1y - cy;
        const float u2x = p2x - cx, u2y = p2y - cy;
        const float cos1 = std::fma(u0y, u1y, u0x * u1x) * inv_Rc2;
        const float sin1 = std::fma(-u0y, u1x, u0x * u1y) * inv_Rc2;
        const float cos2 = std::fma(u0y, u2y, u0x * u2x) * inv_Rc2;
        const float sin2 = std::fma(-u0y, u2x, u0x * u2y) * inv_Rc2;
        const float alpha1 = ::mkfitdev::vdt::fast_atan2f(sin1, cos1);
        const float alpha2 = ::mkfitdev::vdt::fast_atan2f(sin2, cos2);

        // Of the two crossings take the nearer one, the smaller turn in either direction.
        const bool first = std::abs(alpha1) <= std::abs(alpha2);
        const float alpha = first ? alpha1 : alpha2;
        const float sina = first ? sin1 : sin2;
        const float cosa = first ? cos1 : cos2;
        c.x[n] = first ? p1x : p2x;
        c.y[n] = first ? p1y : p2y;

        c.z[n] = std::fma(alpha * k, pz, z);
        c.dalpha[n] = this->dalpha[n] + alpha;

        // The momentum rotates by exactly the same alpha as the radius vector.
        c.px[n] = std::fma(-sina, py, px * cosa);
        c.py[n] = std::fma(sina, px, py * cosa);
        c.pz[n] = pz;
        R_effs[n] = R_eff;
        unreachables[n] = unreachable;
      }
      for (int n = 0; n < N; ++n) {
        if (unreachables[n])
          c.fail_flag[n] = 2;
        else
          c.fail_flag[n] = (std::abs(R_effs[n] - hypotV(c.x[n], c.y[n])) > kReachTolerance) ? 1 : 0;
      }
    }

    // InitialStatePlex::propagate_to_z (PA_Exact, update_momentum = true) to Z[n]; never fails
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void propagate_to_z(const float* Z, StatePlex<N>& c) const {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        const float k = 1.0f / inv_k[n];

        const float dz = Z[n] - this->z[n];
        const float alpha = dz * inv_k[n] / this->pz[n];

        float sina, cosa;
        ::mkfitdev::vdt::fast_sincosf(alpha, sina, cosa);

        c.dalpha[n] = this->dalpha[n] + alpha;
        c.x[n] = std::fma(k, std::fma(-this->py[n], 1.0f - cosa, this->px[n] * sina), this->x[n]);
        c.y[n] = std::fma(k, std::fma(this->px[n], 1.0f - cosa, this->py[n] * sina), this->y[n]);
        c.z[n] = Z[n];

        c.px[n] = std::fma(-this->py[n], sina, this->px[n] * cosa);
        c.py[n] = std::fma(this->px[n], sina, this->py[n] * cosa);
        c.pz[n] = this->pz[n];

        c.fail_flag[n] = 0;
      }
    }
  };

  //-----------------------------------------------------------
  // Hermite interpolation (MiniPropagators.h / .cc): the cubic model of the helix in t, linear in the turning angle.
  // Coefficient k of the cubic in x is m_Hx[k][slot], as MPlex4V aij(k, 0).
  //-----------------------------------------------------------

  constexpr int kHermiteCoeffs = 4;  // coefficients of a cubic (MPlex4V)

  template <int N>
  struct Hermite3D {
    float m_Hx[kHermiteCoeffs][N], m_Hy[kHermiteCoeffs][N], m_Hz[kHermiteCoeffs][N];
    float m_Hderfac[N];  // 1 / (k * dalpha)

    Hermite3D() = default;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit Hermite3D(float val) {
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < kHermiteCoeffs; ++k)
          m_Hx[k][n] = m_Hy[k][n] = m_Hz[k][n] = val;
        m_Hderfac[n] = val;
      }
    }

    // Hermite3D::copyIn(dst_slot, src, src_slot)
    template <int M>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void copyIn(int dst_slot, const Hermite3D<M>& src, int src_slot) {
      for (int k = 0; k < kHermiteCoeffs; ++k) {
        m_Hx[k][dst_slot] = src.m_Hx[k][src_slot];
        m_Hy[k][dst_slot] = src.m_Hy[k][src_slot];
        m_Hz[k][dst_slot] = src.m_Hz[k][src_slot];
      }
      m_Hderfac[dst_slot] = src.m_Hderfac[src_slot];
    }

    // Hermite3D::hermite_1d for slot n
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static void hermite_1d(
        float x1, float p1, float x2, float p2, float derfac, float (&h)[kHermiteCoeffs][N], int n);

    // two-point mode: t = 0 at sp1, t = 1 at sp2 (calculate_coeffs(sp1, sp2, inv_k))
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void calculate_coeffs(const StatePlex<N>& sp1,
                                                              const StatePlex<N>& sp2,
                                                              const float* inv_k);

    // evaluate(t, x, y, z)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void evaluate(const float* t, float* x, float* y, float* z) const;

    // evaluate(t, x, y, z, dx, dy, dz): position and momentum
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void evaluate(
        const float* t, float* x, float* y, float* z, float* dx, float* dy, float* dz) const;

    // evaluate(t, StatePlex& sp): x, y, z, px, py, pz of sp (dalpha and fail_flag untouched)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void evaluate(const float* t, StatePlex<N>& sp) const;
  };

  template <int N>
  struct Hermite3DOnPlane {
    float m_D[kHermiteCoeffs][N];  // cubic of the distance to the plane
    float m_T[N];                  // t at the solution

    // init_coeffs(h, pos, zdir): the distance cubic and the linear first guess T = d0 / (d0 - d1)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void init_coeffs(const Hermite3D<N>& h,
                                                         const ::mkfitdev::MPlexHV<N>& pos,
                                                         const ::mkfitdev::MPlexHV<N>& zdir);

    // evaluate(t, d)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void evaluate(const float* t, float* d) const;

    // evaluate(t, d, dddt)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void evaluate(const float* t, float* d, float* dddt) const;

    // solve(): two Newton steps from m_T
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void solve();
  };

  //===========================================================================
  // MiniPropagators.cc: vectorized version
  //===========================================================================

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void InitialStatePlex<N>::init_momentum_vec_and_k(const float* phi,
                                                                                        const int* chg,
                                                                                        float bf) {
    MPLEX_SIMD
    for (int i = 0; i < N; ++i) {
      const float pt = 1.0f / inv_pt[i];
      ::mkfitdev::vdt::fast_sincosf(phi[i], this->py[i], this->px[i]);
      this->px[i] *= pt;
      this->py[i] *= pt;
      this->pz[i] = pt / ::mkfitdev::vdt::fast_tanf(theta[i]);
      inv_k[i] = ((chg[i] < 0) ? 0.01f : -0.01f) * mconst::sol * bf;
    }
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void InitialStatePlex<N>::init_momentum_vec_and_k(const float* phi,
                                                                                        const int* chg,
                                                                                        const float* bf) {
    init_momentum_vec_and_k(phi, chg, 1.0f);
    for (int i = 0; i < N; ++i)
      inv_k[i] *= bf[i];
  }

  //===========================================================================
  // MiniPropagators.cc: Hermite3D interpolation
  //===========================================================================

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3D<N>::hermite_1d(
      float x1, float p1, float x2, float p2, float derfac, float (&h)[kHermiteCoeffs][N], int n) {
    const float P = x2 - x1;
    const float Q = p1 * derfac;
    const float R = p2 * derfac - 2.0f * P + Q;
    h[0][n] = x1;
    h[1][n] = Q;
    h[2][n] = P - Q - R;
    h[3][n] = R;
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3D<N>::calculate_coeffs(const StatePlex<N>& sp1,
                                                                          const StatePlex<N>& sp2,
                                                                          const float* inv_k) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      m_Hderfac[n] = (sp2.dalpha[n] - sp1.dalpha[n]) / inv_k[n];
      hermite_1d(sp1.x[n], sp1.px[n], sp2.x[n], sp2.px[n], m_Hderfac[n], m_Hx, n);
      hermite_1d(sp1.y[n], sp1.py[n], sp2.y[n], sp2.py[n], m_Hderfac[n], m_Hy, n);
      hermite_1d(sp1.z[n], sp1.pz[n], sp2.z[n], sp2.pz[n], m_Hderfac[n], m_Hz, n);
      m_Hderfac[n] = 1.0f / m_Hderfac[n];
    }
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3D<N>::evaluate(const float* t, float* x, float* y, float* z) const {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float t2 = t[n] * t[n];
      const float t3 = t2 * t[n];
      x[n] = m_Hx[0][n] + t[n] * m_Hx[1][n] + t2 * m_Hx[2][n] + t3 * m_Hx[3][n];
      y[n] = m_Hy[0][n] + t[n] * m_Hy[1][n] + t2 * m_Hy[2][n] + t3 * m_Hy[3][n];
      z[n] = m_Hz[0][n] + t[n] * m_Hz[1][n] + t2 * m_Hz[2][n] + t3 * m_Hz[3][n];
    }
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3D<N>::evaluate(
      const float* t, float* x, float* y, float* z, float* dx, float* dy, float* dz) const {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float t2 = t[n] * t[n];
      const float t3 = t2 * t[n];
      x[n] = m_Hx[0][n] + t[n] * m_Hx[1][n] + t2 * m_Hx[2][n] + t3 * m_Hx[3][n];
      y[n] = m_Hy[0][n] + t[n] * m_Hy[1][n] + t2 * m_Hy[2][n] + t3 * m_Hy[3][n];
      z[n] = m_Hz[0][n] + t[n] * m_Hz[1][n] + t2 * m_Hz[2][n] + t3 * m_Hz[3][n];
      dx[n] = (m_Hx[1][n] + 2.0f * t[n] * m_Hx[2][n] + 3.0f * t2 * m_Hx[3][n]) * m_Hderfac[n];
      dy[n] = (m_Hy[1][n] + 2.0f * t[n] * m_Hy[2][n] + 3.0f * t2 * m_Hy[3][n]) * m_Hderfac[n];
      dz[n] = (m_Hz[1][n] + 2.0f * t[n] * m_Hz[2][n] + 3.0f * t2 * m_Hz[3][n]) * m_Hderfac[n];
    }
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3D<N>::evaluate(const float* t, StatePlex<N>& sp) const {
    evaluate(t, sp.x, sp.y, sp.z, sp.px, sp.py, sp.pz);
  }

  //===========================================================================
  // MiniPropagators.cc: Hermite3DOnPlane intersection
  //===========================================================================

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3DOnPlane<N>::init_coeffs(const Hermite3D<N>& h,
                                                                            const ::mkfitdev::MPlexHV<N>& pos,
                                                                            const ::mkfitdev::MPlexHV<N>& zdir) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      m_D[0][n] = (pos(n, 0, 0) - h.m_Hx[0][n]) * zdir(n, 0, 0) + (pos(n, 1, 0) - h.m_Hy[0][n]) * zdir(n, 1, 0) +
                  (pos(n, 2, 0) - h.m_Hz[0][n]) * zdir(n, 2, 0);
      for (int i = 1; i < kHermiteCoeffs; ++i) {
        m_D[i][n] = -(h.m_Hx[i][n] * zdir(n, 0, 0) + h.m_Hy[i][n] * zdir(n, 1, 0) + h.m_Hz[i][n] * zdir(n, 2, 0));
      }
    }

    float t0[N], t1[N], d0[N], d1[N];
    for (int n = 0; n < N; ++n) {
      t0[n] = 0.0f;
      t1[n] = 1.0f;
    }
    evaluate(t0, d0);
    evaluate(t1, d1);
    for (int n = 0; n < N; ++n)
      m_T[n] = d0[n] / (d0[n] - d1[n]);
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3DOnPlane<N>::evaluate(const float* t, float* d) const {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float t2 = t[n] * t[n];
      d[n] = m_D[0][n] + m_D[1][n] * t[n] + m_D[2][n] * t2 + m_D[3][n] * t2 * t[n];
    }
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3DOnPlane<N>::evaluate(const float* t, float* d, float* dddt) const {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float t2 = t[n] * t[n];
      d[n] = m_D[0][n] + m_D[1][n] * t[n] + m_D[2][n] * t2 + m_D[3][n] * t2 * t[n];
      dddt[n] = m_D[1][n] + 2.0f * m_D[2][n] * t[n] + 3.0f * m_D[3][n] * t2;
    }
  }

  // Two Newton steps. For a plane of constant z the first is exact, since z is linear
  // along the cubic; a barrel module, or a tilted one, needs the second. See
  // MkFitCore doc/MkFinderV2p2-DesignNotes.md, "Layer crossings and the Hermite cubic".
  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Hermite3DOnPlane<N>::solve() {
    float f[N], df[N];
    evaluate(m_T, f, df);
    for (int n = 0; n < N; ++n)
      m_T[n] -= f[n] / df[n];
    evaluate(m_T, f, df);
    for (int n = 0; n < N; ++n)
      m_T[n] -= f[n] / df[n];
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::mini_propagators

#endif
