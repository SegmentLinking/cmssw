#ifndef RecoTracker_MkFitAlpaka_src_alpaka_select_MiniPropagators_h
#define RecoTracker_MkFitAlpaka_src_alpaka_select_MiniPropagators_h

// Portable transliteration of MkFitCore mini propagators (CMSSW_20_1_0_pre2 RecoTracker/MkFitCore/src/
// MiniPropagators.{h,cc}), as used by MkFinder::selectHitIndicesV2 of the LST step.
//   - State / InitialState: the scalar flavour (per-hit propagation). State(par) uses std::cos/sin/tan as MkFitCore.
//   - StatePlex / InitialStatePlex: the MkFitCore "Plex" flavour (used for the bin limits) over N slots. MkFitCore
//     evaluates it element-wise on MPlexQF with vdt fast_sincos / fast_tan; every operation is per element, so any
//     N (CPU batches: kNN, one candidate: 1) is the same arithmetic per slot.
// Only the algorithms the LST step reaches are ported: propagate_to_r(PA_Exact), propagate_to_z(PA_Exact),
// propagate_to_plane(PA_Line). PA_Line/PA_Quadratic of to_r/to_z fall through to PA_Exact; the plane
// PA_Quadratic/PA_Exact throw in MkFitCore and are never called.

#include <cmath>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"
#include "RecoTracker/MkFitAlpaka/interface/math/vdtMath.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::mini_propagators {

  namespace mconst = ::mkfitdev::Const;
  namespace mconfig = ::mkfitdev::Config;

  enum PropAlgo_e { PA_Line, PA_Quadratic, PA_Exact };

  // Module plane of ModuleInfo as used by propagate_to_plane: center position and normal (zdir).
  struct ModulePlane {
    float pos[3];
    float zdir[3];
  };

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
    s.dalpha = 0.f;  // MkFitCore leaves dalpha and fail_flag indeterminate; never read before written
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

    // InitialState::propagate_to_r (PA_Exact); returns the fail flag
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool propagate_to_r(float R, State& c) const {
      // Momentum is always updated -- used as temporary for stepping.
      const float k = 1.0f / inv_k;

      const float curv = 0.5f * inv_k * inv_pt;
      const float oo_curv = 1.0f / curv;  // 2 * radius of curvature
      const float lambda = pz * inv_pt;

      float D = 0;

      c = *this;
      c.dalpha = 0;
      for (int i = 0; i < mconfig::Niter; ++i) {
        // 3-rd order asin for symmetric incidence (shortest arc length).
        float r0 = ::mkfitdev::hipo(c.x, c.y);
        float td = (R - r0) * curv;
        float id = oo_curv * td * (1.0f + 0.16666666f * td * td);
        D += id;

        float alpha = id * inv_pt * inv_k;
        float sina, cosa;
        ::mkfitdev::vdt::fast_sincosf(alpha, sina, cosa);

        c.dalpha += alpha;
        c.x += k * (c.px * sina - c.py * (1.0f - cosa));
        c.y += k * (c.py * sina + c.px * (1.0f - cosa));

        const float o_px = c.px;  // copy before overwriting
        c.px = c.px * cosa - c.py * sina;
        c.py = c.py * cosa + o_px * sina;
      }

      c.z += lambda * D;

      c.fail_flag = std::abs(::mkfitdev::hipo(c.x, c.y) - R) < 0.1f ? 0 : 1;
      return c.fail_flag;
    }

    // InitialState::propagate_to_z (PA_Exact, update_momentum = true); never fails
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool propagate_to_z(float Z, State& c) const {
      const float k = 1.0f / inv_k;

      const float dz = Z - z;
      const float alpha = dz * inv_k / pz;

      float sina, cosa;
      ::mkfitdev::vdt::fast_sincosf(alpha, sina, cosa);

      c.dalpha = alpha;
      c.x = x + k * (px * sina - py * (1.0f - cosa));
      c.y = y + k * (py * sina + px * (1.0f - cosa));
      c.z = Z;

      c.px = px * cosa - py * sina;
      c.py = py * cosa + px * sina;
      c.pz = pz;

      c.fail_flag = 0;
      return c.fail_flag;
    }

    // InitialState::propagate_to_plane (PA_Line): straight step along the momentum to the module plane;
    // MkFitCore returns false (no failure) and does not touch c.fail_flag
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool propagate_to_plane(const ModulePlane& mi, State& c) const {
      // t * p_vec intersects the plane:
      float t = (mi.pos[0] * mi.zdir[0] + mi.pos[1] * mi.zdir[1] + mi.pos[2] * mi.zdir[2] - x * mi.zdir[0] -
                 y * mi.zdir[1] - z * mi.zdir[2]) /
                (px * mi.zdir[0] + py * mi.zdir[1] + pz * mi.zdir[2]);

      c = *this;
      c.x += t * c.px;
      c.y += t * c.py;
      c.z += t * c.pz;
      return false;
    }
  };

  //-----------------------------------------------------------
  // The MkFitCore vectorized flavour (StatePlex / InitialStatePlex) over N slots: every Matriplex expression of
  // MkFitCore is one slot loop (MPLEX_SIMD on the CPU backends); one candidate is N = 1.
  //-----------------------------------------------------------

  // Matriplex::hypot (a*a + b*b, then sqrt) as MkFitCore's -Ofast x86-64-v3 build contracts it (see InitialStatePlex).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float hypotV(float a, float b) { return std::sqrt(std::fma(b, b, a * a)); }

  template <int N>
  struct StatePlex {
    float x[N], y[N], z[N];
    float px[N], py[N], pz[N];
    float dalpha[N];
    int fail_flag[N];
  };

  // MkFitCore evaluates Matriplex expressions that GCC (-Ofast, x86-64-v3) contracts as a*b + c*d -> fma(c, d, a*b),
  // a*b - c*d -> fma(-c, d, a*b), x + k*y -> fma(k, y, x); the explicit fma forms below reproduce that (checked against
  // Bins::sp1/sp2 dumps).
  template <int N>
  struct InitialStatePlex : public StatePlex<N> {
    float inv_pt[N], inv_k[N];
    float theta[N];

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

    // InitialStatePlex::propagate_to_r (PA_Exact) to R[n]; fail if |R - r| > 0.1
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void propagate_to_r(const float* R, StatePlex<N>& c) const {
      float k[N], curv[N], oo_curv[N], lambda[N], D[N];
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        // Momentum is always updated -- used as temporary for stepping.
        k[n] = 1.0f / inv_k[n];
        curv[n] = 0.5f * inv_k[n] * inv_pt[n];
        oo_curv[n] = 1.0f / curv[n];  // 2 * radius of curvature
        lambda[n] = this->pz[n] * inv_pt[n];
        D[n] = 0;
        c.x[n] = this->x[n];
        c.y[n] = this->y[n];
        c.z[n] = this->z[n];
        c.px[n] = this->px[n];
        c.py[n] = this->py[n];
        c.pz[n] = this->pz[n];
        c.dalpha[n] = 0;
      }
      for (int i = 0; i < mconfig::Niter; ++i) {
        float alpha[N], sina[N], cosa[N];
        MPLEX_SIMD
        for (int n = 0; n < N; ++n) {
          const float r0 = hypotV(c.x[n], c.y[n]);
          const float td = (R[n] - r0) * curv[n];
          const float id = oo_curv[n] * td * std::fma(0.16666666f * td, td, 1.0f);
          D[n] += id;
          alpha[n] = id * inv_pt[n] * inv_k[n];
        }
        MPLEX_SIMD
        for (int n = 0; n < N; ++n)
          ::mkfitdev::vdt::fast_sincosf(alpha[n], sina[n], cosa[n]);
        MPLEX_SIMD
        for (int n = 0; n < N; ++n) {
          c.dalpha[n] += alpha[n];
          c.x[n] = std::fma(k[n], std::fma(-c.py[n], 1.0f - cosa[n], c.px[n] * sina[n]), c.x[n]);
          c.y[n] = std::fma(k[n], std::fma(c.px[n], 1.0f - cosa[n], c.py[n] * sina[n]), c.y[n]);

          const float o_px = c.px[n];  // copy before overwriting
          c.px[n] = std::fma(-c.py[n], sina[n], c.px[n] * cosa[n]);
          c.py[n] = std::fma(o_px, sina[n], c.py[n] * cosa[n]);
        }
      }
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        c.z[n] = std::fma(lambda[n], D[n], c.z[n]);
        const float r = hypotV(c.x[n], c.y[n]);
        c.fail_flag[n] = (std::abs(R[n] - r) > 0.1f) ? 1 : 0;
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

        c.dalpha[n] = alpha;
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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::mini_propagators

#endif
