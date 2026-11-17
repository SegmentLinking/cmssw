#ifndef RecoTracker_MkFitAlpaka_src_alpaka_prop_KalmanUtilsMPlex_h
#define RecoTracker_MkFitAlpaka_src_alpaka_prop_KalmanUtilsMPlex_h

// Portable (Alpaka host+device) transliteration of MkFitCore Kalman operations (src/KalmanUtilsMPlex.h/.cc, the
// plane steps of PlaneSteps.h, and the final fit's update of FinalFit.cc), the plane-local flavour used by the
// Phase-2 LST step (Config::usePropToPlane = true): building (kalmanPropagateAndComputeChi2Plane,
// kalmanPropagateAndUpdatePlane), backward fit (propagateHelixToPlaneMPlex + kalmanOperationPlaneLocal),
// final fit (kalmanPropagateAndUpdateAndChi2Plane with PropagationFlags::final_fit = final_fit::propagate_update:
// the covariance update in Joseph form, update_joseph). Same names / argument order, templated on N.
// CPE-correction hook (doCPE + cpe_func of kalmanOperationPlaneLocal / kalmanPropagateAndUpdateAndChi2Plane): a
// functor template argument. Not ported: the barrel (3D/R) and endcap (Z) Kalman flavours and
// kalmanOperationPlane (global-frame plane update), none of which the Phase-2 LST step calls.

#include <cmath>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PropMatriplex.h"
#include "PropMath.h"
#include "PropagationFlags.h"
#include "PropagationMPlex.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using namespace ::mkfitdev::prop;

  enum KalmanFilterOperation { KFO_Calculate_Chi2 = 1, KFO_Update_Params = 2, KFO_Local_Cov = 4 };

  // CPE hook (doCPE + cpe_corr_func, plane::measurement_cpe). A functor with
  // `static constexpr bool enabled` and `bool operator()(int slot, const float (&ltp)[6], float (&lh)[5]) const`:
  // ltp = local (q/p, dxdz, dydz, x, y, pz sign) on the plane, lh = (x, y, exx, exy, eyy) of the hit in the local
  // frame. true replaces the projected hit position and error of that slot; false keeps them, as
  // plane::measurement_cpe.
  struct NoCpe {
    static constexpr bool enabled = false;
    ALPAKA_FN_HOST_ACC bool operator()(int, const float (&)[6], float (&)[5]) const { return false; }
  };

  // LocalStatesOut (final_fit::LocalStatesOut): optional outputs of the plane-local update in the module's local
  // frame (q/p, dx/dz, dy/dz, x, y): the predicted and the updated state and the sign of the local z momentum. Used
  // for the final fit's per-hit states (storeHitStates).
  template <idx_t N>
  struct LocalStatesOut {
    MPlex5V<N>* predPar = nullptr;
    MPlex5S<N>* predErr = nullptr;
    MPlex5V<N>* updPar = nullptr;
    MPlex5S<N>* updErr = nullptr;
    MPlexQI<N>* pzSign = nullptr;
  };

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void kalmanCheckChargeFlip(MPlexLV<N>& outPar, MPlexQI<N>& Chg, int N_proc) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      if (n < N_proc && outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

  namespace kalmandetail {

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MultResidualsAdd(const MPlex52<N>& A,
                                                              const MPlex5V<N>& B,
                                                              const MPlex2V<N>& C,
                                                              MPlex5V<N>& D) {
      // outPar = psPar + kalmanGain*(dPar)
      //   D    =   B         A         C
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      const T* c = C.fArray;
      T* d = D.fArray;

      for (idx_t n = 0; n < N; ++n) {
        d[0 * N + n] = b[0 * N + n] + a[0 * N + n] * c[0 * N + n] + a[1 * N + n] * c[1 * N + n];
        d[1 * N + n] = b[1 * N + n] + a[2 * N + n] * c[0 * N + n] + a[3 * N + n] * c[1 * N + n];
        d[2 * N + n] = b[2 * N + n] + a[4 * N + n] * c[0 * N + n] + a[5 * N + n] * c[1 * N + n];
        d[3 * N + n] = b[3 * N + n] + a[6 * N + n] * c[0 * N + n] + a[7 * N + n] * c[1 * N + n];
        d[4 * N + n] = b[4 * N + n] + a[8 * N + n] * c[0 * N + n] + a[9 * N + n] * c[1 * N + n];
      }
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void Chi2Similarity(const MPlex2V<N>& A,  //resPar
                                                            const MPlex2S<N>& C,  //resErr
                                                            MPlexQF<N>& D)        //outChi2
    {
      // outChi2 = (resPar) * resErr * (resPar)
      //   D     =    A      *    C   *      A
      typedef float T;
      const T* a = A.fArray;
      const T* c = C.fArray;
      T* d = D.fArray;

      for (idx_t n = 0; n < N; ++n) {
        d[0 * N + n] = c[0 * N + n] * a[0 * N + n] * a[0 * N + n] + c[2 * N + n] * a[1 * N + n] * a[1 * N + n] +
                       2 * (c[1 * N + n] * a[1 * N + n] * a[0 * N + n]);
      }
    }

    // C = A * B, C is 2x3, A is 2x3 (first two rows of a 3x3 rotation), B is 3x3 sym
    template <idx_t N, class T1, class T2>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void ProjectResErr(const T1& A, const T2& B, MPlex2H<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;

      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[1 * N + n] + a[2 * N + n] * b[3 * N + n];
        c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[2 * N + n] + a[2 * N + n] * b[4 * N + n];
        c[2 * N + n] = a[0 * N + n] * b[3 * N + n] + a[1 * N + n] * b[4 * N + n] + a[2 * N + n] * b[5 * N + n];
        c[3 * N + n] = a[3 * N + n] * b[0 * N + n] + a[4 * N + n] * b[1 * N + n] + a[5 * N + n] * b[3 * N + n];
        c[4 * N + n] = a[3 * N + n] * b[1 * N + n] + a[4 * N + n] * b[2 * N + n] + a[5 * N + n] * b[4 * N + n];
        c[5 * N + n] = a[3 * N + n] * b[3 * N + n] + a[4 * N + n] * b[4 * N + n] + a[5 * N + n] * b[5 * N + n];
      }
    }

    // C = B * A^T, C is 2x2 sym, A is 2x3 (A^T is 3x2), B is 2x3
    template <idx_t N, class T1>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void ProjectResErrTransp(const T1& A, const MPlex2H<N>& B, MPlex2S<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;

      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        c[0 * N + n] = b[0 * N + n] * a[0 * N + n] + b[1 * N + n] * a[1 * N + n] + b[2 * N + n] * a[2 * N + n];
        c[1 * N + n] = b[0 * N + n] * a[3 * N + n] + b[1 * N + n] * a[4 * N + n] + b[2 * N + n] * a[5 * N + n];
        c[2 * N + n] = b[3 * N + n] * a[3 * N + n] + b[4 * N + n] * a[4 * N + n] + b[5 * N + n] * a[5 * N + n];
      }
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void RotateVectorOnPlane(const MPlexHH<N>& R,
                                                                 const MPlexHV<N>& A,
                                                                 MPlexHV<N>& B) {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        B(n, 0, 0) = R(n, 0, 0) * A(n, 0, 0) + R(n, 0, 1) * A(n, 1, 0) + R(n, 0, 2) * A(n, 2, 0);
        B(n, 1, 0) = R(n, 1, 0) * A(n, 0, 0) + R(n, 1, 1) * A(n, 1, 0) + R(n, 1, 2) * A(n, 2, 0);
        B(n, 2, 0) = R(n, 2, 0) * A(n, 0, 0) + R(n, 2, 1) * A(n, 1, 0) + R(n, 2, 2) * A(n, 2, 0);
      }
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void RotateVectorOnPlaneTransp(const MPlexHH<N>& R,
                                                                       const MPlexHV<N>& A,
                                                                       MPlexHV<N>& B) {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        B(n, 0, 0) = R(n, 0, 0) * A(n, 0, 0) + R(n, 1, 0) * A(n, 1, 0) + R(n, 2, 0) * A(n, 2, 0);
        B(n, 1, 0) = R(n, 0, 1) * A(n, 0, 0) + R(n, 1, 1) * A(n, 1, 0) + R(n, 2, 1) * A(n, 2, 0);
        B(n, 2, 0) = R(n, 0, 2) * A(n, 0, 0) + R(n, 1, 2) * A(n, 1, 0) + R(n, 2, 2) * A(n, 2, 0);
      }
    }

    template <idx_t N, typename T1, typename T2>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void RotateResidualsOnPlane(const T1& R,      //prj - at least MPlex_2_3
                                                                    const T2& A,      //res_glo - at least MPlex_3_1
                                                                    MPlex2V<N>& B) {  //res_loc - MPlex_2_1
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        B(n, 0, 0) = R(n, 0, 0) * A(n, 0, 0) + R(n, 0, 1) * A(n, 1, 0) + R(n, 0, 2) * A(n, 2, 0);
        B(n, 1, 0) = R(n, 1, 0) * A(n, 0, 0) + R(n, 1, 1) * A(n, 1, 0) + R(n, 1, 2) * A(n, 2, 0);
      }
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void JacCCS2Loc(const MPlex55<N>& A, const MPlex56<N>& B, MPlex56<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "JacCCS2Loc.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void PsErrLoc(const MPlex56<N>& A, const MPlexLS<N>& B, MPlex56<N>& C) {
      // C = A * B
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "PsErrLoc.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void PsErrLocTransp(const MPlex56<N>& B, const MPlex56<N>& A, MPlex5S<N>& C) {
      // C = B * AT;
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "PsErrLocTransp.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void PsErrLocUpd(const MPlex55<N>& A, const MPlex5S<N>& B, MPlex5S<N>& C) {
      // C = A * B;
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "PsErrLocUpd.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void JacLoc2CCS(const MPlex65<N>& A, const MPlex55<N>& B, MPlex65<N>& C) {
      // C = A * B;
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "JacLoc2CCS.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void OutErrCCS(const MPlex65<N>& A, const MPlex5S<N>& B, MPlex65<N>& C) {
      // C = A * B
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "OutErrCCS.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void OutErrCCSTransp(const MPlex65<N>& B, const MPlex65<N>& A, MPlexLS<N>& C) {
      // C = B * AT;
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "OutErrCCSTransp.ah"
    }

    // Matriplex::invertCramerSym for 2x2 (MatriplexSym.h CramerInverterSym<T, 2, N>): determinant in double.
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void invertCramerSym(MPlex2S<N>& A) {
      typedef float TT;
      float* a = A.fArray;
      for (idx_t n = 0; n < N; ++n) {
        // Force determinant calculation in double precision.
        const double det = (double)a[0 * N + n] * a[2 * N + n] - (double)a[1 * N + n] * a[1 * N + n];
        const TT s = TT(1) / det;
        const TT tmp = s * a[2 * N + n];
        a[1 * N + n] *= -s;
        a[2 * N + n] = s * a[0 * N + n];
        a[0 * N + n] = tmp;
      }
    }

  }  // namespace kalmandetail

  //============================================================================
  // kalmanOperationPlaneLocal: plane::local_field, to_local, measurement, measurement_cpe, residual, chi2, update
  // (joseph: final_fit update_joseph), to_global
  //============================================================================

  template <idx_t N, typename TCpe = NoCpe>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void kalmanOperationPlaneLocal(const int kfOp,
                                                                     const MPlexLS<N>& psErr,
                                                                     const MPlexLV<N>& psPar,
                                                                     const MPlexQI<N>& inChg,
                                                                     const MPlexHS<N>& msErr,
                                                                     const MPlexHV<N>& msPar,
                                                                     const MPlexHV<N>& plNrm,
                                                                     const MPlexHV<N>& plDir,
                                                                     const MPlexHV<N>& plPnt,
                                                                     MPlexLS<N>& outErr,
                                                                     MPlexLV<N>& outPar,
                                                                     MPlexQF<N>& outChi2,
                                                                     const int N_proc,
                                                                     const TCpe& cpe = TCpe{},
                                                                     const bool use_param_b_field = false,
                                                                     const Config::BFieldParams& bField = {},
                                                                     const LocalStatesOut<N>* localStates = nullptr,
                                                                     const bool joseph = false) {
    using namespace kalmandetail;

    MPlexHH<N> rot;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      rot(n, 0, 0) = plDir(n, 0, 0);
      rot(n, 0, 1) = plDir(n, 1, 0);
      rot(n, 0, 2) = plDir(n, 2, 0);
      rot(n, 1, 0) = plNrm(n, 1, 0) * plDir(n, 2, 0) - plNrm(n, 2, 0) * plDir(n, 1, 0);
      rot(n, 1, 1) = plNrm(n, 2, 0) * plDir(n, 0, 0) - plNrm(n, 0, 0) * plDir(n, 2, 0);
      rot(n, 1, 2) = plNrm(n, 0, 0) * plDir(n, 1, 0) - plNrm(n, 1, 0) * plDir(n, 0, 0);
      rot(n, 2, 0) = plNrm(n, 0, 0);
      rot(n, 2, 1) = plNrm(n, 1, 0);
      rot(n, 2, 2) = plNrm(n, 2, 0);
    }

    // get local parameters
    MPlexHV<N> xd;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      xd(n, 0, 0) = psPar(n, 0, 0) - plPnt(n, 0, 0);
      xd(n, 1, 0) = psPar(n, 1, 0) - plPnt(n, 1, 0);
      xd(n, 2, 0) = psPar(n, 2, 0) - plPnt(n, 2, 0);
    }
    MPlex2V<N> xlo;
    RotateResidualsOnPlane<N>(rot, xd, xlo);

    MPlexQF<N> sinP, sinT, cosP, cosT, pt;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      pt(n, 0, 0) = 1.f / psPar(n, 3, 0);
      sinP(n, 0, 0) = std::sin(psPar(n, 4, 0));
      cosP(n, 0, 0) = std::cos(psPar(n, 4, 0));
      sinT(n, 0, 0) = std::sin(psPar(n, 5, 0));
      cosT(n, 0, 0) = std::cos(psPar(n, 5, 0));
    }

    MPlexHV<N> pgl;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      pgl(n, 0, 0) = cosP(n, 0, 0) * pt(n, 0, 0);
      pgl(n, 1, 0) = sinP(n, 0, 0) * pt(n, 0, 0);
      pgl(n, 2, 0) = cosT(n, 0, 0) * pt(n, 0, 0) / sinT(n, 0, 0);
    }

    MPlexHV<N> plo;
    RotateVectorOnPlane(rot, pgl, plo);
    MPlex5V<N> lp;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      lp(n, 0, 0) = inChg(n, 0, 0) * psPar(n, 3, 0) * sinT(n, 0, 0);
      lp(n, 1, 0) = plo(n, 0, 0) / plo(n, 2, 0);
      lp(n, 2, 0) = plo(n, 1, 0) / plo(n, 2, 0);
      lp(n, 3, 0) = xlo(n, 0, 0);
      lp(n, 4, 0) = xlo(n, 1, 0);
    }
    MPlexQI<N> pzSign;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      pzSign(n, 0, 0) = plo(n, 2, 0) > 0.f ? 1 : -1;
    }

    [[maybe_unused]] float cpeHit[N][5];
    [[maybe_unused]] bool cpeOk[N];
    if constexpr (TCpe::enabled) {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        const float ltp[6] = {lp(n, 0, 0), lp(n, 1, 0), lp(n, 2, 0), lp(n, 3, 0), lp(n, 4, 0), (float)pzSign(n, 0, 0)};
        cpeOk[n] = cpe(n, ltp, cpeHit[n]);
      }
    }

    MPlex56<N> jacCCS2Curv(0.f);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      jacCCS2Curv(n, 0, 3) = inChg(n, 0, 0) * sinT(n, 0, 0);
      jacCCS2Curv(n, 0, 5) = inChg(n, 0, 0) * cosT(n, 0, 0) * psPar(n, 3, 0);
      jacCCS2Curv(n, 1, 5) = -1.f;
      jacCCS2Curv(n, 2, 4) = 1.f;
      jacCCS2Curv(n, 3, 0) = -sinP(n, 0, 0);
      jacCCS2Curv(n, 3, 1) = cosP(n, 0, 0);
      jacCCS2Curv(n, 4, 0) = -cosP(n, 0, 0) * cosT(n, 0, 0);
      jacCCS2Curv(n, 4, 1) = -sinP(n, 0, 0) * cosT(n, 0, 0);
      jacCCS2Curv(n, 4, 2) = sinT(n, 0, 0);
    }

    MPlexHV<N> un;
    MPlexHV<N> vn;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float abslp00 = std::abs(lp(n, 0, 0));
      vn(n, 2, 0) = std::max(1.e-30f, abslp00 * pt(n, 0, 0));
      un(n, 0, 0) = -pgl(n, 1, 0) * abslp00 / vn(n, 2, 0);
      un(n, 1, 0) = pgl(n, 0, 0) * abslp00 / vn(n, 2, 0);
      un(n, 2, 0) = 0.f;
      vn(n, 0, 0) = -pgl(n, 2, 0) * abslp00 * un(n, 1, 0);
      vn(n, 1, 0) = pgl(n, 2, 0) * abslp00 * un(n, 0, 0);
    }
    MPlexHV<N> u;
    RotateVectorOnPlane(rot, un, u);
    MPlexHV<N> v;
    RotateVectorOnPlane(rot, vn, v);
    MPlex55<N> jacCurv2Loc(0.f);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      // plane::local_field: the parametrised field at the state when use_param_b_field (bField: the building's or,
      // with use_fit_b_field, the final fit's constants)
      const float bF =
          use_param_b_field
              ? 0.01f * Const::sol * Config::bFieldFromZR(bField, psPar(n, 2, 0), hipo(psPar(n, 0, 0), psPar(n, 1, 0)))
              : 0.01f * Const::sol * Config::Bfield;
      const float qh2 = bF * lp(n, 0, 0);
      const float t1r = std::sqrt(1.f + lp(n, 1, 0) * lp(n, 1, 0) + lp(n, 2, 0) * lp(n, 2, 0)) * pzSign(n, 0, 0);
      const float t2r = t1r * t1r;
      const float t3r = t1r * t2r;
      jacCurv2Loc(n, 0, 0) = 1.f;
      jacCurv2Loc(n, 1, 1) = -u(n, 1, 0) * t2r;
      jacCurv2Loc(n, 1, 2) = v(n, 1, 0) * vn(n, 2, 0) * t2r;
      jacCurv2Loc(n, 2, 1) = u(n, 0, 0) * t2r;
      jacCurv2Loc(n, 2, 2) = -v(n, 0, 0) * vn(n, 2, 0) * t2r;
      jacCurv2Loc(n, 3, 3) = v(n, 1, 0) * t1r;
      jacCurv2Loc(n, 3, 4) = -u(n, 1, 0) * t1r;
      jacCurv2Loc(n, 4, 3) = -v(n, 0, 0) * t1r;
      jacCurv2Loc(n, 4, 4) = u(n, 0, 0) * t1r;
      const float cosz = -vn(n, 2, 0) * qh2;
      const float ui = u(n, 2, 0) * t3r;
      const float vi = v(n, 2, 0) * t3r;
      jacCurv2Loc(n, 1, 3) = -ui * v(n, 1, 0) * cosz;
      jacCurv2Loc(n, 1, 4) = -vi * v(n, 1, 0) * cosz;
      jacCurv2Loc(n, 2, 3) = ui * v(n, 0, 0) * cosz;
      jacCurv2Loc(n, 2, 4) = vi * v(n, 0, 0) * cosz;
    }

    // jacCCS2Loc = jacCurv2Loc*jacCCS2Curv
    MPlex56<N> jacCCS2Loc;
    JacCCS2Loc(jacCurv2Loc, jacCCS2Curv, jacCCS2Loc);

    // local errors
    MPlex5S<N> psErrLoc;
    MPlex56<N> temp56;
    PsErrLoc(jacCCS2Loc, psErr, temp56);
    PsErrLocTransp(temp56, jacCCS2Loc, psErrLoc);

    // local measurement
    MPlexHV<N> md;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      md(n, 0, 0) = msPar(n, 0, 0) - plPnt(n, 0, 0);
      md(n, 1, 0) = msPar(n, 1, 0) - plPnt(n, 1, 0);
      md(n, 2, 0) = msPar(n, 2, 0) - plPnt(n, 2, 0);
    }
    MPlex2V<N> mslo;
    RotateResidualsOnPlane<N>(rot, md, mslo);
    if constexpr (TCpe::enabled) {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        if (cpeOk[n]) {
          mslo(n, 0, 0) = cpeHit[n][0];
          mslo(n, 1, 0) = cpeHit[n][1];
        }
      }
    }

    MPlex2V<N> res_loc;  //position residual in local coordinates
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      res_loc(n, 0, 0) = mslo(n, 0, 0) - xlo(n, 0, 0);
      res_loc(n, 1, 0) = mslo(n, 1, 0) - xlo(n, 1, 0);
    }

    MPlex2S<N> msErr_loc;
    MPlex2H<N> temp2Hmsl;
    ProjectResErr<N>(rot, msErr, temp2Hmsl);
    ProjectResErrTransp<N>(rot, temp2Hmsl, msErr_loc);
    if constexpr (TCpe::enabled) {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        if (cpeOk[n]) {
          msErr_loc(n, 0, 0) = cpeHit[n][2];
          msErr_loc(n, 0, 1) = cpeHit[n][3];
          msErr_loc(n, 1, 1) = cpeHit[n][4];
        }
      }
    }

    MPlex2S<N> resErr_loc;  //covariance sum in local position coordinates
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      resErr_loc(n, 0, 0) = psErrLoc(n, 3, 3) + msErr_loc(n, 0, 0);
      resErr_loc(n, 0, 1) = psErrLoc(n, 3, 4) + msErr_loc(n, 0, 1);
      resErr_loc(n, 1, 1) = psErrLoc(n, 4, 4) + msErr_loc(n, 1, 1);
    }

    //invert the 2x2 matrix
    invertCramerSym(resErr_loc);

    if (kfOp & KFO_Calculate_Chi2) {
      Chi2Similarity(res_loc, resErr_loc, outChi2);
    }

    if (kfOp & KFO_Update_Params) {
      MPlex52<N> K;  // kalman gain
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        for (int j = 0; j < 5; ++j) {
          K(n, j, 0) = resErr_loc(n, 0, 0) * psErrLoc(n, j, 3) + resErr_loc(n, 0, 1) * psErrLoc(n, j, 4);
          K(n, j, 1) = resErr_loc(n, 0, 1) * psErrLoc(n, j, 3) + resErr_loc(n, 1, 1) * psErrLoc(n, j, 4);
        }
      }

      MPlex5V<N> lp_upd;
      MultResidualsAdd(K, lp, res_loc, lp_upd);

      MPlex55<N> ImKH(0.f);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        for (int j = 0; j < 5; ++j) {
          ImKH(n, j, j) = 1.f;
          ImKH(n, j, 3) -= K(n, j, 0);
          ImKH(n, j, 4) -= K(n, j, 1);
        }
      }
      MPlex5S<N> psErrLoc_upd;
      if (!joseph) {
        PsErrLocUpd(ImKH, psErrLoc, psErrLoc_upd);
      } else {
        // final_fit update_joseph: C' = (I - K H) C (I - K H)^T + K V K^T, V the hit's local covariance (msErr_loc,
        // after the CPE), H picking the local position (parameters 3 and 4); the gain K and the parameters as above
        MPLEX_SIMD
        for (int n = 0; n < N; ++n) {
          float gainV[5][2], fwdTerm[5][5];
          for (int i = 0; i < 5; ++i) {
            gainV[i][0] = K(n, i, 0) * msErr_loc(n, 0, 0) + K(n, i, 1) * msErr_loc(n, 0, 1);
            gainV[i][1] = K(n, i, 0) * msErr_loc(n, 0, 1) + K(n, i, 1) * msErr_loc(n, 1, 1);
          }
          for (int j = 0; j < 5; ++j)
            for (int i = 0; i < 5; ++i)
              fwdTerm[i][j] = psErrLoc(n, i, j) - K(n, i, 0) * psErrLoc(n, 3, j) - K(n, i, 1) * psErrLoc(n, 4, j);
          for (int i = 0; i < 5; ++i)
            for (int j = 0; j <= i; ++j)
              psErrLoc_upd(n, i, j) = fwdTerm[i][j] - fwdTerm[i][3] * K(n, j, 0) - fwdTerm[i][4] * K(n, j, 1) +
                                      gainV[i][0] * K(n, j, 0) + gainV[i][1] * K(n, j, 1);
        }
      }
      if (localStates) {
        if (localStates->predPar)
          *localStates->predPar = lp;
        if (localStates->predErr)
          *localStates->predErr = psErrLoc;
        if (localStates->updPar)
          *localStates->updPar = lp_upd;
        if (localStates->updErr)
          *localStates->updErr = psErrLoc_upd;
        if (localStates->pzSign)
          *localStates->pzSign = pzSign;
      }

      //convert local updated parameters into CCS
      MPlexHV<N> lxu;
      MPlexHV<N> lpu;
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        lxu(n, 0, 0) = lp_upd(n, 3, 0);
        lxu(n, 1, 0) = lp_upd(n, 4, 0);
        lxu(n, 2, 0) = 0.f;
        lpu(n, 2, 0) =
            pzSign(n, 0, 0) / (std::max(std::abs(lp_upd(n, 0, 0)), 1.e-9f) *
                               std::sqrt(1.f + lp_upd(n, 1, 0) * lp_upd(n, 1, 0) + lp_upd(n, 2, 0) * lp_upd(n, 2, 0)));
        lpu(n, 0, 0) = lpu(n, 2, 0) * lp_upd(n, 1, 0);
        lpu(n, 1, 0) = lpu(n, 2, 0) * lp_upd(n, 2, 0);
      }
      MPlexHV<N> gxu;
      RotateVectorOnPlaneTransp(rot, lxu, gxu);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        gxu(n, 0, 0) += plPnt(n, 0, 0);
        gxu(n, 1, 0) += plPnt(n, 1, 0);
        gxu(n, 2, 0) += plPnt(n, 2, 0);
      }
      MPlexHV<N> gpu;
      RotateVectorOnPlaneTransp(rot, lpu, gpu);

      MPlexQF<N> p;
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        pt(n, 0, 0) = std::sqrt(gpu.At(n, 0, 0) * gpu.At(n, 0, 0) + gpu.At(n, 1, 0) * gpu.At(n, 1, 0));
        p(n, 0, 0) = std::sqrt(pt.At(n, 0, 0) * pt.At(n, 0, 0) + gpu.At(n, 2, 0) * gpu.At(n, 2, 0));
        sinP(n, 0, 0) = gpu.At(n, 1, 0) / pt(n, 0, 0);
        cosP(n, 0, 0) = gpu.At(n, 0, 0) / pt(n, 0, 0);
        sinT(n, 0, 0) = pt(n, 0, 0) / p(n, 0, 0);
        cosT(n, 0, 0) = gpu.At(n, 2, 0) / p(n, 0, 0);
      }

      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        outPar(n, 0, 0) = gxu.At(n, 0, 0);
        outPar(n, 1, 0) = gxu.At(n, 1, 0);
        outPar(n, 2, 0) = gxu.At(n, 2, 0);
        outPar(n, 3, 0) = 1.f / pt(n, 0, 0);
        outPar(n, 4, 0) = getPhi(gpu.At(n, 0, 0), gpu.At(n, 1, 0));  //fixme VDT or something?
        outPar(n, 5, 0) = getTheta(pt(n, 0, 0), gpu.At(n, 2, 0));
      }

      MPlex65<N> jacCurv2CCS(0.f);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        jacCurv2CCS(n, 0, 3) = -sinP(n, 0, 0);
        jacCurv2CCS(n, 0, 4) = -cosT(n, 0, 0) * cosP(n, 0, 0);
        jacCurv2CCS(n, 1, 3) = cosP(n, 0, 0);
        jacCurv2CCS(n, 1, 4) = -cosT(n, 0, 0) * sinP(n, 0, 0);
        jacCurv2CCS(n, 2, 4) = sinT(n, 0, 0);
        jacCurv2CCS(n, 3, 0) = inChg(n, 0, 0) / sinT(n, 0, 0);
        jacCurv2CCS(n, 3, 1) = outPar(n, 3, 0) * cosT(n, 0, 0) / sinT(n, 0, 0);
        jacCurv2CCS(n, 4, 2) = 1.f;
        jacCurv2CCS(n, 5, 1) = -1.f;
        if (std::signbit(lp_upd(n, 0, 0)) != std::signbit(lp(n, 0, 0))) {
          outPar(n, 3, 0) = -outPar(n, 3, 0);
          jacCurv2CCS(n, 3, 0) = -jacCurv2CCS(n, 3, 0);
        }
      }

      MPlexHV<N> tnl;
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        const float abslpupd00 = std::max(std::abs(lp_upd(n, 0, 0)), 1.e-9f);
        tnl(n, 0, 0) = lpu(n, 0, 0) * abslpupd00;
        tnl(n, 1, 0) = lpu(n, 1, 0) * abslpupd00;
        tnl(n, 2, 0) = lpu(n, 2, 0) * abslpupd00;
      }
      MPlexHV<N> tn;
      RotateVectorOnPlaneTransp(rot, tnl, tn);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        vn(n, 2, 0) = std::max(1.e-30f, std::sqrt(tn(n, 0, 0) * tn(n, 0, 0) + tn(n, 1, 0) * tn(n, 1, 0)));
        un(n, 0, 0) = -tn(n, 1, 0) / vn(n, 2, 0);
        un(n, 1, 0) = tn(n, 0, 0) / vn(n, 2, 0);
        un(n, 2, 0) = 0.f;
        vn(n, 0, 0) = -tn(n, 2, 0) * un(n, 1, 0);
        vn(n, 1, 0) = tn(n, 2, 0) * un(n, 0, 0);
      }
      MPlex55<N> jacLoc2Curv(0.f);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        const float bF = use_param_b_field
                             ? 0.01f * Const::sol *
                                   Config::bFieldFromZR(bField, psPar(n, 2, 0), hipo(psPar(n, 0, 0), psPar(n, 1, 0)))
                             : 0.01f * Const::sol * Config::Bfield;  //fixme: cache?
        const float qh2 = bF * lp_upd(n, 0, 0);
        const float cosl1 = 1.f / vn(n, 2, 0);
        const float uj = un(n, 0, 0) * rot(n, 0, 0) + un(n, 1, 0) * rot(n, 0, 1);
        const float uk = un(n, 0, 0) * rot(n, 1, 0) + un(n, 1, 0) * rot(n, 1, 1);
        const float vj = vn(n, 0, 0) * rot(n, 0, 0) + vn(n, 1, 0) * rot(n, 0, 1) + vn(n, 2, 0) * rot(n, 0, 2);
        const float vk = vn(n, 0, 0) * rot(n, 1, 0) + vn(n, 1, 0) * rot(n, 1, 1) + vn(n, 2, 0) * rot(n, 1, 2);
        const float cosz = vn(n, 2, 0) * qh2;
        jacLoc2Curv(n, 0, 0) = 1.f;
        jacLoc2Curv(n, 1, 1) = tnl(n, 2, 0) * vj;
        jacLoc2Curv(n, 1, 2) = tnl(n, 2, 0) * vk;
        jacLoc2Curv(n, 2, 1) = tnl(n, 2, 0) * uj * cosl1;
        jacLoc2Curv(n, 2, 2) = tnl(n, 2, 0) * uk * cosl1;
        jacLoc2Curv(n, 3, 3) = uj;
        jacLoc2Curv(n, 3, 4) = uk;
        jacLoc2Curv(n, 4, 3) = vj;
        jacLoc2Curv(n, 4, 4) = vk;
        jacLoc2Curv(n, 2, 3) = tnl(n, 0, 0) * (cosz * cosl1);
        jacLoc2Curv(n, 2, 4) = tnl(n, 1, 0) * (cosz * cosl1);
      }

      MPlex65<N> jacLoc2CCS;
      JacLoc2CCS(jacCurv2CCS, jacLoc2Curv, jacLoc2CCS);

      MPlex65<N> temp65;
      OutErrCCS(jacLoc2CCS, psErrLoc_upd, temp65);
      OutErrCCSTransp(temp65, jacLoc2CCS, outErr);
    }
  }

  //============================================================================
  // Kalman operations - Plane wrappers (KalmanUtilsMPlex.cc)
  //============================================================================

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void kalmanUpdatePlane(const MPlexLS<N>& psErr,
                                                             const MPlexLV<N>& psPar,
                                                             const MPlexQI<N>& Chg,
                                                             const MPlexHS<N>& msErr,
                                                             const MPlexHV<N>& msPar,
                                                             const MPlexHV<N>& plNrm,
                                                             const MPlexHV<N>& plDir,
                                                             const MPlexHV<N>& plPnt,
                                                             MPlexLS<N>& outErr,
                                                             MPlexLV<N>& outPar,
                                                             const int N_proc) {
    MPlexQF<N> dummy_chi2;
    kalmanOperationPlaneLocal(KFO_Update_Params | KFO_Local_Cov,
                              psErr,
                              psPar,
                              Chg,
                              msErr,
                              msPar,
                              plNrm,
                              plDir,
                              plPnt,
                              outErr,
                              outPar,
                              dummy_chi2,
                              N_proc);
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void kalmanPropagateAndUpdatePlane(const MPlexLS<N>& psErr,
                                                                         const MPlexLV<N>& psPar,
                                                                         MPlexQI<N>& Chg,
                                                                         const MPlexHS<N>& msErr,
                                                                         const MPlexHV<N>& msPar,
                                                                         const MPlexHV<N>& plNrm,
                                                                         const MPlexHV<N>& plDir,
                                                                         const MPlexHV<N>& plPnt,
                                                                         MPlexLS<N>& outErr,
                                                                         MPlexLV<N>& outPar,
                                                                         MPlexQI<N>& outFailFlag,
                                                                         const int N_proc,
                                                                         const PropagationFlags& propFlags,
                                                                         const bool propToHit) {
    MPlexQF<N> dummy_chi2;
    if (propToHit) {
      MPlexLS<N> propErr;
      MPlexLV<N> propPar;
      propagateHelixToPlaneMPlex(psErr, psPar, Chg, plPnt, plNrm, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperationPlaneLocal(KFO_Update_Params | KFO_Local_Cov,
                                propErr,
                                propPar,
                                Chg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                outErr,
                                outPar,
                                dummy_chi2,
                                N_proc);
    } else {
      kalmanOperationPlaneLocal(KFO_Update_Params | KFO_Local_Cov,
                                psErr,
                                psPar,
                                Chg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                outErr,
                                outPar,
                                dummy_chi2,
                                N_proc);
    }
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      if (outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

  template <idx_t N, typename TCpe = NoCpe>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void kalmanPropagateAndUpdateAndChi2Plane(
      const MPlexLS<N>& psErr,
      const MPlexLV<N>& psPar,
      MPlexQI<N>& Chg,
      const MPlexHS<N>& msErr,
      const MPlexHV<N>& msPar,
      const MPlexHV<N>& plNrm,
      const MPlexHV<N>& plDir,
      const MPlexHV<N>& plPnt,
      MPlexLS<N>& outErr,
      MPlexLV<N>& outPar,
      MPlexQI<N>& outFailFlag,
      MPlexQF<N>& outChi2,
      const int N_proc,
      const PropagationFlags& propFlags,
      const bool propToHit,
      const MPlexQI<N>* noMatEffPtr = nullptr,
      const TCpe& cpe = TCpe{},
      const MPlexQF<N>* matRadl = nullptr,
      const MPlexQF<N>* matBbxi = nullptr,
      const MPlexQF<N>* msRefP = nullptr,
      const LocalStatesOut<N>* localStates = nullptr) {
    if (propToHit) {
      MPlexLS<N> propErr;
      MPlexLV<N> propPar;

      propagateHelixToPlaneMPlex(psErr,
                                 psPar,
                                 Chg,
                                 plPnt,
                                 plNrm,
                                 propErr,
                                 propPar,
                                 outFailFlag,
                                 N_proc,
                                 propFlags,
                                 noMatEffPtr,
                                 matRadl,
                                 matBbxi,
                                 msRefP);

      kalmanOperationPlaneLocal(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                                propErr,
                                propPar,
                                Chg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                outErr,
                                outPar,
                                outChi2,
                                N_proc,
                                cpe,
                                propFlags.use_param_b_field,
                                propFlags.startField(),
                                localStates,
                                propFlags.final_fit);
    } else {
      kalmanOperationPlaneLocal(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                                psErr,
                                psPar,
                                Chg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                outErr,
                                outPar,
                                outChi2,
                                N_proc,
                                cpe,
                                propFlags.use_param_b_field,
                                propFlags.startField(),
                                localStates,
                                propFlags.final_fit);
    }
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      if (outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void kalmanComputeChi2Plane(const MPlexLS<N>& psErr,
                                                                  const MPlexLV<N>& psPar,
                                                                  const MPlexQI<N>& inChg,
                                                                  const MPlexHS<N>& msErr,
                                                                  const MPlexHV<N>& msPar,
                                                                  const MPlexHV<N>& plNrm,
                                                                  const MPlexHV<N>& plDir,
                                                                  const MPlexHV<N>& plPnt,
                                                                  MPlexQF<N>& outChi2,
                                                                  const int N_proc) {
    // never accessed with KFO_Calculate_Chi2 only (MkFitCore passes file-scope dummies)
    MPlexLS<N> dummy_err;
    MPlexLV<N> dummy_par;
    kalmanOperationPlaneLocal(KFO_Calculate_Chi2,
                              psErr,
                              psPar,
                              inChg,
                              msErr,
                              msPar,
                              plNrm,
                              plDir,
                              plPnt,
                              dummy_err,
                              dummy_par,
                              outChi2,
                              N_proc);
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void kalmanPropagateAndComputeChi2Plane(const MPlexLS<N>& psErr,
                                                                              const MPlexLV<N>& psPar,
                                                                              const MPlexQI<N>& inChg,
                                                                              const MPlexHS<N>& msErr,
                                                                              const MPlexHV<N>& msPar,
                                                                              const MPlexHV<N>& plNrm,
                                                                              const MPlexHV<N>& plDir,
                                                                              const MPlexHV<N>& plPnt,
                                                                              MPlexQF<N>& outChi2,
                                                                              MPlexLV<N>& propPar,
                                                                              MPlexQI<N>& outFailFlag,
                                                                              const int N_proc,
                                                                              const PropagationFlags& propFlags,
                                                                              const bool propToHit) {
    MPlexLS<N> dummy_err;
    MPlexLV<N> dummy_par;
    propPar = psPar;
    if (propToHit) {
      MPlexLS<N> propErr;
      propagateHelixToPlaneMPlex(psErr, psPar, inChg, plPnt, plNrm, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperationPlaneLocal(KFO_Calculate_Chi2,
                                propErr,
                                propPar,
                                inChg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                dummy_err,
                                dummy_par,
                                outChi2,
                                N_proc);
    } else {
      kalmanOperationPlaneLocal(KFO_Calculate_Chi2,
                                psErr,
                                psPar,
                                inChg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                dummy_err,
                                dummy_par,
                                outChi2,
                                N_proc);
    }
  }

  //============================================================================
  // smoothLocalStatesPlane (final_fit::smooth_local_states): two-filter smoother on one module plane, combining the
  // forward updated (xf, Cf) and the backward predicted (xb, Cb) state. S = Cf + Cb = L L^T (Cholesky); with
  // Yf = L^-1 Cf and z = L^-1 (xb - xf): xs = xf + Yf^T z, and Cs = (I - K) Cf (I - K)^T + K Cb K^T in Joseph form
  // with the gain K = Cf S^-1 (K^T = S^-1 Cf by back substitution of L^T K^T = Yf). ok = 0 where S is not positive
  // definite (a pivot of its Cholesky factorisation is not positive). Per slot, the same operations in the same
  // order as MkFitCore.
  //============================================================================
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void smoothLocalStatesPlane(const MPlex5V<N>& xf,
                                                                  const MPlex5S<N>& cf,
                                                                  const MPlex5V<N>& xb,
                                                                  const MPlex5S<N>& cb,
                                                                  MPlex5V<N>& xs,
                                                                  MPlex5S<N>& cs,
                                                                  MPlexQI<N>& ok,
                                                                  const int N_proc) {
    constexpr int D = 5;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      float l[D][D], ld[D], a;
      int okn = 1;
      for (int j = 0; j < D; ++j) {
        a = cf.constAt(n, j, j) + cb.constAt(n, j, j);
        for (int k = 0; k < j; ++k)
          a -= l[j][k] * l[j][k];
        okn &= a > 0.f;
        a = a > 1.e-30f ? a : 1.e-30f;
        l[j][j] = std::sqrt(a);
        ld[j] = 1.f / l[j][j];
        for (int i = j + 1; i < D; ++i) {
          a = cf.constAt(n, i, j) + cb.constAt(n, i, j);
          for (int k = 0; k < j; ++k)
            a -= l[i][k] * l[j][k];
          l[i][j] = a * ld[j];
        }
      }
      ok(n, 0, 0) = okn;
      // forward substitution L y = b for b = xb - xf and the five columns of Cf
      float z[D], yf[D][D];
      for (int i = 0; i < D; ++i) {
        z[i] = xb.constAt(n, i, 0) - xf.constAt(n, i, 0);
        for (int k = 0; k < i; ++k)
          z[i] -= l[i][k] * z[k];
        z[i] *= ld[i];
      }
      for (int c = 0; c < D; ++c)
        for (int i = 0; i < D; ++i) {
          yf[i][c] = cf.constAt(n, i, c);
          for (int k = 0; k < i; ++k)
            yf[i][c] -= l[i][k] * yf[k][c];
          yf[i][c] *= ld[i];
        }
      for (int i = 0; i < D; ++i) {
        a = xf.constAt(n, i, 0);
        for (int k = 0; k < D; ++k)
          a += yf[k][i] * z[k];
        xs(n, i, 0) = a;
      }
      // the gain, K^T = S^-1 Cf, by back substitution L^T K^T = Yf
      float gt[D][D];
      for (int c = 0; c < D; ++c)
        for (int i = D - 1; i >= 0; --i) {
          a = yf[i][c];
          for (int k = i + 1; k < D; ++k)
            a -= l[k][i] * gt[k][c];
          gt[i][c] = a * ld[i];
        }
      // fwdTerm = (I - K) Cf and gainCb = K Cb, with K(i, m) = gt(m, i)
      float fwdTerm[D][D], gainCb[D][D];
      for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j) {
          fwdTerm[i][j] = cf.constAt(n, i, j);
          gainCb[i][j] = 0.f;
          for (int m = 0; m < D; ++m) {
            fwdTerm[i][j] -= gt[m][i] * cf.constAt(n, m, j);
            gainCb[i][j] += gt[m][i] * cb.constAt(n, m, j);
          }
        }
      // Cs = fwdTerm (I - K)^T + gainCb K^T
      for (int i = 0; i < D; ++i)
        for (int j = 0; j <= i; ++j) {
          a = fwdTerm[i][j];
          for (int m = 0; m < D; ++m)
            a += (gainCb[i][m] - fwdTerm[i][m]) * gt[m][j];
          cs(n, i, j) = a;
        }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
