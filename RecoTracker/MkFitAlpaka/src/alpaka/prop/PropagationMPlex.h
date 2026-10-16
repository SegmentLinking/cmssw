#ifndef RecoTracker_MkFitAlpaka_src_alpaka_prop_PropagationMPlex_h
#define RecoTracker_MkFitAlpaka_src_alpaka_prop_PropagationMPlex_h

// Portable (Alpaka host+device) transliteration of MkFitCore propagation (CMSSW_20_1_0_pre2):
//   src/PropagationMPlex.h, PropagationMPlexCommon.cc, PropagationMPlex.cc (helix to R),
//   PropagationMPlexEndcap.cc (helix to Z), PropagationMPlexPlane.cc (helix to plane).
// Same names, same argument order, same operations in the same order; templated on N (tracks per
// Alpaka thread: 1 on GPU, kNN on CPU). Matriplex vector expressions of the MkFitCore plane code are
// written as explicit per-slot loops (same per-element arithmetic). Material comes from
// PropagationFlags::material (MaterialView) instead of TrackerInfo, and so do the field constants
// (MaterialView::bField = mkfit::Config::mag_*). the reference is MkFitCore; its plane
// code (stable getS root, shared start trigonometry, B at the chord midpoint, radial-field kicks, sub-stepped
// propagation, per-module material, pass-signed energy loss, fixed-momentum MS) is transliterated.

#include <cmath>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PropMatriplex.h"
#include "PropMath.h"
#include "PropagationFlags.h"

// Host compilers (serial/TBB backends) compile the R and Z propagators out of line and without IPA, as MkFitCore
// compiles them (exported functions of PropagationMPlex.cc / PropagationMPlexEndcap.cc). GCC's FMA contraction
// (-ffp-contract=fast) then depends only on the propagator body, not on the caller's inlining context, so every
// translation unit (select's K2, the prop test) gets the same, MkFitCore-identical rounding.
// Device compilers keep forced inlining.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define MKFITDEV_PROP_HOST_OUTLINE ALPAKA_FN_INLINE
#else
#define MKFITDEV_PROP_HOST_OUTLINE __attribute__((noipa)) inline
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using namespace ::mkfitdev::prop;

  //============================================================================
  // PropagationMPlex.h inlines
  //============================================================================

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void squashPhiMPlex(MPlexLV<N>& par, const int N_proc) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      if (n < N_proc) {
        if (par(n, 4, 0) >= Const::PI)
          par(n, 4, 0) -= Const::TwoPI;
        if (par(n, 4, 0) < -Const::PI)
          par(n, 4, 0) += Const::TwoPI;
      }
    }
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void squashPhiMPlexGeneral(MPlexLV<N>& par, const int N_proc) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      par(n, 4, 0) -= std::floor(0.5f * Const::InvPI * (par(n, 4, 0) + Const::PI)) * Const::TwoPI;
    }
  }

  //============================================================================
  // Generated multiplications (the .ah files are the MkFitCore scalar branches)
  //============================================================================

  namespace propdetail {

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MultHelixProp(const MPlexLL<N>& A, const MPlexLS<N>& B, MPlexLL<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "MultHelixProp.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MultHelixPropTransp(const MPlexLL<N>& A,
                                                                 const MPlexLL<N>& B,
                                                                 MPlexLS<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "MultHelixPropTransp.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MultHelixPropEndcap(const MPlexLL<N>& A,
                                                                 const MPlexLS<N>& B,
                                                                 MPlexLL<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "MultHelixPropEndcap.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MultHelixPropTranspEndcap(const MPlexLL<N>& A,
                                                                       const MPlexLL<N>& B,
                                                                       MPlexLS<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "MultHelixPropTranspEndcap.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MultHelixPlaneProp(const MPlexLL<N>& A,
                                                                const MPlexLS<N>& B,
                                                                MPlexLL<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "MultHelixPlaneProp.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void MultHelixPlanePropTransp(const MPlexLL<N>& A,
                                                                      const MPlexLL<N>& B,
                                                                      MPlexLS<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "MultHelixPlanePropTransp.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void JacErrPropCurv1(const MPlex65<N>& A, const MPlex55<N>& B, MPlex65<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "JacErrPropCurv1.ah"
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void JacErrPropCurv2(const MPlex65<N>& A, const MPlex56<N>& B, MPlexLL<N>& C) {
      typedef float T;
      const T* a = A.fArray;
      const T* b = B.fArray;
      T* c = C.fArray;
#include "JacErrPropCurv2.ah"
    }

  }  // namespace propdetail

  //============================================================================
  // PropagationMPlexCommon.cc
  //============================================================================

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void applyMaterialEffects(const MPlexQF<N>& hitsRl,
                                                                const MPlexQF<N>& hitsXi,
                                                                const MPlexQF<N>& propSign,
                                                                const MPlexHV<N>& plNrm,
                                                                MPlexLS<N>& outErr,
                                                                MPlexLV<N>& outPar,
                                                                const int N_proc,
                                                                const MPlexQF<N>* msRefP = nullptr) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      if (n >= N_proc)
        continue;
      float radL = hitsRl.constAt(n, 0, 0);
      if (radL < 1e-13f)
        continue;  //ugly, please fixme
      const float theta = outPar.constAt(n, 5, 0);
      const float ipt = outPar.constAt(n, 3, 0);
      const float pt = 1.f / ipt;  //fixme, make sure it is positive?
      const float ipt2 = ipt * ipt;
      float sT;
      float cT;
      vdt::fast_sincosf(theta, sT, cT);
      const float p = pt / sT;
      const float pz = p * cT;
      const float p2 = p * p;
      constexpr float mpi = 0.140;       // m=140 MeV, pion
      constexpr float mpi2 = mpi * mpi;  // m=140 MeV, pion
      const float beta2 = p2 / (p2 + mpi2);
      const float beta = std::sqrt(beta2);
      //radiation lenght, corrected for the crossing angle (cos alpha from dot product of radius vector and momentum)
      float sinP;
      float cosP;
      vdt::fast_sincosf(outPar.constAt(n, 4, 0), sinP, cosP);
      const float invCos = p / std::abs(pt * cosP * plNrm.constAt(n, 0, 0) + pt * sinP * plNrm.constAt(n, 1, 0) +
                                        pz * plNrm.constAt(n, 2, 0));
      radL = radL * invCos;  // general: invCos is p/|p.n| with n the module normal
      if (radL < 1e-13f)
        continue;
      // msRefP (PropagationFlags::ms_ref_p): theta0 at a fixed reference momentum instead of the
      // running estimate; only theta0 changes
      const float pMS = msRefP ? msRefP->constAt(n, 0, 0) : p;
      const float betaMS = msRefP ? std::sqrt(pMS * pMS / (pMS * pMS + mpi2)) : beta;
      const float thetaMSC = 0.0136f * (1.f + 0.038f * vdt::fast_logf(radL)) / (betaMS * pMS);  // eq 32.15
      const float thetaMSC2 = thetaMSC * thetaMSC * radL;
      if constexpr (Config::usePtMultScat) {
        outErr.At(n, 3, 3) += thetaMSC2 * pz * pz * ipt2 * ipt2;
        outErr.At(n, 3, 5) -= thetaMSC2 * pz * ipt2;
        outErr.At(n, 4, 4) += thetaMSC2 * p2 * ipt2;
        outErr.At(n, 5, 5) += thetaMSC2;
      } else {
        outErr.At(n, 4, 4) += thetaMSC2;
        outErr.At(n, 5, 5) += thetaMSC2;
      }
      // energy loss
      const float gamma2 = (p2 + mpi2) / mpi2;
      const float gamma = std::sqrt(gamma2);  //1.f/std::sqrt(1.f - std::min(beta2, 0.999999f));
      constexpr float me = 0.0005;            // m=0.5 MeV, electron
      const float wmax = 2.f * me * beta2 * gamma2 / (1.f + 2.f * gamma * me / mpi + me * me / (mpi * mpi));
      constexpr float I = 16.0e-9 * 10.75;
      const float deltahalf = vdt::fast_logf(28.816e-9f * std::sqrt(2.33f * 0.498f) / I) - 0.5f;
      const float dEdx = beta < 1.f
                             ? (2.f * (hitsXi.constAt(n, 0, 0) * invCos *
                                       (0.5f * vdt::fast_logf(2.f * me * wmax / (I * I)) - beta2 - deltahalf) / beta2))
                             : 0.f;  //protect against infs and nans
      const float dP = propSign.constAt(n, 0, 0) * dEdx / beta;
      outPar.At(n, 3, 0) = p / (std::max(p - dP, 0.001f) * pt);  //stay above 1MeV
      // energy-loss straggling variance, as MkFitCore (the double literal 0.5 is MkFitCore's)
      const float dEdx2 = (hitsXi.constAt(n, 0, 0) * invCos / beta2) * wmax * (1 - beta2 * 0.5);
      outErr.At(n, 3, 3) += dEdx2 / (beta2 * p2 * pt * pt);
    }
  }

  //============================================================================
  // PropagationMPlex.cc : helix to R
  //============================================================================

  namespace propdetail {

    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixAtRFromIterativeCCS_impl(const MPlexLV<N>& __restrict__ inPar,
                                                                           const MPlexQI<N>& __restrict__ inChg,
                                                                           const MPlexQF<N>& __restrict__ msRad,
                                                                           MPlexLV<N>& __restrict__ outPar,
                                                                           MPlexLL<N>& __restrict__ errorProp,
                                                                           MPlexQI<N>& __restrict__ outFailFlag,
                                                                           const int N_proc,
                                                                           const PropagationFlags& pf) {
      // MkFitCore runs every step over all NN slots (nmin = 0, nmax = NN); kept per slot here.
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        errorProp(n, 0, 0) = 1.f;
        errorProp(n, 1, 1) = 1.f;
        errorProp(n, 2, 2) = 1.f;
        errorProp(n, 3, 3) = 1.f;
        errorProp(n, 4, 4) = 1.f;
        errorProp(n, 5, 5) = 1.f;
      }
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        float r0 = hipo(inPar(n, 0, 0), inPar(n, 1, 0));
        float k;
        if (pf.use_param_b_field) {
          k = inChg(n, 0, 0) * 100.f / (-Const::sol * Config::bFieldFromZR(pf.material.bField, inPar(n, 2, 0), r0));
        } else {
          k = inChg(n, 0, 0) * 100.f / (-Const::sol * Config::Bfield);
        }
        const float r = msRad(n, 0, 0);

        const float xin = inPar(n, 0, 0);
        const float yin = inPar(n, 1, 0);
        const float ipt = inPar(n, 3, 0);
        const float phiin = inPar(n, 4, 0);
        const float theta = inPar(n, 5, 0);

        const float kinv = 1.f / k;
        const float pt = 1.f / ipt;

        float D = 0.;
        float cosa, sina, cosah, sinah, id;

        float cosPorT = std::cos(phiin);
        float sinPorT = std::sin(phiin);

        float pxin = cosPorT * pt;
        float pyin = sinPorT * pt;

        float dDdx, dDdy, dDdipt, dDdphi;
        dDdipt = 0.;
        dDdphi = 0.;
        dDdx = r0 > 0.f ? -xin / r0 : 0.f;
        dDdy = r0 > 0.f ? -yin / r0 : 0.f;

        float oodotp, x, y, oor0, dadipt, dadx, dady, pxca, pxsa, pyca, pysa, tmp, tmpx, tmpy, pxinold;

        for (int i = 0; i < Config::Niter; ++i) {
          r0 = hipo(outPar(n, 0, 0), outPar(n, 1, 0));

          oodotp = r0 * pt / (pxin * outPar(n, 0, 0) + pyin * outPar(n, 1, 0));

          if (oodotp > 5.0f || oodotp < 0)  // 0.2 is 78.5 deg
          {
            outFailFlag(n, 0, 0) = 1;
            oodotp = 0.0f;
          } else if (r - r0 < 0.0f && pt < 1.0f) {
            oodotp = 1.0f + (oodotp - 1.0f) * pt;
          }

          id = (r - r0) * oodotp;

          D += id;

          if constexpr (Config::useTrigApprox) {
            sincos4(id * ipt * kinv * 0.5f, sinah, cosah);
          } else {
            cosah = std::cos(id * ipt * kinv * 0.5f);
            sinah = std::sin(id * ipt * kinv * 0.5f);
          }

          cosa = 1.f - 2.f * sinah * sinah;
          sina = 2.f * sinah * cosah;

          if (i + 1 != Config::Niter) {
            x = outPar(n, 0, 0);
            y = outPar(n, 1, 0);
            oor0 = (r0 > 0.f && std::abs(r - r0) > 0.0001f) ? 1.f / r0 : 0.f;
            dadipt = id * kinv;
            dadx = -x * ipt * kinv * oor0;
            dady = -y * ipt * kinv * oor0;
            pxca = pxin * cosa;
            pxsa = pxin * sina;
            pyca = pyin * cosa;
            pysa = pyin * sina;
            tmpx = k * dadx;

            dDdx -= (x * (1.f + tmpx * (pxca - pysa)) + y * tmpx * (pyca + pxsa)) * oor0;

            tmpy = k * dady;
            dDdy -= (x * tmpy * (pxca - pysa) + y * (1.f + tmpy * (pyca + pxsa))) * oor0;
            tmp = dadipt * ipt;
            dDdipt -= k *
                      (x * (pxca * tmp - pysa * tmp - pyca - pxsa + pyin) +
                       y * (pyca * tmp + pxsa * tmp - pysa + pxca - pxin)) *
                      pt * oor0;
            dDdphi += k * (x * (pysa - pxin + pxca) - y * (pxsa - pyin + pyca)) * oor0;
          }

          outPar(n, 0, 0) = outPar(n, 0, 0) + 2.f * k * sinah * (pxin * cosah - pyin * sinah);
          outPar(n, 1, 0) = outPar(n, 1, 0) + 2.f * k * sinah * (pyin * cosah + pxin * sinah);
          pxinold = pxin;  //copy before overwriting
          pxin = pxin * cosa - pyin * sina;
          pyin = pyin * cosa + pxinold * sina;
        }  // iteration loop

        const float alpha = D * ipt * kinv;
        dadx = dDdx * ipt * kinv;
        dady = dDdy * ipt * kinv;
        dadipt = (ipt * dDdipt + D) * kinv;
        const float dadphi = dDdphi * ipt * kinv;

        if constexpr (Config::useTrigApprox) {
          sincos4(alpha, sina, cosa);
        } else {
          cosa = std::cos(alpha);
          sina = std::sin(alpha);
        }

        errorProp(n, 0, 0) = 1.f + k * dadx * (cosPorT * cosa - sinPorT * sina) * pt;
        errorProp(n, 0, 1) = k * dady * (cosPorT * cosa - sinPorT * sina) * pt;
        errorProp(n, 0, 2) = 0.f;
        errorProp(n, 0, 3) =
            k * (cosPorT * (ipt * dadipt * cosa - sina) + sinPorT * ((1.f - cosa) - ipt * dadipt * sina)) * pt * pt;
        errorProp(n, 0, 4) =
            k * (cosPorT * dadphi * cosa - sinPorT * dadphi * sina - sinPorT * sina + cosPorT * cosa - cosPorT) * pt;
        errorProp(n, 0, 5) = 0.f;

        errorProp(n, 1, 0) = k * dadx * (sinPorT * cosa + cosPorT * sina) * pt;
        errorProp(n, 1, 1) = 1.f + k * dady * (sinPorT * cosa + cosPorT * sina) * pt;
        errorProp(n, 1, 2) = 0.f;
        errorProp(n, 1, 3) =
            k * (sinPorT * (ipt * dadipt * cosa - sina) + cosPorT * (ipt * dadipt * sina - (1.f - cosa))) * pt * pt;
        errorProp(n, 1, 4) =
            k * (sinPorT * dadphi * cosa + cosPorT * dadphi * sina + sinPorT * cosa + cosPorT * sina - sinPorT) * pt;
        errorProp(n, 1, 5) = 0.f;

        cosPorT = std::cos(theta);
        sinPorT = std::sin(theta);
        sinPorT = 1.f / sinPorT;

        outPar(n, 2, 0) = inPar(n, 2, 0) + k * alpha * cosPorT * pt * sinPorT;
        errorProp(n, 2, 0) = k * cosPorT * dadx * pt * sinPorT;
        errorProp(n, 2, 1) = k * cosPorT * dady * pt * sinPorT;
        errorProp(n, 2, 2) = 1.f;
        errorProp(n, 2, 3) = k * cosPorT * (ipt * dadipt - alpha) * pt * pt * sinPorT;
        errorProp(n, 2, 4) = k * dadphi * cosPorT * pt * sinPorT;
        errorProp(n, 2, 5) = -k * alpha * pt * sinPorT * sinPorT;

        outPar(n, 3, 0) = ipt;
        errorProp(n, 3, 0) = 0.f;
        errorProp(n, 3, 1) = 0.f;
        errorProp(n, 3, 2) = 0.f;
        errorProp(n, 3, 3) = 1.f;
        errorProp(n, 3, 4) = 0.f;
        errorProp(n, 3, 5) = 0.f;

        outPar(n, 4, 0) = inPar(n, 4, 0) + alpha;
        errorProp(n, 4, 0) = dadx;
        errorProp(n, 4, 1) = dady;
        errorProp(n, 4, 2) = 0.f;
        errorProp(n, 4, 3) = dadipt;
        errorProp(n, 4, 4) = 1.f + dadphi;
        errorProp(n, 4, 5) = 0.f;

        outPar(n, 5, 0) = theta;
        errorProp(n, 5, 0) = 0.f;
        errorProp(n, 5, 1) = 0.f;
        errorProp(n, 5, 2) = 0.f;
        errorProp(n, 5, 3) = 0.f;
        errorProp(n, 5, 4) = 0.f;
        errorProp(n, 5, 5) = 1.f;
      }
    }

  }  // namespace propdetail

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixAtRFromIterativeCCS(const MPlexLV<N>& inPar,
                                                                    const MPlexQI<N>& inChg,
                                                                    const MPlexQF<N>& msRad,
                                                                    MPlexLV<N>& outPar,
                                                                    MPlexLL<N>& errorProp,
                                                                    MPlexQI<N>& outFailFlag,
                                                                    const int N_proc,
                                                                    const PropagationFlags& pflags) {
    errorProp.setVal(0.f);
    outFailFlag.setVal(0.f);

    propdetail::helixAtRFromIterativeCCS_impl(inPar, inChg, msRad, outPar, errorProp, outFailFlag, N_proc, pflags);
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC MKFITDEV_PROP_HOST_OUTLINE void propagateHelixToRMPlex(const MPlexLS<N>& inErr,
                                                                            const MPlexLV<N>& inPar,
                                                                            const MPlexQI<N>& inChg,
                                                                            const MPlexQF<N>& msRad,
                                                                            MPlexLS<N>& outErr,
                                                                            MPlexLV<N>& outPar,
                                                                            MPlexQI<N>& outFailFlag,
                                                                            const int N_proc,
                                                                            const PropagationFlags& pflags,
                                                                            const MPlexQI<N>* noMatEffPtr = nullptr) {
    outErr = inErr;
    outPar = inPar;

    MPlexLL<N> errorProp;

    helixAtRFromIterativeCCS(inPar, inChg, msRad, outPar, errorProp, outFailFlag, N_proc, pflags);

    MPlexLL<N> temp;
    propdetail::MultHelixProp(errorProp, outErr, temp);
    propdetail::MultHelixPropTransp(errorProp, temp, outErr);

    if (pflags.apply_material) {
      MPlexQF<N> hitsRl;
      MPlexQF<N> hitsXi;
      MPlexQF<N> propSign;

      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        if (n < N_proc) {
          if (outFailFlag(n, 0, 0) || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0))) {
            hitsRl(n, 0, 0) = 0.f;
            hitsXi(n, 0, 0) = 0.f;
          } else {
            const auto mat = material_checked(pflags.material, std::abs(outPar(n, 2, 0)), msRad(n, 0, 0));
            hitsRl(n, 0, 0) = mat.radl;
            hitsXi(n, 0, 0) = mat.bbxi;
          }
          const float r0 = hipo(inPar(n, 0, 0), inPar(n, 1, 0));
          const float r = msRad(n, 0, 0);
          propSign(n, 0, 0) = (r > r0 ? 1.f : -1.f);
        }
      }
      MPlexHV<N> plNrm;
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        plNrm(n, 0, 0) = std::cos(outPar.constAt(n, 4, 0));
        plNrm(n, 1, 0) = std::sin(outPar.constAt(n, 4, 0));
        plNrm(n, 2, 0) = 0.f;
      }
      applyMaterialEffects(hitsRl, hitsXi, propSign, plNrm, outErr, outPar, N_proc);
    }

    squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

    for (int i = 0; i < N_proc; ++i) {
      if (outFailFlag(i, 0, 0)) {
        outPar.copySlot(i, inPar);
        outErr.copySlot(i, inErr);
      }
    }
  }

  //============================================================================
  // PropagationMPlexEndcap.cc : helix to Z
  //============================================================================

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixAtZ(const MPlexLV<N>& inPar,
                                                    const MPlexQI<N>& inChg,
                                                    const MPlexQF<N>& msZ,
                                                    MPlexLV<N>& outPar,
                                                    MPlexLL<N>& errorProp,
                                                    MPlexQI<N>& outFailFlag,
                                                    const int N_proc,
                                                    const PropagationFlags& pflags) {
    errorProp.setVal(0.f);
    outFailFlag.setVal(0.f);

    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      errorProp(n, 0, 0) = 1.f;
      errorProp(n, 1, 1) = 1.f;
      errorProp(n, 3, 3) = 1.f;
      errorProp(n, 4, 4) = 1.f;
      errorProp(n, 5, 5) = 1.f;
    }
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      // MkFitCore zero-initialises the per-slot inputs and fills them only for n < N_proc
      float zout = 0.0f, zin = 0.0f, ipt = 0.0f, phiin = 0.0f, theta = 0.0f;
      if (n < N_proc) {
        zout = msZ.constAt(n, 0, 0);
        zin = inPar.constAt(n, 2, 0);
        ipt = inPar.constAt(n, 3, 0);
        phiin = inPar.constAt(n, 4, 0);
        theta = inPar.constAt(n, 5, 0);
      }

      float k;
      if (pflags.use_param_b_field) {
        k = inChg.constAt(n, 0, 0) * 100.f /
            (-Const::sol *
             Config::bFieldFromZR(pflags.material.bField, zin, hipo(inPar.constAt(n, 0, 0), inPar.constAt(n, 1, 0))));
      } else {
        k = inChg.constAt(n, 0, 0) * 100.f / (-Const::sol * Config::Bfield);
      }

      const float kinv = 1.f / k;

      const float pt = 1.f / ipt;

      const float cosP = std::cos(phiin);
      const float sinP = std::sin(phiin);

      const float cosT = std::cos(theta);
      const float sinT = std::sin(theta);

      const float tanT = sinT / cosT;
      const float icos2T = 1.f / (cosT * cosT);
      const float pxin = cosP * pt;
      const float pyin = sinP * pt;

      const float deltaZ = zout - zin;
      const float alpha = deltaZ * tanT * ipt * kinv;

      float cosahTmp, sinahTmp;
      if constexpr (Config::useTrigApprox) {
        sincos4(alpha * 0.5f, sinahTmp, cosahTmp);
      } else {
        cosahTmp = std::cos(alpha * 0.5f);
        sinahTmp = std::sin(alpha * 0.5f);
      }

      const float cosah = cosahTmp;
      const float sinah = sinahTmp;
      const float cosa = 1.f - 2.f * sinah * sinah;
      const float sina = 2.f * sinah * cosah;

      outPar.At(n, 0, 0) = outPar.At(n, 0, 0) + 2.f * k * sinah * (pxin * cosah - pyin * sinah);
      outPar.At(n, 1, 0) = outPar.At(n, 1, 0) + 2.f * k * sinah * (pyin * cosah + pxin * sinah);
      outPar.At(n, 2, 0) = zout;
      outPar.At(n, 4, 0) = phiin + alpha;

      const float pxcaMpysa = pxin * cosa - pyin * sina;

      errorProp(n, 0, 2) = -tanT * ipt * pxcaMpysa;
      errorProp(n, 0, 3) = k * pt * pt * (cosP * (alpha * cosa - sina) + sinP * 2.f * sinah * (sinah - alpha * cosah));
      errorProp(n, 0, 4) = -2.f * k * pt * sinah * (sinP * cosah + cosP * sinah);
      errorProp(n, 0, 5) = deltaZ * ipt * pxcaMpysa * icos2T;

      const float pycaPpxsa = pyin * cosa + pxin * sina;

      errorProp(n, 1, 2) = -tanT * ipt * pycaPpxsa;
      errorProp(n, 1, 3) = k * pt * pt * (sinP * (alpha * cosa - sina) - cosP * 2.f * sinah * (sinah - alpha * cosah));
      errorProp(n, 1, 4) = 2.f * k * pt * sinah * (cosP * cosah - sinP * sinah);
      errorProp(n, 1, 5) = deltaZ * ipt * pycaPpxsa * icos2T;

      errorProp(n, 4, 2) = -ipt * tanT * kinv;
      errorProp(n, 4, 3) = tanT * deltaZ * kinv;
      errorProp(n, 4, 5) = ipt * deltaZ * kinv * icos2T;
    }
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC MKFITDEV_PROP_HOST_OUTLINE void propagateHelixToZMPlex(const MPlexLS<N>& inErr,
                                                                            const MPlexLV<N>& inPar,
                                                                            const MPlexQI<N>& inChg,
                                                                            const MPlexQF<N>& msZ,
                                                                            MPlexLS<N>& outErr,
                                                                            MPlexLV<N>& outPar,
                                                                            MPlexQI<N>& outFailFlag,
                                                                            const int N_proc,
                                                                            const PropagationFlags& pflags,
                                                                            const MPlexQI<N>* noMatEffPtr = nullptr) {
    outErr = inErr;
    outPar = inPar;

    MPlexLL<N> errorProp;

    helixAtZ(inPar, inChg, msZ, outPar, errorProp, outFailFlag, N_proc, pflags);

    MPlexLL<N> temp;
    propdetail::MultHelixPropEndcap(errorProp, outErr, temp);
    propdetail::MultHelixPropTranspEndcap(errorProp, temp, outErr);

    if (pflags.apply_material) {
      MPlexQF<N> hitsRl;
      MPlexQF<N> hitsXi;
      MPlexQF<N> propSign;

      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        if (n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0))) {
          hitsRl(n, 0, 0) = 0.f;
          hitsXi(n, 0, 0) = 0.f;
        } else {
          const float hypo = hipo(outPar(n, 0, 0), outPar(n, 1, 0));
          const auto mat = material_checked(pflags.material, std::abs(msZ(n, 0, 0)), hypo);
          hitsRl(n, 0, 0) = mat.radl;
          hitsXi(n, 0, 0) = mat.bbxi;
        }
        if (n < N_proc) {
          const float zout = msZ.constAt(n, 0, 0);
          const float zin = inPar.constAt(n, 2, 0);
          propSign(n, 0, 0) = (std::abs(zout) > std::abs(zin) ? 1.f : -1.f);
        }
      }
      MPlexHV<N> plNrm;
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        plNrm(n, 0, 0) = 0.f;
        plNrm(n, 1, 0) = 0.f;
        plNrm(n, 2, 0) = 1.f;
      }
      applyMaterialEffects(hitsRl, hitsXi, propSign, plNrm, outErr, outPar, N_proc);
    }

    squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

    for (int i = 0; i < N_proc; ++i) {
      if (outFailFlag(i, 0, 0)) {
        outPar.copySlot(i, inPar);
        outErr.copySlot(i, inErr);
      }
    }
  }

  //============================================================================
  // PropagationMPlexPlane.cc : helix to plane (MkFitCore)
  //============================================================================

  namespace propdetail {

    // getBFieldFromZXY for one slot, with the ES field constants
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float bFieldFromZXY(const PropagationFlags& pf, float z, float x, float y) {
      return Config::bFieldFromZR(pf.material.bField, z, hipo(x, y));
    }

    // StartTrig: trigonometry of a start state, shared by its helix drifts and its Jacobian
    // (sin(theta) from fast_sin for the turning angle, sin/cos of phi and theta from fast_sincos)
    template <idx_t N>
    struct StartTrig {
      float sinTa[N], sinP[N], cosP[N], sinT[N], cosT[N];
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit StartTrig(const MPlexLV<N>& par) {
        MPLEX_SIMD
        for (int n = 0; n < N; ++n) {
          sinTa[n] = vdt::fast_sinf(par.constAt(n, 5, 0));
          vdt::fast_sincosf(par.constAt(n, 4, 0), sinP[n], cosP[n]);
          vdt::fast_sincosf(par.constAt(n, 5, 0), sinT[n], cosT[n]);
        }
      }
    };

    // Per-slot body of parsFromPathL_trig (Matriplex expressions written out for slot n).
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void parsFromPathL_slot(const MPlexLV<N>& __restrict__ inPar,
                                                                MPlexLV<N>& __restrict__ outPar,
                                                                const float kinv,
                                                                const float s,
                                                                const StartTrig<N>& tr,
                                                                const int n) {
      const float alpha = s * tr.sinTa[n] * inPar(n, 3, 0) * kinv;

      float sinah, cosah;
      if constexpr (Config::useTrigApprox) {
        sincos4(0.5f * alpha, sinah, cosah);
      } else {
        vdt::fast_sincosf(0.5f * alpha, sinah, cosah);
      }

      const float sin_mom_phi = tr.sinP[n];
      const float cos_mom_phi = tr.cosP[n];
      const float sin_mom_tht = tr.sinT[n];
      const float cos_mom_tht = tr.cosT[n];

      outPar(n, 0, 0) =
          inPar(n, 0, 0) + 2.f * sinah * (cos_mom_phi * cosah - sin_mom_phi * sinah) / (inPar(n, 3, 0) * kinv);
      outPar(n, 1, 0) =
          inPar(n, 1, 0) + 2.f * sinah * (sin_mom_phi * cosah + cos_mom_phi * sinah) / (inPar(n, 3, 0) * kinv);
      outPar(n, 2, 0) = inPar(n, 2, 0) + alpha / kinv * cos_mom_tht / (inPar(n, 3, 0) * sin_mom_tht);
      outPar(n, 3, 0) = inPar(n, 3, 0);
      outPar(n, 4, 0) = inPar(n, 4, 0) + alpha;
      outPar(n, 5, 0) = inPar(n, 5, 0);
    }

    // parsFromPathL_impl: parsFromPathL_trig with the trigonometry of inPar
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void parsFromPathL_impl(const MPlexLV<N>& __restrict__ inPar,
                                                                MPlexLV<N>& __restrict__ outPar,
                                                                const MPlexQF<N>& kinv,
                                                                const MPlexQF<N>& s) {
      const StartTrig<N> tr(inPar);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n)
        parsFromPathL_slot(inPar, outPar, kinv[n], s[n], tr, n);
    }

    // parsAndErrPropFromPathL_trig: bFld is the field of the step as used for the parameters (kinv)
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void parsAndErrPropFromPathL_trig(const MPlexLV<N>& __restrict__ inPar,
                                                                          const MPlexQI<N>& __restrict__ inChg,
                                                                          MPlexLV<N>& __restrict__ outPar,
                                                                          const MPlexQF<N>& __restrict__ kinv,
                                                                          const MPlexQF<N>& __restrict__ bFld,
                                                                          const MPlexQF<N>& __restrict__ s,
                                                                          MPlexLL<N>& __restrict__ errorProp,
                                                                          const int N_proc,
                                                                          const PropagationFlags& pf,
                                                                          const StartTrig<N>& tr) {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n)
        parsFromPathL_slot(inPar, outPar, kinv[n], s[n], tr, n);

      MPlex55<N> errorPropCurv(0.0f);
      MPlex56<N> jacCCS2Curv(0.0f);
      MPlex65<N> jacCurv2CCS(0.0f);

      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        const float sinPin = tr.sinP[n];
        const float cosPin = tr.cosP[n];
        float sinPout, cosPout;
        vdt::fast_sincosf(outPar(n, 4, 0), sinPout, cosPout);
        const float sinT = tr.sinT[n];
        const float cosT = tr.cosT[n];

        // use code from AnalyticalCurvilinearJacobian::computeFullJacobian for error propagation in curvilinear coordinates, then convert to CCS
        const float qbp = inChg(n, 0, 0) < 0 ? -(sinT * inPar(n, 3, 0)) : (sinT * inPar(n, 3, 0));
        // calculate transport matrix
        // Origin: TRPRFN
        const float t11 = cosPin * sinT;
        const float t12 = sinPin * sinT;
        const float t21 = cosPout * sinT;
        const float t22 = sinPout * sinT;
        const float cosl1 = 1.f / sinT;
        // The field of the step, as used for the parameters (kinv): at the step start, or the chord midpoint
        // with b_field_at_mid
        const float bF = Const::sol_over_100 * bFld[n];
        const float q = -bF * qbp;
        const float theta = q * s[n];
        float sint, cost;
        vdt::fast_sincosf(theta, sint, cost);
        const float dx1 = inPar(n, 0, 0) - outPar(n, 0, 0);
        const float dx2 = inPar(n, 1, 0) - outPar(n, 1, 0);
        const float dx3 = inPar(n, 2, 0) - outPar(n, 2, 0);
        const float u11 = -sinPin;
        const float u12 = cosPin;
        const float v11 = -cosT * u12;
        const float v12 = cosT * u11;
        const float v13 = sinT;
        const float u21 = -sinPout;
        const float u22 = cosPout;
        const float v21 = -cosT * u22;
        const float v22 = cosT * u21;
        const float v23 = sinT;
        // now prepare the transport matrix
        const float omcost = 1.f - cost;
        const float tmsint = theta - sint;

        //   1/p - doesn't change since |p1| = |p2|
        errorPropCurv(n, 0, 0) = 1.f;
        for (int i = 1; i < 5; ++i)
          errorPropCurv(n, 0, i) = 0.f;
        //   lambda
        errorPropCurv(n, 1, 0) = 0.f;
        errorPropCurv(n, 1, 1) =
            cost * (v11 * v21 + v12 * v22 + v13 * v23) + sint * (-v12 * v21 + v11 * v22) + omcost * v13 * v23;
        errorPropCurv(n, 1, 2) = (cost * (u11 * v21 + u12 * v22) + sint * (-u12 * v21 + u11 * v22)) * sinT;
        errorPropCurv(n, 1, 3) = 0.f;
        errorPropCurv(n, 1, 4) = 0.f;
        //   phi
        errorPropCurv(n, 2, 0) = bF * v23 * (t21 * dx1 + t22 * dx2 + cosT * dx3) * cosl1;
        errorPropCurv(n, 2, 1) = (cost * (v11 * u21 + v12 * u22) + sint * (-v12 * u21 + v11 * u22) +
                                  v23 * (-sint * (v11 * t21 + v12 * t22 + v13 * cosT) +
                                         omcost * (-v11 * t22 + v12 * t21) - tmsint * cosT * v13)) *
                                 cosl1;
        errorPropCurv(n, 2, 2) = (cost * (u11 * u21 + u12 * u22) + sint * (-u12 * u21 + u11 * u22) +
                                  v23 * (-sint * (u11 * t21 + u12 * t22) + omcost * (-u11 * t22 + u12 * t21))) *
                                 cosl1 * sinT;
        errorPropCurv(n, 2, 3) = -q * v23 * (u11 * t21 + u12 * t22) * cosl1;
        errorPropCurv(n, 2, 4) = -q * v23 * (v11 * t21 + v12 * t22 + v13 * cosT) * cosl1;

        //   yt
        if (n < N_proc) {
          const float cutCriterion = std::abs(s[n] * sinT * inPar(n, 3, 0));
          const float limit = 5.f;  // valid for propagations with effectively float precision
          if (cutCriterion > limit) {
            const float pp = 1.f / qbp;
            errorPropCurv(n, 3, 0) = pp * (u21 * dx1 + u22 * dx2);
            errorPropCurv(n, 4, 0) = pp * (v21 * dx1 + v22 * dx2 + v23 * dx3);
          } else {
            const float temp1 = -t12 * u21 + t11 * u22;
            const float s2 = s[n] * s[n];
            const float secondOrder41 = -0.5f * bF * temp1 * s2;
            const float temp2 = -t11 * u21 - t12 * u22;
            const float s3 = s2 * s[n];
            const float s4 = s3 * s[n];
            const float h2 = bF * bF;
            const float h3 = h2 * bF;
            const float qbp2 = qbp * qbp;
            const float thirdOrder41 = 1.f / 3 * h2 * s3 * qbp * temp2;
            const float fourthOrder41 = 1.f / 8 * h3 * s4 * qbp2 * temp1;
            errorPropCurv(n, 3, 0) = secondOrder41 + (thirdOrder41 + fourthOrder41);
            const float temp3 = -t12 * v21 + t11 * v22;
            const float secondOrder51 = -0.5f * bF * temp3 * s2;
            const float temp4 = -t11 * v21 - t12 * v22;
            const float thirdOrder51 = 1.f / 3 * h2 * s3 * qbp * temp4;
            const float fourthOrder51 = 1.f / 8 * h3 * s4 * qbp2 * temp3;
            errorPropCurv(n, 4, 0) = secondOrder51 + (thirdOrder51 + fourthOrder51);
          }
        }

        errorPropCurv(n, 3, 1) = (sint * (v11 * u21 + v12 * u22) + omcost * (-v12 * u21 + v11 * u22)) / q;
        errorPropCurv(n, 3, 2) = (sint * (u11 * u21 + u12 * u22) + omcost * (-u12 * u21 + u11 * u22)) * sinT / q;
        errorPropCurv(n, 3, 3) = (u11 * u21 + u12 * u22);
        errorPropCurv(n, 3, 4) = (v11 * u21 + v12 * u22);
        //   zt
        errorPropCurv(n, 4, 1) =
            (sint * (v11 * v21 + v12 * v22 + v13 * v23) + omcost * (-v12 * v21 + v11 * v22) + tmsint * v23 * v13) / q;
        errorPropCurv(n, 4, 2) = (sint * (u11 * v21 + u12 * v22) + omcost * (-u12 * v21 + u11 * v22)) * sinT / q;
        errorPropCurv(n, 4, 3) = (u11 * v21 + u12 * v22);
        errorPropCurv(n, 4, 4) = (v11 * v21 + v12 * v22 + v13 * v23);

        //now we need jacobians to convert to/from curvilinear and CCS
        // code from TrackState::jacobianCCSToCurvilinear
        jacCCS2Curv(n, 0, 3) = inChg(n, 0, 0) < 0 ? -sinT : sinT;
        jacCCS2Curv(n, 0, 5) = inChg(n, 0, 0) < 0 ? -(cosT * inPar(n, 3, 0)) : (cosT * inPar(n, 3, 0));
        jacCCS2Curv(n, 1, 5) = -1.f;
        jacCCS2Curv(n, 2, 4) = 1.f;
        jacCCS2Curv(n, 3, 0) = -sinPin;
        jacCCS2Curv(n, 3, 1) = cosPin;
        jacCCS2Curv(n, 4, 0) = -cosPin * cosT;
        jacCCS2Curv(n, 4, 1) = -sinPin * cosT;
        jacCCS2Curv(n, 4, 2) = sinT;

        // code from TrackState::jacobianCurvilinearToCCS
        jacCurv2CCS(n, 0, 3) = -sinPout;
        jacCurv2CCS(n, 0, 4) = -cosT * cosPout;
        jacCurv2CCS(n, 1, 3) = cosPout;
        jacCurv2CCS(n, 1, 4) = -cosT * sinPout;
        jacCurv2CCS(n, 2, 4) = sinT;
        jacCurv2CCS(n, 3, 0) = inChg(n, 0, 0) < 0 ? -(1.f / sinT) : (1.f / sinT);
        jacCurv2CCS(n, 3, 1) = outPar(n, 3, 0) * cosT / sinT;
        jacCurv2CCS(n, 4, 2) = 1.f;
        jacCurv2CCS(n, 5, 1) = -1.f;
      }

      //need to compute errorProp = jacCurv2CCS*errorPropCurv*jacCCS2Curv
      MPlex65<N> tmp;
      JacErrPropCurv1(jacCurv2CCS, errorPropCurv, tmp);
      JacErrPropCurv2(tmp, jacCCS2Curv, errorProp);
    }

    // from P.Avery's notes (http://www.phys.ufl.edu/~avery/fitting/transport.pdf eq. 5)
    // the root closest to zero in the cancellation-free form (Citardauq), as MkFitCore
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float getS(float delta0,
                                                   float delta1,
                                                   float delta2,
                                                   float eta0,
                                                   float eta1,
                                                   float eta2,
                                                   float sinP,
                                                   float cosP,
                                                   float sinT,
                                                   float cosT,
                                                   float ipt,
                                                   int q,
                                                   float kinv) {
      const float A = delta0 * eta0 + delta1 * eta1 + delta2 * eta2;
      const float p0[3] = {cosP * sinT, sinP * sinT, cosT};
      const float B = (p0[0] * eta0 + p0[1] * eta1 + p0[2] * eta2);
      const float rho = kinv * sinT * ipt;
      const float C = -(eta0 * p0[1] - eta1 * p0[0]) * rho * 0.5f;
      const float s1 = 2.f * A / (-B - std::copysign(std::sqrt(B * B - 4.f * A * C), B));
      return s1;
    }

    // helixAtPlane_impl: want_err = false skips the Jacobian (outPar is the same)
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixAtPlane_impl(const MPlexLV<N>& __restrict__ inPar,
                                                               const MPlexQI<N>& __restrict__ inChg,
                                                               const MPlexHV<N>& __restrict__ plPnt,
                                                               const MPlexHV<N>& __restrict__ plNrm,
                                                               MPlexQF<N>& __restrict__ s,
                                                               MPlexLV<N>& __restrict__ outPar,
                                                               MPlexLL<N>& __restrict__ errorProp,
                                                               MPlexQI<N>& __restrict__ outFailFlag,
                                                               const int N_proc,
                                                               const PropagationFlags& pf,
                                                               const bool want_err = true) {
      MPlexQF<N> kSign, bFld, kinv;
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        kSign[n] = inChg(n, 0, 0) < 0 ? Const::sol_over_100 : -Const::sol_over_100;
        bFld[n] =
            pf.use_param_b_field ? bFieldFromZXY(pf, inPar(n, 2, 0), inPar(n, 0, 0), inPar(n, 1, 0)) : Config::Bfield;
        kinv[n] = kSign[n] * bFld[n];
      }

      // trigonometry of the start state, shared by every drift below and by the Jacobian
      const StartTrig<N> tr(inPar);

      // MkFitCore zero-initialises outParTmp in each step; every element read is overwritten first, so hoisted.
      MPlexLV<N> outParTmp(0.0f);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        float delta0 = inPar(n, 0, 0) - plPnt(n, 0, 0);
        float delta1 = inPar(n, 1, 0) - plPnt(n, 1, 0);
        float delta2 = inPar(n, 2, 0) - plPnt(n, 2, 0);

        float sinP = tr.sinP[n], cosP = tr.cosP[n];
        const float sinT = tr.sinT[n];
        const float cosT = tr.cosT[n];

        // determine solution for straight line
        const float sl = -(plNrm(n, 0, 0) * delta0 + plNrm(n, 1, 0) * delta1 + plNrm(n, 2, 0) * delta2) /
                         (plNrm(n, 0, 0) * cosP * sinT + plNrm(n, 1, 0) * sinP * sinT + plNrm(n, 2, 0) * cosT);

        //first iteration outside the loop
        if (n < N_proc) {
          s[n] = (std::abs(plNrm(n, 2, 0)) < 1.f ? getS(delta0,
                                                        delta1,
                                                        delta2,
                                                        plNrm(n, 0, 0),
                                                        plNrm(n, 1, 0),
                                                        plNrm(n, 2, 0),
                                                        sinP,
                                                        cosP,
                                                        sinT,
                                                        cosT,
                                                        inPar(n, 3, 0),
                                                        inChg(n, 0, 0),
                                                        kinv[n])
                                                 : (plPnt.constAt(n, 2, 0) - inPar.constAt(n, 2, 0)) / cosT);
        }

        for (int i = 0; i < Config::nSStepsInProp2Plane - 1; ++i) {
          parsFromPathL_slot(inPar, outParTmp, kinv[n], s[n], tr, n);

          if (pf.use_param_b_field && pf.b_field_at_mid) {
            // re-sample B at the chord midpoint of the step, 0.5*(start + end), before s is refined
            bFld[n] = bFieldFromZXY(pf,
                                    0.5f * (inPar(n, 2, 0) + outParTmp(n, 2, 0)),
                                    0.5f * (inPar(n, 0, 0) + outParTmp(n, 0, 0)),
                                    0.5f * (inPar(n, 1, 0) + outParTmp(n, 1, 0)));
            kinv[n] = kSign[n] * bFld[n];
          }

          delta0 = outParTmp(n, 0, 0) - plPnt(n, 0, 0);
          delta1 = outParTmp(n, 1, 0) - plPnt(n, 1, 0);
          delta2 = outParTmp(n, 2, 0) - plPnt(n, 2, 0);

          vdt::fast_sincosf(outParTmp(n, 4, 0), sinP, cosP);
          // Note, sinT/cosT not updated

          if (n < N_proc) {
            s[n] += (std::abs(plNrm(n, 2, 0)) < 1.f ? getS(delta0,
                                                           delta1,
                                                           delta2,
                                                           plNrm(n, 0, 0),
                                                           plNrm(n, 1, 0),
                                                           plNrm(n, 2, 0),
                                                           sinP,
                                                           cosP,
                                                           sinT,
                                                           cosT,
                                                           inPar(n, 3, 0),
                                                           inChg(n, 0, 0),
                                                           kinv[n])
                                                    : (plPnt.constAt(n, 2, 0) - outParTmp.constAt(n, 2, 0)) /
                                                          std::cos(outParTmp.constAt(n, 5, 0)));
          }
        }  //end Niter-1

        // use linear approximation if s did not converge (for very high pT tracks)
        if (n < N_proc) {
          if (isFinite(s[n]) == false && isFinite(sl))  // replace with sl even if not fully correct
            s[n] = sl;
        }
      }

      if (want_err) {
        parsAndErrPropFromPathL_trig(inPar, inChg, outPar, kinv, bFld, s, errorProp, N_proc, pf, tr);
      } else {
        MPLEX_SIMD
        for (int n = 0; n < N; ++n)
          parsFromPathL_slot(inPar, outPar, kinv[n], s[n], tr, n);
      }
    }

    // ---- Radial-field correction (PropagationFlags::radial_field_corr; brDeltaRPphiV)
    // D = d(r*p_phi) over one step, from the analytic flux of Bz = Z(z) (a r^2 + 1). Same operations and order.
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float brDeltaRPphi(const MPlexLV<N>& inPar,
                                                           const MPlexLV<N>& outPar,
                                                           const MPlexQI<N>& inChg,
                                                           const Config::BFieldParams& m,
                                                           const int n) {
      const float x0 = inPar.constAt(n, 0, 0), y0 = inPar.constAt(n, 1, 0), z0 = inPar.constAt(n, 2, 0);
      const float x1 = outPar.constAt(n, 0, 0), y1 = outPar.constAt(n, 1, 0), z1 = outPar.constAt(n, 2, 0);
      const float r0sq = x0 * x0 + y0 * y0;
      const float r1sq = x1 * x1 + y1 * y1;
      const float qk = inChg.constAt(n, 0, 0) < 0 ? -Const::sol_over_100 : Const::sol_over_100;
      const float Z0 = (m.b0 * z0 + m.b1) * z0 + m.c1;
      const float Z1 = (m.b0 * z1 + m.b1) * z1 + m.c1;
      const float dz = z0 - z1;
      const float dZ = dz * (m.b0 * (z0 + z1) + m.b1);
      const float zmid = 0.5f * (z0 + z1);
      const float xm = 0.5f * (x0 + x1), ym = 0.5f * (y0 + y1);
      const float rmid2 = xm * xm + ym * ym;
      const float Zmid = (m.b0 * zmid + m.b1) * zmid + m.c1;
      const float Zm = 0.5f * (Z0 + Z1);
      // (Zm - Bc) in factored form: Zm - Zmid = b0 dz^2 / 4
      const float ZmB = 0.25f * m.b0 * dz * dz - Zmid * m.a * rmid2;
      const float D = r0sq - r1sq;
      const float S = r0sq + r1sq;
      const float f0 = 0.25f * m.a * r0sq * r0sq + 0.5f * r0sq;
      const float f1 = 0.25f * m.a * r1sq * r1sq + 0.5f * r1sq;
      return qk * (D * (0.25f * Zm * m.a * S + 0.5f * ZmB) + 0.5f * dZ * (f0 + f1));
    }

    // applyDpPhiV for one slot: rotate p_phi by half_d / r, keeping p_r and |p|; failing guards leave it
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void applyDpPhi(MPlexLV<N>& par, const float half_d, const int n) {
      const float x = par.constAt(n, 0, 0), y = par.constAt(n, 1, 0);
      const float rsq = x * x + y * y;
      const float ipt = par.constAt(n, 3, 0);
      const float rsq_s = (rsq > 1.e-8f) ? rsq : 1.f;
      const float ipt_s = (std::abs(ipt) > 1.e-9f) ? ipt : 1.f;
      float sinP, cosP, sinT, cosT;
      vdt::fast_sincosf(par.constAt(n, 4, 0), sinP, cosP);
      vdt::fast_sincosf(par.constAt(n, 5, 0), sinT, cosT);
      const float sinT_s = (std::abs(sinT) > 1.e-9f) ? sinT : 1.f;
      const float pt0 = 1.f / ipt_s;
      const float pt = ipt_s < 0.f ? -pt0 : pt0;  // Matriplex::negate_if_ltz(1 / ipt_s, ipt_s) = 1/|ipt|
      const float px = pt * cosP, py = pt * sinP;
      const float ptot = pt / sinT_s;
      const float pz = ptot * cosT;
      const float r = std::sqrt(rsq_s);
      const float invr = 1.f / r;
      const float dpphi = half_d / r;
      const float pr = (x * px + y * py) * invr;
      const float pphi = (x * py - y * px) * invr + dpphi;
      const float pt_new = std::sqrt(pr * pr + pphi * pphi);
      const float a2 = ptot * ptot - pt_new * pt_new;
      const float a2_s = (a2 > 0.f) ? a2 : 1.f;
      const float newphi = vdt::fast_atan2f((y * pr + x * pphi) * invr, (x * pr - y * pphi) * invr);
      const float pzn = std::copysign(std::sqrt(a2_s), pz);
      const float newtheta = vdt::fast_atan2f(pt_new, pzn);
      const float newipt = 1.f / pt_new;
      if (!(rsq > 1.e-8f) || !(std::abs(ipt) > 1.e-9f) || !(std::abs(sinT) > 1.e-9f))
        return;
      if (!(a2 > 0.f) || !(pt_new > 1.e-9f))
        return;
      par.At(n, 3, 0) = newipt;
      par.At(n, 4, 0) = newphi;
      par.At(n, 5, 0) = newtheta;
    }

    // helixAtPlaneSel: helixAtPlane with the radial-field half-kicks and the want_err choice
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixAtPlaneSel(const MPlexLV<N>& inPar,
                                                             const MPlexQI<N>& inChg,
                                                             const MPlexHV<N>& plPnt,
                                                             const MPlexHV<N>& plNrm,
                                                             MPlexQF<N>& pathL,
                                                             MPlexLV<N>& outPar,
                                                             MPlexLL<N>& errorProp,
                                                             MPlexQI<N>& outFailFlag,
                                                             const int N_proc,
                                                             const PropagationFlags& pflags,
                                                             const bool want_err) {
      errorProp.setVal(0.f);
      outFailFlag.setVal(0.f);

      if (pflags.use_param_b_field && pflags.radial_field_corr) {
        // D from the uncorrected step, half of it as a kick at the start, the helix step from the kicked state,
        // the other half at the end. The Jacobian is that of the helix step.
        PropagationFlags pf0 = pflags;
        pf0.radial_field_corr = false;

        MPlexLV<N> par0(0.0f);
        MPlexQF<N> pl0(0.0f);
        MPlexLL<N> ep0(0.0f);
        MPlexQI<N> ff0(0);
        helixAtPlane_impl(inPar, inChg, plPnt, plNrm, pl0, par0, ep0, ff0, N_proc, pf0, false);

        MPlexQF<N> halfD;
        MPLEX_SIMD
        for (int n = 0; n < N; ++n)
          halfD[n] = 0.5f * brDeltaRPphi(inPar, par0, inChg, pflags.material.bField, n);
        MPlexLV<N> parH = inPar;
        MPLEX_SIMD
        for (int n = 0; n < N_proc; ++n)
          applyDpPhi(parH, halfD[n], n);
        helixAtPlane_impl(parH, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pf0, want_err);
        MPLEX_SIMD
        for (int n = 0; n < N_proc; ++n)
          applyDpPhi(outPar, halfD[n], n);
        return;
      }

      helixAtPlane_impl(inPar, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pflags, want_err);
    }

    // finishPlanePropagation: 6x6 similarity with errorProp (outErr holds the error to transport), material at
    // the destination (per-module material when matRadl/matBbxi are given; energy-loss sign from the
    // pass with eloss_by_pass; msRefP = pflags.ms_ref_p), phi squash, restore the input on failure.
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void finishPlanePropagation(const MPlexLS<N>& inErr,
                                                                    const MPlexLV<N>& inPar,
                                                                    const MPlexHV<N>& plNrm,
                                                                    const MPlexLL<N>& errorProp,
                                                                    const MPlexQF<N>& pathL,
                                                                    MPlexLS<N>& outErr,
                                                                    MPlexLV<N>& outPar,
                                                                    MPlexQI<N>& outFailFlag,
                                                                    const int N_proc,
                                                                    const PropagationFlags& pflags,
                                                                    const MPlexQI<N>* noMatEffPtr,
                                                                    const MPlexQF<N>* matRadl,
                                                                    const MPlexQF<N>* matBbxi,
                                                                    const MPlexQF<N>* msRefP) {
      // Matriplex version of:
      // result.errors = ROOT::Math::Similarity(errorProp, outErr);
      MPlexLL<N> temp(0.0f);
      MultHelixPlaneProp(errorProp, outErr, temp);
      MultHelixPlanePropTransp(errorProp, temp, outErr);

      if (pflags.apply_material) {
        MPlexQF<N> hitsRl;
        MPlexQF<N> hitsXi;
        MPlexQF<N> propSign;

        // the crossed module's own material instead of the (|z|,r) grid (Config::refitMaterialPerModule:
        // the device fit passes matRadl/matBbxi only when the ES switch is on)
        const bool use_mod_mat = matRadl && matBbxi;
        const float passSign = pflags.eloss_outward ? 1.f : -1.f;
        const bool by_pass = pflags.eloss_by_pass;

        MPLEX_SIMD
        for (int n = 0; n < N; ++n) {
          if (n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0))) {
            hitsRl(n, 0, 0) = 0.f;
            hitsXi(n, 0, 0) = 0.f;
            propSign(n, 0, 0) = -1.f;
          } else {
            if (use_mod_mat) {
              hitsRl(n, 0, 0) = matRadl->constAt(n, 0, 0);
              hitsXi(n, 0, 0) = matBbxi->constAt(n, 0, 0);
            } else {
              const float hypo = hipo(outPar(n, 0, 0), outPar(n, 1, 0));
              const auto mat = material_checked(pflags.material, std::abs(outPar(n, 2, 0)), hypo);
              hitsRl(n, 0, 0) = mat.radl;
              hitsXi(n, 0, 0) = mat.bbxi;
            }
            propSign(n, 0, 0) = by_pass ? passSign : (pathL(n, 0, 0) > 0.f ? 1.f : -1.f);
          }
        }
        applyMaterialEffects(hitsRl, hitsXi, propSign, plNrm, outErr, outPar, N_proc, msRefP);
      }

      squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

      // PROP-FAIL-ENABLE To keep physics changes minimal, we always restore the
      // state to input when propagation fails -- as was the default before.
      for (int i = 0; i < N_proc; ++i) {
        if (outFailFlag(i, 0, 0)) {
          outPar.copySlot(i, inPar);
          outErr.copySlot(i, inErr);
        }
      }
    }

  }  // namespace propdetail

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void helixAtPlane(const MPlexLV<N>& inPar,
                                                        const MPlexQI<N>& inChg,
                                                        const MPlexHV<N>& plPnt,
                                                        const MPlexHV<N>& plNrm,
                                                        MPlexQF<N>& pathL,
                                                        MPlexLV<N>& outPar,
                                                        MPlexLL<N>& errorProp,
                                                        MPlexQI<N>& outFailFlag,
                                                        const int N_proc,
                                                        const PropagationFlags& pflags) {
    propdetail::helixAtPlaneSel(
        inPar, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pflags, true);
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void propagateHelixToPlaneMPlex(const MPlexLS<N>& inErr,
                                                                      const MPlexLV<N>& inPar,
                                                                      const MPlexQI<N>& inChg,
                                                                      const MPlexHV<N>& plPnt,
                                                                      const MPlexHV<N>& plNrm,
                                                                      MPlexLS<N>& outErr,
                                                                      MPlexLV<N>& outPar,
                                                                      MPlexQI<N>& outFailFlag,
                                                                      const int N_proc,
                                                                      const PropagationFlags& pflags,
                                                                      const MPlexQI<N>* noMatEffPtr = nullptr,
                                                                      const MPlexQF<N>* matRadl = nullptr,
                                                                      const MPlexQF<N>* matBbxi = nullptr,
                                                                      const MPlexQF<N>* msRefP = nullptr) {
    outErr = inErr;
    outPar = inPar;

    MPlexQF<N> pathL(0.0f);
    MPlexLL<N> errorProp(0.0f);

    helixAtPlane(inPar, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pflags);

    propdetail::finishPlanePropagation(inErr,
                                       inPar,
                                       plNrm,
                                       errorProp,
                                       pathL,
                                       outErr,
                                       outPar,
                                       outFailFlag,
                                       N_proc,
                                       pflags,
                                       noMatEffPtr,
                                       matRadl,
                                       matBbxi,
                                       msRefP);
  }

  //============================================================================
  // Sub-stepped propagation to a plane (refit backward pass, Config::refitBkwSubSteps)
  //============================================================================

  namespace propdetail {

    // firstPathEstimate: the starting value of helixAtPlane_impl's solve (exact in z for disks), straight
    // line if that is not finite
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void firstPathEstimate(const MPlexLV<N>& inPar,
                                                               const MPlexQI<N>& inChg,
                                                               const MPlexHV<N>& plPnt,
                                                               const MPlexHV<N>& plNrm,
                                                               const int N_proc,
                                                               const PropagationFlags& pf,
                                                               const StartTrig<N>& tr,
                                                               MPlexQF<N>& s0) {
      MPLEX_SIMD
      for (int n = 0; n < N; ++n) {
        s0[n] = 0.f;
        if (n >= N_proc)
          continue;
        const float kSign = inChg(n, 0, 0) < 0 ? Const::sol_over_100 : -Const::sol_over_100;
        const float bFld =
            pf.use_param_b_field ? bFieldFromZXY(pf, inPar(n, 2, 0), inPar(n, 0, 0), inPar(n, 1, 0)) : Config::Bfield;
        const float kinv = kSign * bFld;
        const float d0 = inPar.constAt(n, 0, 0) - plPnt.constAt(n, 0, 0);
        const float d1 = inPar.constAt(n, 1, 0) - plPnt.constAt(n, 1, 0);
        const float d2 = inPar.constAt(n, 2, 0) - plPnt.constAt(n, 2, 0);
        const float e0 = plNrm.constAt(n, 0, 0), e1 = plNrm.constAt(n, 1, 0), e2 = plNrm.constAt(n, 2, 0);
        float s;
        if (std::abs(e2) < 1.f)
          s = getS(d0,
                   d1,
                   d2,
                   e0,
                   e1,
                   e2,
                   tr.sinP[n],
                   tr.cosP[n],
                   tr.sinT[n],
                   tr.cosT[n],
                   inPar.constAt(n, 3, 0),
                   inChg.constAt(n, 0, 0),
                   kinv);
        else
          s = (plPnt.constAt(n, 2, 0) - inPar.constAt(n, 2, 0)) / tr.cosT[n];
        if (!isFinite(s))
          s = -(e0 * d0 + e1 * d1 + e2 * d2) /
              (e0 * tr.cosP[n] * tr.sinT[n] + e1 * tr.sinP[n] * tr.sinT[n] + e2 * tr.cosT[n]);
        s0[n] = isFinite(s) ? s : 0.f;
      }
    }

    // fixedLengthSubStep: one fixed-length sub-step of path length h (per slot; 0 = no move), parameters only,
    // with the same field model as a propagation to a plane
    template <idx_t N>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void fixedLengthSubStep(MPlexLV<N>& par,
                                                                const MPlexQI<N>& inChg,
                                                                const MPlexQF<N>& kSign,
                                                                const MPlexQF<N>& h,
                                                                const int N_proc,
                                                                const PropagationFlags& pf) {
      const MPlexLV<N> p0 = par;
      MPlexQF<N> kinv;
      MPLEX_SIMD
      for (int n = 0; n < N; ++n)
        kinv[n] = kSign[n] *
                  (pf.use_param_b_field ? bFieldFromZXY(pf, p0(n, 2, 0), p0(n, 0, 0), p0(n, 1, 0)) : Config::Bfield);
      MPlexLV<N> p1(0.0f);
      const StartTrig<N> tr0(p0);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n)
        parsFromPathL_slot(p0, p1, kinv[n], h[n], tr0, n);
      if (pf.use_param_b_field && pf.b_field_at_mid) {
        MPLEX_SIMD
        for (int n = 0; n < N; ++n) {
          kinv[n] = kSign[n] * bFieldFromZXY(pf,
                                             0.5f * (p0(n, 2, 0) + p1(n, 2, 0)),
                                             0.5f * (p0(n, 0, 0) + p1(n, 0, 0)),
                                             0.5f * (p0(n, 1, 0) + p1(n, 1, 0)));
          parsFromPathL_slot(p0, p1, kinv[n], h[n], tr0, n);
        }
      }
      if (pf.use_param_b_field && pf.radial_field_corr) {
        MPlexQF<N> dvec;
        MPLEX_SIMD
        for (int n = 0; n < N; ++n)
          dvec[n] = brDeltaRPphi(p0, p1, inChg, pf.material.bField, n);
        MPlexLV<N> ph = p0;
        MPLEX_SIMD
        for (int n = 0; n < N_proc; ++n)
          applyDpPhi(ph, 0.5f * dvec[n], n);
        parsFromPathL_impl(ph, p1, kinv, h);
        MPLEX_SIMD
        for (int n = 0; n < N_proc; ++n)
          applyDpPhi(p1, 0.5f * dvec[n], n);
      }
      squashPhiMPlex(p1, N_proc);
      par = p1;
    }

  }  // namespace propdetail

  // propagateHelixToPlaneSubStepMPlex: nSub sub-steps (parameters only), covariance with the whole-step
  // Jacobian, material at the destination. split[n] = false keeps slot n as one step.
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void propagateHelixToPlaneSubStepMPlex(const MPlexLS<N>& inErr,
                                                                             const MPlexLV<N>& inPar,
                                                                             const MPlexQI<N>& inChg,
                                                                             const MPlexHV<N>& plPnt,
                                                                             const MPlexHV<N>& plNrm,
                                                                             MPlexLS<N>& outErr,
                                                                             MPlexLV<N>& outPar,
                                                                             MPlexQI<N>& outFailFlag,
                                                                             const int N_proc,
                                                                             const PropagationFlags& pflags,
                                                                             const int nSub,
                                                                             const bool (&split)[N],
                                                                             const MPlexQI<N>* noMatEffPtr,
                                                                             const MPlexQF<N>* matRadl,
                                                                             const MPlexQF<N>* matBbxi,
                                                                             const MPlexQF<N>* msRefP) {
    using namespace propdetail;
    PropagationFlags pfs = pflags;
    pfs.apply_material = false;  // intermediate sub-steps: no material (it is applied at the destination)
    outFailFlag.setVal(0);
    MPlexQI<N> ff(0);
    MPlexLL<N> epDummy(0.0f);
    MPlexQF<N> pl(0.0f), sTot(0.0f);
    MPlexLV<N> par = inPar;
    MPlexQF<N> kSign;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n)
      kSign[n] = inChg(n, 0, 0) < 0 ? Const::sol_over_100 : -Const::sol_over_100;

    const StartTrig<N> trIn(inPar);
    MPlexQF<N> h;
    firstPathEstimate(par, inChg, plPnt, plNrm, N_proc, pflags, trIn, h);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n)
      h[n] = (n < N_proc && split[n]) ? h[n] / nSub : 0.f;
    for (int ks = 1; ks < nSub; ++ks) {
      fixedLengthSubStep(par, inChg, kSign, h, N_proc, pfs);
      MPLEX_SIMD
      for (int n = 0; n < N; ++n)
        sTot[n] = sTot[n] + h[n];
    }

    // last sub-step onto the destination plane, parameters only
    helixAtPlaneSel(par, inChg, plPnt, plNrm, pl, outPar, epDummy, ff, N_proc, pfs, false);
    MPLEX_SIMD
    for (int n = 0; n < N_proc; ++n)
      if (ff.constAt(n, 0, 0))
        outFailFlag.At(n, 0, 0) = 1;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n)
      sTot[n] = sTot[n] + pl[n];

    // whole-step Jacobian: from the step start over the accumulated path length, with the field of the step
    MPlexQF<N> bW, kW;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      bW[n] = !pflags.use_param_b_field ? Config::Bfield
              : pflags.b_field_at_mid   ? bFieldFromZXY(pflags,
                                                      0.5f * (inPar(n, 2, 0) + outPar(n, 2, 0)),
                                                      0.5f * (inPar(n, 0, 0) + outPar(n, 0, 0)),
                                                      0.5f * (inPar(n, 1, 0) + outPar(n, 1, 0)))
                                        : bFieldFromZXY(pflags, inPar(n, 2, 0), inPar(n, 0, 0), inPar(n, 1, 0));
      kW[n] = kSign[n] * bW[n];
    }
    MPlexLV<N> parJ(0.0f);
    MPlexLL<N> errorProp(0.0f);
    parsAndErrPropFromPathL_trig(inPar, inChg, parJ, kW, bW, sTot, errorProp, N_proc, pflags, trIn);

    outErr = inErr;
    finishPlanePropagation(inErr,
                           inPar,
                           plNrm,
                           errorProp,
                           sTot,
                           outErr,
                           outPar,
                           outFailFlag,
                           N_proc,
                           pflags,
                           noMatEffPtr,
                           matRadl,
                           matBbxi,
                           msRefP);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
