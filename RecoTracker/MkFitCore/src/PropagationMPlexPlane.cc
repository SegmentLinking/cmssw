#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"

#include "PropagationMPlex.h"

#include <vdt/atan2.h>
#include <cstring>

//#define DEBUG
#include "Debug.h"

namespace {
  using namespace mkfit;

  void MultHelixPlaneProp(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
    // C = A * B

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPlaneProp.ah"
  }

  void MultHelixPlanePropTransp(const MPlexLL& A, const MPlexLL& B, MPlexLS& C) {
    // C = B * AT;

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPlanePropTransp.ah"
  }

}  // namespace

// ============================================================================
// BEGIN STUFF FROM PropagationMPlex.icc
namespace {

  using MPF = MPlexQF;

  // ---- vdt::fast_atan2f over a whole Matriplex, vectorised.
  // The library is compiled without -ffast-math, so GCC keeps -ftrapping-math and will not if-convert the branches
  // inside vdt::fast_atan2f: Matriplex::fast_atan2 runs lane by lane ("not vectorized: control flow in loop").  Here
  // the same operations, in the same order, are written with GCC vector extensions, every branch an explicit blend of
  // values that are all computed.  Bit-identical to Matriplex::fast_atan2 on 1e9 inputs (log-uniform 1e-6..1e3, both
  // signs, zeros, y ~ x), 5.8x its throughput (31.8 -> 5.5 ns per 8 lanes).
  typedef float vfloat __attribute__((vector_size(NN * sizeof(float))));
  typedef int vint __attribute__((vector_size(NN * sizeof(int))));

  MPF atan2V(const MPF& Y, const MPF& X) {
    vfloat y, x;
    std::memcpy(&y, Y.fArray, sizeof(y));
    std::memcpy(&x, X.fArray, sizeof(x));
    const vfloat zero = vfloat{} + 0.f, one = vfloat{} + 1.f;
    const vfloat ax = (vfloat)((vint)x & 0x7fffffff), ay = (vfloat)((vint)y & 0x7fffffff);
    const vint swp = ay > ax;
    const vfloat xx = swp ? ay : ax;
    const vfloat yy = swp ? ax : ay;
    const vfloat oneIfXXZero = (xx == zero) ? one : zero;
    const vfloat t = yy / xx;
    const vint red = t > 0.4142135623730950f;
    const vfloat z = red ? (t - 1.0f) / (t + 1.0f) : t;
    const vfloat z2 = z * z;
    vfloat ret =
        ((((8.05374449538e-2f * z2 - 1.38776856032E-1f) * z2 + 1.99777106478E-1f) * z2 - 3.33329491539E-1f) * z2 * z +
         z);
    ret *= (1.f - oneIfXXZero);
    ret = (y == zero) ? zero : ret;
    ret = red ? ret + vdt::details::PIO4F : ret;
    ret = swp ? vdt::details::PIO2F - ret : ret;
    ret = (x < zero) ? vdt::details::PIF - ret : ret;
    ret = (y < zero) ? -ret : ret;
    MPF R;
    std::memcpy(R.fArray, &ret, sizeof(ret));
    return R;
  }

  MPF getBFieldFromZXY(const MPF& z, const MPF& x, const MPF& y) {
    MPF b;
    for (int n = 0; n < NN; ++n)
      b[n] = Config::bFieldFromZR(z[n], hipo(x[n], y[n]));
    return b;
  }

  void JacErrPropCurv1(const MPlex65& A, const MPlex55& B, MPlex65& C) {
    // C = A * B
    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "JacErrPropCurv1.ah"
  }

  void JacErrPropCurv2(const MPlex65& A, const MPlex56& B, MPlexLL& __restrict__ C) {
    // C = A * B
    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "JacErrPropCurv2.ah"
  }

  // ---- Radial-field correction (PropagationFlags::radial_field_corr) ----------------------------------
  //
  // The propagation models B as purely along z and constant over the step.  The real solenoid field is
  // axially symmetric with Br = -(r/2) dBz/dz (div B = 0), and such a field conserves the canonical
  // angular momentum
  //     L_z = r * p_phi + q * k * Psi(r,z) / (2 pi)
  // with Psi the flux enclosed at (r,z).  The constant-Bz helix conserves the same quantity with the
  // uniform flux q*k*Bc*r^2/2, so over a step it misses a change D of r*p_phi.  For the parametrised
  // field Bz = Z(z) (a r^2 + 1), Z = b0 z^2 + b1 z + c1, the flux is analytic,
  //     G = q*k*Z(z)*(a r^4/4 + r^2/2),
  // so no extra field lookup and no new constant is needed.  p_r is untouched and |p| is conserved, so
  // pT and theta both move -- the degree of freedom the constant-Bz helix freezes.
  //
  // D is exactly antisymmetric under swapping the step's endpoints.  It is applied as two half-kicks,
  // D/(2 r0) before the helix step and D/(2 r1) after it, each converted with the radius where it is
  // applied: reversing the step sends D -> -D and swaps r0 <-> r1, so the step stays its own inverse
  // (a one-sided D/r1 is not, and makes the two refit passes disagree).

  // D = d(r*p_phi) over one step for every lane, B field midpoint Bc = Zmid (a rmid^2 + 1).
  //   D/qk = D2*[ Zm*a*S/4 + (Zm - Bc)/2 ] + dZ*(f0 + f1)/2,  D2 = r0^2 - r1^2,  S = r0^2 + r1^2,
  //   f = a r^4/4 + r^2/2,  Zm = (Z0 + Z1)/2,  dZ = Z0 - Z1.
  // D2 and dZ flip sign under the swap and nothing else does.  Every difference is written in factored
  // form so that no two large, nearly equal numbers are subtracted in float.
  MPlexQF brDeltaRPphiV(const MPlexLV& inPar, const MPlexLV& outPar, const MPlexQI& inChg) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;
    const MPF x0 = inPar(0, 0), y0 = inPar(1, 0), z0 = inPar(2, 0);
    const MPF x1 = outPar(0, 0), y1 = outPar(1, 0), z1 = outPar(2, 0);
    const MPF r0sq = x0 * x0 + y0 * y0;
    const MPF r1sq = x1 * x1 + y1 * y1;
    // q * sol_over_100, without materialising q: +sol where the charge is positive
    const MPF qk = mpt::negate_if_ltz(MPF(Const::sol_over_100), inChg);
    const MPF Z0 = (MPF(Config::mag_b0) * z0 + MPF(Config::mag_b1)) * z0 + MPF(Config::mag_c1);
    const MPF Z1 = (MPF(Config::mag_b0) * z1 + MPF(Config::mag_b1)) * z1 + MPF(Config::mag_c1);
    const MPF dz = z0 - z1;
    const MPF dZ = dz * (MPF(Config::mag_b0) * (z0 + z1) + MPF(Config::mag_b1));
    const MPF zmid = 0.5f * (z0 + z1);
    const MPF xm = 0.5f * (x0 + x1), ym = 0.5f * (y0 + y1);
    const MPF rmid2 = xm * xm + ym * ym;
    const MPF Zmid = (MPF(Config::mag_b0) * zmid + MPF(Config::mag_b1)) * zmid + MPF(Config::mag_c1);
    const MPF Zm = 0.5f * (Z0 + Z1);
    // (Zm - Bc) in factored form: Zm - Zmid = b0 dz^2 / 4
    const MPF ZmB = 0.25f * MPF(Config::mag_b0) * dz * dz - Zmid * MPF(Config::mag_a) * rmid2;
    const MPF D = r0sq - r1sq;
    const MPF S = r0sq + r1sq;
    const MPF f0 = 0.25f * MPF(Config::mag_a) * r0sq * r0sq + 0.5f * r0sq;
    const MPF f1 = 0.25f * MPF(Config::mag_a) * r1sq * r1sq + 0.5f * r1sq;
    return qk * (D * (0.25f * Zm * MPF(Config::mag_a) * S + 0.5f * ZmB) + 0.5f * dZ * (f0 + f1));
  }

  // Rotate p_phi by a kick half_d / r at every lane's own position r, keeping p_r and |p|.  Vectorised
  // (vdt sincos via Matriplex, atan2V); lanes whose guards fail are left untouched.
  void applyDpPhiV(MPlexLV& par, const MPlexQF& half_d, const int N_proc) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;
    const MPF x = par(0, 0), y = par(1, 0);
    const MPF rsq = x * x + y * y;
    const MPF ipt = par(3, 0);

    // Clamp the guarded quantities so every lane can be evaluated; failing lanes are
    // discarded at write-back, exactly as the scalar version leaves them untouched.
    MPF rsq_s = rsq, ipt_s = ipt;
    for (int n = 0; n < NN; ++n) {
      if (!(rsq_s[n] > 1.e-8f))
        rsq_s[n] = 1.f;
      if (!(std::abs(ipt_s[n]) > 1.e-9f))
        ipt_s[n] = 1.f;
    }

    MPF sinP, cosP, sinT, cosT;
    mpt::fast_sincos(par(4, 0), sinP, cosP);
    mpt::fast_sincos(par(5, 0), sinT, cosT);
    MPF sinT_s = sinT;
    for (int n = 0; n < NN; ++n)
      if (!(std::abs(sinT_s[n]) > 1.e-9f))
        sinT_s[n] = 1.f;

    const MPF pt = mpt::negate_if_ltz(MPF(1.f) / ipt_s, ipt_s);  // = 1/|ipt|
    const MPF px = pt * cosP, py = pt * sinP;
    const MPF ptot = pt / sinT_s;
    const MPF pz = ptot * cosT;
    const MPF r = mpt::sqrt(rsq_s);
    const MPF invr = MPF(1.f) / r;
    const MPF dpphi = half_d / r;
    const MPF pr = (x * px + y * py) * invr;
    const MPF pphi = (x * py - y * px) * invr + dpphi;
    const MPF pt_new = mpt::sqrt(pr * pr + pphi * pphi);
    const MPF a2 = ptot * ptot - pt_new * pt_new;

    MPF a2_s = a2;
    for (int n = 0; n < NN; ++n)
      if (!(a2_s[n] > 0.f))
        a2_s[n] = 1.f;
    const MPF newphi = atan2V((y * pr + x * pphi) * invr, (x * pr - y * pphi) * invr);
    MPF pzn = mpt::sqrt(a2_s);
    for (int n = 0; n < NN; ++n)
      pzn[n] = std::copysign(pzn[n], pz[n]);
    const MPF newtheta = atan2V(pt_new, pzn);
    const MPF newipt = MPF(1.f) / pt_new;

    for (int n = 0; n < N_proc; ++n) {
      if (!(rsq[n] > 1.e-8f) || !(std::abs(ipt[n]) > 1.e-9f) || !(std::abs(sinT[n]) > 1.e-9f))
        continue;
      if (!(a2[n] > 0.f) || !(pt_new[n] > 1.e-9f))
        continue;
      par.At(n, 3, 0) = newipt[n];
      par.At(n, 4, 0) = newphi[n];
      par.At(n, 5, 0) = newtheta[n];
    }
  }

  // Trigonometry of a start state, computed once per state and shared by its helix drifts and its Jacobian:
  // sin(theta) from fast_sin for the turning angle (as before), sin/cos of phi and theta from fast_sincos.
  struct StartTrig {
    MPlexQF sinTa, sinP, cosP, sinT, cosT;
    explicit StartTrig(const MPlexLV& par) {
      sinTa = Matriplex::fast_sin(par(5, 0));
      Matriplex::fast_sincos(par(4, 0), sinP, cosP);
      Matriplex::fast_sincos(par(5, 0), sinT, cosT);
    }
  };

  void parsFromPathL_trig(const MPlexLV& __restrict__ inPar,
                          MPlexLV& __restrict__ outPar,
                          const MPlexQF& __restrict__ kinv,
                          const MPlexQF& __restrict__ s,
                          const StartTrig& tr) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;

    const MPF alpha = s * tr.sinTa * inPar(3, 0) * kinv;

    MPF sinah, cosah;
    if constexpr (Config::useTrigApprox) {
      mpt::sincos4(0.5f * alpha, sinah, cosah);
    } else {
      mpt::fast_sincos(0.5f * alpha, sinah, cosah);
    }

    const MPF& sin_mom_phi = tr.sinP;
    const MPF& cos_mom_phi = tr.cosP;
    const MPF& sin_mom_tht = tr.sinT;
    const MPF& cos_mom_tht = tr.cosT;

    outPar.aij(0, 0) = inPar(0, 0) + 2.f * sinah * (cos_mom_phi * cosah - sin_mom_phi * sinah) / (inPar(3, 0) * kinv);
    outPar.aij(1, 0) = inPar(1, 0) + 2.f * sinah * (sin_mom_phi * cosah + cos_mom_phi * sinah) / (inPar(3, 0) * kinv);
    outPar.aij(2, 0) = inPar(2, 0) + alpha / kinv * cos_mom_tht / (inPar(3, 0) * sin_mom_tht);
    outPar.aij(3, 0) = inPar(3, 0);
    outPar.aij(4, 0) = inPar(4, 0) + alpha;
    outPar.aij(5, 0) = inPar(5, 0);
  }

  void parsFromPathL_impl(const MPlexLV& __restrict__ inPar,
                          MPlexLV& __restrict__ outPar,
                          const MPlexQF& __restrict__ kinv,
                          const MPlexQF& __restrict__ s) {
    parsFromPathL_trig(inPar, outPar, kinv, s, StartTrig(inPar));
  }

  //*****************************************************************************************************

  //should kinv and D be templated???
  void parsAndErrPropFromPathL_trig(const MPlexLV& __restrict__ inPar,
                                    const MPlexQI& __restrict__ inChg,
                                    MPlexLV& __restrict__ outPar,
                                    const MPlexQF& __restrict__ kinv,
                                    const MPlexQF& __restrict__ bFld,
                                    const MPlexQF& __restrict__ s,
                                    MPlexLL& __restrict__ errorProp,
                                    const int N_proc,
                                    const PropagationFlags& pf,
                                    const StartTrig& tr) {
    //iteration should return the path length s, then update parameters and compute errors

    namespace mpt = Matriplex;
    using MPF = MPlexQF;

    parsFromPathL_trig(inPar, outPar, kinv, s, tr);

    const MPF& sinPin = tr.sinP;
    const MPF& cosPin = tr.cosP;
    MPF sinPout, cosPout;
    mpt::fast_sincos(outPar(4, 0), sinPout, cosPout);
    const MPF& sinT = tr.sinT;
    const MPF& cosT = tr.cosT;

    // use code from AnalyticalCurvilinearJacobian::computeFullJacobian for error propagation in curvilinear coordinates, then convert to CCS
    // main difference from the above function is that we assume that the magnetic field is purely along z (which also implies that there is no change in pz)
    // this simplifies significantly the code
    const MPF qbp = mpt::negate_if_ltz(sinT * inPar(3, 0), inChg);
    // calculate transport matrix
    // Origin: TRPRFN
    const MPF t11 = cosPin * sinT;
    const MPF t12 = sinPin * sinT;
    const MPF t21 = cosPout * sinT;
    const MPF t22 = sinPout * sinT;
    const MPF cosl1 = 1.f / sinT;
    // The field of the step, as used for the parameters (kinv).  Passed in rather than re-sampled here
    // so that the Jacobian linearises the step the parameters actually took: at the step start as in
    // the original code, or the average over the step as in TRPRFN with PropagationFlags::b_field_at_mid.
    const MPF bF = Const::sol_over_100 * bFld;
    const MPF q = -bF * qbp;
    const MPF theta = q * s;
    MPF sint, cost;
    mpt::fast_sincos(theta, sint, cost);
    const MPF dx1 = inPar(0, 0) - outPar(0, 0);
    const MPF dx2 = inPar(1, 0) - outPar(1, 0);
    const MPF dx3 = inPar(2, 0) - outPar(2, 0);
    const MPF u11 = -sinPin;
    const MPF u12 = cosPin;
    const MPF v11 = -cosT * u12;
    const MPF v12 = cosT * u11;
    const MPF v13 = sinT;
    const MPF u21 = -sinPout;
    const MPF u22 = cosPout;
    const MPF v21 = -cosT * u22;
    const MPF v22 = cosT * u21;
    const MPF v23 = sinT;
    // now prepare the transport matrix
    const MPF omcost = 1.f - cost;
    const MPF tmsint = theta - sint;

    MPlex55 errorPropCurv{0.0f};
    //   1/p - doesn't change since |p1| = |p2|
    errorPropCurv.aij(0, 0) = 1.f;
    for (int i = 1; i < 5; ++i)
      errorPropCurv.aij(0, i) = 0.f;
    //   lambda
    errorPropCurv.aij(1, 0) = 0.f;
    errorPropCurv.aij(1, 1) =
        cost * (v11 * v21 + v12 * v22 + v13 * v23) + sint * (-v12 * v21 + v11 * v22) + omcost * v13 * v23;
    errorPropCurv.aij(1, 2) = (cost * (u11 * v21 + u12 * v22) + sint * (-u12 * v21 + u11 * v22)) * sinT;
    errorPropCurv.aij(1, 3) = 0.f;
    errorPropCurv.aij(1, 4) = 0.f;
    //   phi
    errorPropCurv.aij(2, 0) = bF * v23 * (t21 * dx1 + t22 * dx2 + cosT * dx3) * cosl1;
    errorPropCurv.aij(2, 1) = (cost * (v11 * u21 + v12 * u22) + sint * (-v12 * u21 + v11 * u22) +
                               v23 * (-sint * (v11 * t21 + v12 * t22 + v13 * cosT) + omcost * (-v11 * t22 + v12 * t21) -
                                      tmsint * cosT * v13)) *
                              cosl1;
    errorPropCurv.aij(2, 2) = (cost * (u11 * u21 + u12 * u22) + sint * (-u12 * u21 + u11 * u22) +
                               v23 * (-sint * (u11 * t21 + u12 * t22) + omcost * (-u11 * t22 + u12 * t21))) *
                              cosl1 * sinT;
    errorPropCurv.aij(2, 3) = -q * v23 * (u11 * t21 + u12 * t22) * cosl1;
    errorPropCurv.aij(2, 4) = -q * v23 * (v11 * t21 + v12 * t22 + v13 * cosT) * cosl1;

    //   yt
    for (int n = 0; n < N_proc; ++n) {
      const float cutCriterion = std::abs(s[n] * sinT[n] * inPar(n, 3, 0));
      const float limit = 5.f;  // valid for propagations with effectively float precision
      if (cutCriterion > limit) {
        const float pp = 1.f / qbp[n];
        errorPropCurv(n, 3, 0) = pp * (u21[n] * dx1[n] + u22[n] * dx2[n]);
        errorPropCurv(n, 4, 0) = pp * (v21[n] * dx1[n] + v22[n] * dx2[n] + v23[n] * dx3[n]);
      } else {
        const float temp1 = -t12[n] * u21[n] + t11[n] * u22[n];
        const float s2 = s[n] * s[n];
        const float secondOrder41 = -0.5f * bF[n] * temp1 * s2;
        const float temp2 = -t11[n] * u21[n] - t12[n] * u22[n];
        const float s3 = s2 * s[n];
        const float s4 = s3 * s[n];
        const float h2 = bF[n] * bF[n];
        const float h3 = h2 * bF[n];
        const float qbp2 = qbp[n] * qbp[n];
        const float thirdOrder41 = 1.f / 3 * h2 * s3 * qbp[n] * temp2;
        const float fourthOrder41 = 1.f / 8 * h3 * s4 * qbp2 * temp1;
        errorPropCurv(n, 3, 0) = secondOrder41 + (thirdOrder41 + fourthOrder41);
        const float temp3 = -t12[n] * v21[n] + t11[n] * v22[n];
        const float secondOrder51 = -0.5f * bF[n] * temp3 * s2;
        const float temp4 = -t11[n] * v21[n] - t12[n] * v22[n];
        const float thirdOrder51 = 1.f / 3 * h2 * s3 * qbp[n] * temp4;
        const float fourthOrder51 = 1.f / 8 * h3 * s4 * qbp2 * temp3;
        errorPropCurv(n, 4, 0) = secondOrder51 + (thirdOrder51 + fourthOrder51);
      }
    }

    errorPropCurv.aij(3, 1) = (sint * (v11 * u21 + v12 * u22) + omcost * (-v12 * u21 + v11 * u22)) / q;
    errorPropCurv.aij(3, 2) = (sint * (u11 * u21 + u12 * u22) + omcost * (-u12 * u21 + u11 * u22)) * sinT / q;
    errorPropCurv.aij(3, 3) = (u11 * u21 + u12 * u22);
    errorPropCurv.aij(3, 4) = (v11 * u21 + v12 * u22);
    //   zt
    errorPropCurv.aij(4, 1) =
        (sint * (v11 * v21 + v12 * v22 + v13 * v23) + omcost * (-v12 * v21 + v11 * v22) + tmsint * v23 * v13) / q;
    errorPropCurv.aij(4, 2) = (sint * (u11 * v21 + u12 * v22) + omcost * (-u12 * v21 + u11 * v22)) * sinT / q;
    errorPropCurv.aij(4, 3) = (u11 * v21 + u12 * v22);
    errorPropCurv.aij(4, 4) = (v11 * v21 + v12 * v22 + v13 * v23);

//debug = true;
#ifdef DEBUG
    for (int n = 0; n < NN; ++n) {
      if (debug && g_debug && n < N_proc) {
        dmutex_guard;
        std::cout << n << ": errorPropCurv" << std::endl;
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 0, 0),
               errorPropCurv(n, 0, 1),
               errorPropCurv(n, 0, 2),
               errorPropCurv(n, 0, 3),
               errorPropCurv(n, 0, 4));
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 1, 0),
               errorPropCurv(n, 1, 1),
               errorPropCurv(n, 1, 2),
               errorPropCurv(n, 1, 3),
               errorPropCurv(n, 1, 4));
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 2, 0),
               errorPropCurv(n, 2, 1),
               errorPropCurv(n, 2, 2),
               errorPropCurv(n, 2, 3),
               errorPropCurv(n, 2, 4));
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 3, 0),
               errorPropCurv(n, 3, 1),
               errorPropCurv(n, 3, 2),
               errorPropCurv(n, 3, 3),
               errorPropCurv(n, 3, 4));
        printf("%5f %5f %5f %5f %5f\n",
               errorPropCurv(n, 4, 0),
               errorPropCurv(n, 4, 1),
               errorPropCurv(n, 4, 2),
               errorPropCurv(n, 4, 3),
               errorPropCurv(n, 4, 4));
        printf("\n");
      }
    }
#endif

    //now we need jacobians to convert to/from curvilinear and CCS
    // code from TrackState::jacobianCCSToCurvilinear
    MPlex56 jacCCS2Curv(0.0f);
    jacCCS2Curv.aij(0, 3) = mpt::negate_if_ltz(sinT, inChg);
    jacCCS2Curv.aij(0, 5) = mpt::negate_if_ltz(cosT * inPar(3, 0), inChg);
    jacCCS2Curv.aij(1, 5) = -1.f;
    jacCCS2Curv.aij(2, 4) = 1.f;
    jacCCS2Curv.aij(3, 0) = -sinPin;
    jacCCS2Curv.aij(3, 1) = cosPin;
    jacCCS2Curv.aij(4, 0) = -cosPin * cosT;
    jacCCS2Curv.aij(4, 1) = -sinPin * cosT;
    jacCCS2Curv.aij(4, 2) = sinT;

    // code from TrackState::jacobianCurvilinearToCCS
    MPlex65 jacCurv2CCS(0.0f);
    jacCurv2CCS.aij(0, 3) = -sinPout;
    jacCurv2CCS.aij(0, 4) = -cosT * cosPout;
    jacCurv2CCS.aij(1, 3) = cosPout;
    jacCurv2CCS.aij(1, 4) = -cosT * sinPout;
    jacCurv2CCS.aij(2, 4) = sinT;
    jacCurv2CCS.aij(3, 0) = mpt::negate_if_ltz(1.f / sinT, inChg);
    jacCurv2CCS.aij(3, 1) = outPar(3, 0) * cosT / sinT;
    jacCurv2CCS.aij(4, 2) = 1.f;
    jacCurv2CCS.aij(5, 1) = -1.f;

    //need to compute errorProp = jacCurv2CCS*errorPropCurv*jacCCS2Curv
    MPlex65 tmp;
    JacErrPropCurv1(jacCurv2CCS, errorPropCurv, tmp);
    JacErrPropCurv2(tmp, jacCCS2Curv, errorProp);
    /*
    Matriplex::multiplyGeneral(jacCurv2CCS, errorPropCurv, tmp);
    for (int kk = 0; kk < 1; ++kk) {
      std::cout << "jacCurv2CCS" << std::endl;
      for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 5; ++j)
        std::cout << jacCurv2CCS.constAt(kk, i, j) << " ";
      std::cout << std::endl;;
      }
      std::cout << std::endl;;
      std::cout << "errorPropCurv" << std::endl;
      for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j)
        std::cout << errorPropCurv.constAt(kk, i, j) << " ";
      std::cout << std::endl;;
      }
      std::cout << std::endl;;
      std::cout << "tmp" << std::endl;
      for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 5; ++j)
        std::cout << tmp.constAt(kk, i, j) << " ";
      std::cout << std::endl;;
      }
      std::cout << std::endl;;
      std::cout << "jacCCS2Curv" << std::endl;
      for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 6; ++j)
        std::cout << jacCCS2Curv.constAt(kk, i, j) << " ";
      std::cout << std::endl;;
      }
    }
    Matriplex::multiplyGeneral(tmp, jacCCS2Curv, errorProp);
    */
  }

  // from P.Avery's notes (http://www.phys.ufl.edu/~avery/fitting/transport.pdf eq. 5)
  float getS(float delta0,
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
#ifdef DEBUG
    if (debug)
      std::cout << "A=" << A << " B=" << B << " C=" << C << " s1=" << s1 << std::endl;
#endif
    return s1;
  }

  void helixAtPlane_impl(const MPlexLV& __restrict__ inPar,
                         const MPlexQI& __restrict__ inChg,
                         const MPlexHV& __restrict__ plPnt,
                         const MPlexHV& __restrict__ plNrm,
                         MPlexQF& __restrict__ s,
                         MPlexLV& __restrict__ outPar,
                         MPlexLL& __restrict__ errorProp,
                         MPlexQI& __restrict__ outFailFlag,  // expected to be initialized to 0
                         const int N_proc,
                         const PropagationFlags& pf,
                         // false = parameters only, skip the Jacobian (outPar is the same either way)
                         const bool want_err = true) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;

#ifdef DEBUG
    for (int n = 0; n < N_proc; ++n) {
      dprint_np(n,
                "input parameters" << " inPar(n, 0, 0)=" << std::setprecision(9) << inPar(n, 0, 0)
                                   << " inPar(n, 1, 0)=" << std::setprecision(9) << inPar(n, 1, 0)
                                   << " inPar(n, 2, 0)=" << std::setprecision(9) << inPar(n, 2, 0)
                                   << " inPar(n, 3, 0)=" << std::setprecision(9) << inPar(n, 3, 0)
                                   << " inPar(n, 4, 0)=" << std::setprecision(9) << inPar(n, 4, 0)
                                   << " inPar(n, 5, 0)=" << std::setprecision(9) << inPar(n, 5, 0));
    }
#endif

    const MPF kSign = mpt::negate_if_ltz(MPF(-Const::sol_over_100), inChg);
    MPF bFld = pf.use_param_b_field ? getBFieldFromZXY(inPar(2, 0), inPar(0, 0), inPar(1, 0)) : MPF(Config::Bfield);
    MPF kinv = kSign * bFld;

    MPF delta0 = inPar(0, 0) - plPnt(0, 0);
    MPF delta1 = inPar(1, 0) - plPnt(1, 0);
    MPF delta2 = inPar(2, 0) - plPnt(2, 0);

    // trigonometry of the start state, shared by every drift below and by the Jacobian
    const StartTrig tr(inPar);
    MPF sinP = tr.sinP, cosP = tr.cosP;
    const MPF& sinT = tr.sinT;
    const MPF& cosT = tr.cosT;

    // determine solution for straight line
    MPF sl = -(plNrm(0, 0) * delta0 + plNrm(1, 0) * delta1 + plNrm(2, 0) * delta2) /
             (plNrm(0, 0) * cosP * sinT + plNrm(1, 0) * sinP * sinT + plNrm(2, 0) * cosT);

    // The path-length solves are computed for every lane without a branch, so that they vectorise (a select
    // around the square root and the divisions keeps them scalar), and the plane type picks the solution after.
    MPF sH, sZ;
    //first iteration outside the loop
    for (int n = 0; n < NN; ++n)
      sH[n] = getS(delta0[n],
                   delta1[n],
                   delta2[n],
                   plNrm(n, 0, 0),
                   plNrm(n, 1, 0),
                   plNrm(n, 2, 0),
                   sinP[n],
                   cosP[n],
                   sinT[n],
                   cosT[n],
                   inPar(n, 3, 0),
                   inChg(n, 0, 0),
                   kinv[n]);
    for (int n = 0; n < NN; ++n)
      sZ[n] = (plPnt.constAt(n, 2, 0) - inPar.constAt(n, 2, 0)) / cosT[n];
    for (int n = 0; n < N_proc; ++n)
      s[n] = std::abs(plNrm(n, 2, 0)) < 1.f ? sH[n] : sZ[n];

    CMS_UNROLL_LOOP_COUNT(Config::nSStepsInProp2Plane - 1)
    for (int i = 0; i < Config::nSStepsInProp2Plane - 1; ++i) {
      MPlexLV outParTmp{0.0f};
      parsFromPathL_trig(inPar, outParTmp, kinv, s, tr);

      if (pf.use_param_b_field && pf.b_field_at_mid) {
        // Re-sample B at the chord midpoint of the step, 0.5*(start + end).  That point does not depend
        // on which end the step is taken from, so the outward and inward propagations use the same field
        // and are inverses of each other (sampling at the start makes a round trip out through a track's
        // planes and back miss by ~100 um at 10 GeV, ~1 mm at 1 GeV).  The endpoint is only known once
        // s is, hence here, before s is refined; the 6x6 error propagation still runs once, below.
        bFld = getBFieldFromZXY(0.5f * (inPar(2, 0) + outParTmp(2, 0)),
                                0.5f * (inPar(0, 0) + outParTmp(0, 0)),
                                0.5f * (inPar(1, 0) + outParTmp(1, 0)));
        kinv = kSign * bFld;
      }

      delta0 = outParTmp(0, 0) - plPnt(0, 0);
      delta1 = outParTmp(1, 0) - plPnt(1, 0);
      delta2 = outParTmp(2, 0) - plPnt(2, 0);

      mpt::fast_sincos(outParTmp(4, 0), sinP, cosP);
      // Note, sinT/cosT not updated

      for (int n = 0; n < NN; ++n)
        sH[n] = getS(delta0[n],
                     delta1[n],
                     delta2[n],
                     plNrm(n, 0, 0),
                     plNrm(n, 1, 0),
                     plNrm(n, 2, 0),
                     sinP[n],
                     cosP[n],
                     sinT[n],
                     cosT[n],
                     inPar(n, 3, 0),
                     inChg(n, 0, 0),
                     kinv[n]);
      // disks: the scalar cos only on the lanes that need it
      for (int n = 0; n < N_proc; ++n)
        if (!(std::abs(plNrm(n, 2, 0)) < 1.f))
          sH[n] = (plPnt.constAt(n, 2, 0) - outParTmp.constAt(n, 2, 0)) / std::cos(outParTmp.constAt(n, 5, 0));
      for (int n = 0; n < N_proc; ++n)
        s[n] += sH[n];
    }  //end Niter-1

    // use linear approximation if s did not converge (for very high pT tracks)
    for (int n = 0; n < N_proc; ++n) {
#ifdef DEBUG
      if (debug)
        std::cout << "s[n]=" << s[n] << " sl[n]=" << sl[n] << " std::isnan(s[n])=" << std::isnan(s[n])
                  << " std::isfinite(s[n])=" << std::isfinite(s[n]) << " std::isnormal(s[n])=" << std::isnormal(s[n])
                  << std::endl;
#endif
      if (mkfit::isFinite(s[n]) == false && mkfit::isFinite(sl[n]))  // replace with sl even if not fully correct
        s[n] = sl[n];
    }

#ifdef DEBUG
    if (debug)
      std::cout << "s=" << s[0] << std::endl;
#endif
    if (want_err)
      parsAndErrPropFromPathL_trig(inPar, inChg, outPar, kinv, bFld, s, errorProp, N_proc, pf, tr);
    else
      parsFromPathL_trig(inPar, outPar, kinv, s, tr);
  }

}  // namespace
// END STUFF FROM PropagationMPlex.icc
// ============================================================================

namespace mkfit {

  // helixAtPlane with the choice of skipping the Jacobian (want_err = false: parameters only, the same outPar),
  // for the intermediate sub-steps of propagateHelixToPlaneSubStepMPlex.
  static void helixAtPlaneSel(const MPlexLV& inPar,
                              const MPlexQI& inChg,
                              const MPlexHV& plPnt,
                              const MPlexHV& plNrm,
                              MPlexQF& pathL,
                              MPlexLV& outPar,
                              MPlexLL& errorProp,
                              MPlexQI& outFailFlag,
                              const int N_proc,
                              const PropagationFlags& pflags,
                              const bool want_err) {
    errorProp.setVal(0.f);
    outFailFlag.setVal(0.f);

    if (pflags.use_param_b_field && pflags.radial_field_corr) {
      // Radial-field correction, antisymmetric (see brDeltaRPphiV): D from the uncorrected step, half of
      // it as a kick at the start, the helix step from the kicked state, the other half at the end.
      // The Jacobian is that of the helix step: it describes the transport, not the correction, which
      // is a first-order effect on the weighting and not on the mean.
      PropagationFlags pf0 = pflags;
      pf0.radial_field_corr = false;

      MPlexLV par0{0.0f};
      MPlexQF pl0{0.0f};
      MPlexLL ep0{0.0f};
      MPlexQI ff0{0};
      helixAtPlane_impl(inPar, inChg, plPnt, plNrm, pl0, par0, ep0, ff0, N_proc, pf0, false);

      const MPlexQF halfD = 0.5f * brDeltaRPphiV(inPar, par0, inChg);
      MPlexLV parH = inPar;
      applyDpPhiV(parH, halfD, N_proc);
      helixAtPlane_impl(parH, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pf0, want_err);
      applyDpPhiV(outPar, halfD, N_proc);
      return;
    }

    helixAtPlane_impl(inPar, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pflags, want_err);
  }

  void helixAtPlane(const MPlexLV& inPar,
                    const MPlexQI& inChg,
                    const MPlexHV& plPnt,
                    const MPlexHV& plNrm,
                    MPlexQF& pathL,
                    MPlexLV& outPar,
                    MPlexLL& errorProp,
                    MPlexQI& outFailFlag,
                    const int N_proc,
                    const PropagationFlags& pflags) {
    helixAtPlaneSel(inPar, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pflags, true);
  }

  // The part of a propagation to a plane after the transport: 6x6 error similarity with errorProp (outErr must hold
  // the error to transport on entry), material at the destination, phi squash, restore the input on failure.
  // Shared by propagateHelixToPlaneMPlex and propagateHelixToPlaneSubStepMPlex so the two cannot drift apart.
  static void finishPlanePropagation(const MPlexLS& inErr,
                                     const MPlexLV& inPar,
                                     const MPlexHV& plNrm,
                                     const MPlexLL& errorProp,
                                     const MPlexQF& pathL,
                                     MPlexLS& outErr,
                                     MPlexLV& outPar,
                                     MPlexQI& outFailFlag,
                                     const int N_proc,
                                     const PropagationFlags& pflags,
                                     const MPlexQI* noMatEffPtr,
                                     const MPlexQF* matRadl,
                                     const MPlexQF* matBbxi);

  void propagateHelixToPlaneMPlex(const MPlexLS& inErr,
                                  const MPlexLV& inPar,
                                  const MPlexQI& inChg,
                                  const MPlexHV& plPnt,
                                  const MPlexHV& plNrm,
                                  MPlexLS& outErr,
                                  MPlexLV& outPar,
                                  MPlexQI& outFailFlag,
                                  const int N_proc,
                                  const PropagationFlags& pflags,
                                  const MPlexQI* noMatEffPtr,
                                  const MPlexQF* matRadl,
                                  const MPlexQF* matBbxi) {
    // debug = true;

    outErr = inErr;
    outPar = inPar;

    MPlexQF pathL{0.0f};
    MPlexLL errorProp{0.0f};

    helixAtPlane(inPar, inChg, plPnt, plNrm, pathL, outPar, errorProp, outFailFlag, N_proc, pflags);

#ifdef DEBUG
    for (int n = 0; n < N_proc; ++n) {
      dprint_np(
          n,
          "propagation to plane end, dump parameters\n"
              //<< "   D = " << s[n] << " alpha = " << s[n] * std::sin(inPar(n, 5, 0)) * inPar(n, 3, 0) * kinv[n] << " kinv = " << kinv[n] << std::endl
              << "   pos = " << outPar(n, 0, 0) << " " << outPar(n, 1, 0) << " " << outPar(n, 2, 0) << "\t\t r="
              << std::sqrt(outPar(n, 0, 0) * outPar(n, 0, 0) + outPar(n, 1, 0) * outPar(n, 1, 0)) << std::endl
              << "   mom = " << outPar(n, 3, 0) << " " << outPar(n, 4, 0) << " " << outPar(n, 5, 0) << std::endl
              << " charge = " << inChg(n, 0, 0) << std::endl
              << " cart= " << std::cos(outPar(n, 4, 0)) / outPar(n, 3, 0) << " "
              << std::sin(outPar(n, 4, 0)) / outPar(n, 3, 0) << " " << 1. / (outPar(n, 3, 0) * tan(outPar(n, 5, 0)))
              << "\t\tpT=" << 1. / std::abs(outPar(n, 3, 0)) << std::endl);
    }

    if (debug && g_debug) {
      for (int kk = 0; kk < N_proc; ++kk) {
        dprintf("inPar %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          dprintf("%8f ", inPar.constAt(kk, i, 0));
        }
        dprintf("\n");
        dprintf("inErr %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", inErr.constAt(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");

        for (int kk = 0; kk < N_proc; ++kk) {
          dprintf("plNrm %d\n", kk);
          for (int j = 0; j < 3; ++j)
            dprintf("%8f ", plNrm.constAt(kk, 0, j));
        }
        dprintf("\n");

        for (int kk = 0; kk < N_proc; ++kk) {
          dprintf("pathL %d\n", kk);
          for (int j = 0; j < 1; ++j)
            dprintf("%8f ", pathL.constAt(kk, 0, j));
        }
        dprintf("\n");

        dprintf("errorProp %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", errorProp.At(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");
      }
    }
#endif

    finishPlanePropagation(inErr,
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
                           matBbxi);
  }

  static void finishPlanePropagation(const MPlexLS& inErr,
                                     const MPlexLV& inPar,
                                     const MPlexHV& plNrm,
                                     const MPlexLL& errorProp,
                                     const MPlexQF& pathL,
                                     MPlexLS& outErr,
                                     MPlexLV& outPar,
                                     MPlexQI& outFailFlag,
                                     const int N_proc,
                                     const PropagationFlags& pflags,
                                     const MPlexQI* noMatEffPtr,
                                     const MPlexQF* matRadl,
                                     const MPlexQF* matBbxi) {
    // Matriplex version of:
    // result.errors = ROOT::Math::Similarity(errorProp, outErr);
    MPlexLL temp{0.0f};
    MultHelixPlaneProp(errorProp, outErr, temp);
    MultHelixPlanePropTransp(errorProp, temp, outErr);
    // MultHelixPropFull(errorProp, outErr, temp);
    // for (int kk = 0; kk < 1; ++kk) {
    //   std::cout << "errorProp" << std::endl;
    //   for (int i = 0; i < 6; ++i) {
    // 	for (int j = 0; j < 6; ++j)
    // 	  std::cout << errorProp.constAt(kk, i, j) << " ";
    // 	std::cout << std::endl;;
    //   }
    //   std::cout << std::endl;;
    //   std::cout << "outErr" << std::endl;
    //   for (int i = 0; i < 6; ++i) {
    // 	for (int j = 0; j < 6; ++j)
    // 	  std::cout << outErr.constAt(kk, i, j) << " ";
    // 	std::cout << std::endl;;
    //   }
    //   std::cout << std::endl;;
    //   std::cout << "temp" << std::endl;
    //   for (int i = 0; i < 6; ++i) {
    // 	for (int j = 0; j < 6; ++j)
    // 	  std::cout << temp.constAt(kk, i, j) << " ";
    // 	std::cout << std::endl;;
    //   }
    //   std::cout << std::endl;;
    // }
    // MultHelixPropTranspFull(errorProp, temp, outErr);

#ifdef DEBUG
    if (debug && g_debug) {
      for (int kk = 0; kk < N_proc; ++kk) {
        dprintf("outErr %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", outErr.constAt(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");
      }
    }
#endif

    if (pflags.apply_material) {
      MPlexQF hitsRl;
      MPlexQF hitsXi;
      MPlexQF propSign;

      const TrackerInfo& tinfo = *pflags.tracker_info;
      // the crossed module's own material instead of the (|z|,r) grid (Config::refitMaterialPerModule)
      const bool use_mod_mat = Config::refitMaterialPerModule && matRadl && matBbxi;
      // energy-loss sign from the fit pass (PropagationFlags::eloss_by_pass), else from the path-length sign
      const float passSign = pflags.eloss_outward ? 1.f : -1.f;
      const bool by_pass = pflags.eloss_by_pass;

#if !defined(__clang__)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
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
            const auto mat = tinfo.material_checked(std::abs(outPar(n, 2, 0)), hypo);
            hitsRl(n, 0, 0) = mat.radl;
            hitsXi(n, 0, 0) = mat.bbxi;
          }
          propSign(n, 0, 0) = by_pass ? passSign : (pathL(n, 0, 0) > 0.f ? 1.f : -1.f);
        }
      }
      applyMaterialEffects(hitsRl, hitsXi, propSign, plNrm, outErr, outPar, N_proc, pflags.ms_ref_p);
#ifdef DEBUG
      if (debug && g_debug) {
        for (int kk = 0; kk < N_proc; ++kk) {
          dprintf("propSign %d\n", kk);
          for (int i = 0; i < 1; ++i) {
            dprintf("%8f ", propSign.constAt(kk, i, 0));
          }
          dprintf("\n");
          dprintf("plNrm %d\n", kk);
          for (int i = 0; i < 3; ++i) {
            dprintf("%8f ", plNrm.constAt(kk, i, 0));
          }
          dprintf("\n");
          dprintf("outErr(after material) %d\n", kk);
          for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j)
              dprintf("%8f ", outErr.constAt(kk, i, j));
            dprintf("\n");
          }
          dprintf("\n");
        }
      }
#endif
    }

    squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

    // PROP-FAIL-ENABLE To keep physics changes minimal, we always restore the
    // state to input when propagation fails -- as was the default before.
    // if (pflags.copy_input_state_on_fail) {
    for (int i = 0; i < N_proc; ++i) {
      if (outFailFlag(i, 0, 0)) {
        outPar.copySlot(i, inPar);
        outErr.copySlot(i, inErr);
      }
    }
    // }
  }

  // ============================================================================
  // Sub-stepped propagation to a plane (refit backward pass, Config::refitBkwSubSteps; MkFitter).
  //
  // One propagation over a long step mis-bends near the solenoid ends: B is sampled once (at the chord midpoint)
  // and the radial-field correction is applied only at the two ends, which is exact only to second order in the
  // step length.  nSub sub-steps remove 1 - 1/nSub^2 of that error.  Only the parameters need the sub-steps:
  //   nSub-1 drifts of fixed path length s0/nSub (s0 = the first path-length estimate to the destination plane),
  //   each with the same field model as a full step (midpoint B, radial-field half-kicks), no plane solve; the
  //   last sub-step lands on the plane with the full solve.
  // The covariance is transported once, with the Jacobian of the whole step (the same one-step transport as without
  // sub-stepping), and material is applied at the destination only, as before.  split[n] = false keeps lane n as
  // one step (zero-length intermediate sub-steps).

  // First path-length estimate to the plane: the starting value of helixAtPlane_impl's solve (exact in z for disks),
  // straight line if that is not finite.  Only partitions the step -- the last sub-step lands with the full solve.
  static MPlexQF firstPathEstimate(const MPlexLV& inPar,
                                   const MPlexQI& inChg,
                                   const MPlexHV& plPnt,
                                   const MPlexHV& plNrm,
                                   const int N_proc,
                                   const PropagationFlags& pf,
                                   const StartTrig& tr) {
    namespace mpt = Matriplex;
    const MPlexQF kSign = mpt::negate_if_ltz(MPlexQF(-Const::sol_over_100), inChg);
    const MPlexQF bFld =
        pf.use_param_b_field ? getBFieldFromZXY(inPar(2, 0), inPar(0, 0), inPar(1, 0)) : MPlexQF(Config::Bfield);
    const MPlexQF kinv = kSign * bFld;
    const MPlexQF &sinP = tr.sinP, &cosP = tr.cosP, &sinT = tr.sinT, &cosT = tr.cosT;
    // the three candidates for every lane without a branch (so that they vectorise), then the choice
    MPlexQF sH, sZ, sL;
    for (int n = 0; n < NN; ++n) {
      const float d0 = inPar.constAt(n, 0, 0) - plPnt.constAt(n, 0, 0);
      const float d1 = inPar.constAt(n, 1, 0) - plPnt.constAt(n, 1, 0);
      const float d2 = inPar.constAt(n, 2, 0) - plPnt.constAt(n, 2, 0);
      const float e0 = plNrm.constAt(n, 0, 0), e1 = plNrm.constAt(n, 1, 0), e2 = plNrm.constAt(n, 2, 0);
      sH[n] = getS(d0,
                   d1,
                   d2,
                   e0,
                   e1,
                   e2,
                   sinP[n],
                   cosP[n],
                   sinT[n],
                   cosT[n],
                   inPar.constAt(n, 3, 0),
                   inChg.constAt(n, 0, 0),
                   kinv[n]);
      sZ[n] = (plPnt.constAt(n, 2, 0) - inPar.constAt(n, 2, 0)) / cosT[n];
      sL[n] = -(e0 * d0 + e1 * d1 + e2 * d2) / (e0 * cosP[n] * sinT[n] + e1 * sinP[n] * sinT[n] + e2 * cosT[n]);
    }
    MPlexQF s0{0.0f};
    for (int n = 0; n < N_proc; ++n) {
      float s = std::abs(plNrm.constAt(n, 2, 0)) < 1.f ? sH[n] : sZ[n];
      if (!mkfit::isFinite(s))
        s = sL[n];
      s0[n] = mkfit::isFinite(s) ? s : 0.f;
    }
    return s0;
  }

  // One fixed-length sub-step of path length h (per lane; 0 = no move), parameters only, with the same field model
  // as a propagation to a plane: predictor drift with B at the start, re-drift with B at the chord midpoint
  // (b_field_at_mid), radial-field half-kicks D/(2 r0) before and D/(2 r1) after the drift (radial_field_corr; D
  // between the uncorrected endpoints).  The kicked drift reuses the midpoint field (the kick moves the chord
  // midpoint only at second order).
  static void fixedLengthSubStep(MPlexLV& par,
                                 const MPlexQI& inChg,
                                 const MPlexQF& kSign,
                                 const MPlexQF& h,
                                 const int N_proc,
                                 const PropagationFlags& pf) {
    const MPlexLV& p0 = par;
    MPlexQF kinv =
        kSign * (pf.use_param_b_field ? getBFieldFromZXY(p0(2, 0), p0(0, 0), p0(1, 0)) : MPlexQF(Config::Bfield));
    MPlexLV p1{0.0f};
    const StartTrig tr0(p0);
    parsFromPathL_trig(p0, p1, kinv, h, tr0);
    if (pf.use_param_b_field && pf.b_field_at_mid) {
      kinv = kSign *
             getBFieldFromZXY(0.5f * (p0(2, 0) + p1(2, 0)), 0.5f * (p0(0, 0) + p1(0, 0)), 0.5f * (p0(1, 0) + p1(1, 0)));
      parsFromPathL_trig(p0, p1, kinv, h, tr0);
    }
    if (pf.use_param_b_field && pf.radial_field_corr) {
      const MPlexQF dvec = brDeltaRPphiV(p0, p1, inChg);
      MPlexLV ph = p0;
      applyDpPhiV(ph, 0.5f * dvec, N_proc);
      parsFromPathL_impl(ph, p1, kinv, h);
      applyDpPhiV(p1, 0.5f * dvec, N_proc);
    }
    squashPhiMPlex(p1, N_proc);
    par = p1;
  }

  void propagateHelixToPlaneSubStepMPlex(const MPlexLS& inErr,
                                         const MPlexLV& inPar,
                                         const MPlexQI& inChg,
                                         const MPlexHV& plPnt,
                                         const MPlexHV& plNrm,
                                         MPlexLS& outErr,
                                         MPlexLV& outPar,
                                         MPlexQI& outFailFlag,
                                         const int N_proc,
                                         const PropagationFlags& pflags,
                                         const int nSub,
                                         const bool* split,
                                         const MPlexQI* noMatEffPtr,
                                         const MPlexQF* matRadl,
                                         const MPlexQF* matBbxi) {
    namespace mpt = Matriplex;
    PropagationFlags pfs = pflags;
    pfs.apply_material = false;  // intermediate sub-steps: no material (it is applied at the destination)
    outFailFlag.setVal(0);
    MPlexQI ff{0};
    MPlexLL epDummy{0.0f};
    MPlexQF pl{0.0f}, sTot{0.0f};
    MPlexLV par = inPar;
    const MPlexQF kSign = mpt::negate_if_ltz(MPlexQF(-Const::sol_over_100), inChg);

    const StartTrig trIn(inPar);
    MPlexQF h = firstPathEstimate(par, inChg, plPnt, plNrm, N_proc, pflags, trIn);
    for (int n = 0; n < NN; ++n)
      h[n] = (n < N_proc && (split == nullptr || split[n])) ? h[n] / nSub : 0.f;
    for (int ks = 1; ks < nSub; ++ks) {
      fixedLengthSubStep(par, inChg, kSign, h, N_proc, pfs);
      sTot = sTot + h;
    }

    // last sub-step onto the destination plane, parameters only
    helixAtPlaneSel(par, inChg, plPnt, plNrm, pl, outPar, epDummy, ff, N_proc, pfs, false);
    for (int n = 0; n < N_proc; ++n)
      if (ff.constAt(n, 0, 0))
        outFailFlag.At(n, 0, 0) = 1;
    sTot = sTot + pl;

    // whole-step Jacobian: from the step start over the accumulated path length, with the field of the step
    // (chord midpoint with b_field_at_mid) -- i.e. the one-step transport
    const MPlexQF bW = !pflags.use_param_b_field ? MPlexQF(Config::Bfield)
                       : pflags.b_field_at_mid   ? getBFieldFromZXY(0.5f * (inPar(2, 0) + outPar(2, 0)),
                                                                  0.5f * (inPar(0, 0) + outPar(0, 0)),
                                                                  0.5f * (inPar(1, 0) + outPar(1, 0)))
                                                 : getBFieldFromZXY(inPar(2, 0), inPar(0, 0), inPar(1, 0));
    MPlexLV parJ{0.0f};
    MPlexLL errorProp{0.0f};
    parsAndErrPropFromPathL_trig(inPar, inChg, parJ, kSign * bW, bW, sTot, errorProp, N_proc, pflags, trIn);

    outErr = inErr;
    finishPlanePropagation(
        inErr, inPar, plNrm, errorProp, sTot, outErr, outPar, outFailFlag, N_proc, pflags, noMatEffPtr, matRadl, matBbxi);
  }

}  // namespace mkfit
