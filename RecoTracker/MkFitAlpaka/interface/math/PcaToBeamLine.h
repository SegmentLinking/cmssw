#ifndef RecoTracker_MkFitAlpaka_interface_math_PcaToBeamLine_h
#define RecoTracker_MkFitAlpaka_interface_math_PcaToBeamLine_h
// The device PCA, a portable (ALPAKA_FN_HOST_ACC) transliteration of what the output converter
// does on the host for every track (MkFitOutputTrackConverter / MkFitAlpakaOutputTrackConverter):
//   mkfit::TrackState::convertFromCCSToGlbCurvilinear (MkFitCore/src/Track.cc, float)                  ccsToCurvilinear
//   TSCBLBuilderNoMaterial (TrackingTools/PatternTools/src/TSCBLBuilderNoMaterial.cc)                   pcaToBeamLine
//     = TwoTrackMinimumDistance helix-line (TwoTrackMinimumDistanceHelixLine.cc, Newton on the helix phase, 12 steps,
//       tolerance 1e-6) with the field at the start point only (GlobalTrajectoryParameters' cached field)
//     + AnalyticalCurvilinearJacobian (the USE_EXTVECT variant AnalyticalCurvilinearJacobianEXT.icc, as built in CMSSW)
//     + C' = J C J^T.
// The field at the start point is the menu's TkBfield "3_8T" (interface/math/TkBfield.h, bitwise = the CMSSW
// OAEParametrizedMagneticField inside r < 115 cm, |z| < 280 cm); outside that volume the caller falls back to the host.
// Types follow the host code (TSCBLBuilderNoMaterial): GlobalPoint/GlobalVector are float, the TTMD and the Jacobian
// compute in double. Not bitwise vs the host (FMA contraction, libm vs device sin/cos/atan2): rounding-level
// differences to the host TSCBL.
#include <cmath>
#include <cstdint>

#include <alpaka/core/Common.hpp>

#include "RecoTracker/MkFitAlpaka/interface/math/TkBfield.h"

namespace mkfitdev::vdt {
  // vdt 0.4.3 sincos.h, the DOUBLE fast_sincos (AnalyticalCurvilinearJacobianEXT calls vdt::fast_sincos(double))
  namespace dsincos {
    ALPAKA_FN_HOST_ACC inline double getSinPx(const double x) {
      double px = 1.58962301576546568060E-10;
      px *= x;
      px += -2.50507477628578072866E-8;
      px *= x;
      px += 2.75573136213857245213E-6;
      px *= x;
      px += -1.98412698295895385996E-4;
      px *= x;
      px += 8.33333333332211858878E-3;
      px *= x;
      px += -1.66666666666666307295E-1;
      return px;
    }
    ALPAKA_FN_HOST_ACC inline double getCosPx(const double x) {
      double px = -1.13585365213876817300E-11;
      px *= x;
      px += 2.08757008419747316778E-9;
      px *= x;
      px += -2.75573141792967388112E-7;
      px *= x;
      px += 2.48015872888517045348E-5;
      px *= x;
      px += -1.38888888888730564116E-3;
      px *= x;
      px += 4.16666666666665929218E-2;
      return px;
    }
    ALPAKA_FN_HOST_ACC inline double reduce2quadrant(double x, int32_t& quad) {
      constexpr double DP1 = 7.853981554508209228515625E-1;
      constexpr double DP2 = 7.94662735614792836714E-9;
      constexpr double DP3 = 3.06161699786838294307E-17;
      constexpr double ONEOPIO4 = 4. / M_PI;
      x = std::fabs(x);
      quad = int(ONEOPIO4 * x);
      quad = (quad + 1) & (~1);
      const double y = double(quad);
      return ((x - y * DP1) - y * DP2) - y * DP3;
    }
  }  // namespace dsincos
  ALPAKA_FN_HOST_ACC inline void fast_sincos(const double xx, double& s, double& c) {
    int j;
    const double x = dsincos::reduce2quadrant(xx, j);
    const double signS = (j & 4);
    j -= 2;
    const double signC = (j & 4);
    const double poly = j & 2;
    const double zz = x * x;
    s = x + x * zz * dsincos::getSinPx(zz);
    c = 1.0 - zz * .5 + zz * zz * dsincos::getCosPx(zz);
    if (poly == 0) {
      const double tmp = c;
      c = s;
      s = tmp;
    }
    if (signC == 0.)
      c = -c;
    if (signS != 0.)
      s = -s;
    if (xx < 0.)
      s = -s;
  }
}  // namespace mkfitdev::vdt

namespace mkfitdev::pca {

  // float sin/cos/atan2 as the host's glibc sinf/cosf/atan2f (correctly rounded in practice): evaluated in double and
  // rounded once, so the device's float inputs match the host's (the device float functions are 1-3 ulp)
  ALPAKA_FN_HOST_ACC inline float cosF(float x) { return float(std::cos(double(x))); }
  ALPAKA_FN_HOST_ACC inline float sinF(float x) { return float(std::sin(double(x))); }
  ALPAKA_FN_HOST_ACC inline float atan2F(float y, float x) { return float(std::atan2(double(y), double(x))); }

  struct F3 {
    float x, y, z;
  };
  struct D4 {  // the Vec3D/Vec4D lanes of ExtVec.h
    double v[4];
  };
  ALPAKA_FN_HOST_ACC inline D4 d4(F3 a) { return D4{{double(a.x), double(a.y), double(a.z), 0.}}; }
  ALPAKA_FN_HOST_ACC inline double dot(D4 const& a, D4 const& b) {  // ExtVec dot: ret = 0; ret += a_i b_i, i = 0..3
    double r = 0;
    for (int i = 0; i < 4; ++i)
      r += a.v[i] * b.v[i];
    return r;
  }
  ALPAKA_FN_HOST_ACC inline double dot2(D4 const& a, D4 const& b) { return a.v[0] * b.v[0] + a.v[1] * b.v[1]; }
  ALPAKA_FN_HOST_ACC inline D4 xy(D4 const& a) { return D4{{a.v[0], a.v[1], 0., 0.}}; }
  ALPAKA_FN_HOST_ACC inline D4 cross3(D4 const& x, D4 const& y) {
    // ExtVec cross3: x1200 * y2010 - x2010 * y1200
    return D4{{x.v[1] * y.v[2] - x.v[2] * y.v[1],
               x.v[2] * y.v[0] - x.v[0] * y.v[2],
               x.v[0] * y.v[1] - x.v[1] * y.v[0],
               x.v[0] * y.v[0] - x.v[0] * y.v[0]}};
  }
  // float GlobalVector helpers (Basic3DVector<float>)
  ALPAKA_FN_HOST_ACC inline float mag2(F3 a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
  ALPAKA_FN_HOST_ACC inline float mag(F3 a) { return std::sqrt(mag2(a)); }
  ALPAKA_FN_HOST_ACC inline float perp(F3 a) { return std::sqrt(a.x * a.x + a.y * a.y); }
  ALPAKA_FN_HOST_ACC inline F3 unit(
      F3 a) {  // Basic3DVector::unit: my_mag = mag2; my_mag == 0 ? *this : *this * (1/sqrt)
    float const m2 = mag2(a);
    if (m2 == 0.f)
      return a;
    float const s = 1.f / std::sqrt(m2);
    return F3{a.x * s, a.y * s, a.z * s};
  }

  // ---- inputs / outputs ----
  struct TrackIn {  // the FreeTrajectoryState the converter builds at the first hit
    F3 x, p;
    int charge;
    double C[5][5];  // curvilinear error (q/p, lambda, phi, xT, yT), symmetric
  };
  struct BeamIn {
    F3 pos;  // GlobalPoint(beamSpot.position()) (float)
    F3 dir;  // GlobalVector(dxdz, dydz, 1)
  };
  struct PcaOut {
    F3 x, p;  // state at the PCA (GlobalPoint / GlobalVector, float)
    double C[5][5];
    int status;  // kPcaOk, kPcaFailed (host: invalid TSCBL), kPcaHostFallback (field outside the closed form)
  };
  enum PcaStatus : int8_t { kPcaOk = 0, kPcaFailed = 1, kPcaHostFallback = 2 };

  // mkfit::TrackState::convertFromCCSToGlbCurvilinear on (x, y, z, 1/pT, phi, theta) + the SMatrixSym66 errors
  // (packed lower triangle, TrackSoA order), followed by the converter's FreeTrajectoryState construction: position,
  // momentum (float) and the top-left 5x5 of the converted errors (float -> double). Same float operations as the host.
  ALPAKA_FN_HOST_ACC inline void ccsToCurvilinear(float const* par, float const* err, int charge, TrackIn& t) {
    const float invpt = par[3];
    const float phi = par[4];
    const float theta = par[5];
    const float pt = 1.f / invpt;
    const float cosP = cosF(phi);
    const float sinP = sinF(phi);
    const float cosT = cosF(theta);
    const float sinT = sinF(theta);
    t.x = F3{par[0], par[1], par[2]};
    t.p = F3{cosP * pt, sinP * pt, cosT * pt / sinT};
    t.charge = charge;
    // jacobianCCSToCurvilinear (SMatrix66, zero elsewhere)
    float J[6][6] = {};
    J[3][0] = -sinP;
    J[4][0] = -cosP * cosT;
    J[3][1] = cosP;
    J[4][1] = -sinP * cosT;
    J[4][2] = sinT;
    J[0][3] = charge * sinT;
    J[0][5] = charge * cosT * invpt;
    J[1][5] = -1.f;
    J[2][4] = 1.f;
    auto e = [err](int i, int j) { return i >= j ? err[i * (i + 1) / 2 + j] : err[j * (j + 1) / 2 + i]; };
    // ROOT::Math::Similarity(J, E) in float: (J E) then (J E) J^T, k ascending; only the 5x5 the converter reads
    float JE[5][6];
    for (int i = 0; i < 5; ++i)
      for (int k = 0; k < 6; ++k) {
        float a = 0.f;
        for (int l = 0; l < 6; ++l)
          a += J[i][l] * e(l, k);
        JE[i][k] = a;
      }
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j <= i; ++j) {
        float a = 0.f;
        for (int k = 0; k < 6; ++k)
          a += JE[i][k] * J[j][k];
        t.C[i][j] = t.C[j][i] = double(a);
      }
  }

  // AnalyticalCurvilinearJacobian (the USE_EXTVECT variant CMSSW builds) of the helix from (x1, p1, charge) over the
  // path s to (x2, p2), field B (Tesla) at the start; used by pcaToBeamLine and by the navigation's per-det test
  // (DetPlaneTest.h)
  ALPAKA_FN_HOST_ACC inline void curvilinearJacobian(
      F3 const x1, F3 const p1, int const charge, F3 const x2, F3 const p2, double const s, F3 const B, double J[5][5]) {
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        J[i][j] = i == j ? 1. : 0.;
    float const bza = -2.99792458e-3f * (float(charge) / perp(p1)) * B.z;  // transverseCurvature()
    F3 const p1f = unit(p1);
    if (s * s * std::fabs(bza) > 1.e-5) {
      F3 const hF{2.99792458e-3f * B.x, 2.99792458e-3f * B.y, 2.99792458e-3f * B.z};
      F3 const p2f = unit(p2);
      F3 const dxf{x1.x - x2.x, x1.y - x2.y, x1.z - x2.z};
      double const qbp = float(charge) / mag(p1);  // signedInverseMomentum() (float)
      double const absS = s;
      double const cosl0 = perp(p1f);
      double const cosl1 = 1. / perp(p2f);
      F3 const hnf = unit(hF);
      double const qp = -mag(hF);
      double const q = qp * qbp;
      double const theta = q * absS;
      double sint, cost;
      ::mkfitdev::vdt::fast_sincos(theta, sint, cost);  // as the host (vdt, double)
      D4 const t1 = d4(p1f), t2 = d4(p2f), hn = d4(hnf), dx = d4(dxf);
      double const gamma = dot(hn, t2);
      D4 const an = cross3(hn, t2);
      double const tt0 = t1.v[1] * t1.v[1], tt1 = t1.v[0] * t1.v[0], tt2 = t2.v[1] * t2.v[1], tt3 = t2.v[0] * t2.v[0];
      double const au0 = -std::sqrt(tt1 + tt0), au1 = std::sqrt(tt0 + tt1), au2 = -std::sqrt(tt3 + tt2),
                   au3 = std::sqrt(tt2 + tt3);
      double const uu0 = t1.v[1] / au0, uu1 = t1.v[0] / au1, uu2 = t2.v[1] / au2, uu3 = t2.v[0] / au3;
      D4 const u1{{uu0, uu1, 0., 0.}}, u2{{uu2, uu3, 0., 0.}};
      D4 const u13 = u1, u23 = u2;
      D4 const v1 = cross3(t1, u13);
      D4 const v2 = cross3(t2, u23);
      double const anv = -dot(hn, u23);
      double const anu = dot(hn, v2);
      double const omcost = 1. - cost;
      double const tmsint = theta - sint;
      D4 const hu = cross3(hn, u13);
      D4 const hv = cross3(hn, v1);
      J[1][0] = -qp * anv * dot(t2, dx);
      J[1][1] = cost * dot(v1, v2) + sint * dot(hv, v2) + omcost * dot(hn, v1) * dot(hn, v2) +
                anv * (-sint * dot(v1, t2) + omcost * dot(v1, an) - tmsint * gamma * dot(hn, v1));
      J[1][2] = cost * dot2(u1, v2) + sint * dot(hu, v2) + omcost * dot2(hn, u1) * dot(hn, v2) +
                anv * (-sint * dot2(u1, t2) + omcost * dot2(u1, an) - tmsint * gamma * dot2(hn, u1));
      J[1][2] *= cosl0;
      J[1][3] = -q * anv * dot2(u1, t2);
      J[1][4] = -q * anv * dot(v1, t2);
      J[2][0] = -qp * anu * cosl1 * dot(t2, dx);
      J[2][1] = cost * dot(xy(v1), u2) + sint * dot(xy(hv), u2) + omcost * dot(hn, v1) * dot2(hn, u2) +
                anu * (-sint * dot(v1, t2) + omcost * dot(v1, an) - tmsint * gamma * dot(hn, v1));
      J[2][1] *= cosl1;
      J[2][2] = cost * dot(u1, u2) + sint * dot2(hu, u2) + omcost * dot2(hn, u1) * dot2(hn, u2) +
                anu * (-sint * dot2(u1, t2) + omcost * dot2(u1, an) - tmsint * gamma * dot2(hn, u1));
      J[2][2] *= cosl1 * cosl0;
      J[2][3] = -q * anu * cosl1 * dot2(u1, t2);
      J[2][4] = -q * anu * cosl1 * dot(v1, t2);
      double const overQ = 1. / q;
      J[3][1] = (sint * dot(xy(v1), u2) + omcost * dot2(hv, u2) + tmsint * dot(hn, v1) * dot2(hn, u2)) * overQ;
      J[3][2] = (sint * dot(u1, u2) + omcost * dot2(hu, u2) + tmsint * dot2(hn, u1) * dot2(hn, u2)) * (cosl0 * overQ);
      J[3][3] = dot(u1, u2);
      J[3][4] = dot(xy(v1), u2);
      J[4][1] = (sint * dot(v1, v2) + omcost * dot(hv, v2) + tmsint * dot(hn, v1) * dot(hn, v2)) * overQ;
      J[4][2] = (sint * dot(u1, xy(v2)) + omcost * dot(hu, v2) + tmsint * dot2(hn, u1) * dot(hn, v2)) * (cosl0 * overQ);
      J[4][3] = dot2(u1, v2);
      J[4][4] = dot(v1, v2);
      double const cutCriterion = std::abs(s * qbp);
      if (cutCriterion > 5.) {
        double const pp = 1. / qbp;
        J[3][0] = pp * dot2(u2, dx);
        J[4][0] = pp * dot(v2, dx);
      } else {
        D4 const hp1 = cross3(hn, t1);
        double const temp1 = dot2(hp1, u2);
        D4 const ghnmp{{gamma * hn.v[0] - t1.v[0],
                        gamma * hn.v[1] - t1.v[1],
                        gamma * hn.v[2] - t1.v[2],
                        gamma * hn.v[3] - t1.v[3]}};
        double const temp2 = dot(xy(ghnmp), u2);
        double const qps = qp * s;
        double const h2 = qps * qbp;
        double const h3 = (-1. / 8.) * h2;
        double const secondOrder41 = 0.5 * temp1;
        double const thirdOrder41 = (1. / 3.) * temp2;
        double const fourthOrder41 = h3 * temp1;
        J[3][0] = (s * qps) * (secondOrder41 + h2 * (thirdOrder41 + fourthOrder41));
        double const temp3 = dot(hp1, v2);
        double const temp4 = dot(ghnmp, v2);
        double const secondOrder51 = 0.5 * temp3;
        double const thirdOrder51 = (1. / 3.) * temp4;
        double const fourthOrder51 = h3 * temp3;
        J[4][0] = (s * qps) * (secondOrder51 + h2 * (thirdOrder51 + fourthOrder51));
      }
    } else {  // computeStraightLineJacobian
      double const cosl0 = perp(p1f);
      J[3][2] = cosl0 * s;
      J[4][1] = s;
    }
  }

  // C' = J C J^T (ROOT::Math::Similarity)
  ALPAKA_FN_HOST_ACC inline void similarity5(double const J[5][5], double const C[5][5], double out[5][5]) {
    double JC[5][5];
    for (int i = 0; i < 5; ++i)
      for (int k = 0; k < 5; ++k) {
        double a = 0;
        for (int l = 0; l < 5; ++l)
          a += J[i][l] * C[l][k];
        JC[i][k] = a;
      }
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j <= i; ++j) {
        double a = 0;
        for (int k = 0; k < 5; ++k)
          a += JC[i][k] * J[j][k];
        out[i][j] = out[j][i] = a;
      }
  }

  // TwoTrackMinimumDistanceHelixLine with theH = the track, theL = the beam line; returns 0 on success.
  // B: the field at t.x in Tesla (GlobalTrajectoryParameters::cachedMagneticField).
  ALPAKA_FN_HOST_ACC inline int pcaToBeamLine(TrackIn const& t, F3 const B, BeamIn const& bl, PcaOut& o) {
    // --- updateCoeffs ---
    double const Hn = mag(t.p);
    double const Ln = mag(bl.dir);
    if (Hn == 0. || Ln == 0.)
      return o.status = kPcaFailed;
    F3 const posDiffF{bl.pos.x - t.x.x, bl.pos.y - t.x.y, bl.pos.z - t.x.z};
    double const X = posDiffF.x, Y = posDiffF.y, Z = posDiffF.z;
    double const px = bl.dir.x, py = bl.dir.y, pz = bl.dir.z;
    double const px2 = px * px, py2 = py * py, pz2 = pz * pz;
    double const Bc2kH = B.z * 2.99792458e-3;
    if (Bc2kH == 0. || t.charge == 0)
      return o.status = kPcaFailed;
    float const pzH2 = t.p.z * t.p.z;  // float * float, as the host
    double const theh = -Hn / (t.charge * Bc2kH) * std::sqrt(1 - ((pzH2) / (Hn * Hn)));
    double const tl = -t.p.z / (t.charge * Bc2kH * theh);
    double const phiH0 = double(atan2F(t.p.y, t.p.x));  // GlobalVector::phi() (float)
    double const sinPhiH0 = std::sin(phiH0);
    double const cosPhiH0 = std::cos(phiH0);
    double const aa = (X + theh * sinPhiH0) * (py2 + pz2) - px * (py * Y + pz * Z);
    double const bb = (Y - theh * cosPhiH0) * (px2 + pz2) - py * (px * X + pz * Z);
    double const cc = pz * theh * tl;
    double const dd = theh * px * py;
    double const ee = theh * (px2 - py2);
    double const ff = (px2 + py2) * theh * tl * tl;
    double const baseFct = tl * (Z * (px2 + py2) - pz * (px * X + py * Y));
    double const baseDer = -ff;
    // --- Newton ---
    double phiH = phiH0;
    double const x1 = phiH0 - M_PI, x2 = phiH0 + M_PI;
    bool converged = false;
    for (int j = 1; j <= 12; ++j) {
      double const s = std::sin(phiH), c = std::cos(phiH);
      double fct = baseFct;
      fct -= ff * (phiH - phiH0);
      fct += c * aa;
      fct += s * bb;
      fct += cc * (phiH - phiH0) * (px * c + py * s);
      fct += cc * (px * (s - sinPhiH0) - py * (c - cosPhiH0));
      fct += dd * (s * (s - sinPhiH0) - c * (c - cosPhiH0));
      fct += ee * c * s;
      double der = baseDer;
      der += -s * aa;
      der += c * bb;
      der += cc * (phiH - phiH0) * (py * c - px * s);
      der += 2 * cc * (px * c + py * s);
      der += dd * (4 * c * s - c * sinPhiH0 - s * cosPhiH0);
      der += ee * (c * c - s * s);
      double const dPhiH = fct / der;
      phiH -= dPhiH;
      if ((x1 - phiH) * (phiH - x2) < 0.0)
        phiH += (dPhiH * 0.8);
      if (std::fabs(dPhiH) < 1e-6f) {  // TwoTrackMinimumDistance passes qual as float
        converged = true;
        break;
      }
    }
    if (!converged)
      return o.status = kPcaFailed;
    o.status = kPcaOk;
    // --- finalPoints (helix side) ---
    o.x = F3{float(t.x.x + theh * (std::sin(phiH) - sinPhiH0)),
             float(t.x.y + theh * (-std::cos(phiH) + cosPhiH0)),
             float(t.x.z + theh * (tl * (phiH - phiH0)))};
    double const s = (phiH - phiH0) * (theh * std::sqrt(1 + tl * tl));
    // pTrack = GlobalVector(Cylindrical(perp, phi, z)) in float
    float const rho = perp(t.p), phiF = float(phiH);
    o.p = F3{rho * cosF(phiF), rho * sinF(phiF), t.p.z};

    // --- AnalyticalCurvilinearJacobian(gtp, xTrack, pTrack, s), C' = J C J^T ---
    double J[5][5];
    curvilinearJacobian(t.x, t.p, t.charge, o.x, o.p, s, B, J);
    similarity5(J, t.C, o.C);
    return kPcaOk;
  }

  // The whole per-track device PCA: field at the first-hit state (closed form; outside its volume -> host fallback),
  // then the TSCBL transliteration.
  ALPAKA_FN_HOST_ACC inline int pcaFromFirstHitState(TrackIn const& t, BeamIn const& bl, PcaOut& o) {
    if (!::mkfitdev::field::tkBfieldValid(t.x.x, t.x.y, t.x.z))
      return o.status = kPcaHostFallback;
    F3 B;
    ::mkfitdev::field::tkBfield(t.x.x, t.x.y, t.x.z, B.x, B.y, B.z);
    return pcaToBeamLine(t, B, bl, o);
  }

}  // namespace mkfitdev::pca

#endif
