#ifndef RecoTracker_MkFitAlpaka_interface_math_DetPlaneTest_h
#define RecoTracker_MkFitAlpaka_interface_math_DetPlaneTest_h

// The device per-det test of the missing-hit navigation:
// what GeomDetCompatibilityChecker + AnalyticalPropagator + Chi2MeasurementEstimator(-3 sigma) do for one
// det plane, portable (ALPAKA_FN_HOST_ACC), in double:
//   - the straight-line sagitta pre-check (maxSagitta 2 cm, minTolerance 0.5 cm, bounds widened by the tolerance);
//   - the helix (field along z at the start point, the TSOS transverse curvature) to the plane: Newton on the 3D path
//     length from the straight-line solution, the first crossing in the propagation direction;
//   - AnalyticalPropagator's maxDPhi rule (|rho * s_T| <= 1.6);
//   - the local position errors from the END curvilinear covariance (x_T, y_T block: u = z x t / |z x t|, v = t x u),
//     projected along the track direction onto the plane's local axes;
//   - RectangularPlaneBounds::inside(p, err, -3).
// The curvilinear error transport from the start is pca::curvilinearJacobian + pca::similarity5 (PcaToBeamLine.h, the
// same code as the device PCA); the host check ([outconv DETTEST]) runs both the end-covariance-from-host and the full
// device chain.

#include <cmath>
#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/HelixCrossings.h"
#include "RecoTracker/MkFitAlpaka/interface/math/PcaToBeamLine.h"
#include "RecoTracker/MkFitAlpaka/interface/math/TkBfield.h"

namespace mkfitdev::navdev {

  struct HelixStart {
    double x[3];  // global position (cm)
    double p[3];  // global momentum (GeV)
    double rho;   // transverse curvature (1/cm), as FreeTrajectoryState::transverseCurvature()
  };
  struct DetPlane {
    double pos[3];
    double ax[3], ay[3], az[3];  // local x, y, z (= normal) axes in global coordinates
    double halfWidth, halfLength, halfThickness;
  };
  struct PlaneHit {
    bool valid = false;  // a crossing in the propagation direction within maxDPhi
    double s = 0;        // 3D path length (signed)
    double x[3]{}, t[3]{};
    double lx = 0, ly = 0, lz = 0;  // local position
  };

  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  // the rounding guard of boundsMarginal, also on the two other state-dependent decisions of
  // the det test (the sagitta pre-check and the maxDPhi window); a marginal test sends its search to the host
  constexpr double kNavMarginAbs = 2.e-4;      // cm
  constexpr double kNavMarginRelSag = 1.e-4;   // of the sagitta threshold / tolerance (state-only: no covariance)
  constexpr float kNavMarginRelDPhi = 1.e-4f;  // of the maxDPhi limit (squared)
  // |m| (signed margins of an "inside" decision) within tolerance of an edge: neither surely in nor surely out
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool marginsMarginal(
      double mz, double mx, double my, double tz, double tx, double ty) {
    const bool surelyIn = mz > tz && mx > tx && my > ty;
    const bool surelyOut = mz < -tz || mx < -tx || my < -ty;
    return !surelyIn && !surelyOut;
  }

  // sagitta pre-check of GeomDetCompatibilityChecker::isCompatible: false = rejected without propagation;
  // marginal |= a decision within rounding of its threshold
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool sagittaCheck(
      HelixStart const& st, DetPlane const& pl, bool along, double maxSag, double minTol2, bool& marginal) {
    if (maxSag <= 0)
      return true;
    double d[3] = {st.x[0] - pl.pos[0], st.x[1] - pl.pos[1], st.x[2] - pl.pos[2]};
    const double pz = dot3(pl.az, st.p);
    const double dS = -dot3(pl.az, d) / pz;
    if (std::abs(dS) * std::sqrt(dot3(st.p, st.p)) < kNavMarginAbs)
      marginal = true;  // the straight-line crossing is at the start: its direction decides
    if ((along && dS < 0) || (!along && dS > 0) || !(std::abs(dS) < std::numeric_limits<double>::infinity()))
      return true;  // no straight-line crossing: the host check does not reject
    const double g[3] = {st.x[0] + dS * st.p[0], st.x[1] + dS * st.p[1], st.x[2] + dS * st.p[2]};
    const double tpath2 = (g[0] - st.x[0]) * (g[0] - st.x[0]) + (g[1] - st.x[1]) * (g[1] - st.x[1]);
    const double sagitta = 0.5 * std::abs(tpath2 * st.rho);
    if (std::abs(sagitta - maxSag) <= kNavMarginAbs + kNavMarginRelSag * maxSag)
      marginal = true;
    if (!(sagitta < maxSag))
      return true;
    const double tol = std::sqrt(sagitta * sagitta > minTol2 ? sagitta * sagitta : minTol2);
    const double r[3] = {g[0] - pl.pos[0], g[1] - pl.pos[1], g[2] - pl.pos[2]};
    const double lx = dot3(pl.ax, r), ly = dot3(pl.ay, r), lz = dot3(pl.az, r);
    {  // the widened bounds below (in0 implies them: tol > 0)
      const double e = kNavMarginAbs + kNavMarginRelSag * tol;
      if (marginsMarginal(pl.halfThickness - std::abs(lz),
                          pl.halfWidth + tol - std::abs(lx),
                          pl.halfLength + tol - std::abs(ly),
                          kNavMarginAbs,
                          e,
                          e))
        marginal = true;
    }
    // RectangularPlaneBounds::inside(p, err, 1): inside(p) or |z| < ht && |x| < hw + tol && |y| < hl + tol
    const bool in0 = std::abs(lz) < pl.halfThickness && std::abs(lx) < pl.halfWidth && std::abs(ly) < pl.halfLength;
    return in0 ||
           (std::abs(lz) < pl.halfThickness && std::abs(lx) < pl.halfWidth + tol && std::abs(ly) < pl.halfLength + tol);
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool sagittaOk(
      HelixStart const& st, DetPlane const& pl, bool along, double maxSag, double minTol2) {
    bool marginal = false;
    return sagittaCheck(st, pl, along, maxSag, minTol2, marginal);
  }

  // the helix to the plane (AnalyticalPropagator::propagateWithPath on a plane, error-free part)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE PlaneHit helixToPlane(HelixStart const& st,
                                                               DetPlane const& pl,
                                                               bool along,
                                                               double maxDPhi) {
    PlaneHit h;
    const double pmag = std::sqrt(dot3(st.p, st.p));
    const double pt = std::sqrt(st.p[0] * st.p[0] + st.p[1] * st.p[1]);
    const double sinT = pt / pmag, cosT = st.p[2] / pmag;
    const double phi0 = std::atan2(st.p[1], st.p[0]);
    const double rho = st.rho;
    auto pos = [&](double s, double* x, double* t) MKFITDEV_NAV_LAMBDA_INLINE {
      const double sT = s * sinT;
      const double ph = phi0 + rho * sT;
      const double sp = std::sin(ph), cp = std::cos(ph);
      if (std::abs(rho * sT) < 1e-9) {
        x[0] = st.x[0] + sT * std::cos(phi0);
        x[1] = st.x[1] + sT * std::sin(phi0);
      } else {
        x[0] = st.x[0] + (sp - std::sin(phi0)) / rho;
        x[1] = st.x[1] - (cp - std::cos(phi0)) / rho;
      }
      x[2] = st.x[2] + s * cosT;
      t[0] = cp * sinT;
      t[1] = sp * sinT;
      t[2] = cosT;
    };
    // straight-line start
    const double t0[3] = {st.p[0] / pmag, st.p[1] / pmag, st.p[2] / pmag};
    const double d0[3] = {pl.pos[0] - st.x[0], pl.pos[1] - st.x[1], pl.pos[2] - st.x[2]};
    const double nt0 = dot3(pl.az, t0);
    if (nt0 == 0)
      return h;
    double s = dot3(pl.az, d0) / nt0;
    double x[3], t[3];
    bool conv = false;
    for (int it = 0; it < 30; ++it) {
      pos(s, x, t);
      const double r[3] = {x[0] - pl.pos[0], x[1] - pl.pos[1], x[2] - pl.pos[2]};
      const double f = dot3(pl.az, r), fp = dot3(pl.az, t);
      if (fp == 0)
        break;
      const double ds = f / fp;
      s -= ds;
      if (std::abs(ds) < 1e-10) {
        conv = true;
        break;
      }
    }
    if (!conv || (along && s < 0) || (!along && s > 0))
      return h;
    pos(s, x, t);
    if (std::abs(rho * s * sinT) > maxDPhi)
      return h;
    h.valid = true;
    h.s = s;
    for (int i = 0; i < 3; ++i) {
      h.x[i] = x[i];
      h.t[i] = t[i];
    }
    const double r[3] = {x[0] - pl.pos[0], x[1] - pl.pos[1], x[2] - pl.pos[2]};
    h.lx = dot3(pl.ax, r);
    h.ly = dot3(pl.ay, r);
    h.lz = dot3(pl.az, r);
    return h;
  }

  // local (x, y) position variances on the plane from the curvilinear covariance block of (x_T, y_T) at the crossing
  // (cTT = [C33, C34, C44]), the track direction t at the crossing and the plane axes
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void localPositionErrors(
      DetPlane const& pl, const double* t, const double* cTT, double& sxx, double& syy) {
    const double tT = std::sqrt(t[0] * t[0] + t[1] * t[1]);
    const double u[3] = {-t[1] / tT, t[0] / tT, 0.};
    const double v[3] = {t[1] * u[2] - t[2] * u[1], t[2] * u[0] - t[0] * u[2], t[0] * u[1] - t[1] * u[0]};
    const double nt = dot3(pl.az, t);
    const double nu = dot3(pl.az, u), nv = dot3(pl.az, v);
    // shift by a u + b v, back to the plane along t
    const double jxu = dot3(pl.ax, u) - nu * dot3(pl.ax, t) / nt, jxv = dot3(pl.ax, v) - nv * dot3(pl.ax, t) / nt;
    const double jyu = dot3(pl.ay, u) - nu * dot3(pl.ay, t) / nt, jyv = dot3(pl.ay, v) - nv * dot3(pl.ay, t) / nt;
    sxx = jxu * jxu * cTT[0] + 2 * jxu * jxv * cTT[1] + jxv * jxv * cTT[2];
    syy = jyu * jyu * cTT[0] + 2 * jyu * jyv * cTT[1] + jyv * jyv * cTT[2];
  }

  // RectangularPlaneBounds::inside(p, err, nSigma < 0)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool insideShrunk(
      DetPlane const& pl, PlaneHit const& h, double sxx, double syy, double nSigma) {
    return std::abs(h.lz) < pl.halfThickness && std::abs(h.lx) < pl.halfWidth + std::sqrt(sxx) * nSigma &&
           std::abs(h.ly) < pl.halfLength + std::sqrt(syy) * nSigma;
  }

  // a bounds decision within rounding of its edges. The start state's covariance is not bitwise
  // the host's (MkFitCore is -Ofast: local errors agree to 2.9e-4 relative, crossings to
  // 0.23 um), so a test whose margins are inside these tolerances (x 10) is not decided here: the search goes to the host.
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool boundsMarginal(
      DetPlane const& pl, PlaneHit const& h, double sxx, double syy, double nSigma) {
    constexpr double kAbs = 2.e-4;  // cm
    constexpr double kRel = 3.e-3;  // of the n-sigma error term
    const double ex = std::sqrt(sxx) * nSigma, ey = std::sqrt(syy) * nSigma;
    const double mz = pl.halfThickness - std::abs(h.lz), mx = pl.halfWidth + ex - std::abs(h.lx),
                 my = pl.halfLength + ey - std::abs(h.ly);
    const double tz = kAbs, tx = kAbs + kRel * std::abs(ex), ty = kAbs + kRel * std::abs(ey);
    const bool surelyIn = mz > tz && mx > tx && my > ty;
    const bool surelyOut = mz < -tz || mx < -tx || my < -ty;
    return !surelyIn && !surelyOut;
  }

  // ---- the device start state and the whole per-det test as one function ----
  // The navigation start state as the device has it: the fit row (x, y, z, 1/pT, phi, theta + packed errors) through
  // pca::ccsToCurvilinear (= the converter's FreeTrajectoryState), the field at the start (TkBfield.h) and the
  // transverse curvature as GlobalTrajectoryParameters::transverseCurvature (float).
  struct DetTestStart {
    ::mkfitdev::pca::TrackIn t;  // x, p (float), charge, curvilinear covariance (double)
    ::mkfitdev::pca::F3 b;       // field at t.x (Tesla)
    double rho;                  // transverse curvature (1/cm)
  };
  // false: the start is outside the closed-form field volume (the caller routes the track to the host)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool startFromFitRow(float const* par,
                                                              float const* err,
                                                              int charge,
                                                              DetTestStart& s) {
    namespace pc = ::mkfitdev::pca;
    pc::ccsToCurvilinear(par, err, charge, s.t);
    if (!::mkfitdev::field::tkBfieldValid(s.t.x.x, s.t.x.y, s.t.x.z))
      return false;
    ::mkfitdev::field::tkBfield(s.t.x.x, s.t.x.y, s.t.x.z, s.b.x, s.b.y, s.b.z);
    s.rho = -2.99792458e-3f * (float(charge) / pc::perp(s.t.p)) * s.b.z;
    return true;
  }

  // AnalyticalPropagator::propagateWithPath to a plane: OptimalHelixPlaneCrossing picks HelixBarrelPlaneCrossingByCircle
  // (|n_z| < 1e-6), HelixForwardPlaneCrossing (|n_x|, |n_y| < 1e-6) or HelixArbitraryPlaneCrossing (all three ported in
  // HelixCrossings.h); then the float maxDPhi limit and the direction normalised to the momentum
  // marginal |= the maxDPhi decision within rounding of its limit
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE PlaneHit planeCrossing(DetTestStart const& s,
                                                                DetPlane const& pl,
                                                                bool along,
                                                                bool* marginal = nullptr) {
    namespace pc = ::mkfitdev::pca;
    const float nx = pl.az[0], ny = pl.az[1], nz = pl.az[2];
    const bool barrel = std::abs(nz) < 1.e-6f;
    const bool forward = !barrel && std::abs(nx) < 1.e-6f && std::abs(ny) < 1.e-6f;
    const pc::F3 pp{float(pl.pos[0]), float(pl.pos[1]), float(pl.pos[2])};
    const Crossing x = barrel    ? barrelPlaneCrossing(s.t.x, s.t.p, float(s.rho), along, pp, pc::F3{nx, ny, nz})
                       : forward ? forwardPlaneCrossing(s.t.x, s.t.p, float(s.rho), along, float(pl.pos[2]))
                                 : arbitraryPlaneCrossing(s.t.x, s.t.p, float(s.rho), along, pp, pc::F3{nx, ny, nz});
    PlaneHit h;
    if (x.status != kXingOk)
      return h;
    float dphi2 = float(x.s) * float(s.rho);
    dphi2 = dphi2 * dphi2 * (s.t.p.x * s.t.p.x + s.t.p.y * s.t.p.y);
    const float dphiLim2 = 1.6f * 1.6f * (s.t.p.x * s.t.p.x + s.t.p.y * s.t.p.y + s.t.p.z * s.t.p.z);
    if (marginal && std::abs(dphi2 - dphiLim2) <= kNavMarginRelDPhi * dphiLim2)
      *marginal = true;
    if (dphi2 > dphiLim2)
      return h;
    h.valid = true;
    h.s = x.s;
    h.x[0] = x.x;
    h.x[1] = x.y;
    h.x[2] = x.z;
    const pc::F3 t = pc::unit(pc::F3{x.dx, x.dy, x.dz});
    h.t[0] = t.x;
    h.t[1] = t.y;
    h.t[2] = t.z;
    const double r[3] = {h.x[0] - pl.pos[0], h.x[1] - pl.pos[1], h.x[2] - pl.pos[2]};
    h.lx = dot3(pl.ax, r);
    h.ly = dot3(pl.ay, r);
    h.lz = dot3(pl.az, r);
    return h;
  }

  struct DetTestResult {
    bool sagOk = true;        // false: rejected by the sagitta pre-check (no propagation)
    bool crossed = false;     // a crossing in the propagation direction within maxDPhi
    bool compatible = false;  // the -3 sigma bounds test passed (GeomDetCompatibilityChecker::isCompatible)
    bool marginal = false;    // the bounds decision is within rounding of an edge (boundsMarginal)
    double lx = 0, ly = 0, sxx = 0, syy = 0;
    float gx = 0, gy = 0, gz = 0;   // the crossing (GlobalPoint of the propagated state)
    double tx = 0, ty = 0, tz = 0;  // direction at the crossing
  };

  // GeomDetCompatibilityChecker::isCompatible with the analytic propagator and Chi2MeasurementEstimator(nSigma) for
  // one rectangular det plane: sagitta pre-check, helix to the plane, the (x_T, y_T) curvilinear block transported by
  // pca::curvilinearJacobian (rows 3 and 4 of C' = J C J^T, same order as pca::similarity5), local errors, bounds.
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE DetTestResult
  detTest(DetTestStart const& s, DetPlane const& pl, bool along, double maxSag, double minTol2, double nSigma) {
    namespace pc = ::mkfitdev::pca;
    DetTestResult r;
    const HelixStart st{{s.t.x.x, s.t.x.y, s.t.x.z}, {s.t.p.x, s.t.p.y, s.t.p.z}, s.rho};
    r.sagOk = sagittaCheck(st, pl, along, maxSag, minTol2, r.marginal);
    if (!r.sagOk)
      return r;
    const PlaneHit h = planeCrossing(s, pl, along, &r.marginal);
    if (!h.valid)
      return r;
    r.crossed = true;
    r.gx = float(h.x[0]);
    r.gy = float(h.x[1]);
    r.gz = float(h.x[2]);
    r.tx = h.t[0];
    r.ty = h.t[1];
    r.tz = h.t[2];
    const float pm = pc::mag(s.t.p);
    const pc::F3 x2{float(h.x[0]), float(h.x[1]), float(h.x[2])};
    const pc::F3 p2{float(h.t[0] * pm), float(h.t[1] * pm), float(h.t[2] * pm)};
    double J[5][5];
    pc::curvilinearJacobian(s.t.x, s.t.p, s.t.charge, x2, p2, h.s, s.b, J);
    double JC[2][5];
    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < 5; ++k) {
        double a = 0;
        for (int l = 0; l < 5; ++l)
          a += J[3 + i][l] * s.t.C[l][k];
        JC[i][k] = a;
      }
    double cTT[3];
    for (int i = 0, n = 0; i < 2; ++i)
      for (int j = 0; j <= i; ++j) {
        double a = 0;
        for (int k = 0; k < 5; ++k)
          a += JC[i][k] * J[3 + j][k];
        cTT[n++] = a;  // C33, C43 (= C34), C44
      }
    localPositionErrors(pl, h.t, cTT, r.sxx, r.syy);
    r.lx = h.lx;
    r.ly = h.ly;
    r.compatible = insideShrunk(pl, h, r.sxx, r.syy, nSigma);
    r.marginal = r.marginal || boundsMarginal(pl, h, r.sxx, r.syy, nSigma);
    return r;
  }

}  // namespace mkfitdev::navdev

#endif
