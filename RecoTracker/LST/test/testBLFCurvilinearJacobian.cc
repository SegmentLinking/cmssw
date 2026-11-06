// Unit test of lst::blfToCurvilinearJacobian and lst::blfToCurvilinearCovariance.
//
// The reference is a finite-difference (FD) derivative of the full map from (phi_e, x_T, kappa, cot, z_e) at the
// state point to the curvilinear parameters (q/p, lambda, phi, x_T, y_T) on the plane through the nominal point,
// perpendicular to the nominal momentum. Each input is perturbed in turn: z_e moves the point along z, x_T moves it
// along U = (-sin phi, cos phi, 0). The perturbed state is then stepped along its helix onto that plane.
//
// GlobalTrajectoryParameters and AnalyticalPropagator work in single precision, which cannot resolve a 1e-6 relative
// FD. The 1e-6 comparison therefore uses an exact helix in double precision, whose rotation sense is checked against
// GlobalTrajectoryParameters::transverseCurvature() and whose plane step is checked against AnalyticalPropagator.
// The same FD is also done with GlobalTrajectoryParameters and AnalyticalPropagator; that one is reported, and only
// the sign of the phi-z_e entry is asserted on it.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "MagneticField/UniformEngine/interface/UniformMagneticField.h"
#include "RecoTracker/LST/interface/BLFCurvilinearJacobian.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace {

  constexpr double kTolFD = 1e-6;      // relative, on every analytically non-zero entry
  constexpr double kTolZero = 1e-9;    // absolute (native units), on every analytically zero entry
  constexpr double kTolError = 1e-12;  // relative to sqrt(C_ii C_jj), covariance against an explicit J C J^T
  constexpr float kFieldTesla = 3.8f;

  // FD steps for the double-precision reference: small enough that the truncation error is below 1e-9 relative,
  // large enough that the rounding of the outputs (up to |phi| ~ pi, |x| ~ 300 cm) stays below 1e-9 relative.
  constexpr std::array<double, 5> kStep = {1e-5, 1e-4, 0. /* relative to kappa, below */, 1e-5, 1e-3};
  constexpr double kStepKappaRel = 1e-5;
  // FD steps for the single-precision cross-check (GlobalTrajectoryParameters + AnalyticalPropagator).
  constexpr std::array<double, 5> kStepFloat = {1e-3, 1e-2, 0., 1e-3, 0.5};
  constexpr double kStepKappaRelFloat = 1e-3;

  // generic positive-definite covariance blocks with every correlation non-zero
  constexpr float kCovCircle[6] = {4e-6f, 1e-7f, 1e-5f, 2e-8f, -3e-9f, 1e-9f};
  constexpr float kCovLine[3] = {1e-5f, -2e-6f, 4e-3f};

  double wrapPhi(double dphi) { return std::remainder(dphi, 2. * M_PI); }

  struct Inputs {
    double phi, xT, kappa, cot, z;
  };

  struct Nominal {
    double x0, y0;  // transverse state point
    Inputs in;
    int q;
    double bField;
    // nominal curvilinear frame at the nominal point
    std::array<double, 3> p0, t0, u0, v0;
  };

  Nominal makeNominal(double x0, double y0, Inputs in, int q, double bField) {
    Nominal n{x0, y0, in, q, bField, {}, {}, {}, {}};
    const double cl = 1. / std::sqrt(1. + in.cot * in.cot), sl = in.cot * cl;
    const double cp = std::cos(in.phi), sp = std::sin(in.phi);
    n.p0 = {x0 + in.xT * -sp, y0 + in.xT * cp, in.z};
    n.t0 = {cl * cp, cl * sp, sl};
    n.u0 = {-sp, cp, 0.};
    n.v0 = {-sl * cp, -sl * sp, cl};
    return n;
  }

  double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  // curvilinear parameters (q/p, lambda, phi, x_T, y_T), positions relative to the nominal point in its frame
  using Curv = std::array<double, 5>;

  // Exact helix in a uniform field along +z, in double precision: point (x0, y0, z) + x_T * U(phi_nominal),
  // direction (phi, cot), pT = B/|kappa|, charge q. Its rotation rate dphi/ds_T = -q * B / pT is CMSSW's
  // transverseCurvature() (checked in the test).
  Curv curvilinearDouble(const Nominal& n, const Inputs& in) {
    const double cpn = std::cos(n.in.phi), spn = std::sin(n.in.phi);
    const std::array<double, 3> r0 = {n.x0 + in.xT * -spn, n.y0 + in.xT * cpn, in.z};
    const double cl = 1. / std::sqrt(1. + in.cot * in.cot), sl = in.cot * cl;
    const double pt = n.bField / std::abs(in.kappa);
    const double w = -n.q * n.bField / pt;  // dphi/ds_T
    auto pos = [&](double s) -> std::array<double, 3> {
      const double ph = in.phi + w * cl * s;
      return {
          r0[0] + (std::sin(ph) - std::sin(in.phi)) / w, r0[1] - (std::cos(ph) - std::cos(in.phi)) / w, r0[2] + sl * s};
    };
    auto dir = [&](double s) -> std::array<double, 3> {
      const double ph = in.phi + w * cl * s;
      return {cl * std::cos(ph), cl * std::sin(ph), sl};
    };
    // Newton on f(s) = (r(s) - p0) . t0
    double s = 0.;
    for (int it = 0; it < 50; ++it) {
      const auto r = pos(s);
      const std::array<double, 3> d = {r[0] - n.p0[0], r[1] - n.p0[1], r[2] - n.p0[2]};
      const double f = dot(d, n.t0);
      const double ds = -f / dot(dir(s), n.t0);
      s += ds;
      if (std::abs(ds) < 1e-15)
        break;
    }
    const auto r = pos(s);
    const std::array<double, 3> d = {r[0] - n.p0[0], r[1] - n.p0[1], r[2] - n.p0[2]};
    return {n.q * cl / pt, std::atan(in.cot), in.phi + w * cl * s, dot(d, n.u0), dot(d, n.v0)};
  }

  AlgebraicMatrix55 jacobianMatrix(double kappa, double cot, float bField) {
    double jacobian[5][5];
    lst::blfToCurvilinearJacobian(kappa, cot, bField, jacobian);
    AlgebraicMatrix55 result;
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        result(i, j) = jacobian[i][j];
    return result;
  }

  // the free state with the curvilinear error of the stored (float) values, as a CMSSW consumer builds it
  FreeTrajectoryState freeState(const Nominal& n, const Inputs& in, float bField, const MagneticField& field) {
    const double spn = std::sin(n.in.phi), cpn = std::cos(n.in.phi);
    const float x = n.x0 + in.xT * -spn, y = n.y0 + in.xT * cpn, z = in.z;
    const float phi = in.phi, cot = in.cot, kappa = in.kappa;
    const double pt = bField / std::abs(double(kappa));
    double covariance[5][5];
    lst::blfToCurvilinearCovariance(kappa, cot, bField, kCovCircle, kCovLine, covariance);
    AlgebraicSymMatrix55 error;
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j <= i; ++j)
        error(i, j) = covariance[i][j];
    return FreeTrajectoryState(
        GlobalTrajectoryParameters(
            GlobalPoint(x, y, z), GlobalVector(pt * std::cos(phi), pt * std::sin(phi), pt * cot), n.q, &field),
        CurvilinearTrajectoryError(error));
  }

  // the same map with GlobalTrajectoryParameters and AnalyticalPropagator (single precision)
  bool curvilinearFloat(const Nominal& n,
                        const Inputs& in,
                        const MagneticField& field,
                        const Propagator& prop,
                        const Plane& plane,
                        Curv& out) {
    const auto tsos = prop.propagate(freeState(n, in, n.bField, field), plane);
    if (!tsos.isValid())
      return false;
    const GlobalPoint gp = tsos.globalPosition();
    const GlobalVector gv = tsos.globalMomentum();
    const std::array<double, 3> d = {gp.x() - n.p0[0], gp.y() - n.p0[1], gp.z() - n.p0[2]};
    out = {tsos.charge() / gv.mag(), std::atan2(gv.z(), gv.perp()), gv.phi(), dot(d, n.u0), dot(d, n.v0)};
    return true;
  }

  Inputs perturbed(const Inputs& in, int j, double h) {
    Inputs p = in;
    switch (j) {
      case 0:
        p.phi += h;
        break;
      case 1:
        p.xT += h;
        break;
      case 2:
        p.kappa += h;
        break;
      case 3:
        p.cot += h;
        break;
      default:
        p.z += h;
        break;
    }
    return p;
  }

  template <typename F>
  AlgebraicMatrix55 finiteDifference(F&& f, const Inputs& in, const std::array<double, 5>& steps, bool& ok) {
    AlgebraicMatrix55 jfd;
    ok = true;
    for (int j = 0; j < 5; ++j) {
      Curv plus, minus;
      ok = ok && f(perturbed(in, j, steps[j]), plus) && f(perturbed(in, j, -steps[j]), minus);
      if (!ok)
        return jfd;
      for (int i = 0; i < 5; ++i) {
        const double diff = (i == 2) ? wrapPhi(plus[i] - minus[i]) : plus[i] - minus[i];
        jfd(i, j) = diff / (2. * steps[j]);
      }
    }
    return jfd;
  }

  // a state point: transverse radius and z
  struct Point {
    double radius, z;
  };

}  // namespace

TEST_CASE("BLF state to curvilinear: Jacobian against finite differences", "[BLFCurvilinearJacobian]") {
  const UniformMagneticField field(kFieldTesla);
  // B as the fit takes it; the double-precision reference helix uses the same value
  const float bField = static_cast<float>(1. / field.inverseBzAtOriginInGeV());
  const AnalyticalPropagator analytic(&field, anyDirection);
  const Propagator& prop = analytic;

  const std::vector<double> pts = {0.5, 0.8, 2., 5., 20., 100.};
  const std::vector<double> cots = {-8., -2., -0.5, 0., 0.5, 2., 8.};
  const std::vector<double> phis = {-M_PI + 1e-6, -2.0, -0.3, 1.1, 2.6, M_PI - 1e-6};
  // barrel-like and endcap-like points
  const std::vector<Point> points = {{80., 35.}, {25., 160.}, {60., -250.}};

  std::array<std::array<double, 5>, 5> maxRel{}, maxAbsZero{}, maxRelFloat{};
  double maxErrRel = 0.;
  double sxMin = 1e9, sxMax = -1e9;
  double maxCurvRel = 0., maxPlaneStepPos = 0., maxPlaneStepDir = 0.;
  int nCases = 0, nGenericCount = 0, nFloatSignChecked = 0;
  int minNonZero = 99, maxNonZero = -1;

  for (const Point& point : points) {
    for (int q : {-1, 1}) {
      for (double pt : pts) {
        for (double cot : cots) {
          for (double phi : phis) {
            ++nCases;
            const double kappa = q * bField / pt;
            // the state point at an azimuth a little behind the momentum
            const Inputs in{phi, 0., kappa, cot, point.z};
            const Nominal n =
                makeNominal(point.radius * std::cos(phi - 0.2), point.radius * std::sin(phi - 0.2), in, q, bField);
            CAPTURE(point.radius, point.z, q, pt, cot, phi);

            // the state as a CMSSW consumer builds it: position, momentum, charge, and CMSSW's rotation sense
            const FreeTrajectoryState fts = freeState(n, in, bField, field);
            CHECK(fts.charge() == q);
            CHECK(std::abs(fts.momentum().perp() / pt - 1.) < 1e-6);
            const double wCMSSW = fts.transverseCurvature();
            const double wModel = -q * bField / pt;
            maxCurvRel = std::max(maxCurvRel, std::abs(wCMSSW / wModel - 1.));
            CHECK(std::abs(wCMSSW / wModel - 1.) < 1e-5);
            // transverseCurvature() = -kappa: kappa has the sign of the charge
            CHECK(wCMSSW * kappa < 0.);

            // the double-precision helix against AnalyticalPropagator, over a 20 cm step to a plane perpendicular to
            // the nominal direction
            {
              const std::array<double, 3> far = {
                  n.p0[0] + 20. * n.t0[0], n.p0[1] + 20. * n.t0[1], n.p0[2] + 20. * n.t0[2]};
              Nominal nFar = n;
              nFar.p0 = far;
              const Curv ref = curvilinearDouble(nFar, in);
              const auto farPlane =
                  Plane::build(GlobalPoint(far[0], far[1], far[2]),
                               Surface::RotationType(
                                   n.u0[0], n.u0[1], n.u0[2], n.v0[0], n.v0[1], n.v0[2], n.t0[0], n.t0[1], n.t0[2]));
              Curv flt;
              REQUIRE(curvilinearFloat(nFar, in, field, prop, *farPlane, flt));
              maxPlaneStepPos = std::max({maxPlaneStepPos, std::abs(flt[3] - ref[3]), std::abs(flt[4] - ref[4])});
              maxPlaneStepDir = std::max(maxPlaneStepDir, std::abs(wrapPhi(flt[2] - ref[2])));
              CHECK(std::abs(flt[3] - ref[3]) < 1e-3);
              CHECK(std::abs(flt[4] - ref[4]) < 1e-3);
              CHECK(std::abs(wrapPhi(flt[2] - ref[2])) < 1e-5);
            }

            // double-precision FD of the full map
            const AlgebraicMatrix55 jac = jacobianMatrix(kappa, cot, bField);
            std::array<double, 5> steps = kStep;
            steps[2] = kStepKappaRel * std::abs(kappa);
            bool ok = false;
            const AlgebraicMatrix55 jfd = finiteDifference(
                [&](const Inputs& p, Curv& out) {
                  out = curvilinearDouble(n, p);
                  return true;
                },
                in,
                steps,
                ok);
            REQUIRE(ok);
            for (int i = 0; i < 5; ++i) {
              for (int j = 0; j < 5; ++j) {
                CAPTURE(i, j, jac(i, j), jfd(i, j));
                if (jac(i, j) != 0.) {
                  const double rel = std::abs(jfd(i, j) - jac(i, j)) / std::abs(jac(i, j));
                  maxRel[i][j] = std::max(maxRel[i][j], rel);
                  CHECK(rel < kTolFD);
                } else {
                  maxAbsZero[i][j] = std::max(maxAbsZero[i][j], std::abs(jfd(i, j)));
                  CHECK(std::abs(jfd(i, j)) < kTolZero);
                }
              }
            }
            sxMin = std::min(sxMin, jfd(3, 1));
            sxMax = std::max(sxMax, jfd(3, 1));

            // single-precision FD with GlobalTrajectoryParameters and AnalyticalPropagator;
            // phi-z_e sign is asserted where the entry is resolvable (|J| * step above the float resolution of phi)
            {
              const auto plane =
                  Plane::build(GlobalPoint(n.p0[0], n.p0[1], n.p0[2]),
                               Surface::RotationType(
                                   n.u0[0], n.u0[1], n.u0[2], n.v0[0], n.v0[1], n.v0[2], n.t0[0], n.t0[1], n.t0[2]));
              std::array<double, 5> stepsF = kStepFloat;
              stepsF[2] = kStepKappaRelFloat * std::abs(kappa);
              bool okF = false;
              const AlgebraicMatrix55 jfdF = finiteDifference(
                  [&](const Inputs& p, Curv& out) { return curvilinearFloat(n, p, field, prop, *plane, out); },
                  in,
                  stepsF,
                  okF);
              REQUIRE(okF);
              for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                  if (jac(i, j) != 0.)
                    maxRelFloat[i][j] =
                        std::max(maxRelFloat[i][j], std::abs(jfdF(i, j) - jac(i, j)) / std::abs(jac(i, j)));
              if (std::abs(jac(2, 4)) * stepsF[4] > 1e-5) {
                ++nFloatSignChecked;
                CHECK(jfdF(2, 4) * jac(2, 4) > 0.);
              }
            }

            // the curvilinear covariance is J C J^T, with C block diagonal
            {
              AlgebraicSymMatrix55 cin;
              cin(0, 0) = kCovCircle[0];
              cin(0, 1) = kCovCircle[1];
              cin(1, 1) = kCovCircle[2];
              cin(0, 2) = kCovCircle[3];
              cin(1, 2) = kCovCircle[4];
              cin(2, 2) = kCovCircle[5];
              cin(3, 3) = kCovLine[0];
              cin(3, 4) = kCovLine[1];
              cin(4, 4) = kCovLine[2];
              const float kappaStored = kappa, cotStored = cot;
              const AlgebraicSymMatrix55 expected =
                  ROOT::Math::Similarity(jacobianMatrix(kappaStored, cotStored, bField), cin);
              double got[5][5];
              lst::blfToCurvilinearCovariance(kappaStored, cotStored, bField, kCovCircle, kCovLine, got);
              int nonZero = 0;
              for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                  CHECK(got[i][j] == got[j][i]);
                  if (j < i)
                    continue;
                  if (got[i][j] != 0.)
                    ++nonZero;
                  const double scale = std::sqrt(expected(i, i) * expected(j, j));
                  const double rel = std::abs(got[i][j] - expected(i, j)) / scale;
                  maxErrRel = std::max(maxErrRel, rel);
                  CHECK(rel < kTolError);
                }
              }
              // cov(lambda, x_T) and cov(x_T, y_T) vanish identically
              CHECK(got[1][3] == 0.);
              CHECK(got[3][4] == 0.);
              if (cot != 0.) {
                ++nGenericCount;
                minNonZero = std::min(minNonZero, nonZero);
                maxNonZero = std::max(maxNonZero, nonZero);
                CHECK(nonZero == 13);
              }
            }
          }
        }
      }
    }
  }

  // the phi-z_e entry at pT = 1 GeV, cot = 1, both charges
  std::array<double, 2> phiZFD{}, phiZAnalytic{};
  for (int iq = 0; iq < 2; ++iq) {
    const int q = iq == 0 ? -1 : 1;
    const double kappa = q * bField / 1.;
    const Inputs in{0.4, 0., kappa, 1., 35.};
    const Nominal n = makeNominal(80. * std::cos(0.2), 80. * std::sin(0.2), in, q, bField);
    const double h = kStep[4];
    const double fd =
        wrapPhi(curvilinearDouble(n, perturbed(in, 4, h))[2] - curvilinearDouble(n, perturbed(in, 4, -h))[2]) /
        (2. * h);
    phiZFD[iq] = fd;
    phiZAnalytic[iq] = jacobianMatrix(kappa, 1., bField)(2, 4);
    CHECK(std::abs(fd / phiZAnalytic[iq] - 1.) < kTolFD);
  }

  const char* rows[5] = {"q/p   ", "lambda", "phi   ", "x_T   ", "y_T   "};
  std::cout << "BLF state to curvilinear, " << nCases
            << " cases (points (r, z) = (80, 35), (25, 160), (60, -250) cm; q = +-1; pT 0.5, 0.8, 2, 5, 20, 100 GeV; "
            << "cot -8, -2, -0.5, 0, 0.5, 2, 8; phi -pi+1e-6, -2, -0.3, 1.1, 2.6, pi-1e-6), B = " << bField
            << " GeV/cm\n"
            << "  max rel. FD deviation (double-precision helix) per entry, columns (phi_e, x_T, kappa, cot, z_e); "
            << "'-' = analytically zero; tolerance " << kTolFD << "\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << "    " << rows[i] << ":";
    for (int j = 0; j < 5; ++j) {
      if (maxRel[i][j] > 0.)
        std::cout << ' ' << maxRel[i][j];
      else
        std::cout << " -";
    }
    std::cout << '\n';
  }
  std::cout << "  max |FD| on analytically zero entries (native units), same layout; tolerance " << kTolZero << "\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << "    " << rows[i] << ":";
    for (int j = 0; j < 5; ++j)
      std::cout << ' ' << maxAbsZero[i][j];
    std::cout << '\n';
  }
  std::cout << "  information only: max rel. FD deviation with GlobalTrajectoryParameters and AnalyticalPropagator "
               "(float)\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << "    " << rows[i] << ":";
    for (int j = 0; j < 5; ++j) {
      if (maxRelFloat[i][j] > 0.)
        std::cout << ' ' << maxRelFloat[i][j];
      else
        std::cout << " -";
    }
    std::cout << '\n';
  }
  std::cout << "  phi-z_e at pT = 1 GeV, cot = 1: FD " << phiZFD[0] << " (q = -1), " << phiZFD[1]
            << " (q = +1) rad/cm; analytic +kappa*sl*cl " << phiZAnalytic[0] << ", " << phiZAnalytic[1]
            << "; float-FD sign agreement asserted in " << nFloatSignChecked << " cases\n"
            << "  sx = d x_T / d x_T by FD: " << sxMin << " - " << sxMax << '\n'
            << "  max |covariance - explicit J C J^T| / sqrt(C_ii C_jj): " << maxErrRel << " (tolerance " << kTolError
            << ")\n"
            << "  non-zero curvilinear covariance elements (of 15), cot != 0: " << minNonZero << " - " << maxNonZero
            << " over " << nGenericCount << " cases\n"
            << "  |transverseCurvature / (-q B / pT) - 1| max " << maxCurvRel
            << "; double helix vs AnalyticalPropagator over 20 cm: max |d x_T|, |d y_T| " << maxPlaneStepPos
            << " cm, max |d phi| " << maxPlaneStepDir << " rad" << std::endl;
}
