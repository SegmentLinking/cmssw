// Unit test of brokenline::circleFitOuter and brokenline::lineFitOuter
//
// Built for the serial backend only: the kernel body generates the synthetic helices with std:: containers and
// random engines, so it runs as host code. The functions under test are pure arithmetic templates on TAcc.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include <Eigen/Cholesky>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace bl = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr double kBField = 0.0113921;  // GeV/cm/c, 3.8 T

  // Finite-difference steps: 1 um on the interceptions; 1e-3 of the pre-fitted curvature on u_n, large enough that
  // the rounding of phi (~1e-15 rad) is negligible against the change it produces, small enough that the
  // truncation error of the asin term stays below 1e-7.
  constexpr double kStepU = 1e-4;         // cm
  constexpr double kStepKappaRel = 1e-3;  // relative to 1/R
  constexpr double kTolFD = 1e-6;
  // Noiseless helices: the hits are exact to double precision, so the fast fit is the true circle, every fitted
  // offset is zero up to rounding, and the outer state must reproduce the truth up to rounding amplified by the
  // three-point fast fit. 1e-8 is far above that and far below any detector scale (hit resolution >= 1e-3 cm,
  // phi resolution >= 1e-5 rad).
  constexpr double kTolNoiseless = 1e-8;
  // The opposite bending sign moves phi_e by 2 asin(kappa L / 2) >= L / R: >= 6.8e-4 rad for the shortest chord
  // (1.2 cm) at the largest radius (20 GeV) used here.
  constexpr double kMinWrongSignShift = 1e-5;
  // The overloads of the outer fits that take their work array are run on a buffer in block-shared memory, as the fit
  // kernels pass it, sized for the largest N of the cases below.
  constexpr int kWorkMaxN = 14;
  constexpr int kWorkCapacity = (kWorkMaxN + 1) * (kWorkMaxN + 1);

  struct CaseResult {
    char label[192];
    int n;
    bool noiseless;
    int q, qFit;
    int expectedM, mCircle, mLine;
    bool expectFallback, fallbackCircle, fallbackLine;
    double chord;
    double relJCircle[3];  // max relative FD deviation per column (u_m, u_{n-1}, u_n), non-zero elements
    bool zeroElementsExact;
    double relCovCircle;
    double relJLine[2];  // max relative FD deviation per column (u_m, u_{n-1}), above the FD resolution
    double relCovLine;
    double maxAbsUnresolvedLine;  // max |analytic - FD| over line elements below the FD resolution
    bool lineWithinResolution;    // every line element agrees to kTolFD relative or within the FD resolution
    bool extractionMatchesCircle, extractionMatchesLine;  // within the rounding tolerances below
    bool extractionLineBitwise;                           // information only: l0 == ol bit for bit
    double extractionLineDevPar, extractionLineDevCov;    // information only: |l0 - ol| of the line state
    // largest |deviation| / tolerance of the comparisons with the upstream solve, and the largest tolerances
    double ratioCircle, ratioLine, ratioKappa, ratioUpCot, ratioUpZ;
    double tolPhi, tolPoint, tolKappa, tolCot, tolZ, tolUpCot, tolUpZ;
    bool boundsCircle, boundsLine;  // the Cholesky factorization of the rounding tolerances succeeded
    double rhatBU;
    double nlPhi, nlPos, nlKappaRel, nlCot, nlZ, wrongSignShift, nlCotUpstream;
    double upCot, upZ;  // |cot_out - cot_upstream| and |z_e - (zip_upstream + cot_upstream * s_e)|
    bool dataUnchanged, upstreamIdentical, kappaWithinRounding, workOverloadBitwise;
    bool kappaBitwise;  // information only
    // information only: largest relative deviation of the shared-memory from the stack work array
    double workDevCircle, workDevLine;
    bool orderDependent;
    double orderDPhi, orderDZ;
  };

  double relErr(double a, double b, double floor) {
    return std::abs(a - b) / std::max({std::abs(a), std::abs(b), floor, std::numeric_limits<double>::min()});
  }

  double wrapPhi(double d) { return std::remainder(d, 2 * M_PI); }

  // |deviation| / tolerance: 0 for identical values; infinite for a NaN deviation or tolerance, or a deviation where
  // the tolerance is 0, so that the ratio fails the check and survives std::max (which drops a NaN second argument)
  double devRatio(double dev, double tol) {
    if (dev == 0)
      return 0;
    if (std::isnan(dev) || !(tol > 0))
      return std::numeric_limits<double>::infinity();
    return dev / tol;
  }

  // Rounding tolerances for comparing two double evaluations of the Broken Line solve x = A^-1 b, with the inverse
  // from math::cholesky::invert() (or its in-place copy in the outer fits), that perform the same operations in the
  // same order but differ in which products the compiler fuses into multiply-adds (GCC contracts a * b + c into an FMA
  // depending on the surrounding code). Each such difference is one rounding more or less of one product, so both
  // evaluations stay within the rounding errors of the algorithm, and their difference within the sum of the two.
  // The tolerance is a first-order estimate of those errors, with derived (not fitted) constants. The factorization
  // has the componentwise backward error |dA| <= gamma_{d+1} |R^T| |R| (Higham, Accuracy and Stability of Numerical
  // Algorithms, 2nd ed., Theorem 10.3), which moves the inverse by |A^-1| |dA| |A^-1| to first order; the inversion of
  // R and the product of the two triangular inverses are counted as two more first-order terms of the same form,
  // gamma_d each (an estimate: their errors are not bounded entrywise in that form), and the product X b adds
  // gamma_d |X| |b|. With u = eps / 2 and two evaluations, elementwise:
  //   |dX| <= (3d + 1) eps |X| |L| |L^T| |X|,  |dx| <= |dX| |b| + d eps |X| |b|,
  // with L = R^T from Eigen's LLT of A and X the test's own inverse in place of A^-1. Returns false if the LLT fails.
  // The conditioning enters through |X| |L| |L^T| |X|: for the short or unevenly spaced layouts the elements of X
  // differ between two such evaluations by up to ~1e-7 relative, so no fixed number of ulp of X or x would do.
  template <typename M, typename V>
  bool solveRoundingBounds(const M& a, const M& inv, const V& b, M& tolInv, V& tolSol) {
    constexpr int d = M::ColsAtCompileTime;
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const Eigen::LLT<M> llt(a);
    const M l = llt.matrixL();
    const M absInv = inv.cwiseAbs();
    tolInv = (3 * d + 1) * eps * (absInv * (l.cwiseAbs() * l.cwiseAbs().transpose()) * absInv);
    tolSol = tolInv * b.cwiseAbs() + d * eps * (absInv * b.cwiseAbs());
    return llt.info() == Eigen::Success;
  }

  // helix at transverse arc length s; q > 0 bends clockwise for Bz > 0
  void helixAt(double s,
               double radius,
               int q,
               double phi0,
               double cot,
               double x0,
               double y0,
               double z0,
               double& x,
               double& y,
               double& z,
               double& phi) {
    phi = phi0 - q * s / radius;
    x = x0 + (radius / q) * (std::sin(phi0) - std::sin(phi));
    y = y0 + (radius / q) * (std::cos(phi) - std::cos(phi0));
    z = z0 + cot * s;
  }

  bool sameBits(const bl::OuterCircleState& a, const bl::OuterCircleState& b) {
    return !std::memcmp(a.point.data(), b.point.data(), sizeof(double) * 2) &&
           !std::memcmp(a.par.data(), b.par.data(), sizeof(double) * 3) &&
           !std::memcmp(a.cov.data(), b.cov.data(), sizeof(double) * 9) && a.qCharge == b.qCharge && a.mRef == b.mRef &&
           a.refFallback == b.refFallback;
  }

  bool sameBits(const bl::OuterLineState& a, const bl::OuterLineState& b) {
    return !std::memcmp(a.par.data(), b.par.data(), sizeof(double) * 2) &&
           !std::memcmp(a.cov.data(), b.cov.data(), sizeof(double) * 4) && a.mRef == b.mRef &&
           a.refFallback == b.refFallback;
  }

  // largest element-wise relative deviation of a from b over the point, par and cov (circle) or par and cov (line)
  double relDev(const bl::OuterCircleState& a, const bl::OuterCircleState& b) {
    double d = 0;
    for (int i = 0; i < 2; ++i)
      d = std::max(d, relErr(a.point(i), b.point(i), 0.));
    for (int i = 0; i < 3; ++i)
      d = std::max(d, relErr(a.par(i), b.par(i), 0.));
    for (int i = 0; i < 9; ++i)
      d = std::max(d, relErr(a.cov.data()[i], b.cov.data()[i], 0.));
    return d;
  }

  double relDev(const bl::OuterLineState& a, const bl::OuterLineState& b) {
    double d = 0;
    for (int i = 0; i < 2; ++i)
      d = std::max(d, relErr(a.par(i), b.par(i), 0.));
    for (int i = 0; i < 4; ++i)
      d = std::max(d, relErr(a.cov.data()[i], b.cov.data()[i], 0.));
    return d;
  }

  template <int N>
  bool sameData(const bl::PreparedBrokenLineData<N>& a, const bl::PreparedBrokenLineData<N>& b) {
    return a.qCharge == b.qCharge && !std::memcmp(a.radii.data(), b.radii.data(), sizeof(double) * 2 * N) &&
           !std::memcmp(a.sTransverse.data(), b.sTransverse.data(), sizeof(double) * N) &&
           !std::memcmp(a.sTotal.data(), b.sTotal.data(), sizeof(double) * N) &&
           !std::memcmp(a.zInSZplane.data(), b.zInSZplane.data(), sizeof(double) * N) &&
           !std::memcmp(a.varBeta.data(), b.varBeta.data(), sizeof(double) * N);
  }

  template <int N>
  __attribute__((noinline)) void upstreamFits(const Acc1D& acc,
                                              const ::riemannFit::Matrix3xNd<N>& hits,
                                              const Eigen::Matrix<float, 6, N>& hits_ge,
                                              const ::riemannFit::Vector4d& fast_fit,
                                              bl::PreparedBrokenLineData<N>& data,
                                              bl::karimaki_circle_fit& circle,
                                              ::riemannFit::LineFit& line) {
    bl::lineFit(acc, hits_ge, fast_fit, kBField, data, line);
    bl::circleFit(acc, hits, hits_ge, fast_fit, kBField, data, circle);
  }

  // The m.s. variance of the line reference chord for the toleranced checks, computed out of line so that the compiler
  // cannot merge it with the multScatt() call passed to outerLineExtraction() in runCase (see there).
  __attribute__((noinline)) double chordMultScatt(
      const Acc1D& acc, double length, double radius, int nHits, double slope) {
    return bl::multScatt(acc, length, kBField, radius, nHits, slope);
  }

  template <int N>
  CaseResult runCase(const Acc1D& acc,
                     const char* layout,
                     const std::vector<double>& radii,
                     int expectedM,
                     bool expectFallback,
                     double pt,
                     int q,
                     double phi0,
                     double cot,
                     bool noiseless,
                     std::mt19937& rng) {
    CaseResult res{};
    std::snprintf(res.label,
                  sizeof(res.label),
                  "layout=%s n=%d pt=%.1f q=%+d phi0=%.2f cot=%.2f %s",
                  layout,
                  N,
                  pt,
                  q,
                  phi0,
                  cot,
                  noiseless ? "noiseless" : "smeared");
    res.n = N;
    res.noiseless = noiseless;
    res.q = q;
    res.expectedM = expectedM;
    res.expectFallback = expectFallback;

    // ---- synthetic helix and hits
    const double radius = pt / kBField;
    const double d0 = 0.01, z0 = 0.5;
    const double x0 = -d0 * std::sin(phi0), y0 = d0 * std::cos(phi0);
    std::normal_distribution<double> gaus(0., 1.);
    ::riemannFit::Matrix3xNd<N> hits;
    Eigen::Matrix<float, 6, N> hits_ge;
    double trueX = 0, trueY = 0, trueZ = 0, truePhi = 0;
    double sPrev = 0;
    for (int i = 0; i < N; ++i) {
      double lo = sPrev, hi = M_PI * radius, x, y, z, ph;
      for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        helixAt(mid, radius, q, phi0, cot, x0, y0, z0, x, y, z, ph);
        (std::hypot(x, y) < radii[i] ? lo : hi) = mid;
      }
      helixAt(lo, radius, q, phi0, cot, x0, y0, z0, x, y, z, ph);
      sPrev = lo;
      const double r = std::hypot(x, y);
      const double sigRPhi = r < 20 ? 0.001 : 0.005, sigR = 1e-4, sigZ = r < 20 ? 0.002 : 0.1;
      const double tx = -y / r, ty = x / r, rx = x / r, ry = y / r;
      if (i == N - 1) {
        trueX = x;
        trueY = y;
        trueZ = z;
        truePhi = ph;
      }
      if (!noiseless) {
        const double a = sigRPhi * gaus(rng), b = sigR * gaus(rng);
        x += a * tx + b * rx;
        y += a * ty + b * ry;
        z += sigZ * gaus(rng);
      }
      hits(0, i) = x;
      hits(1, i) = y;
      hits(2, i) = z;
      hits_ge(0, i) = sigRPhi * sigRPhi * tx * tx + sigR * sigR * rx * rx;
      hits_ge(1, i) = sigRPhi * sigRPhi * tx * ty + sigR * sigR * rx * ry;
      hits_ge(2, i) = sigRPhi * sigRPhi * ty * ty + sigR * sigR * ry * ry;
      hits_ge(3, i) = 0;
      hits_ge(4, i) = 0;
      hits_ge(5, i) = sigZ * sigZ;
    }

    ::riemannFit::Vector4d fast_fit;
    bl::fastFit(acc, hits, fast_fit);
    bl::PreparedBrokenLineData<N> data;
    bl::prepareBrokenLineData(acc, hits, fast_fit, kBField, data);
    const bl::PreparedBrokenLineData<N> dataOrig = data;
    bl::PreparedBrokenLineData<N> dataFresh = data;

    // ---- the new functions first; data must not change
    bl::OuterCircleState oc;
    bl::OuterLineState ol;
    bl::circleFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, oc);
    bl::lineFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, ol);
    res.dataUnchanged = sameData(data, dataOrig);

    // ---- the overloads that take the work array, on a buffer in block-shared memory, in the order of the fit kernels
    // (line first, one work array for both). Information only: the same operations on another storage can be
    // contracted differently by the CPU compiler, so the last bits may differ.
    {
      static_assert((N + 1) * (N + 1) <= kWorkCapacity, "the work buffer is too small for this N");
      double* work = alpaka::declareSharedVar<double[kWorkCapacity], __COUNTER__>(acc);
      bl::OuterLineState olWork;
      bl::OuterCircleState ocWork;
      bl::lineFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, olWork, work);
      bl::circleFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, ocWork, work);
      res.workOverloadBitwise = sameBits(oc, ocWork) && sameBits(ol, olWork);
      res.workDevCircle = relDev(oc, ocWork);
      res.workDevLine = relDev(ol, olWork);
    }
    res.qFit = oc.qCharge;
    res.mCircle = oc.mRef;
    res.mLine = ol.mRef;
    res.fallbackCircle = oc.refFallback;
    res.fallbackLine = ol.refFallback;

    // ---- upstream after the new functions (on data) vs upstream alone (on dataFresh), in the LST order
    bl::karimaki_circle_fit c1, c2;
    ::riemannFit::LineFit l1, l2;
    upstreamFits(acc, hits, hits_ge, fast_fit, data, c1, l1);
    upstreamFits(acc, hits, hits_ge, fast_fit, dataFresh, c2, l2);
    res.upstreamIdentical = !std::memcmp(c1.par.data(), c2.par.data(), sizeof(double) * 3) &&
                            !std::memcmp(c1.cov.data(), c2.cov.data(), sizeof(double) * 9) &&
                            !std::memcmp(&c1.chi2, &c2.chi2, sizeof(c1.chi2)) && c1.qCharge == c2.qCharge &&
                            !std::memcmp(l1.par.data(), l2.par.data(), sizeof(double) * 2) &&
                            !std::memcmp(l1.cov.data(), l2.cov.data(), sizeof(double) * 4) &&
                            !std::memcmp(&l1.chi2, &l2.chi2, sizeof(l1.chi2));
    res.kappaBitwise = !std::memcmp(&oc.par(2), &c2.par(2), sizeof(double));

    // ---- information only: the precondition, i.e. the new functions called AFTER upstream circleFit
    {
      bl::OuterCircleState oc2;
      bl::OuterLineState ol2;
      bl::circleFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, oc2);
      bl::lineFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, ol2);
      res.orderDependent = oc2.par != oc.par || ol2.par != ol.par || oc2.cov != oc.cov || ol2.cov != ol.cov;
      res.orderDPhi = std::abs(wrapPhi(oc2.par(0) - oc.par(0)));
      res.orderDZ = std::abs(ol2.par(1) - ol.par(1));
    }

    const int m = oc.mRef;

    // ---- the circle solve of circleFit(), repeated here to obtain iMat and the fitted parameters
    bl::PreparedBrokenLineData<N> d = dataOrig;
    const double slope = -d.qCharge / fast_fit(3);
    d.varBeta *= 1. + ::riemannFit::sqr(slope);
    for (int i = 0; i < N; i++)
      d.zInSZplane(i) = d.radii.block(0, i, 2, 1).norm() - fast_fit(2);
    ::riemannFit::VectorNd<N> weights;
    for (int i = 0; i < N; i++) {
      ::riemannFit::Matrix2d vMat;
      vMat(0, 0) = hits_ge.col(i)[0];
      vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];
      vMat(1, 1) = hits_ge.col(i)[2];
      ::riemannFit::Matrix2d rotMat = bl::rotationMatrix(acc, -d.radii(0, i) / d.radii(1, i));
      weights(i) = 1. / ((rotMat * vMat * rotMat.transpose())(1, 1));
    }
    ::riemannFit::VectorNplusONEd<N> r_u;
    r_u(N) = 0;
    for (int i = 0; i < N; i++)
      r_u(i) = weights(i) * d.zInSZplane(i);
    const auto& sT = d.sTransverse;
    const auto& vB = d.varBeta;
    ::riemannFit::MatrixNplusONEd<N> c_u;
    c_u.block(0, 0, N, N) = bl::matrixC_u(acc, weights, sT, vB);
    c_u(N, N) = 0;
    for (int i = 0; i < N; i++) {
      c_u(i, N) = 0;
      if (i > 0 && i < N - 1)
        c_u(i, N) += -(sT(i + 1) - sT(i - 1)) * (sT(i + 1) - sT(i - 1)) /
                     (2. * vB(i) * (sT(i + 1) - sT(i)) * (sT(i) - sT(i - 1)));
      if (i > 1)
        c_u(i, N) += (sT(i) - sT(i - 2)) / (2. * vB(i - 1) * (sT(i) - sT(i - 1)));
      if (i < N - 2)
        c_u(i, N) += (sT(i + 2) - sT(i)) / (2. * vB(i + 1) * (sT(i + 1) - sT(i)));
      c_u(N, i) = c_u(i, N);
      if (i > 0 && i < N - 1)
        c_u(N, N) += ::riemannFit::sqr(sT(i + 1) - sT(i - 1)) / (4. * vB(i));
    }
    ::riemannFit::MatrixNplusONEd<N> iMat;
    math::cholesky::invert(c_u, iMat);
    const ::riemannFit::VectorNplusONEd<N> u = iMat * r_u;

    const ::riemannFit::Vector2d hitA = hits.block(0, m, 2, 1), hitB = hits.block(0, N - 1, 2, 1);
    const ::riemannFit::Vector2d rhatA = d.radii.col(m) / d.radii.col(m).norm();
    const ::riemannFit::Vector2d rhatB = d.radii.col(N - 1) / d.radii.col(N - 1).norm();
    const double zA = d.zInSZplane(m), zB = d.zInSZplane(N - 1);
    auto extract = [&](double uA, double uB, double uN, bl::OuterCircleState& s, ::riemannFit::Matrix3d& jac) {
      s.qCharge = d.qCharge;
      bl::outerCircleExtraction(acc, hitA, hitB, rhatA, rhatB, uA, zA, uB, zB, uN, fast_fit(2), s, jac);
    };

    bl::OuterCircleState s0;
    ::riemannFit::Matrix3d jAn;
    extract(u(m), u(N - 1), u(N), s0, jAn);
    const ::riemannFit::Vector2d chordVec = s0.point - (hitA + (u(m) - zA) * rhatA);
    res.chord = chordVec.norm();
    res.rhatBU = jAn(1, 1);

    // circleFitOuter() solves the same system as the solve above, with the same operations but compiled in another
    // context (see solveRoundingBounds), so its state is compared with the extraction at this solution within the
    // propagated rounding of the solve plus the rounding of the extraction itself, which can also fuse differently:
    // - eB = hit_b + (u_b - z_b) rhat_b: one fusable product per component, |d eB_c| <= eps |(u_b - z_b) rhat_b,c|;
    // - kappa = q (1 / R + u_n): no fusable product, so only the solve enters;
    // - phi_e = atan2(eB - dA) - asin(kappa L / 2): eB and dA move by at most dE = eps (|z_a - u_a| + |z_b - u_b|),
    //   which turns the chord direction by dE / L and changes L by dE plus eps / 2 relative (the fusable sum of
    //   squares under the square root), hence asin by |kappa L / 2| (dE / L + eps / 2) / sqrt(1 - (kappa L / 2)^2).
    {
      constexpr double eps = std::numeric_limits<double>::epsilon();
      ::riemannFit::MatrixNplusONEd<N> tolInv;
      ::riemannFit::VectorNplusONEd<N> tolU;
      res.boundsCircle = solveRoundingBounds(c_u, iMat, r_u, tolInv, tolU);
      const double tolUm = tolU(m), tolUb = tolU(N - 1), tolUn = tolU(N);
      const double dE = eps * (std::abs(zA - u(m)) + std::abs(zB - u(N - 1)));
      const double halfSine = 0.5 * s0.par(2) * res.chord;
      const double tolPhi = std::abs(jAn(0, 0)) * tolUm + std::abs(jAn(0, 1)) * tolUb + std::abs(jAn(0, 2)) * tolUn +
                            dE / res.chord +
                            std::abs(halfSine) * (dE / res.chord + 0.5 * eps) / std::sqrt(1. - halfSine * halfSine);
      double ratio = devRatio(std::abs(s0.par(0) - oc.par(0)), tolPhi);
      res.tolPoint = 0;
      for (int c = 0; c < 2; ++c) {
        const double tolPoint = std::abs(rhatB(c)) * tolUb + eps * std::abs((u(N - 1) - zB) * rhatB(c));
        ratio = std::max(ratio, devRatio(std::abs(s0.point(c) - oc.point(c)), tolPoint));
        res.tolPoint = std::max(res.tolPoint, tolPoint);
      }
      ratio = std::max(ratio, devRatio(std::abs(s0.par(2) - oc.par(2)), tolUn));
      res.ratioCircle = ratio;
      res.tolPhi = tolPhi;
      res.extractionMatchesCircle = ratio <= 1 && s0.par(1) == oc.par(1);
      // upstream circleFit() returns kappa = q (1 / R + u_n) from its own solve: the same bound on u_n
      res.tolKappa = tolUn;
      res.ratioKappa = devRatio(std::abs(oc.par(2) - c2.par(2)), tolUn);
      res.kappaWithinRounding = res.ratioKappa <= 1;
    }

    // ---- circle Jacobian: central finite differences in (u_m, u_{n-1}, u_n)
    const ::riemannFit::Vector2d normalU(-std::sin(s0.par(0)), std::cos(s0.par(0)));
    ::riemannFit::Matrix3d jFD;
    for (int k = 0; k < 3; ++k) {
      double up[3] = {u(m), u(N - 1), u(N)}, dn[3] = {u(m), u(N - 1), u(N)};
      const double h = k == 2 ? kStepKappaRel / fast_fit(2) : kStepU;
      up[k] += h;
      dn[k] -= h;
      bl::OuterCircleState sp, sm;
      ::riemannFit::Matrix3d jUnused;
      extract(up[0], up[1], up[2], sp, jUnused);
      extract(dn[0], dn[1], dn[2], sm, jUnused);
      jFD(0, k) = wrapPhi(sp.par(0) - sm.par(0)) / (2 * h);
      jFD(1, k) = (sp.point - sm.point).dot(normalU) / (2 * h);
      jFD(2, k) = (sp.par(2) - sm.par(2)) / (2 * h);
    }
    res.zeroElementsExact = true;
    for (int k = 0; k < 3; ++k) {
      res.relJCircle[k] = 0;
      for (int i = 0; i < 3; ++i) {
        const bool analyticZero = (i == 1 && k != 1) || (i == 2 && k != 2);
        if (analyticZero)
          res.zeroElementsExact = res.zeroElementsExact && jAn(i, k) == 0 && jFD(i, k) == 0;
        else
          res.relJCircle[k] = std::max(res.relJCircle[k], relErr(jAn(i, k), jFD(i, k), 0.));
      }
    }

    // ---- circle covariance: J_FD * iMat({m, n-1, n}) * J_FD^T plus the chord-wide m.s. term on phi_e
    ::riemannFit::Matrix3d iBlock;
    const int idx[3] = {m, N - 1, N};
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        iBlock(i, k) = iMat(idx[i], idx[k]);
    ::riemannFit::Matrix3d covExp = jFD * iBlock * jFD.transpose();
    covExp(0, 0) +=
        (1 + slope * slope) * bl::multScatt(acc, d.sTotal(N - 1) - d.sTotal(m), kBField, fast_fit(2), N, slope);
    res.relCovCircle = 0;
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        res.relCovCircle = std::max(res.relCovCircle,
                                    relErr(oc.cov(i, k), covExp(i, k), 1e-9 * std::sqrt(covExp(i, i) * covExp(k, k))));

    // ---- the line solve of lineFit(), repeated here to obtain iMat and the fitted parameters
    const ::riemannFit::Matrix2d rotMatL = bl::rotationMatrix(acc, slope);
    ::riemannFit::VectorNd<N> weightsL;
    for (int i = 0; i < N; i++) {
      ::riemannFit::Matrix3d vMat = ::riemannFit::Matrix3d::Zero();
      ::riemannFit::Matrix2x3d jac = ::riemannFit::Matrix2x3d::Zero();
      vMat(0, 0) = hits_ge.col(i)[0];
      vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];
      vMat(0, 2) = vMat(2, 0) = hits_ge.col(i)[3];
      vMat(1, 1) = hits_ge.col(i)[2];
      vMat(2, 1) = vMat(1, 2) = hits_ge.col(i)[4];
      vMat(2, 2) = hits_ge.col(i)[5];
      const double tmp = 1. / dataOrig.radii.block(0, i, 2, 1).norm();
      jac(0, 0) = dataOrig.qCharge * dataOrig.radii(1, i) * tmp;
      jac(0, 1) = -dataOrig.qCharge * dataOrig.radii(0, i) * tmp;
      jac(1, 2) = 1.;
      weightsL(i) = 1. / ((rotMatL * jac * vMat * jac.transpose() * rotMatL.transpose())(1, 1));
    }
    ::riemannFit::VectorNd<N> r_uL;
    for (int i = 0; i < N; i++)
      r_uL(i) = weightsL(i) * dataOrig.zInSZplane(i);
    ::riemannFit::MatrixNd<N> iMatL;
    math::cholesky::invert(bl::matrixC_u(acc, weightsL, dataOrig.sTotal, dataOrig.varBeta), iMatL);
    const ::riemannFit::VectorNd<N> uL = iMatL * r_uL;
    ::riemannFit::Matrix2d iBlockL;
    iBlockL << iMatL(m, m), iMatL(m, N - 1), iMatL(N - 1, m), iMatL(N - 1, N - 1);
    const double sA = dataOrig.sTotal(m), sB = dataOrig.sTotal(N - 1), sE = dataOrig.sTransverse(N - 1);
    const double msVar = chordMultScatt(acc, sB - sA, fast_fit(2), N, slope);
    auto extractLine = [&](double uA, double uB, bl::OuterLineState& s) {
      bl::outerLineExtraction(acc, uA, uB, iBlockL, sA, sB, sE, rotMatL, msVar, s);
    };
    // The state compared with lineFitOuter() is extracted as lineFitOuter() does it: here, where the library
    // code is inlined too, with the multScatt() call itself as the argument. With floating-point contraction (GCC's
    // default for C++, with the FMA of -march=x86-64-v3) the compiler fuses one of the two products of
    // cov(0, 0) = (...) * idiff^2 + multScatt(...) into the addition: the m.s. one when its last multiplication is
    // inlined at the call and used only there, the other one when the variance arrives as a stored value (msVar, or
    // through the lambda). The two orders differ in the last bit of cov.
    bl::OuterLineState l0;
    bl::outerLineExtraction(acc,
                            uL(m),
                            uL(N - 1),
                            iBlockL,
                            sA,
                            sB,
                            sE,
                            rotMatL,
                            bl::multScatt(acc, sB - sA, kBField, fast_fit(2), N, slope),
                            l0);
    res.extractionLineBitwise = l0.par == ol.par && l0.cov == ol.cov;
    res.extractionLineDevPar = (l0.par - ol.par).cwiseAbs().maxCoeff();
    res.extractionLineDevCov = (l0.cov - ol.cov).cwiseAbs().maxCoeff();

    // ---- line Jacobian: central finite differences in (u_m, u_{n-1}) vs the chain of the three linear steps
    ::riemannFit::Matrix2d jFDL;
    ::riemannFit::Matrix2d dCovL[2];  // d cov / d(u_m, u_{n-1}), for the rounding tolerances below
    for (int k = 0; k < 2; ++k) {
      double up[2] = {uL(m), uL(N - 1)}, dn[2] = {uL(m), uL(N - 1)};
      up[k] += kStepU;
      dn[k] -= kStepU;
      bl::OuterLineState sp, sm;
      extractLine(up[0], up[1], sp);
      extractLine(dn[0], dn[1], sm);
      jFDL.col(k) = (sp.par - sm.par) / (2 * kStepU);
      dCovL[k] = (sp.cov - sm.cov) / (2 * kStepU);
    }
    const double chordS = sB - sA;
    const double slopeRot = (uL(N - 1) - uL(m)) / chordS, zipRot = uL(N - 1) - slopeRot * sB;
    const double tmpRot = rotMatL(0, 0) - slopeRot * rotMatL(0, 1);
    ::riemannFit::Matrix2d jPair, jTrans, jRot, jShift;
    jPair << -1. / chordS, 1. / chordS, 0, 1;
    jTrans << 1, 0, -sB, 1;
    jRot << 1. / (tmpRot * tmpRot), 0, zipRot * rotMatL(0, 1) / (tmpRot * tmpRot), 1. / tmpRot;
    jShift << 1, 0, sE, 1;
    const ::riemannFit::Matrix2d jAnL = jShift * jRot * jTrans * jPair;

    // lineFitOuter() solves the same system as the solve above, compiled in another context (see
    // solveRoundingBounds), so its state is compared with the extraction at this solution within the propagated
    // rounding of the solve plus the rounding of outerLineExtraction() itself, which can also fuse differently. Its
    // fusable products, each one rounding (eps for two evaluations) of the product:
    // - translation zip -= slope S_b: d zip = eps |slope S_b|;
    // - tmp = r00 - slope r01: d tmp = eps |slope r01|, so d(1 / tmp) = d tmp / tmp^2 (as a change of the slope by
    //   eps |slope| for the covariance);
    // - cot = (r01 + slope r00) / tmp: eps |slope r00 / tmp| plus |r01 + slope r00| d(1 / tmp);
    // - z_e = zip / tmp + cot s_e: |zip| d(1 / tmp) + d zip / |tmp| + |s_e| d cot + eps |cot s_e|.
    // The covariance goes through the m.s. sum ((...) idiff^2 + m.s., one fusable product) and three congruences
    // J C J^T, each two levels of 2x2 products with one fusable product per element: 7 levels, each at most one
    // rounding of the absolute value of what it adds up, so 7 eps |J_shift| |J_rot| |J_trans| |C_0| (...)^T with
    // |C_0| = |J_pair| |uCov| |J_pair|^T + m.s. The inverse block uCov enters linearly through the Jacobian of the
    // state, and u_m, u_{n-1} through the dependence of that Jacobian on the slope and intercept (finite differences).
    {
      constexpr double eps = std::numeric_limits<double>::epsilon();
      ::riemannFit::MatrixNd<N> tolInvL;
      ::riemannFit::VectorNd<N> tolUL;
      res.boundsLine = solveRoundingBounds(
          bl::matrixC_u(acc, weightsL, dataOrig.sTotal, dataOrig.varBeta), iMatL, r_uL, tolInvL, tolUL);
      const double tolUm = tolUL(m), tolUb = tolUL(N - 1);
      const double r00 = rotMatL(0, 0), r01 = rotMatL(0, 1), invTmp = 1. / tmpRot;
      const double cotOut = (r01 + slopeRot * r00) * invTmp;
      const double dZipRot = eps * std::abs(slopeRot * sB);
      const double dInvTmp = eps * std::abs(slopeRot * r01) * invTmp * invTmp;
      const double dCot = eps * std::abs(slopeRot * r00 * invTmp) + std::abs(r01 + slopeRot * r00) * dInvTmp;
      const double dZe =
          std::abs(zipRot) * dInvTmp + dZipRot * std::abs(invTmp) + std::abs(sE) * dCot + eps * std::abs(cotOut * sE);
      const ::riemannFit::Matrix2d absJ = jAnL.cwiseAbs();
      const double tolPar[2] = {absJ(0, 0) * tolUm + absJ(0, 1) * tolUb + dCot,
                                absJ(1, 0) * tolUm + absJ(1, 1) * tolUb + dZe};
      ::riemannFit::Matrix2d tolB;
      tolB << tolInvL(m, m), tolInvL(m, N - 1), tolInvL(N - 1, m), tolInvL(N - 1, N - 1);
      // d cov / d slope and d cov / d zip at fixed zip and slope: u_a = zip + slope S_a, u_b = zip + slope S_b
      const ::riemannFit::Matrix2d dCovSlope = sA * dCovL[0] + sB * dCovL[1], dCovZip = dCovL[0] + dCovL[1];
      const ::riemannFit::Matrix2d absChain = jShift.cwiseAbs() * jRot.cwiseAbs() * jTrans.cwiseAbs();
      ::riemannFit::Matrix2d c0Abs = jPair.cwiseAbs() * iBlockL.cwiseAbs() * jPair.cwiseAbs().transpose();
      c0Abs(0, 0) += msVar;
      const ::riemannFit::Matrix2d tolCov =
          absJ * tolB * absJ.transpose() + dCovL[0].cwiseAbs() * tolUm + dCovL[1].cwiseAbs() * tolUb +
          dCovSlope.cwiseAbs() * (eps * std::abs(slopeRot)) + dCovZip.cwiseAbs() * dZipRot +
          7 * eps * (absChain * c0Abs * absChain.transpose());
      double ratio = 0;
      for (int i = 0; i < 2; ++i) {
        ratio = std::max(ratio, devRatio(std::abs(l0.par(i) - ol.par(i)), tolPar[i]));
        for (int k = 0; k < 2; ++k)
          ratio = std::max(ratio, devRatio(std::abs(l0.cov(i, k) - ol.cov(i, k)), tolCov(i, k)));
      }
      res.ratioLine = ratio;
      res.extractionMatchesLine = ratio <= 1;
      res.tolCot = tolPar[0];
      res.tolZ = tolPar[1];

      // Upstream lineFit() takes the line of nodes (0, 1) from its own solve, the outer fit that of nodes
      // (m, n-1): with exact hits both are the truth up to rounding, and each deviates from the exact solution of
      // the double system by at most half the two-evaluation bound, propagated through its own Jacobian. This
      // rounding term is added to kTolNoiseless only on the fallback layout: its 1.2 cm span (0.16 cm between
      // nodes 0 and 1) makes the line solve ill-conditioned, so that upstream lineFit() alone misses the true
      // cot(theta) by more than kTolNoiseless (the reason the truth checks exempt that layout), and two solves
      // that round differently disagree by as much. On every other layout kTolNoiseless holds as it is.
      const double sU0 = dataOrig.sTotal(0), sU1 = dataOrig.sTotal(1), chordU = sU1 - sU0;
      const double slopeU = (uL(1) - uL(0)) / chordU, zipU = uL(1) - slopeU * sU1;
      const double tmpU = rotMatL(0, 0) - slopeU * rotMatL(0, 1);
      ::riemannFit::Matrix2d jPairU, jTransU, jRotU;
      jPairU << -1. / chordU, 1. / chordU, 0, 1;
      jTransU << 1, 0, -sU1, 1;
      jRotU << 1. / (tmpU * tmpU), 0, zipU * rotMatL(0, 1) / (tmpU * tmpU), 1. / tmpU;
      const ::riemannFit::Matrix2d absJU = (jShift * jRotU * jTransU * jPairU).cwiseAbs();
      for (int i = 0; i < 2; ++i) {
        const double rounding =
            0.5 * (absJ(i, 0) * tolUm + absJ(i, 1) * tolUb + absJU(i, 0) * tolUL(0) + absJU(i, 1) * tolUL(1));
        (i == 0 ? res.tolUpCot : res.tolUpZ) = kTolNoiseless + (expectFallback ? rounding : 0.);
      }
    }
    // FD resolution of the line map: (cot, z_e) are evaluated with a rounding of a few ulp of the larger of the two
    // (|z_e| up to ~2e2 cm), so a central difference cannot resolve derivatives below ~16 eps max|par| / h. The
    // element dz_e/du_m falls below that: it is proportional to the residual at node n-1 (exactly 0 for exact hits),
    // because z_e is taken at that node. Such elements are compared at the resolution, all others at kTolFD.
    const double resolutionL = 16 * std::numeric_limits<double>::epsilon() * ol.par.cwiseAbs().maxCoeff() / kStepU;
    res.lineWithinResolution = true;
    res.maxAbsUnresolvedLine = 0;
    for (int k = 0; k < 2; ++k) {
      res.relJLine[k] = 0;
      for (int i = 0; i < 2; ++i) {
        if (std::max(std::abs(jAnL(i, k)), std::abs(jFDL(i, k))) * kTolFD > resolutionL) {
          res.relJLine[k] = std::max(res.relJLine[k], relErr(jAnL(i, k), jFDL(i, k), 0.));
        } else {
          const double deviation = std::abs(jAnL(i, k) - jFDL(i, k));
          res.maxAbsUnresolvedLine = std::max(res.maxAbsUnresolvedLine, deviation);
          res.lineWithinResolution = res.lineWithinResolution && deviation < resolutionL;
        }
      }
    }
    // the m.s. variance sits on the rotated slope at fixed u_{n-1}: d/d(slope) = -(S_b - S_a) d/du_m
    const ::riemannFit::Matrix2d covExpL =
        jFDL * iBlockL * jFDL.transpose() + msVar * chordS * chordS * jFDL.col(0) * jFDL.col(0).transpose();
    res.relCovLine = 0;
    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < 2; ++k)
        res.relCovLine = std::max(res.relCovLine, relErr(ol.cov(i, k), covExpL(i, k), 0.));

    // ---- truth at the last hit
    res.nlPhi = std::abs(wrapPhi(oc.par(0) - truePhi));
    res.nlPos = std::hypot(oc.point(0) - trueX, oc.point(1) - trueY);
    res.nlKappaRel = std::abs(oc.par(2) * radius - q);
    res.nlCot = std::abs(ol.par(0) - cot);
    res.nlCotUpstream = std::abs(l2.par(0) - cot);  // the same quantity from upstream lineFit, for comparison
    res.nlZ = std::abs(ol.par(1) - trueZ);
    // ---- agreement with upstream lineFit on the same inputs (l2, from upstreamFits on dataFresh). lineFit already
    // returns par = (cot(theta), Zip) in the original sz system (rotated back as at its "rotate to the original sz
    // system" step), so the upstream z at the outermost node is Zip + cot(theta) * s_e, with s_e = sTransverse(n-1)
    // of the same prepared data, exactly the last step of outerLineExtraction().
    const double sOuter = dataOrig.sTransverse(N - 1);
    res.upCot = std::abs(ol.par(0) - l2.par(0));
    res.upZ = std::abs(ol.par(1) - (l2.par(1) + l2.par(0) * sOuter));
    res.ratioUpCot = res.upCot / res.tolUpCot;
    res.ratioUpZ = res.upZ / res.tolUpZ;
    // the tangent with the opposite bending sign
    const double phiWrong = std::atan2(chordVec(1), chordVec(0)) + std::asin(0.5 * oc.par(2) * chordVec.norm());
    res.wrongSignShift = std::abs(wrapPhi(phiWrong - truePhi));
    return res;
  }

  // The layouts of the fits the LST input runs: the first N hits of the ladder below, for every N from 3 (the
  // inner-tracker-only fit of three pixel hits) to 14 (the cap on the number of fitted hits). The ladder is four
  // pixel hits followed by the two hits of each outer tracker module pair (0.16 cm apart in radius). Its radii are
  // chosen so that the reference node of the 2 cm rule is known without running the fit: the outermost node more than
  // 2 cm from the last hit, i.e. the last hit of the previous layer (kLadderRefNode[N]).
  constexpr int kLadderMinN = 3, kLadderMaxN = 14;
  constexpr double kLadder[kLadderMaxN] = {3, 7, 11, 16, 23, 23.16, 36, 36.16, 51, 51.16, 68, 68.16, 87, 87.16};
  constexpr int kLadderRefNode[kLadderMaxN + 1] = {-1, -1, -1, 1, 2, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11};

  template <int N>
  void pushLadderCase(std::vector<CaseResult>& results,
                      const Acc1D& acc,
                      double pt,
                      int q,
                      double phi0,
                      double cot,
                      bool noiseless,
                      std::mt19937& rng) {
    const std::vector<double> radii(kLadder, kLadder + N);
    results.push_back(runCase<N>(acc, "ladder", radii, kLadderRefNode[N], false, pt, q, phi0, cot, noiseless, rng));
  }

  template <int... Ns>
  void pushLadderCases(std::integer_sequence<int, Ns...>,
                       std::vector<CaseResult>& results,
                       const Acc1D& acc,
                       double pt,
                       int q,
                       double phi0,
                       double cot,
                       bool noiseless,
                       std::mt19937& rng) {
    (pushLadderCase<kLadderMinN + Ns>(results, acc, pt, q, phi0, cot, noiseless, rng), ...);
  }

  struct OuterFitTestKernel {
    ALPAKA_FN_ACC void operator()(const Acc1D& acc, std::vector<CaseResult>* results) const {
      std::mt19937 rng(12345);
      // transverse radii of the hits (cm); expected m from the 2 cm rule
      const std::vector<double> layoutA = {23, 23.16, 36, 36.16, 51, 51.16, 68, 68.16, 87, 87.16};
      const std::vector<double> layoutB = {3, 7, 11, 16, 23, 23.16, 36, 36.16, 51, 51.16, 68, 68.16};
      const std::vector<double> layoutC = {23, 23.16, 30, 30.16, 32.3, 32.46};
      const std::vector<double> layoutD = {23, 23.16, 51, 51.16, 81, 81.16};
      const std::vector<double> layoutE = {23, 23.16, 36, 36.16, 51, 51.16, 68};
      const std::vector<double> layoutF = {40, 40.16, 40.5, 40.66, 41.2};
      const std::vector<double> layoutG = {3, 7, 11, 16};
      const double pts[4] = {0.8, 2., 5., 20.};
      const double phis[3] = {0.3, 3.1, -1.7};
      const double cots[3] = {0.3, -1.2, 2.5};
      for (bool noiseless : {true, false})
        for (int q : {+1, -1})
          for (double pt : pts)
            for (int a = 0; a < 3; ++a) {
              auto& r = *results;
              r.push_back(runCase<10>(acc, "OT pairs", layoutA, 7, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<12>(acc, "pixel+OT", layoutB, 9, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<6>(acc, "chord 2.3", layoutC, 3, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<6>(acc, "chord 30", layoutD, 3, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<7>(acc, "single last", layoutE, 5, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<5>(acc, "fallback", layoutF, 0, true, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<4>(acc, "pixel quad", layoutG, 2, false, pt, q, phis[a], cots[a], noiseless, rng));
              pushLadderCases(std::make_integer_sequence<int, kLadderMaxN - kLadderMinN + 1>{},
                              r,
                              acc,
                              pt,
                              q,
                              phis[a],
                              cots[a],
                              noiseless,
                              rng);
            }
    }
  };

}  // namespace

TEST_CASE("Broken Line state at the outermost node: " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE), "[BrokenLine]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  REQUIRE(not devices.empty());
  Queue queue(devices.front());
  std::vector<CaseResult> results;
  alpaka::exec<Acc1D>(queue, cms::alpakatools::make_workdiv<Acc1D>(1, 1), OuterFitTestKernel{}, &results);
  alpaka::wait(queue);
  // 2 smearing modes x 2 charges x 4 momenta x 3 directions x (7 named layouts + 12 ladder layouts, N = 3..14)
  REQUIRE(results.size() == 912);

  double maxJCircle[3] = {0, 0, 0}, maxJLine[2] = {0, 0}, maxCovCircle = 0, maxCovLine = 0;
  double maxNlPhi = 0, maxNlPos = 0, maxNlKappa = 0, minWrongShift = 1e9;
  double minChord = 1e9, maxChord = 0, minQRhatBU = 1e9, maxOrderDPhi = 0, maxOrderDZ = 0, maxUnresolvedLine = 0;
  // index 0: all layouts but the fallback one; index 1: the fallback layout
  double maxNlCot[2] = {0, 0}, maxNlZ[2] = {0, 0}, maxNlCotUpstream[2] = {0, 0}, maxUpCot[2] = {0, 0},
         maxUpZ[2] = {0, 0};
  int nFallback = 0, nOrderDependent = 0, nChargeFlips = 0, nLineExtractionDiffers = 0;
  int nCircleExtractionDiffers = 0, nKappaDiffers = 0, nWorkOverloadDiffers = 0;
  double maxWorkDevCircle = 0, maxWorkDevLine = 0;
  // largest |deviation| / tolerance and largest tolerance of the rounding-toleranced comparisons
  double maxRatioCircle = 0, maxRatioLine = 0, maxRatioKappa = 0, maxRatioUpCot[2] = {0, 0}, maxRatioUpZ[2] = {0, 0};
  double maxTolPhi = 0, maxTolPoint = 0, maxTolKappa = 0, maxTolCot = 0, maxTolZ = 0, maxTolUpCot[2] = {0, 0},
         maxTolUpZ[2] = {0, 0};
  double maxLineExtractionDevPar = 0, maxLineExtractionDevCov = 0;
  // per number of fitted hits N (index N): number of cases and the largest deviations over every layout of that N
  struct PerN {
    int cases = 0, noiselessCases = 0;
    double jCircle = 0, jLine = 0, covCircle = 0, covLine = 0;
    double nlPhi = 0, nlPos = 0, nlKappa = 0, nlCot = 0, nlZ = 0;
  };
  PerN perN[kLadderMaxN + 1];

  for (auto const& r : results) {
    INFO(r.label);
    // With smeared hits the generated charge is not always recoverable: over the 1.2 cm "fallback" span and the
    // 2.3 cm chord at 20 GeV the sagitta is below the 50 um hit resolution. The fit's charge is the three-point sign
    // of prepareBrokenLineData(), and the sign conventions below are checked against it.
    if (r.noiseless) {
      INFO("fitted charge differs from the generated charge for exact hits");
      CHECK(r.qFit == r.q);
    }
    {
      INFO("reference node m differs from the 2 cm rule expectation");
      CHECK(r.mCircle == r.expectedM);
    }
    {
      INFO("line and circle use different reference nodes");
      CHECK(r.mLine == r.mCircle);
    }
    {
      INFO("unexpected fallback flag (circle)");
      CHECK(r.fallbackCircle == r.expectFallback);
    }
    {
      INFO("unexpected fallback flag (line)");
      CHECK(r.fallbackLine == r.expectFallback);
    }
    {
      INFO("Cholesky factorization of the rounding tolerances failed: circle " << !r.boundsCircle << " line "
                                                                               << !r.boundsLine);
      CHECK((r.boundsCircle && r.boundsLine));
    }
    {
      INFO("outerCircleExtraction at the solution differs from circleFitOuter beyond rounding, |dev| / tol "
           << r.ratioCircle);
      CHECK(r.extractionMatchesCircle);
    }
    {
      INFO("outerLineExtraction at the solution differs from lineFitOuter beyond rounding, |dev| / tol "
           << r.ratioLine);
      CHECK(r.extractionMatchesLine);
    }
    for (int k = 0; k < 3; ++k) {
      INFO("circle Jacobian column " << k << " (0: u_m, 1: u_{n-1}, 2: u_n), rel. deviation " << r.relJCircle[k]);
      {
        INFO("circle Jacobian disagrees with finite differences");
        CHECK(r.relJCircle[k] < kTolFD);
      }
    }
    {
      INFO("circle Jacobian elements that are analytically zero are not zero");
      CHECK(r.zeroElementsExact);
    }
    INFO("circle covariance rel. deviation " << r.relCovCircle);
    {
      INFO("circle covariance disagrees with J_FD * iMat * J_FD^T + m.s.");
      CHECK(r.relCovCircle < kTolFD);
    }
    for (int k = 0; k < 2; ++k) {
      INFO("line Jacobian column " << k << " (0: u_m, 1: u_{n-1}), rel. deviation " << r.relJLine[k]);
      {
        INFO("line Jacobian disagrees with finite differences");
        CHECK(r.relJLine[k] < kTolFD);
      }
    }
    {
      INFO("line Jacobian element below the FD resolution disagrees beyond it, |deviation| " << r.maxAbsUnresolvedLine);
      CHECK(r.lineWithinResolution);
    }
    INFO("line covariance rel. deviation " << r.relCovLine);
    {
      INFO("line covariance disagrees with J_FD * iMat * J_FD^T + m.s.");
      CHECK(r.relCovLine < kTolFD);
    }
    INFO("rhat_b . U = " << r.rhatBU);
    {
      INFO("d x_T / d u_b does not have the sign of the fitted charge");
      CHECK(r.rhatBU * r.qFit > 0);
    }
    {
      INFO("the new functions modified PreparedBrokenLineData");
      CHECK(r.dataUnchanged);
    }
    {
      INFO("upstream circleFit/lineFit outputs changed by calling the new functions first");
      CHECK(r.upstreamIdentical);
    }
    {
      INFO("circleFitOuter kappa differs from circleFit kappa beyond rounding, |dev| / tol " << r.ratioKappa);
      CHECK(r.kappaWithinRounding);
    }
    if (r.noiseless) {
      INFO("noiseless residuals: phi " << r.nlPhi << " pos " << r.nlPos << " kappa(rel) " << r.nlKappaRel << " cot "
                                       << r.nlCot << " z " << r.nlZ << "; opposite-sign shift " << r.wrongSignShift);
      {
        INFO("phi_e differs from the true tangent at the last hit");
        CHECK(r.nlPhi < kTolNoiseless);
      }
      {
        INFO("eB differs from the true last hit");
        CHECK(r.nlPos < kTolNoiseless);
      }
      {
        INFO("kappa differs from the true curvature");
        CHECK(r.nlKappaRel < kTolNoiseless);
      }
      // The line truth checks exclude the fallback layout (1.2 cm span): there the three-point fast fit is
      // ill-conditioned by construction, and upstream lineFit misses the true cot(theta) by the same amount.
      if (!r.expectFallback) {
        {
          INFO("cot_out differs from the true cot(theta)");
          CHECK(r.nlCot < kTolNoiseless);
        }
        {
          INFO("z_e differs from the true z at the last hit");
          CHECK(r.nlZ < kTolNoiseless);
        }
      }
      // On every layout the outer line must agree with upstream lineFit on the same inputs, within kTolNoiseless,
      // plus the rounding of the two solves on the ill-conditioned fallback layout (see runCase).
      INFO("upstream agreement: |d cot| " << r.upCot << " (tol " << r.tolUpCot << ") |d z_e| " << r.upZ << " (tol "
                                          << r.tolUpZ << ")");
      {
        INFO("cot_out differs from upstream lineFit cot(theta)");
        CHECK(r.upCot < r.tolUpCot);
      }
      {
        INFO("z_e differs from upstream lineFit Zip + cot(theta) * sTransverse(n-1)");
        CHECK(r.upZ < r.tolUpZ);
      }
      {
        INFO("the opposite bending sign also matches the truth");
        CHECK(r.wrongSignShift > kMinWrongSignShift);
      }
      maxNlPhi = std::max(maxNlPhi, r.nlPhi);
      maxNlPos = std::max(maxNlPos, r.nlPos);
      maxNlKappa = std::max(maxNlKappa, r.nlKappaRel);
      const int fb = r.expectFallback ? 1 : 0;  // 1 for the fallback layout
      maxNlCot[fb] = std::max(maxNlCot[fb], r.nlCot);
      maxNlCotUpstream[fb] = std::max(maxNlCotUpstream[fb], r.nlCotUpstream);
      maxNlZ[fb] = std::max(maxNlZ[fb], r.nlZ);
      maxUpCot[fb] = std::max(maxUpCot[fb], r.upCot);
      maxUpZ[fb] = std::max(maxUpZ[fb], r.upZ);
      maxRatioUpCot[fb] = std::max(maxRatioUpCot[fb], r.ratioUpCot);
      maxRatioUpZ[fb] = std::max(maxRatioUpZ[fb], r.ratioUpZ);
      maxTolUpCot[fb] = std::max(maxTolUpCot[fb], r.tolUpCot);
      maxTolUpZ[fb] = std::max(maxTolUpZ[fb], r.tolUpZ);
      minWrongShift = std::min(minWrongShift, r.wrongSignShift);
    }
    {
      auto& p = perN[r.n];
      ++p.cases;
      for (int k = 0; k < 3; ++k)
        p.jCircle = std::max(p.jCircle, r.relJCircle[k]);
      for (int k = 0; k < 2; ++k)
        p.jLine = std::max(p.jLine, r.relJLine[k]);
      p.covCircle = std::max(p.covCircle, r.relCovCircle);
      p.covLine = std::max(p.covLine, r.relCovLine);
      if (r.noiseless) {
        ++p.noiselessCases;
        p.nlPhi = std::max(p.nlPhi, r.nlPhi);
        p.nlPos = std::max(p.nlPos, r.nlPos);
        p.nlKappa = std::max(p.nlKappa, r.nlKappaRel);
        if (!r.expectFallback) {
          p.nlCot = std::max(p.nlCot, r.nlCot);
          p.nlZ = std::max(p.nlZ, r.nlZ);
        }
      }
    }
    for (int k = 0; k < 3; ++k)
      maxJCircle[k] = std::max(maxJCircle[k], r.relJCircle[k]);
    for (int k = 0; k < 2; ++k)
      maxJLine[k] = std::max(maxJLine[k], r.relJLine[k]);
    maxCovCircle = std::max(maxCovCircle, r.relCovCircle);
    maxCovLine = std::max(maxCovLine, r.relCovLine);
    minChord = std::min(minChord, r.chord);
    maxChord = std::max(maxChord, r.chord);
    minQRhatBU = std::min(minQRhatBU, r.qFit * r.rhatBU);
    maxUnresolvedLine = std::max(maxUnresolvedLine, r.maxAbsUnresolvedLine);
    nChargeFlips += (r.qFit != r.q);
    nLineExtractionDiffers += !r.extractionLineBitwise;
    nCircleExtractionDiffers += !(r.ratioCircle == 0);
    nKappaDiffers += !r.kappaBitwise;
    nWorkOverloadDiffers += !r.workOverloadBitwise;
    maxWorkDevCircle = std::max(maxWorkDevCircle, r.workDevCircle);
    maxWorkDevLine = std::max(maxWorkDevLine, r.workDevLine);
    maxRatioCircle = std::max(maxRatioCircle, r.ratioCircle);
    maxRatioLine = std::max(maxRatioLine, r.ratioLine);
    maxRatioKappa = std::max(maxRatioKappa, r.ratioKappa);
    maxTolPhi = std::max(maxTolPhi, r.tolPhi);
    maxTolPoint = std::max(maxTolPoint, r.tolPoint);
    maxTolKappa = std::max(maxTolKappa, r.tolKappa);
    maxTolCot = std::max(maxTolCot, r.tolCot);
    maxTolZ = std::max(maxTolZ, r.tolZ);
    maxLineExtractionDevPar = std::max(maxLineExtractionDevPar, r.extractionLineDevPar);
    maxLineExtractionDevCov = std::max(maxLineExtractionDevCov, r.extractionLineDevCov);
    nFallback += r.fallbackCircle;
    nOrderDependent += r.orderDependent;
    maxOrderDPhi = std::max(maxOrderDPhi, r.orderDPhi);
    maxOrderDZ = std::max(maxOrderDZ, r.orderDZ);
  }

  std::cout << "Summary over " << results.size() << " cases\n"
            << "  max rel. FD deviation, circle Jacobian columns (u_m, u_{n-1}, u_n): " << maxJCircle[0] << ' '
            << maxJCircle[1] << ' ' << maxJCircle[2] << '\n'
            << "  max rel. FD deviation, line Jacobian columns (u_m, u_{n-1}): " << maxJLine[0] << ' ' << maxJLine[1]
            << "; max |deviation| of elements below the FD resolution " << maxUnresolvedLine << '\n'
            << "  max rel. deviation of the covariances, circle / line: " << maxCovCircle << " / " << maxCovLine << '\n'
            << "  noiseless max |phi_e - true| " << maxNlPhi << " rad, |eB - true| " << maxNlPos
            << " cm, |kappa R / q - 1| " << maxNlKappa << '\n'
            << "  noiseless, non-fallback layouts: max |cot_out - true| " << maxNlCot[0] << ", |z_e - true| "
            << maxNlZ[0] << " cm, upstream lineFit |cot - true| " << maxNlCotUpstream[0] << "; vs upstream |d cot| "
            << maxUpCot[0] << ", |d z_e| " << maxUpZ[0] << " cm\n"
            << "  noiseless, fallback layout (truth not asserted): max |cot_out - true| " << maxNlCot[1]
            << ", |z_e - true| " << maxNlZ[1] << " cm, upstream lineFit |cot - true| " << maxNlCotUpstream[1]
            << "; vs upstream |d cot| " << maxUpCot[1] << ", |d z_e| " << maxUpZ[1] << " cm\n"
            << "  noiseless min shift of the opposite-sign tangent " << minWrongShift << " rad\n"
            << "  min q * (rhat_b . U) " << minQRhatBU << "; chord range " << minChord << " - " << maxChord << " cm\n"
            << "  fallback (m = 0) cases " << nFallback << "; smeared cases with a flipped fitted charge "
            << nChargeFlips << '\n'
            << "  information only: outer fits with the work array in block-shared memory differ bitwise from those "
            << "with it on the stack in " << nWorkOverloadDiffers << " of " << results.size()
            << " cases (max relative deviation, circle point/par/cov " << maxWorkDevCircle << ", line par/cov "
            << maxWorkDevLine << ")\n"
            << "  information only: results differ when called after upstream circleFit in " << nOrderDependent
            << " of " << results.size() << " cases (max |d phi_e| " << maxOrderDPhi << " rad, max |d z_e| "
            << maxOrderDZ << " cm)\n"
            << "  information only: outerLineExtraction at the solution differs bitwise from lineFitOuter in "
            << nLineExtractionDiffers << " cases (max |d par| " << maxLineExtractionDevPar << ", max |d cov| "
            << maxLineExtractionDevCov << "), outerCircleExtraction from circleFitOuter in " << nCircleExtractionDiffers
            << ", circleFitOuter kappa from circleFit kappa in " << nKappaDiffers << '\n'
            << "  checked, must be <= 1: max |deviation| / rounding tolerance: circle extraction " << maxRatioCircle
            << ", line extraction " << maxRatioLine << ", kappa vs circleFit " << maxRatioKappa
            << "; noiseless vs upstream lineFit (non-fallback / fallback) cot " << maxRatioUpCot[0] << " / "
            << maxRatioUpCot[1] << ", z_e " << maxRatioUpZ[0] << " / " << maxRatioUpZ[1] << '\n'
            << "  largest rounding tolerances: phi_e " << maxTolPhi << " rad, eB " << maxTolPoint << " cm, kappa "
            << maxTolKappa << " 1/cm, cot " << maxTolCot << ", z_e " << maxTolZ
            << " cm; vs upstream lineFit (non-fallback / fallback) cot " << maxTolUpCot[0] << " / " << maxTolUpCot[1]
            << ", z_e " << maxTolUpZ[0] << " / " << maxTolUpZ[1] << " cm\n"
            << "  per N (tolerances: Jacobians and covariances " << kTolFD << ", noiseless residuals " << kTolNoiseless
            << "):\n";
  for (int n = 0; n <= kLadderMaxN; ++n) {
    auto const& p = perN[n];
    if (p.cases == 0)
      continue;
    std::cout << "    N=" << n << " cases " << p.cases << " (noiseless " << p.noiselessCases
              << ") max rel. dev. J circle " << p.jCircle << " J line " << p.jLine << " cov circle " << p.covCircle
              << " cov line " << p.covLine << "; noiseless max phi " << p.nlPhi << " pos " << p.nlPos << " kappa "
              << p.nlKappa << " cot " << p.nlCot << " z " << p.nlZ << '\n';
  }
  std::cout << std::flush;
}
