// Unit test of lst::closestApproach on the host, for Broken Line Fit states in a uniform field:
//  - against the host chain in the same field: TSCBLBuilderNoMaterial, pixelSegmentImpactParameters,
//    PerigeeConversions::ftsToPerigeeError and pixelSegmentMomentumErrors with the tolerance
//  - against the exact closest approach of the generating helix to the beam line, in double precision
//  - against the uniform-field identities: pT and pz are kept, and ptErr, etaErr have closed forms
//  - on invalid states

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/UniformEngine/interface/UniformMagneticField.h"
#include "RecoTracker/LST/interface/PixelSegmentClosestApproach.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include "RecoTracker/LST/test/BLFTestStates.h"

namespace {

  using blfTestStates::TestState;
  using Result = lst::PixelSegmentClosestApproach;

  constexpr int kNStates = 100000;
  constexpr uint32_t kSeed = 20261018;

  // the precision of the host closest approach, as in testBeamLineClosestApproach: the Newton search stops at
  // |dphi| < 1e-6 on the helix, i.e. 1e-6 times the helix arc length per radian |p| / (c Bz); the transverse radius
  // amplifies the float rounding of |p| (6e-8) by p^2 / pT^2; float positions carry 1e-5 cm
  constexpr double kNewtonTolerance = 1.e-6, kFloatRounding = 6.e-8, kFloatPosition = 1.e-5;
  // tolerances of testBeamLineClosestApproach
  constexpr double kValidMismatchTolerance = 1.e-4, kPositionTolerance = 10.,
                   kMomentumTolerance = 5. * kNewtonTolerance, kErrorTolerance = 1.e-3;

  double positionUnit(TestState const& state, float bField) {
    const double momentumRatio = std::sqrt(1. + state.cot * state.cot);
    const double arcPerRadian = state.pt * momentumRatio / bField;
    return kFloatPosition + (kNewtonTolerance + kFloatRounding * momentumRatio * momentumRatio) * arcPerRadian;
  }

  struct MaxDifference {
    double value = 0;
    int index = -1;
    void update(double difference, int i) {
      if (std::isnan(value))  // a NaN difference is kept
        return;
      if (!(difference <= value)) {  // NaN counts as a difference
        value = difference;
        index = i;
      }
    }
  };

  bool isZero(Result const& result) {
    return result.px == 0.f && result.py == 0.f && result.pz == 0.f && result.dxy == 0.f && result.dz == 0.f &&
           result.ptErr == 0.f && result.etaErr == 0.f && result.charge == 0;
  }

  // the impact parameters of pixelSegmentImpactParameters, in double
  void impactParameters(double const position[3],
                        double const momentum[3],
                        lst::UniformField::Vector3 const& beamSpot,
                        double& dxy,
                        double& dz) {
    const double pt = std::hypot(momentum[0], momentum[1]);
    const double dx = position[0] - beamSpot.x, dy = position[1] - beamSpot.y;
    dxy = (-dx * momentum[1] + dy * momentum[0]) / pt;
    dz = (position[2] - beamSpot.z) - (dx * momentum[0] + dy * momentum[1]) / pt * (momentum[2] / pt);
  }

  // the closest approach of the generating helix to the beam line, in double: Newton on the transverse arc length s
  // from the vertex for f(s) = g(s) . h'(s) = 0, g the component of h(s) minus the line point orthogonal to the line
  bool exactClosestApproach(TestState const& state, double bField, double position[3], double momentum[3]) {
    const auto& line = state.beamLine;
    const double norm =
        std::sqrt(double(line.direction.x) * line.direction.x + double(line.direction.y) * line.direction.y +
                  double(line.direction.z) * line.direction.z);
    const double direction[3] = {line.direction.x / norm, line.direction.y / norm, line.direction.z / norm};
    const double w = -state.charge * bField / state.pt;
    double s = 0.;
    bool converged = false;
    for (int iteration = 0; iteration < 100 && !converged; ++iteration) {
      double h[3], phi;
      blfTestStates::helixAt(state, bField, s, h, phi);
      const double diff[3] = {h[0] - line.position.x, h[1] - line.position.y, h[2] - line.position.z};
      const double along = diff[0] * direction[0] + diff[1] * direction[1] + diff[2] * direction[2];
      const double g[3] = {
          diff[0] - along * direction[0], diff[1] - along * direction[1], diff[2] - along * direction[2]};
      const double h1[3] = {std::cos(phi), std::sin(phi), state.cot};
      const double h2[3] = {-w * std::sin(phi), w * std::cos(phi), 0.};
      const double h1Along = h1[0] * direction[0] + h1[1] * direction[1] + h1[2] * direction[2];
      const double f = g[0] * h1[0] + g[1] * h1[1] + g[2] * h1[2];
      const double derivative = h1[0] * h1[0] + h1[1] * h1[1] + h1[2] * h1[2] - h1Along * h1Along + g[0] * h2[0] +
                                g[1] * h2[1] + g[2] * h2[2];
      const double step = f / derivative;
      s -= step;
      // far below the float precision of the result; the rounding of f limits steps to about 1e-13 cm
      converged = std::abs(step) < 1.e-10;
    }
    double phi;
    blfTestStates::helixAt(state, bField, s, position, phi);
    momentum[0] = state.pt * std::cos(phi);
    momentum[1] = state.pt * std::sin(phi);
    momentum[2] = state.pt * state.cot;
    return converged;
  }

}  // namespace

TEST_CASE("closest approach against TSCBLBuilderNoMaterial in the same uniform field",
          "[PixelSegmentClosestApproach]") {
  const float bField = blfTestStates::bField();
  const lst::UniformField uniformField(bField);
  const UniformMagneticField field(uniformField.inTesla({0.f, 0.f, 0.f}).z);
  const TSCBLBuilderNoMaterial builder;
  const std::vector<TestState> states = blfTestStates::makeStates(kNStates, kSeed);

  int nValid = 0, nValidMismatch = 0;
  MaxDifference dxy, dz, momentum, ptErr, etaErr;
  for (int i = 0; i < kNStates; ++i) {
    TestState const& state = states[i];
    const Result result = lst::closestApproach(state.row, uniformField, state.beamLine);

    beamLineClosestApproach::FreeState lastHit;
    REQUIRE(lst::closestApproachDetail::freeState(state.row, bField, lastHit));
    AlgebraicSymMatrix55 error;
    for (int row = 0; row < 5; ++row)
      for (int col = 0; col <= row; ++col)
        error(row, col) = lastHit.error[row][col];
    const FreeTrajectoryState fts(
        GlobalTrajectoryParameters(GlobalPoint(lastHit.position.x, lastHit.position.y, lastHit.position.z),
                                   GlobalVector(lastHit.momentum.x, lastHit.momentum.y, lastHit.momentum.z),
                                   lastHit.charge,
                                   &field),
        CurvilinearTrajectoryError(error));
    auto const& line = state.beamLine;
    const reco::BeamSpot spot(reco::BeamSpot::Point(line.position.x, line.position.y, line.position.z),
                              5.,
                              line.direction.x,
                              line.direction.y,
                              0.001,
                              reco::BeamSpot::CovarianceMatrix());
    const TrajectoryStateClosestToBeamLine tscbl = builder(fts, spot);

    if ((result.status == Result::kValid) != tscbl.isValid()) {
      ++nValidMismatch;
      UNSCOPED_INFO("state " << i << ": valid on the host " << tscbl.isValid() << ", status " << int(result.status));
      continue;
    }
    if (!tscbl.isValid())
      continue;
    ++nValid;
    FreeTrajectoryState const& atPCA = tscbl.trackStateAtPCA();
    const GlobalPoint hostPosition = atPCA.position();
    const GlobalVector hostMomentum = atPCA.momentum();
    float hostDxy = 0.f, hostDz = 0.f, hostPtErr = 0.f, hostEtaErr = 0.f;
    lst::pixelSegmentImpactParameters(hostPosition.x(),
                                      hostPosition.y(),
                                      hostPosition.z(),
                                      hostMomentum.x(),
                                      hostMomentum.y(),
                                      hostMomentum.z(),
                                      hostMomentum.perp(),
                                      line.position.x,
                                      line.position.y,
                                      line.position.z,
                                      hostDxy,
                                      hostDz);
    const AlgebraicSymMatrix55 perigee = PerigeeConversions::ftsToPerigeeError(atPCA).covarianceMatrix();
    lst::pixelSegmentMomentumErrors(hostMomentum.perp(),
                                    hostMomentum.mag(),
                                    hostMomentum.z(),
                                    atPCA.charge(),
                                    perigee(0, 0),
                                    perigee(0, 1),
                                    perigee(1, 1),
                                    hostPtErr,
                                    hostEtaErr);

    CHECK(result.charge == atPCA.charge());
    const double unit = positionUnit(state, bField);
    dxy.update(std::abs(result.dxy - hostDxy) / unit, i);
    dz.update(std::abs(result.dz - hostDz) / unit, i);
    momentum.update(
        std::hypot(result.px - hostMomentum.x(), result.py - hostMomentum.y(), result.pz - hostMomentum.z()) /
            hostMomentum.mag(),
        i);
    // errors relative to themselves, as the covariance elements relative to sqrt(C_ii C_jj)
    ptErr.update(
        std::abs(double(result.ptErr) * result.ptErr - double(hostPtErr) * hostPtErr) / (double(hostPtErr) * hostPtErr),
        i);
    etaErr.update(std::abs(double(result.etaErr) * result.etaErr - double(hostEtaErr) * hostEtaErr) /
                      (double(hostEtaErr) * hostEtaErr),
                  i);
  }

  std::cout << "closest approach against TSCBLBuilderNoMaterial: " << kNStates << " states, " << nValid
            << " valid on both, " << nValidMismatch << " with different validity (tolerance "
            << kValidMismatchTolerance * kNStates << ")\n"
            << "  max |dxy difference| " << dxy.value << " host-precision units (state " << dxy.index
            << "), |dz difference| " << dz.value << " (state " << dz.index << "); tolerance " << kPositionTolerance
            << "\n"
            << "  max |momentum difference| / |p| " << momentum.value << " (state " << momentum.index << "); tolerance "
            << kMomentumTolerance << "\n"
            << "  max |variance difference| / variance: ptErr " << ptErr.value << " (state " << ptErr.index
            << "), etaErr " << etaErr.value << " (state " << etaErr.index << "); tolerance " << kErrorTolerance
            << std::endl;
  CHECK(nValidMismatch <= kValidMismatchTolerance * kNStates);
  CHECK(nValid > 0.99 * kNStates);
  CHECK(dxy.value < kPositionTolerance);
  CHECK(dz.value < kPositionTolerance);
  CHECK(momentum.value < kMomentumTolerance);
  CHECK(ptErr.value < kErrorTolerance);
  CHECK(etaErr.value < kErrorTolerance);
}

TEST_CASE("closest approach against the exact helix-line closest approach", "[PixelSegmentClosestApproach]") {
  const float bField = blfTestStates::bField();
  const lst::UniformField uniformField(bField);
  const std::vector<TestState> states = blfTestStates::makeStates(kNStates, kSeed);

  // states whose closest approach is not found (by the host as well, compared above) are counted and skipped; the
  // differences are also given per bin of the turning angle from the vertex to the state point. Only states that
  // turned by at most kMaxTurnAngle are asserted: beyond about 1 rad of turn the search can converge to the far side of
  // the circle, as the release's TSCBLBuilderNoMaterial does. Pixel-track states in the Phase-2 HLT sit on the last
  // pixel-track hit, in the pixel detector or the innermost outer-tracker layers, where pLS above the pT cut turn by
  // well under 1 rad; the cut at 0.9 rad keeps a margin. The other states are only counted and printed.
  int nValid = 0, nNotConverged = 0, nAsserted = 0, nReported = 0, nNotConvergedAsserted = 0;
  int nReportedNotConvergedInBin[blfTestStates::kNTurnBins] = {}, nReportedBeyondInBin[blfTestStates::kNTurnBins] = {},
      nReportedPassedInBin[blfTestStates::kNTurnBins] = {};
  MaxDifference dxy[2], dz[2], momentum[2];
  double maxAbsDxy[2] = {}, maxAbsDz[2] = {};
  int nInBin[blfTestStates::kNTurnBins] = {}, nBeyondInBin[blfTestStates::kNTurnBins] = {},
      nNotConvergedInBin[blfTestStates::kNTurnBins] = {};
  MaxDifference positionInBin[blfTestStates::kNTurnBins];
  for (int i = 0; i < kNStates; ++i) {
    TestState const& state = states[i];
    const int bin = blfTestStates::turnBin(state, bField);
    ++nInBin[bin];
    const bool asserted = blfTestStates::turnAngle(state, bField) <= blfTestStates::kMaxTurnAngle;
    ++(asserted ? nAsserted : nReported);
    const Result result = lst::closestApproach(state.row, uniformField, state.beamLine);
    CHECK(result.status != Result::kNonFinite);
    if (result.status != Result::kValid) {
      ++nNotConverged;
      ++nNotConvergedInBin[bin];
      if (asserted)
        ++nNotConvergedAsserted;
      else
        ++nReportedNotConvergedInBin[bin];
      continue;
    }
    ++nValid;
    double position[3], trueMomentum[3];
    const bool exactConverged = exactClosestApproach(state, bField, position, trueMomentum);
    if (asserted)
      REQUIRE(exactConverged);
    else if (!exactConverged) {
      ++nBeyondInBin[bin];
      ++nReportedBeyondInBin[bin];
      continue;
    }
    double trueDxy = 0., trueDz = 0.;
    impactParameters(position, trueMomentum, state.beamLine.position, trueDxy, trueDz);

    // the precision of the host closest approach, plus the float rounding of the stored state (6e-8 of the
    // coordinates, of phi, of kappa and of cot) carried over the path length from the state point to the beam line
    const double pathLength = state.arcToHit * std::sqrt(1. + state.cot * state.cot);
    const double hitDistance = std::abs(state.row.x) + std::abs(state.row.y) + std::abs(state.row.z);
    const double unit = positionUnit(state, bField) + kFloatRounding * (hitDistance + 4. * pathLength);
    const int prompt = state.prompt ? 0 : 1;
    const double dxyUnits = std::abs(result.dxy - trueDxy) / unit, dzUnits = std::abs(result.dz - trueDz) / unit;
    const double momentumDifference =
        std::hypot(result.px - trueMomentum[0], result.py - trueMomentum[1], result.pz - trueMomentum[2]) /
        std::sqrt(trueMomentum[0] * trueMomentum[0] + trueMomentum[1] * trueMomentum[1] +
                  trueMomentum[2] * trueMomentum[2]);
    if (asserted) {
      dxy[prompt].update(dxyUnits, i);
      dz[prompt].update(dzUnits, i);
      momentum[prompt].update(momentumDifference, i);
      maxAbsDxy[prompt] = std::max(maxAbsDxy[prompt], std::abs(result.dxy - trueDxy));
      maxAbsDz[prompt] = std::max(maxAbsDz[prompt], std::abs(result.dz - trueDz));
    }
    positionInBin[bin].update(std::max(dxyUnits, dzUnits), i);
    const bool withinTolerance =
        dxyUnits < kPositionTolerance && dzUnits < kPositionTolerance && momentumDifference < kMomentumTolerance;
    if (!withinTolerance)
      ++nBeyondInBin[bin];
    if (!asserted)
      ++(withinTolerance ? nReportedPassedInBin[bin] : nReportedBeyondInBin[bin]);
  }

  std::cout << "closest approach against the exact helix-line closest approach: " << kNStates << " states, " << nValid
            << " valid, " << nNotConverged << " not converged\n"
            << "  asserted (turning angle up to " << blfTestStates::kMaxTurnAngle << " rad): " << nAsserted
            << " states, " << nNotConvergedAsserted << " of them not converged; reported only: " << nReported
            << " states\n";
  char const* labels[2] = {"vertex within 0.05 cm of the beam line", "vertex within 2 cm of the beam line"};
  for (int prompt = 0; prompt < 2; ++prompt)
    std::cout << "  " << labels[prompt] << ": max |dxy difference| " << maxAbsDxy[prompt] << " cm, "
              << dxy[prompt].value << " units (state " << dxy[prompt].index << "); max |dz difference| "
              << maxAbsDz[prompt] << " cm, " << dz[prompt].value << " units (state " << dz[prompt].index
              << "); max |momentum difference| / |p| " << momentum[prompt].value << " (state " << momentum[prompt].index
              << ")\n";
  std::cout << "  per turning angle from the vertex to the state point (rad): states, not converged, beyond a "
               "tolerance, max dxy or dz difference in units\n";
  for (int bin = 0; bin < blfTestStates::kNTurnBins; ++bin)
    std::cout << "    from " << bin * blfTestStates::kTurnBinWidth << ": " << nInBin[bin] << ", "
              << nNotConvergedInBin[bin] << ", " << nBeyondInBin[bin] << ", " << positionInBin[bin].value << "\n";
  std::cout << "  reported only (turning angle above " << blfTestStates::kMaxTurnAngle
            << " rad), per turning angle (rad): not converged, converged beyond a tolerance (far side or otherwise), "
               "passed\n";
  for (int bin = 0; bin < blfTestStates::kNTurnBins; ++bin)
    if (nReportedNotConvergedInBin[bin] + nReportedBeyondInBin[bin] + nReportedPassedInBin[bin] > 0)
      std::cout << "    from " << bin * blfTestStates::kTurnBinWidth << ": " << nReportedNotConvergedInBin[bin] << ", "
                << nReportedBeyondInBin[bin] << ", " << nReportedPassedInBin[bin] << "\n";
  std::cout << "  tolerances: dxy, dz " << kPositionTolerance << " units, momentum " << kMomentumTolerance << std::endl;
  for (int prompt = 0; prompt < 2; ++prompt) {
    CHECK(dxy[prompt].value < kPositionTolerance);
    CHECK(dz[prompt].value < kPositionTolerance);
    CHECK(momentum[prompt].value < kMomentumTolerance);
  }
  CHECK(nNotConvergedAsserted == 0);
  // guards against the turning angle cut excluding most of the states
  REQUIRE(nAsserted >= 0.95 * kNStates);
}

TEST_CASE("closest approach in a uniform field: kept pT and pz, closed-form errors", "[PixelSegmentClosestApproach]") {
  const float bField = blfTestStates::bField();
  const lst::UniformField uniformField(bField);
  const std::vector<TestState> states = blfTestStates::makeStates(kNStates, kSeed);

  // pT at the closest approach is rebuilt from the float momentum, with tolerance:
  // - sin/cos and products on the state point (about 2 ulp),
  // - the transverse momentum and its rotation in the closest approach (about 3 ulp),
  // - the caller's sqrt (about 2 ulp)
  constexpr double kPtTolerance = 8. * kFloatRounding;
  constexpr double kClosedFormTolerance = 1.e-5;
  MaxDifference pt, ptErr, etaErr;
  int nPzDifferent = 0, nValid = 0;
  for (int i = 0; i < kNStates; ++i) {
    lst::BLFStateRow const& row = states[i].row;
    const Result result = lst::closestApproach(row, uniformField, states[i].beamLine);
    CHECK(result.status != Result::kNonFinite);
    if (result.status != Result::kValid)
      continue;
    ++nValid;
    const float ptIn = bField / std::abs(row.kappa);
    const float pzIn = ptIn * row.cotTheta;
    const float ptPCA = std::sqrt(result.px * result.px + result.py * result.py);
    pt.update(std::abs(ptPCA - ptIn) / ptIn, i);
    if (result.pz != pzIn)
      ++nPzDifferent;

    // var(rho) = var(kappa), cov(rho, theta) = 0, var(theta) = sin^4(theta) var(cot), so
    // ptErr^2 = pT^2 p^2 var(kappa) + pz^2 sin^4(theta) var(cot) and etaErr = sin(theta) sigma(cot)
    const double pT = ptPCA, pz = result.pz, p = std::sqrt(pT * pT + pz * pz);
    const double sinTheta = 1. / std::sqrt(1. + double(row.cotTheta) * row.cotTheta);
    const double varKappa = row.covCircle[5], varCot = row.covLine[0];
    const double ptErrClosed = std::sqrt(pT * pT * p * p * varKappa + pz * pz * std::pow(sinTheta, 4) * varCot);
    const double etaErrClosed = sinTheta * std::sqrt(varCot);
    ptErr.update(std::abs(result.ptErr - ptErrClosed) / ptErrClosed, i);
    etaErr.update(std::abs(result.etaErr - etaErrClosed) / etaErrClosed, i);
  }
  std::cout << "closest approach in a uniform field: " << nValid << " valid states\n"
            << "  max |ptPCA / ptIn - 1| " << pt.value << " (state " << pt.index << "), tolerance " << kPtTolerance
            << "; pzPCA != pz in " << nPzDifferent << " states\n"
            << "  max relative difference from the closed forms: ptErr " << ptErr.value << " (state " << ptErr.index
            << "), etaErr " << etaErr.value << " (state " << etaErr.index << "); tolerance " << kClosedFormTolerance
            << std::endl;
  CHECK(pt.value < kPtTolerance);
  CHECK(nPzDifferent == 0);
  CHECK(ptErr.value < kClosedFormTolerance);
  CHECK(etaErr.value < kClosedFormTolerance);
}

TEST_CASE("closest approach of invalid states", "[PixelSegmentClosestApproach]") {
  const float bField = blfTestStates::bField();
  const lst::UniformField uniformField(bField);
  const std::vector<TestState> states = blfTestStates::makeStates(1, kSeed);
  TestState const& good = states[0];
  REQUIRE(lst::closestApproach(good.row, uniformField, good.beamLine).status == Result::kValid);
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInfinity = std::numeric_limits<float>::infinity();

  auto check = [&](lst::BLFStateRow const& row, uint8_t status, char const* what) {
    const Result result = lst::closestApproach(row, uniformField, good.beamLine);
    INFO(what);
    CHECK(int(result.status) == int(status));
    CHECK(isZero(result));
  };

  {
    auto row = good.row;
    row.kappa = 0.f;
    check(row, Result::kNonFinite, "kappa = 0");
    row.kappa = -0.f;
    check(row, Result::kNonFinite, "kappa = -0");
  }
  {
    auto row = good.row;
    row.charge = 0;
    check(row, Result::kNotConverged, "q = 0");
  }
  {
    // an infinite kappa is rejected by the finiteness check of the input
    auto row = good.row;
    row.kappa = kInfinity;
    check(row, Result::kNonFinite, "kappa = inf");
  }
  for (float bad : {kNaN, kInfinity, -kInfinity}) {
    for (int k = 0; k < 6; ++k) {
      auto row = good.row;
      float* values[6] = {&row.x, &row.y, &row.z, &row.phi, &row.cotTheta, &row.kappa};
      *values[k] = bad;
      check(row, Result::kNonFinite, "non-finite parameter");
    }
    for (int k = 0; k < 6; ++k) {
      auto row = good.row;
      row.covCircle[k] = bad;
      check(row, Result::kNonFinite, "non-finite circle covariance");
    }
    for (int k = 0; k < 3; ++k) {
      auto row = good.row;
      row.covLine[k] = bad;
      check(row, Result::kNonFinite, "non-finite line covariance");
    }
  }
  {
    // finite input whose ptErr overflows a float: an output check
    auto row = good.row;
    row.kappa = std::copysign(1.e-20f, row.kappa);
    row.covCircle[5] = 1.e10f;
    const Result result = lst::closestApproach(row, uniformField, good.beamLine);
    INFO("ptErr overflow, status " << int(result.status));
    CHECK(result.status == Result::kNonFinite);
    CHECK(isZero(result));
  }
}
