#ifndef RecoTracker_LST_test_BLFTestStates_h
#define RecoTracker_LST_test_BLFTestStates_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "RecoTracker/LST/interface/PixelSegmentClosestApproach.h"

// Random Broken Line Fit states for the closest-approach tests: noiseless helices in a uniform 3.8 T field from a
// vertex near a tilted beam line, the stored float state taken at a point of the helix (the last hit), with
// positive-definite covariance blocks of the size the fit gives. Plain types only, shared by the host and the
// device tests.
namespace blfTestStates {

  constexpr float kFieldTesla = 3.8f;
  // the B of the fit, 1 / inverseBzAtOriginInGeV() of a uniform 3.8 T field
  inline float bField() {
    const float inverseBzAtOriginInGeV = 1.f / (kFieldTesla * 2.99792458e-3f);  // as MagneticField::setNominalValue
    return static_cast<float>(1. / inverseBzAtOriginInGeV);
  }

  struct TestState {
    lst::BLFStateRow row;
    beamLineClosestApproach::BeamLine beamLine;
    // the generating helix in double: vertex, momentum azimuth there, pT, cot, charge, and the transverse arc length
    // from the vertex to the state point
    double vertex[3];
    double phi0, pt, cot;
    int charge;
    double arcToHit;
    bool prompt;  // vertex within 0.05 cm of the beam line (else within 2 cm)
  };

  // the helix of a state in double: position and momentum azimuth after the transverse arc length s from the vertex;
  // dphi/ds_T = w = -q B / pT (positive charges turn clockwise for Bz > 0). The chord 2 sin(w s / 2) / w along the
  // mean azimuth avoids the cancellation of sin(phi) - sin(phi0) at large radii.
  inline void helixAt(TestState const& state, double bField, double s, double position[3], double& phi) {
    const double w = -state.charge * bField / state.pt;
    const double halfTurn = 0.5 * w * s;
    const double chord = halfTurn != 0. ? s * std::sin(halfTurn) / halfTurn : s;
    phi = state.phi0 + w * s;
    position[0] = state.vertex[0] + chord * std::cos(state.phi0 + halfTurn);
    position[1] = state.vertex[1] + chord * std::sin(state.phi0 + halfTurn);
    position[2] = state.vertex[2] + state.cot * s;
  }

  // the transverse turning angle of the generated helix from its vertex to the state point, rad
  inline double turnAngle(TestState const& state, double bField) { return bField / state.pt * state.arcToHit; }

  // The tests assert only for states whose turning angle (see turnAngle) is at most this value. The Newton search of
  // the closest approach starts at the helix phase of the state point (as the release's TSCBLBuilderNoMaterial does)
  // and can converge to the far side of the circle, or not converge, beyond about 1 rad of turn. Pixel-track states in
  // the Phase-2 HLT sit on the last pixel-track hit, in the pixel detector or the innermost outer-tracker layers,
  // where pLS above the pT cut turn by well under 1 rad; the cut at 0.9 rad keeps a margin. States beyond this angle
  // are still generated, and only counted and printed.
  constexpr double kMaxTurnAngle = 0.9;

  // states binned in that angle, kTurnBinWidth wide, the last bin open
  constexpr int kNTurnBins = 12;
  constexpr double kTurnBinWidth = 0.25;
  inline int turnBin(TestState const& state, double bField) {
    return std::min(int(turnAngle(state, bField) / kTurnBinWidth), kNTurnBins - 1);
  }

  // positive-definite covariance L L^T of dimension n with the given standard deviations and random correlations
  template <int n>
  void randomCovariance(double const (&sigma)[n], std::mt19937& engine, double (&covariance)[n][n]) {
    std::uniform_real_distribution<double> uniform(0., 1.);
    double lower[n][n] = {};
    for (int row = 0; row < n; ++row)
      for (int col = 0; col <= row; ++col)
        lower[row][col] = sigma[row] * (row == col ? 0.5 + uniform(engine) : 0.6 * (uniform(engine) - 0.5));
    for (int row = 0; row < n; ++row)
      for (int col = 0; col < n; ++col) {
        double sum = 0;
        for (int k = 0; k < n; ++k)
          sum += lower[row][k] * lower[col][k];
        covariance[row][col] = sum;
      }
  }

  // pT log-uniform in 0.5-100 GeV, |eta| < 4, both charges, last-hit transverse distance from the vertex 5-110 cm
  // (at most 1.9 helix radii, |z| at most 280 cm)
  inline std::vector<TestState> makeStates(int nStates, uint32_t seed) {
    std::mt19937 engine(seed);
    std::uniform_real_distribution<double> uniform(0., 1.);
    const double b = bField();
    std::vector<TestState> states(nStates);
    for (TestState& state : states) {
      state.pt = 0.5 * std::exp(std::log(200.) * uniform(engine));
      const double eta = 8. * uniform(engine) - 4.;
      state.cot = std::sinh(eta);
      state.phi0 = 2. * M_PI * uniform(engine) - M_PI;
      state.charge = uniform(engine) < 0.5 ? -1 : 1;

      const float beamX = 0.4 * uniform(engine) - 0.2, beamY = 0.4 * uniform(engine) - 0.2;
      const float beamZ = 10. * uniform(engine) - 5.;
      const float slopeX = 2.e-3 * uniform(engine) - 1.e-3, slopeY = 2.e-3 * uniform(engine) - 1.e-3;
      state.beamLine = {{beamX, beamY, beamZ}, {slopeX, slopeY, 1.f}};
      state.prompt = uniform(engine) < 0.75;
      const double offset = state.prompt ? 0.05 : 2.;
      const double alongLine = 30. * uniform(engine) - 15.;
      state.vertex[0] = beamX + slopeX * alongLine + offset * (2. * uniform(engine) - 1.);
      state.vertex[1] = beamY + slopeY * alongLine + offset * (2. * uniform(engine) - 1.);
      state.vertex[2] = beamZ + alongLine;

      const double radius = state.pt / b;
      const double chord = std::min(5. + 105. * uniform(engine), 1.9 * radius);
      double arc = 2. * radius * std::asin(chord / (2. * radius));
      if (state.cot != 0.)
        arc = std::min(arc, (280. - std::abs(state.vertex[2])) / std::abs(state.cot));
      state.arcToHit = arc;

      double position[3], phi;
      helixAt(state, b, arc, position, phi);
      const double kappa = state.charge * b / state.pt;
      double circle[3][3], line[2][2];
      const double sigmaCircle[3] = {
          2.e-3 * (0.5 + uniform(engine)), 2.e-3 * (0.5 + uniform(engine)), 0.03 * std::abs(kappa) + 2.e-5};
      const double sigmaLine[2] = {2.e-3 * (1. + state.cot * state.cot), 1.e-2 * (0.5 + uniform(engine))};
      randomCovariance(sigmaCircle, engine, circle);
      randomCovariance(sigmaLine, engine, line);

      lst::BLFStateRow& row = state.row;
      row.x = position[0];
      row.y = position[1];
      row.z = position[2];
      row.phi = std::remainder(phi, 2. * M_PI);
      row.cotTheta = state.cot;
      row.kappa = kappa;
      row.covCircle[0] = circle[0][0];
      row.covCircle[1] = circle[0][1];
      row.covCircle[2] = circle[1][1];
      row.covCircle[3] = circle[0][2];
      row.covCircle[4] = circle[1][2];
      row.covCircle[5] = circle[2][2];
      row.covLine[0] = line[0][0];
      row.covLine[1] = line[0][1];
      row.covLine[2] = line[1][1];
      row.charge = state.charge;
    }
    return states;
  }

}  // namespace blfTestStates

#endif  // RecoTracker_LST_test_BLFTestStates_h
