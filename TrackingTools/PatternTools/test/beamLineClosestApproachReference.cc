#include <algorithm>
#include <cmath>
#include <random>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "TrackingTools/PatternTools/test/beamLineClosestApproachReference.h"

namespace {

  // solenoid-like test field: Bz falls with |z|, a radial component grows with |z|
  class TestField final : public MagneticField {
  public:
    TestField() { setNominalValue(); }
    GlobalVector inTesla(const GlobalPoint& point) const override {
      const float z = point.z();
      return GlobalVector(2.e-6f * point.x() * z, 2.e-6f * point.y() * z, 3.8f - 6.e-6f * z * z);
    }
  };

  const TestField testField;

  GlobalPoint globalPoint(beamLineClosestApproachTest::Vector3 const& vec) { return GlobalPoint(vec.x, vec.y, vec.z); }
  GlobalVector globalVector(beamLineClosestApproachTest::Vector3 const& vec) {
    return GlobalVector(vec.x, vec.y, vec.z);
  }
  beamLineClosestApproachTest::Vector3 vector3(GlobalPoint const& point) {
    return beamLineClosestApproachTest::Vector3{point.x(), point.y(), point.z()};
  }
  beamLineClosestApproachTest::Vector3 vector3(GlobalVector const& vec) {
    return beamLineClosestApproachTest::Vector3{vec.x(), vec.y(), vec.z()};
  }

  reco::BeamSpot beamSpot(beamLineClosestApproachTest::BeamLine const& beamLine) {
    return reco::BeamSpot(reco::BeamSpot::Point(beamLine.position.x, beamLine.position.y, beamLine.position.z),
                          5.,
                          beamLine.direction.x,
                          beamLine.direction.y,
                          0.001,
                          reco::BeamSpot::CovarianceMatrix());
  }

}  // namespace

namespace beamLineClosestApproachTest {

  Inputs makeInputs(int nTracks, uint32_t seed) {
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> uniform(0.f, 1.f);
    Inputs inputs;
    inputs.tracks.resize(nTracks);
    inputs.fields.resize(nTracks);
    inputs.beamLines.resize(nTracks);
    for (int i = 0; i < nTracks; ++i) {
      FreeState& track = inputs.tracks[i];
      const float pt = 0.3f * std::exp(std::log(200.f / 0.3f) * uniform(engine));
      const float eta = 8.f * uniform(engine) - 4.f;
      float phiMomentum = float(2 * M_PI) * uniform(engine);
      track.charge = uniform(engine) < 0.5f ? -1 : 1;
      if (i % 4 != 3) {
        // from a vertex near the beam line, moved along a helix (Bz = 3.8 T) to a point inside the tracker
        const float vertexX = 0.4f * uniform(engine) - 0.2f, vertexY = 0.4f * uniform(engine) - 0.2f;
        const float vertexZ = 30.f * uniform(engine) - 15.f;
        const float helixRadius = pt / (2.99792458e-3f * 3.8f);
        const float hitRadius = std::min(3.f + 107.f * uniform(engine), 1.9f * helixRadius);
        float turn = 2.f * std::asin(hitRadius / (2.f * helixRadius));
        const float maxTurn = std::abs(eta) > 0.01f ? 260.f / (helixRadius * std::abs(std::sinh(eta))) : turn;
        turn = std::min(turn, maxTurn);
        const float sign = track.charge;  // positive tracks turn clockwise in +z field
        track.position =
            Vector3{vertexX + sign * helixRadius * (std::sin(phiMomentum) - std::sin(phiMomentum - sign * turn)),
                    vertexY + sign * helixRadius * (std::cos(phiMomentum - sign * turn) - std::cos(phiMomentum)),
                    vertexZ + helixRadius * turn * std::sinh(eta)};
        phiMomentum -= sign * turn;
      } else {
        // anywhere in the tracker, any direction
        const float radius = 110.f * uniform(engine);
        const float phiPosition = float(2 * M_PI) * uniform(engine);
        track.position =
            Vector3{radius * std::cos(phiPosition), radius * std::sin(phiPosition), 540.f * uniform(engine) - 270.f};
      }
      track.momentum = Vector3{pt * std::cos(phiMomentum), pt * std::sin(phiMomentum), pt * std::sinh(eta)};
      // positive-definite curvilinear error L L^T: q/p, lambda, phi, x_T, y_T
      const float momentum = std::sqrt(curvilinearJacobian::mag2(track.momentum));
      const double scale[5] = {0.02 / momentum, 1.e-3, 1.e-3, 5.e-3, 5.e-3};
      double lower[5][5] = {};
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col <= row; ++col)
          lower[row][col] = scale[row] * (row == col ? 0.5 + uniform(engine) : 0.6 * (uniform(engine) - 0.5));
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col < 5; ++col) {
          double sum = 0;
          for (int k = 0; k < 5; ++k)
            sum += lower[row][k] * lower[col][k];
          track.error[row][col] = sum;
        }
      inputs.fields[i] = vector3(testField.inTesla(globalPoint(track.position)));
      BeamLine& beamLine = inputs.beamLines[i];
      beamLine.position =
          Vector3{0.4f * uniform(engine) - 0.2f, 0.4f * uniform(engine) - 0.2f, 10.f * uniform(engine) - 5.f};
      beamLine.direction = Vector3{2.e-3f * uniform(engine) - 1.e-3f, 2.e-3f * uniform(engine) - 1.e-3f, 1.f};
    }
    return inputs;
  }

  std::vector<HostResult> hostResults(Inputs const& inputs) {
    const TSCBLBuilderNoMaterial builder;
    std::vector<HostResult> results(inputs.tracks.size());
    for (std::size_t i = 0; i < inputs.tracks.size(); ++i) {
      FreeState const& track = inputs.tracks[i];
      HostResult& result = results[i];
      AlgebraicSymMatrix55 error;
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col <= row; ++col)
          error(row, col) = track.error[row][col];
      const GlobalTrajectoryParameters parameters(
          globalPoint(track.position), globalVector(track.momentum), track.charge, &testField);
      const FreeTrajectoryState fts(parameters, CurvilinearTrajectoryError(error));
      const reco::BeamSpot spot = beamSpot(inputs.beamLines[i]);

      const TrajectoryStateClosestToBeamLine tscbl = builder(fts, spot);
      result.valid = tscbl.isValid();
      if (!result.valid)
        continue;
      FreeTrajectoryState const& atPCA = tscbl.trackStateAtPCA();
      result.state.position = vector3(atPCA.position());
      result.state.momentum = vector3(atPCA.momentum());
      result.state.charge = atPCA.charge();
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col < 5; ++col)
          result.state.error[row][col] = atPCA.curvilinearError().matrix()(row, col);

      // the Jacobian TSCBLBuilderNoMaterial uses, from the same closest approach
      TwoTrackMinimumDistance ttmd;
      ttmd.calculate(
          parameters,
          GlobalTrajectoryParameters(GlobalPoint(spot.position().x(), spot.position().y(), spot.position().z()),
                                     GlobalVector(spot.dxdz(), spot.dydz(), 1.),
                                     0,
                                     &testField));
      const GlobalPoint endPosition = ttmd.points().first;
      const GlobalVector endMomentum =
          GlobalVector(GlobalVector::Cylindrical(fts.momentum().perp(), ttmd.firstAngle(), fts.momentum().z()));
      result.endPosition = vector3(endPosition);
      result.endMomentum = vector3(endMomentum);
      result.pathLength = ttmd.pathLength().first;
      const AnalyticalCurvilinearJacobian jacobian(parameters, endPosition, endMomentum, result.pathLength);
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col < 5; ++col)
          result.jacobian[row][col] = jacobian.jacobian()(row, col);
    }
    return results;
  }

}  // namespace beamLineClosestApproachTest
