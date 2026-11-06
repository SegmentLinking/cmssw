#ifndef TrackingTools_PatternTools_interface_beamLineClosestApproach_h
#define TrackingTools_PatternTools_interface_beamLineClosestApproach_h

#include <cmath>

#include "TrackingTools/AnalyticalJacobians/interface/curvilinearJacobian.h"
#include "TrackingTools/PatternTools/interface/helixLineClosestApproach.h"

/*
 * TSCBLBuilderNoMaterial as plain constexpr functions, usable on the host and on GPUs: the state of a track at
 * its closest approach to the beam line, with the curvilinear errors transported by the helix Jacobian.
 * The caller provides the magnetic field at the track position.
 */
namespace beamLineClosestApproach {

  using Vector3 = curvilinearJacobian::Vector3;

  // as FreeTrajectoryState: global position (cm), momentum (GeV), charge, curvilinear error
  struct FreeState {
    Vector3 position, momentum;
    int charge;
    double error[5][5];
  };

  // as TSCBLBuilderNoMaterial makes it from the BeamSpot: GlobalPoint(position()), GlobalVector(dxdz(), dydz(), 1)
  struct BeamLine {
    Vector3 position, direction;
  };

  // error' = jacobian error jacobian^T
  constexpr void similarity(double const jacobian[5][5], double const error[5][5], double result[5][5]) {
    double product[5][5];
    for (int row = 0; row < 5; ++row)
      for (int col = 0; col < 5; ++col) {
        double sum = 0;
        for (int k = 0; k < 5; ++k)
          sum += jacobian[row][k] * error[k][col];
        product[row][col] = sum;
      }
    for (int row = 0; row < 5; ++row)
      for (int col = 0; col <= row; ++col) {
        double sum = 0;
        for (int k = 0; k < 5; ++k)
          sum += product[row][k] * jacobian[col][k];
        result[row][col] = result[col][row] = sum;
      }
  }

  // TSCBLBuilderNoMaterial()(track, beamSpot) for a charged track in a field with Bz != 0 at its position
  // (fieldInTesla); returns false where the host state is invalid (no convergence) or the track is not a helix.
  constexpr bool stateNoMaterial(FreeState const& track,
                                 Vector3 fieldInTesla,
                                 BeamLine const& beamLine,
                                 FreeState& result) {
    // TwoTrackMinimumDistanceHelixLine with the track as the helix, the beam line as the line
    constexpr int maxIterations = 12;
    constexpr float tolerance = 1.e-6f;
    const double helixP = std::sqrt(curvilinearJacobian::mag2(track.momentum));
    const double lineP = std::sqrt(curvilinearJacobian::mag2(beamLine.direction));
    const double bc2k = fieldInTesla.z * 2.99792458e-3;
    if (track.charge == 0 || fieldInTesla.z == 0.f || helixP == 0. || lineP == 0.)
      return false;
    const auto coeffs = helixLineClosestApproach::coefficients(beamLine.position.x - track.position.x,
                                                               beamLine.position.y - track.position.y,
                                                               beamLine.position.z - track.position.z,
                                                               beamLine.direction.x,
                                                               beamLine.direction.y,
                                                               beamLine.direction.z,
                                                               helixP,
                                                               track.momentum.z,
                                                               track.charge,
                                                               bc2k,
                                                               std::atan2(track.momentum.y, track.momentum.x));
    double phiH = coeffs.phiH0;
    const double phiMin = coeffs.phiH0 - M_PI, phiMax = coeffs.phiH0 + M_PI;
    bool converged = false;
    for (int iteration = 1; iteration <= maxIterations; ++iteration) {
      double fct = 0, derivative = 0;
      helixLineClosestApproach::functionAndDerivative(coeffs, phiH, fct, derivative);
      const double dPhiH = fct / derivative;
      phiH -= dPhiH;
      if ((phiMin - phiH) * (phiH - phiMax) < 0.0)
        phiH += (dPhiH * 0.8);
      if (std::fabs(dPhiH) < tolerance) {
        converged = true;
        break;
      }
    }
    if (!converged)
      return false;

    double dx = 0, dy = 0, dz = 0;
    helixLineClosestApproach::helixOffset(coeffs, phiH, dx, dy, dz);
    result.position = Vector3{float(track.position.x + dx), float(track.position.y + dy), float(track.position.z + dz)};
    // GlobalVector(GlobalVector::Cylindrical(perp, phi, pz)): float phi, products in double
    const float momentumPerp = curvilinearJacobian::perp(track.momentum);
    const float phiFloat = phiH;
    result.momentum = Vector3{float(momentumPerp * std::cos(double(phiFloat))),
                              float(momentumPerp * std::sin(double(phiFloat))),
                              track.momentum.z};
    result.charge = track.charge;

    double jacobian[5][5];
    curvilinearJacobian::compute(track.position,
                                 track.momentum,
                                 track.charge,
                                 result.position,
                                 result.momentum,
                                 fieldInTesla,
                                 helixLineClosestApproach::helixPathLength(coeffs, phiH),
                                 jacobian);
    similarity(jacobian, track.error, result.error);
    return true;
  }

}  // namespace beamLineClosestApproach

#endif  // TrackingTools_PatternTools_interface_beamLineClosestApproach_h
