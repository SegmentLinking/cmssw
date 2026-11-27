#ifndef TrackingTools_PatternTools_interface_TSCBLBuilderNoMaterialPortable_h
#define TrackingTools_PatternTools_interface_TSCBLBuilderNoMaterialPortable_h

#include <cmath>

#include <Eigen/Core>

#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobianPortable.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixLinePortable.h"

/*
 * TSCBLBuilderNoMaterial, usable on the host and on GPUs: the state of a track at its closest approach to the beam
 * line, with the curvilinear errors transported by the helix Jacobian. The caller provides the magnetic field at the
 * track position.
 */
struct TSCBLBuilderNoMaterialPortable {
  using Vector3 = AnalyticalCurvilinearJacobianPortable::Vector3;
  using Matrix5 = AnalyticalCurvilinearJacobianPortable::Matrix5;

  // as FreeTrajectoryState: global position (cm), momentum (GeV), charge, curvilinear error
  struct FreeState {
    Vector3 position, momentum;
    int charge;
    Matrix5 error;
  };

  // as TSCBLBuilderNoMaterial makes it from the BeamSpot: GlobalPoint(position()), GlobalVector(dxdz(), dydz(), 1)
  struct BeamLine {
    Vector3 position, direction;
  };

  // TSCBLBuilderNoMaterial()(track, beamSpot).trackStateAtPCA() for a charged track in a field with Bz != 0 at its
  // position (fieldInTesla); returns false where the host state is invalid (no convergence) or the track is not a helix.
  EIGEN_DEVICE_FUNC static bool trackStateAtPCA(FreeState const& track,
                                                Vector3 const& fieldInTesla,
                                                BeamLine const& beamLine,
                                                FreeState& result) {
    using HelixLine = TwoTrackMinimumDistanceHelixLinePortable;
    // TwoTrackMinimumDistanceHelixLine with the track as the helix, the beam line as the line
    // |p| summed in the order of GlobalVector::mag: the Newton search amplifies its rounding
    const Vector3& momentum = track.momentum;
    const double helixP =
        std::sqrt(momentum.x() * momentum.x() + momentum.y() * momentum.y() + momentum.z() * momentum.z());
    const double lineP = beamLine.direction.norm();
    const double bc2k = fieldInTesla.z() * 2.99792458e-3;
    if (track.charge == 0 || fieldInTesla.z() == 0.f || helixP == 0. || lineP == 0.)
      return false;
    const Vector3 posDiff = beamLine.position - track.position;
    const auto coeffs = HelixLine::coefficients(posDiff.x(),
                                                posDiff.y(),
                                                posDiff.z(),
                                                beamLine.direction.x(),
                                                beamLine.direction.y(),
                                                beamLine.direction.z(),
                                                helixP,
                                                track.momentum.z(),
                                                track.charge,
                                                bc2k,
                                                std::atan2(track.momentum.y(), track.momentum.x()));
    double phiH = coeffs.phiH0;
    if (!HelixLine::newtonSearch(coeffs, HelixLine::tolerance, phiH))
      return false;

    double dx = 0, dy = 0, dz = 0;
    HelixLine::helixOffset(coeffs, phiH, dx, dy, dz);
    result.position =
        Vector3(float(track.position.x() + dx), float(track.position.y() + dy), float(track.position.z() + dz));
    // GlobalVector(GlobalVector::Cylindrical(perp, phi, pz)): float phi, products in double
    const float momentumPerp = track.momentum.head<2>().norm();
    const float phiFloat = phiH;
    result.momentum = Vector3(float(momentumPerp * std::cos(double(phiFloat))),
                              float(momentumPerp * std::sin(double(phiFloat))),
                              track.momentum.z());
    result.charge = track.charge;

    Matrix5 jacobian;
    AnalyticalCurvilinearJacobianPortable::compute(track.position,
                                                   track.momentum,
                                                   track.charge,
                                                   result.position,
                                                   result.momentum,
                                                   fieldInTesla,
                                                   HelixLine::helixPathLength(coeffs, phiH),
                                                   jacobian);
    result.error = jacobian * track.error * jacobian.transpose();
    return true;
  }
};

#endif  // TrackingTools_PatternTools_interface_TSCBLBuilderNoMaterialPortable_h
