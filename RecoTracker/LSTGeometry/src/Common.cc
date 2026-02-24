#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  double degToRad(double degrees) { return degrees * (std::numbers::pi_v<double> / 180); }

  double phi_mpi_pi(double phi) {
    while (phi >= std::numbers::pi_v<double>)
      phi -= 2 * std::numbers::pi_v<double>;
    while (phi < -std::numbers::pi_v<double>)
      phi += 2 * std::numbers::pi_v<double>;
    return phi;
  }

  double roundAngle(double angle, double tol) {
    const double pi = std::numbers::pi_v<double>;
    if (std::fabs(angle) < tol) {
      return 0.0;
    } else if (std::fabs(angle - pi / 2) < tol) {
      return pi / 2;
    } else if (std::fabs(angle + pi / 2) < tol) {
      return -pi / 2;
    } else if (std::fabs(angle - pi) < tol || std::fabs(angle + pi) < tol) {
      return -pi;
    }
    return angle;
  }

  double roundCoordinate(double coord, double tol) {
    if (std::fabs(coord) < tol) {
      return 0.0;
    }
    return coord;
  }

}  // namespace lstgeometry
