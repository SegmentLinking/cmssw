#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  float degToRad(float degrees) { return degrees * (std::numbers::pi_v<float> / 180); }

  float phi_mpi_pi(float phi) {
    while (phi >= std::numbers::pi_v<float>)
      phi -= 2 * std::numbers::pi_v<float>;
    while (phi < -std::numbers::pi_v<float>)
      phi += 2 * std::numbers::pi_v<float>;
    return phi;
  }

  float roundAngle(float angle, float tol) {
    const float pi = std::numbers::pi_v<float>;
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

  float roundCoordinate(float coord, float tol) {
    if (std::fabs(coord) < tol) {
      return 0.0;
    }
    return coord;
  }

  std::pair<float, float> getEtaPhi(float x, float y, float z, float refphi) {
    float phi = phi_mpi_pi(std::atan2(y, x) - refphi);
    float eta = std::copysign(-std::log(std::tan(std::atan(std::sqrt(x * x + y * y) / std::abs(z)) / 2.)), z);
    return std::make_pair(eta, phi);
  }

  bool isInverted(unsigned int moduleId, Location location, Side side, unsigned int layer) {
    bool moduleIdIsEven = moduleId % 2 == 0;
    if (location == Location::endcap) {
      if (side == Side::NegZ) {
        return !moduleIdIsEven;
      } else if (side == Side::PosZ) {
        return moduleIdIsEven;
      }
    } else if (location == Location::barrel) {
      if (side == Side::Center) {
        if (layer <= 3) {
          return !moduleIdIsEven;
        } else if (layer >= 4) {
          return moduleIdIsEven;
        }
      } else if (side == Side::NegZ || side == Side::PosZ) {
        if (layer <= 2) {
          return !moduleIdIsEven;
        } else if (layer == 3) {
          return moduleIdIsEven;
        }
      }
    }
    return false;
  }

  bool isLower(unsigned int moduleId, Location location, Side side, unsigned int layer, unsigned int detId) {
    return isInverted(moduleId, location, side, layer) ? !(detId & 1) : (detId & 1);
  }

}  // namespace lstgeometry
