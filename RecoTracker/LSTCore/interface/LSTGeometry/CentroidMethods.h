#ifndef RecoTracker_LSTCore_interface_LSTGeometry_CentroidMethods_h
#define RecoTracker_LSTCore_interface_LSTGeometry_CentroidMethods_h

#include <stdexcept>
#include <unordered_map>

#include "Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Module.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Centroid.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/SensorInfo.h"

namespace lstgeometry {

  unsigned int extractBits(unsigned int value, unsigned int start, unsigned int end) {
    unsigned int mask = (1 << (end - start + 1)) - 1;
    return (value >> start) & mask;
  }

  unsigned int firstDigit(unsigned int n) {
    while (n >= 10) {
      n /= 10;
    }
    return n;
  }

  // TODO: refactor to use Module class better
  int parseModuleType(unsigned int detId) {
    // Check if the first digit of detId is '3' for inner tracker
    if (firstDigit(detId) == 3)
      return -1;

    unsigned int subdet = extractBits(detId, 25, 27);
    unsigned int layer = subdet == Module::SubDet::Barrel ? extractBits(detId, 20, 22) : extractBits(detId, 18, 20);
    unsigned int ring = subdet == Module::SubDet::Endcap ? extractBits(detId, 12, 15) : 0;

    bool is_even_det_id = detId % 2 == 0;
    if (subdet == Module::SubDet::Barrel) {
      if (layer <= 3)
        return is_even_det_id ? Module::ModuleType::PSS : Module::ModuleType::PSP;
      else
        return Module::ModuleType::TwoS;
    } else if (subdet == Module::SubDet::Endcap) {
      if (layer <= 2)
        return is_even_det_id && ring <= 10 ? Module::ModuleType::PSS
                                            : (ring <= 10 ? Module::ModuleType::PSP : Module::ModuleType::TwoS);
      else
        return is_even_det_id && ring <= 7 ? Module::ModuleType::PSS
                                           : (ring <= 7 ? Module::ModuleType::PSP : Module::ModuleType::TwoS);
    } else {
      throw std::runtime_error("Invalid subdetector type");
    }
  }

  std::unordered_map<unsigned int, Centroid> computeCentroids(
      std::unordered_map<unsigned int, SensorInfo> const& sensors) {
    std::unordered_map<unsigned int, Centroid> centroids;
    for (auto const& [detId, sensor] : sensors) {
      int moduleType = parseModuleType(detId);

      // Remove sensors from inner tracker
      if (moduleType == -1) {
        continue;
      }

      // Convert from mm to cm
      double z = sensor.sensorCenterZ_cm;
      double rho = sensor.sensorCenterRho_cm;
      double phi = sensor.phi_rad;
      double x = rho * cos(phi);
      double y = rho * sin(phi);

      Centroid centroid{static_cast<unsigned int>(moduleType), x, y, z};
      centroids[detId] = centroid;
    }
    return centroids;
  }

}  // namespace lstgeometry

#endif
