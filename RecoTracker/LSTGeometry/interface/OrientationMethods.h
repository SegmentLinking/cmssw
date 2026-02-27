#ifndef RecoTracker_LSTGeometry_interface_OrientationMethods_h
#define RecoTracker_LSTGeometry_interface_OrientationMethods_h

#include <tuple>
#include <cmath>
#include <unordered_map>
#include <iostream>  /////////////////////// remove

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/SlopeData.h"

namespace lstgeometry {

  bool isStripLayer(Module module, bool isLower) {
    if (module.moduleType == ModuleType::Ph2SS)
      return true;
    if (isInverted(module.moduleId, module.location, module.side, module.layer))
      return isLower;
    return !isLower;
  }

  // Use each sensor's corners to calculate and categorize drdz and dxdy slopes.
  SlopeData calculateSlope(double dx, double dy, double dz) {
    double dr = sqrt(dx * dx + dy * dy);
    double drdz_slope = dz != 0 ? dr / dz : kDefaultSlope;
    double dxdy_slope = dy != 0 ? -dx / dy : kDefaultSlope;
    return SlopeData{drdz_slope, dxdy_slope};
  }

  // Use each sensor's corners to calculate and categorize drdz and dxdy slopes.
  std::tuple<std::unordered_map<unsigned int, SlopeData>, std::unordered_map<unsigned int, SlopeData>> processCorners(
      Modules const& modules, Sensors const& sensors) {
    std::unordered_map<unsigned int, SlopeData> barrel_slopes;
    std::unordered_map<unsigned int, SlopeData> endcap_slopes;

    std::cout << "Number of sensors: " << sensors.size() << std::endl;  /////////////////////// remove

    for (const auto& [detId, sensor] : sensors) {
      double dx = roundCoordinate(sensor.corners(1, 1) - sensor.corners(0, 1));
      double dy = roundCoordinate(sensor.corners(1, 2) - sensor.corners(0, 2));
      double dz = roundCoordinate(sensor.corners(1, 0) - sensor.corners(0, 0));

      SlopeData slope = calculateSlope(dx, dy, dz);

      auto& module = modules.at(sensor.moduleDetId);

      auto location = module.location;
      bool is_tilted = module.side != Side::Center;

      std::cout << "Processing detId " << detId
                << ", location: " << (location == Location::barrel ? "barrel" : "endcap")
                << ", is_tilted: " << is_tilted << ", isstrip: " << isStripLayer(module, sensor.isLower)
                << ", isLower: " << sensor.isLower << ", moduleId: " << module.moduleId << ", layer: " << module.layer
                << ", isInverted: " << isInverted(module.moduleId, module.location, module.side, module.layer)
                << ", drdz_slope: " << slope.drdz_slope << ", dxdy_slope: " << slope.dxdy_slope
                << std::endl;  /////////////////////// remove

      // TODO: Do we need to skip strips?
      if (isStripLayer(module, sensor.isLower))
        continue;

      std::cout << "DetId: " << detId << ", location: " << (location == Location::barrel ? "barrel" : "endcap")
                << ", is_tilted: " << is_tilted << ", drdz_slope: " << slope.drdz_slope
                << ", dxdy_slope: " << slope.dxdy_slope << std::endl;  /////////////////////// remove

      if (location == Location::barrel and is_tilted)
        barrel_slopes[detId] = slope;
      else if (location == Location::endcap)
        endcap_slopes[detId] = slope;
    }

    return std::make_tuple(barrel_slopes, endcap_slopes);
  }
}  // namespace lstgeometry

#endif
