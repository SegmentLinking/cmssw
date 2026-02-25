#ifndef RecoTracker_LSTGeometry_interface_OrientationMethods_h
#define RecoTracker_LSTGeometry_interface_OrientationMethods_h

#include <tuple>
#include <cmath>
#include <unordered_map>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/SlopeData.h"

namespace lstgeometry {

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

    for (const auto& [detId, sensor] : sensors) {
      double dx = roundCoordinate(sensor.corners(1, 1) - sensor.corners(0, 1));
      double dy = roundCoordinate(sensor.corners(1, 2) - sensor.corners(0, 2));
      double dz = roundCoordinate(sensor.corners(1, 0) - sensor.corners(0, 0));

      SlopeData slope = calculateSlope(dx, dy, dz);

      auto& module = modules.at(sensor.moduleDetId);

      auto location = module.location;
      bool is_tilted = module.side != Phase2Tracker::BarrelModuleTilt::flat;

      // TODO: Do we need to skip strips?

      if (location == GeomDetEnumerators::Location::barrel and is_tilted)
        barrel_slopes[detId] = slope;
      else if (location == GeomDetEnumerators::Location::endcap)
        endcap_slopes[detId] = slope;
    }

    return std::make_tuple(barrel_slopes, endcap_slopes);
  }
}  // namespace lstgeometry

#endif
