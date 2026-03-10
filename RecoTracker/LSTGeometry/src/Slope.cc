#include <tuple>
#include <cmath>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Slope.h"

namespace lstgeometry {

  Slope::Slope(float dx, float dy, float dz) {
    float dr = sqrt(dx * dx + dy * dy);
    drdz = dz != 0 ? dr / dz : kDefaultSlope;
    dxdy = dy != 0 ? -dx / dy : kDefaultSlope;
  }

  bool isStripLayer(Module module, bool isLower) {
    if (module.moduleType == ModuleType::Ph2SS)
      return true;
    if (isInverted(module.moduleId, module.location, module.side, module.layer))
      return isLower;
    return !isLower;
  }

  // Use each sensor's corners to calculate and categorize drdz and dxdy slopes.
  std::tuple<Slopes, Slopes> computeSlopes(Modules const& modules, Sensors const& sensors) {
    Slopes barrel_slopes;
    Slopes endcap_slopes;

    for (auto const& [detId, sensor] : sensors) {
      float dx = roundCoordinate(sensor.corners(1, 1) - sensor.corners(0, 1));
      float dy = roundCoordinate(sensor.corners(1, 2) - sensor.corners(0, 2));
      float dz = roundCoordinate(sensor.corners(1, 0) - sensor.corners(0, 0));

      Slope slope(dx, dy, dz);

      auto const& module = modules.at(sensor.moduleDetId);

      auto location = module.location;
      bool is_tilted = module.side != Side::Center;
      bool is_strip = isStripLayer(module, sensor.isLower);

      if (!is_strip)
        continue;

      if (location == Location::barrel and is_tilted)
        barrel_slopes[detId] = slope;
      else if (location == Location::endcap)
        endcap_slopes[detId] = slope;
    }

    return std::make_tuple(barrel_slopes, endcap_slopes);
  }
}  // namespace lstgeometry
