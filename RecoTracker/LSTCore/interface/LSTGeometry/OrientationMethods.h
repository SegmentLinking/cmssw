#ifndef RecoTracker_LSTCore_interface_LSTGeometry_OrientationMethods_h
#define RecoTracker_LSTCore_interface_LSTGeometry_OrientationMethods_h

#include <tuple>
#include <cmath>
#include <unordered_map>

#include "CentroidMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/CentroidMethods.h"

namespace lstgeometry {

  struct SlopeData {
    double drdz_slope;
    double drdy_slope;
  };

  // Use each sensor's corners to calculate and categorize drdz and dxdy slopes.
  SlopeData calculateSlope(double dx, double dy, double dz) {
    double dr = sqrt(dx * dx + dy * dy);
    double drdz_slope = dz != 0 ? dr / dz : kDefaultSlope;
    double drdy_slope = dy != 0 ? -dx / dy : kDefaultSlope;
    return SlopeData{drdz_slope, drdy_slope};
  }

  // Use each sensor's corners to calculate and categorize drdz and dxdy slopes.
  std::tuple<std::unordered_map<unsigned int, SlopeData>, std::unordered_map<unsigned int, SlopeData>> processCorners(
      std::unordered_map<unsigned int, MatrixD4x3>& corners) {
    std::unordered_map<unsigned int, SlopeData> barrel_slopes;
    std::unordered_map<unsigned int, SlopeData> endcap_slopes;

    for (const auto& [detId, corners] : corners) {
      double dx = corners(1, 1) - corners(0, 1);
      double dy = corners(1, 2) - corners(0, 2);
      double dz = corners(1, 0) - corners(0, 0);

      SlopeData slope = calculateSlope(dx, dy, dz);

      unsigned int module_type = static_cast<unsigned int>(parseModuleType(detId));
      Module module(detId, module_type);

      unsigned short subdet = module.subdet();
      bool is_tilted = module.side() != 3;
      bool is_strip = module.moduleLayerType() == 1;

      if (!is_strip)
        continue;

      if (subdet == Module::SubDet::Barrel and is_tilted)
        barrel_slopes[detId] = slope;
      else if (subdet == Module::SubDet::Endcap)
        endcap_slopes[detId] = slope;
    }

    return std::make_tuple(barrel_slopes, endcap_slopes);
  }
}  // namespace lstgeometry

#endif