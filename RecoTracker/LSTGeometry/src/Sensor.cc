#include <cmath>

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "RecoTracker/LSTGeometry/interface/Sensor.h"

using namespace lstgeometry;

// Not sure if there is functionality for this already in CMSSW
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

// This differs from TrackerTopology::isLower since it considers if the module is inverted
bool isLower(unsigned int moduleId, Location location, Side side, unsigned int layer, unsigned int detId) {
  return isInverted(moduleId, location, side, layer) ? !(detId & 1) : (detId & 1);
}

bool isStrip(ModuleType moduleType, bool isInverted, bool isLower) {
  if (moduleType == ModuleType::Ph2SS)
    return true;
  if (isInverted)
    return isLower;
  return !isLower;
}

Sensor::Sensor(unsigned int detId,
               ModuleType moduleType,
               SubDetector subdet,
               Location location,
               Side side,
               unsigned int moduleId,
               unsigned int layer,
               unsigned int ring,
               float centerRho,
               float centerZ,
               float centerPhi,
               Surface const &surface)
    : moduleType(moduleType),
      subdet(subdet),
      location(location),
      side(side),
      moduleId(moduleId),
      layer(layer),
      ring(ring),
      inverted(isInverted(moduleId, location, side, layer)),
      centerRho(centerRho),
      centerZ(centerZ),
      centerPhi(centerPhi),
      lower(isLower(moduleId, location, side, layer, detId)),
      strip(isStrip(moduleType, inverted, lower)),
      centerX(centerRho * std::cos(centerPhi)),
      centerY(centerRho * std::sin(centerPhi)) {
  const Bounds &bounds = surface.bounds();
  const RectangularPlaneBounds &plane_bounds = dynamic_cast<const RectangularPlaneBounds &>(bounds);
  float wid = plane_bounds.width();
  float len = plane_bounds.length();
  auto c1 = GloballyPositioned<float>::LocalPoint(-wid / 2, -len / 2, 0);
  auto c2 = GloballyPositioned<float>::LocalPoint(-wid / 2, len / 2, 0);
  auto c3 = GloballyPositioned<float>::LocalPoint(wid / 2, len / 2, 0);
  auto c4 = GloballyPositioned<float>::LocalPoint(wid / 2, -len / 2, 0);
  auto c1g = surface.toGlobal(c1);
  auto c2g = surface.toGlobal(c2);
  auto c3g = surface.toGlobal(c3);
  auto c4g = surface.toGlobal(c4);
  // store corners with z, x, y ordering
  corners << c1g.z(), c1g.x(), c1g.y(), c2g.z(), c2g.x(), c2g.y(), c3g.z(), c3g.x(), c3g.y(), c4g.z(), c4g.x(), c4g.y();
}
