#include <cmath>

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "RecoTracker/LSTGeometry/interface/Sensor.h"

using namespace lstgeometry;

Sensor::Sensor(unsigned int moduleDetId,
               float centerRho,
               float centerZ,
               float centerPhi,
               bool isLower,
               ModuleType moduleType,
               Surface const &surface)
    : moduleDetId(moduleDetId),
      centerRho(centerRho),
      centerZ(centerZ),
      centerPhi(centerPhi),
      isLower(isLower),
      moduleType(moduleType),
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
