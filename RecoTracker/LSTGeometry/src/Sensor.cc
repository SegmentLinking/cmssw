#include <cmath>

#include "RecoTracker/LSTGeometry/interface/Sensor.h"

using namespace lstgeometry;

Sensor::Sensor(unsigned int moduleDetId, float centerRho, float centerZ, float centerPhi, ModuleType moduleType)
    : moduleDetId(moduleDetId),
      centerRho(centerRho),
      centerZ(centerZ),
      centerPhi(centerPhi),
      corners(MatrixF4x3::Zero()),
      moduleType(moduleType),
      centerX(centerRho * std::cos(centerPhi)),
      centerY(centerRho * std::sin(centerPhi)) {}
