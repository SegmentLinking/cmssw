#include <cmath>

#include "RecoTracker/LSTGeometry/interface/Sensor.h"

using namespace lstgeometry;

Sensor::Sensor(unsigned int moduleDetId,
               float centerRho_cm,
               float centerZ_cm,
               float centerPhi_rad,
               TrackerGeometry::ModuleType moduleType)
    : moduleDetId(moduleDetId),
      centerRho_cm(centerRho_cm),
      centerZ_cm(centerZ_cm),
      centerPhi_rad(centerPhi_rad),
      moduleType(moduleType),
      centerX_cm(centerRho_cm * std::cos(centerPhi_rad)),
      centerY_cm(centerRho_cm * std::sin(centerPhi_rad)) {}
