#ifndef RecoTracker_LSTGeometry_interface_Sensor_h
#define RecoTracker_LSTGeometry_interface_Sensor_h

#include <unordered_map>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  struct Sensor {
    unsigned int moduleDetId;
    float centerRho;
    float centerZ;
    float centerPhi;
    MatrixF4x3 corners;
    // Redundant, but convenient to have them
    ModuleType moduleType;
    float centerX;
    float centerY;

    Sensor() = default;
    Sensor(unsigned int moduleDetId, float centerRho, float centerZ, float centerPhi, ModuleType moduleType);
  };

  using Sensors = std::unordered_map<unsigned int, Sensor>;

}  // namespace lstgeometry

#endif
