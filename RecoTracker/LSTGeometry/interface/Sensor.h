#ifndef RecoTracker_LSTGeometry_interface_Sensor_h
#define RecoTracker_LSTGeometry_interface_Sensor_h

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  struct Sensor {
    unsigned int moduleDetId;
    float centerRho_cm;
    float centerZ_cm;
    float centerPhi_rad;
    MatrixD4x3 corners;
    // Redundant, but convenient to have them
    TrackerGeometry::ModuleType moduleType;
    float centerX_cm;
    float centerY_cm;

    Sensor() = default;
    Sensor(unsigned int moduleDetId,
           float centerRho_cm,
           float centerZ_cm,
           float centerPhi_rad,
           TrackerGeometry::ModuleType moduleType);
  };
}  // namespace lstgeometry

#endif
