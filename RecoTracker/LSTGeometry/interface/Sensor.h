#ifndef RecoTracker_LSTGeometry_interface_Sensor_h
#define RecoTracker_LSTGeometry_interface_Sensor_h

#include <unordered_map>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  struct Sensor {
    unsigned int moduleDetId;
    float centerRho;
    float centerZ;
    float centerPhi;
    bool isLower;
    MatrixF4x3 corners;
    // Redundant, but convenient to have them
    ModuleType moduleType;
    float centerX;
    float centerY;

    Sensor() = default;
    Sensor(
        unsigned int moduleDetId, float centerRho, float centerZ, float centerPhi, bool isLower, ModuleType moduleType, Surface const& surface);
  };

  using Sensors = std::unordered_map<unsigned int, Sensor>;

}  // namespace lstgeometry

#endif
