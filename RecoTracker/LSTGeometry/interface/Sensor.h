#ifndef RecoTracker_LSTGeometry_interface_Sensor_h
#define RecoTracker_LSTGeometry_interface_Sensor_h

#include <unordered_map>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  struct Sensor {
    // Module-level properties
    ModuleType moduleType;
    SubDetector subdet;
    Location location;
    Side side;
    unsigned int moduleId;
    unsigned int layer;
    unsigned int ring;
    bool inverted;
    // Sensor-level properties
    float centerRho;
    float centerZ;
    float centerPhi;
    bool lower;
    bool strip;
    MatrixF4x3 corners;
    // Redundant, but convenient to avoid repeated computations
    float centerX;
    float centerY;

    Sensor() = default;
    Sensor(unsigned int detId,
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
           Surface const &surface);
  };

  using Sensors = std::unordered_map<unsigned int, Sensor>;

}  // namespace lstgeometry

#endif
