#ifndef RecoTracker_LSTGeometry_interface_Module_h
#define RecoTracker_LSTGeometry_interface_Module_h

#include <unordered_map>

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  // A module contains 2 sensors. The common properties of the 2 sensors are stored in the Module struct, and the sensor-specific properties are stored in the Sensor struct.
  struct Module {
    ModuleType moduleType;
    SubDetector subdet;
    Location location;
    Side side;
    unsigned int moduleId;
    unsigned int layer;
    unsigned int ring;
  };

  using Modules = std::unordered_map<unsigned int, Module>;
}  // namespace lstgeometry

#endif
