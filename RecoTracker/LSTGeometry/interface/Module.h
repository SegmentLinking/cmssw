#ifndef RecoTracker_LSTGeometry_interface_Module_h
#define RecoTracker_LSTGeometry_interface_Module_h

#include <unordered_map>

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  struct Module {
    ModuleType moduleType;
    SubDetector subdet;
    Location location;
    Side side;
    unsigned int moduleId;
    unsigned int layer;
    unsigned int ring;
    float centerRho;
    float centerZ;
    float centerPhi;
    float tiltAngle;
    float skewAngle;
    float yawAngle;
    float meanWidth;
    float length;
    float spacing;
    MatrixF8x3 transformedCorners;
  };

  using Modules = std::unordered_map<unsigned int, Module>;
}  // namespace lstgeometry

#endif
