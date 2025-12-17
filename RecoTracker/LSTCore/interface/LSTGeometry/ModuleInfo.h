#ifndef RecoTracker_LSTCore_interface_LSTGeometry_ModuleInfo_h
#define RecoTracker_LSTCore_interface_LSTGeometry_ModuleInfo_h

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"

namespace lstgeometry {

  struct ModuleInfo {
    unsigned int detId;
    double sensorCenterRho_cm;
    double sensorCenterZ_cm;
    double tiltAngle_rad;
    double skewAngle_rad;
    double yawAngle_rad;
    double phi_rad;
    double meanWidth_cm;
    double length_cm;
    double sensorSpacing_cm;
    MatrixD8x3 transformedCorners;
  };

}  // namespace lstgeometry

#endif
