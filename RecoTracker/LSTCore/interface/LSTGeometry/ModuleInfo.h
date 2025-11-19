#ifndef RecoTracker_LSTCore_interface_LSTGeometry_ModuleInfo_h
#define RecoTracker_LSTCore_interface_LSTGeometry_ModuleInfo_h

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"

namespace lstgeometry {

  struct ModuleInfo {
    unsigned int detId;
    double sensorCenterRho_mm;
    double sensorCenterZ_mm;
    double tiltAngle_rad;
    double skewAngle_rad;
    double yawAngle_rad;
    double phi_rad;
    double vtxOneX_mm;
    double vtxOneY_mm;
    double vtxTwoX_mm;
    double vtxTwoY_mm;
    double vtxThreeX_mm;
    double vtxThreeY_mm;
    double vtxFourX_mm;
    double vtxFourY_mm;
    double meanWidth_mm;
    double length_mm;
    double sensorSpacing_mm;
    MatrixD8x3 transformedCorners;
  };

}  // namespace lstgeometry

#endif
