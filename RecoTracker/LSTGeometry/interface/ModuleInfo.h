#ifndef RecoTracker_LSTGeometry_interface_ModuleInfo_h
#define RecoTracker_LSTGeometry_interface_ModuleInfo_h

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  struct ModuleInfo {
    TrackerGeometry::ModuleType moduleType;
    GeomDetEnumerators::SubDetector subdet;
    Phase2Tracker::BarrelModuleTilt side;
    unsigned int layer;
    unsigned int ring;
    bool isLower;
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
