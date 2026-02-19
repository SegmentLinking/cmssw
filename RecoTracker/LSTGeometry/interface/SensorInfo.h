#ifndef RecoTracker_LSTGeometry_interface_SensorInfo_h
#define RecoTracker_LSTGeometry_interface_SensorInfo_h

#include <string>

namespace lstgeometry {

  struct SensorInfo {
    unsigned int detId;
    double sensorCenterRho_cm;
    double sensorCenterZ_cm;
    double phi_rad;
  };
}  // namespace lstgeometry

#endif
