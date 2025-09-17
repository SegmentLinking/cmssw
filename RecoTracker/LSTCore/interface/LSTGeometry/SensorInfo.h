#ifndef RecoTracker_LSTCore_interface_LSTGeometry_SensorInfo_h
#define RecoTracker_LSTCore_interface_LSTGeometry_SensorInfo_h

#include <string>

namespace lst {

  struct SensorInfo {
    unsigned int detId;
    unsigned int binaryDetId;
    std::string section;
    int layer;
    int ring;
    double sensorCenterRho_mm;
    double sensorCenterZ_mm;
    double phi_deg;
  };
}  // namespace lst

#endif
