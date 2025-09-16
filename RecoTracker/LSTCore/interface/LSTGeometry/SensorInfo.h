#ifndef RecoTracker_LSTCore_interface_LSTGeometry_SensorInfo_h
#define RecoTracker_LSTCore_interface_LSTGeometry_SensorInfo_h

#include <string>

namespace lst {

  struct SensorInfo {
    unsigned int DetId;
    char BinaryDetId;
    std::string Section;
    int Layer;
    int Ring;
    double sensorCenterRho_mm;
    double sensorCenterZ_mm;
    double phi_deg;
  };
}  // namespace lst

#endif
