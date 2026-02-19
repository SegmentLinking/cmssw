#ifndef RecoTracker_LSTGeometry_interface_SensorCentroid_h
#define RecoTracker_LSTGeometry_interface_SensorCentroid_h

namespace lstgeometry {

  struct SensorCentroid {
    unsigned int moduleType;
    double x;
    double y;
    double z;
  };

}  // namespace lstgeometry

#endif
