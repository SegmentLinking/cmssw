#ifndef RecoTracker_LSTCore_interface_LSTGeometry_Centroid_h
#define RecoTracker_LSTCore_interface_LSTGeometry_Centroid_h

namespace lstgeometry {

  struct Centroid {
    unsigned int detId;
    unsigned int moduleType;
    double x;
    double y;
    double z;
  };

}  // namespace lstgeometry

#endif
