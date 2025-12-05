#ifndef RecoTracker_LSTCore_interface_LSTGeometry_Helix_h
#define RecoTracker_LSTCore_interface_LSTGeometry_Helix_h

namespace lstgeometry {

  struct Helix {
    double center_x;
    double center_y;
    double center_z;
    double radius;
    double phi;
    double lam;
    int charge;
  };

}  // namespace lstgeometry

#endif
