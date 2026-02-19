#ifndef RecoTracker_LSTGeometry_interface_Helix_h
#define RecoTracker_LSTGeometry_interface_Helix_h

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
