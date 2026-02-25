#ifndef RecoTracker_LSTGeometry_interface_Helix_h
#define RecoTracker_LSTGeometry_interface_Helix_h

namespace lstgeometry {

  struct Helix {
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float phi;
    float lam;
    int charge;
  };

}  // namespace lstgeometry

#endif
