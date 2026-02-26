#ifndef RecoTracker_LSTGeometry_interface_Helix_h
#define RecoTracker_LSTGeometry_interface_Helix_h

namespace lstgeometry {

  struct Helix {
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float phi;
    float lambda;
    int charge;

    Helix(float center_x, float center_y, float center_z, float radius, float phi, float lam, int charge);
    Helix(float pt, float vx, float vy, float vz, float mx, float my, float mz, int charge);

    std::tuple<float, float, float, float> pointFromRadius(float target_r);
    std::tuple<float, float, float, float> pointFromZ(float target_z);
  };

}  // namespace lstgeometry

#endif
