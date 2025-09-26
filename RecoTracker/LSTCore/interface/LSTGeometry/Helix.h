#ifndef RecoTracker_LSTCore_interface_LSTGeometry_Centroid_h
#define RecoTracker_LSTCore_interface_LSTGeometry_Centroid_h

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"

namespace lstgeometry {

  struct Helix {
    ColVectorD3 center;
    double radius;
    double phi;
    double lam;
    int charge;
  };

}  // namespace lstgeometry

#endif