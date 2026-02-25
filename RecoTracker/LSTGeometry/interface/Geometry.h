#ifndef RecoTracker_LSTGeometry_interface_Geometry_h
#define RecoTracker_LSTGeometry_interface_Geometry_h

#include "RecoTracker/LSTGeometry/interface/SlopeData.h"
#include "RecoTracker/LSTGeometry/interface/PixelMap.h"
#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  struct Geometry {
    Sensors sensors;
    std::unordered_map<unsigned int, SlopeData> barrel_slopes;
    std::unordered_map<unsigned int, SlopeData> endcap_slopes;
    PixelMap pixel_map;
    std::unordered_map<unsigned int, std::unordered_set<unsigned int>> merged_line_connections;

    Geometry(Modules &modules,
             Sensors &sensors_input,
             std::vector<float> const &average_r,
             std::vector<float> const &average_z,
             double ptCut);
  };

}  // namespace lstgeometry

#endif
