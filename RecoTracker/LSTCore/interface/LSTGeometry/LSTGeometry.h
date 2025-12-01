#ifndef RecoTracker_LSTCore_interface_LSTGeometry_LSTGeometry_h
#define RecoTracker_LSTCore_interface_LSTGeometry_LSTGeometry_h

#include "RecoTracker/LSTCore/interface/LSTGeometry/CornerMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/CentroidMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/OrientationMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/PixelMapMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/ModuleMapMethods.h"

namespace lstgeometry {

  struct LSTGeometry {
    std::unordered_map<unsigned int, Centroid> centroids;
    std::unordered_map<unsigned int, SlopeData> barrel_slopes;
    std::unordered_map<unsigned int, SlopeData> endcap_slopes;
    PixelMap pixel_map;
    std::unordered_map<unsigned int, std::unordered_set<unsigned int>> merged_line_connections;
    std::unordered_map<unsigned int, SensorInfo> sensor_info;
  };

}  // namespace lstgeometry

#endif
