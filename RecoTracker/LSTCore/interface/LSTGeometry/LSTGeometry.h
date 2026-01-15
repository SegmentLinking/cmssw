#ifndef RecoTracker_LSTCore_interface_LSTGeometry_LSTGeometry_h
#define RecoTracker_LSTCore_interface_LSTGeometry_LSTGeometry_h

#include "RecoTracker/LSTCore/interface/LSTGeometry/Centroid.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/SlopeData.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/PixelMap.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/SensorInfo.h"

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
