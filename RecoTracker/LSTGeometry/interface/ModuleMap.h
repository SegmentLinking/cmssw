#ifndef RecoTracker_LSTGeometry_interface_ModuleMapMethods_h
#define RecoTracker_LSTGeometry_interface_ModuleMapMethods_h

#include <unordered_set>
#include <unordered_map>

#include "RecoTracker/LSTGeometry/interface/DetectorGeometry.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  using ModuleMap = std::unordered_map<unsigned int, std::unordered_set<unsigned int>>;

  ModuleMap buildModuleMap(Sensors const& sensors, DetectorGeometry const& det_geom, float pt_cut);

}  // namespace lstgeometry

#endif
