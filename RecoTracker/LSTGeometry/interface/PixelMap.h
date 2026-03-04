#ifndef RecoTracker_LSTGeometry_interface_PixelMap_h
#define RecoTracker_LSTGeometry_interface_PixelMap_h

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <boost/functional/hash.hpp>

#include "RecoTracker/LSTGeometry/interface/DetectorGeometry.h"
#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  using LayerSubdetChargeKey = std::tuple<unsigned int, unsigned int, int>;
  using LayerSubdetChargeMap = std::unordered_map<LayerSubdetChargeKey,
                                                  std::vector<std::unordered_set<unsigned int>>,
                                                  boost::hash<LayerSubdetChargeKey>>;
  using PixelMap = LayerSubdetChargeMap;

  PixelMap buildPixelMap(Modules const& modules, Sensors const& sensors, DetectorGeometry const& det_geom, float pt_cut);

}  // namespace lstgeometry

#endif
