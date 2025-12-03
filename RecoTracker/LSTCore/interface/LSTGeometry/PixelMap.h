#ifndef RecoTracker_LSTCore_interface_LSTGeometry_PixelMap_h
#define RecoTracker_LSTCore_interface_LSTGeometry_PixelMap_h

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <boost/functional/hash.hpp>

namespace lstgeometry {

  using LayerSubdetChargeKey = std::tuple<unsigned int, unsigned int, int>;
  using LayerSubdetChargeMap = std::unordered_map<LayerSubdetChargeKey,
                                                  std::vector<std::unordered_set<unsigned int>>,
                                                  boost::hash<LayerSubdetChargeKey>>;
  using PixelMap = LayerSubdetChargeMap;

}  // namespace lstgeometry

#endif