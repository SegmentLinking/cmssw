#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTClusterModules_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTClusterModules_h

#include <cstdint>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Phase2TrackerRecHitsKernels.h"

namespace phase2TrackerRecHits {

  // Calls function(ModuleClusters) for each OT module with clusters, in the order of the collection. The clusters of
  // a module are the rows [firstCluster, firstCluster + nClusters) of clusters.data(), which are also the rows of their
  // rechits in a Phase2OTRecHitsSoA made from the same collection.
  template <typename Function>
  void forEachModule(Phase2TrackerCluster1DCollectionNew const& clusters,
                     TrackerGeometry const& geometry,
                     Function&& function) {
    auto const* firstCluster = clusters.data().data();
    for (auto const& detSet : clusters) {
      if (detSet.empty())
        continue;
      function(ModuleClusters{static_cast<uint32_t>(&*detSet.begin() - firstCluster),
                              static_cast<uint32_t>(detSet.size()),
                              static_cast<uint32_t>(geometry.idToDetUnit(DetId(detSet.detId()))->index())});
    }
  }

}  // namespace phase2TrackerRecHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTClusterModules_h
