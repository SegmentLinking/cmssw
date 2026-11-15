#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTClusters_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTClusters_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTHits {

  // the Phase2TrackerCluster1D quantities that the CPE reads, and the row of the cluster's module in the module
  // parameters that the kernel reads
  struct Cluster {
    uint16_t firstStrip;
    uint16_t column;
    uint16_t size;
    uint16_t module;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTClusters_h
