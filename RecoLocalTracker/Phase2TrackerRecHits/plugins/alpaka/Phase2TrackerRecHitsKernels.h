#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2TrackerRecHitsKernels_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2TrackerRecHitsKernels_h

#include <cstdint>

#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEParamsHost.h"

namespace phase2TrackerRecHits {

  // the clusters of one OT module: rows [firstCluster, firstCluster + nClusters) of the cluster collection
  struct ModuleClusters {
    uint32_t firstCluster;
    uint32_t nClusters;
    uint32_t detectorIndex;  // GeomDetUnit::index() of the module
  };

  // the Phase2TrackerCluster1D quantities the CPE reads
  struct ClusterData {
    uint16_t firstStrip;
    uint16_t column;
    uint16_t size;
  };

}  // namespace phase2TrackerRecHits

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2TrackerRecHits {

  // Phase2StripCPE and the local to global transformation for every cluster of the given modules
  void makeRecHits(Queue& queue,
                   Phase2StripCPEParamsSoA::ConstView params,
                   ::phase2TrackerRecHits::ModuleClusters const* modules,
                   uint32_t nModules,
                   ::phase2TrackerRecHits::ClusterData const* clusters,
                   ::reco::Phase2OTRecHitsView recHits);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2TrackerRecHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2TrackerRecHitsKernels_h
