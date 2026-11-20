#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTRecHitsKernels_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTRecHitsKernels_h

#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2OTModuleParamsHost.h"

#include "Phase2OTClusters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTRecHits {

  // one rechit per cluster, in the same order; the cluster's module is its row in modules
  void makeRecHits(Queue& queue,
                   Phase2OTModuleParamsSoA::ConstView modules,
                   phase2OTHits::Cluster const* clusters,
                   ::reco::Phase2OTRecHitsView recHits);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTRecHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTRecHitsKernels_h
