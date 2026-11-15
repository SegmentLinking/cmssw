#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h

#include <cstdint>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2OTCAModuleParamsHost.h"

#include "Phase2OTClusters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits {

  // the cluster's module is its CA module
  using phase2OTHits::Cluster;

  struct BeamSpotPosition {
    double x;
    double y;
    double z;
  };

  // one CA hit per cluster, in the same order, relative to the beam spot; the hits of CA module i get detectorIndex
  // firstDetectorIndex + i
  void makeCAHits(Queue& queue,
                  Phase2OTCAModuleParamsSoA::ConstView modules,
                  Cluster const* clusters,
                  BeamSpotPosition beamSpot,
                  uint16_t firstDetectorIndex,
                  ::reco::TrackingRecHitView hits);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h
