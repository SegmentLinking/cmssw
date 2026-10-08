#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h

#include <cstdint>

#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace phase2OTCAHits {

  // the rechits of one OT module of the CA: rechit rows [firstRecHit, firstRecHit + nHits) become the CA hit rows
  // [firstHit, firstHit + nHits)
  struct ModuleHits {
    uint32_t firstRecHit;
    uint32_t firstHit;
    uint32_t nHits;
  };

  struct BeamSpotPosition {
    double x;
    double y;
    double z;
  };

}  // namespace phase2OTCAHits

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits {

  // fills the CA hits of the given modules, relative to the beam spot, and the module starts; the hits of CA module i
  // get detectorIndex firstDetectorIndex + i and start at nPixelHits + firstHit in the CA's pixel + OT hit indices
  void makeCAHits(Queue& queue,
                  ::reco::Phase2OTRecHitsConstView recHits,
                  ::phase2OTCAHits::ModuleHits const* modules,
                  uint32_t nModules,
                  ::phase2OTCAHits::BeamSpotPosition beamSpot,
                  uint32_t nPixelHits,
                  uint16_t firstDetectorIndex,
                  ::reco::TrackingRecHitView hits,
                  ::reco::HitModuleSoAView hitModules);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h
