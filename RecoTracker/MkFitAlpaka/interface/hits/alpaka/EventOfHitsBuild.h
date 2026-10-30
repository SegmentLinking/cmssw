#ifndef RecoTracker_MkFitAlpaka_interface_hits_alpaka_EventOfHitsBuild_h
#define RecoTracker_MkFitAlpaka_interface_hits_alpaka_EventOfHitsBuild_h

// Host-callable entry point of the device EventOfHits build (kernels: src/alpaka/hits/EventOfHitsKernels.h, compiled
// once in src/alpaka/EventOfHits.dev.cc), so that host (.cc) code of the portable plugins can call it.

#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsDeviceCollections.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {
  // The build on the blocks of the EventOfHits event product (hits and layers blocks filled by the caller). Returns
  // false (nothing enqueued) if the event exceeds the device build limits (the caller skips the event's mkFit).
  bool runBuildEventOfHits(Queue& queue,
                           EventOfHitsDeviceCollection& product,
                           std::vector<::mkfitdev::DeadRegionDev> const& deads);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits

#endif
