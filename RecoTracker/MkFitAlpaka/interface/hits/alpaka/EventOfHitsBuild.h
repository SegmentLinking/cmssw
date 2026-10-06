#ifndef RecoTracker_MkFitAlpaka_interface_hits_alpaka_EventOfHitsBuild_h
#define RecoTracker_MkFitAlpaka_interface_hits_alpaka_EventOfHitsBuild_h

// Host-callable entry point of the device EventOfHits build (kernels: src/alpaka/hits/EventOfHitsKernels.h, compiled once in src/alpaka/EventOfHits.dev.cc), so that host
// (.cc) code of the portable plugins can call it.

#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsDeviceCollections.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {
  EventOfHitsDevice runBuildEventOfHits(Queue& queue,
                                        ::mkfitdev::HitsHostCollection const& hitsHost,
                                        ::mkfitdev::LayersHostCollection const& layersHost,
                                        std::vector<::mkfitdev::DeadRegionDev> const& deads,
                                        bool* built = nullptr);
  // Allocation, and the build on a device EventOfHits whose hits and static layer table are already filled
  // (on CPU backends the device collections are host memory and can be filled directly, without staging).
  EventOfHitsDevice runMakeEventOfHitsDevice(Queue& queue, uint32_t nHits, uint32_t nLayers, uint32_t nBins);
  // Returns false (nothing enqueued) if the event exceeds the device build limits (the caller skips the event's
  // mkFit and logs it; no exception at event time). 'built' of the overload above reports the same.
  bool runBuildEventOfHits(Queue& queue, EventOfHitsDevice& d, std::vector<::mkfitdev::DeadRegionDev> const& deads);
  // Same build on the blocks of the EventOfHits event product (hits and layers blocks filled by the caller).
  bool runBuildEventOfHits(Queue& queue,
                           EventOfHitsDeviceCollection& product,
                           std::vector<::mkfitdev::DeadRegionDev> const& deads);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits

#endif
