// Host-callable entry points of the device EventOfHits build and the device hit input.
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsBuild.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/hits/EventOfHitsKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/inputs/DeviceHitsKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {
  bool runBuildEventOfHits(Queue& queue,
                           EventOfHitsDeviceCollection& product,
                           std::vector<::mkfitdev::DeadRegionDev> const& deads) {
    auto view = product.view();
    return buildEventOfHits(queue, EventOfHitsViews{view.hits(), view.layers(), view.binnedHits(), view.bins()}, deads);
  }

  void runFillHitsFromDeviceInputs(Queue& queue, DeviceHitInputs const& in, ::mkfitdev::HitSoA::View hits) {
    fillHitsFromDeviceInputs(queue, in, hits);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits
