#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsBuild.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/hits/EventOfHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {
  EventOfHitsDevice runBuildEventOfHits(Queue& queue,
                                        ::mkfitdev::HitsHostCollection const& hitsHost,
                                        ::mkfitdev::LayersHostCollection const& layersHost,
                                        std::vector<::mkfitdev::DeadRegionDev> const& deads,
                                        bool* built) {
    return buildEventOfHits(queue, hitsHost, layersHost, deads, built);
  }
  EventOfHitsDevice runMakeEventOfHitsDevice(Queue& queue, uint32_t nHits, uint32_t nLayers, uint32_t nBins) {
    return makeEventOfHitsDevice(queue, nHits, nLayers, nBins);
  }
  bool runBuildEventOfHits(Queue& queue, EventOfHitsDevice& d, std::vector<::mkfitdev::DeadRegionDev> const& deads) {
    return buildEventOfHits(queue, d, deads);
  }
  bool runBuildEventOfHits(Queue& queue,
                           EventOfHitsDeviceCollection& product,
                           std::vector<::mkfitdev::DeadRegionDev> const& deads) {
    auto v = product.view();
    return buildEventOfHits(queue, EventOfHitsViews{v.hits(), v.layers(), v.binnedHits(), v.bins()}, deads);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits

// Device hit input
#include "RecoTracker/MkFitAlpaka/src/alpaka/inputs/DeviceHitsKernel.h"
namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {
  void runFillHitsFromDeviceInputs(Queue& queue, DeviceHitInputs const& in, ::mkfitdev::HitSoA::View hits) {
    fillHitsFromDeviceInputs(queue, in, hits);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits
