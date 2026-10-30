#ifndef RecoTracker_MkFitAlpaka_test_plugins_alpaka_MkFitProductsTestKernels_h
#define RecoTracker_MkFitAlpaka_test_plugins_alpaka_MkFitProductsTestKernels_h

// Test-only kernel behind MkFitAlpakaProductsTest: fills the event products with the known values of
// MkFitProductsTestValues.h, which MkFitAlpakaProductsCheck recomputes on the host.

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/TrackProduct.h"
#include "RecoTracker/MkFitAlpaka/test/plugins/MkFitProductsTestValues.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  void fillProductsTest(Queue& queue, EventOfHitsDeviceCollection& eoh, TrackSoADeviceCollection& trk, uint32_t event);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
