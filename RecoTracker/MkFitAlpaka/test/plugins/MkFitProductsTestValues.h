#ifndef RecoTracker_MkFitAlpaka_test_plugins_MkFitProductsTestValues_h
#define RecoTracker_MkFitAlpaka_test_plugins_MkFitProductsTestValues_h

// Product sizes and known values shared by MkFitAlpakaProductsTest, which writes them on the device, and
// MkFitAlpakaProductsCheck, which recomputes them on the host.

#include <cstdint>

#include <alpaka/alpaka.hpp>

namespace mkfitdev::productstest {
  // value of element `element` of row `row` in event `eventNumber` (exact in float for the sizes used)
  ALPAKA_FN_HOST_ACC inline float val(uint32_t eventNumber, int row, int element) {
    return float((eventNumber % 64) * 1024 + (row % 1024)) + 0.125f * element;
  }
  constexpr int kHits = 300, kLayers = 5, kBins = 512, kTracks = 200;
}  // namespace mkfitdev::productstest

#endif
