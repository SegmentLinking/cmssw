#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitProductsTestKernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitProductsTestKernels_h

// integ: test-only kernels behind MkFitAlpakaProductsTest (fill the event products with known values, and run the
// MPlex <-> SoA packers on the device). The expected values are recomputed on the host by MkFitAlpakaProductsCheck.

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/CandStoreProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/TrackProduct.h"

namespace mkfitdev::productstest {
  // value of element k of row r of event e (exact in float for the sizes used)
  ALPAKA_FN_HOST_ACC inline float val(uint32_t e, int r, int k) {
    return float((e % 64) * 1024 + (r % 1024)) + 0.125f * k;
  }
  constexpr int kHits = 300, kLayers = 5, kBins = 512, kTracks = 200, kSeeds = 40, kHotsPerSeed = 16;
}  // namespace mkfitdev::productstest

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  // fills the three products; then track row r takes params[0..2] = hit r position and errors[0..5] = hit r errors
  // through pack::loadHit + MPlex copyOut, and the candidate slot 0 of seed s takes track s through
  // pack::loadTrack + pack::storeCandState (device instantiation of the packers, N = kNN of the backend)
  void fillProductsTest(Queue& queue,
                        EventOfHitsDeviceCollection& eoh,
                        TrackSoADeviceCollection& trk,
                        CandStoreDeviceCollection& cand,
                        uint32_t event);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
