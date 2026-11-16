#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaSeedExportKernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaSeedExportKernels_h

// Validation export of the build module's seed table: the seed rows with the states building starts from (after the
// pixel seed states and the device LST seed fit) as TrackSoA rows (params, errors, charge, label, hits).

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // rows [0, n) of seeds -> rows [0, n) of out (capacity >= max(n, 1)); out.nTracks = n
  void exportSeedRows(Queue& queue, ::mkfitdev::SeedSoAConstView seeds, int32_t n, ::mkfitdev::TrackSoAView out);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
