#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDevicePixelSeeds_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDevicePixelSeeds_h

// The seeder's selected seeds as the generic device pixel seeds of RecoTracker/MkFitAlpaka's LST input
// (MkFitAlpakaLstInputProducer 'pixelSeeds': TrackSoA rows, hits as (mkFit layer, cluster index) inside-out, the state
// on the last hit, charge 0 = no seed; row = the seed index of the pLS).

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedScoresSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  struct DevicePixelSeedsParameters {
    // LSTInputProducer's stateAtLastPixelHit / stateAtLastPixelHitMaxPt: the pLS of a seed with >= 3 pixel hits that
    // ends on an outer-tracker hit (pT at its last hit below maxPt; maxPt <= 0: all) starts from its state at the last
    // pixel hit (the 'plsStates' rows)
    bool stateAtLastPixelHit;
    float stateAtLastPixelHitMaxPt;
  };

  // The seeds that become rows, of the first *count of cap: selected by the classifier, with a perigee (the seeds of
  // MkSeederPixelTrackProducer's pixel tracks, in their order: row k is pixel track k): flags[i] 1 for such a seed,
  // pos the inclusive scan of flags (pos[cap - 1] is the number of rows); flags and pos have cap entries.
  void flagDevicePixelSeeds(Queue &queue,
                            ::mkseeder::MkSeederSeedsSoA::ConstView seeds,
                            const uint32_t *count,
                            ::mkseeder::MkSeederSeedScoresSoA::ConstView scores,
                            uint32_t cap,
                            uint32_t *flags,
                            uint32_t *pos);
  // The rows (flagDevicePixelSeeds's flags and pos; `rows` zeroed, at least pos[cap - 1] rows): per flagged seed its
  // hits and its state on its last hit; with stateAtLastPixelHit also `plsStates` (same rows: the state at the last
  // pixel hit where it applies, else charge 0).
  void fillDevicePixelSeeds(Queue &queue,
                            ::mkseeder::MkSeederSeedsSoA::ConstView seeds,
                            ::mkseeder::MkSeederSeedScoresSoA::ConstView scores,
                            ::mkseeder::MkSeederSeedPixelStatesSoA::ConstView pixelStates,
                            uint32_t cap,
                            const uint32_t *flags,
                            const uint32_t *pos,
                            DevicePixelSeedsParameters const &p,
                            ::mkfitdev::TrackSoAView rows,
                            ::mkfitdev::TrackSoAView plsStates);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
