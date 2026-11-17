// Component "bkfit": the only translation unit that instantiates the backward-fit kernels (src/alpaka/bkfit/*.h).
// Exported entry points: src/alpaka/bkfit/BkFitLaunch.h.
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/bkfit/BkFitKernel.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/bkfit/BkFitLaunch.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  void backwardFit(Queue& queue,
                   ::mkfitdev::SeedCandsSoA::ConstView seeds,
                   ::mkfitdev::CandSlotsSoA::View slots,
                   ::mkfitdev::CandHotsSoA::View hots,
                   ::mkfitdev::EngineHitInputs const& hits,
                   ::mkfitdev::prop::PropagationFlags const& pflags,
                   int nSeeds,
                   ::mkfitdev::bkfit::OutlierParams const& outliers) {
    BkFitCandsIO io{seeds, slots, hots, hits};
    launchBkFit(queue, io, nSeeds, pflags, outliers);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev
