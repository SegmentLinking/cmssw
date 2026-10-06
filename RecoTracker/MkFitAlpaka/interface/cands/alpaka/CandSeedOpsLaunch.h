#ifndef RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandSeedOpsLaunch_h
#define RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandSeedOpsLaunch_h

// Host entry points of the per-seed bookkeeping ops (one thread per seed). The kernels live in
// src/alpaka/cands/CandSeedOpsKernels.h and are instantiated once, in src/alpaka/Cands.dev.cc (library symbols).
// Callers (producers, tests) use these functions and never launch the kernels themselves (nvlink duplicate-RDC rule).

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // filter_comb_cands per seed: passed[s] = 0/1.
  void filterSeeds(Queue& queue,
                   ::mkfitdev::SeedCandsSoA::View seeds,
                   ::mkfitdev::CandSlotsSoA::View slots,
                   ::mkfitdev::CandHotsSoA::View hots,
                   int8_t* passed,
                   bool bkwRep,
                   bool attemptAllCands,
                   int minHitsQF,
                   int nSeeds);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
