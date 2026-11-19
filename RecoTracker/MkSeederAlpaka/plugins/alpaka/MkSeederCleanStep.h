#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederCleanStep_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederCleanStep_h

// The cleaning (MkSeederClean.h) with its scratch given: the device side of sortAndCleanQuads() (MkSeederDevice.h),
// which allocates the sort's and the cleaning's scratch together.

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "MkSeederClean.h"
#include "MkSeederRadixSort.h"
#include "MkSeederScratchSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // the cleaning's state: per finish index / rank (cap rows), per hit (nTot rows), per hit its items by rank
  // (nTot + 1 offsets, 4 items per quad)
  struct CleanScratch {
    ::mkseeder::MkSeederCleanItemsSoA::View items;
    ::mkseeder::MkSeederHitUseSoA::View hitUse;
    ::mkseeder::MkSeederListOffsetsSoA::View hitItemOffsets;
    ::mkseeder::MkSeederListItemsSoA::View hitItems;
  };

  // Cleans v's quads and triplet seeds (see MkSeederClean.dev.cc) with sort's scratch (the quads' capacity)
  void cleanQuadsAndTriplets(Queue &queue, mkseeder::CleanInputs v, GlobalRadixSort &sort, CleanScratch const &scratch);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
