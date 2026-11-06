#ifndef RecoTracker_MkFitAlpaka_src_alpaka_cands_CandSeedOpsKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_cands_CandSeedOpsKernels_h

// Per-seed view of the clone-engine buffers (makeSeedRef) for the engine kernels.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using namespace ::mkfitdev;

  ALPAKA_FN_ACC inline SeedCandsRef makeSeedRef(
      SeedCandsSoA::View seeds, CandSlotsSoA::View slots, CandHotsSoA::View hots, int s, int hps) {
    SeedCandsRef ref;
    const int cur = seeds.curBuf(s);
    ref.state = &seeds.state(s);
    ref.pickupLayer = &seeds.pickupLayer(s);
    ref.cands = &slots.book(candSlotRow(s, cur, 0));
    ref.states = &slots.state(candSlotRow(s, cur, 0));
    ref.nCands = &seeds.nCands(s);
    ref.bestShort = &seeds.bestShort(s);
    ref.bestShortState = &slots.state(bestShortRow(s));
    ref.bestShortValid = &seeds.bestShortValid(s);
    ref.hots = &hots.node(hotRow(s, 0, hps));
    ref.hotOffset = 0;
    ref.hotCap = hps;
    ref.nHots = &seeds.nHots(s);
    ref.lastHitIdxBeforeBkw = &seeds.lastHitIdxBeforeBkw(s);
    ref.nInsideMinusOneBeforeBkw = &seeds.nInsideMinusOneBeforeBkw(s);
    ref.nTailMinusOneBeforeBkw = &seeds.nTailMinusOneBeforeBkw(s);
    ref.overflowBits = &seeds.overflowBits(s);
    return ref;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
