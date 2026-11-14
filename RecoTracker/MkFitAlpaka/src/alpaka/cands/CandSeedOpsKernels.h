#ifndef RecoTracker_MkFitAlpaka_src_alpaka_cands_CandSeedOpsKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_cands_CandSeedOpsKernels_h

// Per-seed helpers of the candidate kernels, shared by the clone engine (src/alpaka/engine) and the v2p2 kernels
// (src/alpaka/v2p2): the per-seed view of the candidate books (makeSeedRef), the device row count (engineRows) and the
// failing of a seed whose HoT pool overflowed. Templates and host-device inline functions only, as two kernel TUs
// include them (Cands.dev.cc, V2p2.dev.cc).

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using namespace ::mkfitdev;

  // Sync-free driver: grids are sized from the buffer capacity (nSeeds = capacity), the current number of
  // seed rows is read on the device from nDev.
  ALPAKA_FN_HOST_ACC inline int engineRows(int cap, const int32_t* nDev) {
    if (nDev == nullptr)
      return cap;
    const int nRows = *nDev;
    return nRows < cap ? nRows : cap;
  }

  ALPAKA_FN_HOST_ACC inline SeedCandsRef makeSeedRef(
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

  // A seed whose HoT pool overflowed is failed: Finished, no candidates (excluded from export by the filters).
  template <typename TAcc>
  ALPAKA_FN_ACC inline void failSeedOnHotOverflow(TAcc const& acc, SeedCandsSoA::View seeds, int s, uint32_t ovf0) {
    if (seeds.overflowBits(s) & kOverflowHotsBit) {
      seeds.state(s) = kFinished;
      seeds.nCands(s) = 0;
      seeds.nActive(s) = 0;
      seeds.activeMask(s) = 0;
      seeds.nUpdates(s) = 0;
      seeds.bestShortValid(s) = 0;
      if (!(ovf0 & kOverflowHotsBit))
        alpaka::atomicAdd(acc, &seeds.nOverflowHots(), 1u, alpaka::hierarchy::Blocks{});
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
