#ifndef RecoTracker_MkFitAlpaka_src_alpaka_cands_CandSeedOpsKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_cands_CandSeedOpsKernels_h

// Per-seed helpers of the clone-engine kernels (makeSeedRef, countHotOverflow) and the filterSeeds kernel
// (MkBuilder::filter_comb_cands per seed: a pass flag; the caller compacts the surviving seeds in MkFitCore order).

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using namespace ::mkfitdev;

  ALPAKA_FN_ACC inline SeedCandsRef makeSeedRef(
      SeedCandsSoA::View seeds, CandSlotsSoA::View slots, CandHotsSoA::View hots, int s, int hps) {
    SeedCandsRef r;
    const int cur = seeds.curBuf(s);
    r.state = &seeds.state(s);
    r.pickupLayer = &seeds.pickupLayer(s);
    r.cands = &slots.book(candSlotRow(s, cur, 0));
    r.states = &slots.state(candSlotRow(s, cur, 0));
    r.nCands = &seeds.nCands(s);
    r.bestShort = &seeds.bestShort(s);
    r.bestShortState = &slots.state(bestShortRow(s));
    r.bestShortValid = &seeds.bestShortValid(s);
    r.hots = &hots.node(hotRow(s, 0, hps));
    r.hotOffset = 0;
    r.hotCap = hps;
    r.nHots = &seeds.nHots(s);
    r.lastHitIdxBeforeBkw = &seeds.lastHitIdxBeforeBkw(s);
    r.nInsideMinusOneBeforeBkw = &seeds.nInsideMinusOneBeforeBkw(s);
    r.nTailMinusOneBeforeBkw = &seeds.nTailMinusOneBeforeBkw(s);
    r.overflowBits = &seeds.overflowBits(s);
    return r;
  }

  ALPAKA_FN_ACC inline void countHotOverflow(Acc1D const& acc, SeedCandsSoA::View seeds, int s, uint32_t ovf0) {
    if ((seeds.overflowBits(s) & kOverflowHotsBit) && !(ovf0 & kOverflowHotsBit))
      alpaka::atomicAdd(acc, &seeds.nOverflowHots(), 1u, alpaka::hierarchy::Blocks{});
  }

  class KernelFilterSeeds {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::View seeds,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::View hots,
                                  int8_t* passed,
                                  bool bkwRep,
                                  bool attemptAllCands,
                                  int minHitsQF,
                                  int nSeeds) const {
      const int hps = seeds.hotsPerSeed();
      for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
        if (seeds.nCands(s) <= 0) {
          passed[s] = 0;
          continue;
        }
        SeedCandsRef r = makeSeedRef(seeds, slots, hots, s, hps);
        passed[s] = filterSeedCands(r, bkwRep, attemptAllCands, minHitsQF) ? 1 : 0;
      }
    }
  };

  inline void filterSeedsImpl(Queue& queue,
                              SeedCandsSoA::View seeds,
                              CandSlotsSoA::View slots,
                              CandHotsSoA::View hots,
                              int8_t* passed,
                              bool bkwRep,
                              bool attemptAllCands,
                              int minHitsQF,
                              int nSeeds) {
    if (nSeeds <= 0)
      return;
    const auto wd = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nSeeds, 128), 128);
    alpaka::exec<Acc1D>(
        queue, wd, KernelFilterSeeds{}, seeds, slots, hots, passed, bkwRep, attemptAllCands, minHitsQF, nSeeds);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
