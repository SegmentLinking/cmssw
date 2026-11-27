#ifndef RecoTracker_MkFitAlpaka_src_alpaka_cands_CandSeedOpsKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_cands_CandSeedOpsKernels_h

// Per-seed helpers of the clone-engine and v2p2 kernels (makeSeedRef, engineRows, the HoT pool: reserveHots,
// attachHots, failSeedOnHotOverflow); inline only, as two kernel TUs include them.

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

  // The HoT pool of the seed rows as plain pointers and sizes (CandsSoA.h hotRow), taken once per kernel thread.
  struct HotPool {
    HoTNode* nodes;
    int32_t* ext;    // SeedCandsSoA::hotExt
    int32_t* nHots;  // SeedCandsSoA::nHots
    uint32_t* nExt;  // SeedCandsSoA::nHotExt
    int seedRows;
    int home;
    int perSeed;
    int extSlots;
  };

  ALPAKA_FN_HOST_ACC inline HotPool hotPoolOf(SeedCandsSoA::View seeds, CandHotsSoA::View hots) {
    return HotPool{hots.metadata().addressOf_node(),
                   seeds.metadata().addressOf_hotExt(),
                   seeds.metadata().addressOf_nHots(),
                   &seeds.nHotExt(),
                   seeds.metadata().size(),
                   seeds.hotsHome(),
                   seeds.hotsPerSeed(),
                   seeds.hotExtSlots()};
  }

  // The HoT nodes of seed s in the pool: home rows, and the extension rows if it holds a slot.
  template <typename TRef>
  ALPAKA_FN_HOST_ACC inline void attachHots(HotPool const& pool, int s, TRef& ref) {
    const int ext = pool.ext[s];
    ref.hots = pool.nodes + hotRow(s, 0, ext, pool.seedRows, pool.home, pool.perSeed);
    ref.extHots = ext >= 0 ? pool.nodes + hotRow(s, pool.home, ext, pool.seedRows, pool.home, pool.perSeed) : nullptr;
    ref.hotHome = pool.home;
    ref.hotCap = ext >= 0 ? pool.perSeed : pool.home;
  }

  ALPAKA_FN_HOST_ACC inline SeedCandsRef makeSeedRef(SeedCandsSoA::View seeds,
                                                     CandSlotsSoA::View slots,
                                                     CandHotsSoA::View hots,
                                                     int s) {
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
    attachHots(hotPoolOf(seeds, hots), s, ref);
    ref.nHots = &seeds.nHots(s);
    ref.lastHitIdxBeforeBkw = &seeds.lastHitIdxBeforeBkw(s);
    ref.nInsideMinusOneBeforeBkw = &seeds.nInsideMinusOneBeforeBkw(s);
    ref.nTailMinusOneBeforeBkw = &seeds.nTailMinusOneBeforeBkw(s);
    ref.overflowBits = &seeds.overflowBits(s);
    return ref;
  }

  // Before appending up to `need` HoT nodes: a seed that could pass its home rows takes an extension slot (once, to the
  // end of the event); with none left it stays at home, and a node past it fails the seed (counted).
  template <typename TAcc>
  ALPAKA_FN_ACC inline void reserveHots(TAcc const& acc, HotPool const& pool, int s, int need) {
    if (pool.ext[s] >= 0 || pool.nHots[s] + need <= pool.home)
      return;
    const uint32_t slot = alpaka::atomicAdd(acc, pool.nExt, 1u, alpaka::hierarchy::Blocks{});
    if (slot < uint32_t(pool.extSlots))
      pool.ext[s] = int32_t(slot);
  }

  // The HoT pool of one seed for code that appends nodes from several lanes (MkFinderV2p2 select_and_materialise):
  // one lane reserves, then every lane attaches its own SeedCandsRef.
  struct SeedHotPool {
    HotPool const* pool;
    int s;
    template <typename TAcc>
    ALPAKA_FN_ACC void reserve(TAcc const& acc, int need) const {
      reserveHots(acc, *pool, s, need);
    }
    ALPAKA_FN_HOST_ACC void attach(SeedCandsRef& ref) const { attachHots(*pool, s, ref); }
  };

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
