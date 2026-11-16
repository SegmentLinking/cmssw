#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederCleanKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederCleanKernels_h

// The kernels of the quad / triplet cleaning (MkSeederClean.h); the algorithm is described at
// cleanQuadsAndTriplets() in MkSeederClean.dev.cc.

#include <bit>
#include <cstdint>
#include <cstring>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederClean.h"
#include "MkSeederDeviceTypes.h"
#include "MkSeederPhysicsConstants.h"
#include "MkSeederRadixSort.h"
#include "MkSeederScratchSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::clean {

  using SortKeysView = ::mkseeder::MkSeederSortKeysSoA::View;
  using CleanItemsView = ScratchView<::mkseeder::MkSeederCleanItemsSoA>;
  using HitUseView = ScratchView<::mkseeder::MkSeederHitUseSoA>;
  using ::mkseeder::MkSeederItemLists;
  using ::mkseeder::MkSeederSortedHits;

  // the cleaning's inputs and outputs with its state: per finish index / rank (cap rows), per hit (nTot rows), per
  // hit its items by rank
  struct CleanState : CleanInputs {
    mutable CleanItemsView items;
    mutable HitUseView hitUse;
    MkSeederItemLists *hitItems;
  };

  ALPAKA_FN_HOST_ACC inline uint32_t floatBits(float f) { return std::bit_cast<uint32_t>(f); }
  // the order-preserving integer key of a float (the triplets' score)
  ALPAKA_FN_HOST_ACC inline uint32_t fkey(float f) {
    uint32_t u = floatBits(f);
    if (u == 0x80000000u)
      u = 0;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  }
  template <typename TLayers>
  ALPAKA_FN_HOST_ACC inline uint32_t packLayers(const TLayers &l) {
    return uint32_t(uint8_t(l(0) + 1)) << 24 | uint32_t(uint8_t(l(1) + 1)) << 16 | uint32_t(uint8_t(l(2) + 1)) << 8 |
           uint32_t(uint8_t(l(3) + 1));
  }
  // a hit's cluster-order position
  ALPAKA_FN_ACC inline uint32_t cpos(const CleanInputs &v, int layer, uint32_t hit) {
    return v.layerGeometry[layer].isPixel() ? v.pixelClusters.hitPosition(hit) : v.otClusters.hitPosition(hit);
  }
  ALPAKA_FN_ACC inline int nShared(const MkSeederSortedHits &a, const MkSeederSortedHits &b) {
    int ns = 0;
    for (int x = 0; x < 4; ++x)
      for (int y = 0; y < 4; ++y)
        ns += a.hit[x] == b.hit[y] && a.hit[x] != 0xffffffffu;
    return ns;
  }
  // the items on hit h by rank: [first, last)
  ALPAKA_FN_ACC inline const uint32_t *listBegin(const MkSeederItemLists *lists, uint32_t h) { return lists->begin(h); }
  ALPAKA_FN_ACC inline const uint32_t *listEnd(const MkSeederItemLists *lists, uint32_t h) { return lists->end(h); }

  // per quad: the key of the pattern order (its pattern's position, np for any other combination)
  class KernelFillPatternKeys {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v, SortKeysView keys) const {
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const auto q = v.quads[v.items.quadOfSortedIndex(i)];
        const uint32_t key = packLayers(q.layers());
        int lo0 = 0, hi0 = v.parameters.nPatternTableEntries;
        while (lo0 < hi0) {
          const int m = (lo0 + hi0) / 2;
          if (v.patterns[m].layerKey() < key)
            lo0 = m + 1;
          else
            hi0 = m;
        }
        const int pi = lo0 < v.parameters.nPatternTableEntries && v.patterns[lo0].layerKey() == key
                           ? v.patterns[lo0].patternIndex()
                           : v.parameters.nPatterns;
        keys.keyHigh(i) = 0, keys.keyLow(i) = uint64_t(pi), keys.index(i) = i;
      }
    }
  };

  // per finish index f (the position in the pattern order): the quad (the pattern order's sorted indices' quads), the rank key (quads: high 0, the
  // cleaning key; triplets: high 1, (failed the triplet MLP, fkey(score)))
  class KernelFillRankKeys {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  CleanState v,
                                  ::mkseeder::MkSeederSortKeysSoA::ConstView patternOrder,
                                  SortKeysView keys) const {
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t f : cms::alpakatools::uniform_elements(acc, n)) {
        const uint32_t qi = v.items.quadOfSortedIndex(patternOrder.index(f));
        v.items.quadOfFinishIndex(f) = qi;
        const auto q = v.quads[qi];
        keys.index(f) = f;
        if (q.layers()(3) >= 0) {
          int t = 0;
          for (int k = 0; k < 4; ++k)
            t += !v.layerGeometry[q.layers()(k)].isPixelEnvelope();
          const uint32_t sb = floatBits(q.score());
          uint32_t key;
          if (v.parameters.dedupInner) {
            const int l0 = q.layers()(0);
            const int rk0 = l0 < 16 ? l0 : (l0 < 38 ? l0 - 12 : l0 - 34);
            const uint32_t rk = uint32_t(rk0 < 31 ? rk0 : 31);
            key = uint32_t(t) << 29 | rk << 24 | sb >> 8;
          } else
            key = uint32_t(t) << 29 | sb >> 3;
          keys.keyHigh(f) = 0, keys.keyLow(f) = key;
          alpaka::atomicAdd(acc, &v.counters->nQuads, 1u, alpaka::hierarchy::Blocks{});
        } else {
          const uint64_t tf = v.parameters.flagFailedTriplets && q.tripletFailedMlp() ? 1 : 0;
          keys.keyHigh(f) = 1, keys.keyLow(f) = tf << 32 | fkey(q.score());
          alpaka::atomicAdd(acc, &v.counters->nTriplets, 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // per rank r: the maps, the sorted hits, the initial state, the hits' counts
  class KernelFillRankFields {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  CleanState v,
                                  ::mkseeder::MkSeederSortKeysSoA::ConstView rankOrder) const {
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      const uint32_t nQ = v.counters->nQuads;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, n)) {
        const uint32_t f = rankOrder.index(r);
        v.items.finishIndexOfRank(r) = f, v.items.rankOfFinishIndex(f) = r;
        const auto q = v.quads[v.items.quadOfFinishIndex(f)];
        const int nh = q.layers()(3) < 0 ? 3 : 4;
        MkSeederSortedHits g{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};
        for (int k = 0; k < nh; ++k)
          g.hit[k] = cpos(v, q.layers()(k), q.hits()(k));
        // sorted (the sentinel stays last)
        for (int a = 1; a < 4; ++a)
          for (int b = a; b > 0 && g.hit[b] < g.hit[b - 1]; --b) {
            const uint32_t x = g.hit[b];
            g.hit[b] = g.hit[b - 1], g.hit[b - 1] = x;
          }
        v.items.sortedHits(r) = g;
        for (int k = 0; k < nh; ++k)
          v.hitItems->count(acc, g.hit[k]);
        uint8_t s = 0;
        if (r < nQ && v.parameters.dedup <= 0)
          s = 1;
        v.items.state(r) = s;
        v.items.nAmbiguous(r) = 0;
      }
    }
  };

  class KernelFillHitLists {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v) const {
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, n)) {
        const MkSeederSortedHits g = v.items.sortedHits(r);
        for (int k = 0; k < 4; ++k) {
          if (g.hit[k] == 0xffffffffu)
            break;
          v.hitItems->fill(acc, g.hit[k], r);
        }
      }
    }
  };

  // per one of lists: its items in increasing order (insertion sort: the lists are short; the fill's order is the
  // atomics')
  class KernelSortHitLists {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, MkSeederItemLists *lists, uint32_t nOnes) const {
      for (uint32_t h : cms::alpakatools::uniform_elements(acc, nOnes)) {
        uint32_t *e = lists->content.data() + lists->off[h];
        const uint32_t m = lists->off[h + 1] - lists->off[h];
        for (uint32_t a = 1; a < m; ++a) {
          const uint32_t x = e[a];
          uint32_t b = a;
          for (; b > 0 && e[b - 1] > x; --b)
            e[b] = e[b - 1];
          e[b] = x;
        }
      }
    }
  };

  // the quads' cleaning of the undecided quad of rank r: dropped if a kept better-ranked quad shares >= dedup hits
  // with it, kept if every better-ranked quad sharing >= dedup hits is dropped; false: still undecided. The states
  // only go from undecided to final, so the latest states of the others (other threads' included) can be read
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool quadStep(const TAcc &acc, const CleanState &v, uint32_t r) {
    const MkSeederSortedHits g = v.items.sortedHits(r);
    bool drop = false, blocked = false;
    for (int a = 0; a < 4 && !drop; ++a) {
      const uint32_t h = g.hit[a];
      for (const uint32_t *e = listBegin(v.hitItems, h), *e1 = listEnd(v.hitItems, h); e < e1; ++e) {
        const uint32_t r2 = *e;
        if (r2 >= r)
          break;
        const uint8_t s2 = v.items.state(r2);
        if (s2 == 2)
          continue;
        if (nShared(g, v.items.sortedHits(r2)) < v.parameters.dedup)
          continue;
        if (s2 == 1) {
          drop = true;
          break;
        }
        blocked = true;
      }
    }
    if (drop)
      v.items.state(r) = 2;
    else if (!blocked)
      v.items.state(r) = 1;
    return !drop && blocked;
  }

  // one round of the quads' cleaning
  class KernelCleanQuadsRound {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v, int round) const {
      if (round > 0 && v.counters->undecidedAfterQuadRound[round - 1] == 0)
        return;
      const uint32_t nQ = v.counters->nQuads;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, v.cap)) {
        if (r >= nQ)
          break;
        if (v.items.state(r) == 0 && quadStep(acc, v, r))
          alpaka::atomicAdd(acc, &v.counters->undecidedAfterQuadRound[round], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // one block: the passes over the n items that decide the undecided ones (step(r): false when decided) until none
  // is left; the best-ranked undecided item is decided in every pass (all the better-ranked ones are), so the
  // passes end, with the decisions of the rounds (each is fixed by the better-ranked items' final states)
  template <typename TAcc, typename TStep>
  ALPAKA_FN_ACC inline void completeInOneBlock(const TAcc &acc, uint32_t first, uint32_t n, const TStep &step) {
    auto &undecided = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    while (true) {
      if (cms::alpakatools::once_per_block(acc))
        undecided = 0;
      alpaka::syncBlockThreads(acc);
      for (uint32_t r : cms::alpakatools::independent_group_elements(acc, first, n))
        if (step(r))
          alpaka::atomicAdd(acc, &undecided, 1u, alpaka::hierarchy::Threads{});
      alpaka::syncBlockThreads(acc);
      const uint32_t left = undecided;
      alpaka::syncBlockThreads(acc);
      if (left == 0)
        break;
    }
  }

  // after the rounds (one block): the quads they left undecided
  class KernelCleanQuadsComplete {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v, int rounds) const {
      if (rounds > 0 && v.counters->undecidedAfterQuadRound[rounds - 1] == 0)
        return;
      completeInOneBlock(
          acc, 0u, v.counters->nQuads, [&](uint32_t r) { return v.items.state(r) == 0 && quadStep(acc, v, r); });
    }
  };

  // a dropped quad's n_amb charge: the walk over the dedup-th shortest kept lists (lengths when it was cleaned,
  // ties by the hit's position), each from its latest (best-ranked below) kept quad; the first sharing >= dedup hits
  class KernelResolveAmbiguousQuads {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v) const {
      const uint32_t nQ = v.counters->nQuads;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, v.cap)) {
        if (r >= nQ)
          break;
        if (v.items.state(r) != 2)
          continue;
        const MkSeederSortedHits g = v.items.sortedHits(r);
        int ln[4];
        for (int a = 0; a < 4; ++a) {
          int c = 0;
          for (const uint32_t *e = listBegin(v.hitItems, g.hit[a]), *e1 = listEnd(v.hitItems, g.hit[a]); e < e1; ++e) {
            const uint32_t r2 = *e;
            if (r2 >= r)
              break;
            c += v.items.state(r2) == 1;
          }
          ln[a] = c;
        }
        int pos[4];
        for (int a = 0; a < 4; ++a) {
          int k = 0;
          for (int b = 0; b < 4; ++b)
            k += ln[b] < ln[a] || (ln[b] == ln[a] && b < a);
          pos[k] = a;
        }
        bool done = false;
        for (int w = 0; w < v.parameters.nSharedHitWalk && !done; ++w) {
          const uint32_t h = g.hit[pos[w]];
          const uint32_t *e0 = listBegin(v.hitItems, h);
          for (const uint32_t *e = listEnd(v.hitItems, h); e > e0; --e) {
            const uint32_t r2 = *(e - 1);
            if (r2 >= r || v.items.state(r2) != 1)
              continue;
            if (nShared(g, v.items.sortedHits(r2)) >= v.parameters.dedup) {
              alpaka::atomicAdd(acc, &v.items.nAmbiguous(r2), 1, alpaka::hierarchy::Blocks{});
              done = true;
              break;
            }
          }
        }
      }
    }
  };

  // whether the kept quad of rank r takes part in the triplet veto (min_pix_ot: not with >= 2 outer-tracker hits)
  ALPAKA_FN_ACC inline bool quadVetoesTriplets(const CleanState &v, uint32_t r) {
    const auto q = v.quads[v.items.quadOfFinishIndex(v.items.finishIndexOfRank(r))];
    // trip_veto_fk: a fake-like kept quad does not veto
    if (v.parameters.tripletVetoMaxQuadFakeScore > 0 && !(q.fakeScore() < v.parameters.tripletVetoMaxQuadFakeScore))
      return false;
    if (v.parameters.outerTrackerQuadsVetoTriplets)
      return true;
    int t = 0;
    for (int k = 0; k < 4; ++k)
      t += !v.layerGeometry[q.layers()(k)].isPixelEnvelope();
    return t < 2;
  }

  class KernelMarkQuadHitsUsed {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v) const {
      const uint32_t nQ = v.counters->nQuads;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, v.cap)) {
        if (r >= nQ)
          break;
        if (v.items.state(r) == 1 && quadVetoesTriplets(v, r)) {
          const MkSeederSortedHits g = v.items.sortedHits(r);
          for (int k = 0; k < 4; ++k)
            v.hitUse.usedByKeptQuad(g.hit[k]) = 1;
        }
      }
    }
  };

  // the triplet veto of the undecided triplet of rank r: dropped when >= trip_clean_min of its hits are used by kept
  // quads (trip_veto_quad: held by one kept quad) or kept better-ranked triplets, kept when that cannot happen any
  // more; false: still undecided (the quads are all decided before the triplets)
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool tripletStep(const TAcc &acc, const CleanState &v, uint32_t r, uint32_t nQ) {
    const MkSeederSortedHits g = v.items.sortedHits(r);
    const bool single = v.parameters.tripletVetoBySingleQuad;
    if (single) {
      int best = 0;
      for (int a = 0; a < 3; ++a)
        for (const uint32_t *e = listBegin(v.hitItems, g.hit[a]), *e1 = listEnd(v.hitItems, g.hit[a]); e < e1; ++e) {
          if (*e >= nQ)
            break;
          if (v.items.state(*e) == 1 && quadVetoesTriplets(v, *e))
            best = std::max(best, nShared(g, v.items.sortedHits(*e)));
        }
      if (best >= v.parameters.tripCleanMin) {
        v.items.state(r) = 2;
        return false;
      }
    }
    int used = 0, unknown = 0;
    for (int a = 0; a < 3; ++a) {
      const uint32_t h = g.hit[a];
      if (!single && v.hitUse.usedByKeptQuad(h)) {
        ++used;
        continue;
      }
      bool u = false, blocked = false;
      for (const uint32_t *e = listBegin(v.hitItems, h), *e1 = listEnd(v.hitItems, h); e < e1; ++e) {
        const uint32_t r2 = *e;
        if (r2 < nQ)
          continue;
        if (r2 >= r)
          break;
        // trip_veto_tsc: a better triplet with a score at or above it never vetoes (kept or not)
        if (v.parameters.tripletVetoMaxTripletScore > 0 &&
            !(v.quads[v.items.quadOfFinishIndex(v.items.finishIndexOfRank(r2))].score() <
              v.parameters.tripletVetoMaxTripletScore))
          continue;
        const uint8_t s2 = v.items.state(r2);
        if (s2 == 1) {
          u = true;
          break;
        }
        blocked |= s2 == 0;
      }
      used += u;
      unknown += !u && blocked;
    }
    if (used >= v.parameters.tripCleanMin)
      v.items.state(r) = 2;
    else if (used + unknown < v.parameters.tripCleanMin)
      v.items.state(r) = 1;
    else
      return true;
    return false;
  }

  // one round of the triplet veto
  class KernelCleanTripletsRound {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v, int round) const {
      if (round > 0 && v.counters->undecidedAfterTripletRound[round - 1] == 0)
        return;
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      const uint32_t nQ = v.counters->nQuads;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, v.cap)) {
        if (r >= n)
          break;
        if (r >= nQ && v.items.state(r) == 0 && tripletStep(acc, v, r, nQ))
          alpaka::atomicAdd(acc, &v.counters->undecidedAfterTripletRound[round], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // after the rounds (one block): the triplets they left undecided
  class KernelCleanTripletsComplete {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v, int rounds) const {
      if (rounds > 0 && v.counters->undecidedAfterTripletRound[rounds - 1] == 0)
        return;
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      const uint32_t nQ = v.counters->nQuads;
      completeInOneBlock(acc, nQ, n, [&](uint32_t r) { return v.items.state(r) == 0 && tripletStep(acc, v, r, nQ); });
    }
  };

  // per finish index: kept first (the output order)
  class KernelFillKeptKeys {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, CleanState v, SortKeysView keys) const {
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t f : cms::alpakatools::uniform_elements(acc, n)) {
        const bool kept = v.items.state(v.items.rankOfFinishIndex(f)) == 1;
        keys.keyHigh(f) = 0, keys.keyLow(f) = kept ? 0 : 1, keys.index(f) = f;
        if (kept)
          alpaka::atomicAdd(acc, &v.counters->nKept, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // per label k: the kept seed (with its hits' bin-order rows, the fit's input); the kept triplets that failed the
  // triplet MLP counted (probeOn: their probe items, KernelBuildProbeItems)
  class KernelBuildKeptSeeds {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  CleanState v,
                                  ::mkseeder::MkSeederSortKeysSoA::ConstView keptOrder) const {
      const uint32_t nk = v.counters->nKept;
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, v.cap)) {
        if (k >= nk)
          break;
        const uint32_t f = keptOrder.index(k);
        const uint32_t r = v.items.rankOfFinishIndex(f);
        const auto q = v.quads[v.items.quadOfFinishIndex(f)];
        const bool trip = q.layers()(3) < 0;
        const int tfail = v.parameters.flagFailedTriplets && trip && q.tripletFailedMlp() ? 1 : 0;
        const int nh = trip ? 3 : 4;
        auto o = v.kept[k];
        o.nHits() = nh;
        for (int j = 0; j < 4; ++j) {
          o.layers()(j) = q.layers()(j);
          o.hitRows()(j) = j < nh ? v.clusterOrder.binOrderPosition(cpos(v, q.layers()(j), q.hits()(j))) : 0u;
        }
        o.score() = q.score(), o.fakeScore() = q.fakeScore(), o.nAmbiguous() = v.items.nAmbiguous(r);
        o.tripletFailedMlp() = tfail;
        if (tfail && v.parameters.probesEnabled)
          alpaka::atomicAdd(acc, &v.counters->nProbes, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // per kept triplet that failed the triplet MLP (probeOn): its probe item (the extension of its hits) with its label
  // (the items' order is the atomics': each item is extended on its own and read by its label)
  class KernelBuildProbeItems {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  KeptSeedsConstView kept,
                                  HitsView hits,
                                  MkSeederCleanCounters *counters,
                                  SeedHitsView probe,
                                  uint32_t nKept) const {
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, nKept)) {
        const auto o = kept[k];
        if (!o.tripletFailedMlp())
          continue;
        const uint32_t p = alpaka::atomicAdd(acc, &counters->nProbesWritten, 1u, alpaka::hierarchy::Blocks{});
        const int nh = o.nHits();
        WorkingSeed ps;
        std::memset(&ps, 0, sizeof(WorkingSeed));
        ps.tripletFailedMlp = 1;
        ps.nHits = nh;
        for (int j = 0; j < nh; ++j)
          ps.layers[j] = o.layers()(j), ps.hitRows[j] = o.hitRows()(j);
        // cot from the first and the last hit (the double sqrt of the exact double squares, rounded: what
        // std::hypot of floats gives)
        const uint32_t a = ps.hitRows[0], c = ps.hitRows[nh - 1];
        const float dx = hits.x(c) - hits.x(a), dy = hits.y(c) - hits.y(a);
        const float dr = float(alpaka::math::sqrt(acc, double(dx) * double(dx) + double(dy) * double(dy)));
        ps.cot = dr > 0 ? (hits.z(c) - hits.z(a)) / dr : 1e30f;
        storeSeed(probe, p, ps);
        probe.label(p) = k;
      }
    }
  };

  // The duplicate cleaning's helix parameters of a fitted state (mkFit's CCS parameters x, y, z, 1/pT, phi, theta); zeros
  // for a failed fit (checkFitOk); the first *count (nullptr: cap) seeds
  class KernelComputeSeedHelices {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  SeedStatesConstView states,
                                  bool checkFitOk,
                                  uint32_t cap,
                                  const uint32_t *count,
                                  SeedHelicesView helices,
                                  float bField) const {
      const uint32_t n = count && *count < cap ? *count : cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const auto t = states[i];
        float qpt = 0, cot = 0, phi0 = 0, z0 = 0, eta = 0, vqpt = 0, vcot = 0;
        if (!checkFitOk || t.fitOk()) {
          const int q = t.charge();
          const auto par = t.parameters();
          const float phi = par(4), th = par(5);
          const float sth = alpaka::math::sin(acc, th);
          const float R = 1.f / (par(3) * kSolOver100 * bField);
          const float cx = par(0) + q * R * alpaka::math::sin(acc, phi),
                      cy = par(1) - q * R * alpaka::math::cos(acc, phi);
          const float rc = std::hypot(cx, cy);
          const float ax = -cx / rc * R, ay = -cy / rc * R;
          const float bx = par(0) - cx, by = par(1) - cy;
          const float alpha = alpaka::math::abs(acc, alpaka::math::atan2(acc, ax * by - ay * bx, ax * bx + ay * by));
          cot = alpaka::math::cos(acc, th) / sth;
          eta = alpaka::math::asinh(acc, cot);
          phi0 = phi + q * alpha;
          z0 = par(2) - cot * R * alpha;
          qpt = q * par(3);
          vqpt = t.covariance()(9);
          vcot = t.covariance()(20) / (sth * sth * sth * sth);
        }
        auto h = helices[i];
        h.qOverPt() = qpt, h.cot() = cot, h.phi0() = phi0, h.z0() = z0, h.eta() = eta, h.varQOverPt() = vqpt;
        h.varCot() = vcot;
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::clean

#endif
