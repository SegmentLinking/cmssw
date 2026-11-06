#ifndef RecoTracker_MkFitAlpaka_src_alpaka_seeds_SeedKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_seeds_SeedKernels_h

// Device seed import (run_OneIteration up to find_tracks_load_seeds) and best-candidate export
// (MkBuilder::export_best_comb_cands + TrackCand::exportTrack). Nothing depends on atomic order:
//   S1 KernelSeedPrepare   per input seed: seed_post_cleaning flag, phase2:1 region, binnor key, counting bin
//                          (atomic count), last-layer min/max per region (atomic min/max)
//   S2 KernelSeedScan      one block: cleanIdx = exclusive scan of the kept flags (erase() order), bin starts,
//                          region ends (m_seedEtaSeparators), nKept
//   S3 KernelSeedFill      per kept seed: atomic slot in its bin (order inside the bin fixed in S4)
//   S4 KernelSeedRank      per kept seed: pos = binStart + #{bin members with (key, row) < own (key, row)}
//                          (= stable radix order on the key, region-major as import_seeds)
//   S5 KernelSeedInitCands per import position: CombCandidate::importSeed into the cands SoA rows
//   E1 KernelExportFlags / E2 scan / E3 KernelExportBestCands: per seed with candidates, the front candidate ->
//                          TrackSoA row (exportTrack(remove_missing_hits = true)), stable in seed order.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/seeds/SeedFunctions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::seeds {

  using namespace ::mkfitdev;
  using namespace ::mkfitdev::seeds;

  constexpr int kScanBlock = 128;

  // Deterministic exclusive scan by one block: thread t owns a contiguous chunk. Returns the total.
  // partial must be block-shared with >= nThreads + 1 entries.
  template <typename TAcc, typename Get, typename Put>
  ALPAKA_FN_ACC inline int32_t blockExclusiveScan(TAcc const& acc, int32_t n, Get get, Put put, int32_t* partial) {
    const int32_t nT = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    const int32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    const int32_t chunk = (n + nT - 1) / nT;
    const int32_t b = t * chunk;
    const int32_t e = (b + chunk < n) ? b + chunk : n;
    int32_t s = 0;
    for (int32_t i = b; i < e; ++i)
      s += get(i);
    partial[t] = s;
    alpaka::syncBlockThreads(acc);
    if (t == 0) {
      int32_t run = 0;
      for (int32_t k = 0; k < nT; ++k) {
        const int32_t v = partial[k];
        partial[k] = run;
        run += v;
      }
      partial[nT] = run;
    }
    alpaka::syncBlockThreads(acc);
    s = partial[t];
    for (int32_t i = b; i < e; ++i) {
      const int32_t v = get(i);
      put(i, s);
      s += v;
    }
    const int32_t total = partial[nT];
    alpaka::syncBlockThreads(acc);
    return total;
  }

  // ---- S1 ----
  class KernelSeedPrepare {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedSoAView seeds,
                                  SeedPartitionLimits lim,
                                  int32_t* __restrict__ binCount,
                                  float const* __restrict__ hitX,
                                  float const* __restrict__ hitY,
                                  float const* __restrict__ hitZ,
                                  uint32_t const* __restrict__ layerHitBase) const {
      const int32_t n = seeds.nSeeds();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const bool silly = seedHasSillyValues(seeds[i].errors().v);
        seeds[i].silly() = silly ? 1 : 0;
        seeds[i].pos() = -1;
        if (silly) {
          seeds[i].region() = -1;
          seeds[i].key() = 0;
          continue;
        }
        const SeedKin kin = seedKin(seeds[i].params().v, seeds[i].charge());
        const int reg = seedRegion(kin, lim);
        const int nh = seeds[i].nHits();
        float hx = seeds[i].lastX(), hy = seeds[i].lastY(), hz = seeds[i].lastZ();
        if (layerHitBase != nullptr && nh > 0) {
          // device EventOfHits: MkFitCore eoh[layer].refHit(index) == HitSoA row layers.hitBase[layer] + index
          const auto& last = seeds[i].hits().hot[nh - 1];
          const uint32_t row = layerHitBase[last.layer] + uint32_t(last.index);
          hx = hitX[row];
          hy = hitY[row];
          hz = hitZ[row];
        }
        const float phi = ::mkfitdev::getPhi(hx, hy);      // Hit::phi()
        const float eta = ::mkfitdev::getEta(hx, hy, hz);  // Hit::eta()
        const uint32_t key = seedBinnorKey(phi, eta);
        seeds[i].region() = reg;
        seeds[i].key() = key;
        alpaka::atomicAdd(acc, &binCount[seedCountBin(reg, key)], 1, alpaka::hierarchy::Blocks{});
        const int lastLayer = nh > 0 ? seeds[i].hits().hot[nh - 1].layer : 0;
        alpaka::atomicMin(acc, &seeds.minLastLayer().v[reg], lastLayer, alpaka::hierarchy::Blocks{});
        alpaka::atomicMax(acc, &seeds.maxLastLayer().v[reg], lastLayer, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // ---- S2 (one block) ----
  class KernelSeedScan {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedSoAView seeds,
                                  int32_t const* __restrict__ binCount,
                                  int32_t* __restrict__ binStart,
                                  SeedCandsSoA::View sc,
                                  int32_t hotsPerSeed) const {
      auto& partial = alpaka::declareSharedVar<int32_t[kScanBlock + 1], __COUNTER__>(acc);
      const int32_t n = seeds.nSeeds();
      blockExclusiveScan(
          acc,
          n,
          [&](int32_t i) { return seeds[i].silly() ? 0 : 1; },
          [&](int32_t i, int32_t v) { seeds[i].cleanIdx() = seeds[i].silly() ? -1 : v; },
          partial);
      const int32_t total = blockExclusiveScan(
          acc,
          kSeedNBins,
          [&](int32_t i) { return binCount[i]; },
          [&](int32_t i, int32_t v) { binStart[i] = v; },
          partial);
      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        binStart[kSeedNBins] = total;
        seeds.nKept() = total;
        // cands SoA scalars (before S5 counts into them)
        sc.hotsPerSeed() = hotsPerSeed;
        sc.nOverflowHots() = 0;
        sc.nOverflowOpts() = 0;
        sc.nOverflowExtras() = 0;
        for (int r = 0; r < kNSeedRegions; ++r) {
          seeds.regionEnd().v[r] = binStart[(r + 1) * kSeedBinsPerRegion];
          // MkBuilder::import_seeds: "Fix min/max layers"
          if (seeds.minLastLayer().v[r] == 9999)
            seeds.minLastLayer().v[r] = -1;
          if (seeds.maxLastLayer().v[r] == 0)
            seeds.maxLastLayer().v[r] = -1;
        }
      }
    }
  };

  // ---- S3 ----
  class KernelSeedFill {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedSoAConstView seeds,
                                  int32_t const* __restrict__ binStart,
                                  int32_t* __restrict__ binFill,
                                  int32_t* __restrict__ members) const {
      const int32_t n = seeds.nSeeds();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        if (seeds[i].silly())
          continue;
        const int b = seedCountBin(seeds[i].region(), seeds[i].key());
        const int32_t slot = alpaka::atomicAdd(acc, &binFill[b], 1, alpaka::hierarchy::Blocks{});
        members[binStart[b] + slot] = i;
      }
    }
  };

  // ---- S4 ----
  class KernelSeedRank {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedSoAView seeds,
                                  int32_t const* __restrict__ binStart,
                                  int32_t const* __restrict__ members) const {
      const int32_t n = seeds.nSeeds();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        if (seeds[i].silly())
          continue;
        const uint32_t key = seeds[i].key();
        const int b = seedCountBin(seeds[i].region(), key);
        const int32_t s = binStart[b], e = binStart[b + 1];
        int32_t rank = 0;
        for (int32_t k = s; k < e; ++k) {
          const int32_t m = members[k];
          const uint32_t km = seeds[m].key();
          rank += (km < key || (km == key && m < i)) ? 1 : 0;
        }
        seeds[i].pos() = s + rank;
        seeds[s + rank].order() = i;
      }
    }
  };

  // ---- S5: CombCandidate::importSeed (TrackStructures.cc:53-72) at row p of the cands SoA ----
  class KernelSeedInitCands {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedSoAConstView seeds,
                                  SeedCandsSoA::View sc,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::View hots,
                                  int32_t hotsPerSeed) const {
      const int32_t n = seeds.nKept();
      const int32_t cap = sc.metadata().size();
      for (int32_t p : cms::alpakatools::uniform_elements(acc, cap)) {
        if (p >= n) {  // unused rows: no candidates (per-seed kernels may loop over the capacity)
          sc[p].nCands() = 0;
          sc[p].curBuf() = 0;
          sc[p].state() = kFinished;
          sc[p].overflowBits() = 0;
          sc[p].bestShortValid() = 0;
          continue;
        }
        const int32_t i = seeds[p].order();
        const int nh = seeds[i].nHits();
        const int reg = seeds[i].region();
        sc[p].nCands() = 1;
        sc[p].curBuf() = 0;
        sc[p].state() = kDormant;
        sc[p].pickupLayer() = nh > 0 ? seeds[i].hits().hot[nh - 1].layer : -1;  // seed.getLastHitLyr()
        sc[p].layer() = -1;
        sc[p].region() = reg;
        sc[p].activeMask() = 0;
        sc[p].nActive() = 0;
        sc[p].seedOriginIdx() = seeds[i].cleanIdx();
        sc[p].nHots() = 0;
        sc[p].nExtras() = 0;
        sc[p].nUpdates() = 0;
        sc[p].nOverlapUpdates() = 0;
        CandBook bs{};
        bs.score = scoreWorstPossible();  // CombCandidate::reset: m_best_short_cand.setScore(worst)
        bs.lastHitIdx = -1;
        bs.originIndex = -1;
        bs.overlaps.reset();
        sc[p].bestShort() = bs;
        sc[p].bestShortValid() = 0;
        sc[p].lastHitIdxBeforeBkw() = -1;
        sc[p].nInsideMinusOneBeforeBkw() = -1;
        sc[p].nTailMinusOneBeforeBkw() = -1;
        sc[p].overflowBits() = 0;

        // TrackCand(seed, this): TrackBase copy, lastHitIdx_ = -1, nFoundHits_ = 0, TrackCand members default
        CandState& st = slots[candSlotRow(p, 0, 0)].state();
        for (int k = 0; k < 6; ++k)
          st.par[k] = seeds[i].params().v[k];
        for (int k = 0; k < 21; ++k)
          st.err[k] = seeds[i].errors().v[k];
        st.charge = seeds[i].charge();
        st.label = seeds[i].label();
        st.status = statusWithSeedHitsAndRegion(seeds[i].status(), nh, reg);  // setNSeedHits, setEtaRegion
        st.nSeedHits = int32_t(uint32_t(nh) & kStatusNSeedHitsMask);          // getNSeedHits(): 4-bit field
        CandBook c{};
        c.score = 0.f;
        c.chi2 = seeds[i].chi2();
        c.lastHitIdx = -1;
        c.nFound = 0;
        c.nMissing = 0;
        c.nOverlap = 0;
        c.nInsideMinusOne = 0;
        c.nTailMinusOne = 0;
        c.originIndex = -1;
        c.overlaps.reset();

        SeedCandsRef r;
        r.hots = &hots[hotRow(p, 0, hotsPerSeed)].node();
        r.hotOffset = 0;
        r.hotCap = hotsPerSeed;
        r.nHots = &sc[p].nHots();
        r.overflowBits = &sc[p].overflowBits();
        for (int h = 0; h < nh; ++h)
          seedAddHitIdx(r, c, seeds[i].hits().hot[h].index, seeds[i].hits().hot[h].layer, 0.0f);
        c.score = getScoreCand(c, std::abs(1.f / st.par[3]));  // getScoreCand(score_func, cand) defaults
        slots[candSlotRow(p, 0, 0)].book() = c;
        if (sc[p].overflowBits() & kOverflowHotsBit)
          alpaka::atomicAdd(acc, &sc.nOverflowHots(), 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // The device seed count can never address rows past the store (defensive: a count buffer left unwritten).
  ALPAKA_FN_HOST_ACC inline int32_t clampSeedCount(int32_t n, int32_t rows) {
    return n < 0 ? 0 : (n > rows ? rows : n);
  }

  // ---- E1: export flags (seed rows with at least one candidate) ----
  class KernelExportScan {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::ConstView sc,
                                  int32_t const* __restrict__ nSeedsPtr,
                                  int8_t const* __restrict__ passed,
                                  int32_t* __restrict__ dest,
                                  TrackSoAView out) const {
      auto& partial = alpaka::declareSharedVar<int32_t[kScanBlock + 1], __COUNTER__>(acc);
      const int32_t n = clampSeedCount(*nSeedsPtr, sc.metadata().size());
      auto keep = [&](int32_t s) { return sc[s].nCands() > 0 && (passed == nullptr || passed[s] != 0); };
      const int32_t total = blockExclusiveScan(
          acc,
          n,
          [&](int32_t s) { return keep(s) ? 1 : 0; },
          [&](int32_t s, int32_t v) { dest[s] = keep(s) ? v : -1; },
          partial);
      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        const int32_t cap = out.metadata().size();
        out.nTracks() = total < cap ? total : cap;
        out.nOverflowTracks() = total < cap ? 0 : total - cap;
        out.nOverflowHits() = 0;
      }
    }
  };

  // ---- E3: TrackCand::exportTrack(remove_missing_hits = true) of the front candidate (TrackStructures.cc:14-47) ----
  class KernelExportBestCands {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::ConstView sc,
                                  CandSlotsSoA::ConstView slots,
                                  CandHotsSoA::ConstView hots,
                                  int32_t hotsPerSeed,
                                  int32_t const* __restrict__ nSeedsPtr,
                                  int32_t const* __restrict__ dest,
                                  TrackSoAView out,
                                  uint32_t* __restrict__ brokenChains) const {
      const int32_t n = clampSeedCount(*nSeedsPtr, sc.metadata().size());
      const int32_t cap = out.metadata().size();
      for (int32_t s : cms::alpakatools::uniform_elements(acc, n)) {
        const int32_t r = dest[s];
        if (r < 0 || r >= cap)
          continue;
        const int row = candSlotRow(s, sc[s].curBuf(), 0);
        const CandBook& c = slots[row].book();
        const CandState& st = slots[row].state();
        for (int k = 0; k < 6; ++k)
          out[r].params().v[k] = st.par[k];
        for (int k = 0; k < 21; ++k)
          out[r].errors().v[k] = st.err[k];
        out[r].charge() = st.charge;
        out[r].chi2() = c.chi2;
        out[r].score() = c.score;
        out[r].label() = st.label;
        const int nFound = c.nFound;
        out[r].nFoundHits() = nFound;
        out[r].nSeedHits() = int8_t((st.status >> kStatusNSeedHitsShift) & kStatusNSeedHitsMask);
        out[r].etaRegion() = int8_t((st.status >> kStatusEtaRegionShift) & kStatusEtaRegionMask);
        out[r].algorithm() = int8_t((st.status >> 7) & 0x3fu);
        out[r].duplicate() = int8_t((st.status >> 6) & 1u);
        out[r].nOverlaps() = int8_t(c.nOverlap);  // res.setNOverlapHits(nOverlapHits())
        const bool fits = nFound <= kMaxTrkHits;
        const int nKeep = fits ? nFound : kMaxTrkHits;
        out[r].nTotalHits() = nKeep;
        auto& hl = out[r].hits().hot;
        for (int h = 0; h < nKeep; ++h) {
          hl[h].index = -1;
          hl[h].layer = -1;
        }
        int nh = c.nFound + c.nMissing;
        int ch = c.lastHitIdx;
        int good = nFound;
        while (--nh >= 0 && ch >= 0) {
          const HoTNode& node = hots[hotRow(s, ch, hotsPerSeed)].node();
          if (node.index >= 0) {
            --good;
            if (good < nKeep) {
              hl[good].index = node.index;
              hl[good].layer = node.layer;
            }
          }
          ch = node.prev;
        }
        if (!fits)
          alpaka::atomicAdd(acc, &out.nOverflowHits(), 1, alpaka::hierarchy::Blocks{});
        // chain check (C2): a consistent chain gives exactly nFound valid hits in nFound + nMissing nodes
        if (brokenChains != nullptr && (good != 0 || nh >= 0))
          alpaka::atomicAdd(acc, brokenChains, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::seeds

#endif
