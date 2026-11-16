// GlobalRadixSort::sort() and its kernels (MkSeederRadixSort.h)

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederRadixSort.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  namespace {
    constexpr uint32_t kRadixScanThreads = 256;  // the row scan's tile
    constexpr uint32_t kRadixBlocks = 1024;      // the tile kernels' grid limit

    ALPAKA_FN_HOST_ACC inline uint32_t radixDigit(uint64_t keyHigh, uint64_t keyLow, int b) {
      return b < 8 ? uint32_t(keyLow >> (8 * b)) & 0xffu : uint32_t(keyHigh >> (8 * (b - 8))) & 0xffu;
    }
    // the digit of a key in the word that holds byte b
    ALPAKA_FN_HOST_ACC inline uint32_t radixWordDigit(uint64_t word, int b) {
      return uint32_t(word >> (8 * (b & 7))) & 0xffu;
    }

    // The GPU's tile kernels: a block per tile of kRadixTile keys, kRadixWarps warps, warp w on the tile's w-th
    // kRadixWarp keys; a warp reads its keys coalesced, warpSize at a time, and ranks them stably by digit with ballots
    // (the lanes with the same digit as this lane among the active ones: peers; its rank: the peers below it)
    template <typename TAcc>
    ALPAKA_FN_ACC inline auto digitPeers(const TAcc &acc, uint32_t d, bool active) {
      auto peers = alpaka::warp::ballot(acc, active);
      for (int k = 0; k < 8; ++k) {
        const auto bk = alpaka::warp::ballot(acc, (d >> k) & 1u);
        peers &= ((d >> k) & 1u) ? bk : ~bk;
      }
      return peers;
    }
    template <typename TMask>
    ALPAKA_FN_ACC inline TMask lanesBelow(TMask peers, uint32_t lane) {
      return peers & ((TMask(1) << lane) - 1);
    }
    // the number of keys (at most capacity)
    ALPAKA_FN_ACC inline uint32_t radixCount(const uint32_t *count, uint32_t capacity) {
      return *count < capacity ? *count : capacity;
    }
    // per warp of a tile its keys' digit counts, cnt[w][d] (zeroed first); all threads of the block call it
    template <typename TAcc, typename TCounts>
    ALPAKA_FN_ACC inline void warpDigitCounts(
        const TAcc &acc, const uint64_t *keyWord, int b, uint32_t n, uint32_t tile, TCounts &cnt) {
      const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const uint32_t ws = uint32_t(alpaka::warp::getSize(acc)), w = t / ws, lane = t % ws;
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] == kRadixWarps * ws));
      for (uint32_t d = lane; d < 256; d += ws)
        cnt[w][d] = 0;
      alpaka::syncBlockThreads(acc);
      for (uint32_t k = 0; k < kRadixWarp; k += ws) {
        const uint32_t e = tile * kRadixTile + w * kRadixWarp + k + lane;
        const bool active = e < n;
        const uint32_t d = active ? radixWordDigit(keyWord[e], b) : 0u;
        const auto peers = digitPeers(acc, d, active);
        if (active && lanesBelow(peers, lane) == 0)
          cnt[w][d] += alpaka::popcount(acc, peers);
        alpaka::syncBlockThreads(acc);
      }
    }

    // per tile of the first *count keys its digit counts: tileCounts[d * nTiles + tile]
    class KernelRadixHistogram {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t capacity,
                                    const uint64_t *keyWord,
                                    int b,
                                    uint32_t *tileCounts,
                                    uint32_t nTiles) const {
        // per warp its counts per digit (rows padded against bank conflicts)
        auto &cnt = alpaka::declareSharedVar<uint32_t[kRadixWarps][257], __COUNTER__>(acc);
        const uint32_t n = radixCount(count, capacity), nt = cms::alpakatools::divide_up_by(n, kRadixTile);
        const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        const uint32_t nth = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
        for (uint32_t tile : cms::alpakatools::independent_groups(acc, nt)) {
          warpDigitCounts(acc, keyWord, b, n, tile, cnt);
          for (uint32_t d = t; d < 256; d += nth) {
            uint32_t sum = 0;
            for (uint32_t w = 0; w < kRadixWarps; ++w)
              sum += cnt[w][d];
            tileCounts[d * nTiles + tile] = sum;
          }
          alpaka::syncBlockThreads(acc);
        }
      }
    };

    // a block per digit d: the exclusive prefix of the digit's row of tile counts (the first nt tiles) in place, in
    // coalesced runs of kRadixScanThreads tiles with a block prefix scan and a carried offset; the row's total in
    // totals[d]
    class KernelRadixRowScan {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t capacity,
                                    uint32_t *tileCounts,
                                    uint32_t nTiles,
                                    uint32_t *totals) const {
        auto &run = alpaka::declareSharedVar<uint32_t[kRadixScanThreads], __COUNTER__>(acc);
        auto &wsum = alpaka::declareSharedVar<uint32_t[kRadixScanThreads], __COUNTER__>(acc);
        const uint32_t n = radixCount(count, capacity), nt = cms::alpakatools::divide_up_by(n, kRadixTile);
        const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] == kRadixScanThreads));
        for (uint32_t d : cms::alpakatools::independent_groups(acc, 256u)) {
          uint32_t *row = tileCounts + d * nTiles;
          uint32_t carry = 0;
          for (uint32_t base = 0; base < nt; base += kRadixScanThreads) {
            const uint32_t c = base + t;
            const uint32_t x = c < nt ? row[c] : 0u;
            run[t] = x;
            cms::alpakatools::blockPrefixScan(acc, run, int32_t(kRadixScanThreads), wsum);
            if (c < nt)
              row[c] = carry + run[t] - x;
            carry += run[kRadixScanThreads - 1];
            alpaka::syncBlockThreads(acc);
          }
          if (t == 0)
            totals[d] = carry;
        }
      }
    };

    // the stable scatter of the tiles: the digits' offsets (the exclusive prefix of the totals) per block, a warp's
    // first position of digit d: the offset + the earlier tiles' (the row scan) + the tile's earlier warps' keys of d;
    // a pass whose digit is the same for every key copies the keys as they are (coalesced)
    class KernelRadixScatter {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t capacity,
                                    const uint64_t *keyHigh,
                                    const uint64_t *keyLow,
                                    const uint32_t *index,
                                    uint64_t *keyHighOut,
                                    uint64_t *keyLowOut,
                                    uint32_t *indexOut,
                                    int b,
                                    const uint32_t *tileCounts,
                                    uint32_t nTiles,
                                    const uint32_t *totals) const {
        auto &offs = alpaka::declareSharedVar<uint32_t[256], __COUNTER__>(acc);
        auto &tot = alpaka::declareSharedVar<uint32_t[256], __COUNTER__>(acc);
        auto &wsum = alpaka::declareSharedVar<uint32_t[256], __COUNTER__>(acc);
        // per warp its counts per digit, then its first positions
        auto &next = alpaka::declareSharedVar<uint32_t[kRadixWarps][257], __COUNTER__>(acc);
        const uint32_t n = radixCount(count, capacity), nt = cms::alpakatools::divide_up_by(n, kRadixTile);
        if (n == 0)
          return;
        const uint64_t *keyWord = b < 8 ? keyLow : keyHigh;
        if (totals[radixWordDigit(keyWord[0], b)] == n) {
          for (uint32_t e : cms::alpakatools::uniform_elements(acc, n))
            keyHighOut[e] = keyHigh[e], keyLowOut[e] = keyLow[e], indexOut[e] = index[e];
          return;
        }
        const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        const uint32_t nth = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
        const uint32_t ws = uint32_t(alpaka::warp::getSize(acc)), w = t / ws, lane = t % ws;
        for (uint32_t d = t; d < 256; d += nth)
          tot[d] = totals[d];
        alpaka::syncBlockThreads(acc);
        cms::alpakatools::blockPrefixScan(acc, tot, offs, 256, wsum);
        for (uint32_t d = t; d < 256; d += nth)
          offs[d] -= tot[d];
        alpaka::syncBlockThreads(acc);
        for (uint32_t tile : cms::alpakatools::independent_groups(acc, nt)) {
          warpDigitCounts(acc, keyWord, b, n, tile, next);
          for (uint32_t d = t; d < 256; d += nth) {
            uint32_t run = offs[d] + tileCounts[d * nTiles + tile];
            for (uint32_t v = 0; v < kRadixWarps; ++v) {
              const uint32_t x = next[v][d];
              next[v][d] = run;
              run += x;
            }
          }
          alpaka::syncBlockThreads(acc);
          for (uint32_t k = 0; k < kRadixWarp; k += ws) {
            const uint32_t e = tile * kRadixTile + w * kRadixWarp + k + lane;
            const bool active = e < n;
            const uint32_t d = active ? radixWordDigit(keyWord[e], b) : 0u;
            const auto peers = digitPeers(acc, d, active);
            const auto below = lanesBelow(peers, lane);
            if (active) {
              const uint32_t p = next[w][d] + alpaka::popcount(acc, below);
              keyHighOut[p] = keyHigh[e], keyLowOut[p] = keyLow[e], indexOut[p] = index[e];
            }
            alpaka::syncBlockThreads(acc);
            if (active && below == 0)
              next[w][d] += alpaka::popcount(acc, peers);
            alpaka::syncBlockThreads(acc);
          }
        }
      }
    };

    // a CPU backend: the whole sort in one thread (bit b of digitMask: sort by byte b), the result in keys0
    class KernelRadixSortSerial {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t capacity,
                                    ::mkseeder::MkSeederSortKeysSoA::View keys0,
                                    ::mkseeder::MkSeederSortKeysSoA::View keys1,
                                    uint32_t digitMask) const {
        if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>) {
          if (!cms::alpakatools::once_per_grid(acc))
            return;
          const uint32_t n = *count < capacity ? *count : capacity;
          uint64_t *hs = keys0.keyHigh().data(), *ls = keys0.keyLow().data();
          uint64_t *hd = keys1.keyHigh().data(), *ld = keys1.keyLow().data();
          uint32_t *is = keys0.index().data(), *id = keys1.index().data();
          bool inZero = true;
          std::vector<uint32_t> cnt;
          for (int b = 0; b < 16 && n > 0; ++b) {
            if (!(digitMask >> b & 1u))
              continue;
            // two adjacent bytes in one 16-bit pass
            const int b0 = b, width = b + 1 < 16 && (digitMask >> (b + 1) & 1u) ? 16 : 8;
            if (width == 16)
              ++b;
            auto dig = [&](uint32_t e) -> uint32_t {
              const uint32_t lo = radixDigit(hs[e], ls[e], b0);
              return width == 16 ? lo | radixDigit(hs[e], ls[e], b0 + 1) << 8 : lo;
            };
            cnt.assign(1u << width, 0u);
            for (uint32_t e = 0; e < n; ++e)
              ++cnt[dig(e)];
            if (cnt[dig(0)] == n)
              continue;  // one digit: the order stays
            uint32_t run = 0;
            for (auto &c : cnt) {
              const uint32_t x = c;
              c = run, run += x;
            }
            for (uint32_t e = 0; e < n; ++e) {
              const uint32_t p = cnt[dig(e)]++;
              hd[p] = hs[e], ld[p] = ls[e], id[p] = is[e];
            }
            std::swap(hs, hd), std::swap(ls, ld), std::swap(is, id);
            inZero = !inZero;
          }
          if (!inZero)
            for (uint32_t e = 0; e < n; ++e)
              hd[e] = hs[e], ld[e] = ls[e], id[e] = is[e];
        }
      }
    };
  }  // namespace

  ::mkseeder::MkSeederSortKeysSoA::ConstView GlobalRadixSort::sort(Queue &queue,
                                                                   const uint32_t *count,
                                                                   std::initializer_list<int> digits) {
    using namespace cms::alpakatools;
    const uint32_t nTiles_ = nTiles(capacity_);
    if constexpr (!kGpu) {
      uint32_t mask = 0;
      for (int b : digits)
        mask |= 1u << b;
      alpaka::exec<Acc1D>(
          queue, make_workdiv<Acc1D>(1u, 1u), KernelRadixSortSerial{}, count, capacity_, keys0_, keys1_, mask);
      return keys0_;
    } else {
      // a block per tile (at most kRadixBlocks blocks, grid-stride); a block per digit for the scan
      const uint32_t warp = uint32_t(alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
      const auto wt = make_workdiv<Acc1D>(std::min(nTiles_, kRadixBlocks), kRadixWarps * warp);
      const auto ws = make_workdiv<Acc1D>(256u, kRadixScanThreads);
      uint32_t *totals = counts_.count().data(), *tileCounts = totals + 256;
      ::mkseeder::MkSeederSortKeysSoA::View buffer[2] = {keys0_, keys1_};
      int s = 0;
      for (int b : digits) {
        const int d = s ^ 1;
        const uint64_t *hi = buffer[s].keyHigh().data(), *lo = buffer[s].keyLow().data();
        alpaka::exec<Acc1D>(
            queue, wt, KernelRadixHistogram{}, count, capacity_, b < 8 ? lo : hi, b, tileCounts, nTiles_);
        alpaka::exec<Acc1D>(queue, ws, KernelRadixRowScan{}, count, capacity_, tileCounts, nTiles_, totals);
        alpaka::exec<Acc1D>(queue,
                            wt,
                            KernelRadixScatter{},
                            count,
                            capacity_,
                            hi,
                            lo,
                            (const uint32_t *)buffer[s].index().data(),
                            buffer[d].keyHigh().data(),
                            buffer[d].keyLow().data(),
                            buffer[d].index().data(),
                            b,
                            (const uint32_t *)tileCounts,
                            nTiles_,
                            (const uint32_t *)totals);
        s = d;
      }
      return buffer[s];
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder
