#ifndef RecoTracker_MkFitAlpaka_src_alpaka_clean_CleanKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_clean_CleanKernels_h

// Kernels of the device duplicate cleaner (StdSeq::clean_duplicates_sharedhits_pixelseed) and of the LST-step
// quality filter (phase1:qfilter_n_hits_pixseed + qfilter_nan_n_silly), with deterministic stable compaction.

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/clean/CleanFunctions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::clean {

  using namespace ::mkfitdev::clean;
  using ::mkfitdev::TrackSoAConstView;
  using ::mkfitdev::TrackSoAView;

  // (phi, cot theta) cells. Cell widths >= the MkFitCore pre-cuts (|dphi| <= 0.37, |dcot theta| <= 0.37), so every pair
  // MkFitCore can flag lies in the same or an adjacent cell. phi is cyclic; cot theta outside the range is clamped into
  // the edge cells, which keeps the neighbour property. Tracks with a non-finite or absurd phi / cot theta are
  // "wildcards" and are paired with every track (MkFitCore would evaluate those pairs too).
  constexpr int kNPhiCells = 16;  // width 2 pi / 16 = 0.393
  constexpr int kNCtCells = 64;   // width 0.375 over [-12, 12)
  constexpr float kCtCellWidth = 0.375f;
  constexpr float kCtMin = -0.5f * kNCtCells * kCtCellWidth;
  constexpr int kNCells = kNPhiCells * kNCtCells;
  constexpr float kMaxBinnablePhi = 100.f;

  // counters (device int buffer)
  enum Counter : int {
    kCntWildcard = 0,
    kCntCellOverflow = 1,
    kNCounters = 8
  };  // overflow must stay 0 (consistency check)

  // Exclusive prefix scan of in[0..n) into out[0..n], out[n] = total; one block, deterministic. n is read on device
  // from *nPtr (+ nExtra) so that no host synchronisation is needed.
  struct KernelScanOneBlock {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, const int* in, int* out, const int32_t* nPtr, int nConst) const {
      const int n = nPtr ? *nPtr : nConst;
      auto& partial = alpaka::declareSharedVar<int[256], __COUNTER__>(acc);  // <= 256 threads per block
      const int nThr = static_cast<int>(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const int tid = static_cast<int>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const int chunk = (n + nThr - 1) / nThr;
      const int b = tid * chunk;
      const int e = (b + chunk < n) ? b + chunk : n;
      int s = 0;
      for (int k = b; k < e; ++k)
        s += in[k];
      partial[tid] = s;
      alpaka::syncBlockThreads(acc);
      if (tid == 0) {
        int run = 0;
        for (int k = 0; k < nThr; ++k) {
          const int v = partial[k];
          partial[k] = run;
          run += v;
        }
        out[n] = run;
      }
      alpaka::syncBlockThreads(acc);
      int run = partial[tid];
      for (int k = b; k < e; ++k) {
        const int v = in[k];
        out[k] = run;
        run += v;
      }
    }
  };

  // Per track: cot theta (MkFitCore: 1.f / std::tan(theta)), hasPixel (pixelPriority), cell, clear flags, count cell
  // occupancy.
  struct KernelPrepare {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TrackSoAConstView trk,
                                  DupCleanParams p,
                                  float* ct,
                                  int8_t* hasPix,
                                  int* cell,
                                  int* flags,
                                  int* cellCount,
                                  int* wild,
                                  int* counters) const {
      const int n = trk.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        flags[i] = 0;
        const float c = 1.f / alpaka::math::tan(acc, trk[i].params().v[5]);
        ct[i] = c;
        hasPix[i] = (p.pixelPriority && hasPixelHits(trk, i, p)) ? 1 : 0;
        const float phi = trk[i].params().v[4];
        if (!isFiniteBits(c) || !isFiniteBits(phi) || alpaka::math::abs(acc, phi) > kMaxBinnablePhi) {
          cell[i] = -1;
          const int w = alpaka::atomicAdd(acc, &counters[kCntWildcard], 1, alpaka::hierarchy::Grids{});
          wild[w] = i;
          continue;
        }
        // full wrap into [-pi, pi) for binning only (the decision uses MkFitCore's squashPhiMinimal)
        float ph = phi - kTwoPI * alpaka::math::floor(acc, (phi + kPI) / kTwoPI);
        int ip = static_cast<int>((ph + kPI) * (kNPhiCells / kTwoPI));
        ip = ip < 0 ? 0 : (ip >= kNPhiCells ? kNPhiCells - 1 : ip);
        const float fc = (c - kCtMin) / kCtCellWidth;
        int ic = fc < 0.f ? 0 : (fc >= float(kNCtCells) ? kNCtCells - 1 : static_cast<int>(fc));
        const int cl = ic * kNPhiCells + ip;
        cell[i] = cl;
        alpaka::atomicAdd(acc, &cellCount[cl], 1, alpaka::hierarchy::Grids{});
      }
    }
  };

  // Fill cell contents (order inside a cell is irrelevant for the result).
  struct KernelFill {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TrackSoAConstView trk,
                                  const int* cell,
                                  const int* cellStart,
                                  int* cellCursor,
                                  int* cellContent,
                                  int* counters) const {
      const int n = trk.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const int cl = cell[i];
        if (cl < 0)
          continue;
        const int pos = cellStart[cl] + alpaka::atomicAdd(acc, &cellCursor[cl], 1, alpaka::hierarchy::Grids{});
        if (pos < cellStart[cl + 1])
          cellContent[pos] = i;
        else
          alpaka::atomicAdd(acc, &counters[kCntCellOverflow], 1, alpaka::hierarchy::Grids{});
      }
    }
  };

  // Neighbour cell k (0..8) of cell cl, or -1 if outside the cot theta range.
  ALPAKA_FN_HOST_ACC inline int neighbourCell(int cl, int k) {
    const int ic = cl / kNPhiCells + (k / 3 - 1);
    if (ic < 0 || ic >= kNCtCells)
      return -1;
    const int ip = (cl % kNPhiCells + (k % 3 - 1) + kNPhiCells) % kNPhiCells;
    return ic * kNPhiCells + ip;
  }

  // Number of partner slots of each track: total occupancy of its 3x3 neighbourhood.
  struct KernelCountPairs {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, TrackSoAConstView trk, const int* cell, const int* cellStart, int* pairCount) const {
      const int n = trk.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const int cl = cell[i];
        int s = 0;
        if (cl >= 0) {
          for (int k = 0; k < 9; ++k) {
            const int nc = neighbourCell(cl, k);
            if (nc >= 0)
              s += cellStart[nc + 1] - cellStart[nc];
          }
        }
        pairCount[i] = s;
      }
    }
  };

  ALPAKA_FN_ACC inline void applyPair(Acc1D const& acc,
                                      TrackSoAConstView trk,
                                      const float* ct,
                                      const int8_t* hasPix,
                                      int i,
                                      int j,
                                      DupCleanParams const& p,
                                      int* flags,
                                      int* counters) {
    const int loser = dupPairDecision(trk, ct, hasPix, i, j, p);
    if (loser >= 0)
      alpaka::atomicOr(acc, &flags[loser], 1, alpaka::hierarchy::Grids{});
  }

  // One thread per candidate pair: global pair index -> (track i, neighbour slot) -> partner j; pairs with j <= i
  // are skipped so each unordered pair is decided once, oriented as MkFitCore (itrack < jtrack).
  struct KernelPairs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TrackSoAConstView trk,
                                  const float* ct,
                                  const int8_t* hasPix,
                                  const int* cell,
                                  const int* cellStart,
                                  const int* cellContent,
                                  const int* pairOffset,
                                  DupCleanParams p,
                                  int* flags,
                                  int* counters) const {
      const int n = trk.nTracks();
      const int nPairs = pairOffset[n];
      // Per-thread walk state. Consecutive pair indices of one thread (CPU backends: contiguous element ranges) are
      // advanced incrementally; otherwise (GPU grid stride) the track is found by binary search. Both give the same
      // (i, j) for a given pair index.
      int i = -1, k = 0, s = 0, gPrev = -2;
      for (int32_t g : cms::alpakatools::uniform_elements(acc, nPairs)) {
        if (g == gPrev + 1 && i >= 0) {
          if (g < pairOffset[i + 1]) {
            ++s;  // next slot of the same track
          } else {
            while (pairOffset[i + 1] <= g)  // next track with slots
              ++i;
            k = 0;
            s = g - pairOffset[i];
          }
        } else {
          // last i with pairOffset[i] <= g (binary search; pairOffset is non-decreasing)
          int lo = 0, hi = n - 1;
          while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (pairOffset[mid] <= g)
              lo = mid;
            else
              hi = mid - 1;
          }
          i = lo;
          k = 0;
          s = g - pairOffset[i];
        }
        gPrev = g;
        const int cl = cell[i];
        // walk the neighbour cells from k: find the cell holding slot s
        int nc = neighbourCell(cl, k);
        int sz = nc >= 0 ? cellStart[nc + 1] - cellStart[nc] : 0;
        while (s >= sz) {
          s -= sz;
          ++k;
          nc = neighbourCell(cl, k);
          sz = nc >= 0 ? cellStart[nc + 1] - cellStart[nc] : 0;
        }
        const int j = cellContent[cellStart[nc] + s];
        if (j <= i)
          continue;
        applyPair(acc, trk, ct, hasPix, i, j, p, flags, counters);
      }
    }
  };

  // Wildcard tracks against every other track (expected to be empty on real events; counted).
  struct KernelWildcardPairs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TrackSoAConstView trk,
                                  const float* ct,
                                  const int8_t* hasPix,
                                  const int* wild,
                                  DupCleanParams p,
                                  int* flags,
                                  int* counters) const {
      const int n = trk.nTracks();
      const int nw = counters[kCntWildcard];
      for (int32_t g : cms::alpakatools::uniform_elements(acc, nw * n)) {
        const int w = wild[g / n];
        const int t = g % n;
        if (t == w)
          continue;
        applyPair(acc, trk, ct, hasPix, w < t ? w : t, w < t ? t : w, p, flags, counters);
      }
    }
  };

  // duplicate column from the flags; keep = !duplicate
  struct KernelWriteDupFlags {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, TrackSoAView trk, const int* flags, int* keep) const {
      const int n = trk.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        trk[i].duplicate() = flags[i] ? 1 : 0;
        keep[i] = flags[i] ? 0 : 1;
      }
    }
  };

  // LST-step candidate filter on finished tracks: keep = qfilter_n_hits_pixseed && !hasNanNSillyValues
  struct KernelFilterTracks {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, TrackSoAConstView trk, int minHitsQF, int* keep) const {
      const int n = trk.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        keep[i] = lstStepCandFilter(trk[i].nFoundHits(), trk[i].errors().v, minHitsQF) ? 1 : 0;
      }
    }
  };

  // Stable compaction of rows with keep != 0 (std::remove_if / filter_comb_cands order); dest from the exclusive scan.
  struct KernelCompact {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, TrackSoAConstView in, TrackSoAView out, const int* keep, const int* dest) const {
      const int n = in.nTracks();
      if (cms::alpakatools::once_per_grid(acc)) {
        out.nTracks() = dest[n];
        out.nOverflowTracks() = in.nOverflowTracks();
        out.nOverflowHits() = in.nOverflowHits();
      }
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        if (!keep[i])
          continue;
        const int d = dest[i];
        out[d].params() = in[i].params();
        out[d].errors() = in[i].errors();
        out[d].charge() = in[i].charge();
        out[d].chi2() = in[i].chi2();
        out[d].score() = in[i].score();
        out[d].label() = in[i].label();
        out[d].nTotalHits() = in[i].nTotalHits();
        out[d].nFoundHits() = in[i].nFoundHits();
        out[d].nSeedHits() = in[i].nSeedHits();
        out[d].etaRegion() = in[i].etaRegion();
        out[d].algorithm() = in[i].algorithm();
        out[d].nOverlaps() = in[i].nOverlaps();
        out[d].duplicate() = in[i].duplicate();
        const int nh = in[i].nTotalHits();
        for (int h = 0; h < nh; ++h)
          out[d].hits().hot[h] = in[i].hits().hot[h];
      }
    }
  };

  // ---- Building blocks for filter_comb_cands on the candidate store of the building ----

  // filter_comb_cands(filter, attempt_all_cands = true) per seed: index of the first candidate (front first, then
  // 1, 2, ... in stored order) that passes the LST-step filter, or -1 (seed removed). nFound(j), err(j) are callables.
  template <typename TNFound, typename TErr>
  ALPAKA_FN_HOST_ACC inline int firstPassingCand(int nCands, TNFound nFound, TErr err, int minHitsQF) {
    for (int j = 0; j < nCands; ++j)
      if (lstStepCandFilter(nFound(j), err(j), minHitsQF))
        return j;
    return -1;
  }

  // New region separators after the stable compaction (m_seedEtaSeparators update of filter_comb_cands):
  // newSep[r] = number of passing seeds before the old separator = dest[oldSep[r]] of the exclusive scan.
  struct KernelNewSeparators {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, const int* dest, const int* oldSep, int* newSep, int nRegions) const {
      for (int32_t r : cms::alpakatools::uniform_elements(acc, nRegions))
        newSep[r] = dest[oldSep[r]];
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::clean

#endif
