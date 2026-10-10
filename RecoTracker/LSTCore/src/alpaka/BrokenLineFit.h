#ifndef RecoTracker_LSTCore_src_alpaka_BrokenLineFit_h
#define RecoTracker_LSTCore_src_alpaka_BrokenLineFit_h

#include <cstdint>
#include <iterator>
#include <limits>

#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelTrackFitting/interface/FitResult.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

#include "LSTEvent.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Hit counts N for which a Kernel_LSTBLFit<N> instantiation exists, descending.
  // Keep in sync with the launchBLFKernelN* calls in BrokenLineFit.dev.cc.
  inline constexpr int kBLFitSizes[] = {14, 12, 10, 8, 6, 5};
  inline constexpr int kBLFitNumSizes = std::size(kBLFitSizes);
  inline constexpr int kBLFitMaxHits = kBLFitSizes[0];
  inline constexpr int kBLFitMinHits = kBLFitSizes[kBLFitNumSizes - 1];
  static_assert(kBLFitNumSizes == 6,
                "blfFitHits tests one size per instantiation, spelled out; adding or removing an entry in "
                "kBLFitSizes means editing it by hand");
  static_assert(kBLFitSizes[1] == 12 && kBLFitSizes[2] == 10 && kBLFitSizes[3] == 8 && kBLFitSizes[4] == 6,
                "blfFitHits spells the intermediate sizes out; keep it in step with kBLFitSizes");

  // The two sensor hits of an OT mini-doublet are collapsed into a SINGLE fitted hit when their
  // global (x, y) lie within 1e-4 cm of each other. Squared, in cm^2, so the test is a plain
  // dx * dx + dy * dy comparison with no square root in device code.
  // The value sits in the empty part of the measured endcap 2S separation spectrum -- 2778
  // pairs at exactly 0, 76 in [1e-7, 1e-5) cm, then nothing until 0.0044 cm -- so every
  // tolerance in that gap flags the same pairs, while exact equality misses the 76 that
  // differ only in the float rounding of the global transform at two different z.
  inline constexpr float kMaxDegenSepXY2 = 1e-4f * 1e-4f;

  // Largest instantiated hit count not exceeding nHits, or 0 when nHits is below the
  // smallest one, in which case no kernel claims the candidate and it stays unfit.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr int blfFitHits(int nHits) {
    if (nHits >= kBLFitMaxHits)
      return kBLFitMaxHits;
    if (nHits >= 12)
      return 12;
    if (nHits >= 10)
      return 10;
    if (nHits >= 8)
      return 8;
    if (nHits >= 6)
      return 6;
    if (nHits >= kBLFitMinHits)
      return kBLFitMinHits;
    return 0;
  }

  // Loop through the list of hits with a constant increment and force the last entry,
  // which keeps the maximum lever arm. Requires nSrc >= M, which blfFitHits guarantees.
  // For nSrc == M the increment is exactly 1 and the mapping is the identity,
  // so such a candidate is fitted on all the hits in order.
  template <int M>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void selectFitHits(const unsigned int* src, int nSrc, unsigned int (&dst)[M]) {
    ALPAKA_ASSERT_ACC(nSrc >= M);
    float incr = static_cast<float>(nSrc) / static_cast<float>(M);
    if (incr < 1.f)
      incr = 1.f;
    float n = 0.f;
    for (int i = 0; i < M; ++i) {
      int j = static_cast<int>(n + 0.5f);  // round
      if (M - 1 == i)
        j = nSrc - 1;
      ALPAKA_ASSERT_ACC(j < nSrc);
      n += incr;
      dst[i] = src[j];
    }
  }

  // Shed exactly one hit, keeping the order of the others.
  template <int M>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void dropFitHit(const unsigned int* src, int dropIdx, unsigned int (&dst)[M]) {
    ALPAKA_ASSERT_ACC(dropIdx >= 0 && dropIdx <= M);
    for (int i = 0; i < M; ++i)
      dst[i] = src[i + (i >= dropIdx ? 1 : 0)];
  }

#if LST_BLF_PIXEL_HITS
  // Everything from here to the matching #endif exists only to support pixel seed rec hits as
  // fit hits. With LST_BLF_PIXEL_HITS off none of it is compiled and the kernel below takes
  // the outer-tracker-only path.

  // Stands for a fitted hit that is not a Phase-2 OT rec hit, so it can never collide with one.
  inline constexpr unsigned int kBLFNoOTIdx = 0xFFFFFFFF;

  // The two blocks a fit hit's measurement can live in, and the row index that separates them,
  // bundled so that the hit accessors take one argument instead of three and so that nHitsOT is
  // read out of the SoA once per candidate rather than once per lookup.
  struct BLFHits {
    HitsBaseConst hitsBase;
    SeedHitsConst seedHits;
    unsigned int nHitsOT;
  };

  // A fitted hit's measurement lives in a different block depending on which region of HitsBaseSoA
  // its row falls in:
  //   - A row below nHitsOT is a Phase-2 OT rec hit and carries its position in
  //     xs/ys/zs and its error in ge.
  //   - A row at or above nHitsOT is a pixel seed row, whose xs/ys/zs hold seed-state pseudo-hits
  //     (r3PCA, a (pT, eta, phi) triple, r3LH) and whose ge is zero, so its measurement is the seed
  //     rec hit position and error in SeedHitsSoA, at row hIdx - nHitsOT.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void blfHitPosition(
      BLFHits const& blfHits, unsigned int hIdx, float& x, float& y, float& z) {
    if (hIdx >= blfHits.nHitsOT) {
      const unsigned int sIdx = hIdx - blfHits.nHitsOT;
      x = blfHits.seedHits.xs()[sIdx];
      y = blfHits.seedHits.ys()[sIdx];
      z = blfHits.seedHits.zs()[sIdx];
    } else {
      x = blfHits.hitsBase.xs()[hIdx];
      y = blfHits.hitsBase.ys()[hIdx];
      z = blfHits.hitsBase.zs()[hIdx];
    }
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE ArrayFx6 const& blfHitError(BLFHits const& blfHits, unsigned int hIdx) {
    return hIdx >= blfHits.nHitsOT ? blfHits.seedHits.ge()[hIdx - blfHits.nHitsOT] : blfHits.hitsBase.ge()[hIdx];
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int blfHitOTIdx(BLFHits const& blfHits, unsigned int hIdx) {
    if (hIdx < blfHits.nHitsOT)
      return hIdx;
    return blfHits.hitsBase.detid()[hIdx] == kPixelModuleId ? kBLFNoOTIdx : blfHits.hitsBase.idxs()[hIdx];
  }

  // Whether a Phase-2 OT rec hit is already among the collected hits. A kBLFNoOTIdx query is
  // never a match: it means "not an OT rec hit", not one particular one.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool blfHasOTIdx(const unsigned int* otIdxs, int nIdxs, unsigned int otIdx) {
    if (otIdx == kBLFNoOTIdx)
      return false;
    for (int i = 0; i < nIdxs; ++i)
      if (otIdxs[i] == otIdx)
        return true;
    return false;
  }

  // Order the collected hits by 3D distance from the origin, innermost first.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void blfSortHitsInsideOut(BLFHits const& blfHits, unsigned int* hitIdxs, int nIdxs) {
    ALPAKA_ASSERT_ACC(nIdxs <= static_cast<int>(Params_TC::kHits));
    float d2[Params_TC::kHits];
    for (int i = 0; i < nIdxs; ++i) {
      float x, y, z;
      blfHitPosition(blfHits, hitIdxs[i], x, y, z);
      d2[i] = x * x + y * y + z * z;
    }
    for (int i = 1; i < nIdxs; ++i) {
      const unsigned int hIdx = hitIdxs[i];
      const float key = d2[i];
      int j = i - 1;
      while (j >= 0 && d2[j] > key) {
        hitIdxs[j + 1] = hitIdxs[j];
        d2[j + 1] = d2[j];
        --j;
      }
      hitIdxs[j + 1] = hIdx;
      d2[j + 1] = key;
    }
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool blfIsPixelHit(BLFHits const& blfHits, unsigned int hIdx) {
    return hIdx >= blfHits.nHitsOT && blfHits.hitsBase.detid()[hIdx] == kPixelModuleId;
  }

  // The pixel hits are kept in full and the OT hits are strided down to fill the remainder,
  // up to M with the LAST OT hit forced.
  template <int M>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void selectFitHitsKeepingPixels(BLFHits const& blfHits,
                                                                 const unsigned int* src,
                                                                 int nSrc,
                                                                 unsigned int (&dst)[M]) {
    ALPAKA_ASSERT_ACC(nSrc >= M);
    int nPix = 0;
    for (int i = 0; i < nSrc; ++i)
      if (blfIsPixelHit(blfHits, src[i]))
        ++nPix;
    ALPAKA_ASSERT_ACC(nPix <= M);

    const int nOT = nSrc - nPix;   // strideable hits available
    const int nWantOT = M - nPix;  // strideable hits to keep
    float incr = nWantOT > 0 ? static_cast<float>(nOT) / static_cast<float>(nWantOT) : 0.f;
    if (incr < 1.f)
      incr = 1.f;

    // Target position, among the OT hits only, of the k-th OT hit to keep. The last one kept is
    // forced to the outermost OT hit for the lever arm, exactly as selectFitHits forces its last
    // entry; the rest are the same rounded uniform stride.
    int nOut = 0, iOT = 0, kOT = 0;
    int nextOT = nWantOT > 0 ? (nWantOT == 1 ? nOT - 1 : 0) : -1;
    for (int i = 0; i < nSrc && nOut < M; ++i) {
      if (blfIsPixelHit(blfHits, src[i])) {
        dst[nOut++] = src[i];
        continue;
      }
      const int nOTLeft = nOT - iOT;  // this one included
      if (iOT == nextOT || nOTLeft <= nWantOT - kOT) {
        dst[nOut++] = src[i];
        ++kOT;
        if (kOT >= nWantOT)
          nextOT = -1;
        else if (kOT == nWantOT - 1)
          nextOT = nOT - 1;  // force the last OT hit, for the lever arm
        else
          nextOT = static_cast<int>(static_cast<float>(kOT) * incr + 0.5f);
      }
      ++iOT;
    }
    ALPAKA_ASSERT_ACC(nOut == M);
  }

#endif  // LST_BLF_PIXEL_HITS
  // Initialise every fit-result column before the fit kernels run:
  //   pt   = -1: the unfit flag every consumer keys on (LSTOutputConverter gates the reco::Track on
  //              pt >= 0; LST.cc and the standalone trkCore.cc count pt != -1). It must not change.
  //   chi2 = -1: chi2 is non-negative by construction, so a negative value cannot be mistaken for a
  //              fit, whereas 0 would look like a perfect one. LSTOutputConverter rebuilds
  //              chi2total = chi2stored * ndof, so a leaked chi2 = 0 would sail through the chi2n
  //              cut in the downstream track selection.
  //   charge = 0: not a valid track charge.
  //   eta, phi, tip, zip and both covariances = 0: a zero covariance is a zero error, which is loud
  //              rather than plausible. The authoritative unfit test remains pt < 0, not these.
  //   nFit = 0: the number of hits the fit used. Only the instantiated hit counts
  //              (5, 6, 8, 10, 12, 14) are valid, so 0 is unphysical.
  //   nDegen = 0: the number of OT mini-doublets whose outer sensor was dropped as degenerate.
  //              Hits are also dropped for other reasons -- a surplus over the instantiated hit
  //              count, and with LST_BLF_PIXEL_HITS on a hit collected twice -- and none of those
  //              is counted here.
  //              0 is also a legal value for a fitted candidate, so it is not a flag
  struct Kernel_InitBLFFit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, TrackCandidatesBLFFit fitResults, unsigned int nTC) const {
      for (unsigned int tcIdx : cms::alpakatools::uniform_elements(acc, nTC)) {
        fitResults.pt()[tcIdx] = -1.f;
        fitResults.eta()[tcIdx] = 0.f;
        fitResults.phi()[tcIdx] = 0.f;
        fitResults.tip()[tcIdx] = 0.f;
        fitResults.zip()[tcIdx] = 0.f;
        fitResults.charge()[tcIdx] = 0;
        fitResults.chi2()[tcIdx] = -1.f;
        fitResults.nFit()[tcIdx] = 0;
        fitResults.nDegen()[tcIdx] = 0;
        auto& cCircle = fitResults.covCircle()[tcIdx];
        for (unsigned int i = 0; i < cCircle.size(); ++i)
          cCircle[i] = 0.f;
        auto& cLine = fitResults.covLine()[tcIdx];
        for (unsigned int i = 0; i < cLine.size(); ++i)
          cLine[i] = 0.f;
      }
    }
  };

  // BLF kernel for TCs whose surviving hit count rounds down to exactly N hits.
  // With LST_BLF_PIXEL_HITS on that count is the pixel seed's rec hits and the OT hits together;
  // with it off it is the OT hits alone.
  // TCs that round down to a different hit count are skipped (handled by another N
  // instantiation), and TCs left with fewer than kBLFitMinHits hits are not fitted at all.
  template <int N>
  struct Kernel_LSTBLFit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const float bField,
                                  TrackCandidatesBaseConst candsBase,
                                  HitsBaseConst hitsBase,
#if LST_BLF_PIXEL_HITS
                                  SeedHitsConst seedHits,
#endif
                                  TrackCandidatesBLFFit fitResults) const {
      const double bFieldD = static_cast<double>(bField);
      const unsigned int nTC = candsBase.nTrackCandidates();
#if LST_BLF_PIXEL_HITS
      // Built once per launch: neither nHitsOT nor hasSeedHitPos is per-candidate.
      const BLFHits blfHits{hitsBase, seedHits, hitsBase.nHitsOT()};
      const bool haveSeedHits = hitsBase.hasSeedHitPos() != 0 && seedHits.metadata().size() != 0;
#endif
      for (unsigned int tcIdx : cms::alpakatools::uniform_elements(acc, nTC)) {
        auto const& hitSlots = candsBase.hitIndices()[tcIdx];

#if LST_BLF_PIXEL_HITS
        // Collect both sensor hits per doublet layer slot (inner sensor first,
        // then outer sensor), skipping empty slots.
        //
        // The outer sensor is dropped when its global (x, y) is within kMaxDegenSepXY2 of
        // the inner one's. sTransverse depends on the hit only through (x, y), so such a
        // pair has a zero or sub-ulp first difference; matrixC_u divides by first
        // differences of sTransverse and invertNN then turns the resulting infinities into
        // a NaN pt, which LSTOutputConverter silently drops. A separation too small to
        // divide by but not exactly zero is as bad: it gives c_uMat entries of order 1e23
        // and a meaningless pt that passes the pt >= 0 gate, which is why the test is a
        // tolerance and not an equality. It is the two sensors of a Phase-2 endcap 2S
        // module that do this, once the track is straight enough for the same strip to fire
        // in both. The line fit is unaffected either way, since it uses sTotal, which still
        // separates the two hits through z.

        unsigned int survHitIdxs[Params_TC::kHits];
        unsigned int survOTIdxs[Params_TC::kHits];
        int nSurv = 0;
        int nDegen = 0;
        // Skipping pLSs as they are fitted upstream.
        // The pixel slots are skipped as well when no seed rec hit positions were supplied, which
        // is the standalone mode currently.
        auto const tcType = candsBase.trackCandidateType()[tcIdx];
        const bool usePixelSlots = haveSeedHits && (tcType == LSTObjType::pT5 || tcType == LSTObjType::pT3);
        for (int slot = usePixelSlots ? 0 : Params_TC::kPixelLayerSlots; slot < Params_TC::kLayers; ++slot) {
          unsigned int h0 = hitSlots[slot][0];
          if (h0 == kTCEmptyHitIdx)
            continue;
          unsigned int h1 = hitSlots[slot][1];

          // Cross-slot duplicate removal, replicating LSTOutputConverter's sharesInput guard:
          // a hit already collected is dropped, so the copy collected first wins.
          const unsigned int ot0 = blfHitOTIdx(blfHits, h0);
          const unsigned int ot1 = blfHitOTIdx(blfHits, h1);
          const bool dup0 = blfHasOTIdx(survOTIdxs, nSurv, ot0);
          const bool dup1 =
              blfHasOTIdx(survOTIdxs, nSurv, ot1) || (!dup0 && (h0 == h1 || (ot0 != kBLFNoOTIdx && ot0 == ot1)));
          if (dup0 && dup1)
            continue;

          if (dup0 || dup1) {
            survOTIdxs[nSurv] = dup0 ? ot1 : ot0;
            survHitIdxs[nSurv++] = dup0 ? h1 : h0;
            continue;
          }

          float x0, y0, z0, x1, y1, z1;
          blfHitPosition(blfHits, h0, x0, y0, z0);
          blfHitPosition(blfHits, h1, x1, y1, z1);

          float d0sq = x0 * x0 + y0 * y0 + z0 * z0;
          float d1sq = x1 * x1 + y1 * y1 + z1 * z1;
          bool swap = d0sq > d1sq;
          survOTIdxs[nSurv] = swap ? ot1 : ot0;
          survHitIdxs[nSurv++] = swap ? h1 : h0;
          float dx = x1 - x0, dy = y1 - y0;
          float sep2 = dx * dx + dy * dy;
          if (sep2 < kMaxDegenSepXY2) {
            if (!blfIsPixelHit(blfHits, h0) && !blfIsPixelHit(blfHits, h1))
              ++nDegen;
            continue;
          }
          survOTIdxs[nSurv] = swap ? ot0 : ot1;
          survHitIdxs[nSurv++] = swap ? h0 : h1;
        }

        // Round the surviving hit count down to the nearest instantiated hit count; the
        // instantiation it lands on claims the candidate, and selectFitHitsKeepingPixels sheds
        // the remaining hits. Without the degeneracy drop and the duplicate removal, nSurv is always
        // even and equal to one of those counts, so this is the identity and the candidate is
        // fitted exactly.
        if (blfFitHits(nSurv) != N)
          continue;

        blfSortHitsInsideOut(blfHits, survHitIdxs, nSurv);

        int dropIdx = -1;
        if (nSurv == N + 1) {
          float dropSep2 = std::numeric_limits<float>::max();
          for (int i = 1; i < nSurv; ++i) {
            float xa, ya, za, xb, yb, zb;
            blfHitPosition(blfHits, survHitIdxs[i - 1], xa, ya, za);
            blfHitPosition(blfHits, survHitIdxs[i], xb, yb, zb);
            const float dxs = xb - xa, dys = yb - ya;
            const float sep2Adj = dxs * dxs + dys * dys;
            // Never name a pixel hit: under option (a) the surplus is shed from the OT hits
            // only, and this drop is one of the two ways a hit is shed.
            if (sep2Adj < dropSep2 && !blfIsPixelHit(blfHits, survHitIdxs[i])) {
              dropSep2 = sep2Adj;
              dropIdx = i;
            }
          }
        }

        unsigned int fitHitIdxs[N];
        if (nSurv == N + 1 && dropIdx >= 0)
          dropFitHit<N>(survHitIdxs, dropIdx, fitHitIdxs);
        else
          selectFitHitsKeepingPixels<N>(blfHits, survHitIdxs, nSurv, fitHitIdxs);
#else
        // Collect both sensor hits per OT doublet layer slot (inner sensor first,
        // then outer sensor), skipping pixel layer slots and empty slots.
        //
        // The outer sensor is dropped when its global (x, y) is within kMaxDegenSepXY2 of
        // the inner one's. sTransverse depends on the hit only through (x, y), so such a
        // pair has a zero or sub-ulp first difference; matrixC_u divides by first
        // differences of sTransverse and invertNN then turns the resulting infinities into
        // a NaN pt, which LSTOutputConverter silently drops. A separation too small to
        // divide by but not exactly zero is as bad: it gives c_uMat entries of order 1e23
        // and a meaningless pt that passes the pt >= 0 gate, which is why the test is a
        // tolerance and not an equality. It is the two sensors of a Phase-2 endcap 2S
        // module that do this, once the track is straight enough for the same strip to fire
        // in both. The line fit is unaffected either way, since it uses sTotal, which still
        // separates the two hits through z.
        unsigned int survHitIdxs[Params_TC::kHitsPerLayer * (Params_TC::kLayers - Params_TC::kPixelLayerSlots)];
        int nSurv = 0;
        int nDegen = 0;
        // Position in survHitIdxs of the outer sensor of the least separated pair that was
        // NOT dropped as degenerate, and that separation.
        // The two sensors of a nearly coincident pair carry almost the same
        // transverse information, so removing one costs the least.
        int dropIdx = -1;
        float dropSep2 = std::numeric_limits<float>::max();
        for (int slot = Params_TC::kPixelLayerSlots; slot < Params_TC::kLayers; ++slot) {
          unsigned int h0 = hitSlots[slot][0];
          if (h0 == kTCEmptyHitIdx)
            continue;
          unsigned int h1 = hitSlots[slot][1];
          float x0 = hitsBase.xs()[h0], y0 = hitsBase.ys()[h0], z0 = hitsBase.zs()[h0];
          float x1 = hitsBase.xs()[h1], y1 = hitsBase.ys()[h1], z1 = hitsBase.zs()[h1];
          float d0sq = x0 * x0 + y0 * y0 + z0 * z0;
          float d1sq = x1 * x1 + y1 * y1 + z1 * z1;
          bool swap = d0sq > d1sq;
          survHitIdxs[nSurv++] = swap ? h1 : h0;
          float dx = x1 - x0, dy = y1 - y0;
          float sep2 = dx * dx + dy * dy;
          if (sep2 < kMaxDegenSepXY2) {
            ++nDegen;
            continue;
          }
          survHitIdxs[nSurv++] = swap ? h0 : h1;
          if (sep2 < dropSep2) {
            dropSep2 = sep2;
            dropIdx = nSurv - 1;
          }
        }

        // Round the surviving hit count down to the nearest instantiated hit count; the
        // instantiation it lands on claims the candidate, and selectFitHits sheds the
        // remaining hits. Without the degeneracy drop, nSurv is always even and equal to one
        // of those counts, so this is the identity and the candidate is fitted exactly.
        if (blfFitHits(nSurv) != N)
          continue;

        // Drop the extra hit. Two facts together make it at most one hit today:
        // no gap in kBLFitSizes is wider than 2, AND nSurv cannot exceed kBLFitMaxHits,
        // because Params_T5::kLayers = 7 caps the OT hit count at 14.
        // The argmin drop for a surplus of exactly one, the uniform stride otherwise.
        // The stride also covers the case where every pair was degenerate,
        // which leaves no spare hit to name and dropIdx at -1.
        unsigned int fitHitIdxs[N];
        if (nSurv == N + 1 && dropIdx >= 0)
          dropFitHit<N>(survHitIdxs, dropIdx, fitHitIdxs);
        else
          selectFitHits<N>(survHitIdxs, nSurv, fitHitIdxs);
#endif  // LST_BLF_PIXEL_HITS

        Eigen::Matrix<double, 3, N> hits;
        Eigen::Matrix<float, 6, N> hits_ge;
        for (int i = 0; i < N; ++i) {
          const unsigned int hIdx = fitHitIdxs[i];
#if LST_BLF_PIXEL_HITS
          float x, y, z;
          blfHitPosition(blfHits, hIdx, x, y, z);
          hits(0, i) = static_cast<double>(x);
          hits(1, i) = static_cast<double>(y);
          hits(2, i) = static_cast<double>(z);
          ArrayFx6 const& ge = blfHitError(blfHits, hIdx);
#else
          hits(0, i) = static_cast<double>(hitsBase.xs()[hIdx]);
          hits(1, i) = static_cast<double>(hitsBase.ys()[hIdx]);
          hits(2, i) = static_cast<double>(hitsBase.zs()[hIdx]);
          auto const& ge = hitsBase.ge()[hIdx];
#endif
          hits_ge(0, i) = ge[0];
          hits_ge(1, i) = ge[1];
          hits_ge(2, i) = ge[2];
          hits_ge(3, i) = ge[3];
          hits_ge(4, i) = ge[4];
          hits_ge(5, i) = ge[5];
        }

        ::riemannFit::Vector4d fast_fit;
        brokenline::fastFit(acc, hits, fast_fit);

        brokenline::PreparedBrokenLineData<N> data;
        brokenline::prepareBrokenLineData(acc, hits, fast_fit, bFieldD, data);

        brokenline::karimaki_circle_fit circle;
        ::riemannFit::LineFit line;
        brokenline::lineFit(acc, hits_ge, fast_fit, bFieldD, data, line);
        brokenline::circleFit(acc, hits, hits_ge, fast_fit, bFieldD, data, circle);

        fitResults.phi()[tcIdx] = static_cast<float>(circle.par(0));
        fitResults.tip()[tcIdx] = static_cast<float>(circle.par(1));
        fitResults.pt()[tcIdx] = static_cast<float>(bFieldD / alpaka::math::abs(acc, circle.par(2)));
        fitResults.eta()[tcIdx] = static_cast<float>(alpaka::math::asinh(acc, line.par(0)));
        fitResults.zip()[tcIdx] = static_cast<float>(line.par(1));
        fitResults.charge()[tcIdx] = static_cast<int8_t>(circle.qCharge);
        fitResults.chi2()[tcIdx] = static_cast<float>((circle.chi2 + line.chi2) / (2 * N - 5));
        fitResults.nFit()[tcIdx] = static_cast<uint8_t>(N);
        fitResults.nDegen()[tcIdx] = static_cast<uint8_t>(nDegen);

        // Circle covariance upper triangle: (phi-phi, phi-tip, tip-tip, phi-k, tip-k, k-k)
        auto& cCircle = fitResults.covCircle()[tcIdx];
        cCircle[0] = static_cast<float>(circle.cov(0, 0));
        cCircle[1] = static_cast<float>(circle.cov(0, 1));
        cCircle[2] = static_cast<float>(circle.cov(1, 1));
        cCircle[3] = static_cast<float>(circle.cov(0, 2));
        cCircle[4] = static_cast<float>(circle.cov(1, 2));
        cCircle[5] = static_cast<float>(circle.cov(2, 2));

        // Line covariance upper triangle: (slope-slope, slope-zip, zip-zip)
        auto& cLine = fitResults.covLine()[tcIdx];
        cLine[0] = static_cast<float>(line.cov(0, 0));
        cLine[1] = static_cast<float>(line.cov(0, 1));
        cLine[2] = static_cast<float>(line.cov(1, 1));
      }
    }
  };

  void launchLSTBrokenLineKernels(Queue& queue,
                                  float bField,
                                  TrackCandidatesBaseConst candsBase,
                                  HitsBaseConst hitsBase,
#if LST_BLF_PIXEL_HITS
                                  SeedHitsConst seedHits,
#endif
                                  TrackCandidatesBLFFit fitResults,
                                  unsigned int nTrackCandidates);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif  // RecoTracker_LSTCore_src_alpaka_BrokenLineFit_h
