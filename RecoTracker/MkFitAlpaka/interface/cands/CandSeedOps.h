#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandSeedOps_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandSeedOps_h

// Per-seed bookkeeping operations of the clone engine, transliterated from MkFitCore (one thread per seed):
//   activateSeedCands          MkBuilder::find_tracks_unroll_candidates, per seed (MkBuilder.cc:641-704)
//   mergeCandsAndBestShortOne  CombCandidate::mergeCandsAndBestShortOne (TrackStructures.cc:72-117)
//   compactifyHitStorageForBestCand, beginBkwSearch, repackCandPostBkwSearch (TrackStructures.cc:149-254)
//   filterSeedCands            the per-seed body of MkBuilder::filter_comb_cands (MkBuilder.cc:314-346) with the LST
//                              step filters phase1:qfilter_n_hits_pixseed && qfilter_nan_n_silly
// All work on a SeedCandsRef: pointers into one seed's rows (current candidate buffer, best short, HoT pool).

#include <cstdint>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"

namespace mkfitdev {

  struct SeedCandsRef {
    int8_t* state;
    int16_t* pickupLayer;
    CandBook* cands;    // current buffer, kMaxCandsPerSeed slots
    CandState* states;  // states of the current buffer (may be nullptr for ops that do not move states)
    int32_t* nCands;
    CandBook* bestShort;
    CandState* bestShortState;
    int8_t* bestShortValid;
    HoTNode* hots;  // node n at hots[n - hotOffset]
    int hotOffset;
    int hotCap;
    int32_t* nHots;
    int16_t* lastHitIdxBeforeBkw;
    int16_t* nInsideMinusOneBeforeBkw;
    int16_t* nTailMinusOneBeforeBkw;
    uint32_t* overflowBits;

    ALPAKA_FN_HOST_ACC inline HoTNode& hot(int n) const { return hots[n - hotOffset]; }
  };

  // CombCandidate::addHit + TrackCand::addHitIdx
  ALPAKA_FN_HOST_ACC inline void seedAddHitIdx(SeedCandsRef& r, CandBook& c, int hitIdx, int layer, float chi2) {
    const int n = *r.nHots;
    if (n < r.hotCap) {
      r.hot(n) = HoTNode{hitIdx, layer, chi2, c.lastHitIdx};
      *r.nHots = n + 1;
      c.lastHitIdx = n;
    } else {
      *r.overflowBits |= kOverflowHotsBit;
    }
    c.addHitCounters(hitIdx, chi2);
  }

  // Kinematics MkFitCore reads from the candidate state in the unroll step (TrackCand::pT(), posRsq(), posPhi() =
  // vdt::fast_atan2f(y, x), momPhi() = par[4]); computed by the caller with the portable math of interface/math.
  // MkFitCore is built for x86-64-v3 and GCC contracts posRsq = x*x + y*y into fma(x, x, y*y): measured on 191,605
  // dumped cands, fma(x, x, y*y) matches MkFitCore bitwise in all, plain x*x + y*y differs in 16%. mplex's getPhi
  // compiled for x86-64-v3 matches posPhi bitwise in all. Use the explicit fma form on every backend.
  struct CandKin {
    float pt;
    float posRsq;
    float posPhi;
    float momPhi;
  };

  // MkBuilder::find_tracks_unroll_candidates for one seed. activeIc gets the listed candidates in MkFitCore order.
  ALPAKA_FN_HOST_ACC inline void activateSeedCands(SeedCandsRef& r,
                                                   const CandKin* kin,
                                                   int layer,
                                                   int prevLayer,
                                                   bool pickupOnly,
                                                   bool fwdSearch,
                                                   float minPtCut,
                                                   int32_t* activeIc,
                                                   int32_t* nActive) {
    *nActive = 0;
    if (*r.state == kDormant && *r.pickupLayer == prevLayer) {
      *r.state = kFinding;
    }
    if (!pickupOnly && *r.state == kFinding) {
      bool active = false;
      for (int ic = 0; ic < *r.nCands; ++ic) {
        CandBook& c = r.cands[ic];
        if (r.hot(c.lastHitIdx).index != kHitStopIdx) {
          // Stop candidates with pT<X GeV
          if (kin[ic].pt < minPtCut) {
            seedAddHitIdx(r, c, kHitStopIdx, layer, 0.0f);
            continue;
          }
          // Check if the candidate is close to it's max_r, pi/2 - 0.2 rad (11.5 deg)
          if (fwdSearch && kin[ic].pt < 1.2f) {
            const float d = kin[ic].posPhi - kin[ic].momPhi;
            const float dphi = d < 0.f ? -d : d;
            if (kin[ic].posRsq > 625.f && dphi > 1.371f && dphi < 4.512f) {
              seedAddHitIdx(r, c, kHitStopIdx, layer, 0.0f);
              continue;
            }
          }
          active = true;
          activeIc[(*nActive)++] = ic;
          c.overlaps.reset();
        }
      }
      if (!active) {
        *r.state = kFinished;
      }
    }
  }

  // CombCandidate::mergeCandsAndBestShortOne. pt[i] = pT() of candidate i, ptBest = pT() of the best short one.
  // The std::sort of <= maxCandsPerSeed candidates is libstdc++'s __insertion_sort (n <= 16), transliterated.
  ALPAKA_FN_HOST_ACC inline void mergeCandsAndBestShortOne(
      SeedCandsRef& r, int maxCandsPerSeed, bool updateScore, bool sortCands, const float* pt, float ptBest) {
    const bool hasBest = *r.bestShortValid != 0;
    int n = *r.nCands;
    float ptl[kMaxCandsPerSeed];
    for (int i = 0; i < n; ++i)
      ptl[i] = pt[i];

    if (n > 0) {
      if (updateScore) {
        for (int i = 0; i < n; ++i)
          r.cands[i].score = getScoreCand(r.cands[i], ptl[i]);
        if (hasBest)
          r.bestShort->score = getScoreCand(*r.bestShort, ptBest);
      }
      if (sortCands) {
        // std::__insertion_sort with comp = sortByScoreTrackCand (a.score > b.score)
        for (int i = 1; i < n; ++i) {
          const CandBook vb = r.cands[i];
          const CandState vs = r.states[i];
          const float vp = ptl[i];
          int j = i;
          if (vb.score > r.cands[0].score) {
            for (; j > 0; --j) {
              r.cands[j] = r.cands[j - 1];
              r.states[j] = r.states[j - 1];
              ptl[j] = ptl[j - 1];
            }
          } else {
            while (j > 0 && vb.score > r.cands[j - 1].score) {
              r.cands[j] = r.cands[j - 1];
              r.states[j] = r.states[j - 1];
              ptl[j] = ptl[j - 1];
              --j;
            }
          }
          r.cands[j] = vb;
          r.states[j] = vs;
          ptl[j] = vp;
        }
      }
      if (hasBest && r.bestShort->score > r.cands[n - 1].score) {
        int ci = 0;
        while (r.cands[ci].score > r.bestShort->score)
          ++ci;
        if (n >= maxCandsPerSeed)
          --n;  // pop_back
        for (int k = n; k > ci; --k) {
          r.cands[k] = r.cands[k - 1];
          r.states[k] = r.states[k - 1];
        }
        r.cands[ci] = *r.bestShort;
        r.states[ci] = *r.bestShortState;
        ++n;
      }
    } else if (hasBest) {
      r.cands[0] = *r.bestShort;
      r.states[0] = *r.bestShortState;
      n = 1;
    }
    if (hasBest) {  // TrackCand::resetShortTrack()
      r.bestShort->score = scoreWorstPossible();
      *r.bestShortValid = 0;
    }
    *r.nCands = n;
  }

  // CombCandidate::compactifyHitStorageForBestCand (in place in the seed's pool, as MkFitCore).
  ALPAKA_FN_HOST_ACC inline void compactifyHitStorageForBestCand(SeedCandsRef& r,
                                                                 bool removeSeedHits,
                                                                 int backwardFitMinHits) {
    *r.nCands = 1;
    CandBook& tc = r.cands[0];
    if (removeSeedHits && tc.nFound <= backwardFitMinHits) {
      removeSeedHits = false;
    }
    const int stashEnd = *r.nHots;
    int stashPos = stashEnd;
    int idx = tc.lastHitIdx;

    if (removeSeedHits) {
      const int nSeedHits = r.states[0].nSeedHits;
      const int a = tc.nFound - nSeedHits;
      int nToPick = a > backwardFitMinHits ? a : backwardFitMinHits;  // std::max
      while (nToPick > 0) {
        r.hot(--stashPos) = r.hot(idx);
        if (r.hot(idx).index >= 0)
          --nToPick;
        idx = r.hot(idx).prev;
      }
      *r.nHots = 0;
      tc.lastHitIdx = -1;
      tc.nFound = 0;
      tc.nMissing = 0;
      tc.nInsideMinusOne = 0;
      tc.nTailMinusOne = 0;
      while (stashPos != stashEnd && r.hot(stashPos).index < 0)
        ++stashPos;
      while (stashPos != stashEnd) {
        const HoTNode hn = r.hot(stashPos);
        seedAddHitIdx(r, tc, hn.index, hn.layer, hn.chi2);
        ++stashPos;
      }
    } else {
      while (idx != -1) {
        r.hot(--stashPos) = r.hot(idx);
        idx = r.hot(idx).prev;
      }
      int pos = 0;
      while (stashPos != stashEnd) {
        r.hot(pos).index = r.hot(stashPos).index;
        r.hot(pos).layer = r.hot(stashPos).layer;
        r.hot(pos).chi2 = r.hot(stashPos).chi2;
        r.hot(pos).prev = pos - 1;
        ++pos;
        ++stashPos;
      }
      *r.nHots = pos;
      tc.lastHitIdx = pos - 1;
    }
  }

  // CombCandidate::beginBkwSearch
  ALPAKA_FN_HOST_ACC inline void beginBkwSearch(SeedCandsRef& r) {
    CandBook& tc = r.cands[0];
    *r.state = kDormant;
    *r.pickupLayer = r.hot(0).layer;
    *r.lastHitIdxBeforeBkw = tc.lastHitIdx;
    *r.nInsideMinusOneBeforeBkw = tc.nInsideMinusOne;
    *r.nTailMinusOneBeforeBkw = tc.nTailMinusOne;
    tc.lastHitIdx = 0;
    tc.nInsideMinusOne = 0;
    tc.nTailMinusOne = 0;
  }

  // TrackBase::d0BeamSpot(x_bs, y_bs, linearize = false) (Track.cc) of a candidate state
  ALPAKA_FN_HOST_ACC inline float d0BeamSpot(const CandState& st, float xBs, float yBs) {
    const float k = ((st.charge < 0) ? 100.0f : -100.0f) / (Const::sol * Config::Bfield);
    const float pt = std::abs(1.f / st.par[3]);
    const float absOocHalf = std::abs(k * pt);
    const float xCenter = st.par[0] - k * (pt * std::sin(st.par[4]));
    const float yCenter = st.par[1] + k * (pt * std::cos(st.par[4]));
    return hipo(xCenter - xBs, yCenter - yBs) - absOocHalf;
  }

  // EventOfCombCandidates::gateBkwSearch helper: distinct pixel layers (layer < 64, as MkFitCore) with found
  // hits that the backward search added to candidate 0 (nodes after node 0, the pickup hit).
  ALPAKA_FN_HOST_ACC inline int bkwSearchPixelLayers(SeedCandsRef& r, const uint64_t* pixelLayers) {
    uint64_t layers = 0;
    for (int idx = r.cands[0].lastHitIdx; idx > 0; idx = r.hot(idx).prev) {
      const HoTNode& h = r.hot(idx);
      if (h.index >= 0 && h.layer < 64 && ((pixelLayers[h.layer >> 6] >> (h.layer & 63)) & 1u))
        layers |= uint64_t(1) << h.layer;
    }
    int n = 0;
    for (; layers; layers &= layers - 1)
      ++n;
    return n;
  }

  // CombCandidate::repackCandPostBkwSearch
  ALPAKA_FN_HOST_ACC inline void repackCandPostBkwSearch(SeedCandsRef& r, int i) {
    CandBook& tc = r.cands[i];
    int currIdx = tc.lastHitIdx;
    if (currIdx != 0) {
      int lastIdx = -1, prevIdx;
      do {
        prevIdx = r.hot(currIdx).prev;
        r.hot(currIdx).prev = lastIdx;
        lastIdx = currIdx;
        currIdx = prevIdx;
      } while (prevIdx != -1);
    }
    tc.lastHitIdx = *r.lastHitIdxBeforeBkw;
    tc.nInsideMinusOne = *r.nInsideMinusOneBeforeBkw + tc.nInsideMinusOne;
    tc.nTailMinusOne = *r.nTailMinusOneBeforeBkw + tc.nTailMinusOne;
  }

  // TrackBase::hasNanNSillyValues (Track.cc:177): a diagonal error < 0 or any element not finite.
  // err is the 21-element lower triangle in SMatrixSym66 order, element (i, j <= i) at i * (i + 1) / 2 + j.
  ALPAKA_FN_HOST_ACC inline bool hasNanNSillyValues(const CandState& s) {
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j <= i; ++j) {
        const float e = s.err[i * (i + 1) / 2 + j];
        if ((i == j && e < 0) || !isFinite(e))
          return true;
      }
    }
    return false;
  }

  // phase1:qfilter_n_hits_pixseed && qfilter_nan_n_silly (the LST step's pre and post backward-fit filters)
  ALPAKA_FN_HOST_ACC inline bool passLstStepFilter(const CandBook& c, const CandState& s, int minHitsQF) {
    return c.nFound >= minHitsQF && !hasNanNSillyValues(s);
  }

  // Per-seed body of MkBuilder::filter_comb_cands: repack (backward representation) and filter cand 0, then, if it
  // fails and attemptAllCands, repack/filter cands 1.. and copy the first passing one into slot 0 (MkFitCore order of
  // the in-place repacks kept). Returns whether the seed survives; the caller drops failing seeds keeping the order.
  ALPAKA_FN_HOST_ACC inline bool filterSeedCands(SeedCandsRef& r, bool bkwRep, bool attemptAllCands, int minHitsQF) {
    if (bkwRep)
      repackCandPostBkwSearch(r, 0);
    bool passed = passLstStepFilter(r.cands[0], r.states[0], minHitsQF);
    if (!passed && attemptAllCands) {
      for (int j = 1; j < *r.nCands; ++j) {
        if (bkwRep)
          repackCandPostBkwSearch(r, j);
        if (passLstStepFilter(r.cands[j], r.states[j], minHitsQF)) {
          r.cands[0] = r.cands[j];
          r.states[0] = r.states[j];
          passed = true;
          break;
        }
      }
    }
    return passed;
  }

}  // namespace mkfitdev

#endif
