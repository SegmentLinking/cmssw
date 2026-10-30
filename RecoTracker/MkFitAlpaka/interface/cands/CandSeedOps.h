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
  ALPAKA_FN_HOST_ACC inline void seedAddHitIdx(SeedCandsRef& seed, CandBook& cand, int hitIdx, int layer, float chi2) {
    const int node = *seed.nHots;
    if (node < seed.hotCap) {
      seed.hot(node) = HoTNode{hitIdx, layer, chi2, cand.lastHitIdx};
      *seed.nHots = node + 1;
      cand.lastHitIdx = node;
    } else {
      *seed.overflowBits |= kOverflowHotsBit;
    }
    cand.addHitCounters(hitIdx, chi2);
  }

  // Kinematics MkFitCore reads from the candidate state in the unroll step (TrackCand::pT(), posRsq(), posPhi() =
  // vdt::fast_atan2f(y, x), momPhi() = par[4]); computed by the caller with the portable math of interface/math.
  // MkFitCore is built for x86-64-v3 and GCC contracts posRsq = x*x + y*y into fma(x, x, y*y): the explicit fma form
  // reproduces it bitwise on every backend.
  struct CandKin {
    float pt;
    float posRsq;
    float posPhi;
    float momPhi;
  };

  // MkBuilder::find_tracks_unroll_candidates for one seed. activeIc gets the listed candidates in MkFitCore order.
  ALPAKA_FN_HOST_ACC inline void activateSeedCands(SeedCandsRef& seed,
                                                   const CandKin* kin,
                                                   int layer,
                                                   int prevLayer,
                                                   bool pickupOnly,
                                                   bool fwdSearch,
                                                   float minPtCut,
                                                   int32_t* activeIc,
                                                   int32_t* nActive) {
    *nActive = 0;
    if (*seed.state == kDormant && *seed.pickupLayer == prevLayer) {
      *seed.state = kFinding;
    }
    if (!pickupOnly && *seed.state == kFinding) {
      bool active = false;
      for (int ic = 0; ic < *seed.nCands; ++ic) {
        CandBook& tc = seed.cands[ic];
        if (seed.hot(tc.lastHitIdx).index != kHitStopIdx) {
          // Stop candidates with pT<X GeV
          if (kin[ic].pt < minPtCut) {
            seedAddHitIdx(seed, tc, kHitStopIdx, layer, 0.0f);
            continue;
          }
          // Check if the candidate is close to it's max_r, pi/2 - 0.2 rad (11.5 deg)
          if (fwdSearch && kin[ic].pt < 1.2f) {
            const float dphiSigned = kin[ic].posPhi - kin[ic].momPhi;
            const float dphi = dphiSigned < 0.f ? -dphiSigned : dphiSigned;
            if (kin[ic].posRsq > 625.f && dphi > 1.371f && dphi < 4.512f) {
              seedAddHitIdx(seed, tc, kHitStopIdx, layer, 0.0f);
              continue;
            }
          }
          active = true;
          activeIc[(*nActive)++] = ic;
          tc.overlaps.reset();
        }
      }
      if (!active) {
        *seed.state = kFinished;
      }
    }
  }

  // CombCandidate::mergeCandsAndBestShortOne. pt[i] = pT() of candidate i, ptBest = pT() of the best short one.
  // The std::sort of <= maxCandsPerSeed candidates is libstdc++'s __insertion_sort (n <= 16), transliterated.
  ALPAKA_FN_HOST_ACC inline void mergeCandsAndBestShortOne(
      SeedCandsRef& seed, int maxCandsPerSeed, bool updateScore, bool sortCands, const float* pt, float ptBest) {
    const bool hasBest = *seed.bestShortValid != 0;
    int nCands = *seed.nCands;
    float ptl[kMaxCandsPerSeed];
    for (int i = 0; i < nCands; ++i)
      ptl[i] = pt[i];

    if (nCands > 0) {
      if (updateScore) {
        for (int i = 0; i < nCands; ++i)
          seed.cands[i].score = getScoreCand(seed.cands[i], ptl[i]);
        if (hasBest)
          seed.bestShort->score = getScoreCand(*seed.bestShort, ptBest);
      }
      if (sortCands) {
        // std::__insertion_sort with comp = sortByScoreTrackCand (a.score > b.score)
        for (int i = 1; i < nCands; ++i) {
          const CandBook vb = seed.cands[i];
          const CandState vs = seed.states[i];
          const float vp = ptl[i];
          int j = i;
          if (vb.score > seed.cands[0].score) {
            for (; j > 0; --j) {
              seed.cands[j] = seed.cands[j - 1];
              seed.states[j] = seed.states[j - 1];
              ptl[j] = ptl[j - 1];
            }
          } else {
            while (j > 0 && vb.score > seed.cands[j - 1].score) {
              seed.cands[j] = seed.cands[j - 1];
              seed.states[j] = seed.states[j - 1];
              ptl[j] = ptl[j - 1];
              --j;
            }
          }
          seed.cands[j] = vb;
          seed.states[j] = vs;
          ptl[j] = vp;
        }
      }
      if (hasBest && seed.bestShort->score > seed.cands[nCands - 1].score) {
        int ci = 0;
        while (seed.cands[ci].score > seed.bestShort->score)
          ++ci;
        if (nCands >= maxCandsPerSeed)
          --nCands;  // pop_back
        for (int k = nCands; k > ci; --k) {
          seed.cands[k] = seed.cands[k - 1];
          seed.states[k] = seed.states[k - 1];
        }
        seed.cands[ci] = *seed.bestShort;
        seed.states[ci] = *seed.bestShortState;
        ++nCands;
      }
    } else if (hasBest) {
      seed.cands[0] = *seed.bestShort;
      seed.states[0] = *seed.bestShortState;
      nCands = 1;
    }
    if (hasBest) {  // TrackCand::resetShortTrack()
      seed.bestShort->score = scoreWorstPossible();
      *seed.bestShortValid = 0;
    }
    *seed.nCands = nCands;
  }

  // CombCandidate::compactifyHitStorageForBestCand with remove_seed_hits = false (in place in the seed's pool).
  ALPAKA_FN_HOST_ACC inline void compactifyHitStorageForBestCand(SeedCandsRef& seed) {
    *seed.nCands = 1;
    CandBook& tc = seed.cands[0];
    const int stashEnd = *seed.nHots;
    int stashPos = stashEnd;
    int idx = tc.lastHitIdx;
    while (idx != -1) {
      seed.hot(--stashPos) = seed.hot(idx);
      idx = seed.hot(idx).prev;
    }
    int pos = 0;
    while (stashPos != stashEnd) {
      seed.hot(pos).index = seed.hot(stashPos).index;
      seed.hot(pos).layer = seed.hot(stashPos).layer;
      seed.hot(pos).chi2 = seed.hot(stashPos).chi2;
      seed.hot(pos).prev = pos - 1;
      ++pos;
      ++stashPos;
    }
    *seed.nHots = pos;
    tc.lastHitIdx = pos - 1;
  }

  // CombCandidate::beginBkwSearch
  ALPAKA_FN_HOST_ACC inline void beginBkwSearch(SeedCandsRef& seed) {
    CandBook& tc = seed.cands[0];
    *seed.state = kDormant;
    *seed.pickupLayer = seed.hot(0).layer;
    *seed.lastHitIdxBeforeBkw = tc.lastHitIdx;
    *seed.nInsideMinusOneBeforeBkw = tc.nInsideMinusOne;
    *seed.nTailMinusOneBeforeBkw = tc.nTailMinusOne;
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
  ALPAKA_FN_HOST_ACC inline int bkwSearchPixelLayers(SeedCandsRef& seed, const uint64_t* pixelLayers) {
    uint64_t layers = 0;
    for (int idx = seed.cands[0].lastHitIdx; idx > 0; idx = seed.hot(idx).prev) {
      const HoTNode& node = seed.hot(idx);
      if (node.index >= 0 && node.layer < 64 && ((pixelLayers[node.layer >> 6] >> (node.layer & 63)) & 1u))
        layers |= uint64_t(1) << node.layer;
    }
    int nLayers = 0;
    for (; layers; layers &= layers - 1)
      ++nLayers;
    return nLayers;
  }

  // CombCandidate::repackCandPostBkwSearch
  ALPAKA_FN_HOST_ACC inline void repackCandPostBkwSearch(SeedCandsRef& seed, int i) {
    CandBook& tc = seed.cands[i];
    int currIdx = tc.lastHitIdx;
    if (currIdx != 0) {
      int lastIdx = -1, prevIdx;
      do {
        prevIdx = seed.hot(currIdx).prev;
        seed.hot(currIdx).prev = lastIdx;
        lastIdx = currIdx;
        currIdx = prevIdx;
      } while (prevIdx != -1);
    }
    tc.lastHitIdx = *seed.lastHitIdxBeforeBkw;
    tc.nInsideMinusOne = *seed.nInsideMinusOneBeforeBkw + tc.nInsideMinusOne;
    tc.nTailMinusOne = *seed.nTailMinusOneBeforeBkw + tc.nTailMinusOne;
  }

  // TrackBase::hasNanNSillyValues (Track.cc:177): a diagonal error < 0 or any element not finite.
  // err is the 21-element lower triangle in SMatrixSym66 order, element (i, j <= i) at i * (i + 1) / 2 + j.
  ALPAKA_FN_HOST_ACC inline bool hasNanNSillyValues(const float* err) {
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j <= i; ++j) {
        const float elem = err[i * (i + 1) / 2 + j];
        if ((i == j && elem < 0) || !isFinite(elem))
          return true;
      }
    }
    return false;
  }

  ALPAKA_FN_HOST_ACC inline bool hasNanNSillyValues(const CandState& state) { return hasNanNSillyValues(state.err); }

  // phase1:qfilter_n_hits_pixseed && qfilter_nan_n_silly (the LST step's pre and post backward-fit filters)
  ALPAKA_FN_HOST_ACC inline bool passLstStepFilter(const CandBook& cand, const CandState& state, int minHitsQF) {
    return cand.nFound >= minHitsQF && !hasNanNSillyValues(state);
  }

  // Per-seed body of MkBuilder::filter_comb_cands: repack (backward representation) and filter cand 0, then, if it
  // fails and attemptAllCands, repack/filter cands 1.. and copy the first passing one into slot 0 (MkFitCore order of
  // the in-place repacks kept). Returns whether the seed survives; the caller drops failing seeds keeping the order.
  ALPAKA_FN_HOST_ACC inline bool filterSeedCands(SeedCandsRef& seed, bool bkwRep, bool attemptAllCands, int minHitsQF) {
    if (bkwRep)
      repackCandPostBkwSearch(seed, 0);
    bool passed = passLstStepFilter(seed.cands[0], seed.states[0], minHitsQF);
    if (!passed && attemptAllCands) {
      for (int j = 1; j < *seed.nCands; ++j) {
        if (bkwRep)
          repackCandPostBkwSearch(seed, j);
        if (passLstStepFilter(seed.cands[j], seed.states[j], minHitsQF)) {
          seed.cands[0] = seed.cands[j];
          seed.states[0] = seed.states[j];
          passed = true;
          break;
        }
      }
    }
    return passed;
  }

}  // namespace mkfitdev

#endif
