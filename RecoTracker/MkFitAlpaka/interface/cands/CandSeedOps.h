#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandSeedOps_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandSeedOps_h

// Per-seed bookkeeping operations of the clone engine and of MkFinderV2p2, transliterated from MkFitCore (one thread
// per seed):
//   activateSeedCands          MkBuilder::find_tracks_unroll_candidates, per seed
//   mergeCandsAndBestShortOne  CombCandidate::mergeCandsAndBestShortOne
//   compactifyHitStorageForBestCand, beginBkwSearch, addPreBkwSearchHoles, repackCandPostBkwSearch (CombCandidate)
//   passesBkwSearchGate, restorePreBkwSearchCand (EventOfCombCandidates)
//   filterSeedCands            the per-seed body of MkBuilder::filter_comb_cands with the LST step filters
//                              phase1:qfilter_n_hits_pixseed && qfilter_nan_n_silly
// All work on a SeedCandsRef: pointers into one seed's rows (current candidate buffer, best short, HoT pool).

#include <cstdint>
#include <limits>

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
    HoTNode* hots;     // node n < hotHome at hots[n]
    HoTNode* extHots;  // node n >= hotHome at extHots[n - hotHome] (the seed's extension; nullptr: none)
    int hotHome;
    int hotCap;  // node indices must stay below hotCap
    int32_t* nHots;
    int16_t* lastHitIdxBeforeBkw;
    int16_t* nInsideMinusOneBeforeBkw;
    int16_t* nTailMinusOneBeforeBkw;
    uint32_t* overflowBits;

    ALPAKA_FN_HOST_ACC inline HoTNode& hot(int n) const { return hotNodeAt(hots, extHots, hotHome, n); }
  };

  // CombCandidate::addHit + TrackCand::addHitIdx into the HoT node chosen by the caller; false if it is past the pool
  ALPAKA_FN_HOST_ACC inline bool seedSetHitIdx(
      SeedCandsRef& seed, CandBook& cand, int node, int hitIdx, int layer, float chi2) {
    const bool fits = node < seed.hotCap;
    if (fits) {
      seed.hot(node) = HoTNode{hitIdx, layer, chi2, cand.lastHitIdx};
      cand.lastHitIdx = node;
    }
    cand.addHitCounters(hitIdx, chi2);
    return fits;
  }

  // CombCandidate::addHit + TrackCand::addHitIdx
  ALPAKA_FN_HOST_ACC inline void seedAddHitIdx(SeedCandsRef& seed, CandBook& cand, int hitIdx, int layer, float chi2) {
    const int node = *seed.nHots;
    if (seedSetHitIdx(seed, cand, node, hitIdx, layer, chi2))
      *seed.nHots = node + 1;
    else
      *seed.overflowBits |= kOverflowHotsBit;
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
          // Stop a low-pT candidate near its maximum radius, a looper: the
          // transverse angle between position and momentum exceeds
          // pi/2 - 0.2 rad (78.5 deg). dphi = |posPhi - momPhi| lies in
          // [0, 2pi), so the angle is past the limit when
          // kMaxAngPosMom < dphi < TwoPI - kMaxAngPosMom; the upper bound is
          // the wrap image of the lower.
          if (fwdSearch && kin[ic].pt < 1.2f) {
            constexpr float kMaxAngPosMom = Const::PIOver2 - 0.2f;
            const float dphiSigned = kin[ic].posPhi - kin[ic].momPhi;
            const float dphi = dphiSigned < 0.f ? -dphiSigned : dphiSigned;
            if (kin[ic].posRsq > 625.f && dphi > kMaxAngPosMom && dphi < Const::TwoPI - kMaxAngPosMom) {
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

  // CombCandidate::mergeCandsAndBestShortOne(params, score_func, update_score, sort_cands) with the scorer chosen by
  // name. pt[i] = pT() of candidate i, ptBest = pT() of the best short one.
  // The std::sort of <= maxCandsPerSeed candidates is libstdc++'s __insertion_sort (n <= 16), transliterated.
  ALPAKA_FN_HOST_ACC inline void mergeCandsAndBestShortOne(SeedCandsRef& seed,
                                                           int maxCandsPerSeed,
                                                           TrackScorer scorer,
                                                           bool updateScore,
                                                           bool sortCands,
                                                           const float* pt,
                                                           float ptBest) {
    const bool hasBest = *seed.bestShortValid != 0;
    int nCands = *seed.nCands;
    float ptl[kMaxCandsPerSeed];
    for (int i = 0; i < nCands; ++i)
      ptl[i] = pt[i];

    if (nCands > 0) {
      if (updateScore) {
        for (int i = 0; i < nCands; ++i)
          seed.cands[i].score = getScoreCand(scorer, seed.cands[i], ptl[i], seed.states[i].nSeedHits);
        if (hasBest)
          seed.bestShort->score = getScoreCand(scorer, *seed.bestShort, ptBest, seed.bestShortState->nSeedHits);
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

  // CombCandidate::addPreBkwSearchHoles
  ALPAKA_FN_HOST_ACC inline void addPreBkwSearchHoles(SeedCandsRef& seed) {
    // Called at the end of the backward search, before its final sort, so that the
    // final sort scores whole tracks. The hole counts were zeroed in beginBkwSearch()
    // so that the search's stopping cuts see only the holes the search makes.

    auto add = [&seed](CandBook& tc) {
      tc.nInsideMinusOne = *seed.nInsideMinusOneBeforeBkw + tc.nInsideMinusOne;
      tc.nTailMinusOne = *seed.nTailMinusOneBeforeBkw + tc.nTailMinusOne;
    };
    for (int ic = 0; ic < *seed.nCands; ++ic)
      add(seed.cands[ic]);
    if (*seed.bestShortValid)
      add(*seed.bestShort);
  }

  // Transverse helix of the state par = (x, y, z, 1/pT, phi, theta) as TrackBase::d0BeamSpot (Track.cc) computes it:
  // k = signed radius per GeV of pT, the center (x, y)
  struct HelixCenter {
    float k;
    float pt;
    float x;
    float y;
  };
  ALPAKA_FN_HOST_ACC inline HelixCenter helixCenter(const float* par, int charge) {
    const float k = ((charge < 0) ? 100.0f : -100.0f) / (Const::sol * Config::Bfield);
    const float pt = std::abs(1.f / par[3]);
    return {k, pt, par[0] - k * (pt * std::sin(par[4])), par[1] + k * (pt * std::cos(par[4]))};
  }

  // TrackBase::d0BeamSpot(x_bs, y_bs, linearize = false)
  ALPAKA_FN_HOST_ACC inline float d0BeamSpot(const float* par, int charge, float xBs, float yBs) {
    const HelixCenter center = helixCenter(par, charge);
    const float absOocHalf = std::abs(center.k * center.pt);
    return hipo(center.x - xBs, center.y - yBs) - absOocHalf;
  }

  // ... of a candidate state
  ALPAKA_FN_HOST_ACC inline float d0BeamSpot(const CandState& st, float xBs, float yBs) {
    return d0BeamSpot(st.par, st.charge, xBs, yBs);
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
    // The hole counts were restored by addPreBkwSearchHoles().
  }

  // TrackBase::hasNanNSillyValues = TrackState::hasNanNSillyValues (TrackState.cc): a diagonal error < 0 or any element
  // not finite.
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

  // TrackerInfo is_pixel() of layer from a pixel-layer bit mask (bit l of word l / 64, ESConfig::pixel_layer_mask)
  ALPAKA_FN_HOST_ACC inline bool isPixelLayer(const uint64_t* mask, int layer) {
    constexpr int kBits = std::numeric_limits<uint64_t>::digits;
    return (mask[layer / kBits] >> (layer % kBits)) & 1u;
  }

  // The backward-search gate of one seed (EventOfCombCandidates m_pre_bkw_cands[i], m_bkw_min_pixel_layers[i]):
  // pixelLayers = bit mask of the pixel layers (TrackerInfo is_pixel).
  struct SeedBkwSearchGate {
    const CandBook* preBkwBook;
    const CandState* preBkwState;
    int minPixelLayers;
    const uint64_t* pixelLayers;
  };

  // EventOfCombCandidates::passesBkwSearchGate
  ALPAKA_FN_HOST_ACC inline bool passesBkwSearchGate(SeedCandsRef& seed,
                                                     const CandBook& tc,
                                                     const SeedBkwSearchGate& gate) {
    // Distinct pixel layers with found hits that the backward search added to tc. After
    // compactifyHitStorageForBestCand() the pre-search hits are nodes 0 .. lastHitIdxBeforeBkwSearch(),
    // and the search only appends, so this holds before and after repackCandPostBkwSearch().
    uint64_t layers = 0;
    for (int idx = tc.lastHitIdx; idx >= 0; idx = seed.hot(idx).prev) {
      const HoTNode& hot = seed.hot(idx);
      if (idx > *seed.lastHitIdxBeforeBkw && hot.index >= 0 && hot.layer < std::numeric_limits<uint64_t>::digits &&
          isPixelLayer(gate.pixelLayers, hot.layer))
        layers |= uint64_t(1) << hot.layer;
    }
    int nLayers = 0;  // std::popcount
    for (; layers; layers &= layers - 1)
      ++nLayers;
    return nLayers >= gate.minPixelLayers;
  }

  // EventOfCombCandidates::restorePreBkwSearchCand
  ALPAKA_FN_HOST_ACC inline void restorePreBkwSearchCand(SeedCandsRef& seed, const SeedBkwSearchGate& gate) {
    // repackCandPostBkwSearch() linked node 0, the innermost pre-search hit, to the hits the
    // search added. The other pre-search nodes are as they were.
    seed.hot(0).prev = -1;
    seed.cands[0] = *gate.preBkwBook;
    seed.states[0] = *gate.preBkwState;
    *seed.nCands = 1;
  }

  // Per-seed body of MkBuilder::filter_comb_cands: repack (backward representation) cand 0, apply the backward-search
  // gate to it (bkwGate != nullptr), filter it, then, if it fails and attemptAllCands, repack/filter cands 1.. and
  // copy the first passing one into slot 0 (MkFitCore order of the in-place repacks kept). Returns whether the seed
  // survives; the caller drops failing seeds keeping the order.
  ALPAKA_FN_HOST_ACC inline bool filterSeedCands(SeedCandsRef& seed,
                                                 bool bkwRep,
                                                 bool attemptAllCands,
                                                 int minHitsQF,
                                                 const SeedBkwSearchGate* bkwGate = nullptr) {
    if (bkwRep)
      repackCandPostBkwSearch(seed, 0);
    if (bkwGate && !passesBkwSearchGate(seed, seed.cands[0], *bkwGate))
      restorePreBkwSearchCand(seed, *bkwGate);  // leaves one candidate
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
