#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandSelection_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandSelection_h

// Device equivalent of CandCloner::processSeedRange (CandCloner.cc:127-252) for ONE seed.
//
// MkFitCore std::sort's the seed's option list by score (descending) and walks it: "stop" options (hit -2) may
// replace the best-short candidate, the other options become the new candidates, held-back ("extra")
// candidates that score strictly better than the option at hand are squeezed in first, and the walk stops
// once maxCandsPerSeed candidates are pushed. Only the first maxCandsPerSeed non-stop options and the
// best stop option can influence the result, so here they are found with a fixed-size top-k insertion
// (score descending, input order on equal score) and the MkFitCore walk is replayed on that short sequence.
//
// Ties: equal scores keep the option input order (stable). That is exactly MkFitCore's order for n <= 16 options
// (libstdc++ std::sort is an insertion sort there). For n > 16 MkFitCore's order of EQUAL scores is introsort's; it is
// not reproduced (doc/SIMPLIFICATIONS.txt): measured 1 exact score tie among a seed's options in 426k
// records, none with n > 16.
//
// HoT nodes: MkFitCore appends a node for every option it looks at (also skipped stop options); here a node is
// appended only for the pushed candidates, their overlap hits and an accepted best-short candidate. Hit
// chains (hit, layer, chi2 sequence) are identical, node indices are not.

#include <cstdint>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"

namespace mkfitdev {

  struct SeedSelParams {
    int layer;            // the kernel takes it per seed from SeedCandsSoA::layer
    int maxCandsPerSeed;  // <= kMaxCandsPerSeed
    float pTCutOverlap;
    bool recheckOverlap;
  };

  static_assert(kMaxOptsPerSeed <= 64, "SeedSelIO::liveOpts has one bit per option slot");

  // Per-seed I/O, plain pointers into the seed's rows (see CandsSoA.h for the row conventions).
  struct SeedSelIO {
    // inputs
    const CandBook* candsIn;  // [nCandsIn] current candidates (parents)
    const float* candsInPt;   // [nCandsIn] TrackCand::pT() of each parent = |1/par[3]|
    int nCandsIn;
    const CandExtra* extras;  // [nExtras] held-back candidates, MkFitCore order (score descending)
    int nExtras;
    const CandOption* opts;  // [nOptSlots] option slots in MkFitCore input order; empty: hitIdx == kOptEmpty
    int nOptSlots;
    uint64_t liveOpts;  // bit j: slot j was written for this step (the other slots are not read)
    int state;          // SeedState before the selection
    // in/out
    CandBook* bestShort;
    int8_t* bestShortValid;
    int32_t* bestShortSrc;  // set to the parent index when the best-short candidate is replaced
    HoTNode* hots;          // node n lives at hots[n - hotOffset]
    int hotOffset;
    int hotCap;  // node indices must stay below hotCap
    int32_t* nHots;
    // outputs
    CandBook* candsOut;    // [maxCandsPerSeed]
    int32_t* candsOutSrc;  // state source of each output candidate (index into the parents)
    int32_t* nCandsOut;
    CandUpdate* upd;  // [maxCandsPerSeed] kalman update list of this seed
    int32_t* nUpd;
    CandUpdate* ovl;  // [maxCandsPerSeed] overlap re-check list (recheckOverlap only)
    int32_t* nOvl;
    uint32_t* overflowBits;
  };

  // CombCandidate::addHit + TrackCand::addHitIdx
  ALPAKA_FN_HOST_ACC inline void selAddHitIdx(SeedSelIO& io, CandBook& c, int hitIdx, int layer, float chi2) {
    const int n = *io.nHots;
    if (n < io.hotCap) {
      io.hots[n - io.hotOffset] = HoTNode{hitIdx, layer, chi2, c.lastHitIdx};
      *io.nHots = n + 1;
      c.lastHitIdx = n;
    } else {
      *io.overflowBits |= kOverflowHotsBit;  // seed must be treated as failed by the caller
    }
    c.addHitCounters(hitIdx, chi2);
  }

  // Pushes the candidate made from option h (CandCloner.cc:180-210 after the break checks).
  ALPAKA_FN_HOST_ACC inline void selPushOption(const SeedSelParams& p,
                                               SeedSelIO& io,
                                               const CandOption& h,
                                               int& nPushed) {
    CandBook tc = io.candsIn[h.trkIdx];
    selAddHitIdx(io, tc, h.hitIdx, p.layer, h.chi2_hit);
    tc.score = h.score;
    if (h.hitIdx >= 0) {
      const HitMatchPair& parentOv = io.candsIn[h.trkIdx].overlaps;
      int om = -1;
      if (io.candsInPt[h.trkIdx] > p.pTCutOverlap && (om = parentOv.findOverlap(h.hitIdx, (int)h.module)) >= 0) {
        const int ovHit = parentOv.M[om].hit;
        if (p.recheckOverlap) {
          io.ovl[(*io.nOvl)++] = CandUpdate{nPushed, h.hitIdx, ovHit};
        } else {
          selAddHitIdx(io, tc, ovHit, p.layer, 0.f);
          ++tc.nOverlap;
          io.upd[(*io.nUpd)++] = CandUpdate{nPushed, h.hitIdx, -1};
        }
      } else {
        io.upd[(*io.nUpd)++] = CandUpdate{nPushed, h.hitIdx, -1};
      }
    }
    io.candsOut[nPushed] = tc;
    io.candsOutSrc[nPushed] = h.trkIdx;
    ++nPushed;
  }

  // A stop option (hit -2) as MkFitCore handles it: replaces the best-short candidate if strictly better.
  ALPAKA_FN_HOST_ACC inline void selConsiderStop(const SeedSelParams& p, SeedSelIO& io, const CandOption& h) {
    if (h.score > io.bestShort->score) {
      CandBook tc = io.candsIn[h.trkIdx];
      selAddHitIdx(io, tc, h.hitIdx, p.layer, h.chi2_hit);
      tc.score = h.score;
      *io.bestShort = tc;
      *io.bestShortValid = 1;
      *io.bestShortSrc = h.trkIdx;
    }
  }

  ALPAKA_FN_HOST_ACC inline void selPushExtra(SeedSelIO& io, int& ei, int& nPushed) {
    io.candsOut[nPushed] = io.extras[ei].book;
    io.candsOutSrc[nPushed] = io.extras[ei].stateSrc;
    ++nPushed;
    ++ei;
  }

  // Returns true if the seed's candidate list was rebuilt (candsOut/nCandsOut valid), false if MkFitCore leaves the
  // CombCandidate untouched (no options and state != Finding). Extras are consumed in both cases.
  ALPAKA_FN_HOST_ACC inline bool selectSeedCandidates(const SeedSelParams& p, SeedSelIO& io) {
    const int K = p.maxCandsPerSeed;
    *io.nUpd = 0;
    *io.nOvl = 0;

    // ---- pass over the options: top-K non-stop options and the first best stop option ----
    int topIdx[kMaxCandsPerSeed];
    float topScore[kMaxCandsPerSeed];
    int nTop = 0;
    int nOpts = 0;
    int stopIdx = -1;
    float stopScore = 0.f;
    for (int j = 0; j < io.nOptSlots; ++j) {
      if (!((io.liveOpts >> j) & 1u))
        continue;
      const CandOption& o = io.opts[j];
      if (o.hitIdx == kOptEmpty)
        continue;
      ++nOpts;
      if (o.hitIdx == kHitStopIdx) {
        if (stopIdx < 0 || o.score > stopScore) {
          stopIdx = j;
          stopScore = o.score;
        }
        continue;
      }
      // stable position: after every kept entry with score >= o.score
      int pos = nTop;
      while (pos > 0 && topScore[pos - 1] < o.score)
        --pos;
      if (pos >= K)
        continue;
      const int last = nTop < K ? nTop : K - 1;
      for (int q = last; q > pos; --q) {
        topIdx[q] = topIdx[q - 1];
        topScore[q] = topScore[q - 1];
      }
      topIdx[pos] = j;
      topScore[pos] = o.score;
      if (nTop < K)
        ++nTop;
    }

    if (nOpts == 0) {
      // MkFitCore: if (ccand.state() == Finding) { ccand.clear(); append all extras; }
      if (io.state == kFinding) {
        for (int e = 0; e < io.nExtras; ++e) {
          io.candsOut[e] = io.extras[e].book;
          io.candsOutSrc[e] = io.extras[e].stateSrc;
        }
        *io.nCandsOut = io.nExtras;
        return true;
      }
      return false;
    }

    // ---- replay of the MkFitCore walk over the top-K entries and the best stop option ----
    int nPushed = 0;
    int ei = 0;
    bool stopPending = stopIdx >= 0;
    int t = 0;
    while (true) {
      const bool takeStop =
          stopPending && (t >= nTop || stopScore > topScore[t] || (stopScore == topScore[t] && stopIdx < topIdx[t]));
      if (takeStop) {
        stopPending = false;
        selConsiderStop(p, io, io.opts[stopIdx]);
        continue;
      }
      if (t >= nTop)
        break;
      const CandOption& h = io.opts[topIdx[t++]];
      // squeeze in extras that are strictly better (sortByScoreTrackCand(*extra_i, tc))
      while (ei < io.nExtras && io.extras[ei].book.score > h.score && nPushed < K)
        selPushExtra(io, ei, nPushed);
      if (nPushed >= K)
        break;
      selPushOption(p, io, h, nPushed);
      if (nPushed >= K)
        break;
    }
    // remaining extras while there is room
    while (ei < io.nExtras && nPushed < K)
      selPushExtra(io, ei, nPushed);
    *io.nCandsOut = nPushed;
    return true;
  }

}  // namespace mkfitdev

#endif
