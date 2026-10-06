#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandTypes_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandTypes_h

// Candidate bookkeeping of the clone engine: device-usable PODs that mirror MkFitCore
// (TrackStructures.h: TrackCand, HoTNode, HitMatch(Pair); IdxChi2List.h; MkFinder.h: UpdateIndices).
// All functions are ALPAKA_FN_HOST_ACC and transliterate the MkFitCore code (same operations, same order).

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/math/HitConstants.h"

namespace mkfitdev {

  // ---- capacities (fixed; every buffer has an overflow counter, see CandsSoA.h) ----
  // maxCandsPerSeed of the LST step is 5 (forward and backward); the slots are sized for it.
  constexpr int kMaxCandsPerSeed = 5;
  // MkFinder::selectHitIndicesV2 keeps at most NEW_MAX_HIT = 6 hits per candidate (MkFinder.cc:793).
  constexpr int kMaxHitsPerCand = 6;
  // Option slots per seed: per candidate kMaxHitsPerCand hit options + 1 invalid ("no hit") option.
  // Layout written by the chi2 kernel: slot = j * kMaxCandsPerSeed + ic, j = hit slot (0..5) or 6 (invalid),
  // i.e. hit-slot major like MkFitCore's MkFinder::findCandidatesCloneEngine loop (hit_cnt outer, track inner).
  constexpr int kMaxOptsPerSeed = (kMaxHitsPerCand + 1) * kMaxCandsPerSeed;  // 35
  // Held-back ("extra") candidates per seed: at most one per live candidate.
  constexpr int kMaxExtrasPerSeed = kMaxCandsPerSeed;
  // Default HoT node pool per seed (runtime value: SeedCandsSoA::hotsPerSeed). MkFitCore reserves 128 and grows.
  // Overflow is flagged and counted, never silent.
  constexpr int kDefaultHotsPerSeed = 256;

  // Hit index codes (Hit.h): interface/math/HitConstants.h
  // Empty option slot (not a MkFitCore code).
  constexpr int kOptEmpty = -1000000;

  // Per-seed overflow flags (SeedCandsSoA::overflowBits); a flagged seed must be treated as failed.
  enum CandOverflow : uint32_t { kOverflowHotsBit = 1u, kOverflowOptsBit = 2u, kOverflowExtrasBit = 4u };

  // CombCandidate::SeedState_e
  enum SeedState : int8_t { kDormant = 0, kFinding = 1, kFinished = 2 };

  // MkFitCore: struct HoTNode { HitOnTrack m_hot; float m_chi2; int m_prev_idx; } (HitOnTrack = index:24, layer:8)
  struct HoTNode {
    int32_t index;
    int32_t layer;
    float chi2;
    int32_t prev;
  };

  // MkFitCore: struct HitMatch / HitMatchPair (TrackStructures.h:28-89)
  struct HitMatch {  // MkFitCore defaults: hit -1, module -1, chi2 1e9 (set by HitMatchPair::reset)
    int32_t hit;
    int32_t module;
    float chi2;
  };

  struct HitMatchPair {
    HitMatch M[2];

    ALPAKA_FN_HOST_ACC inline void reset() {
      M[0] = HitMatch{-1, -1, 1e9f};
      M[1] = HitMatch{-1, -1, 1e9f};
    }

    ALPAKA_FN_HOST_ACC inline void considerHitForOverlap(int hit_idx, int module_id, float chi2) {
      if (module_id == M[0].module) {
        if (chi2 < M[0].chi2) {
          M[0].chi2 = chi2;
          M[0].hit = hit_idx;
        }
      } else if (module_id == M[1].module) {
        if (chi2 < M[1].chi2) {
          M[1].chi2 = chi2;
          M[1].hit = hit_idx;
        }
      } else {
        if (M[0].chi2 > M[1].chi2) {
          if (chi2 < M[0].chi2) {
            M[0] = {hit_idx, module_id, chi2};
          }
        } else {
          if (chi2 < M[1].chi2) {
            M[1] = {hit_idx, module_id, chi2};
          }
        }
      }
    }

    // Returns the index (0/1) of the overlap match, or -1 (MkFitCore returns HitMatch* or nullptr).
    ALPAKA_FN_HOST_ACC inline int findOverlap(int hit_idx, int module_id) const {
      if (module_id == M[0].module) {
        if (M[1].hit >= 0)
          return 1;
      } else if (module_id == M[1].module) {
        if (M[0].hit >= 0)
          return 0;
      } else {
        if (M[0].chi2 <= M[1].chi2) {
          if (M[0].hit >= 0)
            return 0;
        } else {
          if (M[1].hit >= 0)
            return 1;
        }
      }
      return -1;
    }
  };

  // Bookkeeping part of a TrackCand (TrackBase::score_/chi2_/lastHitIdx_/nFoundHits_ + TrackCand members).
  // The track state (parameters, errors, charge, label, status) lives in the candidate slot SoA next to it and is
  // copied from the parent when a candidate is cloned.
  struct CandBook {
    float score;
    float chi2;
    int32_t lastHitIdx;  // index into the seed's HoT pool, -1 = none
    int16_t nFound;
    int16_t nMissing;
    int16_t nOverlap;
    int16_t nInsideMinusOne;
    int16_t nTailMinusOne;
    int16_t originIndex;
    HitMatchPair overlaps;

    // TrackCand::addHitIdx (TrackStructures.h:526-543) given the node index already allocated
    ALPAKA_FN_HOST_ACC inline void addHitCounters(int hitIdx, float hchi2) {
      if (hitIdx >= 0 || hitIdx == kHitCCCFilterIdx) {
        ++nFound;
        chi2 += hchi2;
        nInsideMinusOne += nTailMinusOne;
        nTailMinusOne = 0;
      } else {
        ++nMissing;
        if (hitIdx == kHitMissIdx)
          ++nTailMinusOne;
      }
    }
  };

  // Track state of a candidate (TrackBase: TrackState parameters/errors/charge, label_, status_).
  // err holds the 21 lower-triangle elements of the symmetric 6x6 in SMatrixSym66 / MatriplexSym order.
  struct CandState {
    float par[6];
    float err[21];
    int32_t charge;
    int32_t label;
    uint32_t status;    // TrackBase::Status bits (prod type, region, ...)
    int32_t nSeedHits;  // TrackBase::getNSeedHits()
  };

  // IdxChi2List (IdxChi2List.h), same members and order
  struct CandOption {
    uint32_t module;
    int32_t hitIdx;
    int32_t trkIdx;
    int32_t nhits;
    int32_t ntailholes;
    int32_t noverlaps;
    int32_t nholes;
    float pt;
    float chi2;
    float chi2_hit;
    float score;
  };

  // UpdateIndices (MkFinder.h:29-36), seed index implicit (per-seed fixed slots)
  struct CandUpdate {
    int32_t cand_idx;
    int32_t hit_idx;
    int32_t ovlp_idx;
  };

  // A held-back candidate (MkBuilder::find_tracks_handle_missed_layers): bookkeeping copy + the slot its state
  // comes from (index into the seed's input candidates).
  struct CandExtra {
    CandBook book;
    int32_t stateSrc;
    int32_t needsStop;  // barrel-region copy of a barrel-layer WSR_Failed cand: append a -2 stop node (K4)
  };

  // ---- scorer 'phase1:default' (MkStdSeqs.cc:738-756, Config.h:82-86) ----
  ALPAKA_FN_HOST_ACC inline float trackScoreDefault(const int nfoundhits,
                                                    const int ntailholes,
                                                    const int noverlaphits,
                                                    const int nmisshits,
                                                    const float chi2,
                                                    const float pt,
                                                    const bool inFindCandidates) {
    constexpr float validHitBonus_ = 4;
    constexpr float validHitSlope_ = 0.2;
    constexpr float overlapHitBonus_ = 0;
    constexpr float missingHitPenalty_ = 8;
    constexpr float tailMissingHitPenalty_ = 3;
    float maxBonus = 8.0;
    float bonus = validHitSlope_ * nfoundhits + validHitBonus_;
    float penalty = missingHitPenalty_;
    float tailPenalty = tailMissingHitPenalty_;
    float overlapBonus = overlapHitBonus_;
    if (pt < 0.9) {
      penalty *= inFindCandidates ? 1.7f : 1.5f;
      float b = bonus * (inFindCandidates ? 0.9f : 1.0f);
      bonus = (maxBonus < b) ? maxBonus : b;  // std::min(b, maxBonus), same NaN behaviour
    }
    float score =
        bonus * nfoundhits + overlapBonus * noverlaphits - penalty * nmisshits - tailPenalty * ntailholes - chi2;
    return score;
  }

  // getScoreStruct (Track.h:629-640)
  ALPAKA_FN_HOST_ACC inline float getScoreStruct(const CandOption& c) {
    float chi2 = c.chi2;
    if (chi2 < 0)
      chi2 = 0.f;
    return trackScoreDefault(c.nhits, c.ntailholes, c.noverlaps, c.nholes, chi2, c.pt, true);
  }

  // getScoreCand for TrackCand (TrackStructures.h:245-260); pt = |1/par[3]| of the candidate's state
  ALPAKA_FN_HOST_ACC inline float getScoreCand(const CandBook& c,
                                               float pt,
                                               bool penalizeTailMissHits = false,
                                               bool inFindCandidates = false) {
    int ntailmisshits = penalizeTailMissHits ? c.nTailMinusOne : 0;
    float chi2 = c.chi2;
    if (chi2 < 0)
      chi2 = 0.f;
    return trackScoreDefault(c.nFound, ntailmisshits, c.nOverlap, c.nInsideMinusOne, chi2, pt, inFindCandidates);
  }

  // getScoreWorstPossible()
  ALPAKA_FN_HOST_ACC inline constexpr float scoreWorstPossible() { return -3.402823466e+38f; }

}  // namespace mkfitdev

#endif
