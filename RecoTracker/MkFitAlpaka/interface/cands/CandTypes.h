#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandTypes_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandTypes_h

// Candidate bookkeeping of the clone engine and of MkFinderV2p2: device-usable PODs that mirror MkFitCore
// (TrackStructures.h: TrackCand, HoTNode, HitMatch(Pair); IdxChi2List.h; MkFinder.h: UpdateIndices).
// All functions are ALPAKA_FN_HOST_ACC and transliterate the MkFitCore code (same operations, same order).

#include <algorithm>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/HitConstants.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"

namespace mkfitdev {

  // ---- capacities (fixed; options and extras cannot overflow by construction, the HoT pool is counted) ----
  // maxCandsPerSeed of the LST step is 5 (forward and backward); the slots are sized for it.
  constexpr int kMaxCandsPerSeed = 5;
  // MkFinder::selectHitIndicesV2 keeps at most NEW_MAX_HIT = 6 hits per candidate (MkFinder.cc).
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
  enum CandOverflow : uint32_t { kOverflowHotsBit = 1u };

  // CombCandidate::SeedState_e
  enum SeedState : int8_t { kDormant = 0, kFinding = 1, kFinished = 2 };

  // MkFitCore: struct HoTNode { HitOnTrack m_hot; float m_chi2; int m_prev_idx; } (HitOnTrack = index:24, layer:8)
  struct HoTNode {
    int32_t index;
    int32_t layer;
    float chi2;
    int32_t prev;
  };

  // MkFitCore: struct HitMatch / HitMatchPair (TrackStructures.h)
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

    // TrackCand::addHitIdx (TrackStructures.h) given the node index already allocated
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

  // TrackCand::recountHolesAndOverlaps (TrackStructures.cc): the hole and overlap counts from the HoT chain, after the
  // backward fit's outlier rejection. hot(idx) = node idx of the chain; nSeedHits = TrackBase::getNSeedHits().
  template <typename THot>
  ALPAKA_FN_HOST_ACC inline void recountHolesAndOverlaps(CandBook& tc, int nSeedHits, THot const& hot) {
    // Nodes are numbered from the first of the chain; the first nSeedHits are the seed's.
    int n_nodes = 0;
    for (int idx = tc.lastHitIdx; idx >= 0; idx = hot(idx).prev)
      ++n_nodes;
    const int n_seed = nSeedHits;
    auto found = [](const HoTNode& h) { return h.index >= 0 || h.index == kHitCCCFilterIdx; };

    int n_inside = 0, n_tail = 0, n_overlap = 0;
    bool seen_found = false;  // a found hit later in the chain
    int later_found_layer = -1;
    // After the seed's hits, a run of consecutive nodes on one layer is one crossing of the layer:
    // one hole if it has a miss and no found hit. Rejecting one hit of an overlap pair leaves no
    // hole, rejecting both leaves one.
    int run_layer = -1;
    bool run_found = false, run_miss = false, run_seen_found = false;
    auto close_run = [&]() {
      if (run_miss && !run_found)
        ++(run_seen_found ? n_inside : n_tail);
      run_layer = -1;
      run_found = run_miss = false;
    };
    int pos = n_nodes;
    for (int idx = tc.lastHitIdx; idx >= 0; idx = hot(idx).prev) {
      --pos;
      const HoTNode& h = hot(idx);
      const bool after_seed = pos >= n_seed;
      if (!after_seed || h.layer != run_layer) {
        close_run();
        if (after_seed) {
          run_layer = h.layer;
          run_seen_found = seen_found;
        }
      }
      if (found(h)) {
        if (h.layer == later_found_layer && after_seed)
          ++n_overlap;
        later_found_layer = h.layer;
        seen_found = true;
        if (after_seed)
          run_found = true;
      } else if (h.index == kHitMissIdx) {
        if (after_seed)
          run_miss = true;
        else
          ++(seen_found ? n_inside : n_tail);
      }
    }
    close_run();
    tc.nInsideMinusOne = n_inside;
    tc.nTailMinusOne = n_tail;
    tc.nOverlap = n_overlap;
  }

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

  // UpdateIndices (MkFinder.h), seed index implicit (per-seed fixed slots)
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

  // What a track scorer sees of a track or candidate (FunctionTypes.h TrackScoreInput). Hole counts are always the
  // real ones; penalize_tail_holes says whether the caller wants the tail holes charged. score_in is the score the
  // object carries when the scorer is called, 0 where it carries none.
  struct TrackScoreInput {
    int n_found_hits = 0;
    int n_tail_holes = 0;
    int n_overlap_hits = 0;
    int n_inside_holes = 0;
    int n_seed_hits = 0;
    float chi2 = 0;
    float pt = 0;
    float score_in = 0;
    bool penalize_tail_holes = false;
    bool in_find_candidates = false;
  };

  // ---- scorer 'default' / 'phase1:default' (MkStdSeqs.cc trackScoreDefault, Config.h) ----
  ALPAKA_FN_HOST_ACC inline float trackScoreDefault(const TrackScoreInput& in) {
    const int ntailholes = in.penalize_tail_holes ? in.n_tail_holes : 0;
    float maxBonus = 8.0;
    float bonus = Config::validHitSlope_ * in.n_found_hits + Config::validHitBonus_;
    float penalty = Config::missingHitPenalty_;
    float tailPenalty = Config::tailMissingHitPenalty_;
    float overlapBonus = Config::overlapHitBonus_;
    if (in.pt < 0.9) {
      penalty *= in.in_find_candidates ? 1.7f : 1.5f;
      float scaledBonus = bonus * (in.in_find_candidates ? 0.9f : 1.0f);
      bonus = (maxBonus < scaledBonus) ? maxBonus : scaledBonus;  // std::min(scaledBonus, maxBonus), same NaN behaviour
    }
    float score = bonus * in.n_found_hits + overlapBonus * in.n_overlap_hits - penalty * in.n_inside_holes -
                  tailPenalty * ntailholes - in.chi2;
    return score;
  }

  // getScoreStruct (Track.h)
  ALPAKA_FN_HOST_ACC inline float getScoreStruct(const CandOption& cand1) {
    TrackScoreInput in;
    in.n_found_hits = cand1.nhits;
    in.n_tail_holes = cand1.ntailholes;
    in.n_overlap_hits = cand1.noverlaps;
    in.n_inside_holes = cand1.nholes;
    // Do not allow for chi2<0 in score calculation
    in.chi2 = std::max(cand1.chi2, 0.f);
    in.pt = cand1.pt;
    in.penalize_tail_holes = true;
    in.in_find_candidates = true;
    return trackScoreDefault(in);
  }

  // ---- scorer 'v2p2:llh' (MkBuilder.cc trackScoreV2p2Llh) ----
  // For the final pick of the forward search, where score_in is the summed
  // layer-step score of the search (V2p2Score.h). It compares candidates of one
  // seed only, since that sum starts at 0 at pickup
  // (MkFinderV2p2::awaken_candidates()).
  ALPAKA_FN_HOST_ACC inline float trackScoreV2p2Llh(const TrackScoreInput& in) {
    namespace po = Config::V2p2::Policy;
    return in.score_in - po::final_pick_hole_penalty * in.n_inside_holes -
           po::final_pick_tail_penalty * in.n_tail_holes;
  }

  // ---- scorer 'keep' (MkStdSeqs.cc trackScoreKeep) ----
  // Keeps the score the track carries. For a task scorer (final pick, after the
  // backward fit, before the duplicate cleaner) that should not re-score.
  ALPAKA_FN_HOST_ACC inline float trackScoreKeep(const TrackScoreInput& in) { return in.score_in; }

  // The track scorer of a task by its ES name (es/ESConfig.h TrackScorer): Default = trackScoreDefault,
  // V2p2Llh = trackScoreV2p2Llh, Keep = trackScoreKeep. None never reaches a task (esFill resolves it to the
  // iteration's scorer) and scores as Default.
  ALPAKA_FN_HOST_ACC inline float trackScore(TrackScorer scorer, const TrackScoreInput& in) {
    switch (scorer) {
      case TrackScorer::V2p2Llh:
        return trackScoreV2p2Llh(in);
      case TrackScorer::Keep:
        return trackScoreKeep(in);
      default:
        return trackScoreDefault(in);
    }
  }

  // getScoreCand(score_func, cand, ...) for TrackCand (TrackStructures.h) with the scorer chosen by name; pt =
  // |1/par[3]| and nSeedHits = getNSeedHits() of the candidate's state
  ALPAKA_FN_HOST_ACC inline float getScoreCand(TrackScorer scorer,
                                               const CandBook& cand1,
                                               float pt,
                                               int nSeedHits,
                                               bool penalizeTailMissHits = false,
                                               bool inFindCandidates = false) {
    TrackScoreInput in;
    in.n_found_hits = cand1.nFound;
    in.n_tail_holes = cand1.nTailMinusOne;
    in.n_overlap_hits = cand1.nOverlap;
    in.n_inside_holes = cand1.nInsideMinusOne;
    in.n_seed_hits = nSeedHits;
    // Do not allow for chi2<0 in score calculation
    in.chi2 = std::max(cand1.chi2, 0.f);
    in.pt = pt;
    in.score_in = cand1.score;
    in.penalize_tail_holes = penalizeTailMissHits;
    in.in_find_candidates = inFindCandidates;
    return trackScore(scorer, in);
  }

  // ... with the scorer 'default' (the clone engine's search and merge)
  ALPAKA_FN_HOST_ACC inline float getScoreCand(const CandBook& cand1,
                                               float pt,
                                               int nSeedHits,
                                               bool penalizeTailMissHits = false,
                                               bool inFindCandidates = false) {
    return getScoreCand(TrackScorer::Default, cand1, pt, nSeedHits, penalizeTailMissHits, inFindCandidates);
  }

  // getScoreWorstPossible()
  ALPAKA_FN_HOST_ACC inline constexpr float scoreWorstPossible() { return -3.402823466e+38f; }

}  // namespace mkfitdev

#endif
