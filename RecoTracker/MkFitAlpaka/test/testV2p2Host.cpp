// Host checks of the candidate bookkeeping against MkFitCore: the v2p2 plan tables (interface/v2p2/V2p2Plan.h) against
// the walk of SteeringParams::iterator in findTracksStandardv2p2, the track scorer v2p2:llh against the registered
// one, the final sorting with it (mergeCandsAndBestShortOne, interface/cands/CandSeedOps.h) against
// CombCandidate::mergeCandsAndBestShortOne on random seeds with many equal scores, and the status counter names.

#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "RecoTracker/MkFitAlpaka/interface/StatusProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Plan.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/SteeringParams.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

namespace {

  int nFailed = 0;

  void check(bool good, const char* what) {
    nFailed += !good;
    std::printf("%s %s\n", good ? "ok  " : "FAIL", what);
  }

  // One region of findTracksStandardv2p2: the entry searched at each step, the entry whose awaken_candidates() runs
  // before it (setup, then end_layer() of the previous entry) and the plan index; the last step only awakens.
  struct PlanWalk {
    std::vector<int> layer, prevLayer, planIndex;
  };

  PlanWalk walkV2p2(const mkfit::SteeringParams& sp, mkfit::SteeringParams::IterationType_e type) {
    PlanWalk walk;
    if (type == mkfit::SteeringParams::IT_BkwSearch && !sp.has_bksearch_plan())
      return walk;
    auto it = sp.make_iterator(type);
    int prev = it.layer();
    while (++it) {
      walk.layer.push_back(it.layer());
      walk.prevLayer.push_back(prev);
      walk.planIndex.push_back(it.index());
      prev = it.layer();
    }
    walk.layer.push_back(-1);
    walk.prevLayer.push_back(prev);
    walk.planIndex.push_back(-1);
    return walk;
  }

  // Regions with plans of different lengths and pickups (backward pickup -1 = no backward-search plan).
  bool checkV2p2StepTables() {
    const std::vector<int> nPlan = {9, 12, 5, 12, 1};
    const std::vector<int> fwdPickup = {0, 3, 4, 11, 0};
    const std::vector<int> bkwPickup = {8, 0, -1, 6, 0};
    std::vector<mkfit::SteeringParams> sps(nPlan.size());
    mkfitdev::EnginePlan plan;
    for (size_t r = 0; r < nPlan.size(); ++r) {
      for (int i = 0; i < nPlan[r]; ++i)
        sps[r].append_plan(3 * i + int(r));
      sps[r].set_iterator_limits(fwdPickup[r], 0, bkwPickup[r]);
      plan.layers.emplace_back();
      for (const auto& lc : sps[r].m_layer_plan)
        plan.layers.back().push_back(lc.m_layer);
      plan.fwdPickup.push_back(fwdPickup[r]);
      plan.bkwPickup.push_back(bkwPickup[r]);
    }
    bool same = true;
    for (bool fwd : {true, false}) {
      const mkfitdev::V2p2StepTablesHost tables = mkfitdev::makeV2p2StepTables(plan, fwd);
      const auto& st = tables.steps;
      const mkfitdev::EngineStepTablesHost engineTables = mkfitdev::makeEngineStepTables(plan, fwd);
      same = same && st.nRegions == int(nPlan.size());
      size_t maxSteps = 0;
      for (size_t r = 0; r < nPlan.size(); ++r) {
        // the regions the search runs on: findTracksStandardv2p2 / findTracksCloneEngine skip a region without a
        // backward-search plan
        const bool searched = fwd || sps[r].has_bksearch_plan();
        same = same && bool(st.plans.hasPlan[r]) == searched && bool(engineTables.plans.hasPlan[r]) == searched;
        const PlanWalk walk =
            walkV2p2(sps[r], fwd ? mkfit::SteeringParams::IT_FwdSearch : mkfit::SteeringParams::IT_BkwSearch);
        maxSteps = walk.layer.size() > maxSteps ? walk.layer.size() : maxSteps;
        for (int step = 0; step < st.nSteps; ++step) {
          const size_t row = step * st.nRegions + r;
          const bool inWalk = size_t(step) < walk.layer.size();
          same = same && st.layer[row] == (inWalk ? walk.layer[step] : -1) &&
                 st.prevLayer[row] == (inWalk ? walk.prevLayer[step] : -1) &&
                 tables.planIndex[row] == (inWalk ? walk.planIndex[step] : -1);
        }
      }
      same = same && size_t(st.nSteps) == maxSteps;
    }
    return same;
  }

  bool checkTrackScoreV2p2Llh() {
    const mkfit::track_score_func ref = mkfit::IterationConfig::get_track_scorer("v2p2:llh");
    bool same = true;
    for (float scoreIn : {0.f, -3.25f, 17.5f, 123.456f})
      for (int nInside = 0; nInside < 4; ++nInside)
        for (int nTail = 0; nTail < 4; ++nTail) {
          mkfit::TrackScoreInput in;
          mkfitdev::TrackScoreInput dev;
          in.score_in = dev.score_in = scoreIn;
          in.n_inside_holes = dev.n_inside_holes = nInside;
          in.n_tail_holes = dev.n_tail_holes = nTail;
          in.n_found_hits = dev.n_found_hits = 9;
          in.chi2 = dev.chi2 = 4.5f;
          in.pt = dev.pt = 1.1f;
          const float a = ref(in);
          const float b = mkfitdev::trackScore(mkfitdev::TrackScorer::V2p2Llh, dev);
          same = same && std::memcmp(&a, &b, sizeof(a)) == 0;
        }
    return same;
  }

  // The device's bookkeeping copy of a TrackCand; originIndex carries the label to identify it after the sort.
  mkfitdev::CandBook bookOf(const mkfit::TrackCand& tc) {
    mkfitdev::CandBook book{};
    book.score = tc.score();
    book.chi2 = tc.chi2();
    book.nFound = tc.nFoundHits();
    book.nOverlap = tc.nOverlapHits();
    book.nInsideMinusOne = tc.nInsideMinusOneHits();
    book.nTailMinusOne = tc.nTailMinusOneHits();
    book.originIndex = tc.label();
    return book;
  }

  bool checkMergeV2p2Llh() {
    constexpr int kNumSeeds = 3000;
    constexpr int kLayer = 20;
    std::mt19937 rng(27182);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    auto pick = [&rng](int n) { return std::uniform_int_distribution<int>(0, n - 1)(rng); };
    mkfit::IterationParams params;
    params.maxCandsPerSeed = mkfitdev::kMaxCandsPerSeed;
    const mkfit::track_score_func scorer = mkfit::IterationConfig::get_track_scorer("v2p2:llh");
    mkfit::EventOfCombCandidates combs;
    combs.reset(kNumSeeds, params.maxCandsPerSeed);
    bool same = true;
    for (int seed = 0; seed < kNumSeeds; ++seed) {
      mkfit::CombCandidate& comb = combs[seed];
      const int nCands = pick(mkfitdev::kMaxCandsPerSeed + 1);
      const int labelBestShort = mkfitdev::kMaxCandsPerSeed;
      auto makeCand = [&](int label) {
        mkfit::TrackState state;
        state.parameters.At(3) = 1.f / (0.5f + 2.f * uni(rng));
        state.charge = 1;
        mkfit::TrackCand cand(mkfit::TrackBase(state, 0.f, label), &comb);
        for (int hit = 0, nHits = 2 + pick(8); hit < nHits; ++hit) {
          const float u = uni(rng);
          cand.addHitIdx(u < 0.7f ? pick(1000) : (u < 0.9f ? mkfit::Hit::kHitMissIdx : mkfit::Hit::kHitEdgeIdx),
                         kLayer - nHits + hit,
                         10.f * uni(rng));
        }
        cand.setScore(0.5f * pick(40) - 5.f);  // summed layer-step score, coarse: many equal final scores
        return cand;
      };
      mkfitdev::CandBook books[mkfitdev::kMaxCandsPerSeed];
      mkfitdev::CandState states[mkfitdev::kMaxCandsPerSeed] = {};
      float pt[mkfitdev::kMaxCandsPerSeed];
      for (int ic = 0; ic < nCands; ++ic) {
        comb.push_back(makeCand(ic));
        books[ic] = bookOf(comb[ic]);
        states[ic].par[3] = comb[ic].parameters().At(3);
        pt[ic] = comb[ic].pT();
      }
      mkfitdev::CandBook bestShort{};
      mkfitdev::CandState bestShortState{};
      int8_t bestShortValid = 0;
      if (uni(rng) < 0.5f) {
        comb.setBestShortCand(makeCand(labelBestShort));
        bestShort = bookOf(comb.refBestShortCand());
        bestShortValid = 1;
      }
      const float ptBest = bestShortValid ? comb.refBestShortCand().pT() : 0.f;
      mkfitdev::HoTNode hots[1] = {};
      int8_t state = mkfitdev::kFinding;
      int16_t pickupLayer = 0, lastHitIdxBeforeBkw = 0, nInsideBeforeBkw = 0, nTailBeforeBkw = 0;
      int32_t nCandsDev = nCands, nHots = 0;
      uint32_t overflowBits = 0;
      mkfitdev::SeedCandsRef ref{&state,
                                 &pickupLayer,
                                 books,
                                 states,
                                 &nCandsDev,
                                 &bestShort,
                                 &bestShortState,
                                 &bestShortValid,
                                 hots,
                                 nullptr,
                                 1,
                                 1,
                                 &nHots,
                                 &lastHitIdxBeforeBkw,
                                 &nInsideBeforeBkw,
                                 &nTailBeforeBkw,
                                 &overflowBits};
      mkfitdev::mergeCandsAndBestShortOne(
          ref, params.maxCandsPerSeed, mkfitdev::TrackScorer::V2p2Llh, true, true, pt, ptBest);
      comb.mergeCandsAndBestShortOne(params, scorer, true, true);
      same = same && nCandsDev == int(comb.size()) && bestShortValid == 0;
      for (int ic = 0; ic < nCandsDev && ic < int(comb.size()); ++ic) {
        const float scoreRef = comb[ic].score();
        same = same && books[ic].originIndex == comb[ic].label() &&
               std::memcmp(&books[ic].score, &scoreRef, sizeof(scoreRef)) == 0;
      }
    }
    return same;
  }

  bool checkStatusCounterNames() {
    std::set<std::string> names;
    for (int i = 0; i < mkfitdev::kNumStatusCounters; ++i)
      names.insert(mkfitdev::statusCounterName(i));
    return int(names.size()) == mkfitdev::kNumStatusCounters && names.count("unused") == 0;
  }

}  // namespace

int main() {
  check(checkV2p2StepTables(),
        "makeV2p2StepTables = SteeringParams::iterator walk of findTracksStandardv2p2, regions searched as MkBuilder");
  check(checkTrackScoreV2p2Llh(), "trackScore(V2p2Llh) = MkFitCore v2p2:llh");
  check(checkMergeV2p2Llh(), "mergeCandsAndBestShortOne(V2p2Llh) = CombCandidate::mergeCandsAndBestShortOne");
  check(checkStatusCounterNames(), "status counter names distinct and named");
  std::printf("%s: %d check(s) failed\n", nFailed ? "FAIL" : "PASS", nFailed);
  return nFailed ? 1 : 0;
}
