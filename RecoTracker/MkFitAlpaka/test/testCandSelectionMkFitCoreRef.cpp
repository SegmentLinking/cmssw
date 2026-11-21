// The clone engine's candidate selection (interface/cands/CandSelection.h, selectSeedCandidates, one seed per call on
// every backend) against MkFitCore's CandCloner::processSeedRange on random seeds: parents with hit chains and overlap
// candidates, held-back (extra) candidates, option lists with stop, missing and found hits and many exact score ties,
// the best-short candidate. Compared per seed: the new candidates (bookkeeping, hit chain, state source), the kalman
// update list and the best-short candidate. Node indices of the hit pools are not compared (the device appends nodes
// only for what it keeps), the chains are.

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandSelection.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"
// package-private MkFitCore headers on purpose: angle brackets keep scram's private-header check off this test
#include <RecoTracker/MkFitCore/src/CandCloner.h>
#include <RecoTracker/MkFitCore/src/MkFinder.h>

namespace {

  constexpr int kNumSeeds = 4000;
  constexpr int kLayer = 20;
  constexpr int kHotCap = 1024;
  // with more options than this the order of equal scores is not reproduced (doc/SIMPLIFICATIONS.txt)
  constexpr int kMaxOptionsWithTies = 16;

  struct Chain {
    std::vector<int> hit, layer;
    std::vector<float> chi2;
    bool operator==(Chain const&) const = default;
  };

  Chain chainOf(const mkfit::CombCandidate& comb, int node) {
    Chain chain;
    for (; node >= 0; node = comb.hot_node(node).m_prev_idx) {
      chain.hit.push_back(comb.hot_node(node).m_hot.index);
      chain.layer.push_back(comb.hot_node(node).m_hot.layer);
      chain.chi2.push_back(comb.hot_node(node).m_chi2);
    }
    return chain;
  }

  Chain chainOf(const std::vector<mkfitdev::HoTNode>& hots, int node) {
    Chain chain;
    for (; node >= 0; node = hots[node].prev) {
      chain.hit.push_back(hots[node].index);
      chain.layer.push_back(hots[node].layer);
      chain.chi2.push_back(hots[node].chi2);
    }
    return chain;
  }

  // the device bookkeeping of a TrackCand; the overlap candidates are replayed on both sides
  mkfitdev::CandBook bookOf(const mkfit::TrackCand& cand, const mkfitdev::HitMatchPair& overlaps) {
    mkfitdev::CandBook book{};
    book.score = cand.score();
    book.chi2 = cand.chi2();
    book.lastHitIdx = cand.lastCcIndex();
    book.nFound = cand.nFoundHits();
    book.nMissing = cand.nMissingHits();
    book.nOverlap = cand.nOverlapHits();
    book.nInsideMinusOne = cand.nInsideMinusOneHits();
    book.nTailMinusOne = cand.nTailMinusOneHits();
    book.originIndex = cand.originIndex();
    book.overlaps = overlaps;
    return book;
  }

  bool sameCand(const mkfit::CombCandidate& comb,
                const mkfit::TrackCand& ref,
                int refLabel,
                const std::vector<mkfitdev::HoTNode>& hots,
                const mkfitdev::CandBook& got,
                int gotLabel) {
    return ref.score() == got.score && ref.chi2() == got.chi2 && ref.nFoundHits() == got.nFound &&
           ref.nMissingHits() == got.nMissing && ref.nOverlapHits() == got.nOverlap &&
           ref.nInsideMinusOneHits() == got.nInsideMinusOne && ref.nTailMinusOneHits() == got.nTailMinusOne &&
           refLabel == gotLabel && chainOf(comb, ref.lastCcIndex()) == chainOf(hots, got.lastHitIdx);
  }

  struct SeedInput {
    std::vector<mkfitdev::CandBook> parents;
    std::vector<float> parentPt;
    std::vector<mkfitdev::CandExtra> extras;
    std::vector<mkfitdev::CandOption> options;
    std::vector<mkfitdev::HoTNode> hots;  // the pool before the selection, then room up to kHotCap
    int nHots;
    mkfitdev::CandBook bestShort;
    float bestShortScore;
    int state;
  };

}  // namespace

int main() {
  std::mt19937 rng(16180);
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  auto pick = [&rng](int n) { return std::uniform_int_distribution<int>(0, n - 1)(rng); };

  mkfit::IterationParams params;
  params.maxCandsPerSeed = mkfitdev::kMaxCandsPerSeed;
  params.recheckOverlap = false;  // the only supported setting (SupportedConfig)
  int nFailed = 0, nSeedsWithTies = 0, nStopAccepted = 0, nOverlapsAdded = 0;
  for (float pTCutOverlap : {0.f, 1.f}) {
    params.pTCutOverlap = pTCutOverlap;
    mkfit::EventOfCombCandidates combs;
    combs.reset(kNumSeeds, params.maxCandsPerSeed);
    std::vector<std::vector<mkfit::TrackCand>> extras(kNumSeeds);
    std::vector<SeedInput> inputs(kNumSeeds);
    mkfit::CandCloner cloner;
    cloner.setup(params);
    std::vector<mkfit::UpdateIndices> updates, overlapUpdates;
    cloner.begin_eta_bin(&combs, &updates, &overlapUpdates, &extras, 0, kNumSeeds);
    cloner.begin_layer(kLayer);
    cloner.begin_iteration();
    for (int seed = 0; seed < kNumSeeds; ++seed) {
      mkfit::CombCandidate& comb = combs[seed];
      SeedInput& in = inputs[seed];
      in.state = uni(rng) < 0.9f ? mkfitdev::kFinding : mkfitdev::kFinished;
      comb.setState(in.state == mkfitdev::kFinding ? mkfit::CombCandidate::Finding : mkfit::CombCandidate::Finished);
      // parents: own hit chains, scores with ties, pT around the overlap cut, overlap candidates on a few modules
      const int nParents = 1 + pick(mkfitdev::kMaxCandsPerSeed);
      for (int parent = 0; parent < nParents; ++parent) {
        mkfit::TrackState state;
        state.parameters.At(3) = 1.f / (0.5f + 2.f * uni(rng));
        state.charge = 1;
        mkfit::TrackCand cand(mkfit::TrackBase(state, 0.f, parent), &comb);
        const int nHits = 1 + pick(4);
        for (int hit = 0; hit < nHits; ++hit)
          cand.addHitIdx(uni(rng) < 0.8f ? pick(1000) : -1, kLayer - nHits + hit, 10.f * uni(rng));
        cand.setScore(0.5f * pick(20));
        mkfitdev::HitMatchPair overlaps;
        overlaps.reset();
        for (int ov = pick(4); ov > 0; --ov) {
          const int hit = pick(1000), module = pick(5);
          const float chi2 = 10.f * uni(rng);
          cand.considerHitForOverlap(hit, module, chi2);
          overlaps.considerHitForOverlap(hit, module, chi2);
        }
        comb.push_back(cand);
        in.parents.push_back(bookOf(cand, overlaps));
        in.parentPt.push_back(cand.pT());
      }
      // extras: copies of parents that missed the layer, score descending (MkFitCore's order)
      for (int extra = pick(nParents + 1); extra > 0; --extra) {
        const int parent = pick(nParents);
        mkfit::TrackCand cand(comb[parent]);
        cand.addHitIdx(mkfit::Hit::kHitMissIdx, kLayer, 0.f);
        cand.setScore(0.5f * pick(20));
        extras[seed].push_back(cand);
      }
      std::stable_sort(extras[seed].begin(), extras[seed].end(), mkfit::sortByScoreTrackCand);
      for (auto const& cand : extras[seed]) {
        mkfitdev::HitMatchPair overlaps = in.parents[cand.label()].overlaps;
        in.extras.push_back({bookOf(cand, overlaps), cand.label(), 0});
      }
      // the best-short candidate
      mkfit::TrackCand bestShort(comb[0]);
      bestShort.setScore(uni(rng) < 0.5f ? mkfit::getScoreWorstPossible() : 0.5f * pick(20));
      comb.setBestShortCand(bestShort);
      in.bestShort = bookOf(bestShort, in.parents[0].overlaps);
      in.bestShortScore = bestShort.score();
      // options: found, missing, edge, gap and stop hits; exact score ties only up to kMaxOptionsWithTies options
      const int nOptions = uni(rng) < 0.05f ? 0 : pick(mkfitdev::kMaxOptsPerSeed + 1);
      const bool ties = nOptions <= kMaxOptionsWithTies;
      nSeedsWithTies += ties && nOptions > 1;
      const int codes[] = {
          mkfit::Hit::kHitMissIdx, mkfit::Hit::kHitStopIdx, mkfit::Hit::kHitEdgeIdx, mkfit::Hit::kHitInGapIdx};
      for (int option = 0; option < nOptions; ++option) {
        mkfit::IdxChi2List entry;
        entry.reset();
        entry.trkIdx = pick(nParents);
        entry.hitIdx = uni(rng) < 0.6f ? pick(1000) : codes[pick(4)];
        entry.module = pick(5);
        entry.chi2_hit = 10.f * uni(rng);
        entry.score = ties ? 0.5f * pick(20) : 10.f * uni(rng) + 1e-3f * option;
        cloner.add_cand(seed, entry);
        in.options.push_back({entry.module,
                              entry.hitIdx,
                              entry.trkIdx,
                              entry.nhits,
                              entry.ntailholes,
                              entry.noverlaps,
                              entry.nholes,
                              entry.pt,
                              entry.chi2,
                              entry.chi2_hit,
                              entry.score});
      }
      for (int node = 0; node < comb.hotsSize(); ++node)
        in.hots.push_back({comb.hot_node(node).m_hot.index,
                           comb.hot_node(node).m_hot.layer,
                           comb.hot_node(node).m_chi2,
                           comb.hot_node(node).m_prev_idx});
      in.nHots = comb.hotsSize();
    }
    cloner.end_iteration();
    cloner.end_layer();
    cloner.end_eta_bin();
    std::vector<std::vector<mkfit::UpdateIndices>> updatesBySeed(kNumSeeds);
    for (auto const& update : updates)
      updatesBySeed[update.seed_idx].push_back(update);

    // the device selection, seed by seed
    for (int seed = 0; seed < kNumSeeds; ++seed) {
      SeedInput& in = inputs[seed];
      in.hots.resize(kHotCap);
      int32_t nHots = in.nHots;
      mkfitdev::SeedSelParams selParams{kLayer, params.maxCandsPerSeed, params.pTCutOverlap};
      mkfitdev::CandBook bestShort = in.bestShort;
      int8_t bestShortValid = 1;
      int32_t bestShortSrc = -1, nCandsOut = 0, nUpd = 0;
      uint32_t overflowBits = 0;
      mkfitdev::CandBook candsOut[mkfitdev::kMaxCandsPerSeed];
      int32_t candsOutSrc[mkfitdev::kMaxCandsPerSeed];
      mkfitdev::CandUpdate upd[mkfitdev::kMaxCandsPerSeed];
      mkfitdev::SeedSelIO io{};
      io.candsIn = in.parents.data();
      io.candsInPt = in.parentPt.data();
      io.nCandsIn = in.parents.size();
      io.extras = in.extras.data();
      io.nExtras = in.extras.size();
      io.opts = in.options.data();
      io.nOptSlots = in.options.size();
      io.liveOpts = in.options.empty() ? 0 : (~uint64_t(0) >> (64 - in.options.size()));
      io.state = in.state;
      io.bestShort = &bestShort;
      io.bestShortValid = &bestShortValid;
      io.bestShortSrc = &bestShortSrc;
      io.hots = in.hots.data();
      io.extHots = nullptr;
      io.hotHome = kHotCap;
      io.hotCap = kHotCap;
      io.nHots = &nHots;
      io.candsOut = candsOut;
      io.candsOutSrc = candsOutSrc;
      io.nCandsOut = &nCandsOut;
      io.upd = upd;
      io.nUpd = &nUpd;
      io.overflowBits = &overflowBits;
      const bool rebuilt = mkfitdev::selectSeedCandidates(selParams, io);

      const mkfit::CombCandidate& comb = combs[seed];
      bool good = overflowBits == 0;
      if (rebuilt) {
        good = good && int(comb.size()) == nCandsOut;
        for (int k = 0; good && k < nCandsOut; ++k)
          good = sameCand(comb, comb[k], comb[k].label(), in.hots, candsOut[k], candsOutSrc[k]);
      } else {
        good = good && comb.size() == in.parents.size();
        for (int k = 0; good && k < int(comb.size()); ++k)
          good = sameCand(comb, comb[k], comb[k].label(), in.hots, in.parents[k], k);
      }
      const auto& refUpdates = updatesBySeed[seed];
      good = good && int(refUpdates.size()) == (rebuilt ? nUpd : 0);
      for (int k = 0; good && k < int(refUpdates.size()); ++k)
        good = refUpdates[k].cand_idx == upd[k].cand_idx && refUpdates[k].hit_idx == upd[k].hit_idx &&
               refUpdates[k].ovlp_idx == upd[k].ovlp_idx;
      for (int k = 0; rebuilt && k < nCandsOut; ++k)
        nOverlapsAdded += candsOut[k].nOverlap > in.parents[candsOutSrc[k]].nOverlap;
      const mkfit::TrackCand& refShort = comb.refBestShortCand();
      const bool replaced = refShort.score() != in.bestShortScore;
      nStopAccepted += replaced;
      good = good && refShort.score() == bestShort.score && replaced == (bestShortSrc >= 0);
      if (good && replaced)
        good = sameCand(comb, refShort, refShort.label(), in.hots, bestShort, bestShortSrc);
      if (!good && nFailed < 5)
        std::printf("seed %d (pTCutOverlap %.1f): %zu parents, %zu extras, %zu options differ from MkFitCore\n",
                    seed,
                    pTCutOverlap,
                    in.parents.size(),
                    in.extras.size(),
                    in.options.size());
      nFailed += !good;
    }
  }
  std::printf(
      "%s: %d of %d seeds differ from CandCloner::processSeedRange (%d with tied option scores, %d best-short "
      "replacements, %d overlap hits added)\n",
      nFailed ? "FAIL" : "PASS",
      nFailed,
      2 * kNumSeeds,
      nSeedsWithTies,
      nStopAccepted,
      nOverlapsAdded);
  return nFailed ? 3 : 0;
}
