#ifndef TruthPixelSeeds_h
#define TruthPixelSeeds_h

#include <map>
#include <set>
#include <vector>

// The same per-seed inputs prepareInput() (interface/LSTPrepareInput.h) takes from CMSSW's
// see_* branches, built instead from Monte Carlo truth. Feeding these into the same,
// unmodified prepareInput() yields pLS that are guaranteed to truth-match their sim track.
struct TruthPixelSeedVectors {
  std::vector<float> see_px;
  std::vector<float> see_py;
  std::vector<float> see_pz;
  std::vector<float> see_dxy;
  std::vector<float> see_dz;
  std::vector<float> see_ptErr;
  std::vector<float> see_etaErr;
  std::vector<float> see_stateTrajGlbX;
  std::vector<float> see_stateTrajGlbY;
  std::vector<float> see_stateTrajGlbZ;
  std::vector<float> see_stateTrajGlbPx;
  std::vector<float> see_stateTrajGlbPy;
  std::vector<float> see_stateTrajGlbPz;
  std::vector<int> see_q;
  std::vector<std::vector<int>> see_hitIdx;
  std::vector<std::vector<int>> see_hitType;
};

// Inverts pix_simHitIdx (pixel-hit -> simhit indices) + simhit_simTrkIdx (simhit -> sim-track
// index) into sim-track -> pixel-hit-array-index, for the current event.
std::map<int, std::vector<int>> buildSimTrkToPixHitMap(std::vector<std::vector<int>> const& pix_simHitIdx,
                                                        std::vector<int> const& simhit_simTrkIdx);

// For every sim track with >=3 true pixel hits in distinct layers, except those listed in
// excludeSimTrkIdx, builds a synthetic, truth-derived pixel seed (quad if >=4 distinct layers,
// else triplet). Reads the current event's branches from the global `trk` (Trktree.h), same
// convention as the rest of code/core/. excludeSimTrkIdx defaults to empty (the --idealpls
// full-replacement behavior); pass the result of findSimTrkIdxsWithRealSeed() below to instead
// only fill in sim tracks CMSSW's real seeding produced nothing for (--fillmissingpls).
TruthPixelSeedVectors buildTruthPixelSeeds(std::set<int> const& excludeSimTrkIdx = {});

// Returns the set of sim-track indices that already have at least one real seed
// (see_hitIdx/see_hitType, as read from the input ntuple) truth-matching them at the standard
// 0.75 hit-purity threshold (matchedSimTrkIdxs(), trkCore.h) -- i.e. sim tracks CMSSW's real
// seeding did NOT fail for. Used to build the exclusion set for buildTruthPixelSeeds() in
// --fillmissingpls mode, so synthetic seeds are only injected for the genuine gap.
std::set<int> findSimTrkIdxsWithRealSeed(std::vector<std::vector<int>> const& see_hitIdx,
                                          std::vector<std::vector<int>> const& see_hitType,
                                          std::vector<int> const& simhit_simTrkIdx,
                                          std::vector<std::vector<int>> const& ph2_simHitIdx,
                                          std::vector<std::vector<int>> const& pix_simHitIdx);

#endif
