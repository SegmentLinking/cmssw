#ifndef RecoTracker_MkFitAlpaka_test_dupCleanTestIO_h
#define RecoTracker_MkFitAlpaka_test_dupCleanTestIO_h

// Record format shared by the MkFitCore duplicate-cleaner reference writer (testDupCleanMkFitCoreRef.cpp) and the
// port comparator (alpaka/testDupClean.dev.cc): per case, the cleaner with its IterationConfig values and beam spot,
// the input tracks, and which tracks MkFitCore keeps.

#include <cstdint>
#include <cstdio>
#include <vector>

namespace mkfitdev::dupcleantest {

  // The cleaner (mkfitdev::DuplicateCleaner value) and the IterationConfig dc_* values
  struct CaseConfig {
    int32_t cleaner;
    float fracSharedHits;
    float drthCentral;
    float drthObarrel;
    float drthForward;
    float fracSharedHitsLowPt;
    float lowPtRampStart;
    float lowPtRampEnd;
    float lowPtMaxRelDiffInvPt;
    int32_t lowPtRequireSameCharge;
    float lowPtMaxD0;
    float lowPtMaxDRTimesPt;
    int32_t minUniqueHitsToKeep;
    float minPtUniqueHitsToKeep;
    float bsX;
    float bsY;
    uint64_t pixelLayers[4];
  };

  struct Hit {
    int32_t index;
    int32_t layer;
  };

  struct TrackRec {
    float par[6];  // x, y, z, 1/pT, phi, theta
    int32_t charge;
    int32_t label;
    float score;
    int32_t nFoundHits;
    std::vector<Hit> hits;
    int32_t keep;  // MkFitCore output: the track survives the cleaner
  };

  struct Case {
    CaseConfig config;
    std::vector<TrackRec> tracks;
  };

  inline void writeCases(std::FILE* file, std::vector<Case> const& cases) {
    const int32_t nCases = cases.size();
    std::fwrite(&nCases, sizeof(nCases), 1, file);
    for (auto const& entry : cases) {
      std::fwrite(&entry.config, sizeof(CaseConfig), 1, file);
      const int32_t nTracks = entry.tracks.size();
      std::fwrite(&nTracks, sizeof(nTracks), 1, file);
      for (auto const& track : entry.tracks) {
        const int32_t nHits = track.hits.size();
        std::fwrite(track.par, sizeof(float), 6, file);
        std::fwrite(&track.charge, sizeof(int32_t), 1, file);
        std::fwrite(&track.label, sizeof(int32_t), 1, file);
        std::fwrite(&track.score, sizeof(float), 1, file);
        std::fwrite(&track.nFoundHits, sizeof(int32_t), 1, file);
        std::fwrite(&track.keep, sizeof(int32_t), 1, file);
        std::fwrite(&nHits, sizeof(nHits), 1, file);
        std::fwrite(track.hits.data(), sizeof(Hit), nHits, file);
      }
    }
  }

  // false on a truncated or malformed file
  inline bool readCases(std::FILE* file, std::vector<Case>& cases) {
    int32_t nCases = 0;
    if (std::fread(&nCases, sizeof(nCases), 1, file) != 1 || nCases < 0)
      return false;
    cases.resize(nCases);
    for (auto& entry : cases) {
      int32_t nTracks = 0;
      if (std::fread(&entry.config, sizeof(CaseConfig), 1, file) != 1 ||
          std::fread(&nTracks, sizeof(nTracks), 1, file) != 1 || nTracks < 0)
        return false;
      entry.tracks.resize(nTracks);
      for (auto& track : entry.tracks) {
        int32_t nHits = 0;
        bool ok = std::fread(track.par, sizeof(float), 6, file) == 6 &&
                  std::fread(&track.charge, sizeof(int32_t), 1, file) == 1 &&
                  std::fread(&track.label, sizeof(int32_t), 1, file) == 1 &&
                  std::fread(&track.score, sizeof(float), 1, file) == 1 &&
                  std::fread(&track.nFoundHits, sizeof(int32_t), 1, file) == 1 &&
                  std::fread(&track.keep, sizeof(int32_t), 1, file) == 1 &&
                  std::fread(&nHits, sizeof(nHits), 1, file) == 1 && nHits >= 0;
        if (!ok)
          return false;
        track.hits.resize(nHits);
        if (std::fread(track.hits.data(), sizeof(Hit), nHits, file) != size_t(nHits))
          return false;
      }
    }
    return true;
  }

}  // namespace mkfitdev::dupcleantest

#endif
