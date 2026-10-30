// Runs the device duplicate cleaner of MkFitAlpaka (CleanAlgo::flagDuplicates) on the cases written by
// testMkFitAlpakaDupCleanMkFitCoreRef and compares its keep / remove decisions with MkFitCore's.
// Usage: testMkFitAlpakaDupClean<Backend> <mkfitcore_ref.bin>

#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/tracks/TrackSoADeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/DupCleanParams.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/clean/CleanAlgo.h"
#include "RecoTracker/MkFitAlpaka/test/dupCleanTestIO.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ::mkfitdev::dupcleantest;
namespace mkd = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev;

namespace {

  // the cleaner parameters as the build module makes them from the ES configuration
  ::mkfitdev::clean::DupCleanParams makeParams(CaseConfig const& config) {
    ::mkfitdev::ESConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.duplicate_cleaner = ::mkfitdev::DuplicateCleaner(config.cleaner);
    cfg.dc_fracSharedHits = config.fracSharedHits;
    cfg.dc_drth_central = config.drthCentral;
    cfg.dc_drth_obarrel = config.drthObarrel;
    cfg.dc_drth_forward = config.drthForward;
    cfg.dc_fracSharedHitsLowPt = config.fracSharedHitsLowPt;
    cfg.dc_lowPtRampStart = config.lowPtRampStart;
    cfg.dc_lowPtRampEnd = config.lowPtRampEnd;
    cfg.dc_lowPtMaxRelDiffInvPt = config.lowPtMaxRelDiffInvPt;
    cfg.dc_lowPtRequireSameCharge = config.lowPtRequireSameCharge != 0;
    cfg.dc_lowPtMaxD0 = config.lowPtMaxD0;
    cfg.dc_lowPtMaxDRTimesPt = config.lowPtMaxDRTimesPt;
    cfg.dc_minUniqueHitsToKeep = config.minUniqueHitsToKeep;
    cfg.dc_minPtUniqueHitsToKeep = config.minPtUniqueHitsToKeep;
    for (int word = 0; word < 4; ++word)
      cfg.pixel_layer_mask[word] = config.pixelLayers[word];
    ::mkfitdev::clean::DupCleanParams params = ::mkfitdev::clean::makeDupCleanParams(cfg);
    params.bsX = config.bsX;
    params.bsY = config.bsY;
    return params;
  }

  const char* cleanerName(int cleaner) {
    switch (::mkfitdev::DuplicateCleaner(cleaner)) {
      case ::mkfitdev::DuplicateCleaner::SharedHitsPixelSeed:
        return "phase1:clean_duplicates_sharedhits_pixelseed";
      case ::mkfitdev::DuplicateCleaner::SharedHitsPixelPriority:
        return "phase2:clean_duplicates_sharedhits_pixelpriority";
      case ::mkfitdev::DuplicateCleaner::SharedHitsPixelPriorityPtAdaptive:
        return "phase2:clean_duplicates_sharedhits_pixelpriority_ptadaptive";
      default:
        return "?";
    }
  }

  struct Counts {
    int cases = 0;
    int tracks = 0;
    int removed = 0;
    int mismatches = 0;
  };

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <mkfitcore_ref.bin>\n", argv[0]);
    return 1;
  }
  const auto& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::printf("SKIP %s: no device\n", EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE));
    return 0;
  }
  std::FILE* file = std::fopen(argv[1], "rb");
  std::vector<Case> cases;
  if (!file || !readCases(file, cases)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  std::fclose(file);

  Queue queue(devices[0]);
  std::map<int, Counts> counts;  // per cleaner
  int nMismatchTotal = 0;
  for (auto const& entry : cases) {
    const int n = entry.tracks.size();
    ::mkfitdev::TrackSoAHostCollection host(queue, n);
    std::memset(host.buffer().data(), 0, alpaka::getExtentProduct(host.buffer()));
    auto view = host.view();
    view.nTracks() = n;
    for (int i = 0; i < n; ++i) {
      TrackRec const& rec = entry.tracks[i];
      if (rec.hits.size() > size_t(::mkfitdev::kMaxTrkHits)) {
        std::fprintf(stderr, "track with %d hits > kMaxTrkHits\n", int(rec.hits.size()));
        return 1;
      }
      for (int k = 0; k < 6; ++k)
        view[i].params().v[k] = rec.par[k];
      view[i].charge() = rec.charge;
      view[i].label() = rec.label;
      view[i].score() = rec.score;
      view[i].nTotalHits() = rec.hits.size();
      view[i].nFoundHits() = rec.nFoundHits;
      for (size_t h = 0; h < rec.hits.size(); ++h) {
        view[i].hits().hot[h].index = rec.hits[h].index;
        view[i].hits().hot[h].layer = rec.hits[h].layer;
      }
    }
    mkd::TrackSoADeviceCollection device(queue, n);
    alpaka::memcpy(queue, device.buffer(), host.buffer());
    mkd::clean::CleanAlgo algo(queue, n);
    algo.flagDuplicates(queue, device.view(), makeParams(entry.config));
    alpaka::memcpy(queue, host.buffer(), device.buffer());
    alpaka::wait(queue);

    Counts& count = counts[entry.config.cleaner];
    ++count.cases;
    for (int i = 0; i < n; ++i) {
      const bool removed = host.const_view()[i].duplicate() != 0;
      ++count.tracks;
      count.removed += removed;
      if (removed == (entry.tracks[i].keep != 0)) {
        ++count.mismatches;
        ++nMismatchTotal;
      }
    }
  }
  for (auto const& [cleaner, count] : counts)
    std::printf("%-60s cases %4d tracks %7d removed %6d decisions differing from MkFitCore %d\n",
                cleanerName(cleaner),
                count.cases,
                count.tracks,
                count.removed,
                count.mismatches);
  std::printf("%s %s: %d decision(s) differ from MkFitCore\n",
              nMismatchTotal ? "FAIL" : "PASS",
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE),
              nMismatchTotal);
  return nMismatchTotal ? 3 : 0;
}
