// MkFitCore reference for the duplicate-cleaner test: generates events of tracks with duplicates (shared hits, close
// directions, low and high pT, prompt and displaced, with and without pixel hits), runs the MkStdSeqs.cc cleaners on
// them with several IterationConfig settings and writes the inputs and the kept tracks for
// testMkFitAlpakaDupClean<Backend>. Also checks that the duplicate-cleaner constants of the port equal MkFitCore's.
// Usage: testMkFitAlpakaDupCleanMkFitCoreRef <out.bin> [nevents=40] [seed=2718]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "RecoTracker/MkFitCMS/interface/MkStdSeqs.h"
#include "RecoTracker/MkFitCore/interface/BeamSpot.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"

#include "dupCleanTestIO.h"

using namespace mkfitdev::dupcleantest;

namespace {

  constexpr int kNLayers = 30;
  constexpr int kNPixelLayers = 10;
  constexpr int kHitsPerLayer = 1000;

  struct Particle {
    float pt, eta, phi0, offset;
    int charge;
    std::vector<Hit> hits;
  };

  // A state on the helix of the particle (center as TrackBase::d0BeamSpot) after a path of the given length from its
  // point of closest approach to the origin, which is 'offset' away from the origin.
  TrackRec stateAt(Particle const& particle, float pathLength) {
    const float k = ((particle.charge < 0) ? 100.0f : -100.0f) / (mkfit::Const::sol * mkfit::Config::Bfield);
    const float radius = k * particle.pt;  // signed
    const float pcaX = -particle.offset * std::sin(particle.phi0), pcaY = particle.offset * std::cos(particle.phi0);
    const float centerX = pcaX - radius * std::sin(particle.phi0), centerY = pcaY + radius * std::cos(particle.phi0);
    const float phi = particle.phi0 + pathLength / radius;
    TrackRec track{};
    track.par[0] = centerX + radius * std::sin(phi);
    track.par[1] = centerY - radius * std::cos(phi);
    track.par[2] = pathLength * std::sinh(particle.eta);
    track.par[3] = 1.f / particle.pt;
    track.par[4] = mkfit::squashPhiMinimal(phi);
    track.par[5] = 2.f * std::atan(std::exp(-particle.eta));
    track.charge = particle.charge;
    track.hits = particle.hits;
    return track;
  }

  std::vector<TrackRec> makeEvent(std::mt19937& rng, int nParticles) {
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::vector<TrackRec> tracks;
    int label = 0;
    for (int ip = 0; ip < nParticles; ++ip) {
      Particle particle;
      particle.pt = 0.6f * std::pow(100.f, uni(rng));  // 0.6 - 60 GeV, log-uniform
      particle.eta = -3.f + 6.f * uni(rng);
      particle.phi0 = -3.14159265f + 6.2831853f * uni(rng);
      particle.offset = (uni(rng) < 0.8f ? 0.5f : 5.f) * (2.f * uni(rng) - 1.f);
      particle.charge = uni(rng) < 0.5f ? -1 : 1;
      // layers in order: 0 or 2-4 pixel layers, then strip layers
      const int nPixel = uni(rng) < 0.2f ? 0 : 2 + int(3 * uni(rng));
      const int nHits = nPixel + 4 + int(26 * uni(rng));
      int layer = int(3 * uni(rng));
      for (int ih = 0; ih < nHits && layer < kNLayers; ++ih) {
        if (ih == nPixel && layer < kNPixelLayers)
          layer = kNPixelLayers;
        particle.hits.push_back({uni(rng) < 0.05f ? -1 : int(kHitsPerLayer * uni(rng)), layer});
        layer += 1 + (uni(rng) < 0.2f ? 1 : 0);
      }
      TrackRec first = stateAt(particle, 3.f + 40.f * uni(rng));
      first.label = label++;
      first.score = 100.f * uni(rng);
      tracks.push_back(first);
      // duplicates: a fraction of the hits replaced, a perturbed state, rarely the same label / score / other charge
      const int nDup = uni(rng) < 0.5f ? 1 + int(3 * uni(rng)) : 0;
      for (int id = 0; id < nDup; ++id) {
        Particle copy = particle;
        copy.pt *= 1.f + 0.1f * gauss(rng);
        copy.eta += 0.01f * gauss(rng);
        copy.phi0 += 0.01f * gauss(rng);
        copy.offset += 0.3f * gauss(rng);
        if (uni(rng) < 0.05f)
          copy.charge = -copy.charge;
        const float replace = uni(rng);
        for (auto& hit : copy.hits)
          if (uni(rng) < replace)
            hit.index = uni(rng) < 0.1f ? -1 : int(kHitsPerLayer * uni(rng));
        if (uni(rng) < 0.3f && copy.hits.size() > 4)
          copy.hits.resize(copy.hits.size() - 1 - int(3 * uni(rng)));
        TrackRec dup = stateAt(copy, 3.f + 40.f * uni(rng));
        dup.label = uni(rng) < 0.03f ? first.label : label++;
        dup.score = uni(rng) < 0.05f ? first.score : 100.f * uni(rng);
        tracks.push_back(dup);
      }
    }
    std::shuffle(tracks.begin(), tracks.end(), rng);
    for (auto& track : tracks) {
      track.nFoundHits = 0;
      for (auto const& hit : track.hits)
        track.nFoundHits += hit.index >= 0;
    }
    return tracks;
  }

  // the IterationConfig settings of the cases
  std::vector<CaseConfig> caseConfigs() {
    using mkfitdev::DuplicateCleaner;
    // the pT-adaptive cleaner, values like the LST step's
    CaseConfig ptAdaptive{};
    ptAdaptive.cleaner = int32_t(DuplicateCleaner::SharedHitsPixelPriorityPtAdaptive);
    ptAdaptive.fracSharedHits = 0.24f;
    ptAdaptive.fracSharedHitsLowPt = 0.10f;
    ptAdaptive.lowPtRampStart = 5.f;
    ptAdaptive.lowPtRampEnd = 10.f;
    ptAdaptive.lowPtMaxRelDiffInvPt = 0.10f;
    ptAdaptive.lowPtRequireSameCharge = 1;
    ptAdaptive.lowPtMaxD0 = 2.f;
    ptAdaptive.lowPtMaxDRTimesPt = 0.10f;
    ptAdaptive.minUniqueHitsToKeep = 9;
    ptAdaptive.minPtUniqueHitsToKeep = 10.f;
    ptAdaptive.bsX = 0.01f;
    ptAdaptive.bsY = -0.02f;
    for (int layer = 0; layer < kNPixelLayers; ++layer)
      ptAdaptive.pixelLayers[layer >> 6] |= 1ull << (layer & 63);
    // the phase-1 pixelseed cleaner with a direction-only test
    CaseConfig pixelSeed = ptAdaptive;
    pixelSeed.cleaner = int32_t(DuplicateCleaner::SharedHitsPixelSeed);
    pixelSeed.drthCentral = pixelSeed.drthObarrel = pixelSeed.drthForward = 0.001f;
    CaseConfig pixelPriority = ptAdaptive;
    pixelPriority.cleaner = int32_t(DuplicateCleaner::SharedHitsPixelPriority);
    // the pT-adaptive cleaner with a direction-only test (every pair compared), with its conditions off and with
    // negative dR thresholds (only pairs that share a hit, but a direction-only test) and no low-pT fraction
    CaseConfig directionTest = ptAdaptive;
    directionTest.drthCentral = 0.01f;
    directionTest.drthObarrel = 0.02f;
    directionTest.drthForward = 0.03f;
    CaseConfig conditionsOff = ptAdaptive;
    conditionsOff.fracSharedHitsLowPt = 0.05f;
    conditionsOff.lowPtMaxRelDiffInvPt = 0.f;
    conditionsOff.lowPtRequireSameCharge = 0;
    conditionsOff.lowPtMaxD0 = 0.f;
    conditionsOff.lowPtMaxDRTimesPt = 0.f;
    conditionsOff.minUniqueHitsToKeep = 0;
    CaseConfig negativeDR = ptAdaptive;
    negativeDR.drthCentral = negativeDR.drthObarrel = negativeDR.drthForward = -0.02f;
    negativeDR.fracSharedHitsLowPt = -1.f;
    return {pixelSeed, pixelPriority, ptAdaptive, directionTest, conditionsOff, negativeDR};
  }

  // MkFitCore's cleaner on the tracks; keep = the track survives (chi2 carries the index through remove_duplicates)
  void runMkFitCore(CaseConfig const& config, mkfit::TrackerInfo const& trackerInfo, std::vector<TrackRec>& tracks) {
    mkfit::IterationConfig itconf;
    itconf.dc_fracSharedHits = config.fracSharedHits;
    itconf.dc_drth_central = config.drthCentral;
    itconf.dc_drth_obarrel = config.drthObarrel;
    itconf.dc_drth_forward = config.drthForward;
    itconf.dc_fracSharedHitsLowPt = config.fracSharedHitsLowPt;
    itconf.dc_lowPtRampStart = config.lowPtRampStart;
    itconf.dc_lowPtRampEnd = config.lowPtRampEnd;
    itconf.dc_lowPtMaxRelDiffInvPt = config.lowPtMaxRelDiffInvPt;
    itconf.dc_lowPtRequireSameCharge = config.lowPtRequireSameCharge != 0;
    itconf.dc_lowPtMaxD0 = config.lowPtMaxD0;
    itconf.dc_lowPtMaxDRTimesPt = config.lowPtMaxDRTimesPt;
    itconf.dc_minUniqueHitsToKeep = config.minUniqueHitsToKeep;
    itconf.dc_minPtUniqueHitsToKeep = config.minPtUniqueHitsToKeep;
    const mkfit::BeamSpot beamSpot(config.bsX, config.bsY, 0.f, 5.f, 5e-4f, 5e-4f, 0.f, 0.f);

    mkfit::TrackVec input;
    for (size_t i = 0; i < tracks.size(); ++i) {
      TrackRec const& rec = tracks[i];
      mkfit::TrackState state;
      for (int k = 0; k < 6; ++k)
        state.parameters[k] = rec.par[k];
      state.charge = rec.charge;
      std::vector<mkfit::HitOnTrack> hots;
      for (auto const& hit : rec.hits)
        hots.push_back({hit.index, hit.layer});
      mkfit::Track track(state, float(i), rec.label, hots.size(), hots.data());
      track.setScore(rec.score);
      input.push_back(track);
    }
    using mkfitdev::DuplicateCleaner;
    switch (DuplicateCleaner(config.cleaner)) {
      case DuplicateCleaner::SharedHitsPixelSeed:
        mkfit::StdSeq::clean_duplicates_sharedhits_pixelseed(input, itconf, trackerInfo, beamSpot);
        break;
      case DuplicateCleaner::SharedHitsPixelPriority:
        mkfit::StdSeq::clean_duplicates_sharedhits_pixelpriority(input, itconf, trackerInfo, beamSpot);
        break;
      case DuplicateCleaner::SharedHitsPixelPriorityPtAdaptive:
        mkfit::StdSeq::clean_duplicates_sharedhits_pixelpriority_ptadaptive(input, itconf, trackerInfo, beamSpot);
        break;
      default:
        std::fprintf(stderr, "unexpected cleaner %d\n", config.cleaner);
        std::exit(2);
    }
    for (auto& rec : tracks)
      rec.keep = 0;
    for (auto const& track : input)
      tracks[int(track.chi2())].keep = 1;
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <out.bin> [nevents=40] [seed=2718]\n", argv[0]);
    return 1;
  }
  const int nEvents = argc > 2 ? std::atoi(argv[2]) : 40;
  std::mt19937 rng(argc > 3 ? std::atoi(argv[3]) : 2718);

  // the port's copies of the cleaner constants (interface/math/Config.h)
  const bool sameConstants =
      mkfit::Config::maxd1pt == mkfitdev::Config::maxd1pt && mkfit::Config::maxdphi == mkfitdev::Config::maxdphi &&
      mkfit::Config::maxdcth == mkfitdev::Config::maxdcth && mkfit::Config::maxcth_ob == mkfitdev::Config::maxcth_ob &&
      mkfit::Config::maxcth_fw == mkfitdev::Config::maxcth_fw &&
      mkfit::Config::mindeta_p2 == mkfitdev::Config::mindeta_p2 &&
      mkfit::Config::maxeta_ob_p2 == mkfitdev::Config::maxeta_ob_p2 &&
      mkfit::Config::maxeta_fw_p2 == mkfitdev::Config::maxeta_fw_p2;
  if (!sameConstants) {
    std::fprintf(stderr, "FAIL: duplicate-cleaner constants of interface/math/Config.h differ from MkFitCore's\n");
    return 3;
  }

  mkfit::TrackerInfo trackerInfo;
  trackerInfo.create_layers(kNLayers, 0, 0);
  for (int layer = 0; layer < kNLayers; ++layer)
    trackerInfo.layer_nc(layer).set_is_pixel(layer < kNPixelLayers);

  const std::vector<CaseConfig> configs = caseConfigs();
  std::vector<Case> cases;
  int nTracks = 0, nKept = 0;
  for (int event = 0; event < nEvents; ++event) {
    const std::vector<TrackRec> tracks = makeEvent(rng, 100 + int(rng() % 200));
    for (auto const& config : configs) {
      Case entry{config, tracks};
      runMkFitCore(config, trackerInfo, entry.tracks);
      for (auto const& track : entry.tracks) {
        ++nTracks;
        nKept += track.keep;
      }
      cases.push_back(std::move(entry));
    }
  }
  std::FILE* file = std::fopen(argv[1], "wb");
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  writeCases(file, cases);
  std::fclose(file);
  std::printf("testMkFitAlpakaDupCleanMkFitCoreRef: %d cases, %d tracks, %d kept by MkFitCore\n",
              int(cases.size()),
              nTracks,
              nKept);
  return 0;
}
