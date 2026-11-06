// Unit test of the kernels of the device LST input (RecoTracker/LST/plugins/alpaka/LSTInputDeviceKernels.h), run end
// to end through their launchers in the order of LSTInputDeviceProducer: prepareTracks, fitAllTracks (the
// inner-tracker-only fits and the full fits of 4, 6, 8, 10, 12 and 14 hits), makePixelSegments, indexPixelSegments,
// fillLSTInput, fillSeedStates. The inputs are small synthetic events in the producer's input formats: a pixel rechit
// SoA (positions relative to the beam spot), an OT rechit SoA, the CA's OT module starts and the first OT rechit of
// each CA module, a pixel-track SoA with its hit offsets and hit indices, the map to the EDM tracks, the module frames
// and a tilted beam spot. The tracks are noiseless helices (pixel only, pixel plus OT, more than 14 hits,
// inner-tracker-only candidates with 3 and fewer inner-tracker hits), tracks without an EDM track or with an
// unsupported hit count, and tracks made to fail each check of the fitted state.
//
// Checked: the hit gathering against the input rows and a host computation of the covariances; the size of each full
// fit and its list, one list per fit size; the states against the noiseless truth and their statuses; the hit
// selection against the per-track functions run in a test kernel on the same hits, and the states and pLS against
// them run on the gathered fit inputs (states with identical bits, the closest approach within the tolerances of two
// compilations of it); the seed, pLS and hit indices; the LST input and the seed-state product with every counter;
// events without pixel tracks, without OT hits, and without any input; and the independence of every per-track
// result from the order of the tracks.

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LST/interface/LSTSeedFitModulesHost.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/PixelSeedStatesHostCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/PixelSeedStatesDeviceCollection.h"
#include "RecoTracker/LST/test/BLFTestStates.h"
#include "RecoTracker/LST/test/alpaka/LSTInputDeviceTestTracks.h"

// The launchers are compiled into the plugin library of the producer, which a test binary cannot link: their sources
// are compiled here.
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceKernels.dev.cc"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceFit03.dev.cc"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceFit04.dev.cc"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceFit06.dev.cc"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceFit08.dev.cc"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceFit10.dev.cc"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceFit12.dev.cc"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceFit14.dev.cc"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDeviceTest;

  using ::lstInputDevice::Counters;
  using ::lstInputDevice::kITFitList;
  using ::lstInputDevice::kNumFitLists;
  using ::lstInputDevice::Totals;

  template <typename T>
  using DeviceCollection = ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>;

  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

  // ---------------------------------------------------------------------------------------------------------------
  // The tracks of the test events

  // a pixel track: its hits inside out (global positions, module frames, local variances) and what it tests
  struct EventTrack {
    std::string label;
    std::vector<TestHit> hits;
    GeneratedTrack generated;  // the truth, when the hits are a noiseless generated helix
    bool hasTruth = false;
    bool hasEdmTrack = true;
    int sharesHitsOf = -1;  // the track whose input rows this track uses, as pixel tracks sharing their hits
    // the statuses of the full fit and of the inner-tracker-only fit, when they run
    SeedStateStatus expectedStatus = SeedStateStatus::kOk;
    SeedStateStatus expectedITStatus = SeedStateStatus::kOk;
    // expectedStatus holds only when every hit is fitted; with fewer, the status is the per-track functions'
    bool statusOfAllHits = false;
  };

  // the prompt helix from the tilted beam line at z = beamSpotZ + dz
  Helix prompt(Settings const& settings, double pt, int charge, double phi0, double cot, double dz = 0.) {
    return Helix{{settings.beamSpotX + settings.beamSlopeX * dz,
                  settings.beamSpotY + settings.beamSlopeY * dz,
                  settings.beamSpotZ + dz},
                 phi0,
                 pt,
                 cot,
                 charge};
  }

  struct TrackBuilder {
    Settings settings;
    double bField;
    std::vector<EventTrack> tracks;
    std::vector<std::string> incomplete;  // helices that missed a surface of their layout: a failure

    EventTrack& add(std::string const& label, Helix const& helix, std::vector<Surface> const& surfaces) {
      EventTrack track;
      track.label = label;
      track.generated = generate(helix, surfaces, bField, nullptr);
      if (!track.generated.complete)
        incomplete.push_back(label);
      track.hits = track.generated.hits;
      track.hasTruth = true;
      tracks.push_back(track);
      return tracks.back();
    }
  };

  std::string describe(std::string const& what, double pt, int q, double cot) {
    return what + " pt=" + std::to_string(pt).substr(0, 4) + " q=" + std::to_string(q) +
           " cot=" + std::to_string(cot).substr(0, 5);
  }

  // inner-tracker barrel layers, then outer-tracker barrel layers
  std::vector<Surface> itLayout(int nIT, int nOT) {
    std::vector<Surface> surfaces;
    for (int i = 0; i < nIT; ++i)
      surfaces.push_back(pixelBarrel(kPixelBarrelRadii[i]));
    for (int i = 0; i < nOT; ++i)
      surfaces.push_back(otBarrel(kOTBarrelRadii[i]));
    return surfaces;
  }

  // nHits barrel layers evenly from r = 3 cm to r = 108 cm, pixel below 20 cm
  std::vector<Surface> denseLayout(int nHits) {
    std::vector<Surface> surfaces;
    for (int i = 0; i < nHits; ++i) {
      const double r = 3. + 105. * i / (nHits - 1);
      surfaces.push_back(r < 20. ? pixelBarrel(r) : otBarrel(r));
    }
    return surfaces;
  }

  std::vector<EventTrack> buildTracks(TrackBuilder& builder) {
    std::mt19937 rng(20261019);
    std::uniform_real_distribution<double> uniform(0., 1.);
    Settings const& settings = builder.settings;
    auto phi = [&]() { return 2. * kPi * uniform(rng) - kPi; };

    // pixel only: four barrel layers; two barrel layers and four discs
    for (double pt : {1.5, 20.})
      for (int q : {-1, 1}) {
        builder.add(
            describe("pixel only, 4 barrel", pt, q, 0.3), prompt(settings, pt, q, phi(), 0.3), centralLayout(4));
        builder.add(describe("pixel only, 2 barrel + 4 discs", pt, q, 2.5),
                    prompt(settings, pt, q, phi(), 2.5),
                    forwardLayout(6, 2.5));
      }
    // pixel plus OT, every full-fit size of 5 to 14 hits
    for (int n = 5; n <= 14; ++n)
      for (double pt : {0.9, 4.}) {
        const int q = n % 2 ? 1 : -1;
        const double cot = n % 3 ? 0.4 : -0.7;
        builder.add(describe("pixel + OT n=" + std::to_string(n), pt, q, cot),
                    prompt(settings, pt, q, phi(), cot, 6. * uniform(rng) - 3.),
                    centralLayout(n));
      }
    for (int n : {8, 11})
      builder.add(describe("pixel + OT forward n=" + std::to_string(n), 3., 1, -2.5),
                  prompt(settings, 3., 1, phi(), -2.5),
                  forwardLayout(n, -2.5));
    // more than 14 hits: the stride
    for (int n : {15, 20, 29, 64})
      builder.add(describe("dense n=" + std::to_string(n), 10., -1, 0.2),
                  prompt(settings, 10., -1, phi(), 0.2),
                  denseLayout(n));
    // below the pT cut: a seed without a pLS
    builder.add(describe("below the pT cut n=7", 0.5, 1, 0.2), prompt(settings, 0.5, 1, phi(), 0.2), centralLayout(7));
    // large turns to the last hit, where the closest approach may fail
    for (int q : {-1, 1})
      builder.add(describe("large turn n=10", 0.85, q, 0.1), prompt(settings, 0.85, q, phi(), 0.1), centralLayout(10));
    // inner-tracker-only candidates: last hit in the OT and at most three inner-tracker hits
    for (int nOT = 1; nOT <= 6; ++nOT)
      builder.add(describe("3 IT + " + std::to_string(nOT) + " OT", 2., nOT % 2 ? 1 : -1, 0.4),
                  prompt(settings, 2., nOT % 2 ? 1 : -1, phi(), 0.4),
                  itLayout(3, nOT));
    builder.add(describe("2 IT + 4 OT", 4., 1, 0.4), prompt(settings, 4., 1, phi(), 0.4), itLayout(2, 4));
    builder.add(describe("1 IT + 4 OT", 4., -1, 0.4), prompt(settings, 4., -1, phi(), 0.4), itLayout(1, 4));
    builder.add(describe("0 IT + 5 OT", 4., 1, 0.4), prompt(settings, 4., 1, phi(), 0.4), itLayout(0, 5));
    builder.add(describe("4 IT + 2 OT", 4., -1, 0.4), prompt(settings, 4., -1, phi(), 0.4), itLayout(4, 2));

    // not fitted: no EDM track, 3 hits, 256 hits
    builder.add("no EDM track", prompt(settings, 4., 1, phi(), 0.2), centralLayout(8)).hasEdmTrack = false;
    builder.add("3 hits", prompt(settings, 4., 1, phi(), 0.2), centralLayout(3));
    {
      std::vector<Surface> many;
      for (int i = 0; i < 256; ++i)
        many.push_back(otBarrel(3. + 105. * i / 255.));
      builder.add("256 hits", prompt(settings, 10., 1, phi(), 0.2), many);
    }
    // a track sharing every hit of another track
    {
      const int original = 10;
      EventTrack copy = builder.tracks[original];
      copy.label = "the hits of " + copy.label;
      copy.sharesHitsOf = original;
      builder.tracks.push_back(copy);
    }

    // the checks of the state
    {
      EventTrack& track = builder.add("NaN pixel hit position", prompt(settings, 4., -1, phi(), 0.2), centralLayout(8));
      track.hits[2].x = kNaN;
      track.expectedStatus = SeedStateStatus::kNonFinite;
    }
    {
      // negative local variances make every hit weight negative: the solve stays finite, the variances do not stay
      // positive
      EventTrack& track = builder.add("negative hit variances", prompt(settings, 20., 1, phi(), 0.2), centralLayout(8));
      for (auto& hit : track.hits) {
        hit.varX = -hit.varX;
        hit.varY = -hit.varY;
      }
      track.expectedStatus = SeedStateStatus::kNonPositiveVariance;
    }
    {
      // A zigzag: the three hits that set the fit's charge (first, middle, last) bend one way by 1 um, the two others
      // lie 50 um on the other side; with 100 um errors and the 20 GeV multiple scattering, the fitted curvature has
      // the opposite sign. The whole track is rotated about the beam axis by an azimuth that is no multiple of 45
      // degrees, so that no fitted quantity the test compares bit for bit is an exact cancellation (a coordinate that
      // is exactly 0 in a symmetric position has a rounding residue that depends on the compilation of the fit).
      EventTrack track;
      track.label = "zigzag";
      track.expectedStatus = SeedStateStatus::kChargeMismatch;
      track.statusOfAllHits = true;
      const double xs[5] = {3., 7., 11., 15., 19.}, ys[5] = {0., -50e-4, 1e-4, -50e-4, 0.};
      const double rotation = 0.6;  // rad, about 34.4 degrees
      for (int i = 0; i < 5; ++i) {
        TestHit hit{};
        hit.x = std::cos(rotation) * xs[i] - std::sin(rotation) * ys[i];
        hit.y = std::sin(rotation) * xs[i] + std::cos(rotation) * ys[i];
        hit.z = 0.1 * xs[i];
        const double azimuth = std::atan2(hit.y, hit.x);
        const float axes[9] = {float(-std::sin(azimuth)),
                               float(std::cos(azimuth)),
                               0.f,
                               0.f,
                               0.f,
                               1.f,
                               float(std::cos(azimuth)),
                               float(std::sin(azimuth)),
                               0.f};
        std::copy(axes, axes + 9, hit.rotation);
        hit.origin[0] = hit.x;
        hit.origin[1] = hit.y;
        hit.origin[2] = hit.z;
        hit.varX = hit.varY = 1e-4;
        hit.detId = ::lst::kPixelModuleId;
        hit.clusterSize = 1;
        track.hits.push_back(hit);
      }
      builder.tracks.push_back(track);
    }
    {
      // a NaN on an inner-tracker hit: both fits non-finite
      EventTrack& track = builder.add("3 IT + 3 OT, NaN IT hit", prompt(settings, 4., 1, phi(), 0.4), itLayout(3, 3));
      track.hits[1].z = kNaN;
      track.expectedStatus = SeedStateStatus::kNonFinite;
      track.expectedITStatus = SeedStateStatus::kNonFinite;
    }
    {
      // a NaN on an OT hit: the full fit non-finite, the inner-tracker-only fit runs regardless and succeeds
      EventTrack& track = builder.add("3 IT + 3 OT, NaN OT hit", prompt(settings, 4., -1, phi(), 0.4), itLayout(3, 3));
      track.hits[4].y = kNaN;
      track.expectedStatus = SeedStateStatus::kNonFinite;
    }
    return builder.tracks;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // The input rows of an event

  // where a hit of a track is in the inputs
  struct HitRow {
    bool isOT;
    uint32_t row;  // pixel rechit row or OT rechit row
  };

  // the rows of the hit inputs, shared by the events made of the same tracks in different orders
  struct EventHits {
    std::vector<TestHit> pixelRows;  // x, y, z relative to the beam spot, as the pixel rechits
    std::vector<TestHit> otRows;     // global x, y, z
    std::vector<uint32_t> pixelModule, otModule;
    std::vector<TestHit> moduleFrames;  // the frame of each module row (origin and rotation)
    std::vector<uint32_t> caModuleStart, caRecHitStart;
    std::vector<std::vector<HitRow>> trackRows;   // per track, per hit
    std::vector<std::vector<uint32_t>> trackIds;  // per track, per hit: the CA hit index
    // per track, per hit: the hit as the fit is to see it (positions, frame, variances, row index, detId, cluster
    // size)
    std::vector<std::vector<TestHit>> expected;
  };

  // The rows of the pixel and OT rechits of the tracks, and with `noise` unused rows, unused and empty OT modules,
  // in shuffled orders: the pixel rows, the OT modules in the rechits, the CA modules and the module frames. Each hit
  // of a track is on a module of its own; an OT module of the CA has its track hit among up to two other rechits.
  EventHits buildHits(std::vector<EventTrack> const& tracks, Settings const& settings, std::mt19937& rng, bool noise) {
    EventHits hits;
    std::uniform_int_distribution<int> extra(0, 2);
    std::uniform_int_distribution<int> clusterSize(1, 6);
    hits.trackRows.resize(tracks.size());

    // the modules (frames), in creation order; shuffled at the end
    std::vector<TestHit> frames;

    // pixel rows: the hits of the tracks, then unused hits near them, then shuffled
    struct PixelEntry {
      TestHit hit;
      uint32_t module;
      int track, position;  // -1 for unused rows
    };
    std::vector<PixelEntry> pixel;
    for (std::size_t t = 0; t < tracks.size(); ++t) {
      if (tracks[t].sharesHitsOf >= 0)
        continue;
      for (std::size_t k = 0; k < tracks[t].hits.size(); ++k)
        if (!tracks[t].hits[k].isOT) {
          frames.push_back(tracks[t].hits[k]);
          pixel.push_back({tracks[t].hits[k], uint32_t(frames.size() - 1), int(t), int(k)});
        }
    }
    const std::size_t nTrackPixel = pixel.size();
    for (std::size_t i = 0; noise && nTrackPixel > 0 && i < nTrackPixel / 3 + 2; ++i) {
      PixelEntry entry = pixel[(7 * i) % nTrackPixel];
      entry.hit.z += 0.3f;
      entry.track = entry.position = -1;
      pixel.push_back(entry);
    }
    std::shuffle(pixel.begin(), pixel.end(), rng);

    // OT modules: one per OT hit of a track, with up to two other rechits; unused modules; empty CA modules
    struct OTModule {
      std::vector<TestHit> rows;
      std::vector<std::pair<int, int>> owners;  // (track, position) per row, (-1, -1) if unused
      uint32_t frame;
      bool ca;
    };
    std::vector<OTModule> modules;
    uint32_t nextDetId = 0x14000000u;
    auto rechit = [&](TestHit hit, float shift) {
      for (int k = 0; k < 3; ++k)
        (k == 0 ? hit.x : k == 1 ? hit.y : hit.z) += shift * hit.rotation[k];
      hit.clusterSize = clusterSize(rng);
      return hit;
    };
    for (std::size_t t = 0; t < tracks.size(); ++t) {
      if (tracks[t].sharesHitsOf >= 0)
        continue;
      for (std::size_t k = 0; k < tracks[t].hits.size(); ++k) {
        if (!tracks[t].hits[k].isOT)
          continue;
        frames.push_back(tracks[t].hits[k]);
        OTModule module{{}, {}, uint32_t(frames.size() - 1), true};
        const int nRows = 1 + (noise ? extra(rng) : 0);
        const int trackRow = std::uniform_int_distribution<int>(0, nRows - 1)(rng);
        const uint32_t detId = nextDetId;
        nextDetId += 4;
        for (int row = 0; row < nRows; ++row) {
          TestHit hit = row == trackRow ? tracks[t].hits[k] : rechit(tracks[t].hits[k], 0.05f * (row + 1));
          if (row == trackRow)
            hit.clusterSize = clusterSize(rng);
          hit.detId = detId;
          module.rows.push_back(hit);
          module.owners.emplace_back(row == trackRow ? int(t) : -1, row == trackRow ? int(k) : -1);
        }
        modules.push_back(module);
      }
    }
    const std::size_t nTrackModules = modules.size();
    for (std::size_t i = 0; noise && nTrackModules > 0 && i < nTrackModules / 4 + 2; ++i) {
      // an OT module outside the CA, with two rechits
      OTModule const& near = modules[(5 * i) % nTrackModules];
      OTModule module{{}, {}, near.frame, false};
      for (int row = 0; row < 2; ++row) {
        TestHit hit = rechit(near.rows.front(), -0.07f * (row + 1));
        hit.detId = nextDetId;
        module.rows.push_back(hit);
        module.owners.emplace_back(-1, -1);
      }
      nextDetId += 4;
      modules.push_back(module);
    }
    for (std::size_t i = 0; noise && i < 3; ++i)
      modules.push_back(OTModule{{}, {}, 0, true});  // an empty CA module
    std::shuffle(modules.begin(), modules.end(), rng);

    // the module frames, shuffled
    std::vector<uint32_t> frameRow(frames.size());
    std::iota(frameRow.begin(), frameRow.end(), 0u);
    std::shuffle(frameRow.begin(), frameRow.end(), rng);
    hits.moduleFrames.resize(frames.size() + (noise ? 2 : 0), frames.empty() ? TestHit{} : frames.front());
    for (std::size_t m = 0; m < frames.size(); ++m)
      hits.moduleFrames[frameRow[m]] = frames[m];

    // the pixel rechits; the beam spot is subtracted as for the pixel rechits
    for (auto const& entry : pixel) {
      TestHit row = entry.hit;
      row.x = entry.hit.x - settings.beamSpotX;
      row.y = entry.hit.y - settings.beamSpotY;
      row.z = entry.hit.z - settings.beamSpotZ;
      if (entry.track >= 0)
        hits.trackRows[entry.track].resize(tracks[entry.track].hits.size());
      if (entry.track >= 0)
        hits.trackRows[entry.track][entry.position] = HitRow{false, uint32_t(hits.pixelRows.size())};
      hits.pixelRows.push_back(row);
      hits.pixelModule.push_back(frameRow[entry.module]);
    }
    // the OT rechits, module after module in the shuffled order
    std::vector<uint32_t> firstRow(modules.size());
    for (std::size_t m = 0; m < modules.size(); ++m) {
      firstRow[m] = hits.otRows.size();
      for (std::size_t r = 0; r < modules[m].rows.size(); ++r) {
        auto const [track, position] = modules[m].owners[r];
        if (track >= 0) {
          hits.trackRows[track].resize(tracks[track].hits.size());
          hits.trackRows[track][position] = HitRow{true, uint32_t(hits.otRows.size())};
        }
        hits.otRows.push_back(modules[m].rows[r]);
        hits.otModule.push_back(frameRow[modules[m].frame]);
      }
    }
    // the CA modules, in an order of their own
    std::vector<uint32_t> caModules;
    for (std::size_t m = 0; m < modules.size(); ++m)
      if (modules[m].ca)
        caModules.push_back(m);
    std::shuffle(caModules.begin(), caModules.end(), rng);
    std::vector<int> caModuleOfRow(hits.otRows.size(), -1);
    uint32_t caIndex = hits.pixelRows.size();
    for (std::size_t i = 0; i < caModules.size(); ++i) {
      const uint32_t m = caModules[i];
      hits.caModuleStart.push_back(caIndex);
      hits.caRecHitStart.push_back(firstRow[m]);
      for (std::size_t r = 0; r < modules[m].rows.size(); ++r)
        caModuleOfRow[firstRow[m] + r] = i;
      caIndex += modules[m].rows.size();
    }
    hits.caModuleStart.push_back(caIndex);

    // the CA hit indices of the tracks and the hits as the fit is to see them
    hits.trackIds.resize(tracks.size());
    hits.expected.resize(tracks.size());
    for (std::size_t t = 0; t < tracks.size(); ++t) {
      if (tracks[t].sharesHitsOf >= 0)
        hits.trackRows[t] = hits.trackRows[tracks[t].sharesHitsOf];
      for (HitRow const& hitRow : hits.trackRows[t]) {
        TestHit hit;
        uint32_t module;
        if (hitRow.isOT) {
          const int caModule = caModuleOfRow[hitRow.row];
          hits.trackIds[t].push_back(hits.caModuleStart[caModule] + (hitRow.row - hits.caRecHitStart[caModule]));
          hit = hits.otRows[hitRow.row];
          module = hits.otModule[hitRow.row];
        } else {
          hits.trackIds[t].push_back(hitRow.row);
          TestHit const& row = hits.pixelRows[hitRow.row];
          hit = row;
          hit.x = row.x + settings.beamSpotX;
          hit.y = row.y + settings.beamSpotY;
          hit.z = row.z + settings.beamSpotZ;
          hit.detId = ::lst::kPixelModuleId;
          hit.clusterSize = 1;
          module = hits.pixelModule[hitRow.row];
        }
        std::copy(hits.moduleFrames[module].origin, hits.moduleFrames[module].origin + 3, hit.origin);
        std::copy(hits.moduleFrames[module].rotation, hits.moduleFrames[module].rotation + 9, hit.rotation);
        hit.index = hitRow.row;
        hits.expected[t].push_back(hit);
      }
    }
    return hits;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Running the kernels

  // The per-track functions on the fit inputs that the producer's kernels gathered: the full fit and the
  // inner-tracker-only fit of the sizes the producer chose (nFit of its states), and the pLS of a full state of status
  // kOk. The states start as copies of the producer's, so that stateHit is kept. Starting from the gathered inputs
  // makes the comparison exact: the hit covariances computed in two different kernels may differ in the last bits,
  // the same float expression being contracted to fused multiply-adds differently. For the same reason the fits' work
  // array is where the producer's fit kernels keep it, one slice of block-shared memory per thread, sized for the
  // largest fit; the launch uses kRefitThreads threads per block.
  constexpr uint32_t kRefitWork = ALPAKA_ACCELERATOR_NAMESPACE::lst::fitWorkSize(kMaxFitHits);
  constexpr uint32_t kRefitThreads = ALPAKA_ACCELERATOR_NAMESPACE::lst::fitThreadsPerBlock(kMaxFitHits);

  struct RefitKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::lstInputDevice::FitInput const* inputs,
                                  Settings settings,
                                  uint32_t nTracks,
                                  FittedState* states,
                                  FittedState* itStates,
                                  PixelSegmentCandidate* candidates) const {
      constexpr uint32_t kSlots = cms::alpakatools::requires_single_thread_per_block_v<Acc1D> ? 1 : kRefitThreads;
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] <= kSlots));
      auto& workArrays = alpaka::declareSharedVar<double[kSlots * kRefitWork], __COUNTER__>(acc);
      double* work = workArrays + alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] * kRefitWork;
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        if (states[t].nFit > 0)
          fitFull(acc, states[t].nFit, inputs[t].hits, settings.bField, states[t], work);
        if (itStates[t].nFit == kITFitHits)
          fit::fitState<kITFitHits>(acc, inputs[t].itHits, settings.bField, itStates[t], work);
        if (states[t].status == static_cast<uint8_t>(SeedStateStatus::kOk))
          fit::makePixelSegment(states[t], settings, candidates[t]);
      }
    }
  };

  struct PipelineOutput {
    uint32_t nTracks = 0, nOTHits = 0;
    std::vector<::lstInputDevice::FitInput> inputs;
    std::vector<PixelSegmentCandidate> candidates;
    std::vector<FittedState> states, itStates;
    // the per-track functions on the gathered fit inputs (RefitKernel)
    std::vector<PixelSegmentCandidate> refitCandidates;
    std::vector<FittedState> refitStates, refitITStates;
    std::vector<uint32_t> lists, seedIndex, segmentIndex, hitOffset;
    Counters counters{};
    Totals totals{};
    std::optional<::lst::LSTInputHostCollection> lstInput;
    std::optional<::lst::PixelSeedStatesHostCollection> seedStates;
  };

  template <typename T>
  cms::alpakatools::device_buffer<Device, T[]> toDevice(Queue& queue, std::vector<T> const& values) {
    auto buffer = cms::alpakatools::make_device_buffer<T[]>(queue, values.size());
    if (!values.empty())
      alpaka::memcpy(queue, buffer, cms::alpakatools::make_host_view(values.data(), values.size()));
    return buffer;
  }

  template <typename T, typename TBuffer>
  std::vector<T> toHost(Queue& queue, TBuffer const& buffer, uint32_t size) {
    if (size == 0)
      return {};
    auto host = cms::alpakatools::make_host_buffer<T[]>(queue, size);
    alpaka::memcpy(queue, host, buffer);
    alpaka::wait(queue);
    return std::vector<T>(host.data(), host.data() + size);
  }

  template <typename T, typename TBuffer>
  T scalarToHost(Queue& queue, TBuffer const& buffer) {
    auto host = cms::alpakatools::make_host_buffer<T>(queue);
    alpaka::memcpy(queue, host, buffer);
    alpaka::wait(queue);
    return *host.data();
  }

  // The device inputs of an event whose pixel track t is track order[t], the launchers in the order of the producer,
  // and the host copies of every buffer and product.
  PipelineOutput runPipeline(Queue& queue,
                             EventHits const& hits,
                             std::vector<EventTrack> const& tracks,
                             std::vector<uint32_t> const& order,
                             Settings settings) {
    namespace input = ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice;
    const uint32_t nTracks = order.size();
    const uint32_t nPixelHits = hits.pixelRows.size();
    const uint32_t nOTHits = hits.otRows.size();
    settings.nPixelHits = nPixelHits;
    settings.nCAOTModules = hits.caModuleStart.size() - 1;

    // the host inputs
    PortableHostCollection<reco::TrackingRecHitSoA> pixelHost(cms::alpakatools::host(), nPixelHits);
    for (uint32_t row = 0; row < nPixelHits; ++row) {
      auto out = pixelHost.view()[row];
      TestHit const& hit = hits.pixelRows[row];
      out.xLocal() = out.yLocal() = 0.f;
      out.xerrLocal() = hit.varX;
      out.yerrLocal() = hit.varY;
      out.xGlobal() = hit.x;
      out.yGlobal() = hit.y;
      out.zGlobal() = hit.z;
      out.rGlobal() = std::hypot(hit.x, hit.y);
      out.detectorIndex() = hits.pixelModule[row];
    }
    PortableHostCollection<reco::Phase2OTRecHitsSoA> otHost(cms::alpakatools::host(), nOTHits);
    for (uint32_t row = 0; row < nOTHits; ++row) {
      auto out = otHost.view()[row];
      TestHit const& hit = hits.otRows[row];
      out.detId() = hit.detId;
      out.detectorIndex() = hits.otModule[row];
      out.clusterSize() = hit.clusterSize;
      out.xLocal() = out.yLocal() = 0.f;
      out.xerrLocal() = hit.varX;
      out.yerrLocal() = hit.varY;
      out.xGlobal() = hit.x;
      out.yGlobal() = hit.y;
      out.zGlobal() = hit.z;
    }
    const uint32_t nModules = hits.moduleFrames.size();
    ::lst::LSTSeedFitModulesHost modulesHost(cms::alpakatools::host(), nModules);
    for (uint32_t row = 0; row < nModules; ++row)
      modulesHost.view()[row].frame() = frameOf(hits.moduleFrames[row]);
    uint32_t nTrackHits = 0;
    for (uint32_t t = 0; t < nTracks; ++t)
      nTrackHits += hits.trackIds[order[t]].size();
    PortableHostCollection<reco::TrackBlocks> tracksHost(cms::alpakatools::host(), nTracks, nTrackHits);
    std::vector<int32_t> edmIndex(nTracks);
    {
      auto trackView = tracksHost.view().tracks();
      auto hitView = tracksHost.view().trackHits();
      trackView.nTracks() = nTracks;
      uint32_t offset = 0;
      for (uint32_t t = 0; t < nTracks; ++t) {
        for (uint32_t id : hits.trackIds[order[t]]) {
          hitView[offset].id() = id;
          hitView[offset].detId() = 0;
          ++offset;
        }
        trackView[t].hitOffsets() = offset;
        edmIndex[t] = tracks[order[t]].hasEdmTrack ? static_cast<int32_t>(order[t]) : -1;
      }
    }

    // the device inputs
    DeviceCollection<reco::TrackingRecHitSoA> pixelDevice(queue, nPixelHits);
    alpaka::memcpy(queue, pixelDevice.buffer(), pixelHost.buffer());
    DeviceCollection<reco::Phase2OTRecHitsSoA> otDevice(queue, nOTHits);
    alpaka::memcpy(queue, otDevice.buffer(), otHost.buffer());
    DeviceCollection<::lst::LSTSeedFitModulesSoA> modulesDevice(queue, nModules);
    alpaka::memcpy(queue, modulesDevice.buffer(), modulesHost.buffer());
    DeviceCollection<reco::TrackBlocks> tracksDevice(queue, nTracks, nTrackHits);
    alpaka::memcpy(queue, tracksDevice.buffer(), tracksHost.buffer());
    auto edmIndexDevice = toDevice(queue, edmIndex);
    auto caModuleStartDevice = toDevice(queue, hits.caModuleStart);
    auto caRecHitStartDevice = toDevice(queue, hits.caRecHitStart);

    // the buffers of the producer, and its kernels in its order
    auto inputs = cms::alpakatools::make_device_buffer<::lstInputDevice::FitInput[]>(queue, nTracks);
    auto lists = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kNumFitLists * nTracks);
    auto counters = cms::alpakatools::make_device_buffer<Counters>(queue);
    alpaka::memset(queue, counters, 0);
    auto candidates = cms::alpakatools::make_device_buffer<PixelSegmentCandidate[]>(queue, nTracks);
    auto states = cms::alpakatools::make_device_buffer<FittedState[]>(queue, nTracks);
    auto itStates = cms::alpakatools::make_device_buffer<FittedState[]>(queue, nTracks);
    input::prepareTracks(queue,
                         tracksDevice.const_view(),
                         nTracks,
                         edmIndexDevice.data(),
                         pixelDevice.const_view(),
                         caModuleStartDevice.data(),
                         caRecHitStartDevice.data(),
                         otDevice.const_view(),
                         modulesDevice.const_view(),
                         settings,
                         inputs.data(),
                         candidates.data(),
                         states.data(),
                         itStates.data(),
                         lists.data(),
                         counters.data());
    const float bField = settings.bField;
    input::fitAllTracks(
        queue, nTracks, inputs.data(), bField, lists.data(), counters.data(), states.data(), itStates.data());
    input::makePixelSegments(queue, nTracks, settings, states.data(), candidates.data(), counters.data());
    auto refitStates = cms::alpakatools::make_device_buffer<FittedState[]>(queue, nTracks);
    auto refitITStates = cms::alpakatools::make_device_buffer<FittedState[]>(queue, nTracks);
    auto refitCandidates = cms::alpakatools::make_device_buffer<PixelSegmentCandidate[]>(queue, nTracks);
    if (nTracks > 0) {
      alpaka::memcpy(queue, refitStates, states);
      alpaka::memcpy(queue, refitITStates, itStates);
      alpaka::memcpy(queue, refitCandidates, candidates);
      alpaka::exec<Acc1D>(
          queue,
          cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTracks, kRefitThreads), kRefitThreads),
          RefitKernel{},
          inputs.data(),
          settings,
          nTracks,
          refitStates.data(),
          refitITStates.data(),
          refitCandidates.data());
    }
    auto seedIndex = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks);
    auto segmentIndex = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks);
    auto hitOffset = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks);
    auto totalsDevice = cms::alpakatools::make_device_buffer<Totals>(queue);
    input::indexPixelSegments(queue,
                              states.data(),
                              candidates.data(),
                              nTracks,
                              seedIndex.data(),
                              segmentIndex.data(),
                              hitOffset.data(),
                              totalsDevice.data());

    PipelineOutput out;
    out.nTracks = nTracks;
    out.nOTHits = nOTHits;
    out.totals = scalarToHost<Totals>(queue, totalsDevice);
    Totals const& totals = out.totals;
    ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTInputDeviceCollection lstInput(
        queue, nOTHits + totals.nSegmentHits, totals.nSegments, totals.nSegmentHits);
    input::fillLSTInput(queue,
                        lstInput.view(),
                        otDevice.const_view(),
                        nOTHits,
                        bField,
                        candidates.data(),
                        nTracks,
                        seedIndex.data(),
                        segmentIndex.data(),
                        hitOffset.data());
    ALPAKA_ACCELERATOR_NAMESPACE::lst::PixelSeedStatesDeviceCollection seedStates(
        queue, totals.nSeeds, totals.nSeeds, totals.nSeeds);
    input::fillSeedStates(queue,
                          seedStates.view(),
                          bField,
                          states.data(),
                          itStates.data(),
                          candidates.data(),
                          nTracks,
                          seedIndex.data(),
                          counters.data());

    // the host copies
    out.lstInput.emplace(
        cms::alpakatools::host(), nOTHits + totals.nSegmentHits, totals.nSegments, totals.nSegmentHits);
    alpaka::memcpy(queue, out.lstInput->buffer(), lstInput.buffer());
    out.seedStates.emplace(cms::alpakatools::host(), totals.nSeeds, totals.nSeeds, totals.nSeeds);
    alpaka::memcpy(queue, out.seedStates->buffer(), seedStates.buffer());
    out.inputs = toHost<::lstInputDevice::FitInput>(queue, inputs, nTracks);
    out.candidates = toHost<PixelSegmentCandidate>(queue, candidates, nTracks);
    out.states = toHost<FittedState>(queue, states, nTracks);
    out.itStates = toHost<FittedState>(queue, itStates, nTracks);
    out.refitStates = toHost<FittedState>(queue, refitStates, nTracks);
    out.refitITStates = toHost<FittedState>(queue, refitITStates, nTracks);
    out.refitCandidates = toHost<PixelSegmentCandidate>(queue, refitCandidates, nTracks);
    out.lists = toHost<uint32_t>(queue, lists, kNumFitLists * nTracks);
    out.seedIndex = toHost<uint32_t>(queue, seedIndex, nTracks);
    out.segmentIndex = toHost<uint32_t>(queue, segmentIndex, nTracks);
    out.hitOffset = toHost<uint32_t>(queue, hitOffset, nTracks);
    out.counters = scalarToHost<Counters>(queue, counters);
    alpaka::wait(queue);
    return out;
  }

  // the per-track functions in the test kernel, on the hits as the fit is to see them
  std::vector<TestOutput> runReference(Queue& queue,
                                       EventHits const& hits,
                                       std::vector<EventTrack> const& tracks,
                                       Settings const& settings) {
    const uint32_t nTracks = tracks.size();
    if (nTracks == 0)
      return {};
    std::vector<TestTrack> testTracks;
    std::vector<TestHit> testHits;
    for (uint32_t t = 0; t < nTracks; ++t) {
      testTracks.push_back(TestTrack{static_cast<uint32_t>(testHits.size()),
                                     static_cast<uint32_t>(hits.expected[t].size()),
                                     static_cast<uint8_t>(tracks[t].hasEdmTrack),
                                     settings});
      testHits.insert(testHits.end(), hits.expected[t].begin(), hits.expected[t].end());
    }
    auto tracksDevice = toDevice(queue, testTracks);
    auto hitsDevice = toDevice(queue, testHits);
    auto outputs = cms::alpakatools::make_device_buffer<TestOutput[]>(queue, nTracks);
    constexpr uint32_t kThreads = 64;
    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTracks, kThreads), kThreads),
        TrackFitKernel{},
        tracksDevice.data(),
        hitsDevice.data(),
        settings.bField,
        outputs.data(),
        nTracks);
    return toHost<TestOutput>(queue, outputs, nTracks);
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Expectations and comparisons

  // what the inputs of a track make of it
  struct Expected {
    bool fitted;
    uint32_t nHits, nFit, nIT;
    int lastIT;  // position of the last inner-tracker hit, -1 if none
    bool itCandidate, itFit;
    SeedStateStatus status, itStatus;
    std::vector<uint32_t> itPositions;
  };

  Expected expectation(EventTrack const& track, Settings const& settings) {
    Expected e{};
    e.nHits = track.hits.size();
    e.fitted = track.hasEdmTrack && e.nHits <= kMaxTrackHits && expectedFitSize(e.nHits) > 0;
    e.nFit = e.fitted ? expectedFitSize(e.nHits) : 0;
    e.lastIT = -1;
    for (uint32_t k = 0; k < e.nHits; ++k)
      if (!track.hits[k].isOT) {
        e.itPositions.push_back(k);
        e.lastIT = k;
      }
    e.nIT = e.itPositions.size();
    e.itCandidate = e.fitted && track.hits.back().isOT && e.nIT <= settings.maxITHitsToDropOTHits;
    e.itFit = e.itCandidate && e.nIT == kITFitHits;
    e.status = e.fitted ? track.expectedStatus : SeedStateStatus::kNotFitted;
    e.itStatus = e.itFit ? track.expectedITStatus : SeedStateStatus::kNotFitted;
    return e;
  }

  bool isOk(FittedState const& state) { return state.status == static_cast<uint8_t>(SeedStateStatus::kOk); }

  bool sameHit(FitHit const& a, FitHit const& b) {
    bool result = same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z);
    for (int k = 0; k < 6; ++k)
      result = result && same(a.cov[k], b.cov[k]);
    return result;
  }

  // the same value: the same bits, or both NaN (a NaN computed on the device may have other bits than on the host)
  bool sameValue(float a, float b) { return same(a, b) || (std::isnan(a) && std::isnan(b)); }

  // Every finite value of the state bit for bit; two NaNs count as the same value. The fit of a track with a NaN hit
  // returns NaN elements whose sign bit depends on the operand order: the compiler fuses products into multiply-adds
  // differently in the kernel and in the per-track functions, an instruction with NaN operands returns the one chosen
  // by its position, and the fit negates intermediates (e.g. -s in the Cholesky inversion), so either sign can reach
  // the result.
  bool sameRow(::lst::BLFStateRow const& a, ::lst::BLFStateRow const& b) {
    bool result = sameValue(a.x, b.x) && sameValue(a.y, b.y) && sameValue(a.z, b.z) && sameValue(a.phi, b.phi) &&
                  sameValue(a.cotTheta, b.cotTheta) && sameValue(a.kappa, b.kappa) && a.charge == b.charge;
    for (int k = 0; k < 6; ++k)
      result = result && sameValue(a.covCircle[k], b.covCircle[k]);
    for (int k = 0; k < 3; ++k)
      result = result && sameValue(a.covLine[k], b.covLine[k]);
    return result;
  }

  bool sameState(FittedState const& a, FittedState const& b) {
    return sameRow(a.row, b.row) && a.stateHit == b.stateHit && a.status == b.status && a.nFit == b.nFit &&
           a.refFallback == b.refFallback;
  }

  // the fields that prepareTrack sets; the slot covariances only if withCovariance
  bool samePreparation(PixelSegmentCandidate const& a, PixelSegmentCandidate const& b, bool withCovariance) {
    bool result = a.nHits == b.nHits && a.nITHits == b.nITHits && a.nSlots == b.nSlots && a.hitDetBits == b.hitDetBits;
    for (uint32_t slot = 0; result && slot < a.nSlots; ++slot)
      result = a.slotDetId[slot] == b.slotDetId[slot] && a.slotClusterSize[slot] == b.slotClusterSize[slot] &&
               a.slotHitIndex[slot] == b.slotHitIndex[slot] &&
               (withCovariance ? sameHit(a.slotHits[slot], b.slotHits[slot])
                               : sameValue(a.slotHits[slot].x, b.slotHits[slot].x) &&
                                     sameValue(a.slotHits[slot].y, b.slotHits[slot].y) &&
                                     sameValue(a.slotHits[slot].z, b.slotHits[slot].z));
    return result;
  }

  bool sameKinematics(::lst::PixelSegmentKinematics const& a, ::lst::PixelSegmentKinematics const& b) {
    return same(a.pcaX, b.pcaX) && same(a.pcaY, b.pcaY) && same(a.pcaZ, b.pcaZ) && same(a.ptPCA, b.ptPCA) &&
           same(a.etaPCA, b.etaPCA) && same(a.phiPCA, b.phiPCA) && same(a.lastHitX, b.lastHitX) &&
           same(a.lastHitY, b.lastHitY) && same(a.lastHitZ, b.lastHitZ) && same(a.dxy, b.dxy) && same(a.dz, b.dz);
  }

  // the fields that makePixelSegment sets, for a full state of status kOk
  bool samePixelSegment(PixelSegmentCandidate const& a, PixelSegmentCandidate const& b) {
    if (a.pixelType != b.pixelType || a.pcaStatus != b.pcaStatus)
      return false;
    if (a.pixelType == ::lst::PixelType::kInvalid)
      return true;
    return a.charge == b.charge && a.superbin == b.superbin && same(a.ptIn, b.ptIn) && same(a.ptErr, b.ptErr) &&
           same(a.px, b.px) && same(a.py, b.py) && same(a.pz, b.pz) && same(a.etaErr, b.etaErr) && same(a.eta, b.eta) &&
           same(a.phi, b.phi) && same(a.deltaPhi, b.deltaPhi) && sameKinematics(a.kinematics, b.kinematics);
  }

  // Two pLS of the same state, by the producer's kernel and by the per-track function in the test kernel: the
  // discrete fields and the last-hit quantities identical, the outputs of the closest approach within the tolerances
  // of two compilations of it
  struct SegmentDeviation {
    bool identical;   // pixelType, pcaStatus, charge, superbin, ptIn, p, eta, phi, deltaPhi, last hit
    double position;  // cm: dxy, dz, PCA position
    double momentum;  // ptPCA (relative), etaPCA and phiPCA (rad)
    double error;     // relative: ptErr, etaErr
  };

  SegmentDeviation compareSegments(PixelSegmentCandidate const& a, PixelSegmentCandidate const& b) {
    SegmentDeviation d{a.pixelType == b.pixelType && a.pcaStatus == b.pcaStatus, 0., 0., 0.};
    if (!d.identical || a.pixelType == ::lst::PixelType::kInvalid)
      return d;
    auto const& ka = a.kinematics;
    auto const& kb = b.kinematics;
    d.identical = a.charge == b.charge && a.superbin == b.superbin && same(a.ptIn, b.ptIn) && same(a.px, b.px) &&
                  same(a.py, b.py) && same(a.pz, b.pz) && same(a.eta, b.eta) && same(a.phi, b.phi) &&
                  same(a.deltaPhi, b.deltaPhi) && same(ka.lastHitX, kb.lastHitX) && same(ka.lastHitY, kb.lastHitY) &&
                  same(ka.lastHitZ, kb.lastHitZ);
    auto difference = [](float x, float y) {
      if (sameValue(x, y))
        return 0.;
      return std::isnan(x) || std::isnan(y) ? std::numeric_limits<double>::infinity() : std::abs(double(x) - y);
    };
    auto relative = [&](float x, float y) { return x == y ? 0. : difference(x, y) / std::abs(double(y)); };
    d.position = std::max({difference(ka.dxy, kb.dxy),
                           difference(ka.dz, kb.dz),
                           difference(ka.pcaX, kb.pcaX),
                           difference(ka.pcaY, kb.pcaY),
                           difference(ka.pcaZ, kb.pcaZ)});
    d.momentum = std::max({relative(ka.ptPCA, kb.ptPCA),
                           difference(ka.etaPCA, kb.etaPCA),
                           std::abs(std::remainder(difference(ka.phiPCA, kb.phiPCA), 2. * kPi))});
    d.error = std::max(relative(a.ptErr, b.ptErr), relative(a.etaErr, b.etaErr));
    return d;
  }

  // the largest element deviation of a covariance from the host computation, relative to its largest diagonal element
  double covarianceDeviation(float const* cov, TestHit const& hit) {
    float host[6];
    fit::hitCovariance(frameOf(hit), hit.varX, hit.varY, host);
    const double scale = std::max({std::abs(double(host[0])), std::abs(double(host[2])), std::abs(double(host[5]))});
    double deviation = 0.;
    for (int k = 0; k < 6; ++k)
      deviation = std::max(deviation, std::abs(double(cov[k]) - host[k]) / scale);
    return deviation;
  }

  bool samePosition(FitHit const& a, TestHit const& b) {
    return sameValue(a.x, b.x) && sameValue(a.y, b.y) && sameValue(a.z, b.z);
  }

  template <typename Row, typename View>
  bool sameSeedStateRow(Row const& row, View const& view, int index, FittedState const& state) {
    auto const& s = state.row;
    bool result = same(row.x(), s.x) && same(row.y(), s.y) && same(row.z(), s.z) && same(row.phi(), s.phi) &&
                  same(row.cotTheta(), s.cotTheta) && same(row.kappa(), s.kappa) && row.charge() == s.charge &&
                  row.status() == state.status && row.nFit() == state.nFit && row.refFallback() == state.refFallback &&
                  row.stateHit() == state.stateHit;
    for (int k = 0; k < 6; ++k)
      result = result && same(view.covCircle()[index][k], s.covCircle[k]);
    for (int k = 0; k < 3; ++k)
      result = result && same(view.covLine()[index][k], s.covLine[k]);
    return result;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // The checks of one event

  struct Summary {
    uint32_t nTracks = 0, nFitted = 0, nOk = 0, nSegments = 0, nITFits = 0, nPCAFailures = 0, nPCAFailedSegments = 0;
    std::vector<std::string> pcaFailures;  // the tracks whose closest approach failed
  };

  void evaluate(Checks& checks,
                std::string const& event,
                EventHits const& hits,
                std::vector<EventTrack> const& tracks,
                std::vector<uint32_t> const& order,
                std::vector<TestOutput> const& reference,
                PipelineOutput const& out,
                Settings const& settings,
                Summary& summary) {
    const uint32_t nTracks = order.size();
    const uint32_t nOTHits = hits.otRows.size();
    const double bField = settings.bField;
    summary.nTracks += nTracks;

    // per track in the order of the event: the expectations, and the indices from them and the test kernel
    std::vector<Expected> expected;
    Counters counts{};
    uint32_t nOk = 0, nSegments = 0, nSegmentHits = 0;
    std::vector<uint32_t> seedIndex(nTracks), segmentIndex(nTracks), hitOffset(nTracks);
    for (uint32_t t = 0; t < nTracks; ++t) {
      expected.push_back(expectation(tracks[order[t]], settings));
      if (tracks[order[t]].statusOfAllHits && expected.back().fitted && expected.back().nFit < expected.back().nHits)
        expected.back().status = static_cast<SeedStateStatus>(reference[order[t]].state.status);
      Expected const& e = expected.back();
      counts.nNotFitted += !e.fitted;
      counts.nNonFinite += e.status == SeedStateStatus::kNonFinite;
      counts.nChargeMismatch += e.status == SeedStateStatus::kChargeMismatch;
      counts.nNonPositiveVariance += e.status == SeedStateStatus::kNonPositiveVariance;
      counts.nITCandidates += e.itCandidate;
      counts.nITTooFewHits += e.itCandidate && e.nIT < kITFitHits;
      counts.nITNonFinite += e.itStatus == SeedStateStatus::kNonFinite;
      counts.nITChargeMismatch += e.itStatus == SeedStateStatus::kChargeMismatch;
      counts.nITNonPositiveVariance += e.itStatus == SeedStateStatus::kNonPositiveVariance;
      seedIndex[t] = nOk;
      segmentIndex[t] = nSegments;
      hitOffset[t] = nSegmentHits;
      if (e.status != SeedStateStatus::kOk)
        continue;
      ++nOk;
      PixelSegmentCandidate const& ref = out.refitCandidates[t];
      counts.nPCANotConverged += ref.pcaStatus == ::lst::PixelSegmentClosestApproach::kNotConverged;
      counts.nPCANonFinite += ref.pcaStatus == ::lst::PixelSegmentClosestApproach::kNonFinite;
      if (ref.pixelType != ::lst::PixelType::kInvalid) {
        ++nSegments;
        nSegmentHits += std::min(e.nHits, ::lst::kMaxPLSHitsInHitsSoA);
      }
    }

    // ---- per track: gathering, lists, states, pLS
    // the list of the inner-tracker-only fits, then one list per fit size
    checks("lists: one per fit size and the inner-tracker-only list", 0., "")
        .require(kNumFitLists == 1 + kExpectedFitSizes.size() && kITFitList == 0, event);
    std::vector<std::vector<int>> inList(nTracks, std::vector<int>(kNumFitLists, 0));
    for (uint32_t list = 0; list < kNumFitLists; ++list) {
      const uint32_t size = out.counters.listSize[list];
      checks("lists: size at most the number of tracks", 0., "").require(size <= nTracks, event);
      for (uint32_t i = 0; i < size && i < nTracks; ++i) {
        const uint32_t track = out.lists[list * nTracks + i];
        checks("lists: entries are tracks of the event", 0., "").require(track < nTracks, event);
        if (track < nTracks)
          ++inList[track][list];
      }
    }
    for (uint32_t t = 0; t < nTracks; ++t) {
      const uint32_t c = order[t];
      EventTrack const& track = tracks[c];
      Expected const& e = expected[t];
      TestOutput const& ref = reference[c];
      FittedState const& state = out.states[t];
      FittedState const& itState = out.itStates[t];
      PixelSegmentCandidate const& candidate = out.candidates[t];
      std::vector<TestHit> const& source = hits.expected[c];
      const std::string label = event + ": " + track.label;

      // lists: a fitted track in the list of its full fit only, and in the inner-tracker-only list iff it has that fit
      bool lists = inList[t][kITFitList] == (e.itFit ? 1 : 0);
      for (uint32_t list = 1; list < kNumFitLists; ++list)
        lists = lists && inList[t][list] == (e.fitted && list == expectedFitList(e.nFit) ? 1 : 0);
      checks("lists: each fitted track once, in the list of its fit size; IT-only list iff 3 IT hits", 0., "")
          .require(lists, label);

      checks("states: full-fit status as expected", 0., "")
          .require(state.status == static_cast<uint8_t>(e.status), label + " status " + statusName(state.status));
      checks("states: IT-only status as expected", 0., "")
          .require(itState.status == static_cast<uint8_t>(e.itStatus), label + " status " + statusName(itState.status));
      if (!e.fitted) {
        checks("not fitted: kNotFitted, nFit 0, NaN kappa, invalid stateHit, no pLS, no slots", 0., "")
            .require(
                state.nFit == 0 && std::isnan(state.row.kappa) && state.stateHit == ::lst::kInvalidU32Idx &&
                    itState.nFit == 0 && std::isnan(itState.row.kappa) && itState.stateHit == ::lst::kInvalidU32Idx &&
                    candidate.pixelType == ::lst::PixelType::kInvalid && candidate.nHits == 0 && candidate.nSlots == 0,
                label);
        continue;
      }
      ++summary.nFitted;

      // gathering: the fitted hits, the inner-tracker-only hits and the slots against the input rows
      std::vector<uint32_t> positions(e.nFit);
      fit::fitHitPositions(e.nHits, e.nFit, positions.data());
      for (uint32_t i = 0; i < e.nFit; ++i) {
        FitHit const& hit = out.inputs[t].hits[i];
        checks("gathering: fit-hit positions = pixel rows + beam spot, or OT rows of the CA starts", 0., "")
            .require(samePosition(hit, source[positions[i]]), label);
        checks("gathering: fit-hit covariance from the module frame, host vs kernel", kHitCovariance, "rel")
            .add(covarianceDeviation(hit.cov, source[positions[i]]), label);
      }
      if (e.itFit)
        for (uint32_t k = 0; k < kITFitHits; ++k) {
          FitHit const& hit = out.inputs[t].itHits[k];
          checks("gathering: IT-only fit hits are the IT hits", 0., "")
              .require(samePosition(hit, source[e.itPositions[k]]), label);
          checks("gathering: IT-only fit-hit covariance, host vs kernel", kHitCovariance, "rel")
              .add(covarianceDeviation(hit.cov, source[e.itPositions[k]]), label);
        }
      const uint32_t nSlots = std::min(e.nHits, ::lst::kMaxPLSHitsInHitsSoA);
      const uint32_t nBits = std::min(e.nHits, ::lst::kMaxPLSHitBitsInHitsSoA);
      uint8_t bits = 0;
      for (uint32_t bit = 0; bit < nBits; ++bit)
        bits |= source[::lst::pixelSegmentHitOfSlot(bit, nBits, e.nHits)].isOT << bit;
      checks("gathering: nHits, nITHits, nSlots, hitDetBits", 0., "")
          .require(candidate.nHits == e.nHits && candidate.nITHits == e.nIT && candidate.nSlots == nSlots &&
                       candidate.hitDetBits == bits,
                   label);
      for (uint32_t slot = 0; slot < nSlots; ++slot) {
        TestHit const& hit = source[::lst::pixelSegmentHitOfSlot(slot, nSlots, e.nHits)];
        checks("gathering: slot row index, detId, cluster size, position", 0., "")
            .require(candidate.slotHitIndex[slot] == hit.index && candidate.slotDetId[slot] == hit.detId &&
                         candidate.slotClusterSize[slot] == hit.clusterSize &&
                         samePosition(candidate.slotHits[slot], hit),
                     label);
        checks("gathering: slot covariance, host vs kernel", kHitCovariance, "rel")
            .add(covarianceDeviation(candidate.slotHits[slot].cov, hit), label);
      }

      // prepareTrack in the test kernel on the hits read by the test's reader: the same hits, slots, counts and bits;
      // the covariances to the contraction differences of kHitCovariance
      bool samePositions = ref.prepared == 1 && ref.nFit == e.nFit && ref.itFit == e.itFit;
      double covariance = 0.;
      auto compare = [&](FitHit const& a, FitHit const& b) {
        samePositions = samePositions && sameValue(a.x, b.x) && sameValue(a.y, b.y) && sameValue(a.z, b.z);
        const double scale =
            std::max({std::abs(double(b.cov[0])), std::abs(double(b.cov[2])), std::abs(double(b.cov[5]))});
        for (int k = 0; k < 6; ++k)
          covariance = std::max(covariance, std::abs(double(a.cov[k]) - b.cov[k]) / scale);
      };
      for (uint32_t i = 0; i < e.nFit; ++i)
        compare(out.inputs[t].hits[i], ref.input.hits[i]);
      for (uint32_t k = 0; e.itFit && k < kITFitHits; ++k)
        compare(out.inputs[t].itHits[k], ref.input.itHits[k]);
      for (uint32_t slot = 0; slot < candidate.nSlots && slot < ::lst::kMaxPLSHitsInHitsSoA; ++slot)
        compare(candidate.slotHits[slot], ref.candidate.slotHits[slot]);
      checks("per-track functions on the test's hit reader: fit-hit and slot positions identical", 0., "")
          .require(samePositions, label);
      checks("per-track functions on the test's hit reader: slots, hit counts, hitDetBits identical", 0., "")
          .require(samePreparation(candidate, ref.candidate, false), label);
      checks("per-track functions on the test's hit reader: hit covariances", kHitCovariance, "rel")
          .add(covariance, label);
      // the fits and the pLS of the per-track functions on the gathered inputs: identical bits
      checks("per-track functions on the gathered inputs: full state identical", 0., "")
          .require(sameState(state, out.refitStates[t]), label);
      checks("per-track functions on the gathered inputs: IT-only state identical", 0., "")
          .require(sameState(itState, out.refitITStates[t]), label);

      // the states
      checks("states: nFit = the largest fit size not above nHits, stateHit = last hit", 0., "")
          .require(state.nFit == e.nFit && state.stateHit == ::lst::encodeStateHitPixel(e.nHits - 1), label);
      checks("states: IT-only stateHit = last IT hit", 0., "")
          .require(itState.stateHit == (e.lastIT >= 0 ? ::lst::encodeStateHitPixel(e.lastIT) : ::lst::kInvalidU32Idx),
                   label);
      if (e.itFit) {
        ++summary.nITFits;
        checks("states: IT-only nFit = 3", 0., "").require(itState.nFit == kITFitHits, label);
        if (isOk(itState) && track.hasTruth)
          checkTruth(checks, "IT-only ", itState, track.generated, e.lastIT, bField, label);
      } else {
        checks("states: IT-only not run -> NaN kappa, nFit 0", 0., "")
            .require(std::isnan(itState.row.kappa) && itState.nFit == 0, label);
      }
      if (!isOk(state)) {
        checks("pLS: none without a full state of status kOk", 0., "")
            .require(candidate.pixelType == ::lst::PixelType::kInvalid, label);
        continue;
      }
      ++summary.nOk;
      if (track.hasTruth)
        checkTruth(checks, "", state, track.generated, e.nHits - 1, bField, label);

      // the closest approach and the pLS
      const SegmentDeviation deviation = compareSegments(candidate, out.refitCandidates[t]);
      checks(
          "per-track functions on the gathered inputs: pLS type, PCA status, charge, superbin, last-hit values", 0., "")
          .require(deviation.identical, label);
      checks("per-track functions on the gathered inputs: pLS dxy, dz, PCA position", kPLSPosition, "cm")
          .add(deviation.position, label);
      checks("per-track functions on the gathered inputs: pLS ptPCA (rel), etaPCA, phiPCA (rad)", kPLSMomentum, "")
          .add(deviation.momentum, label);
      checks("per-track functions on the gathered inputs: pLS ptErr, etaErr", kPLSError, "rel")
          .add(deviation.error, label);
      const bool pcaFailed = candidate.pcaStatus != ::lst::PixelSegmentClosestApproach::kValid;
      summary.nPCAFailures += pcaFailed;
      if (pcaFailed && track.hasTruth)
        summary.pcaFailures.push_back(
            label + ", turn to the last hit " +
            std::to_string(bField / track.generated.helix.pt * track.generated.truth.back().s).substr(0, 4) + " rad");
      if (candidate.pixelType == ::lst::PixelType::kInvalid)
        continue;
      ++summary.nSegments;
      if (pcaFailed) {
        // a failed closest approach keeps the pLS with zero momentum, impact parameters, errors and charge at the
        // closest approach; the PCA position is rebuilt from them and is not finite
        ++summary.nPCAFailedSegments;
        auto const& k = candidate.kinematics;
        checks("pLS: failed closest approach -> zero dxy, dz, ptErr, etaErr, ptPCA, charge; NaN PCA position", 0., "")
            .require(k.dxy == 0.f && k.dz == 0.f && candidate.ptErr == 0.f && candidate.etaErr == 0.f &&
                         k.ptPCA == 0.f && candidate.charge == 0 && std::isnan(k.pcaX) && std::isnan(k.pcaY) &&
                         std::isnan(k.pcaZ),
                     label);
      }
    }

    // ---- counters
    Counters const& n = out.counters;
    uint32_t pcaNotConverged = 0, pcaNonFinite = 0;
    for (uint32_t t = 0; t < nTracks; ++t)
      if (isOk(out.states[t])) {
        pcaNotConverged += out.candidates[t].pcaStatus == ::lst::PixelSegmentClosestApproach::kNotConverged;
        pcaNonFinite += out.candidates[t].pcaStatus == ::lst::PixelSegmentClosestApproach::kNonFinite;
      }
    checks("counters: full-fit rejections and not fitted as expected", 0., "")
        .require(n.nNotFitted == counts.nNotFitted && n.nNonFinite == counts.nNonFinite &&
                     n.nChargeMismatch == counts.nChargeMismatch &&
                     n.nNonPositiveVariance == counts.nNonPositiveVariance,
                 event);
    checks("counters: IT-only candidates, too few hits, rejections as expected", 0., "")
        .require(n.nITCandidates == counts.nITCandidates && n.nITTooFewHits == counts.nITTooFewHits &&
                     n.nITNonFinite == counts.nITNonFinite && n.nITChargeMismatch == counts.nITChargeMismatch &&
                     n.nITNonPositiveVariance == counts.nITNonPositiveVariance,
                 event);
    checks("counters: closest-approach failures over the kOk full fits", 0., "")
        .require(n.nPCANotConverged == pcaNotConverged && n.nPCANonFinite == pcaNonFinite &&
                     n.nPCANotConverged == counts.nPCANotConverged && n.nPCANonFinite == counts.nPCANonFinite,
                 event);

    // ---- indices
    checks("index: totals (seeds, pLS, pLS hits)", 0., "")
        .require(
            out.totals.nSeeds == nOk && out.totals.nSegments == nSegments && out.totals.nSegmentHits == nSegmentHits,
            event);
    for (uint32_t t = 0; t < nTracks; ++t) {
      const std::string label = event + ": " + tracks[order[t]].label;
      if (expected[t].status == SeedStateStatus::kOk)
        checks("index: seed index dense over the kOk full fits, in track order", 0., "")
            .require(out.seedIndex[t] == seedIndex[t], label);
      if (out.candidates[t].pixelType != ::lst::PixelType::kInvalid)
        checks("index: pLS index and first pLS hit dense over the pLS, in track order", 0., "")
            .require(out.segmentIndex[t] == segmentIndex[t] && out.hitOffset[t] == hitOffset[t], label);
    }

    // ---- the LST input
    auto const lstInput = out.lstInput->const_view();
    auto const lstHits = lstInput.hits();
    auto const lstHitsIT = lstInput.hitsIT();
    auto const segments = lstInput.pixelSeeds();
    checks("LST input: block sizes", 0., "")
        .require(uint32_t(lstHits.metadata().size()) == nOTHits + nSegmentHits &&
                     uint32_t(segments.metadata().size()) == nSegments &&
                     uint32_t(lstHitsIT.metadata().size()) == nSegmentHits,
                 event);
    checks("LST input: nHitsOT, B", 0., "")
        .require(lstHits.nHitsOT() == nOTHits && lstHits.B() == settings.bField, event);
    for (uint32_t row = 0; row < nOTHits; ++row) {
      TestHit const& hit = hits.otRows[row];
      checks("LST input: OT rows xs, ys, zs, detid, clustsize equal to the OT rechits", 0., "")
          .require(same(lstHits[row].xs(), hit.x) && same(lstHits[row].ys(), hit.y) && same(lstHits[row].zs(), hit.z) &&
                       lstHits[row].detid() == hit.detId && lstHits[row].clustsize() == hit.clusterSize,
                   event + " OT row " + std::to_string(row));
    }
    for (uint32_t t = 0; t < nTracks; ++t) {
      PixelSegmentCandidate const& candidate = out.candidates[t];
      if (candidate.pixelType == ::lst::PixelType::kInvalid)
        continue;
      const uint32_t c = order[t];
      Expected const& e = expected[t];
      const std::string label = event + ": " + tracks[c].label;
      std::vector<uint32_t> positions(e.nFit);
      fit::fitHitPositions(e.nHits, e.nFit, positions.data());
      const uint32_t first = nOTHits + hitOffset[t];
      for (uint32_t slot = 0; slot < candidate.nSlots; ++slot) {
        const uint32_t row = first + slot;
        const uint32_t itRow = hitOffset[t] + slot;
        float x, y, z;
        ::lst::pixelSegmentHitColumns(candidate.kinematics, slot, x, y, z);
        checks("LST input: pLS-hit rows: packed kinematics, slot detid and clustsize", 0., "")
            .require(same(lstHits[row].xs(), x) && same(lstHits[row].ys(), y) && same(lstHits[row].zs(), z) &&
                         lstHits[row].detid() == candidate.slotDetId[slot] &&
                         lstHits[row].clustsize() == candidate.slotClusterSize[slot],
                     label);
        // the slot hit with the values of the fit: the fit input itself when the hit is fitted, else computed alike
        // (its covariance against the host above)
        const uint32_t position = ::lst::pixelSegmentHitOfSlot(slot, candidate.nSlots, e.nHits);
        FitHit expectedHit = candidate.slotHits[slot];
        for (uint32_t i = 0; i < e.nFit; ++i)
          if (positions[i] == position)
            expectedHit = out.inputs[t].hits[i];
        FitHit stored{lstHitsIT[itRow].xs(), lstHitsIT[itRow].ys(), lstHitsIT[itRow].zs(), {}};
        for (int k = 0; k < 6; ++k)
          stored.cov[k] = lstHitsIT.cov()[itRow][k];
        checks("LST input: hitsIT idxs, xs, ys, zs, cov identical to the fit's values of the slot hit", 0., "")
            .require(lstHitsIT[itRow].idxs() == hits.expected[c][position].index && sameHit(stored, expectedHit) &&
                         samePosition(stored, hits.expected[c][position]),
                     label);
      }
      auto const segment = segments[segmentIndex[t]];
      checks("LST input: pLS rows equal to the pLS of the producer's kernels", 0., "")
          .require(segment.firstHit() == first && segment.nHits() == candidate.nHits &&
                       segment.hitDetBits() == candidate.hitDetBits && same(segment.deltaPhi(), candidate.deltaPhi) &&
                       segment.seedIdx() == seedIndex[t] && segment.charge() == candidate.charge &&
                       segment.superbin() == candidate.superbin && segment.pixelType() == candidate.pixelType &&
                       segment.isQuad() == (candidate.nHits > 3) && same(segment.ptIn(), candidate.ptIn) &&
                       same(segment.ptErr(), candidate.ptErr) && same(segment.px(), candidate.px) &&
                       same(segment.py(), candidate.py) && same(segment.pz(), candidate.pz) &&
                       same(segment.etaErr(), candidate.etaErr) && same(segment.eta(), candidate.eta) &&
                       same(segment.phi(), candidate.phi),
                   label);
    }

    // ---- the seed-state product
    auto const product = out.seedStates->const_view();
    auto const stateBlock = product.state();
    auto const itStateBlock = product.itState();
    auto const extra = product.extra();
    checks("seed states: one row per kOk full fit in each block", 0., "")
        .require(uint32_t(stateBlock.metadata().size()) == nOk && uint32_t(itStateBlock.metadata().size()) == nOk &&
                     uint32_t(extra.metadata().size()) == nOk,
                 event);
    checks("seed states: B in both state blocks", 0., "")
        .require(stateBlock.B() == settings.bField && itStateBlock.B() == settings.bField, event);
    checks("seed states: counters equal to the kernels' counters", 0., "")
        .require(extra.nTracks() == nTracks && extra.nNotFitted() == n.nNotFitted &&
                     extra.nNonFinite() == n.nNonFinite && extra.nChargeMismatch() == n.nChargeMismatch &&
                     extra.nNonPositiveVariance() == n.nNonPositiveVariance &&
                     extra.nITCandidates() == n.nITCandidates && extra.nITTooFewHits() == n.nITTooFewHits &&
                     extra.nITNonFinite() == n.nITNonFinite && extra.nITChargeMismatch() == n.nITChargeMismatch &&
                     extra.nITNonPositiveVariance() == n.nITNonPositiveVariance &&
                     extra.nPCANotConverged() == n.nPCANotConverged && extra.nPCANonFinite() == n.nPCANonFinite,
                 event);
    checks("seed states: nTracks = nNotFitted + kOk + nNonFinite + nChargeMismatch + nNonPositiveVariance", 0., "")
        .require(extra.nTracks() == extra.nNotFitted() + uint32_t(stateBlock.metadata().size()) + extra.nNonFinite() +
                                        extra.nChargeMismatch() + extra.nNonPositiveVariance(),
                 event);
    for (uint32_t t = 0; t < nTracks; ++t) {
      if (!isOk(out.states[t]) || out.seedIndex[t] >= nOk)
        continue;
      const int row = out.seedIndex[t];
      const std::string label = event + ": " + tracks[order[t]].label;
      checks("seed states: the full state at its seed index; type pLS, pixelSeedIndex = row", 0., "")
          .require(sameSeedStateRow(stateBlock[row], stateBlock, row, out.states[t]) &&
                       stateBlock[row].type() == ::lst::LSTObjType::pLS &&
                       stateBlock[row].pixelSeedIndex() == uint32_t(row),
                   label);
      checks("seed states: the IT-only state at the same row; type pLS, pixelSeedIndex = row", 0., "")
          .require(sameSeedStateRow(itStateBlock[row], itStateBlock, row, out.itStates[t]) &&
                       itStateBlock[row].type() == ::lst::LSTObjType::pLS &&
                       itStateBlock[row].pixelSeedIndex() == uint32_t(row),
                   label);
      checks("seed states: nITHits", 0., "").require(extra[row].nITHits() == expected[t].nIT, label);
    }
  }

  // an event without pixel tracks or without any input: empty products with their scalars, zero counters
  void checkEmpty(Checks& checks, std::string const& event, PipelineOutput const& out, uint32_t nOTHits, float bField) {
    auto const product = out.seedStates->const_view();
    auto const extra = product.extra();
    checks("empty: no seeds, pLS or pLS hits; LST input of the OT rows only", 0., "")
        .require(out.totals.nSeeds == 0 && out.totals.nSegments == 0 && out.totals.nSegmentHits == 0 &&
                     uint32_t(out.lstInput->const_view().hits().metadata().size()) == nOTHits &&
                     out.lstInput->const_view().hits().nHitsOT() == nOTHits &&
                     out.lstInput->const_view().hits().B() == bField,
                 event);
    checks("empty: seed-state product without rows, B set, nTracks and every counter zero", 0., "")
        .require(product.state().metadata().size() == 0 && product.state().B() == bField &&
                     product.itState().B() == bField && extra.nTracks() == 0 && extra.nNotFitted() == 0 &&
                     extra.nNonFinite() == 0 && extra.nChargeMismatch() == 0 && extra.nNonPositiveVariance() == 0 &&
                     extra.nITCandidates() == 0 && extra.nITTooFewHits() == 0 && extra.nITNonFinite() == 0 &&
                     extra.nITChargeMismatch() == 0 && extra.nITNonPositiveVariance() == 0 &&
                     extra.nPCANotConverged() == 0 && extra.nPCANonFinite() == 0,
                 event);
  }

  // the per-track results of two orders of the same tracks
  void compareOrders(Checks& checks,
                     Settings const& settings,
                     std::vector<EventTrack> const& tracks,
                     std::vector<uint32_t> const& orderA,
                     PipelineOutput const& a,
                     std::vector<uint32_t> const& orderB,
                     PipelineOutput const& b) {
    std::vector<uint32_t> positionB(tracks.size());
    for (uint32_t t = 0; t < orderB.size(); ++t)
      positionB[orderB[t]] = t;
    auto const segmentsA = a.lstInput->const_view().pixelSeeds();
    auto const segmentsB = b.lstInput->const_view().pixelSeeds();
    for (uint32_t ta = 0; ta < orderA.size(); ++ta) {
      const uint32_t c = orderA[ta], tb = positionB[c];
      Expected const e = expectation(tracks[c], settings);
      bool identical = sameState(a.states[ta], b.states[tb]) && sameState(a.itStates[ta], b.itStates[tb]) &&
                       a.candidates[ta].pixelType == b.candidates[tb].pixelType;
      if (e.fitted) {
        identical = identical && samePreparation(a.candidates[ta], b.candidates[tb], true);
        for (uint32_t i = 0; i < e.nFit; ++i)
          identical = identical && sameHit(a.inputs[ta].hits[i], b.inputs[tb].hits[i]);
      }
      if (isOk(a.states[ta]))
        identical = identical && samePixelSegment(a.candidates[ta], b.candidates[tb]);
      if (identical && a.candidates[ta].pixelType != ::lst::PixelType::kInvalid) {
        auto const sa = segmentsA[a.segmentIndex[ta]];
        auto const sb = segmentsB[b.segmentIndex[tb]];
        identical = sa.nHits() == sb.nHits() && sa.hitDetBits() == sb.hitDetBits() &&
                    same(sa.deltaPhi(), sb.deltaPhi()) && sa.charge() == sb.charge() &&
                    sa.superbin() == sb.superbin() && sa.pixelType() == sb.pixelType() && same(sa.ptIn(), sb.ptIn()) &&
                    same(sa.ptErr(), sb.ptErr()) && same(sa.px(), sb.px()) && same(sa.py(), sb.py()) &&
                    same(sa.pz(), sb.pz()) && same(sa.etaErr(), sb.etaErr()) && same(sa.eta(), sb.eta()) &&
                    same(sa.phi(), sb.phi());
      }
      checks("track order: fit inputs, states, pLS and pLS rows identical per track", 0., "")
          .require(identical, tracks[c].label);
    }
  }

}  // namespace

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  const float bField = blfTestStates::bField();
  const Settings settings{0.08f, -0.04f, 1.2f, 2e-4f, -1e-4f, 0.8f, bField, 0, 0, 3};
  TrackBuilder builder{settings, bField, {}, {}};
  const std::vector<EventTrack> tracks = buildTracks(builder);

  // the events: the tracks in their order and permuted, with unused rows and modules; no pixel tracks; pixel-only
  // tracks without OT hits; nothing
  std::mt19937 rng(42);
  const EventHits hits = buildHits(tracks, settings, rng, true);
  std::vector<uint32_t> order(tracks.size()), permuted(tracks.size());
  std::iota(order.begin(), order.end(), 0u);
  std::iota(permuted.begin(), permuted.end(), 0u);
  std::shuffle(permuted.begin(), permuted.end(), rng);
  std::vector<EventTrack> pixelTracks;
  for (auto const& track : tracks)
    if (std::none_of(track.hits.begin(), track.hits.end(), [](TestHit const& hit) { return hit.isOT; }) &&
        track.sharesHitsOf < 0)
      pixelTracks.push_back(track);
  const EventHits pixelHits = buildHits(pixelTracks, settings, rng, false);
  std::vector<uint32_t> pixelOrder(pixelTracks.size());
  std::iota(pixelOrder.begin(), pixelOrder.end(), 0u);
  const EventHits noHits = buildHits({}, settings, rng, false);

  bool failed = false;
  {
    Checks checks;
    for (auto const& label : builder.incomplete)
      checks("tracks: every helix crosses every surface of its layout", 0., "").require(false, label);
    checks("tracks: every helix crosses every surface of its layout", 0., "").require(true, "all");
    checks("events: pixel-only tracks without OT hits", 0., "")
        .require(!pixelTracks.empty() && pixelHits.otRows.empty() && pixelHits.caModuleStart.size() == 1, "");
    std::cout << tracks.size() << " tracks; " << hits.pixelRows.size() << " pixel rechits, " << hits.otRows.size()
              << " OT rechits, " << hits.caModuleStart.size() - 1 << " CA OT modules, " << hits.moduleFrames.size()
              << " module frames; " << pixelTracks.size() << " pixel-only tracks\n";
    failed = checks.report();
  }

  for (auto const& device : devices) {
    Queue queue(device);
    std::cout << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " " << alpaka::getName(device) << "\n";
    Checks checks;
    Summary summary;
    const std::vector<TestOutput> reference = runReference(queue, hits, tracks, settings);
    const PipelineOutput inOrder = runPipeline(queue, hits, tracks, order, settings);
    evaluate(checks, "in order", hits, tracks, order, reference, inOrder, settings, summary);
    const PipelineOutput inPermutedOrder = runPipeline(queue, hits, tracks, permuted, settings);
    evaluate(checks, "permuted", hits, tracks, permuted, reference, inPermutedOrder, settings, summary);
    compareOrders(checks, settings, tracks, order, inOrder, permuted, inPermutedOrder);

    const PipelineOutput noTracks = runPipeline(queue, hits, tracks, {}, settings);
    evaluate(checks, "no pixel tracks", hits, tracks, {}, reference, noTracks, settings, summary);
    checkEmpty(checks, "no pixel tracks", noTracks, hits.otRows.size(), bField);
    const std::vector<TestOutput> pixelReference = runReference(queue, pixelHits, pixelTracks, settings);
    const PipelineOutput noOTHits = runPipeline(queue, pixelHits, pixelTracks, pixelOrder, settings);
    evaluate(checks, "no OT hits", pixelHits, pixelTracks, pixelOrder, pixelReference, noOTHits, settings, summary);
    const PipelineOutput nothing = runPipeline(queue, noHits, {}, {}, settings);
    evaluate(checks, "no input", noHits, {}, {}, {}, nothing, settings, summary);
    checkEmpty(checks, "no input", nothing, 0, bField);

    std::cout << "  over the events: " << summary.nTracks << " tracks, " << summary.nFitted << " fitted, "
              << summary.nOk << " kOk full fits, " << summary.nITFits << " IT-only fits, " << summary.nSegments
              << " pLS; closest approach failed for " << summary.nPCAFailures << " kOk fits ("
              << summary.nPCAFailedSegments << " pLS)\n  list sizes in order (IT-only, then N =";
    for (uint32_t size : kExpectedFitSizes)
      std::cout << " " << size;
    std::cout << "):";
    for (uint32_t list = 0; list < kNumFitLists; ++list)
      std::cout << " " << inOrder.counters.listSize[list];
    std::cout << "\n";
    for (auto const& label : summary.pcaFailures)
      std::cout << "  closest approach failed: " << label << "\n";
    failed = checks.report() || failed;
  }
  std::cout << (failed ? "FAILED" : "passed") << "\n";
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
