// Unit test of the seed-state fit of the LST track candidates (RecoTracker/LSTCore/interface/alpaka/TCSeedStateFit.h),
// run end to end through its launcher on every device of the backend, on a synthetic event: noiseless helices in a
// uniform 3.8 T field (T5, T4, pT3, pT5 of 8 to 18 hits, and edge cases) and candidates made to end in every status.
// Checked: rows, statuses, fitted-hit counts and state hits against expectations; the kOk states against the truth
// with the tolerances of LSTInputDeviceTestTracks.h; the covariances of the fitted OT hits, which the fit computes from
// the OT rechits and the module frames, against the host; the counters; rows independent of the candidate order and
// of junk rows beyond the candidate count.

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/alpaka/Phase2OTRecHitsSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/LSTCore/interface/OTHitCovariance.h"
#include "RecoTracker/LSTCore/interface/TCSeedStatesHostCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStateFit.h"
#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStatesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"
#include "RecoTracker/LST/test/BLFTestStates.h"
#include "RecoTracker/LST/test/alpaka/LSTInputDeviceTestTracks.h"
#include "RecoTracker/LST/test/alpaka/TCSeedStateTestEvent.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  namespace tc = ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState;
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDeviceTest;
  // the test surfaces, not the geometry class of the same name that LST.h brings in
  using ALPAKA_ACCELERATOR_NAMESPACE::lstInputDeviceTest::Surface;
  using ::lst::LSTObjType;
  using tcSeedStateTest::Candidate;
  using tcSeedStateTest::Event;
  using tcSeedStateTest::Ref;
  using tcSeedStateTest::Row;
  using tcSeedStateTest::sameBits;

  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

  // ---------------------------------------------------------------------------------------------------------------
  // The event

  // what a candidate must give
  struct Expected {
    SeedStateStatus status;
    uint32_t nFit;
    uint32_t stateHit;
    int track = -1;  // the generated track and its hit the state sits on, for the truth (kOk)
    int node = -1;
  };

  struct Builder {
    float bField;
    Event event;
    std::vector<GeneratedTrack> tracks;
    std::vector<Expected> expected;  // one per candidate
    std::vector<std::string> incomplete;
    std::map<std::pair<bool, double>, TestHit> moduleFrames;  // the first hit on each OT surface (barrel, radius or z)

    // a row from a hit of a generated track: its position and its global covariance from its module frame
    static Row rowOf(TestHit const& hit, uint32_t detId, uint32_t idx) {
      Row row{hit.x, hit.y, hit.z, {}, detId, idx};
      fit::hitCovariance(frameOf(hit), hit.varX, hit.varY, row.cov);
      return row;
    }

    // an OT row, with the local variances and the module frame from which the fit computes its covariance
    Ref addOT(TestHit const& hit) {
      Row row = rowOf(hit, 0x10000000u + event.nOT(), 0);
      row.varX = hit.varX;
      row.varY = hit.varY;
      row.module = event.moduleOf(frameOf(hit));
      event.ot.push_back(row);
      return {false, event.nOT() - 1};
    }

    // a pixel track of nHits hits whose slots hold the given pixel hits, then copies of the given OT rows
    uint32_t addPixelTrack(std::vector<TestHit> const& pixelHits, std::vector<Ref> const& otHits, uint32_t nHits) {
      tcSeedStateTest::PixelTrack track{static_cast<uint32_t>(event.it.size()),
                                        static_cast<uint32_t>(pixelHits.size() + otHits.size()),
                                        nHits,
                                        static_cast<uint32_t>(10 + 2 * event.pixelTracks.size())};
      for (auto const& hit : pixelHits)
        event.it.push_back(rowOf(hit, ::lst::kPixelModuleId, 7000u + uint32_t(event.it.size())));
      for (Ref ot : otHits) {
        Row row = event.ot[ot.row];
        row.idx = ot.row;
        event.it.push_back(row);
      }
      event.pixelTracks.push_back(track);
      return event.pixelTracks.size() - 1;
    }

    int generateTrack(std::string const& label, Helix const& helix, std::vector<Surface> const& surfaces) {
      tracks.push_back(generate(helix, surfaces, bField, nullptr));
      // the OT hits on a surface that an earlier track crossed sit on the module of that surface and take its frame, so
      // that the rows of different tracks share a module
      for (std::size_t i = 0; i < tracks.back().hits.size(); ++i) {
        if (!surfaces[i].isOT)
          continue;
        TestHit& hit = tracks.back().hits[i];
        auto const [module, isNew] = moduleFrames.try_emplace({surfaces[i].barrel, surfaces[i].value}, hit);
        if (!isNew) {
          std::copy(module->second.origin, module->second.origin + 3, hit.origin);
          std::copy(module->second.rotation, module->second.rotation + 9, hit.rotation);
        }
      }
      if (!tracks.back().complete)
        incomplete.push_back(label);
      return tracks.size() - 1;
    }

    // A candidate from a helix through nPixel pixel surfaces (the slots of its pixel track, nHits hits) and then OT
    // modules of two sensors (the OT layer slots from 2 up): every hit fitted, or for more than 14 hits the pixel
    // hits and a stride of the OT hits; the state on the outermost OT hit.
    Candidate& addHelixCandidate(std::string const& label,
                                 LSTObjType type,
                                 Helix const& helix,
                                 std::vector<Surface> const& surfaces,
                                 int nPixel,
                                 uint32_t nHits,
                                 uint32_t nFit) {
      const int track = generateTrack(label, helix, surfaces);
      std::vector<TestHit> const& hits = tracks[track].hits;
      std::vector<TestHit> pixelHits(hits.begin(), hits.begin() + std::min<std::size_t>(nPixel, hits.size()));
      std::vector<Ref> ot;
      for (std::size_t i = nPixel; i < hits.size(); ++i)
        ot.push_back(addOT(hits[i]));
      Candidate& candidate = event.addCandidate(label, type, ::lst::kInvalidU32Idx);
      if (nPixel > 0)
        event.setPixelSlots(candidate, addPixelTrack(pixelHits, {}, nHits));
      for (std::size_t m = 0; m + 1 < ot.size(); m += 2) {
        candidate.slots[2 + m / 2][0] = ot[m];
        candidate.slots[2 + m / 2][1] = ot[m + 1];
      }
      expected.push_back({SeedStateStatus::kOk,
                          nFit,
                          ot.empty() ? ::lst::kInvalidU32Idx : ::lst::encodeStateHitOT(ot.back().row),
                          track,
                          int(hits.size()) - 1});
      return candidate;
    }
  };

  // OT barrel modules of two sensors 0.4 cm apart at the given radii
  std::vector<Surface> barrelModules(std::vector<double> const& radii) {
    std::vector<Surface> surfaces;
    for (double r : radii) {
      surfaces.push_back(otBarrel(r));
      surfaces.push_back(otBarrel(r + 0.4));
    }
    return surfaces;
  }
  // OT endcap modules of two sensors 0.4 cm apart at the given |z|, on the side of cot
  std::vector<Surface> endcapModules(std::vector<double> const& zs, double cot) {
    const double side = cot > 0. ? 1. : -1.;
    std::vector<Surface> surfaces;
    for (double z : zs) {
      surfaces.push_back(otDisc(side * z));
      surfaces.push_back(otDisc(side * (z + 0.4)));
    }
    return surfaces;
  }
  std::vector<Surface> pixelBarrel4() {
    std::vector<Surface> surfaces;
    surfaces.reserve(kPixelBarrelRadii.size());
    for (double r : kPixelBarrelRadii)
      surfaces.push_back(pixelBarrel(r));
    return surfaces;
  }
  template <typename... Lists>
  std::vector<Surface> join(Lists const&... lists) {
    std::vector<Surface> all;
    (all.insert(all.end(), lists.begin(), lists.end()), ...);
    return all;
  }

  const std::vector<double> kRadii7 = {23.0, 35.7, 50.8, 68.6, 88.6, 100.0, 108.0};
  std::vector<double> radii(int n) { return std::vector<double>(kRadii7.begin(), kRadii7.begin() + n); }

  Helix helix(double pt, int charge, double phi0, double cot) { return {{0.02, -0.01, 0.5}, phi0, pt, cot, charge}; }

  Builder buildEvent(float bField) {
    Builder builder{bField, {}, {}, {}, {}, {}};
    Event& event = builder.event;

    // every hit fitted, T5 and T4 of 8 to 14 hits, central and forward, both charges
    builder.addHelixCandidate("T5, 10 hits", LSTObjType::T5, helix(5., 1, 0.3, 0.4), barrelModules(radii(5)), 0, 0, 10);
    builder.addHelixCandidate(
        "T5, 12 hits", LSTObjType::T5, helix(2., -1, 2.1, -0.7), barrelModules(radii(6)), 0, 0, 12);
    builder.addHelixCandidate(
        "T5, 14 hits", LSTObjType::T5, helix(50., 1, -2.8, 0.1), barrelModules(radii(7)), 0, 0, 14);
    builder.addHelixCandidate(
        "T4, 8 hits", LSTObjType::T4, helix(1.2, -1, -1.0, 0.9), barrelModules(radii(4)), 0, 0, 8);
    builder.addHelixCandidate("T5 forward, 10 hits",
                              LSTObjType::T5,
                              helix(3., 1, 1.3, 4.),
                              endcapModules({131., 155., 185., 220., 265.}, 4.),
                              0,
                              0,
                              10);
    // a T5 holding a pixel seed index: its row holds none
    builder
        .addHelixCandidate(
            "T5 holding pixelSeedIndex 7", LSTObjType::T5, helix(8., -1, 0.9, -0.2), barrelModules(radii(5)), 0, 0, 10)
        .pixelSeedIndex = 7;

    // pT3 and pT5: the pixel hits from hitsIT
    builder.addHelixCandidate("pT3, 10 hits",
                              LSTObjType::pT3,
                              helix(4., -1, -0.4, 0.3),
                              join(pixelBarrel4(), barrelModules(radii(3))),
                              4,
                              4,
                              10);
    builder.addHelixCandidate("pT5, 14 hits",
                              LSTObjType::pT5,
                              helix(20., 1, 2.6, -0.5),
                              join(pixelBarrel4(), barrelModules(radii(5))),
                              4,
                              4,
                              14);
    // a pixel track of 6 hits: its slots hold its hits 0, 1, 2 and 5
    builder.addHelixCandidate("pT5, 14 hits, a pixel track of 6 hits",
                              LSTObjType::pT5,
                              helix(10., -1, 1.7, 0.6),
                              join(pixelBarrel4(), barrelModules(radii(5))),
                              4,
                              6,
                              14);
    // 4 pixel and 14 OT hits: the pixel hits and a stride of 10 OT hits, the outermost last
    builder.addHelixCandidate("pT5, 18 hits",
                              LSTObjType::pT5,
                              helix(30., 1, -1.9, 0.2),
                              join(pixelBarrel4(), barrelModules(radii(7))),
                              4,
                              4,
                              14);
    {
      // the pixel track's last slot is the candidate's first OT hit: the copy in the pLS slot is fitted, the OT slot's
      // is not; 9 hits, one dropped
      const std::string label = "pT3, an OT hit shared with the pixel track";
      std::vector<Surface> pixel3 = pixelBarrel4();
      pixel3.resize(3);
      const int track = builder.generateTrack(label, helix(6., 1, 0.1, -0.3), join(pixel3, barrelModules(radii(3))));
      auto const& hits = builder.tracks[track].hits;
      std::vector<Ref> ot;
      for (std::size_t i = 3; i < hits.size(); ++i)
        ot.push_back(builder.addOT(hits[i]));
      const uint32_t pixel =
          builder.addPixelTrack(std::vector<TestHit>(hits.begin(), hits.begin() + 3), {ot.front()}, 5);
      Candidate& candidate = event.addCandidate(label, LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int m = 0; m < 3; ++m) {
        candidate.slots[2 + m][0] = ot[2 * m];
        candidate.slots[2 + m][1] = ot[2 * m + 1];
      }
      builder.expected.push_back(
          {SeedStateStatus::kOk, 8, ::lst::encodeStateHitOT(ot.back().row), track, int(hits.size()) - 1});
    }
    {
      // the outer hit of the outermost module moved onto the transverse position of its inner hit: the pair is
      // degenerate, its inner hit is the last node; 9 hits, one dropped
      const std::string label = "T5, a degenerate outermost pair";
      const int track = builder.generateTrack(label, helix(15., -1, -2.2, 0.5), barrelModules(radii(5)));
      TestHit moved = builder.tracks[track].hits[9];
      moved.x = builder.tracks[track].hits[8].x;
      moved.y = builder.tracks[track].hits[8].y;
      std::vector<Ref> ot;
      ot.reserve(9);
      for (int i = 0; i < 9; ++i)
        ot.push_back(builder.addOT(builder.tracks[track].hits[i]));
      ot.push_back(builder.addOT(moved));
      Candidate& candidate = event.addCandidate(label, LSTObjType::T5, ::lst::kInvalidU32Idx);
      for (int m = 0; m < 5; ++m) {
        candidate.slots[2 + m][0] = ot[2 * m + 1];
        candidate.slots[2 + m][1] = ot[2 * m];
      }
      builder.expected.push_back({SeedStateStatus::kOk, 8, ::lst::encodeStateHitOT(ot[8].row), track, 8});
    }
    {
      // the pixel hits outside the OT hits (a pT3 of outer pixel hits): the state on pixel slot 3 of a 6-hit pixel
      // track, its hit 5
      const std::string label = "pT3, the state on a pixel hit";
      std::vector<Surface> outer;
      for (double r : {68.6, 88.6, 100.0, 108.0})
        outer.push_back(pixelBarrel(r));
      const int track = builder.generateTrack(label, helix(25., 1, 0.6, 0.2), join(barrelModules(radii(3)), outer));
      auto const& hits = builder.tracks[track].hits;
      std::vector<Ref> ot;
      ot.reserve(6);
      for (int i = 0; i < 6; ++i)
        ot.push_back(builder.addOT(hits[i]));
      const uint32_t pixel = builder.addPixelTrack(std::vector<TestHit>(hits.begin() + 6, hits.end()), {}, 6);
      Candidate& candidate = event.addCandidate(label, LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int m = 0; m < 3; ++m) {
        candidate.slots[2 + m][0] = ot[2 * m];
        candidate.slots[2 + m][1] = ot[2 * m + 1];
      }
      builder.expected.push_back({SeedStateStatus::kOk, 10, ::lst::encodeStateHitPixel(5), track, 9});
    }

    // not fitted
    {
      const std::string label = "pLS";
      const int track = builder.generateTrack(label, helix(3., -1, 1.1, 0.1), pixelBarrel4());
      const uint32_t pixel = builder.addPixelTrack(builder.tracks[track].hits, {}, 4);
      Candidate& candidate = event.addCandidate(label, LSTObjType::pLS, 0);
      event.setPLSSlots(candidate, pixel);
      builder.expected.push_back({SeedStateStatus::kNotFitted, 0, ::lst::kInvalidU32Idx});
    }
    {
      const std::string label = "pLS, a pixel track of 5 hits";
      const int track = builder.generateTrack(label, helix(1.5, 1, -2.5, -0.8), pixelBarrel4());
      const uint32_t pixel = builder.addPixelTrack(builder.tracks[track].hits, {}, 5);
      Candidate& candidate = event.addCandidate(label, LSTObjType::pLS, 0);
      event.setPLSSlots(candidate, pixel);
      builder.expected.push_back({SeedStateStatus::kNotFitted, 0, ::lst::kInvalidU32Idx});
    }
    builder.addHelixCandidate(
        "T4, 4 hits: too few", LSTObjType::T4, helix(2., 1, 0.2, 0.), barrelModules(radii(2)), 0, 0, 0);
    builder.expected.back() = {SeedStateStatus::kNotFitted, 0, ::lst::kInvalidU32Idx};

    // the checks of the state
    {
      builder.addHelixCandidate(
          "T5, a NaN position", LSTObjType::T5, helix(4., -1, 2.9, 0.3), barrelModules(radii(5)), 0, 0, 10);
      event.ot[event.index(event.candidates.back().slots[4][0])].x = kNaN;
      builder.expected.back().status = SeedStateStatus::kNonFinite;
    }
    {
      // negative hit variances make every hit weight negative: the solve stays finite, the variances do not stay
      // positive
      builder.addHelixCandidate(
          "T5, negative hit variances", LSTObjType::T5, helix(20., 1, -0.7, -0.2), barrelModules(radii(5)), 0, 0, 10);
      for (int slot = 2; slot <= 6; ++slot)
        for (int k = 0; k < 2; ++k) {
          Row& row = event.ot[event.candidates.back().slots[slot][k].row];
          for (float& value : row.cov)
            value = -value;
          row.varX = -row.varX;
          row.varY = -row.varY;
        }
      builder.expected.back().status = SeedStateStatus::kNonPositiveVariance;
    }
    {
      // The zigzag of the device LST input's test (testLSTInputDeviceKernels.dev.cc): the three hits that set the
      // fit's charge (first, middle, last) bend one way by 1 um, the two others lie 50 um on the other side; with
      // 100 um errors the fitted curvature has the opposite sign. Its 5 hits in three slots, the last one holding one
      // hit twice.
      const double xs[5] = {3., 7., 11., 15., 19.}, ys[5] = {0., -50e-4, 1e-4, -50e-4, 0.};
      std::vector<Ref> ot;
      for (int i = 0; i < 5; ++i) {
        TestHit hit{};
        hit.x = xs[i];
        hit.y = ys[i];
        hit.z = 0.1 * xs[i];
        const double azimuth = std::atan2(ys[i], xs[i]);
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
        ot.push_back(builder.addOT(hit));
      }
      Candidate& candidate = event.addCandidate("T5, zigzag", LSTObjType::T5, ::lst::kInvalidU32Idx);
      candidate.slots[2][0] = ot[0];
      candidate.slots[2][1] = ot[1];
      candidate.slots[3][0] = ot[2];
      candidate.slots[3][1] = ot[3];
      candidate.slots[4][0] = ot[4];
      candidate.slots[4][1] = ot[4];
      builder.expected.push_back({SeedStateStatus::kChargeMismatch, 5, ::lst::encodeStateHitOT(ot[4].row)});
    }
    return builder;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // The launcher on an event whose candidate i is candidate order[i]

  // The candidates followed by nSpare rows of junk that the fit must not read, as a track-candidate collection
  // larger than its candidate count holds: every type and an invalid one, pixel seed indices out of range, and hit
  // indices out of range or of a hit of NaN position and covariance (nanHit).
  ::lst::TrackCandidatesBaseHostCollection withSpareRows(::lst::TrackCandidatesBaseHostCollection const& candidates,
                                                         uint32_t nSpare,
                                                         uint32_t nanHit,
                                                         uint32_t nHits) {
    auto const in = candidates.const_view();
    const uint32_t n = in.metadata().size();
    ::lst::TrackCandidatesBaseHostCollection out(cms::alpakatools::host(), n + nSpare);
    auto view = out.view();
    view.nTrackCandidates() = in.nTrackCandidates();
    for (uint32_t i = 0; i < n; ++i) {
      view.trackCandidateType()[i] = in.trackCandidateType()[i];
      view.pixelSeedIndex()[i] = in.pixelSeedIndex()[i];
      view.hitIndices()[i] = in.hitIndices()[i];
    }
    const LSTObjType types[] = {
        LSTObjType::T5, LSTObjType::T4, LSTObjType::pT3, LSTObjType::pT5, LSTObjType::pLS, static_cast<LSTObjType>(90)};
    for (uint32_t s = 0; s < nSpare; ++s) {
      view.trackCandidateType()[n + s] = types[s % 6];
      view.pixelSeedIndex()[n + s] = 0xbad00000u + s;
      for (uint32_t slot = 0; slot < tcSeedStateTest::kLayers; ++slot)
        for (uint32_t k = 0; k < 2; ++k) {
          const uint32_t kind = (s + slot + k) % 3;
          view.hitIndices()[n + s][slot][k] = kind == 0 ? nanHit : kind == 1 ? nHits + 1000 * slot : 0x7ffffff0u + slot;
        }
    }
    return out;
  }

  // with fitInputs, also the hits of each candidate's fit as the fit read them, in candidate order
  ::lst::TCSeedStatesHostCollection runFit(Queue& queue,
                                           Event const& event,
                                           std::vector<uint32_t> const& order,
                                           float bField,
                                           uint32_t nSpare = 0,
                                           uint32_t nanHit = ::lst::kTCEmptyHitIdx,
                                           std::vector<tc::FitInput>* fitInputs = nullptr) {
    const auto inputHost = tcSeedStateTest::makeInput(event, bField);
    const auto otRecHitsHost = tcSeedStateTest::makeOTRecHits(event);
    const uint32_t nOT = event.nOT(), nIT = event.it.size(), nPLS = event.pixelTracks.size();
    const auto framesHost = event.frameTable();
    const uint32_t nFrames = framesHost.size();
    const uint32_t nTCs = order.size();
    const auto candidatesHost =
        nSpare == 0 ? tcSeedStateTest::makeCandidates(event, order)
                    : withSpareRows(tcSeedStateTest::makeCandidates(event, order), nSpare, nanHit, nOT + nIT);
    ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTInputDeviceCollection input(queue, nOT + nIT, nPLS, nIT);
    alpaka::memcpy(queue, input.buffer(), inputHost.buffer());
    ALPAKA_ACCELERATOR_NAMESPACE::reco::Phase2OTRecHitsSoACollection otRecHits(queue, nOT);
    alpaka::memcpy(queue, otRecHits.buffer(), otRecHitsHost.buffer());
    auto frames = cms::alpakatools::make_device_buffer<SOAFrame<float>[]>(queue, nFrames);
    alpaka::memcpy(queue, frames, cms::alpakatools::make_host_view(framesHost.data(), nFrames));
    ALPAKA_ACCELERATOR_NAMESPACE::lst::TrackCandidatesBaseDeviceCollection candidates(queue, nTCs + nSpare);
    alpaka::memcpy(queue, candidates.buffer(), candidatesHost.buffer());

    ALPAKA_ACCELERATOR_NAMESPACE::lst::TCSeedStatesDeviceCollection seedStates(queue, nTCs, nTCs);
    auto inputs = cms::alpakatools::make_device_buffer<tc::FitInput[]>(queue, nTCs);
    auto states = cms::alpakatools::make_device_buffer<FittedState[]>(queue, nTCs);
    auto lists = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, tc::kNumFitSizes * nTCs);
    auto counters = cms::alpakatools::make_device_buffer<tc::Counters>(queue);
    tc::fitSeedStates(queue,
                      input.const_view(),
                      ::lst::OTHitCovarianceInputs{otRecHits.const_view(), frames.data()},
                      candidates.const_view(),
                      nTCs,
                      seedStates.view(),
                      {inputs.data(), states.data(), lists.data(), counters.data()});
    ::lst::TCSeedStatesHostCollection host(cms::alpakatools::host(), nTCs, nTCs);
    alpaka::memcpy(queue, host.buffer(), seedStates.buffer());
    if (fitInputs != nullptr) {
      fitInputs->resize(nTCs);
      alpaka::memcpy(queue, cms::alpakatools::make_host_view(fitInputs->data(), nTCs), inputs);
    }
    alpaka::wait(queue);
    return host;
  }

  // The covariance of every fitted OT hit, which the fit computes from the OT rechit of its row and the module frame,
  // against the host computation of its row (Builder::rowOf). A fitted hit is an OT row's if its position is that
  // row's, bit for bit; hits of NaN position are left out.
  void checkOTCovariances(Checks& checks,
                          Builder const& builder,
                          std::vector<uint32_t> const& order,
                          std::vector<tc::FitInput> const& fitInputs,
                          ::lst::TCSeedStatesHostCollection const& result) {
    auto const state = result.const_view().state();
    for (uint32_t i = 0; i < order.size(); ++i) {
      const std::string& label = builder.event.candidates[order[i]].label;
      for (uint32_t k = 0; k < state[i].nFit(); ++k) {
        ::lst::FitHit const& hit = fitInputs[i].hits[k];
        if (std::isnan(hit.x) || std::isnan(hit.y) || std::isnan(hit.z))
          continue;
        for (Row const& row : builder.event.ot) {
          if (!(sameBits(hit.x, row.x) && sameBits(hit.y, row.y) && sameBits(hit.z, row.z)))
            continue;
          const double scale =
              std::max({std::abs(double(row.cov[0])), std::abs(double(row.cov[2])), std::abs(double(row.cov[5]))});
          double deviation = 0.;
          for (int c = 0; c < 6; ++c)
            deviation = std::max(deviation, std::abs(double(hit.cov[c]) - row.cov[c]) / scale);
          checks("fit inputs: OT-hit covariance from the OT rechit and the module frame, host vs kernel",
                 kHitCovariance,
                 "rel")
              .add(deviation, label);
          break;
        }
      }
    }
  }

  // a row as the fit's state
  FittedState fittedState(::lst::TCSeedStatesSoA::ConstView const& view, uint32_t i) {
    auto const state = view.state();
    FittedState fitted;
    fitted.row.x = state[i].x();
    fitted.row.y = state[i].y();
    fitted.row.z = state[i].z();
    fitted.row.phi = state[i].phi();
    fitted.row.cotTheta = state[i].cotTheta();
    fitted.row.kappa = state[i].kappa();
    for (int k = 0; k < 6; ++k)
      fitted.row.covCircle[k] = state.covCircle()[i][k];
    for (int k = 0; k < 3; ++k)
      fitted.row.covLine[k] = state.covLine()[i][k];
    fitted.row.charge = state[i].charge();
    fitted.stateHit = state[i].stateHit();
    fitted.status = state[i].status();
    fitted.nFit = state[i].nFit();
    fitted.refFallback = state[i].refFallback();
    return fitted;
  }

  // every column of row i of a and row j of b, bit for bit
  bool sameRow(::lst::TCSeedStatesSoA::ConstView const& a,
               uint32_t i,
               ::lst::TCSeedStatesSoA::ConstView const& b,
               uint32_t j) {
    const FittedState x = fittedState(a, i), y = fittedState(b, j);
    bool same = sameBits(x.row.x, y.row.x) && sameBits(x.row.y, y.row.y) && sameBits(x.row.z, y.row.z) &&
                sameBits(x.row.phi, y.row.phi) && sameBits(x.row.cotTheta, y.row.cotTheta) &&
                sameBits(x.row.kappa, y.row.kappa) && x.row.charge == y.row.charge && x.stateHit == y.stateHit &&
                x.status == y.status && x.nFit == y.nFit && x.refFallback == y.refFallback;
    for (int k = 0; k < 6; ++k)
      same = same && sameBits(x.row.covCircle[k], y.row.covCircle[k]);
    for (int k = 0; k < 3; ++k)
      same = same && sameBits(x.row.covLine[k], y.row.covLine[k]);
    return same && a.state()[i].type() == b.state()[j].type() &&
           a.state()[i].pixelSeedIndex() == b.state()[j].pixelSeedIndex();
  }

  bool sameCounters(::lst::TCSeedStatesHostCollection const& a, ::lst::TCSeedStatesHostCollection const& b) {
    auto const x = a.const_view().extra(), y = b.const_view().extra();
    return x.nTCs() == y.nTCs() && x.nNotFittedPLS() == y.nNotFittedPLS() && x.nTooFewHits() == y.nTooFewHits() &&
           x.nNonFinite() == y.nNonFinite() && x.nChargeMismatch() == y.nChargeMismatch() &&
           x.nNonPositiveVariance() == y.nNonPositiveVariance();
  }

  struct Counts {
    uint32_t nOk = 0, nNotFittedPLS = 0, nTooFewHits = 0, nNonFinite = 0, nChargeMismatch = 0, nNonPositiveVariance = 0;
  };

  // the rows of an event in candidate order against the expectations, and the counters
  Counts evaluate(Checks& checks,
                  std::string const& prefix,
                  Builder const& builder,
                  std::vector<uint32_t> const& order,
                  ::lst::TCSeedStatesHostCollection const& result) {
    auto const view = result.const_view();
    auto const state = view.state();
    auto const extra = view.extra();
    const uint32_t n = order.size();
    checks(prefix + "rows: one per candidate", 0., "")
        .require(uint32_t(state.metadata().size()) == n && extra.nTCs() == n, prefix);
    checks(prefix + "B: the field of the input", 0., "").require(sameBits(state.B(), builder.bField), prefix);
    Counts counts;
    for (uint32_t i = 0; i < n; ++i) {
      Candidate const& candidate = builder.event.candidates[order[i]];
      Expected const& expected = builder.expected[order[i]];
      const std::string& label = candidate.label;
      const FittedState fitted = fittedState(view, i);
      const bool pixelSeed =
          candidate.type == LSTObjType::pT3 || candidate.type == LSTObjType::pT5 || candidate.type == LSTObjType::pLS;
      checks(prefix + "type", 0., "").require(state[i].type() == candidate.type, label);
      checks(prefix + "pixelSeedIndex: the candidate's for pT3, pT5, pLS, none for T5, T4", 0., "")
          .require(state[i].pixelSeedIndex() == (pixelSeed ? candidate.pixelSeedIndex : ::lst::kInvalidU32Idx), label);
      checks(prefix + "status", 0., "")
          .require(fitted.status == static_cast<uint8_t>(expected.status),
                   label + ": " + statusName(fitted.status) + ", expected " +
                       statusName(static_cast<uint8_t>(expected.status)));
      checks(prefix + "number of fitted hits", 0., "")
          .require(fitted.nFit == expected.nFit,
                   label + ": " + std::to_string(fitted.nFit) + ", expected " + std::to_string(expected.nFit));
      checks(prefix + "state hit", 0., "").require(fitted.stateHit == expected.stateHit, label);
      switch (static_cast<SeedStateStatus>(fitted.status)) {
        case SeedStateStatus::kOk:
          ++counts.nOk;
          break;
        case SeedStateStatus::kNotFitted:
          if (candidate.type == LSTObjType::pLS)
            ++counts.nNotFittedPLS;
          else
            ++counts.nTooFewHits;
          break;
        case SeedStateStatus::kNonFinite:
          ++counts.nNonFinite;
          break;
        case SeedStateStatus::kChargeMismatch:
          ++counts.nChargeMismatch;
          break;
        case SeedStateStatus::kNonPositiveVariance:
          ++counts.nNonPositiveVariance;
          break;
      }
      if (expected.status == SeedStateStatus::kNotFitted) {
        auto const& row = fitted.row;
        bool allNaN = std::isnan(row.x) && std::isnan(row.y) && std::isnan(row.z) && std::isnan(row.phi) &&
                      std::isnan(row.cotTheta) && std::isnan(row.kappa);
        for (float value : row.covCircle)
          allNaN = allNaN && std::isnan(value);
        for (float value : row.covLine)
          allNaN = allNaN && std::isnan(value);
        checks(prefix + "unfit rows: NaN state, no charge, no fallback", 0., "")
            .require(allNaN && row.charge == 0 && fitted.refFallback == 0, label);
      }
      if (expected.status == SeedStateStatus::kOk && fitted.status == static_cast<uint8_t>(SeedStateStatus::kOk) &&
          expected.track >= 0) {
        checkTruth(checks, prefix, fitted, builder.tracks[expected.track], expected.node, builder.bField, label);
        checks(prefix + "no reference fallback (earlier nodes more than 2 cm away)", 0., "")
            .require(fitted.refFallback == 0, label);
      }
    }
    checks(prefix + "counters: the statuses of the rows", 0., "")
        .require(extra.nNotFittedPLS() == counts.nNotFittedPLS && extra.nTooFewHits() == counts.nTooFewHits &&
                     extra.nNonFinite() == counts.nNonFinite && extra.nChargeMismatch() == counts.nChargeMismatch &&
                     extra.nNonPositiveVariance() == counts.nNonPositiveVariance,
                 prefix);
    checks(prefix + "counters: an exact partition of the rows", 0., "")
        .require(counts.nOk + extra.nNotFittedPLS() + extra.nTooFewHits() + extra.nNonFinite() +
                         extra.nChargeMismatch() + extra.nNonPositiveVariance() ==
                     extra.nTCs(),
                 prefix);
    return counts;
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
  const Builder builder = buildEvent(bField);
  const uint32_t nCandidates = builder.event.candidates.size();
  std::vector<uint32_t> order(nCandidates), permuted(nCandidates);
  std::iota(order.begin(), order.end(), 0u);
  std::iota(permuted.begin(), permuted.end(), 0u);
  std::mt19937 rng(7);
  std::shuffle(permuted.begin(), permuted.end(), rng);

  bool failed = false;
  {
    Checks checks;
    for (auto const& label : builder.incomplete)
      checks("tracks: every helix crosses every surface of its layout", 0., "").require(false, label);
    checks("tracks: every helix crosses every surface of its layout", 0., "").require(true, "all");
    // an LST that has not computed the seed states (here one that has not run) hands out none
    ALPAKA_ACCELERATOR_NAMESPACE::lst::LST lst;
    checks("LST without the seed states: none", 0., "").require(lst.getTCSeedStates() == nullptr, "");
    std::cout << nCandidates << " candidates; " << builder.event.nOT() << " OT rows, " << builder.event.it.size()
              << " pLS slots, " << builder.event.pixelTracks.size() << " pixel tracks\n";
    failed = checks.report();
  }

  for (auto const& device : devices) {
    Queue queue(device);
    std::cout << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " " << alpaka::getName(device) << "\n";
    Checks checks;
    std::vector<tc::FitInput> fitInputs;
    const auto inOrder = runFit(queue, builder.event, order, bField, 0, ::lst::kTCEmptyHitIdx, &fitInputs);
    const Counts counts = evaluate(checks, "", builder, order, inOrder);
    checkOTCovariances(checks, builder, order, fitInputs, inOrder);
    const auto inPermutedOrder = runFit(queue, builder.event, permuted, bField);
    evaluate(checks, "permuted: ", builder, permuted, inPermutedOrder);
    for (uint32_t i = 0; i < nCandidates; ++i)
      checks("permuted: the rows of the permuted candidates, bit for bit", 0., "")
          .require(sameRow(inPermutedOrder.const_view(), i, inOrder.const_view(), permuted[i]),
                   builder.event.candidates[permuted[i]].label);
    checks("permuted: the same counters", 0., "").require(sameCounters(inOrder, inPermutedOrder), "");
    {
      // The candidate collection with spare rows of junk beyond the candidate count, as the track-candidate builder
      // leaves its unused rows: the launcher must read the first nTCs rows only. The event gets an OT row of NaN
      // position and covariance that only the spare rows refer to, in both runs.
      constexpr uint32_t kSpareRows = 37;
      tcSeedStateTest::Event event = builder.event;
      const uint32_t nanHit = event.nOT();
      event.ot.push_back(
          {kNaN, kNaN, kNaN, {kNaN, kNaN, kNaN, kNaN, kNaN, kNaN}, 0x10000000u + nanHit, 0, kNaN, kNaN, 0});
      const auto plain = runFit(queue, event, order, bField);
      const auto spare = runFit(queue, event, order, bField, kSpareRows, nanHit);
      checks("spare rows: nTCs rows", 0., "")
          .require(uint32_t(spare.const_view().state().metadata().size()) == nCandidates &&
                       spare.const_view().extra().nTCs() == nCandidates,
                   "");
      for (uint32_t i = 0; i < nCandidates; ++i)
        checks("spare rows: the rows of the candidates, bit for bit", 0., "")
            .require(sameRow(spare.const_view(), i, plain.const_view(), i), builder.event.candidates[i].label);
      checks("spare rows: the same counters", 0., "").require(sameCounters(spare, plain), "");
    }
    const auto empty = runFit(queue, builder.event, {}, bField);
    evaluate(checks, "no candidates: ", builder, {}, empty);
    {
      auto const extra = empty.const_view().extra();
      checks("no candidates: every counter zero", 0., "")
          .require(extra.nTCs() == 0 && extra.nNotFittedPLS() == 0 && extra.nTooFewHits() == 0 &&
                       extra.nNonFinite() == 0 && extra.nChargeMismatch() == 0 && extra.nNonPositiveVariance() == 0,
                   "");
    }
    std::cout << "  rows: " << counts.nOk << " kOk, " << counts.nNotFittedPLS << " pLS, " << counts.nTooFewHits
              << " too few hits, " << counts.nNonFinite << " kNonFinite, " << counts.nChargeMismatch
              << " kChargeMismatch, " << counts.nNonPositiveVariance << " kNonPositiveVariance\n";
    failed = checks.report() || failed;
  }
  std::cout << (failed ? "FAILED" : "passed") << "\n";
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
