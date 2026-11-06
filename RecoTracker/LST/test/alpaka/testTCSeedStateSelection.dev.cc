// Unit test of the hit selection of the seed-state fit of the LST track candidates
// (RecoTracker/LSTCore/interface/alpaka/TCSeedStateSelection.h): which hits are collected, in which order, which are
// fitted, and on which hit the state sits. Run in a kernel on every device of the backend on a synthetic LST input and
// synthetic candidates: hand-built cases against their expected hits, then random candidates against the properties
// every selection must have. The hits-block rows behind the pixel slots hold junk, so that reading them would change
// the order and the fit inputs. Every check is exact: the selection only moves and compares values, and the global
// covariance it computes for an OT hit from the OT rechit and the module frame is exact for the frames used here.

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/Phase2OTRecHitsSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/OTHitCovariance.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesHostCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStateSelection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"
#include "RecoTracker/LST/test/alpaka/TCSeedStateTestEvent.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  namespace tc = ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState;
  // not a using-directive: Event is also a type of the backend namespace
  using ::lst::LSTObjType;
  using tcSeedStateTest::Candidate;
  using tcSeedStateTest::Event;
  using tcSeedStateTest::expectedStateHit;
  using tcSeedStateTest::isPixel;
  using tcSeedStateTest::kEmpty;
  using tcSeedStateTest::kLayers;
  using tcSeedStateTest::makeCandidates;
  using tcSeedStateTest::makeInput;
  using tcSeedStateTest::makeOTRecHits;
  using tcSeedStateTest::otIndex;
  using tcSeedStateTest::PixelTrack;
  using tcSeedStateTest::Ref;
  using tcSeedStateTest::Row;
  using tcSeedStateTest::sameBits;

  // ---------------------------------------------------------------------------------------------------------------
  // The synthetic event

  // a covariance unique to every row, so that a mix-up of rows shows
  void fillCovariance(Row& row, uint32_t serial) {
    for (int k = 0; k < 6; ++k)
      row.cov[k] = 1e-4f * (serial + 1) + 1e-6f * k;
  }

  // The OT rows sit on kModules modules, row r on module r % kModules, so that rows share a module and share its frame;
  // the table of the frames lists the modules in an order different from that of the rows (Event::moduleOf). The
  // rotation of a module only permutes and flips the axes: the global covariance from the local variances is then
  // exact, whatever fused multiply-adds the compilers of the host and of the device form, and the fit computes cov of
  // the row bit for bit. The four rotations (rows: the local axes in the global frame) and local variances unique to
  // every row show a mix-up of rows, of modules or of the two variances.
  constexpr uint32_t kModules = 6;
  constexpr float kRotations[4][9] = {{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f},
                                      {0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f},
                                      {0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f},
                                      {-1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f}};

  Ref addOT(Event& event, float x, float y, float z) {
    const uint32_t serial = event.ot.size();
    Row row{x, y, z, {}, 0x10000000u + serial, 0};
    row.varX = 1e-4f * (serial + 1);
    row.varY = 1e-4f * (serial + 1) + 3e-6f;
    const uint32_t module = serial % kModules;
    row.module = event.moduleOf(SOAFrame<float>(float(module), 0.f, 0.f, SOARotation<float>(kRotations[module % 4])));
    tcSeedStateTest::setOTCovariance(event, row);
    event.ot.push_back(row);
    return {false, serial};
  }

  // an OT module at radius r: the inner hit at (r, 1, r / 2), the outer one at (r + dx, 1 + dy, r / 2 + dz)
  struct Module {
    Ref inner, outer;
  };
  Module addModule(Event& event, float r, float dx, float dy = 0.f, float dz = 0.f) {
    const Ref inner = addOT(event, r, 1.f, 0.5f * r);
    const Ref outer = addOT(event, r + dx, 1.f + dy, 0.5f * r + dz);
    return {inner, outer};
  }

  // the standard module of OT layer slot `slot` (2 to 12): radius 20 + 10 (slot - 2) cm, the two hits
  // 0.10 + 0.02 (slot - 2) cm apart, so that the smallest separation of a candidate is in its innermost module
  float layerRadius(int slot) { return 20.f + 10.f * (slot - 2); }
  float layerSeparation(int slot) { return 0.10f + 0.02f * (slot - 2); }
  Module addLayer(Event& event, int slot) { return addModule(event, layerRadius(slot), layerSeparation(slot)); }

  // a slot hit of a pixel track: a pixel hit at radius r (y = 1, z = r / 2), or a copy of OT row `ot`
  struct SlotHit {
    float r;
    int ot = -1;
  };

  // a pixel track of nHits hits with the given slot hits (at most 4); its hits rows hold junk
  uint32_t addPixelTrack(Event& event, std::vector<SlotHit> const& slots, uint32_t nHits) {
    PixelTrack track{static_cast<uint32_t>(event.it.size()),
                     static_cast<uint32_t>(slots.size()),
                     nHits,
                     static_cast<uint32_t>(100 + 3 * event.pixelTracks.size())};
    for (auto const& slot : slots) {
      Row row;
      if (slot.ot >= 0) {
        row = event.ot[slot.ot];
        row.idx = slot.ot;
      } else {
        row = Row{slot.r, 1.f, 0.5f * slot.r, {}, ::lst::kPixelModuleId, 5000u + uint32_t(event.it.size())};
        fillCovariance(row, 1000 + event.it.size());
      }
      event.it.push_back(row);
    }
    event.pixelTracks.push_back(track);
    return event.pixelTracks.size() - 1;
  }

  void setModule(Candidate& candidate, int slot, Module const& module, bool outerFirst = false) {
    candidate.slots[slot][0] = outerFirst ? module.outer : module.inner;
    candidate.slots[slot][1] = outerFirst ? module.inner : module.outer;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // The kernel

  struct Output {
    uint8_t preparation;
    tc::Collected collected;
    tc::Selection selection;
    tc::FitInput fitInput;
    ::lst::FittedState state;
    uint32_t rowPixelSeedIndex;
  };

  struct SelectionKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::lst::LSTInputConstView input,
                                  ::lst::OTHitCovarianceInputs otHits,
                                  ::lst::TrackCandidatesBaseConst candidates,
                                  uint32_t nCandidates,
                                  Output* outputs) const {
      const tc::InputHits hits{input.hits(), input.hitsIT(), input.pixelSeeds(), input.hits().nHitsOT(), otHits};
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nCandidates)) {
        Output& out = outputs[i];
        const LSTObjType type = candidates.trackCandidateType()[i];
        out.preparation = static_cast<uint8_t>(tc::prepareCandidate(
            hits, candidates.hitIndices()[i], type, out.collected, out.selection, out.fitInput, out.state));
        out.rowPixelSeedIndex = tc::rowPixelSeedIndex(type, candidates.pixelSeedIndex()[i]);
      }
    }
  };

  std::vector<Output> run(Queue& queue, Event const& event) {
    const uint32_t nOT = event.nOT(), nIT = event.it.size(), nPLS = event.pixelTracks.size();
    const uint32_t nCandidates = event.candidates.size();
    std::vector<uint32_t> order(nCandidates);
    for (uint32_t i = 0; i < nCandidates; ++i)
      order[i] = i;
    const auto inputHost = makeInput(event, 3.8f * 2.99792458e-3f);
    const auto candidatesHost = makeCandidates(event, order);

    ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTInputDeviceCollection inputDevice(queue, nOT + nIT, nPLS, nIT);
    alpaka::memcpy(queue, inputDevice.buffer(), inputHost.buffer());
    const auto otRecHitsHost = makeOTRecHits(event);
    ALPAKA_ACCELERATOR_NAMESPACE::reco::Phase2OTRecHitsSoACollection otRecHitsDevice(queue, nOT);
    alpaka::memcpy(queue, otRecHitsDevice.buffer(), otRecHitsHost.buffer());
    const auto framesHost = event.frameTable();
    const uint32_t nFrames = framesHost.size();
    auto framesDevice = cms::alpakatools::make_device_buffer<SOAFrame<float>[]>(queue, nFrames);
    alpaka::memcpy(queue, framesDevice, cms::alpakatools::make_host_view(framesHost.data(), nFrames));
    ALPAKA_ACCELERATOR_NAMESPACE::lst::TrackCandidatesBaseDeviceCollection candidatesDevice(queue, nCandidates);
    alpaka::memcpy(queue, candidatesDevice.buffer(), candidatesHost.buffer());
    auto outputs = cms::alpakatools::make_device_buffer<Output[]>(queue, nCandidates);
    constexpr uint32_t kThreads = 32;
    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nCandidates, kThreads), kThreads),
        SelectionKernel{},
        inputDevice.const_view(),
        ::lst::OTHitCovarianceInputs{otRecHitsDevice.const_view(), framesDevice.data()},
        candidatesDevice.const_view(),
        nCandidates,
        outputs.data());
    auto outputsHost = cms::alpakatools::make_host_buffer<Output[]>(queue, nCandidates);
    alpaka::memcpy(queue, outputsHost, outputs);
    alpaka::wait(queue);
    return std::vector<Output>(outputsHost.data(), outputsHost.data() + nCandidates);
  }

  // the squared 3D distance from the origin, by which the selection orders the hits
  float distance2(Row const& row) { return row.x * row.x + row.y * row.y + row.z * row.z; }

  // ---------------------------------------------------------------------------------------------------------------
  // Checks

  struct Checks {
    int failures = 0, count = 0;
    void require(bool condition, std::string const& what, std::string const& label) {
      ++count;
      if (!condition) {
        ++failures;
        if (failures <= 40)
          std::cout << "    FAILED " << what << " [" << label << "]\n";
      }
    }
  };

  std::string describe(Event const& event, uint32_t const* hits, uint32_t n) {
    std::string text;
    for (uint32_t i = 0; i < n; ++i) {
      const Ref ref = event.ref(hits[i]);
      text += (i ? " " : "") + std::string(ref.it ? "it" : "ot") + std::to_string(ref.row);
    }
    return text;
  }

  // The properties of every output: the collected hits ordered inside out, no OT hit twice; the fit size of their
  // count; the fitted hits a subset in the same order, with every pixel hit and the outermost hit last; the fit inputs
  // the values of the rows they come from (hitsIT for a pLS slot, never the junk of its hits row); the state unfit with
  // the hit of the last fitted hit; the pixelSeedIndex of the row.
  void checkProperties(Checks& checks, Event const& event, Candidate const& candidate, Output const& out) {
    const std::string& label = candidate.label;
    auto const& unfit = out.state.row;
    bool allNaN = std::isnan(unfit.x) && std::isnan(unfit.y) && std::isnan(unfit.z) && std::isnan(unfit.phi) &&
                  std::isnan(unfit.cotTheta) && std::isnan(unfit.kappa);
    for (float value : unfit.covCircle)
      allNaN = allNaN && std::isnan(value);
    for (float value : unfit.covLine)
      allNaN = allNaN && std::isnan(value);
    checks.require(allNaN && unfit.charge == 0 && out.state.nFit == 0 && out.state.refFallback == 0 &&
                       out.state.status == static_cast<uint8_t>(::lst::SeedStateStatus::kNotFitted),
                   "the state is unfit until the fit",
                   label);
    const bool pixelSeed =
        candidate.type == LSTObjType::pT3 || candidate.type == LSTObjType::pT5 || candidate.type == LSTObjType::pLS;
    checks.require(out.rowPixelSeedIndex == (pixelSeed ? candidate.pixelSeedIndex : ::lst::kInvalidU32Idx),
                   "the pixelSeedIndex of the row",
                   label);
    if (candidate.type == LSTObjType::pLS) {
      checks.require(
          out.preparation == static_cast<uint8_t>(tc::Preparation::kPLS) && out.state.stateHit == ::lst::kInvalidU32Idx,
          "a pLS is not fitted, and has no state hit",
          label);
      return;
    }
    tc::Collected const& collected = out.collected;
    tc::Selection const& selection = out.selection;
    bool ordered = true, distinct = true;
    for (uint32_t i = 0; i < collected.n; ++i) {
      if (i > 0 &&
          distance2(event.row(event.ref(collected.hits[i]))) < distance2(event.row(event.ref(collected.hits[i - 1]))))
        ordered = false;
      for (uint32_t j = 0; j < i; ++j)
        if (collected.hits[i] == collected.hits[j] ||
            (otIndex(event, collected.hits[i]) >= 0 &&
             otIndex(event, collected.hits[i]) == otIndex(event, collected.hits[j])))
          distinct = false;
    }
    checks.require(ordered, "the collected hits are ordered inside out", label);
    checks.require(distinct, "no hit and no OT hit is collected twice", label);
    checks.require(selection.nCollected == collected.n && selection.nFit == tc::fitSize(collected.n),
                   "the fit size of the collected hits",
                   label);
    if (selection.nFit == 0) {
      checks.require(out.preparation == static_cast<uint8_t>(tc::Preparation::kTooFewHits) &&
                         out.state.stateHit == ::lst::kInvalidU32Idx,
                     "too few hits: not fitted, no state hit",
                     label);
      return;
    }
    checks.require(out.preparation == static_cast<uint8_t>(tc::Preparation::kFit), "fitted", label);
    uint32_t position = 0;
    bool subset = true;
    for (uint32_t i = 0; i < selection.nFit; ++i) {
      while (position < collected.n && collected.hits[position] != selection.hits[i])
        ++position;
      subset = subset && position < collected.n;
      ++position;
    }
    checks.require(subset, "the fitted hits are collected hits, in their order", label);
    bool pixelsKept = true;
    for (uint32_t i = 0; i < collected.n; ++i)
      if (isPixel(event, collected.hits[i]))
        pixelsKept = pixelsKept && std::find(selection.hits, selection.hits + selection.nFit, collected.hits[i]) !=
                                       selection.hits + selection.nFit;
    checks.require(pixelsKept, "every pixel hit is fitted", label);
    checks.require(selection.hits[selection.nFit - 1] == collected.hits[collected.n - 1],
                   "the outermost collected hit is the last fitted hit",
                   label);
    bool inputs = true;
    for (uint32_t i = 0; i < selection.nFit; ++i) {
      Row const& row = event.row(event.ref(selection.hits[i]));
      ::lst::FitHit const& hit = out.fitInput.hits[i];
      inputs = inputs && sameBits(hit.x, row.x) && sameBits(hit.y, row.y) && sameBits(hit.z, row.z);
      for (int k = 0; k < 6; ++k)
        inputs = inputs && sameBits(hit.cov[k], row.cov[k]);
    }
    checks.require(inputs, "the fit inputs are the rows of the fitted hits (hitsIT for a pLS slot)", label);
    checks.require(out.state.stateHit == expectedStateHit(event, selection.hits[selection.nFit - 1]),
                   "the state hit is the last fitted hit",
                   label);
  }

  // a hand-built case and what it must give
  struct Expectation {
    tc::Preparation preparation;
    uint32_t nCollected;
    std::vector<Ref> fit;   // the fitted hits, inside out
    uint32_t stateHit;      // encoded
    std::vector<Ref> kept;  // hits that must be collected
    std::vector<Ref> lost;  // hits that must not be collected
  };

  void checkExpectation(
      Checks& checks, Event const& event, Candidate const& candidate, Output const& out, Expectation const& expected) {
    const std::string& label = candidate.label;
    checks.require(out.preparation == static_cast<uint8_t>(expected.preparation), "the outcome", label);
    if (expected.preparation == tc::Preparation::kPLS)
      return;
    checks.require(
        out.collected.n == expected.nCollected,
        "collected " + std::to_string(out.collected.n) + " hits, expected " + std::to_string(expected.nCollected),
        label);
    std::vector<uint32_t> fit;
    fit.reserve(expected.fit.size());
    for (Ref ref : expected.fit)
      fit.push_back(event.index(ref));
    const bool sameHits = out.selection.nFit == fit.size() && std::equal(fit.begin(), fit.end(), out.selection.hits);
    checks.require(sameHits,
                   "fitted " + describe(event, out.selection.hits, out.selection.nFit) + ", expected " +
                       describe(event, fit.data(), fit.size()),
                   label);
    checks.require(out.state.stateHit == expected.stateHit, "the encoded state hit", label);
    auto collected = [&](Ref ref) {
      return std::find(out.collected.hits, out.collected.hits + out.collected.n, event.index(ref)) !=
             out.collected.hits + out.collected.n;
    };
    for (Ref ref : expected.kept)
      checks.require(collected(ref), "collected: " + describe(event, std::vector{event.index(ref)}.data(), 1), label);
    for (Ref ref : expected.lost)
      checks.require(
          !collected(ref), "not collected: " + describe(event, std::vector{event.index(ref)}.data(), 1), label);
  }

  uint32_t otHit(Ref ref) { return ::lst::encodeStateHitOT(ref.row); }

  // ---------------------------------------------------------------------------------------------------------------
  // The hand-built cases

  struct Cases {
    Event event;
    std::vector<Expectation> expectations;  // one per candidate
  };

  // the standard modules of slots first..last
  std::vector<Module> addLayers(Event& event, int first, int last) {
    std::vector<Module> modules(kLayers);
    for (int slot = first; slot <= last; ++slot)
      modules[slot] = addLayer(event, slot);
    return modules;
  }

  // the inner and outer hits of the modules of slots first..last, inside out
  std::vector<Ref> hitsOf(std::vector<Module> const& modules, int first, int last) {
    std::vector<Ref> refs;
    for (int slot = first; slot <= last; ++slot) {
      refs.push_back(modules[slot].inner);
      refs.push_back(modules[slot].outer);
    }
    return refs;
  }

  // the four slot hits of a standard pixel track at radii 3, 7, 11 and 16 cm
  uint32_t addStandardPixelTrack(Event& event, uint32_t nHits = 4) {
    return addPixelTrack(event, {{3.f}, {7.f}, {11.f}, {16.f}}, nHits);
  }
  std::vector<Ref> pixelHitsOf(Event const& event, uint32_t pixelTrack) {
    std::vector<Ref> refs;
    refs.reserve(event.pixelTracks[pixelTrack].nSlots);
    for (uint32_t slot = 0; slot < event.pixelTracks[pixelTrack].nSlots; ++slot)
      refs.push_back(event.slot(pixelTrack, slot));
    return refs;
  }

  template <typename... Lists>
  std::vector<Ref> join(Lists const&... lists) {
    std::vector<Ref> all;
    (all.insert(all.end(), lists.begin(), lists.end()), ...);
    return all;
  }

  // positions in a list
  std::vector<Ref> pick(std::vector<Ref> const& list, std::vector<int> const& positions) {
    std::vector<Ref> picked;
    picked.reserve(positions.size());
    for (int position : positions)
      picked.push_back(list[position]);
    return picked;
  }

  // the list without the entry at position
  std::vector<Ref> without(std::vector<Ref> list, int position) {
    list.erase(list.begin() + position);
    return list;
  }

  Cases buildCases() {
    Cases cases;
    Event& event = cases.event;
    auto expect = [&](Expectation expectation) { cases.expectations.push_back(std::move(expectation)); };
    // the first pixel track (pLS row 0, its slot 0 at hitsIT row 0): its slot 0 is the outermost hit of a candidate
    // below, so that the search of the pixel track of a state hit is checked at its lower end
    const uint32_t first = addPixelTrack(event, {{250.f}, {7.f}, {11.f}, {16.f}}, 5);
    // a pixel track whose slot rows a T5 or a T4 must ignore
    const uint32_t ignored = addStandardPixelTrack(event);

    // identity: every hit fitted, inside out (some slots list their outer hit first)
    for (int last : {6, 7, 8}) {
      const auto modules = addLayers(event, 2, last);
      Candidate& candidate =
          event.addCandidate("T5, " + std::to_string(2 * (last - 1)) + " hits", LSTObjType::T5, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= last; ++slot)
        setModule(candidate, slot, modules[slot], slot % 2 == 0);
      event.setPixelSlots(candidate, ignored);
      candidate.pixelSeedIndex = ::lst::kInvalidU32Idx;
      const auto hits = hitsOf(modules, 2, last);
      expect({tc::Preparation::kFit, uint32_t(hits.size()), hits, otHit(hits.back()), {}, pixelHitsOf(event, ignored)});
    }
    {
      const auto modules = addLayers(event, 2, 5);
      Candidate& candidate = event.addCandidate("T4, 8 hits", LSTObjType::T4, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 5; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto hits = hitsOf(modules, 2, 5);
      expect({tc::Preparation::kFit, 8, hits, otHit(hits.back()), {}, {}});
    }
    {
      // an empty slot between the filled ones
      auto modules = addLayers(event, 2, 7);
      Candidate& candidate = event.addCandidate("T5, 10 hits, slot 4 empty", LSTObjType::T5, ::lst::kInvalidU32Idx);
      for (int slot : {2, 3, 5, 6, 7})
        setModule(candidate, slot, modules[slot]);
      const auto hits = join(hitsOf(modules, 2, 3), hitsOf(modules, 5, 7));
      expect({tc::Preparation::kFit, 10, hits, otHit(hits.back()), {}, {modules[4].inner, modules[4].outer}});
    }
    // the pixelSeedIndex of a T5 or a T4 is none, whatever the candidate holds
    {
      const auto modules = addLayers(event, 2, 6);
      Candidate& candidate = event.addCandidate("T5 holding pixelSeedIndex 7", LSTObjType::T5, 7);
      for (int slot = 2; slot <= 6; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto hits = hitsOf(modules, 2, 6);
      expect({tc::Preparation::kFit, 10, hits, otHit(hits.back()), {}, {}});
    }

    // pT3 and pT5: the pixel hits from hitsIT, then the OT hits
    {
      const uint32_t pixel = addStandardPixelTrack(event);
      const auto modules = addLayers(event, 2, 4);
      Candidate& candidate = event.addCandidate("pT3, 10 hits", LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto hits = join(pixelHitsOf(event, pixel), hitsOf(modules, 2, 4));
      expect({tc::Preparation::kFit, 10, hits, otHit(hits.back()), {}, {}});
    }
    {
      const uint32_t pixel = addStandardPixelTrack(event, 7);
      const auto modules = addLayers(event, 2, 6);
      Candidate& candidate = event.addCandidate("pT5, 14 hits, a pixel track of 7 hits", LSTObjType::pT5, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 6; ++slot)
        setModule(candidate, slot, modules[slot], true);
      const auto hits = join(pixelHitsOf(event, pixel), hitsOf(modules, 2, 6));
      expect({tc::Preparation::kFit, 14, hits, otHit(hits.back()), {}, {}});
    }
    {
      // 4 pixel and 14 OT hits: the pixel hits and 10 OT hits by the rounded stride 14 / 10 from the first, the
      // outermost forced: OT positions round(1.4 k) for k = 0..8, then 13
      const uint32_t pixel = addStandardPixelTrack(event);
      const auto modules = addLayers(event, 2, 8);
      Candidate& candidate = event.addCandidate("pT5, 18 hits: stride", LSTObjType::pT5, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 8; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto ot = hitsOf(modules, 2, 8);
      const auto hits = join(pixelHitsOf(event, pixel), pick(ot, {0, 1, 3, 4, 6, 7, 8, 10, 11, 13}));
      expect({tc::Preparation::kFit, 18, hits, otHit(ot.back()), {}, {}});
    }
    {
      // 4 pixel and 12 OT hits: OT positions round(1.2 k) for k = 0..8, then 11
      const uint32_t pixel = addStandardPixelTrack(event);
      const auto modules = addLayers(event, 2, 7);
      Candidate& candidate = event.addCandidate("pT5, 16 hits: stride", LSTObjType::pT5, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 7; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto ot = hitsOf(modules, 2, 7);
      const auto hits = join(pixelHitsOf(event, pixel), pick(ot, {0, 1, 2, 4, 5, 6, 7, 8, 10, 11}));
      expect({tc::Preparation::kFit, 16, hits, otHit(ot.back()), {}, {}});
    }

    // an OT hit of the pixel track that is also an OT hit of the candidate: the pLS copy, collected first, is kept
    {
      const auto modules = addLayers(event, 2, 4);
      const uint32_t pixel = addPixelTrack(event, {{3.f}, {7.f}, {11.f}, {0.f, int(modules[2].inner.row)}}, 5);
      Candidate& candidate = event.addCandidate("pT3, a pLS slot sharing an OT hit", LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      // 9 hits: one too many, the smallest separation is the innermost module (0.10 cm): its outer hit is dropped
      const auto collected = join(pixelHitsOf(event, pixel), hitsOf(modules, 2, 4));
      const auto all = without(collected, 4);  // the OT copy of the shared hit
      expect(
          {tc::Preparation::kFit, 9, without(all, 4), otHit(all.back()), {event.slot(pixel, 3)}, {modules[2].inner}});
    }
    {
      // the shared hit is the outermost hit: the state sits on the pLS copy, encoded by its OT index
      const auto modules = addLayers(event, 2, 4);
      const uint32_t pixel = addPixelTrack(event, {{3.f}, {7.f}, {11.f}, {0.f, int(modules[4].outer.row)}}, 4);
      Candidate& candidate = event.addCandidate("pT3, the outermost hit shared with a pLS slot", LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto pixels = pixelHitsOf(event, pixel);
      // 9 hits, the outer hit of the innermost module dropped
      const std::vector<Ref> hits = {pixels[0],
                                     pixels[1],
                                     pixels[2],
                                     modules[2].inner,
                                     modules[3].inner,
                                     modules[3].outer,
                                     modules[4].inner,
                                     pixels[3]};
      expect({tc::Preparation::kFit, 9, hits, otHit(modules[4].outer), {}, {modules[4].outer}});
    }
    // The shared hit's slot partner degenerate with it (the same transverse position, 0.3 cm apart in z): the pLS copy
    // stands for the shared hit, so the partner is tested against it, else the two would be consecutive nodes of no
    // transverse separation; the outer one of the two is dropped.
    for (bool sharedFirst : {true, false}) {
      // the partner outer: dropped, whether the shared hit is listed first in its slot or second
      auto modules = addLayers(event, 2, 4);
      modules[2] = addModule(event, layerRadius(2), 0.f, 0.f, 0.3f);
      const uint32_t pixel = addPixelTrack(event, {{3.f}, {7.f}, {11.f}, {0.f, int(modules[2].inner.row)}}, 5);
      Candidate& candidate =
          event.addCandidate(std::string("pT3, a shared OT hit with an outer degenerate partner, listed ") +
                                 (sharedFirst ? "first" : "second"),
                             LSTObjType::pT3,
                             0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot], !sharedFirst && slot == 2);
      // 8 hits: the pixel hits with the pLS copy of the shared hit, then modules 3 and 4
      const auto hits = join(pixelHitsOf(event, pixel), hitsOf(modules, 3, 4));
      expect({tc::Preparation::kFit,
              8,
              hits,
              otHit(modules[4].outer),
              {event.slot(pixel, 3)},
              {modules[2].inner, modules[2].outer}});
    }
    {
      // the partner inner: it takes the entry of the pLS copy, the outer hit of the pair
      auto modules = addLayers(event, 2, 4);
      modules[2] = addModule(event, layerRadius(2), 0.f, 0.f, 0.3f);
      const uint32_t pixel = addPixelTrack(event, {{3.f}, {7.f}, {11.f}, {0.f, int(modules[2].outer.row)}}, 5);
      Candidate& candidate =
          event.addCandidate("pT3, a shared OT hit with an inner degenerate partner", LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      // 8 hits: the three pixel hits, the partner, modules 3 and 4
      const auto pixels = pixelHitsOf(event, pixel);
      const auto hits = join(pick(pixels, {0, 1, 2}), std::vector<Ref>{modules[2].inner}, hitsOf(modules, 3, 4));
      expect({tc::Preparation::kFit,
              8,
              hits,
              otHit(modules[4].outer),
              {modules[2].inner},
              {event.slot(pixel, 3), modules[2].outer}});
    }

    // two equal indices in a slot count once
    {
      const auto modules = addLayers(event, 2, 5);
      Candidate& candidate =
          event.addCandidate("T4, a slot with two equal indices", LSTObjType::T4, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 5; ++slot)
        setModule(candidate, slot, modules[slot]);
      candidate.slots[3][1] = modules[3].inner;
      // 7 hits: the outer hit of the innermost module dropped
      expect(
          {tc::Preparation::kFit,
           7,
           {modules[2].inner, modules[3].inner, modules[4].inner, modules[4].outer, modules[5].inner, modules[5].outer},
           otHit(modules[5].outer),
           {},
           {}});
    }
    {
      // a pixel track with 3 slot hits: its last pixel slot repeats its third hit
      const uint32_t pixel = addPixelTrack(event, {{3.f}, {7.f}, {11.f}}, 3);
      const auto modules = addLayers(event, 2, 4);
      Candidate& candidate = event.addCandidate("pT3, a pixel track of 3 hits", LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto all = join(pixelHitsOf(event, pixel), hitsOf(modules, 2, 4));
      expect({tc::Preparation::kFit, 9, without(all, 4), otHit(all.back()), {}, {}});
    }

    // degenerate pairs: the outer hit of a pair closer than 1e-4 cm transversely is dropped (here 0.3 cm apart in z,
    // the farther one listed first), the inner one kept
    for (float dy : {0.f, 0.9e-4f, 1.1e-4f}) {
      auto modules = addLayers(event, 2, 6);
      modules[4] = addModule(event, layerRadius(4), 0.f, dy, 0.3f);
      Candidate& candidate =
          event.addCandidate("T5, a pair " + std::to_string(dy) + " cm apart", LSTObjType::T5, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 6; ++slot)
        setModule(candidate, slot, modules[slot], true);
      const auto all = hitsOf(modules, 2, 6);
      if (dy < 1e-4f) {
        // 9 hits: the outer hit of the innermost module dropped
        expect({tc::Preparation::kFit,
                9,
                without(without(all, 5), 1),
                otHit(all.back()),
                {modules[4].inner},
                {modules[4].outer}});
      } else {
        expect({tc::Preparation::kFit, 10, all, otHit(all.back()), {modules[4].inner, modules[4].outer}, {}});
      }
    }
    {
      // a degenerate outermost pair: the state sits on its inner hit
      auto modules = addLayers(event, 2, 6);
      modules[6] = addModule(event, layerRadius(6), 0.f, 0.f, 0.3f);
      Candidate& candidate =
          event.addCandidate("T5, a degenerate outermost pair", LSTObjType::T5, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 6; ++slot)
        setModule(candidate, slot, modules[slot], true);
      const auto all = hitsOf(modules, 2, 6);
      expect({tc::Preparation::kFit,
              9,
              without(without(all, 9), 1),
              otHit(modules[6].inner),
              {modules[6].inner},
              {modules[6].outer}});
    }
    {
      // one hit too many (slot 3 lists its inner hit twice) with the smallest separation (0.01 cm) at the outermost
      // pair: that pair is kept, the next smallest (the innermost module, 0.10 cm) loses its outer hit
      auto modules = addLayers(event, 2, 6);
      modules[6] = addModule(event, layerRadius(6), 0.01f);
      Candidate& candidate =
          event.addCandidate("T5, one hit too many, the closest pair outermost", LSTObjType::T5, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 6; ++slot)
        setModule(candidate, slot, modules[slot]);
      candidate.slots[3][1] = modules[3].inner;
      expect({tc::Preparation::kFit,
              9,
              {modules[2].inner,
               modules[3].inner,
               modules[4].inner,
               modules[4].outer,
               modules[5].inner,
               modules[5].outer,
               modules[6].inner,
               modules[6].outer},
              otHit(modules[6].outer),
              {},
              {}});
    }

    // the smallest sizes, and 13 and 15 hits
    {
      const auto modules = addLayers(event, 2, 3);
      Candidate& candidate = event.addCandidate("T4, 4 hits: too few", LSTObjType::T4, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 3; ++slot)
        setModule(candidate, slot, modules[slot]);
      expect({tc::Preparation::kTooFewHits, 4, {}, ::lst::kInvalidU32Idx, {}, {}});
    }
    {
      auto modules = addLayers(event, 2, 4);
      modules[3] = addModule(event, layerRadius(3), 0.f, 0.f, 0.3f);
      Candidate& candidate = event.addCandidate("T4, 5 hits", LSTObjType::T4, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      expect({tc::Preparation::kFit,
              5,
              {modules[2].inner, modules[2].outer, modules[3].inner, modules[4].inner, modules[4].outer},
              otHit(modules[4].outer),
              {},
              {}});
    }
    {
      const auto modules = addLayers(event, 2, 4);
      Candidate& candidate = event.addCandidate("T4, 6 hits", LSTObjType::T4, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto hits = hitsOf(modules, 2, 4);
      expect({tc::Preparation::kFit, 6, hits, otHit(hits.back()), {}, {}});
    }
    {
      auto modules = addLayers(event, 2, 8);
      modules[5] = addModule(event, layerRadius(5), 0.f, 0.f, 0.3f);
      Candidate& candidate = event.addCandidate("T5, 13 hits", LSTObjType::T5, ::lst::kInvalidU32Idx);
      for (int slot = 2; slot <= 8; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto all = hitsOf(modules, 2, 8);
      expect({tc::Preparation::kFit, 13, without(without(all, 7), 1), otHit(all.back()), {}, {}});
    }
    {
      const uint32_t pixel = addStandardPixelTrack(event);
      auto modules = addLayers(event, 2, 7);
      modules[5] = addModule(event, layerRadius(5), 0.f, 0.f, 0.3f);
      Candidate& candidate = event.addCandidate("pT5, 15 hits", LSTObjType::pT5, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 7; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto all = join(pixelHitsOf(event, pixel), hitsOf(modules, 2, 7));
      // the outer hit of slot 5 is degenerate, then the innermost module loses its outer hit (never a pixel hit)
      expect({tc::Preparation::kFit, 15, without(without(all, 11), 5), otHit(all.back()), {}, {}});
    }

    // a pLS is not fitted
    {
      const uint32_t pixel = addStandardPixelTrack(event);
      Candidate& candidate = event.addCandidate("pLS", LSTObjType::pLS, 0);
      event.setPLSSlots(candidate, pixel);
      expect({tc::Preparation::kPLS, 0, {}, ::lst::kInvalidU32Idx, {}, {}});
    }

    // the state on a pixel hit (pixel hits placed outside the OT hits): its position in its pixel track, from its
    // slot and the hit count of the track: slot 3 of a 6-hit track is hit 5, slot 2 is hit 2
    {
      const uint32_t pixel = addPixelTrack(event, {{3.f}, {7.f}, {11.f}, {250.f}}, 6);
      const auto modules = addLayers(event, 2, 4);
      Candidate& candidate = event.addCandidate("pT3, the state on pixel slot 3", LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto pixels = pixelHitsOf(event, pixel);
      const auto hits = join(pick(pixels, {0, 1, 2}), hitsOf(modules, 2, 4), pick(pixels, {3}));
      expect({tc::Preparation::kFit, 10, hits, ::lst::encodeStateHitPixel(5), {}, {}});
    }
    {
      const uint32_t pixel = addPixelTrack(event, {{3.f}, {7.f}, {260.f}, {250.f}}, 6);
      const auto modules = addLayers(event, 2, 4);
      Candidate& candidate = event.addCandidate("pT3, the state on pixel slot 2", LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, pixel);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto pixels = pixelHitsOf(event, pixel);
      const auto hits = join(pick(pixels, {0, 1}), hitsOf(modules, 2, 4), pick(pixels, {3, 2}));
      expect({tc::Preparation::kFit, 10, hits, ::lst::encodeStateHitPixel(2), {}, {}});
    }
    {
      // slot 0 of the first pixel track, the hit at its firstHit: hit 0
      const auto modules = addLayers(event, 2, 4);
      Candidate& candidate =
          event.addCandidate("pT3, the state on slot 0 of the first pixel track", LSTObjType::pT3, 0);
      event.setPixelSlots(candidate, first);
      for (int slot = 2; slot <= 4; ++slot)
        setModule(candidate, slot, modules[slot]);
      const auto pixels = pixelHitsOf(event, first);
      const auto hits = join(pick(pixels, {1, 2, 3}), hitsOf(modules, 2, 4), pick(pixels, {0}));
      expect({tc::Preparation::kFit, 10, hits, ::lst::encodeStateHitPixel(0), {}, {}});
    }
    return cases;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Random candidates: OT modules at increasing random radii, some degenerate, some sharing a hit with the pixel track
  // or listing one hit twice; pixel tracks of 3 to 12 hits, some with an OT hit in a slot.

  Event buildRandom(int nCandidates, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.f, 1.f);
    Event event;
    for (int c = 0; c < nCandidates; ++c) {
      const float pick = uniform(rng);
      const LSTObjType type = pick < 0.3f    ? LSTObjType::T5
                              : pick < 0.5f  ? LSTObjType::T4
                              : pick < 0.75f ? LSTObjType::pT3
                              : pick < 0.95f ? LSTObjType::pT5
                                             : LSTObjType::pLS;
      const int nLayers = type == LSTObjType::T4 ? 4 : type == LSTObjType::pT3 ? 3 : 5 + int(3 * uniform(rng));
      std::vector<int> slots;
      for (int slot = 2; slot < int(kLayers); ++slot)
        slots.push_back(slot);
      std::shuffle(slots.begin(), slots.end(), rng);
      slots.resize(std::min<std::size_t>(nLayers, slots.size()));
      std::sort(slots.begin(), slots.end());
      std::vector<Module> modules(kLayers);
      float r = 20.f;
      for (int slot : slots) {
        r += 3.f + 12.f * uniform(rng);
        const float kind = uniform(rng);
        if (kind < 0.15f)
          modules[slot] = addModule(event, r, 0.f, 0.f, 0.2f);  // degenerate
        else if (kind < 0.25f)
          modules[slot] = addModule(event, r, 0.f, 2e-4f * uniform(rng), 0.2f);  // around the tolerance
        else
          modules[slot] = addModule(event, r, 0.005f + 0.5f * uniform(rng), 0.01f * uniform(rng));
      }
      const bool pixelSlots = type == LSTObjType::pT3 || type == LSTObjType::pT5 || type == LSTObjType::pLS;
      uint32_t pixel = 0;
      if (pixelSlots) {
        const uint32_t nHits = 3 + int(10 * uniform(rng));
        std::vector<SlotHit> slotHits = {{3.f}, {7.f}, {11.f}, {16.f}};
        slotHits.resize(std::min(nHits, 4u));
        if (nHits > 3 && uniform(rng) < 0.4f && !slots.empty())
          slotHits[3] = {0.f, int(modules[slots[0]].inner.row)};
        pixel = addPixelTrack(event, slotHits, nHits);
      }
      Candidate& candidate =
          event.addCandidate("random " + std::to_string(c), type, pixelSlots ? 0u : uint32_t(uniform(rng) * 10));
      if (pixelSlots)
        event.setPixelSlots(candidate, pixel);
      for (int slot : slots) {
        setModule(candidate, slot, modules[slot], uniform(rng) < 0.5f);
        if (uniform(rng) < 0.05f)
          candidate.slots[slot][1] = candidate.slots[slot][0];
      }
    }
    return event;
  }

}  // namespace

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  const Cases cases = buildCases();
  const tcSeedStateTest::Event randomEvent = buildRandom(400, 1234);
  bool failed = false;
  for (auto const& device : devices) {
    Queue queue(device);
    std::cout << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " " << alpaka::getName(device) << "\n";
    Checks checks;
    const std::vector<Output> outputs = run(queue, cases.event);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      checkProperties(checks, cases.event, cases.event.candidates[i], outputs[i]);
      checkExpectation(checks, cases.event, cases.event.candidates[i], outputs[i], cases.expectations[i]);
    }
    std::cout << "  " << outputs.size() << " hand-built candidates: " << checks.count - checks.failures << " of "
              << checks.count << " checks passed\n";
    Checks randomChecks;
    const std::vector<Output> randomOutputs = run(queue, randomEvent);
    int counts[16] = {};
    for (std::size_t i = 0; i < randomOutputs.size(); ++i) {
      checkProperties(randomChecks, randomEvent, randomEvent.candidates[i], randomOutputs[i]);
      ++counts[std::min(randomOutputs[i].selection.nFit, 15u)];
    }
    std::cout << "  " << randomOutputs.size() << " random candidates: " << randomChecks.count - randomChecks.failures
              << " of " << randomChecks.count << " checks passed; fit sizes:";
    for (int n = 0; n < 16; ++n)
      if (counts[n] > 0)
        std::cout << " " << n << ":" << counts[n];
    std::cout << "\n";
    failed = failed || checks.failures > 0 || randomChecks.failures > 0;
  }
  std::cout << (failed ? "FAILED" : "passed") << "\n";
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
