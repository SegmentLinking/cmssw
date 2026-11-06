// Unit test of the per-track work of the device LST input (RecoTracker/LST/plugins/alpaka/LSTInputDeviceTrackFit.h),
// run in a kernel on every device of the backend, in the order of the producer's kernels: the hits of the fits of a
// pixel track (prepareTrack), the Broken Line Fit with its state at the last hit and its checks (fitState), and the pLS
// computed from the state (makePixelSegment). The tracks are synthetic helices through pixel and outer-tracker layers
// in a uniform 3.8 T field, with module frames as the detector's and the hit errors of its sensors.
//
// Checked: the state against the noiseless truth at the last hit; its invariance under rotations of the input about
// the beam axis; every status of the checks; the range of the stored phi; the sign of deltaPhi and the pixel type it
// gives; the size of the full fit of each hit count (4, 6, 8, 10, 12 or 14 hits) and the hits the stride keeps, the
// first hit first and the last hit the last node; the inner-tracker-only selection and fit; the pLS hit slots. Each
// tolerance is defined and justified where it is declared; the largest deviation of each check is printed with it.

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceTrackFit.h"
#include "RecoTracker/LST/test/BLFTestStates.h"
#include "RecoTracker/LST/test/alpaka/LSTInputDeviceTestTracks.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDeviceTest;

  // ---------------------------------------------------------------------------------------------------------------
  // Tolerances of the rotations and of the pLS (those of the noiseless truth, of the float storage, of the hit
  // covariances and of the closest approach are in LSTInputDeviceTestTracks.h)
  //
  // Rotations by multiples of 90 degrees map every float input exactly (coordinates swap and change sign), so only
  // the order of the double operations inside the fit differs: about 1e-15 relative, then rounded to float. A
  // parameter may then differ by a float ulp; 1e-4 sigma or two ulps of the value covers it with a large margin. The
  // covariance elements: 1e-5 of sqrt(C_ii C_jj), a hundred float roundings.
  constexpr double kRotationSigmas = 1e-4;
  constexpr double kRotationPosition = 2e-5;  // cm, two float ulps at |x| < 128 cm
  constexpr double kRotationCovariance = 1e-5;
  // A rotation by a generic angle rounds the rotated hits to float again: the noiseless-truth bound applies.
  constexpr double kGenericRotationSigmas = kTruthSigmas;
  constexpr double kGenericRotationPosition = kTruthPosition;
  constexpr double kGenericRotationCovariance = 1e-3;  // the hit weights change by the input rounding, ~1e-6
  // deltaPhi is a float difference of angles of size pi: four float ulps at pi
  constexpr double kPLSDeltaPhi = 1e-6;  // rad
  // deltaPhi against the truth: the phi tolerance plus the azimuth of the position tolerance at r >= 10 cm
  constexpr double kDeltaPhiPosition = 1e-5;  // rad
  // the uniform-field identity ptPCA = ptIn, to float rounding (as the closest-approach unit test)
  constexpr double kPtIdentity = 5e-6;

  // rotations about the beam axis: exact for multiples of 90 degrees (quarterTurns), else by `angle` in double
  void rotateVector(double angle, int quarterTurns, float& x, float& y) {
    if (quarterTurns >= 0) {
      for (int k = 0; k < quarterTurns; ++k) {
        const float oldX = x;
        x = -y;
        y = oldX;
      }
      return;
    }
    const double c = std::cos(angle), s = std::sin(angle);
    const double newX = c * x - s * y, newY = s * x + c * y;
    x = newX;
    y = newY;
  }

  TestHit rotateHit(TestHit hit, double angle, int quarterTurns) {
    rotateVector(angle, quarterTurns, hit.x, hit.y);
    rotateVector(angle, quarterTurns, hit.origin[0], hit.origin[1]);
    for (int axis = 0; axis < 3; ++axis)
      rotateVector(angle, quarterTurns, hit.rotation[3 * axis], hit.rotation[3 * axis + 1]);
    return hit;
  }

  Settings rotateSettings(Settings settings, double angle, int quarterTurns) {
    rotateVector(angle, quarterTurns, settings.beamSpotX, settings.beamSpotY);
    rotateVector(angle, quarterTurns, settings.beamSlopeX, settings.beamSlopeY);
    return settings;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Test cases and their checks

  enum class Kind { kTruth, kRotated, kStatus, kStride, kIT };

  struct Case {
    std::string label;
    Kind kind;
    TestTrack track;
    std::vector<TestHit> hits;
    GeneratedTrack generated;  // the truth, when the hits are a generated helix
    bool noiseless = false;
    // kRotated: the case it is a rotation of, and the rotation
    int base = -1;
    double angle = 0.;
    int quarterTurns = -1;
    // expectations
    SeedStateStatus expectedStatus = SeedStateStatus::kOk;
    bool expectPrepared = true;
    // kIT: whether the case is an inner-tracker-only candidate, as its name says (the selection itself is checked
    // against the rule on every case)
    bool expectITCandidate = false;
    // the status of the inner-tracker-only fit when it runs
    SeedStateStatus expectedITFitStatus = SeedStateStatus::kOk;
    // the expected status holds only when every hit is fitted
    bool statusOfAllHits = false;
  };

  // the slots, hitDetBits and the hit counts of a prepared track against its source hits
  void checkSlots(Checks& checks, Case const& c, TestOutput const& out) {
    const uint32_t nHits = c.track.nHits;
    PixelSegmentCandidate const& candidate = out.candidate;
    const uint32_t nSlots = std::min<uint32_t>(nHits, ::lst::kMaxPLSHitsInHitsSoA);
    checks("slots: nHits, nSlots", 0., "").require(candidate.nHits == nHits && candidate.nSlots == nSlots, c.label);
    std::vector<uint32_t> positions(out.nFit);
    fit::fitHitPositions(nHits, out.nFit, positions.data());
    for (uint32_t slot = 0; slot < nSlots; ++slot) {
      const uint32_t position = ::lst::pixelSegmentHitOfSlot(slot, nSlots, nHits);
      TestHit const& source = c.hits[position];
      FitHit const& hit = candidate.slotHits[slot];
      checks("slots: hit index, detId, cluster size, position", 0., "")
          .require(candidate.slotHitIndex[slot] == source.index && candidate.slotDetId[slot] == source.detId &&
                       candidate.slotClusterSize[slot] == source.clusterSize && same(hit.x, source.x) &&
                       same(hit.y, source.y) && same(hit.z, source.z),
                   c.label);
      // a slot hit that entered the full fit carries exactly the fit's values
      for (uint32_t i = 0; i < out.nFit; ++i)
        if (positions[i] == position) {
          bool identical = true;
          for (int k = 0; k < 6; ++k)
            identical = identical && same(hit.cov[k], out.input.hits[i].cov[k]);
          checks("slots: covariance identical to the fit's input", 0., "").require(identical, c.label);
        }
    }
    const uint32_t nBits = std::min<uint32_t>(nHits, ::lst::kMaxPLSHitBitsInHitsSoA);
    uint8_t bits = 0;
    for (uint32_t bit = 0; bit < nBits; ++bit)
      bits |= c.hits[::lst::pixelSegmentHitOfSlot(bit, nBits, nHits)].isOT << bit;
    checks("slots: hitDetBits", 0., "").require(candidate.hitDetBits == bits, c.label);
    uint32_t nIT = 0;
    for (auto const& hit : c.hits)
      nIT += hit.isOT ? 0 : 1;
    checks("slots: nITHits", 0., "").require(candidate.nITHits == nIT && out.nITHits == nIT, c.label);
  }

  // the inputs of the full fit against the source hits at the positions of the stride
  void checkFitInputs(Checks& checks, Case const& c, TestOutput const& out) {
    std::vector<uint32_t> positions(out.nFit);
    fit::fitHitPositions(c.track.nHits, out.nFit, positions.data());
    for (uint32_t i = 0; i < out.nFit; ++i) {
      TestHit const& source = c.hits[positions[i]];
      FitHit const& hit = out.input.hits[i];
      checks("fit inputs: positions of the selected hits", 0., "")
          .require(same(hit.x, source.x) && same(hit.y, source.y) && same(hit.z, source.z), c.label);
      float cov[6];
      fit::hitCovariance(frameOf(source), source.varX, source.varY, cov);
      // relative to the largest diagonal element of the hit
      const double scale = std::max({std::abs(double(cov[0])), std::abs(double(cov[2])), std::abs(double(cov[5]))});
      double deviation = 0.;
      for (int k = 0; k < 6; ++k)
        deviation = std::max(deviation, std::abs(double(hit.cov[k]) - cov[k]) / scale);
      checks("fit inputs: hit covariance, host vs kernel", kHitCovariance, "rel").add(deviation, c.label);
    }
  }

  // the stride of the pixel-track fit, as written in RecoTracker/PixelSeeding/plugins/alpaka/BrokenLineFit.h
  std::vector<uint32_t> referenceStride(uint32_t nHits, uint32_t hitsInFit) {
    std::vector<uint32_t> positions;
    float incr = std::max(1.f, float(nHits) / float(hitsInFit));
    float n = 0;
    for (uint32_t i = 0; i < hitsInFit; ++i) {
      int j = int(n + 0.5f);
      if (hitsInFit - 1 == i)
        j = nHits - 1;
      n += incr;
      positions.push_back(j);
    }
    return positions;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Building the cases

  struct Builder {
    double bField;
    Settings settings;
    std::mt19937 rng{20261019};
    std::vector<Case> cases;
    // per category (the label up to " n=", or the whole label): the cases kept, since a helix that does not cross
    // every surface of its layout is dropped
    std::map<std::string, int> kept;

    // a case from a generated track; false if the helix did not cross every surface
    bool add(std::string const& label,
             Kind kind,
             Helix const& helix,
             std::vector<Surface> const& surfaces,
             bool noiseless,
             Case expectations = {}) {
      GeneratedTrack generated = generate(helix, surfaces, bField, noiseless ? nullptr : &rng);
      const std::string category = label.substr(0, label.find(" n="));
      kept.try_emplace(category, 0);
      if (!generated.complete)
        return false;
      ++kept[category];
      Case c = expectations;
      c.label = label;
      c.kind = kind;
      c.noiseless = noiseless;
      c.hits = generated.hits;
      c.generated = generated;
      c.track = TestTrack{0, static_cast<uint32_t>(c.hits.size()), 1, settings};
      cases.push_back(c);
      return true;
    }

    // the prompt helix from the beam line at z = settings.beamSpotZ + dz
    Helix prompt(double pt, int charge, double phi0, double cot, double dz = 0.) const {
      return Helix{{settings.beamSpotX + settings.beamSlopeX * dz,
                    settings.beamSpotY + settings.beamSlopeY * dz,
                    settings.beamSpotZ + dz},
                   phi0,
                   pt,
                   cot,
                   charge};
    }
  };

  std::string describe(char const* what, int n, double pt, int q, double cot, double phi0) {
    std::ostringstream label;
    label << what << " n=" << n << " pt=" << pt << " q=" << q << " cot=" << cot << " phi0=" << std::setprecision(4)
          << phi0;
    return label.str();
  }

  void buildCases(Builder& builder) {
    std::uniform_real_distribution<double> uniform(0., 1.);
    auto& rng = builder.rng;

    // noiseless and smeared helices, every full-fit size, both charges, central and forward
    for (int n = 4; n <= 14; ++n)
      for (double pt : {0.9, 1.5, 4., 20.})
        for (int q : {-1, 1}) {
          for (double cot : {-0.6, 0.05, 0.9}) {
            const double phi0 = 2. * kPi * uniform(rng) - kPi;
            const Helix helix = builder.prompt(pt, q, phi0, cot, 10. * uniform(rng) - 5.);
            builder.add(describe("central", n, pt, q, cot, phi0), Kind::kTruth, helix, centralLayout(n), true);
            builder.add(describe("central smeared", n, pt, q, cot, phi0), Kind::kTruth, helix, centralLayout(n), false);
          }
          if (n <= 11)
            for (double cot : {-2.5, 2.5}) {
              const double phi0 = 2. * kPi * uniform(rng) - kPi;
              const Helix helix = builder.prompt(pt, q, phi0, cot);
              builder.add(describe("forward", n, pt, q, cot, phi0), Kind::kTruth, helix, forwardLayout(n, cot), true);
            }
        }

    // phi at the last hit within 0.02 rad of pi, on both sides: the stored phi is reduced to [-pi, pi]
    for (double pt : {1.5, 4.})
      for (int q : {-1, 1})
        for (double offset : {-0.02, -1e-3, 1e-3, 0.02}) {
          const Helix straight = builder.prompt(pt, q, 0., 0.3);
          const GeneratedTrack probe = generate(straight, centralLayout(8), builder.bField, nullptr);
          const double turn = probe.truth.back().phi;
          const double phi0 = kPi + offset - turn;
          builder.add(describe("phi near pi", 8, pt, q, 0.3, phi0) + " offset=" + std::to_string(offset),
                      Kind::kTruth,
                      builder.prompt(pt, q, phi0, 0.3),
                      centralLayout(8),
                      true);
        }

    // rotations of the input about the beam axis, of noiseless and smeared tracks
    const std::size_t nTruth = builder.cases.size();
    for (std::size_t i = 0; i < nTruth; i += 3) {
      for (int quarterTurns : {1, 2, 3, -1}) {
        Case rotated = builder.cases[i];
        rotated.kind = Kind::kRotated;
        rotated.base = i;
        rotated.quarterTurns = quarterTurns;
        rotated.angle = quarterTurns >= 0 ? quarterTurns * 0.5 * kPi : 0.7;
        for (auto& hit : rotated.hits)
          hit = rotateHit(hit, rotated.angle, quarterTurns);
        rotated.track.settings = rotateSettings(rotated.track.settings, rotated.angle, quarterTurns);
        rotated.label += quarterTurns >= 0 ? " rotated by " + std::to_string(quarterTurns) + " x 90 deg"
                                           : std::string(" rotated by 0.7 rad");
        builder.cases.push_back(rotated);
      }
    }

    // more than 14 hits: the stride, the last hit last; up to the largest supported count
    for (int nHits : {15, 16, 18, 20, 21, 25, 28, 29, 40, 64, 128, 255}) {
      std::vector<Surface> surfaces;
      for (int i = 0; i < nHits; ++i) {
        const double r = 3. + 105. * i / (nHits - 1);
        surfaces.push_back(r < 20. ? pixelBarrel(r) : otBarrel(r));
      }
      builder.add(
          describe("stride", nHits, 10., 1, 0.2, 0.4), Kind::kStride, builder.prompt(10., 1, 0.4, 0.2), surfaces, true);
    }

    // statuses
    const std::vector<Surface> layout8 = centralLayout(8);
    {
      Case expectations;
      expectations.expectedStatus = SeedStateStatus::kNotFitted;
      expectations.expectPrepared = false;
      builder.add("no EDM track", Kind::kStatus, builder.prompt(4., 1, 1., 0.2), layout8, true, expectations);
      builder.cases.back().track.hasEdmTrack = 0;
      builder.add("3 hits",
                  Kind::kStatus,
                  builder.prompt(4., 1, 1., 0.2),
                  std::vector<Surface>(layout8.begin(), layout8.begin() + 3),
                  true,
                  expectations);
      std::vector<Surface> many;
      for (int i = 0; i < 256; ++i)
        many.push_back(otBarrel(3. + 105. * i / 255.));
      builder.add("256 hits", Kind::kStatus, builder.prompt(10., 1, 1., 0.2), many, true, expectations);
    }
    {
      Case expectations;
      expectations.expectedStatus = SeedStateStatus::kNonFinite;
      builder.add("NaN hit position", Kind::kStatus, builder.prompt(4., -1, 2., 0.2), layout8, true, expectations);
      builder.cases.back().hits[3].x = std::numeric_limits<float>::quiet_NaN();
    }
    {
      // negative local variances make every hit weight negative: the solve stays finite, the variances do not stay
      // positive
      Case expectations;
      expectations.expectedStatus = SeedStateStatus::kNonPositiveVariance;
      builder.add(
          "negative hit variances", Kind::kStatus, builder.prompt(20., 1, -2., 0.2), layout8, true, expectations);
      for (auto& hit : builder.cases.back().hits) {
        hit.varX = -hit.varX;
        hit.varY = -hit.varY;
      }
    }
    {
      // A zigzag: the three hits that set the fit's charge (first, middle, last) bend one way by 1 um, the two others
      // lie 50 um on the other side; with 100 um errors and the 20 GeV multiple scattering, the fitted curvature has
      // the opposite sign.
      Case expectations;
      expectations.expectedStatus = SeedStateStatus::kChargeMismatch;
      Case c = expectations;
      c.label = "zigzag";
      c.kind = Kind::kStatus;
      const double xs[5] = {3., 7., 11., 15., 19.}, ys[5] = {0., -50e-4, 1e-4, -50e-4, 0.};
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
        hit.detId = ::lst::kPixelModuleId;
        hit.clusterSize = 1;
        c.hits.push_back(hit);
      }
      c.track = TestTrack{0, 5, 1, builder.settings};
      c.statusOfAllHits = true;  // with fewer hits, the status depends on the hits dropped
      builder.cases.push_back(c);
    }

    // inner-tracker-only candidates: last hit in the OT and at most maxITHitsToDropOTHits IT hits
    auto itLayout = [](int nIT, int nOT) {
      std::vector<Surface> surfaces;
      for (int i = 0; i < nIT; ++i)
        surfaces.push_back(pixelBarrel(kPixelBarrelRadii[i]));
      for (int i = 0; i < nOT; ++i)
        surfaces.push_back(otBarrel(kOTBarrelRadii[i]));
      return surfaces;
    };
    for (int nOT = 1; nOT <= 6; ++nOT)
      for (double pt : {0.9, 4.})
        for (int q : {-1, 1}) {
          Case expectations;
          expectations.expectITCandidate = true;
          const double phi0 = 2. * kPi * uniform(rng) - kPi;
          builder.add(describe("3 IT + OT", 3 + nOT, pt, q, 0.4, phi0),
                      Kind::kIT,
                      builder.prompt(pt, q, phi0, 0.4),
                      itLayout(3, nOT),
                      true,
                      expectations);
        }
    {
      Case expectations;
      expectations.expectITCandidate = true;  // too few IT hits for the fit
      builder.add("2 IT + 4 OT", Kind::kIT, builder.prompt(4., 1, 0.3, 0.4), itLayout(2, 4), true, expectations);
      builder.add("1 IT + 4 OT", Kind::kIT, builder.prompt(4., 1, 0.3, 0.4), itLayout(1, 4), true, expectations);
      builder.add("0 IT + 5 OT", Kind::kIT, builder.prompt(4., 1, 0.3, 0.4), itLayout(0, 5), true, expectations);
      expectations.expectITCandidate = false;
      builder.add("4 IT + 2 OT", Kind::kIT, builder.prompt(4., -1, 0.3, 0.4), itLayout(4, 2), true, expectations);
      builder.add("4 IT", Kind::kIT, builder.prompt(4., -1, 0.3, 0.4), itLayout(4, 0), true, expectations);
      builder.add("3 IT + 3 OT, maxITHitsToDropOTHits = 2",
                  Kind::kIT,
                  builder.prompt(4., -1, 0.3, 0.4),
                  itLayout(3, 3),
                  true,
                  expectations);
      builder.cases.back().track.settings.maxITHitsToDropOTHits = 2;
    }
    {
      // a NaN on an inner-tracker hit: both fits non-finite
      Case expectations;
      expectations.expectedStatus = SeedStateStatus::kNonFinite;
      expectations.expectITCandidate = true;
      expectations.expectedITFitStatus = SeedStateStatus::kNonFinite;
      const Helix helix = builder.prompt(4., 1, 0.3, 0.4);
      builder.add("3 IT + 3 OT, NaN IT hit", Kind::kStatus, helix, itLayout(3, 3), true, expectations);
      builder.cases.back().hits[1].z = std::numeric_limits<float>::quiet_NaN();
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Evaluation

  bool isOk(FittedState const& state) { return state.status == static_cast<uint8_t>(SeedStateStatus::kOk); }

  bool evaluate(std::vector<Case> const& cases, TestOutput const* outputs, double bField) {
    Checks checks;
    int nNotConvergedPCA = 0, nBeyondTurn = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
      Case const& c = cases[i];
      TestOutput const& out = outputs[i];
      FittedState const& state = out.state;
      const uint32_t nHits = c.track.nHits;

      // statuses and the prepared fits
      checks("prepared iff an EDM track and 4 to 255 hits", 0., "").require(out.prepared == c.expectPrepared, c.label);
      if (c.kind == Kind::kTruth && !c.noiseless) {
        // smeared hits: the 1/pT resolution can flip the sign of a stiff track's curvature; not asserted
      } else if (c.statusOfAllHits && expectedFitSize(nHits) < nHits) {
        // the expected status needs every hit fitted; not asserted
      } else if (c.kind != Kind::kRotated) {
        checks("status as expected", 0., "")
            .require(state.status == static_cast<uint8_t>(c.expectedStatus),
                     c.label + " status " + statusName(state.status));
      }
      if (!out.prepared) {
        checks("unprepared: kNotFitted, nFit 0, NaN kappa, invalid stateHit", 0., "")
            .require(state.status == static_cast<uint8_t>(SeedStateStatus::kNotFitted) && state.nFit == 0 &&
                         std::isnan(state.row.kappa) && state.stateHit == ::lst::kInvalidU32Idx &&
                         out.itState.status == static_cast<uint8_t>(SeedStateStatus::kNotFitted),
                     c.label);
        continue;
      }
      checks("full fit: nFit = the largest fit size not above nHits, stateHit = last hit", 0., "")
          .require(out.nFit == expectedFitSize(nHits) && state.stateHit == ::lst::encodeStateHitPixel(nHits - 1) &&
                       (state.status == static_cast<uint8_t>(SeedStateStatus::kNonFinite) || state.nFit == out.nFit),
                   c.label);
      {
        // the hits kept: the stride of the pixel-track fit, increasing, the first hit first and the last hit the
        // last node
        std::vector<uint32_t> positions(out.nFit);
        fit::fitHitPositions(nHits, out.nFit, positions.data());
        bool increasing = !positions.empty() && positions.front() == 0 && positions.back() == nHits - 1;
        for (uint32_t k = 1; k < positions.size(); ++k)
          increasing = increasing && positions[k] > positions[k - 1];
        checks("fitted hits: the stride of the pixel-track fit, first hit first, last hit last", 0., "")
            .require(increasing && positions == referenceStride(nHits, out.nFit), c.label);
        checks("fitted hits: the last fit input is the last hit", 0., "")
            .require(out.nFit > 0 && same(out.input.hits[out.nFit - 1].x, c.hits[nHits - 1].x) &&
                         same(out.input.hits[out.nFit - 1].y, c.hits[nHits - 1].y) &&
                         same(out.input.hits[out.nFit - 1].z, c.hits[nHits - 1].z),
                     c.label);
      }
      checkSlots(checks, c, out);
      checkFitInputs(checks, c, out);

      // the inner-tracker-only selection, against the rule of LSTOutputConverter
      std::vector<uint32_t> itPositions;
      for (uint32_t k = 0; k < nHits; ++k)
        if (!c.hits[k].isOT)
          itPositions.push_back(k);
      const bool expectCandidate =
          c.hits[nHits - 1].isOT && itPositions.size() <= c.track.settings.maxITHitsToDropOTHits;
      const bool expectFit = expectCandidate && itPositions.size() == kITFitHits;
      checks("IT-only: candidate iff last hit OT and nIT <= maxITHitsToDropOTHits; fit iff also nIT = 3", 0., "")
          .require(out.itCandidate == expectCandidate && out.itFit == expectFit, c.label);
      if (c.kind == Kind::kIT)
        checks("IT-only: the case is what it is meant to test", 0., "")
            .require(expectCandidate == c.expectITCandidate, c.label);
      const SeedStateStatus expectedITStatus = expectFit ? c.expectedITFitStatus : SeedStateStatus::kNotFitted;
      checks("IT-only: status as expected", 0., "")
          .require(out.itState.status == static_cast<uint8_t>(expectedITStatus),
                   c.label + " status " + statusName(out.itState.status));
      checks("IT-only: stateHit = last IT hit", 0., "")
          .require(out.itState.stateHit ==
                       (itPositions.empty() ? ::lst::kInvalidU32Idx : ::lst::encodeStateHitPixel(itPositions.back())),
                   c.label);
      if (expectFit && isOk(out.itState)) {
        checks("IT-only: nFit = 3", 0., "").require(out.itState.nFit == kITFitHits, c.label);
        checkTruth(checks, "IT-only ", out.itState, c.generated, itPositions.back(), bField, c.label);
        bool identical = true;
        for (uint32_t k = 0; k < kITFitHits; ++k) {
          TestHit const& source = c.hits[itPositions[k]];
          identical = identical && same(out.input.itHits[k].x, source.x) && same(out.input.itHits[k].y, source.y) &&
                      same(out.input.itHits[k].z, source.z);
        }
        checks("IT-only: the fitted hits are the IT hits", 0., "").require(identical, c.label);
      }
      if (!expectFit)
        checks("IT-only: not fitted -> NaN kappa, nFit 0", 0., "")
            .require(std::isnan(out.itState.row.kappa) && out.itState.nFit == 0, c.label);

      // the stride
      if (c.kind == Kind::kStride) {
        std::vector<uint32_t> positions(out.nFit);
        fit::fitHitPositions(nHits, out.nFit, positions.data());
        bool increasing = positions.front() == 0 && positions.back() == nHits - 1;
        for (uint32_t k = 1; k < out.nFit; ++k)
          increasing = increasing && positions[k] > positions[k - 1];
        checks("stride: 14 increasing positions, first and last hit", 0., "")
            .require(out.nFit == kMaxFitHits && increasing, c.label);
        checks("stride: as the pixel-track fit", 0., "")
            .require(positions == referenceStride(nHits, kMaxFitHits), c.label);
      }

      if (!isOk(state) || !c.noiseless || c.kind == Kind::kRotated)
        continue;

      // the noiseless truth at the last hit
      checkTruth(checks, "", state, c.generated, nHits - 1, bField, c.label);

      // the pLS: deltaPhi and its sign, the pixel type, the uniform-field identities
      Helix const& helix = c.generated.helix;
      Truth const& truth = c.generated.truth.back();
      const double turn = bField / helix.pt * truth.s;
      if (turn > blfTestStates::kMaxTurnAngle) {
        ++nBeyondTurn;
        continue;
      }
      PixelSegmentCandidate const& candidate = out.candidate;
      if (candidate.pcaStatus != ::lst::PixelSegmentClosestApproach::kValid) {
        ++nNotConvergedPCA;
        checks("pLS: closest approach valid within the turn cut", 0., "").require(false, c.label);
        continue;
      }
      const double ptIn = bField / std::abs(state.row.kappa);
      const bool expectPLS = ptIn > c.track.settings.ptCut;  // ptErr > 0 widens the acceptance only
      checks("pLS: made for ptIn > ptCut", 0., "")
          .require(!expectPLS || candidate.pixelType != ::lst::PixelType::kInvalid, c.label);
      if (candidate.pixelType == ::lst::PixelType::kInvalid)
        continue;
      const double deltaPhiTruth = wrap(std::atan2(truth.position[1], truth.position[0]) - truth.phi);
      checks("pLS: deltaPhi vs truth", kTruthSigmas, "sigma")
          .add(std::max(0., std::abs(candidate.deltaPhi - deltaPhiTruth) - kDeltaPhiPosition) /
                   std::sqrt(state.row.covCircle[0]),
               c.label);
      // the sign of deltaPhi decides the low-pT pixel type; at high pT the turn to the last hit can be smaller than the
      // azimuth of the beam-spot offset seen from the origin
      if (ptIn < 2.)
        checks("pLS: sign(deltaPhi) = charge (prompt, ptIn < 2 GeV)", 0., "")
            .require((candidate.deltaPhi > 0.f) == (helix.charge > 0), c.label);
      const ::lst::PixelType expectedType =
          ptIn >= 2. ? ::lst::PixelType::kHighPt
                     : (helix.charge > 0 ? ::lst::PixelType::kLowPtPosCurv : ::lst::PixelType::kLowPtNegCurv);
      checks("pLS: pixel type from ptIn and the sign of deltaPhi", 0., "")
          .require(candidate.pixelType == expectedType, c.label);
      checks("pLS: ptIn = B/|kappa|, phi = stored phi, charge", 0., "")
          .require(candidate.ptIn == c.track.settings.bField / std::abs(state.row.kappa) &&
                       candidate.phi == state.row.phi && candidate.charge == state.row.charge,
                   c.label);
      // the device asinh against the host's: a few ulps
      checks("pLS: eta = asinh(cot)", 4., "ulp")
          .add(std::abs(candidate.eta - std::asinh(state.row.cotTheta)) / ulp(candidate.eta), c.label);
      checks("pLS: ptPCA = ptIn (uniform field)", kPtIdentity, "rel")
          .add(std::abs(candidate.kinematics.ptPCA - candidate.ptIn) / candidate.ptIn, c.label);
    }

    // rotations
    for (std::size_t i = 0; i < cases.size(); ++i) {
      Case const& c = cases[i];
      if (c.kind != Kind::kRotated)
        continue;
      FittedState const& rotated = outputs[i].state;
      FittedState const& base = outputs[c.base].state;
      checks("rotation: same status", 0., "").require(rotated.status == base.status, c.label);
      if (!isOk(rotated) || !isOk(base))
        continue;
      const bool exact = c.quarterTurns >= 0;
      const double sigmas = exact ? kRotationSigmas : kGenericRotationSigmas;
      const double position = exact ? kRotationPosition : kGenericRotationPosition;
      const double covariance = exact ? kRotationCovariance : kGenericRotationCovariance;
      const std::string kind = exact ? "rotation by k x 90 deg: " : "rotation by 0.7 rad: ";
      auto const& r = rotated.row;
      auto const& b = base.row;
      auto inSigmas = [](double deviation, double sigma, double value) {
        return std::max(0., std::abs(deviation) - kUlps * ulp(value)) / sigma;
      };
      checks(kind + "phi", sigmas, "sigma")
          .add(inSigmas(wrap(r.phi - b.phi - c.angle), std::sqrt(b.covCircle[0]), kPi), c.label);
      checks(kind + "kappa, cot(theta), z", sigmas, "sigma")
          .add(std::max({inSigmas(r.kappa - b.kappa, std::sqrt(b.covCircle[5]), b.kappa),
                         inSigmas(r.cotTheta - b.cotTheta, std::sqrt(b.covLine[0]), b.cotTheta),
                         inSigmas(r.z - b.z, std::sqrt(b.covLine[2]), b.z)}),
               c.label);
      float x = b.x, y = b.y;
      rotateVector(c.angle, c.quarterTurns, x, y);
      checks(kind + "position", position, "cm").add(std::hypot(r.x - x, r.y - y, r.z - b.z), c.label);
      double deviation = 0.;
      const int diagonalCircle[3] = {0, 2, 5};
      for (int row = 0, k = 0; row < 3; ++row)
        for (int col = 0; col <= row; ++col, ++k) {
          // covCircle is (pp, px, xx, pk, xk, kk): element k is (col, row) of the upper triangle
          const double scale = std::sqrt(b.covCircle[diagonalCircle[row]] * b.covCircle[diagonalCircle[col]]);
          deviation = std::max(deviation, std::abs(r.covCircle[k] - b.covCircle[k]) / scale);
        }
      const int diagonalLine[2] = {0, 2};
      for (int row = 0, k = 0; row < 2; ++row)
        for (int col = 0; col <= row; ++col, ++k) {
          const double scale = std::sqrt(b.covLine[diagonalLine[row]] * b.covLine[diagonalLine[col]]);
          deviation = std::max(deviation, std::abs(r.covLine[k] - b.covLine[k]) / scale);
        }
      checks(kind + "covariance", covariance, "rel").add(deviation, c.label);
      checks(kind + "charge, nFit, refFallback", 0., "")
          .require(r.charge == b.charge && rotated.nFit == base.nFit && rotated.refFallback == base.refFallback,
                   c.label);

      // the pLS, exact rotations, within the turn cut of the closest approach
      Case const& baseCase = cases[c.base];
      if (!exact || !baseCase.noiseless)
        continue;
      const double turn = bField / baseCase.generated.helix.pt * baseCase.generated.truth.back().s;
      PixelSegmentCandidate const& pr = outputs[i].candidate;
      PixelSegmentCandidate const& pb = outputs[c.base].candidate;
      if (turn > blfTestStates::kMaxTurnAngle || pb.pixelType == ::lst::PixelType::kInvalid)
        continue;
      checks(kind + "pLS made, same type", 0., "").require(pr.pixelType == pb.pixelType, c.label);
      if (pr.pixelType == ::lst::PixelType::kInvalid)
        continue;
      checks(kind + "pLS dxy, dz", kPLSPosition, "cm")
          .add(std::max(std::abs(pr.kinematics.dxy - pb.kinematics.dxy), std::abs(pr.kinematics.dz - pb.kinematics.dz)),
               c.label);
      checks(kind + "pLS ptIn, ptPCA", kPLSMomentum, "rel")
          .add(std::max(std::abs(pr.ptIn - pb.ptIn) / pb.ptIn,
                        std::abs(pr.kinematics.ptPCA - pb.kinematics.ptPCA) / pb.kinematics.ptPCA),
               c.label);
      checks(kind + "pLS deltaPhi", kPLSDeltaPhi, "rad").add(std::abs(pr.deltaPhi - pb.deltaPhi), c.label);
      checks(kind + "pLS ptErr, etaErr", kPLSError, "rel")
          .add(std::max(std::abs(pr.ptErr - pb.ptErr) / pb.ptErr, std::abs(pr.etaErr - pb.etaErr) / pb.etaErr),
               c.label);
    }

    std::cout << "  " << cases.size() << " cases; noiseless states beyond the closest-approach turn cut of "
              << blfTestStates::kMaxTurnAngle << " rad (pLS not checked): " << nBeyondTurn
              << "; closest approach not valid within it: " << nNotConvergedPCA << "\n";
    return checks.report();
  }

  // Every category of cases keeps at least its minimum number of cases, so that no category can silently become
  // empty when helices miss a surface of their layout. With the current seed and layouts every helix is kept: 264
  // central and 264 smeared, 128 forward, 16 phi near pi, 12 stride, 24 "3 IT + OT", one case in each other category.
  // The minimums are these counts less about 5% (at least one case), and one for a single-case category; a category
  // without a minimum fails.
  bool checkCaseCounts(Builder const& builder) {
    const std::map<std::string, int> minimum = {
        {"central", 250},
        {"central smeared", 250},
        {"forward", 121},
        {"phi near pi", 15},
        {"stride", 11},
        {"no EDM track", 1},
        {"3 hits", 1},
        {"256 hits", 1},
        {"NaN hit position", 1},
        {"negative hit variances", 1},
        {"3 IT + OT", 22},
        {"2 IT + 4 OT", 1},
        {"1 IT + 4 OT", 1},
        {"0 IT + 5 OT", 1},
        {"4 IT + 2 OT", 1},
        {"4 IT", 1},
        {"3 IT + 3 OT, maxITHitsToDropOTHits = 2", 1},
        {"3 IT + 3 OT, NaN IT hit", 1},
    };
    Checks checks;
    for (auto const& [category, count] : builder.kept) {
      std::cout << "  cases kept: " << count << " " << category << "\n";
      auto const found = minimum.find(category);
      checks("case counts: every category has a minimum", 0., "").require(found != minimum.end(), category);
      if (found != minimum.end())
        checks("case counts: at least the minimum per category", 0., "")
            .require(count >= found->second, category + ": " + std::to_string(count));
    }
    for (auto const& [category, count] : minimum)
      checks("case counts: every minimum has its category", 0., "").require(builder.kept.contains(category), category);
    return checks.report();
  }

  // the functions without a kernel: checkState, fitHitPositions
  bool hostChecks() {
    Checks checks;
    ::lst::BLFStateRow row{
        10.f, 5.f, 20.f, 0.5f, 0.3f, 2e-3f, {1e-6f, 0.f, 1e-6f, 0.f, 0.f, 1e-10f}, {1e-4f, 0.f, 1e-3f}, 1};
    auto status = [&](::lst::BLFStateRow const& r) { return fit::checkState(r); };
    checks("checkState: a valid state", 0., "").require(status(row) == SeedStateStatus::kOk, "valid");
    for (int element = 0; element < 15; ++element) {
      ::lst::BLFStateRow bad = row;
      float* values[15] = {&bad.x,
                           &bad.y,
                           &bad.z,
                           &bad.phi,
                           &bad.cotTheta,
                           &bad.kappa,
                           &bad.covCircle[0],
                           &bad.covCircle[1],
                           &bad.covCircle[2],
                           &bad.covCircle[3],
                           &bad.covCircle[4],
                           &bad.covCircle[5],
                           &bad.covLine[0],
                           &bad.covLine[1],
                           &bad.covLine[2]};
      *values[element] = element % 2 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
      checks("checkState: a non-finite element -> kNonFinite", 0., "")
          .require(status(bad) == SeedStateStatus::kNonFinite, "element " + std::to_string(element));
    }
    {
      ::lst::BLFStateRow bad = row;
      bad.charge = -1;
      checks("checkState: sign(kappa) != charge -> kChargeMismatch", 0., "")
          .require(status(bad) == SeedStateStatus::kChargeMismatch, "kappa > 0, q < 0");
      bad = row;
      bad.kappa = 0.f;
      checks("checkState: kappa = 0 -> kChargeMismatch", 0., "")
          .require(status(bad) == SeedStateStatus::kChargeMismatch, "kappa = 0");
      bad = row;
      bad.charge = 0;
      checks("checkState: charge 0 -> kChargeMismatch", 0., "")
          .require(status(bad) == SeedStateStatus::kChargeMismatch, "q = 0");
    }
    for (int diagonal : {0, 2, 5, 6, 8}) {
      for (float value : {0.f, -1e-12f}) {
        ::lst::BLFStateRow bad = row;
        (diagonal < 6 ? bad.covCircle[diagonal] : bad.covLine[diagonal - 6]) = value;
        checks("checkState: a diagonal variance <= 0 -> kNonPositiveVariance", 0., "")
            .require(status(bad) == SeedStateStatus::kNonPositiveVariance, "diagonal " + std::to_string(diagonal));
      }
    }
    for (uint32_t nHits = 4; nHits <= 255; ++nHits) {
      const uint32_t nFit = fit::nFitHits(nHits);
      std::vector<uint32_t> positions(nFit);
      fit::fitHitPositions(nHits, nFit, positions.data());
      checks("fitHitPositions: as the pixel-track fit, 4 to 255 hits", 0., "")
          .require(positions == referenceStride(nHits, nFit), "nHits " + std::to_string(nHits));
    }
    // the size of the full fit, and the stride of that many hits, increasing, from the first to the last hit
    for (uint32_t nHits = 0; nHits <= 300; ++nHits) {
      const std::string label = "nHits " + std::to_string(nHits);
      const uint32_t nFit = fit::nFitHits(nHits);
      checks("sizes: the largest fit size not above nHits; fitted iff 4 to 255 hits", 0., "")
          .require(nFit == expectedFitSize(nHits) &&
                       fit::isFitted(nHits) == (expectedFitSize(nHits) > 0 && nHits <= kMaxTrackHits) &&
                       fit::isFitted(nHits) == (nHits >= 4 && nHits <= 255),
                   label);
      if (nFit == 0 || nHits > kMaxTrackHits)
        continue;
      std::vector<uint32_t> positions(nFit);
      fit::fitHitPositions(nHits, nFit, positions.data());
      bool increasing = positions.front() == 0 && positions.back() == nHits - 1;
      for (uint32_t k = 1; k < nFit; ++k)
        increasing = increasing && positions[k] > positions[k - 1];
      checks("sizes: the stride of the pixel-track fit, increasing, first and last hit", 0., "")
          .require(increasing && positions == referenceStride(nHits, nFit), label);
    }
    return checks.report();
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
  Builder builder;
  builder.bField = bField;
  builder.settings = Settings{0.08f, -0.04f, 1.2f, 2e-4f, -1e-4f, 0.8f, bField, 0, 0, 3};
  buildCases(builder);
  std::vector<Case>& cases = builder.cases;

  // the hits of every case, one after the other
  const uint32_t nTracks = cases.size();
  uint32_t nHits = 0;
  for (auto& c : cases) {
    c.track.firstHit = nHits;
    c.track.nHits = c.hits.size();
    for (auto& hit : c.hits)
      hit.index = nHits++;
  }
  auto tracksHost = cms::alpakatools::make_host_buffer<TestTrack[], Platform>(nTracks);
  auto hitsHost = cms::alpakatools::make_host_buffer<TestHit[], Platform>(nHits);
  auto outputsHost = cms::alpakatools::make_host_buffer<TestOutput[], Platform>(nTracks);
  for (uint32_t t = 0; t < nTracks; ++t) {
    tracksHost[t] = cases[t].track;
    std::copy(cases[t].hits.begin(), cases[t].hits.end(), hitsHost.data() + cases[t].track.firstHit);
  }

  std::cout << "host functions\n";
  bool failed = hostChecks();
  failed = checkCaseCounts(builder) || failed;
  for (auto const& device : devices) {
    Queue queue(device);
    auto tracks = cms::alpakatools::make_device_buffer<TestTrack[]>(queue, nTracks);
    auto hits = cms::alpakatools::make_device_buffer<TestHit[]>(queue, nHits);
    auto outputs = cms::alpakatools::make_device_buffer<TestOutput[]>(queue, nTracks);
    alpaka::memcpy(queue, tracks, tracksHost);
    alpaka::memcpy(queue, hits, hitsHost);
    constexpr uint32_t kThreads = 64;
    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTracks, kThreads), kThreads),
        TrackFitKernel{},
        tracks.data(),
        hits.data(),
        bField,
        outputs.data(),
        nTracks);
    alpaka::memcpy(queue, outputsHost, outputs);
    alpaka::wait(queue);
    std::cout << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " " << alpaka::getName(device) << "\n";
    failed = evaluate(cases, outputsHost.data(), bField) || failed;
  }
  std::cout << (failed ? "FAILED" : "passed") << "\n";
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
