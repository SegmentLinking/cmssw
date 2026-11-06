#ifndef RecoTracker_LST_test_alpaka_LSTInputDeviceTestTracks_h
#define RecoTracker_LST_test_alpaka_LSTInputDeviceTestTracks_h

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LST/plugins/alpaka/LSTInputDeviceTrackFit.h"

// Shared by the unit tests of the device LST input: synthetic helices through pixel and outer-tracker layers in a
// uniform field, with module frames as the detector's and the hit errors of its sensors; the per-track work of the
// producer (prepareTrack, fitState, makePixelSegment) run in a test kernel on such hits; the truth tolerances; and the
// bookkeeping of the checks (the largest deviation of each against its tolerance).
namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDeviceTest {

  namespace fit = ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice;

  using ::lst::SeedStateStatus;
  using ::lstInputDevice::FitHit;
  using ::lstInputDevice::FitInput;
  using ::lstInputDevice::FittedState;
  using ::lstInputDevice::kITFitHits;
  using ::lstInputDevice::kMaxFitHits;
  using ::lstInputDevice::kMaxTrackHits;
  using ::lstInputDevice::PixelSegmentCandidate;
  using ::lstInputDevice::Settings;

  constexpr double kPi = std::numbers::pi;

  // ---------------------------------------------------------------------------------------------------------------
  // Tolerances
  //
  // Noiseless truth. The hits lie on the helix in double precision but enter the fit as float, rounded by at most
  // 3.8e-6 cm per coordinate (half a float ulp at |x| < 128 cm). For a least-squares fit, an input shift delta moves a
  // parameter by at most sqrt(chi2(delta)) of its standard deviation, chi2(delta) being the chi2 of the shift itself;
  // with at least 7 um hit errors, 14 hits and three coordinates this is below 0.01 sigma. The Broken Line Fit is
  // linearised around the fast fit, which on exact hits is the true circle, so the bound holds to second order.
  // 0.05 sigma leaves a factor 5. The state point is the last hit corrected along the radius: 1e-4 cm is 25 times the
  // input rounding and 7 times below the best hit resolution.
  constexpr double kTruthSigmas = 0.05;
  constexpr double kTruthPosition = 1e-4;  // cm
  // Float storage of the state: two ulps of the stored value.
  constexpr double kUlps = 2.;
  // the global covariance of a hit computed on the host and in the kernel by the same float expression, possibly
  // contracted differently: a few float roundings
  constexpr double kHitCovariance = 1e-6;
  // The closest approach of the same state computed twice, from rotated inputs or in two kernels whose float
  // expressions are contracted differently: the tolerances of its host-device comparison
  // (TrackingTools/PatternTools/test/alpaka/testBeamLineClosestApproach.dev.cc): positions to 10 host-precision units
  // (here 1e-4 cm at |x| < 128 cm in float), momenta 5e-6 relative, errors 1e-3 relative.
  constexpr double kPLSPosition = 1e-4;  // cm, dxy and dz
  constexpr double kPLSMomentum = 5e-6;  // relative
  constexpr double kPLSError = 1e-3;     // relative, ptErr and etaErr

  // ---------------------------------------------------------------------------------------------------------------
  // The sizes of the full fit, written out here independently of the producer's functions

  inline const std::vector<uint32_t> kExpectedFitSizes = {4, 6, 8, 10, 12, 14};

  // the size of the full fit of a track of nHits hits: the largest size not above nHits, 0 if none
  inline uint32_t expectedFitSize(uint32_t nHits) {
    uint32_t size = 0;
    for (uint32_t s : kExpectedFitSizes)
      if (s <= nHits && s > size)
        size = s;
    return size;
  }

  // the list of the full fits of nFit hits: after the inner-tracker-only list, one list per size, the smallest first
  inline uint32_t expectedFitList(uint32_t nFit) {
    uint32_t list = 1;
    for (uint32_t s : kExpectedFitSizes)
      list += s < nFit ? 1 : 0;
    return list;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Test inputs and outputs, plain types shared by the host and the device

  struct TestHit {
    float x, y, z;
    float origin[3];
    float rotation[9];  // rows: the local axes in the global frame, as SOARotation
    float varX, varY;   // local variances, uncorrelated
    uint32_t index, detId;
    uint16_t clusterSize;
    uint8_t isOT;
  };

  ALPAKA_FN_HOST_ACC inline SOAFrame<float> frameOf(TestHit const& hit) {
    return SOAFrame<float>(hit.origin[0], hit.origin[1], hit.origin[2], SOARotation<float>(hit.rotation));
  }

  // the hits of one test track as prepareTrack reads them, as the producer reads the pixel-track hits
  struct TestTrackHits {
    TestHit const* hits;

    ALPAKA_FN_ACC bool isOT(uint32_t k) const { return hits[k].isOT != 0; }

    ALPAKA_FN_ACC fit::TrackHit hit(uint32_t k) const {
      TestHit const& source = hits[k];
      fit::TrackHit result;
      result.fit.x = source.x;
      result.fit.y = source.y;
      result.fit.z = source.z;
      fit::hitCovariance(frameOf(source), source.varX, source.varY, result.fit.cov);
      result.index = source.index;
      result.detId = source.detId;
      result.clusterSize = source.clusterSize;
      return result;
    }
  };

  struct TestTrack {
    uint32_t firstHit, nHits;
    uint8_t hasEdmTrack;
    Settings settings;
  };

  struct TestOutput {
    FitInput input;
    PixelSegmentCandidate candidate;
    FittedState state, itState;
    uint32_t nFit, nITHits;
    uint8_t prepared, itCandidate, itFit;
  };

  // work: the fit's work array, ALPAKA_ACCELERATOR_NAMESPACE::lst::fitWorkSize(kMaxFitHits) doubles
  template <typename TAcc>
  ALPAKA_FN_ACC void fitFull(
      TAcc const& acc, uint32_t nFit, FitHit const* hits, float bField, FittedState& state, double* work) {
    switch (nFit) {
      case 4:
        fit::fitState<4>(acc, hits, bField, state, work);
        break;
      case 6:
        fit::fitState<6>(acc, hits, bField, state, work);
        break;
      case 8:
        fit::fitState<8>(acc, hits, bField, state, work);
        break;
      case 10:
        fit::fitState<10>(acc, hits, bField, state, work);
        break;
      case 12:
        fit::fitState<12>(acc, hits, bField, state, work);
        break;
      case 14:
        fit::fitState<14>(acc, hits, bField, state, work);
        break;
      default:
        break;
    }
  }

  // the producer's per-track sequence: prepare, full fit, inner-tracker-only fit, pLS of a state of status kOk
  struct TrackFitKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TestTrack const* tracks,
                                  TestHit const* hits,
                                  float bField,
                                  TestOutput* outputs,
                                  uint32_t nTracks) const {
      // the fits' work array, on the stack of the thread (the producer's fit kernels keep it in block-shared memory)
      double work[ALPAKA_ACCELERATOR_NAMESPACE::lst::fitWorkSize(kMaxFitHits)];
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, nTracks)) {
        TestTrack const& track = tracks[t];
        TestOutput& out = outputs[t];
        out.prepared = out.itCandidate = out.itFit = 0;
        out.nFit = out.nITHits = 0;
        out.candidate.pixelType = ::lst::PixelType::kInvalid;
        if (!track.hasEdmTrack || !fit::isFitted(track.nHits)) {
          fit::setUnfit(out.state, ::lst::kInvalidU32Idx);
          fit::setUnfit(out.itState, ::lst::kInvalidU32Idx);
          continue;
        }
        const fit::TrackPreparation preparation = fit::prepareTrack(TestTrackHits{hits + track.firstHit},
                                                                    track.nHits,
                                                                    track.settings.maxITHitsToDropOTHits,
                                                                    out.input,
                                                                    out.candidate,
                                                                    out.state,
                                                                    out.itState);
        out.prepared = 1;
        out.nFit = preparation.nFit;
        out.nITHits = preparation.nITHits;
        out.itCandidate = preparation.itCandidate;
        out.itFit = preparation.itFit;
        fitFull(acc, preparation.nFit, out.input.hits, bField, out.state, work);
        if (preparation.itFit)
          fit::fitState<kITFitHits>(acc, out.input.itHits, bField, out.itState, work);
        if (out.state.status == static_cast<uint8_t>(SeedStateStatus::kOk))
          fit::makePixelSegment(out.state, track.settings, out.candidate);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------------
  // Synthetic tracks

  // a helix: vertex, momentum azimuth at the vertex, pT, cot(theta), charge; positive charges turn clockwise for Bz > 0
  struct Helix {
    double vertex[3];
    double phi0, pt, cot;
    int charge;
  };

  // position and momentum azimuth after the transverse arc length s; dphi/ds = -q B / pT
  inline void helixAt(Helix const& helix, double bField, double s, double position[3], double& phi) {
    const double w = -helix.charge * bField / helix.pt;
    const double halfTurn = 0.5 * w * s;
    const double chord = halfTurn != 0. ? s * std::sin(halfTurn) / halfTurn : s;
    phi = helix.phi0 + w * s;
    position[0] = helix.vertex[0] + chord * std::cos(helix.phi0 + halfTurn);
    position[1] = helix.vertex[1] + chord * std::sin(helix.phi0 + halfTurn);
    position[2] = helix.vertex[2] + helix.cot * s;
  }

  // a detector surface: a barrel cylinder of radius `value` or an endcap disc at z = `value`, and the local errors of
  // its sensors (local x along r phi; local y along z in the barrel, along r in the endcaps)
  struct Surface {
    bool barrel;
    double value;
    double sigmaX, sigmaY;
    bool isOT;
  };

  constexpr double kPixelSigmaX = 25e-4 / std::numbers::sqrt3 / 2., kPixelSigmaY = 100e-4 / std::numbers::sqrt3 / 2.;
  constexpr double kPSSigmaX = 100e-4 / std::numbers::sqrt3 / 2., kPSSigmaY = 1500e-4 / std::numbers::sqrt3 / 2.;
  constexpr double k2SSigmaX = 90e-4 / std::numbers::sqrt3 / 2., k2SSigmaY = 5. / std::numbers::sqrt3 / 2.;

  inline Surface pixelBarrel(double r) { return {true, r, kPixelSigmaX, kPixelSigmaY, false}; }
  inline Surface pixelDisc(double z) { return {false, z, kPixelSigmaX, kPixelSigmaY, false}; }
  inline Surface otBarrel(double r) {
    return r < 60. ? Surface{true, r, kPSSigmaX, kPSSigmaY, true} : Surface{true, r, k2SSigmaX, k2SSigmaY, true};
  }
  inline Surface otDisc(double z) { return {false, z, kPSSigmaX, kPSSigmaY, true}; }

  inline const std::vector<double> kPixelBarrelRadii = {3.0, 6.8, 10.9, 16.0};
  inline const std::vector<double> kOTBarrelRadii = {23.0, 35.7, 50.8, 68.6, 88.6, 108.0};

  // the transverse arc length of the first crossing of a surface after sMin within the first half turn, or -1
  inline double crossing(Helix const& helix, double bField, Surface const& surface, double sMin) {
    const double sHalf = kPi * helix.pt / bField;
    double position[3], phi;
    if (!surface.barrel) {
      if (helix.cot == 0.)
        return -1.;
      const double s = (surface.value - helix.vertex[2]) / helix.cot;
      return s > sMin && s < sHalf ? s : -1.;
    }
    auto radius = [&](double s) {
      helixAt(helix, bField, s, position, phi);
      return std::hypot(position[0], position[1]);
    };
    if (radius(sMin) >= surface.value || radius(sHalf) <= surface.value)
      return -1.;
    double low = sMin, high = sHalf;
    for (int iteration = 0; iteration < 200; ++iteration) {
      const double middle = 0.5 * (low + high);
      (radius(middle) < surface.value ? low : high) = middle;
    }
    return 0.5 * (low + high);
  }

  struct Truth {
    double position[3];
    double phi;
    double s;
  };

  struct GeneratedTrack {
    Helix helix;
    std::vector<TestHit> hits;
    std::vector<Truth> truth;  // per hit
    bool complete;             // every surface was crossed, in order
  };

  // the hits of a helix on the surfaces, in the order of the surfaces; smeared in the module plane if rng is given
  inline GeneratedTrack generate(Helix const& helix,
                                 std::vector<Surface> const& surfaces,
                                 double bField,
                                 std::mt19937* rng) {
    GeneratedTrack track{helix, {}, {}, true};
    std::normal_distribution<double> gauss(0., 1.);
    double sPrevious = 0.;
    for (std::size_t layer = 0; layer < surfaces.size(); ++layer) {
      Surface const& surface = surfaces[layer];
      const double s = crossing(helix, bField, surface, sPrevious);
      if (s < 0.) {
        track.complete = false;
        break;
      }
      sPrevious = s;
      Truth truth;
      truth.s = s;
      helixAt(helix, bField, s, truth.position, truth.phi);
      const double azimuth = std::atan2(truth.position[1], truth.position[0]);
      const double sinA = std::sin(azimuth), cosA = std::cos(azimuth);
      const double axisX[3] = {-sinA, cosA, 0.};
      const double axisY[3] = {surface.barrel ? 0. : cosA, surface.barrel ? 0. : sinA, surface.barrel ? 1. : 0.};
      const double axisZ[3] = {surface.barrel ? cosA : 0., surface.barrel ? sinA : 0., surface.barrel ? 0. : -1.};
      double position[3] = {truth.position[0], truth.position[1], truth.position[2]};
      if (rng != nullptr) {
        const double dx = surface.sigmaX * gauss(*rng), dy = surface.sigmaY * gauss(*rng);
        for (int k = 0; k < 3; ++k)
          position[k] += dx * axisX[k] + dy * axisY[k];
      }
      TestHit hit{};
      hit.x = position[0];
      hit.y = position[1];
      hit.z = position[2];
      for (int k = 0; k < 3; ++k) {
        hit.origin[k] = position[k];
        hit.rotation[k] = axisX[k];
        hit.rotation[3 + k] = axisY[k];
        hit.rotation[6 + k] = axisZ[k];
      }
      hit.varX = surface.sigmaX * surface.sigmaX;
      hit.varY = surface.sigmaY * surface.sigmaY;
      hit.detId = surface.isOT ? 0x10000000u + layer : ::lst::kPixelModuleId;
      hit.clusterSize = surface.isOT ? 2 : 1;
      hit.isOT = surface.isOT;
      track.hits.push_back(hit);
      track.truth.push_back(truth);
    }
    return track;
  }

  // pixel barrel layers, then outer barrel layers, then the second sensors (+0.4 cm) of the first outer layers
  inline std::vector<Surface> centralLayout(int n) {
    std::vector<Surface> surfaces;
    for (double r : kPixelBarrelRadii)
      surfaces.push_back(pixelBarrel(r));
    std::vector<double> otRadii;
    for (int i = 0; i < std::min(n - 4, 6); ++i)
      otRadii.push_back(kOTBarrelRadii[i]);
    for (int i = 0; i < n - 10; ++i)
      otRadii.push_back(kOTBarrelRadii[i] + 0.4);
    std::sort(otRadii.begin(), otRadii.end());
    for (double r : otRadii)
      surfaces.push_back(otBarrel(r));
    surfaces.resize(n);
    return surfaces;
  }

  // two pixel barrel layers, the pixel discs, the outer discs, on the side of cot
  inline std::vector<Surface> forwardLayout(int n, double cot) {
    const double side = cot > 0. ? 1. : -1.;
    std::vector<Surface> surfaces = {pixelBarrel(3.0), pixelBarrel(6.8)};
    for (double z : {25., 32., 41., 52.})
      surfaces.push_back(pixelDisc(side * z));
    for (double z : {131., 155., 185., 220., 265.})
      surfaces.push_back(otDisc(side * z));
    surfaces.resize(n);
    return surfaces;
  }

  // the largest deviation of one check against its tolerance
  struct Check {
    std::string name;
    double tolerance;
    std::string unit;
    double worst = 0.;
    std::string worstCase;
    int count = 0, failures = 0;

    void add(double deviation, std::string const& label) {
      ++count;
      if (!(deviation <= tolerance)) {
        if (failures < 5)
          std::cout << "    FAILED " << name << ": " << deviation << " " << unit << " > " << tolerance << " in "
                    << label << "\n";
        ++failures;
      }
      if (!(deviation <= worst) && !std::isnan(worst)) {
        worst = deviation;
        worstCase = label;
      }
    }
    // a condition that must hold
    void require(bool condition, std::string const& label) { add(condition ? 0. : 1., label); }
  };

  struct Checks {
    std::vector<Check> list;
    Check& operator()(std::string const& name, double tolerance, std::string const& unit) {
      for (auto& check : list)
        if (check.name == name)
          return check;
      Check check;
      check.name = name;
      check.tolerance = tolerance;
      check.unit = unit;
      list.push_back(check);
      return list.back();
    }
    bool report() const {
      bool failed = false;
      for (auto const& check : list) {
        std::cout << "  " << (check.failures ? "FAILED " : "passed ") << std::setw(58) << std::left << check.name
                  << std::right << " max " << std::setw(11) << check.worst << " tol " << std::setw(9) << check.tolerance
                  << " " << std::setw(6) << std::left << check.unit << std::right << " over " << check.count
                  << (check.failures ? " (" + std::to_string(check.failures) + " failures)" : std::string()) << "  ["
                  << check.worstCase << "]\n";
        failed = failed || check.failures != 0;
      }
      return failed;
    }
  };

  inline double ulp(double value) {
    const float f = std::abs(static_cast<float>(value));
    return std::nextafter(f, std::numeric_limits<float>::infinity()) - f;
  }

  inline double wrap(double angle) { return std::remainder(angle, 2. * kPi); }

  // the same float bits: a copied value, NaN included
  inline bool same(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }

  inline std::string statusName(uint8_t status) {
    switch (static_cast<SeedStateStatus>(status)) {
      case SeedStateStatus::kOk:
        return "kOk";
      case SeedStateStatus::kNotFitted:
        return "kNotFitted";
      case SeedStateStatus::kNonFinite:
        return "kNonFinite";
      case SeedStateStatus::kChargeMismatch:
        return "kChargeMismatch";
      case SeedStateStatus::kNonPositiveVariance:
        return "kNonPositiveVariance";
    }
    return "unknown";
  }

  // the state against the noiseless truth at hit `hit` of the generated track
  inline void checkTruth(Checks& checks,
                         std::string const& prefix,
                         FittedState const& state,
                         GeneratedTrack const& generated,
                         std::size_t hit,
                         double bField,
                         std::string const& label) {
    auto const& row = state.row;
    Truth const& truth = generated.truth[hit];
    Helix const& helix = generated.helix;
    const double kappa = helix.charge * bField / helix.pt;
    const double sigmaPhi = std::sqrt(row.covCircle[0]), sigmaKappa = std::sqrt(row.covCircle[5]);
    const double sigmaCot = std::sqrt(row.covLine[0]), sigmaZ = std::sqrt(row.covLine[2]);
    auto inSigmas = [](double deviation, double sigma, double value) {
      return std::max(0., std::abs(deviation) - kUlps * ulp(value)) / sigma;
    };
    checks(prefix + "truth: phi", kTruthSigmas, "sigma").add(inSigmas(wrap(row.phi - truth.phi), sigmaPhi, kPi), label);
    checks(prefix + "truth: kappa", kTruthSigmas, "sigma").add(inSigmas(row.kappa - kappa, sigmaKappa, kappa), label);
    checks(prefix + "truth: cot(theta)", kTruthSigmas, "sigma")
        .add(inSigmas(row.cotTheta - helix.cot, sigmaCot, helix.cot), label);
    checks(prefix + "truth: z", kTruthSigmas, "sigma")
        .add(inSigmas(row.z - truth.position[2], sigmaZ, truth.position[2]), label);
    checks(prefix + "truth: position", kTruthPosition, "cm")
        .add(std::hypot(row.x - truth.position[0], row.y - truth.position[1], row.z - truth.position[2]), label);
    checks(prefix + "charge", 0., "").require(row.charge == helix.charge, label);
    checks(prefix + "stored phi in [-float(pi), float(pi)]", 0., "")
        .require(row.phi >= -std::numbers::pi_v<float> && row.phi <= std::numbers::pi_v<float>, label);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDeviceTest

#endif
