// Runs lst::closestApproach, lst::blfToCurvilinearJacobian and lst::blfToCurvilinearCovariance in a kernel on each
// device of the backend, for random Broken Line Fit states and a few invalid ones, and compares with the same calls
// on the host: bit-identical on CPU backends; on GPUs within the tolerances of the host chain comparison of
// testPixelSegmentClosestApproach (which are those of TrackingTools/PatternTools/test/alpaka/
// testBeamLineClosestApproach.dev.cc), the GPU sine, cosine and contractions differing from the host ones. The
// comparison with tolerances asserts only for states that turned by at most blfTestStates::kMaxTurnAngle; the others
// are counted and printed. The bit-identity on CPU backends covers all states.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LST/interface/BLFCurvilinearJacobian.h"
#include "RecoTracker/LST/interface/PixelSegmentClosestApproach.h"

#include "RecoTracker/LST/test/BLFTestStates.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  using Result = lst::PixelSegmentClosestApproach;
  using beamLineClosestApproach::BeamLine;

  struct Matrix5 {
    double element[5][5];
  };

  struct ClosestApproachKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  lst::BLFStateRow const* __restrict__ rows,
                                  BeamLine const* __restrict__ beamLines,
                                  float bField,
                                  Result* __restrict__ results,
                                  Matrix5* __restrict__ jacobians,
                                  Matrix5* __restrict__ covariances,
                                  int32_t nStates) const {
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nStates)) {
        lst::BLFStateRow const& row = rows[i];
        results[i] = lst::closestApproach(row, lst::UniformField(bField), beamLines[i]);
        lst::blfToCurvilinearJacobian(row.kappa, row.cotTheta, bField, jacobians[i].element);
        lst::blfToCurvilinearCovariance(
            row.kappa, row.cotTheta, bField, row.covCircle, row.covLine, covariances[i].element);
      }
    }
  };

  struct MaxDifference {
    double value = 0;
    int index = -1;
    void update(double difference, int i) {
      if (std::isnan(value))  // a NaN difference is kept
        return;
      if (!(difference <= value)) {  // NaN counts as a difference
        value = difference;
        index = i;
      }
    }
  };

  bool sameBits(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }
  bool sameBits(double a, double b) { return std::bit_cast<uint64_t>(a) == std::bit_cast<uint64_t>(b); }

  bool sameBits(Result const& a, Result const& b) {
    return sameBits(a.px, b.px) && sameBits(a.py, b.py) && sameBits(a.pz, b.pz) && sameBits(a.dxy, b.dxy) &&
           sameBits(a.dz, b.dz) && sameBits(a.ptErr, b.ptErr) && sameBits(a.etaErr, b.etaErr) && a.charge == b.charge &&
           a.status == b.status;
  }

  bool sameBits(Matrix5 const& a, Matrix5 const& b) {
    for (int row = 0; row < 5; ++row)
      for (int col = 0; col < 5; ++col)
        if (!sameBits(a.element[row][col], b.element[row][col]))
          return false;
    return true;
  }

  // bit-identical results on CPU backends; unused on the others
  [[maybe_unused]] bool compareBits(std::vector<Result> const& host,
                                    std::vector<Matrix5> const& hostJacobians,
                                    std::vector<Matrix5> const& hostCovariances,
                                    Result const* device,
                                    Matrix5 const* deviceJacobians,
                                    Matrix5 const* deviceCovariances) {
    int nResults = 0, nJacobians = 0, nCovariances = 0;
    for (std::size_t i = 0; i < host.size(); ++i) {
      if (!sameBits(host[i], device[i])) {
        if (nResults < 10)
          std::cout << "  state " << i << ": different closest approach\n";
        ++nResults;
      }
      nJacobians += sameBits(hostJacobians[i], deviceJacobians[i]) ? 0 : 1;
      nCovariances += sameBits(hostCovariances[i], deviceCovariances[i]) ? 0 : 1;
    }
    std::cout << "  states with a result not bit-identical to the host: closest approach " << nResults << ", Jacobian "
              << nJacobians << ", covariance " << nCovariances << " of " << host.size() << "\n";
    const bool failed = nResults != 0 || nJacobians != 0 || nCovariances != 0;
    std::cout << (failed ? "  FAILED" : "  passed") << " (tolerance: bit-identical)\n";
    return failed;
  }

  // GPU results within the tolerances of the host chain comparison
  //
  // Only states that turned by at most kMaxTurnAngle are asserted: beyond about 1 rad of turn the Newton search can
  // converge to the far side of the circle, or not converge, and does so differently with the GPU sine and cosine, as
  // the release's TSCBLBuilderNoMaterial does on the host. Pixel-track states in the Phase-2 HLT sit on the last
  // pixel-track hit, in the pixel detector or the innermost outer-tracker layers, where pLS above the pT cut turn by
  // well under 1 rad; the cut at 0.9 rad keeps a margin. The other states are counted and printed.
  // The states after the first nGenerated are invalid by construction and always asserted.
  bool compareTolerances(std::vector<blfTestStates::TestState> const& states,
                         int nGenerated,
                         float bField,
                         std::vector<Result> const& host,
                         std::vector<Matrix5> const& hostJacobians,
                         std::vector<Matrix5> const& hostCovariances,
                         Result const* device,
                         Matrix5 const* deviceJacobians,
                         Matrix5 const* deviceCovariances) {
    constexpr double kNewtonTolerance = 1.e-6, kFloatRounding = 6.e-8, kFloatPosition = 1.e-5;
    constexpr double kStatusMismatchTolerance = 1.e-4, kPositionTolerance = 10.,
                     kMomentumTolerance = 5. * kNewtonTolerance, kErrorTolerance = 1.e-3, kJacobianTolerance = 1.e-5;
    const int nStates = host.size();
    int nStatusMismatch = 0, nValid = 0, nChargeMismatch = 0, nInvalidNotZero = 0, nNonFiniteMismatch = 0;
    int nAsserted = 0, nReported = 0;
    int nNotConvergedInBin[blfTestStates::kNTurnBins] = {}, nBeyondInBin[blfTestStates::kNTurnBins] = {},
        nPassedInBin[blfTestStates::kNTurnBins] = {};
    MaxDifference dxy, dz, momentum, ptErr, etaErr, jacobian, covariance;
    // a non-finite element (from a non-finite input) must be non-finite of the same kind on both
    auto sameNonFinite = [](double a, double b) {
      return std::isnan(a) == std::isnan(b) && std::isinf(a) == std::isinf(b);
    };
    for (int i = 0; i < nStates; ++i) {
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col < 5; ++col) {
          const double reference = hostJacobians[i].element[row][col];
          const double onDevice = deviceJacobians[i].element[row][col];
          if (!std::isfinite(reference))
            nNonFiniteMismatch += sameNonFinite(reference, onDevice) ? 0 : 1;
          else
            jacobian.update(std::abs(onDevice - reference) / (reference != 0. ? std::abs(reference) : 1.), i);
          const double referenceCovariance = hostCovariances[i].element[row][col];
          const double covarianceOnDevice = deviceCovariances[i].element[row][col];
          const double scale = std::sqrt(hostCovariances[i].element[row][row] * hostCovariances[i].element[col][col]);
          if (!std::isfinite(referenceCovariance) || !std::isfinite(scale))
            nNonFiniteMismatch += sameNonFinite(referenceCovariance, covarianceOnDevice) ? 0 : 1;
          else
            covariance.update(std::abs(covarianceOnDevice - referenceCovariance) / (scale != 0. ? scale : 1.), i);
        }
      blfTestStates::TestState const& state = states[i];
      const int bin = blfTestStates::turnBin(state, bField);
      const bool asserted = i >= nGenerated || blfTestStates::turnAngle(state, bField) <= blfTestStates::kMaxTurnAngle;
      ++(asserted ? nAsserted : nReported);
      if (device[i].status != Result::kValid) {
        nInvalidNotZero +=
            (device[i].px == 0.f && device[i].py == 0.f && device[i].pz == 0.f && device[i].dxy == 0.f &&
             device[i].dz == 0.f && device[i].ptErr == 0.f && device[i].etaErr == 0.f && device[i].charge == 0)
                ? 0
                : 1;
      }
      if (!asserted && (host[i].status != Result::kValid || device[i].status != Result::kValid)) {
        ++nNotConvergedInBin[bin];
        continue;
      }
      if (host[i].status != device[i].status) {
        ++nStatusMismatch;
        std::cout << "  state " << i << ": status on the host " << int(host[i].status) << ", on the device "
                  << int(device[i].status) << "\n";
        continue;
      }
      if (host[i].status != Result::kValid)
        continue;
      if (asserted) {
        ++nValid;
        nChargeMismatch += host[i].charge == device[i].charge ? 0 : 1;
      }
      const double momentumRatio = std::sqrt(1. + state.cot * state.cot);
      const double arcPerRadian = state.pt * momentumRatio / bField;
      const double unit =
          kFloatPosition + (kNewtonTolerance + kFloatRounding * momentumRatio * momentumRatio) * arcPerRadian;
      const double dxyUnits = std::abs(device[i].dxy - host[i].dxy) / unit;
      const double dzUnits = std::abs(device[i].dz - host[i].dz) / unit;
      const double momentumDifference =
          std::hypot(device[i].px - host[i].px, device[i].py - host[i].py, device[i].pz - host[i].pz) /
          std::hypot(host[i].px, host[i].py, host[i].pz);
      const bool withinTolerance =
          dxyUnits < kPositionTolerance && dzUnits < kPositionTolerance && momentumDifference < kMomentumTolerance;
      if (!asserted) {
        ++(withinTolerance ? nPassedInBin[bin] : nBeyondInBin[bin]);
        continue;
      }
      dxy.update(dxyUnits, i);
      dz.update(dzUnits, i);
      momentum.update(momentumDifference, i);
      if (!withinTolerance)
        std::cout << "  state " << i << " beyond a tolerance: turning angle " << blfTestStates::turnAngle(state, bField)
                  << ", pT " << state.pt << ", dxy " << host[i].dxy << " on the host, " << device[i].dxy
                  << " on the device, dz " << host[i].dz << ", " << device[i].dz << "\n";
      ptErr.update(std::abs(double(device[i].ptErr) * device[i].ptErr - double(host[i].ptErr) * host[i].ptErr) /
                       (double(host[i].ptErr) * host[i].ptErr),
                   i);
      etaErr.update(std::abs(double(device[i].etaErr) * device[i].etaErr - double(host[i].etaErr) * host[i].etaErr) /
                        (double(host[i].etaErr) * host[i].etaErr),
                    i);
    }
    std::cout << "  " << nStates << " states, " << nAsserted << " asserted (turning angle up to "
              << blfTestStates::kMaxTurnAngle << " rad, and the invalid ones), " << nReported
              << " reported only; asserted: " << nValid << " valid on both, " << nStatusMismatch
              << " with a different status, " << nChargeMismatch << " with a different charge, " << nInvalidNotZero
              << " invalid with non-zero outputs, " << nNonFiniteMismatch
              << " non-finite Jacobian or covariance elements not matched\n"
              << "  max |dxy difference| " << dxy.value << " host-precision units (state " << dxy.index
              << "), |dz difference| " << dz.value << " (state " << dz.index << ")\n"
              << "  max |momentum difference| / |p| " << momentum.value << " (state " << momentum.index << ")\n"
              << "  max |variance difference| / variance: ptErr " << ptErr.value << " (state " << ptErr.index
              << "), etaErr " << etaErr.value << " (state " << etaErr.index << ")\n"
              << "  max relative Jacobian difference " << jacobian.value << " (state " << jacobian.index
              << "), max |covariance difference| / sqrt(C_ii C_jj) " << covariance.value << " (state "
              << covariance.index << ")\n"
              << "  reported only, per turning angle from the vertex to the state point (rad): not converged on the "
                 "host or the device, converged beyond a tolerance (far side or otherwise), passed\n";
    for (int bin = 0; bin < blfTestStates::kNTurnBins; ++bin)
      if (nNotConvergedInBin[bin] + nBeyondInBin[bin] + nPassedInBin[bin] > 0)
        std::cout << "    from " << bin * blfTestStates::kTurnBinWidth << ": " << nNotConvergedInBin[bin] << ", "
                  << nBeyondInBin[bin] << ", " << nPassedInBin[bin] << "\n";
    // the lower bound on the asserted states guards against the turning angle cut excluding most of them
    const bool failed = !(nStatusMismatch <= kStatusMismatchTolerance * nStates) || !(nAsserted >= 0.95 * nGenerated) ||
                        nChargeMismatch != 0 || nInvalidNotZero != 0 || nNonFiniteMismatch != 0 ||
                        !(dxy.value < kPositionTolerance) || !(dz.value < kPositionTolerance) ||
                        !(momentum.value < kMomentumTolerance) || !(ptErr.value < kErrorTolerance) ||
                        !(etaErr.value < kErrorTolerance) || !(jacobian.value < kJacobianTolerance) ||
                        !(covariance.value < kErrorTolerance);
    std::cout << (failed ? "  FAILED" : "  passed") << " (tolerances: status mismatches " << kStatusMismatchTolerance
              << " of the states, dxy and dz " << kPositionTolerance << " units, momentum " << kMomentumTolerance
              << ", ptErr and etaErr variances " << kErrorTolerance << ", Jacobian " << kJacobianTolerance
              << ", covariance " << kErrorTolerance << ")\n";
    return failed;
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
  std::vector<blfTestStates::TestState> states = blfTestStates::makeStates(100000, 20261019);
  const int nGenerated = states.size();
  // invalid states at the end: kappa = 0, q = 0, a NaN parameter, a NaN covariance element
  for (int k = 0; k < 4; ++k) {
    blfTestStates::TestState state = states[k];
    if (k == 0)
      state.row.kappa = 0.f;
    else if (k == 1)
      state.row.charge = 0;
    else if (k == 2)
      state.row.phi = std::numeric_limits<float>::quiet_NaN();
    else
      state.row.covLine[1] = std::numeric_limits<float>::quiet_NaN();
    states.push_back(state);
  }
  const int32_t nStates = states.size();

  const lst::UniformField field(bField);
  std::vector<Result> host(nStates);
  std::vector<Matrix5> hostJacobians(nStates), hostCovariances(nStates);
  auto rowsHost = cms::alpakatools::make_host_buffer<lst::BLFStateRow[], Platform>(nStates);
  auto beamLinesHost = cms::alpakatools::make_host_buffer<BeamLine[], Platform>(nStates);
  auto resultsHost = cms::alpakatools::make_host_buffer<Result[], Platform>(nStates);
  auto jacobiansHost = cms::alpakatools::make_host_buffer<Matrix5[], Platform>(nStates);
  auto covariancesHost = cms::alpakatools::make_host_buffer<Matrix5[], Platform>(nStates);
  for (int32_t i = 0; i < nStates; ++i) {
    lst::BLFStateRow const& row = states[i].row;
    rowsHost[i] = row;
    beamLinesHost[i] = states[i].beamLine;
    host[i] = lst::closestApproach(row, field, states[i].beamLine);
    lst::blfToCurvilinearJacobian(row.kappa, row.cotTheta, bField, hostJacobians[i].element);
    lst::blfToCurvilinearCovariance(
        row.kappa, row.cotTheta, bField, row.covCircle, row.covLine, hostCovariances[i].element);
  }

  bool failed = false;
  for (auto const& device : devices) {
    Queue queue(device);
    auto rows = cms::alpakatools::make_device_buffer<lst::BLFStateRow[]>(queue, nStates);
    auto beamLines = cms::alpakatools::make_device_buffer<BeamLine[]>(queue, nStates);
    auto results = cms::alpakatools::make_device_buffer<Result[]>(queue, nStates);
    auto jacobians = cms::alpakatools::make_device_buffer<Matrix5[]>(queue, nStates);
    auto covariances = cms::alpakatools::make_device_buffer<Matrix5[]>(queue, nStates);
    alpaka::memcpy(queue, rows, rowsHost);
    alpaka::memcpy(queue, beamLines, beamLinesHost);

    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nStates, 128), 128);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        ClosestApproachKernel{},
                        rows.data(),
                        beamLines.data(),
                        bField,
                        results.data(),
                        jacobians.data(),
                        covariances.data(),
                        nStates);
    alpaka::memcpy(queue, resultsHost, results);
    alpaka::memcpy(queue, jacobiansHost, jacobians);
    alpaka::memcpy(queue, covariancesHost, covariances);
    alpaka::wait(queue);

    std::cout << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " " << alpaka::getName(device) << "\n";
    if constexpr (std::is_same_v<Device, alpaka::DevCpu>)
      failed =
          compareBits(
              host, hostJacobians, hostCovariances, resultsHost.data(), jacobiansHost.data(), covariancesHost.data()) ||
          failed;
    else
      failed = compareTolerances(states,
                                 nGenerated,
                                 bField,
                                 host,
                                 hostJacobians,
                                 hostCovariances,
                                 resultsHost.data(),
                                 jacobiansHost.data(),
                                 covariancesHost.data()) ||
               failed;
  }
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
