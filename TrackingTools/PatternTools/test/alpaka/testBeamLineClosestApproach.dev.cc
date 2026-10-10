// Compares the portable beam-line closest approach (beamLineClosestApproach.h) and curvilinear Jacobian
// (curvilinearJacobian.h), run on each device of the backend, with TSCBLBuilderNoMaterial and
// AnalyticalCurvilinearJacobian on the host, for random tracks in a non-uniform field.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "TrackingTools/PatternTools/interface/beamLineClosestApproach.h"

#include "TrackingTools/PatternTools/test/beamLineClosestApproachReference.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace beamLineClosestApproachTest;

namespace {

  struct Matrix5 {
    double element[5][5];
  };

  struct BeamLineKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  FreeState const* __restrict__ tracks,
                                  Vector3 const* __restrict__ fields,
                                  BeamLine const* __restrict__ beamLines,
                                  Vector3 const* __restrict__ endPositions,
                                  Vector3 const* __restrict__ endMomenta,
                                  double const* __restrict__ pathLengths,
                                  FreeState* __restrict__ states,
                                  uint8_t* __restrict__ valid,
                                  Matrix5* __restrict__ jacobians,
                                  int32_t nTracks) const {
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nTracks)) {
        valid[i] = beamLineClosestApproach::stateNoMaterial(tracks[i], fields[i], beamLines[i], states[i]) ? 1 : 0;
        curvilinearJacobian::compute(tracks[i].position,
                                     tracks[i].momentum,
                                     tracks[i].charge,
                                     endPositions[i],
                                     endMomenta[i],
                                     fields[i],
                                     pathLengths[i],
                                     jacobians[i].element);
      }
    }
  };

  struct MaxDifference {
    double value = 0;
    int index = -1;
    void update(double difference, int track) {
      if (!(difference <= value)) {  // NaN counts as a difference
        value = difference;
        index = track;
      }
    }
  };

  // Differences, each normalised to the precision the host itself has for that quantity:
  //  - position: the closest approach is found to |dphi| < 1e-6 on the helix, i.e. to 1e-6 times the helix arc length
  //    per radian |p| / (c Bz); the transverse radius |p| sqrt(1 - pz^2 / p^2) / (c Bz) also amplifies the float
  //    rounding of |p| (6e-8) by p^2 / pT^2, which matters for tracks almost parallel to the beam line
  //  - Jacobian: float inputs (unit vectors, positions) carry 6e-8 relative errors, amplified by cancellations in
  //    the largest terms: |p| |s| for the position rows of the q/p column, |s| for the other position-angle elements
  //  - errors: relative to the uncertainties, sqrt(C_ii C_jj)
  // tracks i % 4 != 3 come from a vertex near the beam line (prompt), the others start anywhere in any direction
  bool compare(char const* label,
               Inputs const& inputs,
               std::vector<HostResult> const& host,
               FreeState const* states,
               uint8_t const* valid,
               Matrix5 const* jacobians,
               bool prompt) {
    constexpr double newtonTolerance = 1.e-6, floatRounding = 6.e-8, floatPosition = 1.e-5;
    int nTracks = 0, nValid = 0, nValidMismatch = 0;
    MaxDifference position, positionNormalised, momentum, error[5][5], jacobianDiff[5][5];
    for (int32_t i = 0; i < int32_t(host.size()); ++i) {
      if ((i % 4 != 3) != prompt)
        continue;
      ++nTracks;
      if ((valid[i] != 0) != host[i].valid) {
        ++nValidMismatch;
        std::cout << "  track " << i << ": valid on the host " << host[i].valid << ", on the device " << int(valid[i])
                  << "\n";
        continue;
      }
      if (!host[i].valid)
        continue;
      ++nValid;
      const double momentumMag = std::sqrt(curvilinearJacobian::mag2(inputs.tracks[i].momentum));
      const double pathLength = std::abs(host[i].pathLength);
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col < 5; ++col) {
          const double reference = host[i].jacobian[row][col];
          double scale = std::max(1., std::abs(reference));
          if (row >= 3 && col == 0)
            scale = std::max(scale, momentumMag * pathLength);
          else if (row >= 3 && col <= 2)
            scale = std::max(scale, pathLength);
          jacobianDiff[row][col].update(std::abs(jacobians[i].element[row][col] - reference) / scale, i);
        }
      FreeState const& onDevice = states[i];
      FreeState const& reference = host[i].state;
      const double positionDifference = std::hypot(onDevice.position.x - reference.position.x,
                                                   onDevice.position.y - reference.position.y,
                                                   onDevice.position.z - reference.position.z);
      const double arcPerRadian = momentumMag / (2.99792458e-3 * std::abs(inputs.fields[i].z));
      const double momentumRatio = momentumMag / curvilinearJacobian::perp(inputs.tracks[i].momentum);
      position.update(positionDifference, i);
      positionNormalised.update(
          positionDifference /
              (floatPosition + (newtonTolerance + floatRounding * momentumRatio * momentumRatio) * arcPerRadian),
          i);
      momentum.update(std::hypot(onDevice.momentum.x - reference.momentum.x,
                                 onDevice.momentum.y - reference.momentum.y,
                                 onDevice.momentum.z - reference.momentum.z) /
                          momentumMag,
                      i);
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col <= row; ++col)
          error[row][col].update(std::abs(onDevice.error[row][col] - reference.error[row][col]) /
                                     std::sqrt(reference.error[row][row] * reference.error[col][col]),
                                 i);
    }

    std::cout << " " << label << ": " << nTracks << " tracks, " << nValid << " valid on both, " << nValidMismatch
              << " with different validity\n";
    std::cout << "  max |position difference| " << position.value << " cm (track " << position.index
              << "), in units of the host precision: " << positionNormalised.value << " (track "
              << positionNormalised.index << ")\n";
    std::cout << "  max |momentum difference| / |p| " << momentum.value << " (track " << momentum.index << ")\n";
    double maxError = 0, maxJacobian = 0;
    std::cout << "  max |error difference| / sqrt(C_ii C_jj), lower triangle:\n";
    for (int row = 0; row < 5; ++row) {
      std::cout << "   ";
      for (int col = 0; col <= row; ++col) {
        std::cout << " " << error[row][col].value;
        maxError = std::max(maxError, error[row][col].value);
      }
      std::cout << "\n";
    }
    std::cout << "  max |Jacobian difference| / scale of its largest terms:\n";
    for (int row = 0; row < 5; ++row) {
      std::cout << "   ";
      for (int col = 0; col < 5; ++col) {
        std::cout << " " << jacobianDiff[row][col].value;
        maxJacobian = std::max(maxJacobian, jacobianDiff[row][col].value);
      }
      std::cout << "\n";
    }
    // validity may differ only for tracks at the 12-iteration limit; momentum direction differs like the phase
    constexpr double validMismatchTolerance = 1.e-4, positionTolerance = 10., momentumTolerance = 5. * newtonTolerance,
                     errorTolerance = 1.e-3, jacobianTolerance = 1.e-5;
    const bool failed = !(nValidMismatch <= validMismatchTolerance * nTracks) ||
                        !(positionNormalised.value < positionTolerance) || !(momentum.value < momentumTolerance) ||
                        !(maxError < errorTolerance) || !(maxJacobian < jacobianTolerance);
    std::cout << (failed ? "  FAILED" : "  passed") << " (tolerances: validity mismatches " << validMismatchTolerance
              << " of the tracks, position " << positionTolerance << " units, momentum " << momentumTolerance
              << ", errors " << errorTolerance << ", Jacobian " << jacobianTolerance << ")\n";
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

  constexpr int32_t nTracks = 200000;
  const Inputs inputs = makeInputs(nTracks, 20261008);
  const std::vector<HostResult> host = hostResults(inputs);

  auto tracksHost = cms::alpakatools::make_host_buffer<FreeState[], Platform>(nTracks);
  auto fieldsHost = cms::alpakatools::make_host_buffer<Vector3[], Platform>(nTracks);
  auto beamLinesHost = cms::alpakatools::make_host_buffer<BeamLine[], Platform>(nTracks);
  auto endPositionsHost = cms::alpakatools::make_host_buffer<Vector3[], Platform>(nTracks);
  auto endMomentaHost = cms::alpakatools::make_host_buffer<Vector3[], Platform>(nTracks);
  auto pathLengthsHost = cms::alpakatools::make_host_buffer<double[], Platform>(nTracks);
  auto statesHost = cms::alpakatools::make_host_buffer<FreeState[], Platform>(nTracks);
  auto validHost = cms::alpakatools::make_host_buffer<uint8_t[], Platform>(nTracks);
  auto jacobiansHost = cms::alpakatools::make_host_buffer<Matrix5[], Platform>(nTracks);
  for (int32_t i = 0; i < nTracks; ++i) {
    tracksHost[i] = inputs.tracks[i];
    fieldsHost[i] = inputs.fields[i];
    beamLinesHost[i] = inputs.beamLines[i];
    // the Jacobian is also compared where the host state is invalid, with the start state as the end state
    endPositionsHost[i] = host[i].valid ? host[i].endPosition : inputs.tracks[i].position;
    endMomentaHost[i] = host[i].valid ? host[i].endMomentum : inputs.tracks[i].momentum;
    pathLengthsHost[i] = host[i].valid ? host[i].pathLength : 0.;
  }

  bool failed = false;
  for (auto const& device : devices) {
    Queue queue(device);
    auto tracks = cms::alpakatools::make_device_buffer<FreeState[]>(queue, nTracks);
    auto fields = cms::alpakatools::make_device_buffer<Vector3[]>(queue, nTracks);
    auto beamLines = cms::alpakatools::make_device_buffer<BeamLine[]>(queue, nTracks);
    auto endPositions = cms::alpakatools::make_device_buffer<Vector3[]>(queue, nTracks);
    auto endMomenta = cms::alpakatools::make_device_buffer<Vector3[]>(queue, nTracks);
    auto pathLengths = cms::alpakatools::make_device_buffer<double[]>(queue, nTracks);
    auto states = cms::alpakatools::make_device_buffer<FreeState[]>(queue, nTracks);
    auto valid = cms::alpakatools::make_device_buffer<uint8_t[]>(queue, nTracks);
    auto jacobians = cms::alpakatools::make_device_buffer<Matrix5[]>(queue, nTracks);
    alpaka::memcpy(queue, tracks, tracksHost);
    alpaka::memcpy(queue, fields, fieldsHost);
    alpaka::memcpy(queue, beamLines, beamLinesHost);
    alpaka::memcpy(queue, endPositions, endPositionsHost);
    alpaka::memcpy(queue, endMomenta, endMomentaHost);
    alpaka::memcpy(queue, pathLengths, pathLengthsHost);

    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTracks, 128), 128);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        BeamLineKernel{},
                        tracks.data(),
                        fields.data(),
                        beamLines.data(),
                        endPositions.data(),
                        endMomenta.data(),
                        pathLengths.data(),
                        states.data(),
                        valid.data(),
                        jacobians.data(),
                        nTracks);
    alpaka::memcpy(queue, statesHost, states);
    alpaka::memcpy(queue, validHost, valid);
    alpaka::memcpy(queue, jacobiansHost, jacobians);
    alpaka::wait(queue);

    std::cout << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " " << alpaka::getName(device) << ", " << nTracks
              << " tracks\n";
    const bool promptFailed = compare("from a vertex near the beam line",
                                      inputs,
                                      host,
                                      statesHost.data(),
                                      validHost.data(),
                                      jacobiansHost.data(),
                                      true);
    const bool otherFailed = compare("anywhere in the tracker, any direction",
                                     inputs,
                                     host,
                                     statesHost.data(),
                                     validHost.data(),
                                     jacobiansHost.data(),
                                     false);
    failed = failed || promptFailed || otherFailed;
  }
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
