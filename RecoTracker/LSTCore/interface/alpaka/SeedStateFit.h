#ifndef RecoTracker_LSTCore_interface_alpaka_SeedStateFit_h
#define RecoTracker_LSTCore_interface_alpaka_SeedStateFit_h

#include <Eigen/Core>  // Eigen::Matrix; before any SoA header

#include <cmath>
#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"
#include "RecoTracker/LSTCore/interface/SeedStateFitTypes.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

// The Broken Line Fit of a hit list, with its state at the last hit as the seed-state product stores it, and the
// checks of that state.
namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // An unfit state: NaN parameters and covariance, so that a consumer testing only std::isfinite rejects it.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void setUnfit(::lst::FittedState& state, uint32_t stateHit) {
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    auto& row = state.row;
    row.x = row.y = row.z = row.phi = row.cotTheta = row.kappa = kNaN;
    for (int i = 0; i < 6; ++i)
      row.covCircle[i] = kNaN;
    for (int i = 0; i < 3; ++i)
      row.covLine[i] = kNaN;
    row.charge = 0;
    state.stateHit = stateHit;
    state.status = static_cast<uint8_t>(::lst::SeedStateStatus::kNotFitted);
    state.nFit = 0;
    state.refFallback = 0;
  }

  // the checks of a stored state, shared with the host (RecoTracker/LSTCore/interface/SeedStateFitTypes.h)
  using ::lst::checkState;

  // Doubles in the work array of the fit of n hits: one (n+1)x(n+1) matrix, used by lineFitOuter (its leading n*n
  // doubles) and then by circleFitOuter.
  constexpr uint32_t fitWorkSize(uint32_t n) { return (n + 1) * (n + 1); }

  // Static shared memory per block: 48 KiB on the GPUs (every CUDA architecture CMSSW builds for; HIP allows 64 KiB),
  // and alpaka's block-shared capacity on the CPU backends (47 KiB by default). alpaka checks the CPU capacity only
  // with an assertion that a release build may compile out, so the fit kernels check their arrays against both
  // limits at compile time.
  constexpr uint32_t kFitSharedBytesGPU = 48u << 10;
  constexpr uint32_t kFitSharedBytesCPU = alpaka::BlockSharedDynMemberAllocKiB << 10;

  // Threads per block of the fit kernels of n hits on the GPUs, where each thread has its work array in block-shared
  // memory: the largest multiple of 8, at most 64, whose work arrays fit in kFitSharedBytesGPU. The kernels and their
  // launches use the same number.
  constexpr uint32_t fitThreadsPerBlock(uint32_t n) {
    uint32_t threads = 64;
    while (threads > 8 && threads * fitWorkSize(n) * sizeof(double) > kFitSharedBytesGPU)
      threads -= 8;
    return threads;
  }

  // The Broken Line Fit of N hits, inside out, in the global frame, double precision with float hit errors,
  // the state at the last hit: fastFit, prepareBrokenLineData, lineFitOuter, circleFitOuter, in this order
  // (the outer functions must run before the upstream circleFit, which modifies their input;
  // neither circleFit nor lineFit is run). work: fitWorkSize(N) doubles for the matrix of the two outer fits, which
  // run one after the other (see brokenline::circleFitOuter); its content on entry is not used.
  template <int N, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fitState(
      TAcc const& acc, ::lst::FitHit const* fitHits, float bField, ::lst::FittedState& state, double* work) {
    ::riemannFit::Matrix3xNd<N> hits;
    Eigen::Matrix<float, 6, N> hitsCov;
    for (int i = 0; i < N; ++i) {
      hits(0, i) = fitHits[i].x;
      hits(1, i) = fitHits[i].y;
      hits(2, i) = fitHits[i].z;
      for (int k = 0; k < 6; ++k)
        hitsCov(k, i) = fitHits[i].cov[k];
    }
    const double field = bField;
    ::riemannFit::Vector4d fastFit;
    brokenline::fastFit(acc, hits, fastFit);
    brokenline::PreparedBrokenLineData<N> data;
    brokenline::prepareBrokenLineData(acc, hits, fastFit, field, data);
    brokenline::OuterLineState line;
    brokenline::lineFitOuter(acc, hits, hitsCov, fastFit, field, data, line, work);
    brokenline::OuterCircleState circle;
    brokenline::circleFitOuter(acc, hits, hitsCov, fastFit, field, data, circle, work);

    auto& row = state.row;
    row.x = circle.point(0);
    row.y = circle.point(1);
    row.z = line.par(1);
    // reduced to [-pi, pi] in double, then rounded to float
    row.phi = static_cast<float>(cms::alpakatools::reducePhiRange(acc, circle.par(0)));
    row.cotTheta = line.par(0);
    row.kappa = circle.par(2);
    // upper triangles: (phi, x_T, kappa) and (cot(theta), z)
    row.covCircle[0] = circle.cov(0, 0);
    row.covCircle[1] = circle.cov(0, 1);
    row.covCircle[2] = circle.cov(1, 1);
    row.covCircle[3] = circle.cov(0, 2);
    row.covCircle[4] = circle.cov(1, 2);
    row.covCircle[5] = circle.cov(2, 2);
    row.covLine[0] = line.cov(0, 0);
    row.covLine[1] = line.cov(0, 1);
    row.covLine[2] = line.cov(1, 1);
    row.charge = circle.qCharge;
    state.nFit = N;
    state.refFallback = circle.refFallback;
    state.status = static_cast<uint8_t>(checkState(row));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
