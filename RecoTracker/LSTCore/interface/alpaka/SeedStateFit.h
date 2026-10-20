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

  // The status of a stored state: every parameter and covariance element finite, else kNonFinite; sign(kappa) equal
  // to the charge (kappa = 0 is a mismatch), else kChargeMismatch; the diagonal variances of the native covariance
  // blocks positive, else kNonPositiveVariance.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE ::lst::SeedStateStatus checkState(::lst::BLFStateRow const& row) {
    bool finite = std::isfinite(row.x) && std::isfinite(row.y) && std::isfinite(row.z) && std::isfinite(row.phi) &&
                  std::isfinite(row.cotTheta) && std::isfinite(row.kappa);
    for (int i = 0; i < 6; ++i)
      finite = finite && std::isfinite(row.covCircle[i]);
    for (int i = 0; i < 3; ++i)
      finite = finite && std::isfinite(row.covLine[i]);
    if (!finite)
      return ::lst::SeedStateStatus::kNonFinite;
    if (!((row.kappa > 0.f && row.charge > 0) || (row.kappa < 0.f && row.charge < 0)))
      return ::lst::SeedStateStatus::kChargeMismatch;
    if (!(row.covCircle[0] > 0.f && row.covCircle[2] > 0.f && row.covCircle[5] > 0.f && row.covLine[0] > 0.f &&
          row.covLine[2] > 0.f))
      return ::lst::SeedStateStatus::kNonPositiveVariance;
    return ::lst::SeedStateStatus::kOk;
  }

  // The Broken Line Fit of N hits, inside out, in the global frame, double precision with float hit errors,
  // the state at the last hit: fastFit, prepareBrokenLineData, lineFitOuter, circleFitOuter, in this order
  // (the outer functions must run before the upstream circleFit, which modifies their input;
  // neither circleFit nor lineFit is run).
  template <int N, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fitState(TAcc const& acc,
                                               ::lst::FitHit const* fitHits,
                                               float bField,
                                               ::lst::FittedState& state) {
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
    brokenline::lineFitOuter(acc, hits, hitsCov, fastFit, field, data, line);
    brokenline::OuterCircleState circle;
    brokenline::circleFitOuter(acc, hits, hitsCov, fastFit, field, data, circle);

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
