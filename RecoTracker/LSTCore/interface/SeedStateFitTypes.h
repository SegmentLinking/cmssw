#ifndef RecoTracker_LSTCore_interface_SeedStateFitTypes_h
#define RecoTracker_LSTCore_interface_SeedStateFitTypes_h

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

namespace lst {

  // Outcome of the Broken Line Fit and of its checks for one seed state.
  enum class SeedStateStatus : uint8_t {
    kOk = 0,
    // No fit was attempted or possible. Pixel track: no reco::Track, hit count out of range, or IT-only state not
    // computed. Track candidate: a pLS, or fewer hits than the smallest fit size after the hit selection.
    kNotFitted,
    // A parameter or a covariance element is not finite.
    kNonFinite,
    // sign(kappa) differs from the charge (kappa = 0 counts as a mismatch).
    kChargeMismatch,
    // A diagonal variance of the native covariance blocks is not positive.
    kNonPositiveVariance,
  };

  // One stored Broken Line Fit state at the outermost fitted node.
  struct BLFStateRow {
    float x, y, z;
    float phi, cotTheta, kappa;
    float covCircle[6];
    float covLine[3];
    int8_t charge;
  };

  // A hit as it enters the fit: global position (cm) and global covariance (cm^2) in the order of the Broken Line
  // Fit's hit errors: xx, xy, yy, xz, yz, zz.
  struct FitHit {
    float x, y, z;
    float cov[6];
  };

  // A fitted state as the seed-state product stores it.
  struct FittedState {
    ::lst::BLFStateRow row;
    uint32_t stateHit;    // the encoded hit the state sits on: lst::encodeStateHitOT or lst::encodeStateHitPixel
    uint8_t status;       // lst::SeedStateStatus
    uint8_t nFit;         // fitted hits, 0 if unfit
    uint8_t refFallback;  // the reference node of the outer chord is the first fitted node
  };

  // The status of a stored state: every parameter and covariance element finite, else kNonFinite; sign(kappa) equal
  // to the charge (kappa = 0 is a mismatch), else kChargeMismatch; the diagonal variances of the native covariance
  // blocks positive, else kNonPositiveVariance. Shared by the device fits and the host seed building.
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

}  // namespace lst

#endif
