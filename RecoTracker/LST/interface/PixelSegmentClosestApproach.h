#ifndef RecoTracker_LST_interface_PixelSegmentClosestApproach_h
#define RecoTracker_LST_interface_PixelSegmentClosestApproach_h

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/LST/interface/BLFCurvilinearJacobian.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentParameters.h"
#include "RecoTracker/LSTCore/interface/SeedStateFitTypes.h"
#include "TrackingTools/PatternTools/interface/beamLineClosestApproach.h"
#include "TrackingTools/TrajectoryState/interface/curvilinear2PerigeeJacobian.h"

namespace lst {

  // The result of closestApproach. On a failure (status not kValid) the momentum,
  // dxy, dz, ptErr, etaErr and the charge are zero.
  struct PixelSegmentClosestApproach {
    enum Status : uint8_t {
      kValid = 0,
      // the closest approach was not found: no convergence, or a state that is not a helix (q = 0, Bz = 0, p = 0)
      kNotConverged = 1,
      // a non-finite input, momentum or output
      kNonFinite = 2,
    };
    float px, py, pz;     // momentum at the closest approach, GeV
    float dxy, dz;        // relative to the beam-spot point, as pixelSegmentImpactParameters
    float ptErr, etaErr;  // as pixelSegmentMomentumErrors, from the perigee (transverse curvature, theta) errors
    int8_t charge;
    uint8_t status;
  };

  class UniformField {
  public:
    using Vector3 = beamLineClosestApproach::Vector3;

    ALPAKA_FN_HOST_ACC constexpr explicit UniformField(float bField) : bField_(bField) {}

    ALPAKA_FN_HOST_ACC constexpr float bField() const { return bField_; }
    ALPAKA_FN_HOST_ACC constexpr Vector3 inTesla(Vector3 const&) const { return Vector3{0.f, 0.f, bField_ / kC}; }
    ALPAKA_FN_HOST_ACC constexpr Vector3 inInverseGeV(Vector3 const& position) const {
      const Vector3 tesla = inTesla(position);
      return Vector3{kC * tesla.x, kC * tesla.y, kC * tesla.z};
    }

  private:
    float bField_;
  };

  namespace closestApproachDetail {

    struct SinCos {
      ALPAKA_FN_HOST_ACC void operator()(double angle, double& sinAngle, double& cosAngle) const {
        sinAngle = std::sin(angle);
        cosAngle = std::cos(angle);
      }
    };

    // the free state at the stored point:
    // momentum pT (cos phi, sin phi, cot) and the curvilinear covariance of blfToCurvilinearCovariance;
    // false if an input or the momentum is not finite (kappa = 0 included).
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool freeState(BLFStateRow const& row,
                                                       float bField,
                                                       beamLineClosestApproach::FreeState& state) {
      bool finite = std::isfinite(row.x) && std::isfinite(row.y) && std::isfinite(row.z) && std::isfinite(row.phi) &&
                    std::isfinite(row.cotTheta) && std::isfinite(row.kappa);
      for (int i = 0; i < 6; ++i)
        finite = finite && std::isfinite(row.covCircle[i]);
      for (int i = 0; i < 3; ++i)
        finite = finite && std::isfinite(row.covLine[i]);
      const float pt = bField / std::abs(row.kappa);
      state.position = {row.x, row.y, row.z};
      state.momentum = {pt * std::cos(row.phi), pt * std::sin(row.phi), pt * row.cotTheta};
      state.charge = row.charge;
      if (!(finite && std::isfinite(state.momentum.x) && std::isfinite(state.momentum.y) &&
            std::isfinite(state.momentum.z)))
        return false;
      blfToCurvilinearCovariance(row.kappa, row.cotTheta, bField, row.covCircle, row.covLine, state.error);
      return true;
    }

    // the (transverse curvature, theta) block of the perigee errors of a state,
    // as PerigeeConversions::ftsToPerigeeError, with the field fieldInInverseGeV at its position, ptErr and etaErr
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void momentumErrors(beamLineClosestApproach::FreeState const& state,
                                                            beamLineClosestApproach::Vector3 const& fieldInInverseGeV,
                                                            float& ptErr,
                                                            float& etaErr) {
      using curvilinear2PerigeeJacobian::Vector3;
      const Vector3 momentum(state.momentum.x, state.momentum.y, state.momentum.z);
      const float pt = momentum.perp();
      curvilinear2PerigeeJacobian::Matrix5 jacobian;
      curvilinear2PerigeeJacobian::compute(momentum,
                                           Vector3(fieldInInverseGeV.x, fieldInInverseGeV.y, fieldInInverseGeV.z),
                                           state.charge / momentum.mag(),
                                           -state.charge / pt * fieldInInverseGeV.z,
                                           SinCos{},
                                           jacobian);
      double perigee[2][2] = {};
      for (int row = 0; row < 2; ++row)
        for (int col = 0; col <= row; ++col) {
          double sum = 0.;
          for (int a = 0; a < 5; ++a)
            for (int b = 0; b < 5; ++b)
              sum += jacobian(row, a) * state.error[a][b] * jacobian(col, b);
          perigee[row][col] = perigee[col][row] = sum;
        }
      pixelSegmentMomentumErrors(
          pt, momentum.mag(), momentum.z(), state.charge, perigee[0][0], perigee[0][1], perigee[1][1], ptErr, etaErr);
    }

  }  // namespace closestApproachDetail

  // The pLS quantities at the closest approach to the beam line {(x0, y0, z0), (dxdz, dydz, 1)} of the stored state
  // row, propagated without material in field
  template <typename Field>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PixelSegmentClosestApproach
  closestApproach(BLFStateRow const& row, Field const& field, beamLineClosestApproach::BeamLine const& beamLine) {
    PixelSegmentClosestApproach result{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0, PixelSegmentClosestApproach::kValid};
    beamLineClosestApproach::FreeState lastHit;
    if (!closestApproachDetail::freeState(row, field.bField(), lastHit)) {
      result.status = PixelSegmentClosestApproach::kNonFinite;
      return result;
    }
    beamLineClosestApproach::FreeState pca;
    if (!beamLineClosestApproach::stateNoMaterial(lastHit, field.inTesla(lastHit.position), beamLine, pca)) {
      result.status = PixelSegmentClosestApproach::kNotConverged;
      return result;
    }

    const float px = pca.momentum.x, py = pca.momentum.y, pz = pca.momentum.z;
    float dxy = 0.f, dz = 0.f;
    pixelSegmentImpactParameters(pca.position.x,
                                 pca.position.y,
                                 pca.position.z,
                                 px,
                                 py,
                                 pz,
                                 std::sqrt(px * px + py * py),
                                 beamLine.position.x,
                                 beamLine.position.y,
                                 beamLine.position.z,
                                 dxy,
                                 dz);
    float ptErr = 0.f, etaErr = 0.f;
    closestApproachDetail::momentumErrors(pca, field.inInverseGeV(pca.position), ptErr, etaErr);

    if (!(std::isfinite(px) && std::isfinite(py) && std::isfinite(pz) && std::isfinite(dxy) && std::isfinite(dz) &&
          std::isfinite(ptErr) && std::isfinite(etaErr))) {
      result.status = PixelSegmentClosestApproach::kNonFinite;
      return result;
    }
    result.px = px;
    result.py = py;
    result.pz = pz;
    result.dxy = dxy;
    result.dz = dz;
    result.ptErr = ptErr;
    result.etaErr = etaErr;
    result.charge = row.charge;
    return result;
  }

}  // namespace lst

#endif  // RecoTracker_LST_interface_PixelSegmentClosestApproach_h
