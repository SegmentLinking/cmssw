#ifndef RecoTracker_TkSeedGenerator_interface_SeedFromConsecutiveHitsFit_h
#define RecoTracker_TkSeedGenerator_interface_SeedFromConsecutiveHitsFit_h

// The fit of SeedFromConsecutiveHitsCreator::makeSeed callable from GPU code, one track at a time, with MkFitCore's
// portable propagation and Kalman update (parametrised field, module material as PropagatorWithMaterial).

#include <cmath>

#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/portable/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitCore/interface/portable/Macros.h"
#include "RecoTracker/MkFitCore/interface/portable/PropagationMPlex.h"

namespace seedFromConsecutiveHits {

  // one hit of the seed: global position and covariance, and the plane, material and local x axis of its module
  struct Hit {
    float x, y, z;
    float covXX, covYX, covYY, covZX, covZY, covZZ;
    float planeX, planeY, planeZ;
    float normalX, normalY, normalZ;
    float axisX, axisY, axisZ;
    float radLen, xi;
  };

  // GlobalTrackingRegion of the creator: origin (x, y), ptMin, and the squared origin radius and half-length
  struct Region {
    float originX, originY;
    float ptMin;
    float originRadius2;
    float originHalfLength2;
  };

  // creator and propagator settings; env holds MkFitCore's field parametrisation
  struct Settings {
    float minOneOverPtError = 1.f;         // SeedCreatorPSet MinOneOverPtError
    float maxDPhi = 1.6f;                  // PropagatorWithMaterial MaxDPhi
    float backwardTolerance = 0.01f;       // cm: a step shorter than this against the momentum is still accepted
    float retryAngleVariance = 0.01f;      // retry after a float failure: capped prior of the angles (rad^2)
    float retryTransverseVariance = 25.f;  // ... and of the transverse position along the track (cm^2)
    mkfit::PropagationEnv env;
  };

  enum class Status : int8_t { ok = 0, rejectedDPhi = 1, rejectedBackward = 2, invalid = 3 };

  // state on the last hit in MkFitCore's CCS (x, y, z, 1/pT, phi, theta), errors in its packed lower triangle
  struct State {
    float parameters[6];
    float errors[21];
    int charge;
  };

}  // namespace seedFromConsecutiveHits

namespace seedFromConsecutiveHits::inline MKFIT_PORTABLE_NAMESPACE {

  using namespace mkfit::portable;

  // initialKinematic (FastHelix through hit 1, hit 0 and the region origin) and initialError of the creator
  MKFIT_HOST_DEVICE inline void startState(Hit const* hits,
                                           Region const& region,
                                           Settings const& settings,
                                           bool cappedPrior,
                                           MPlexLV<1>& parameters,
                                           MPlexLS<1>& errors,
                                           MPlexQI<1>& charge) {
    const double vx = region.originX, vy = region.originY;
    const double mx = hits[0].x, my = hits[0].y, mz = hits[0].z;
    const double ox = hits[1].x, oy = hits[1].y, oz = hits[1].z;
    constexpr double kCm2GeV = 0.01 * 0.3 * 3.8;  // the creator's nominal field
    constexpr double kMaxPt = 10000.;
    constexpr double kMaxRho = kMaxPt / kCm2GeV;
    // circle through the three points (FastCircle is exact for three points)
    const double ax = mx - vx, ay = my - vy, bx = ox - vx, by = oy - vy;
    const double cross = ax * by - ay * bx;
    double centerX = 0., centerY = 0., rho = 0.;
    const bool circle = std::abs(cross) > 1e-12 * (ax * ax + ay * ay) * (bx * bx + by * by) / (1. + ax * ax + ay * ay);
    if (circle) {
      const double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by;
      centerX = vx + (by * a2 - ay * b2) / (2. * cross);
      centerY = vy + (ax * b2 - bx * a2) / (2. * cross);
      rho = std::sqrt((centerX - vx) * (centerX - vx) + (centerY - vy) * (centerY - vy));
    }
    bool helix = circle && rho < kMaxRho;
    double cosDPhi = 0.;
    if (helix) {
      cosDPhi = ((ox - centerX) * (mx - centerX) + (oy - centerY) * (my - centerY)) / (rho * rho);
      helix = std::abs(cosDPhi) < 1.;
    }
    double px, py, pz, zv;
    int chargeSign = 1;
    if (helix) {  // FastHelix::helixStateAtVertex
      const double pt = kCm2GeV * rho;
      px = -kCm2GeV * (vy - centerY);
      py = kCm2GeV * (vx - centerX);
      if (px * (mx - vx) + py * (my - vy) < 0.) {
        px = -px;
        py = -py;
      }
      const double dzdrphi = (oz - mz) / (rho * std::acos(cosDPhi));
      pz = pt * dzdrphi;
      if (centerX * py - centerY * px < 0)
        chargeSign = -chargeSign;
      zv = mz;
      double arc = ((vx - centerX) * (mx - centerX) + (vy - centerY) * (my - centerY)) / (rho * rho);
      if (std::abs(arc) < 1.) {
        zv -= rho * std::acos(arc) * dzdrphi;
      } else {
        const double dmv2 = (mx - vx) * (mx - vx) + (my - vy) * (my - vy);
        const double dom2 = (ox - mx) * (ox - mx) + (oy - my) * (oy - my);
        zv -= std::sqrt(dmv2 / dom2) * (oz - mz);
      }
    } else {  // FastHelix::straightLineStateAtVertex
      const double chord = std::sqrt(ax * ax + ay * ay);
      px = kMaxPt * ax / chord;
      py = kMaxPt * ay / chord;
      const double rm = std::sqrt(mx * mx + my * my), ro = std::sqrt(ox * ox + oy * oy);
      const double dzdr = (oz - mz) / (ro - rm);
      pz = kMaxPt * dzdr;
      zv = mz - rm * dzdr;
    }
    const double ptVertex = std::sqrt(px * px + py * py);
    const double momentum = std::sqrt(px * px + py * py + pz * pz);
    const float sinL = pz / momentum, cosL = ptVertex / momentum, sinF = py / ptVertex, cosF = px / ptVertex;
    const float invPt = 1. / ptVertex;
    parameters(0, 0, 0) = vx;
    parameters(0, 1, 0) = vy;
    parameters(0, 2, 0) = zv;
    parameters(0, 3, 0) = invPt;
    parameters(0, 4, 0) = std::atan2(py, px);
    parameters(0, 5, 0) = std::atan2(ptVertex, pz);
    charge(0, 0, 0) = chargeSign;

    // curvilinear diag(q/p, lambda, phi, xT, yT) of the region rotated to the CCS
    const float sin2Theta = cosL * cosL;
    float varQoverP =
        std::max(sin2Theta / (region.ptMin * region.ptMin), settings.minOneOverPtError * settings.minOneOverPtError);
    float varLambda = 1.f, varPhi = 1.f;  // as the creator
    const float varXT = region.originRadius2;
    float varYT = region.originHalfLength2 * sin2Theta + varXT * (1.f - sin2Theta);
    if (cappedPrior) {
      const float sigmaQoverP = (invPt + 0.01f) * cosL;
      varQoverP = std::min(varQoverP, sigmaQoverP * sigmaQoverP);
      varLambda = settings.retryAngleVariance;
      varPhi = settings.retryAngleVariance;
      varYT = std::min(varYT, settings.retryTransverseVariance);
    }
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j)
        errors(0, i, j) = 0.f;
    errors(0, 0, 0) = sinF * sinF * varXT + sinL * sinL * cosF * cosF * varYT;
    errors(0, 1, 0) = -sinF * cosF * varXT + sinL * sinL * sinF * cosF * varYT;
    errors(0, 1, 1) = cosF * cosF * varXT + sinL * sinL * sinF * sinF * varYT;
    errors(0, 2, 0) = -sinL * cosL * cosF * varYT;
    errors(0, 2, 1) = -sinL * cosL * sinF * varYT;
    errors(0, 2, 2) = cosL * cosL * varYT;
    const float dInvPtdLambda = invPt * sinL / cosL;
    errors(0, 3, 3) = varQoverP / (cosL * cosL) + dInvPtdLambda * dInvPtdLambda * varLambda;
    errors(0, 4, 4) = varPhi;
    errors(0, 5, 3) = -dInvPtdLambda * varLambda;
    errors(0, 5, 5) = varLambda;
  }

  // buildSeed: per hit, propagation to its module plane, the module material, then the update
  MKFIT_HOST_DEVICE inline Status forwardPass(Hit const* hits,
                                              int nHits,
                                              Settings const& settings,
                                              MPlexLV<1>& parameters,
                                              MPlexLS<1>& errors,
                                              MPlexQI<1>& charge) {
    mkfit::PropagationFlags flags(mkfit::PF_use_param_b_field);
    flags.env = settings.env;
    MPlexLS<1> propErrors, stepErrors;
    MPlexLV<1> propParameters, stepParameters;
    MPlexQI<1> failFlag;
    MPlexQF<1> radLen, xi, propSign;
    MPlexHS<1> hitErrors;
    MPlexHV<1> hitPosition, normal, axis, point;
    propSign(0, 0, 0) = 1.f;  // along the momentum
    for (int iHit = 0; iHit < nHits; ++iHit) {
      Hit const& hit = hits[iHit];
      hitPosition(0, 0, 0) = hit.x;
      hitPosition(0, 1, 0) = hit.y;
      hitPosition(0, 2, 0) = hit.z;
      hitErrors(0, 0, 0) = hit.covXX;
      hitErrors(0, 1, 0) = hit.covYX;
      hitErrors(0, 1, 1) = hit.covYY;
      hitErrors(0, 2, 0) = hit.covZX;
      hitErrors(0, 2, 1) = hit.covZY;
      hitErrors(0, 2, 2) = hit.covZZ;
      point(0, 0, 0) = hit.planeX;
      point(0, 1, 0) = hit.planeY;
      point(0, 2, 0) = hit.planeZ;
      normal(0, 0, 0) = hit.normalX;
      normal(0, 1, 0) = hit.normalY;
      normal(0, 2, 0) = hit.normalZ;
      axis(0, 0, 0) = hit.axisX;
      axis(0, 1, 0) = hit.axisY;
      axis(0, 2, 0) = hit.axisZ;
      radLen(0, 0, 0) = hit.radLen;
      xi(0, 0, 0) = hit.xi;
      failFlag(0, 0, 0) = 0;
      stepParameters = parameters;
      stepErrors = errors;
      // the long step from the origin to the first hit is refined by a second propagation from near the plane
      const int nSteps = iHit == 0 ? 2 : 1;
      for (int step = 0; step < nSteps; ++step) {
        propagateHelixToPlaneMPlex<1>(
            stepErrors, stepParameters, charge, point, normal, propErrors, propParameters, failFlag, 1, flags);
        if (failFlag(0, 0, 0))
          return Status::invalid;
        stepParameters = propParameters;
        stepErrors = propErrors;
      }
      applyMaterialEffects<1>(radLen, xi, propSign, normal, propErrors, propParameters, 1, settings.env);
      // AnalyticalPropagator: a step turning by more than MaxDPhi fails; the propagation is along the momentum
      MPlexLV<1> turn = propParameters;
      turn(0, 4, 0) -= parameters(0, 4, 0);
      squashPhiMPlexGeneral<1>(turn, 1);
      if (std::abs(turn(0, 4, 0)) > settings.maxDPhi)
        return Status::rejectedDPhi;
      const float sinTheta = std::sin(parameters(0, 5, 0));
      const float alongMomentum =
          (propParameters(0, 0, 0) - parameters(0, 0, 0)) * std::cos(parameters(0, 4, 0)) * sinTheta +
          (propParameters(0, 1, 0) - parameters(0, 1, 0)) * std::sin(parameters(0, 4, 0)) * sinTheta +
          (propParameters(0, 2, 0) - parameters(0, 2, 0)) * std::cos(parameters(0, 5, 0));
      if (alongMomentum < -settings.backwardTolerance)
        return Status::rejectedBackward;
      kalmanPropagateAndUpdatePlane<1>(propErrors,
                                       propParameters,
                                       charge,
                                       hitErrors,
                                       hitPosition,
                                       normal,
                                       axis,
                                       point,
                                       errors,
                                       parameters,
                                       failFlag,
                                       1,
                                       flags,
                                       false);
      for (int k = 0; k < 6; ++k)
        if (!std::isfinite(parameters(0, k, 0)))
          return Status::invalid;
    }
    for (int k = 0; k < 6; ++k)
      if (!(std::isfinite(errors(0, k, k)) && errors(0, k, k) > 0.f))
        return Status::invalid;
    return Status::ok;
  }

  // the seed state on the last hit; a float failure with the creator's prior is refitted once with a capped prior
  MKFIT_HOST_DEVICE inline Status fit(
      Hit const* hits, int nHits, Region const& region, Settings const& settings, State& state) {
    if (nHits < 2)
      return Status::invalid;
    MPlexLV<1> parameters;
    MPlexLS<1> errors;
    MPlexQI<1> charge;
    Status status = Status::invalid;
    for (int attempt = 0; attempt < 2 && status == Status::invalid; ++attempt) {
      startState(hits, region, settings, attempt == 1, parameters, errors, charge);
      status = forwardPass(hits, nHits, settings, parameters, errors, charge);
    }
    if (status != Status::ok)
      return status;
    for (int k = 0; k < 6; ++k)
      state.parameters[k] = parameters(0, k, 0);
    errors.copyOut(0, state.errors);
    state.charge = charge(0, 0, 0);
    return status;
  }

}  // namespace seedFromConsecutiveHits::inline MKFIT_PORTABLE_NAMESPACE

#endif
