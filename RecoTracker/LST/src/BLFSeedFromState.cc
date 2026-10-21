#include "RecoTracker/LST/interface/BLFSeedFromState.h"

#include <cmath>
#include <utility>

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/LST/interface/BLFCurvilinearJacobian.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

namespace lst {

  BLFStateOnPlane blfStateOnPlane(BLFStateRow const& row,
                                  uint8_t nFit,
                                  float bField,
                                  Plane const& plane,
                                  MagneticField const& magneticField,
                                  Propagator const& propagator,
                                  bool attachIfStepFails) {
    if (propagator.propagationDirection() != anyDirection)
      throw edm::Exception(edm::errors::LogicError)
          << "blfStateOnPlane: the propagator must propagate in anyDirection, the state's point can lie on either side "
             "of the hit's surface";
    if (nFit == 0)
      return {BLFSeedStatus::kNotFitted, 0., {}};
    switch (checkState(row)) {
      case SeedStateStatus::kOk:
        break;
      case SeedStateStatus::kNotFitted:  // not returned by checkState
        return {BLFSeedStatus::kNotFitted, 0., {}};
      case SeedStateStatus::kNonFinite:
        return {BLFSeedStatus::kNonFinite, 0., {}};
      case SeedStateStatus::kChargeMismatch:
        return {BLFSeedStatus::kChargeMismatch, 0., {}};
      case SeedStateStatus::kNonPositiveVariance:
        return {BLFSeedStatus::kNonPositiveVariance, 0., {}};
    }

    double covariance[5][5];
    blfToCurvilinearCovariance(row.kappa, row.cotTheta, bField, row.covCircle, row.covLine, covariance);
    for (int i = 0; i < 5; ++i)
      if (!(std::isfinite(covariance[i][i]) && covariance[i][i] > 0.))
        return {BLFSeedStatus::kCurvilinearErrorNotPositive, 0., {}};
    AlgebraicSymMatrix55 curvilinear;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        curvilinear(i, j) = covariance[i][j];

    // pT = bField / |kappa|, in float
    const float pt = bField / std::abs(row.kappa);
    const FreeTrajectoryState freeState(
        GlobalTrajectoryParameters(GlobalPoint(row.x, row.y, row.z),
                                   GlobalVector(pt * std::cos(row.phi), pt * std::sin(row.phi), pt * row.cotTheta),
                                   row.charge,
                                   &magneticField),
        CurvilinearTrajectoryError(curvilinear));

    // the step follows the helix: the device has no detector surfaces, and the state point is off the detector plane
    // by up to a few cm in the endcaps
    auto stepped = propagator.propagateWithPath(freeState, plane);
    if (stepped.first.isValid())
      return {BLFSeedStatus::kOk, stepped.second, std::move(stepped.first)};
    if (!attachIfStepFails)
      return {BLFSeedStatus::kStepFailed, 0., {}};
    return {BLFSeedStatus::kAttachedAfterFailedStep, 0., TrajectoryStateOnSurface(freeState, plane)};
  }

  BLFSeed blfSeedFromState(BLFStateRow const& row,
                           uint8_t nFit,
                           float bField,
                           std::span<TrackingRecHit const* const> hits,
                           MagneticField const& magneticField,
                           Propagator const& propagator,
                           bool attachIfStepFails) {
    if (hits.empty())
      throw edm::Exception(edm::errors::LogicError) << "blfSeedFromState: a seed needs at least one hit";
    for (TrackingRecHit const* hit : hits)
      if (hit == nullptr)
        throw edm::Exception(edm::errors::LogicError) << "blfSeedFromState: a null hit";
    TrackingRecHit const* lastHit = hits.back();
    if (lastHit->det() == nullptr)
      throw edm::Exception(edm::errors::LogicError) << "blfSeedFromState: the last hit has no detector";

    BLFStateOnPlane onPlane =
        blfStateOnPlane(row, nFit, bField, lastHit->det()->surface(), magneticField, propagator, attachIfStepFails);
    BLFSeed result{onPlane.status, onPlane.pathLength, std::nullopt};
    if (!onPlane.state.isValid())
      return result;

    TrajectorySeed::RecHitContainer seedHits;
    seedHits.reserve(hits.size());
    for (TrackingRecHit const* hit : hits)
      seedHits.push_back(hit->clone());
    result.seed.emplace(trajectoryStateTransform::persistentState(onPlane.state, lastHit->geographicalId().rawId()),
                        std::move(seedHits),
                        alongMomentum);
    return result;
  }

}  // namespace lst
