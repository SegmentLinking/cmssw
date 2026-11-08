#ifndef RecoTracker_LST_interface_BLFSeedFromState_h
#define RecoTracker_LST_interface_BLFSeedFromState_h

#include <cstdint>
#include <optional>
#include <span>

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoTracker/LSTCore/interface/SeedStateFitTypes.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class MagneticField;
class Plane;
class Propagator;

// A TrajectorySeed from a Broken Line Fit state stored by LST, with no refit: the state is checked, turned into a free
// state with its curvilinear error, and stepped along its helix onto the surface of the seed's last hit. Every module
// that builds seeds from these states calls these functions, compiled once in this package's library, so that the
// same row gives bit-identical seeds in all of them.
namespace lst {

  enum class BLFSeedStatus : uint8_t {
    kOk = 0,
    // nFit = 0
    kNotFitted,
    // the three rejections of lst::checkState
    kNonFinite,
    kChargeMismatch,
    kNonPositiveVariance,
    // a diagonal element of the curvilinear error is not positive or not finite; for a row that passes
    // lst::checkState this needs a field value bField that is zero or not finite
    kCurvilinearErrorNotPositive,
    // the helix step onto the surface failed; no state
    kStepFailed,
    // the helix step failed and the free state was attached to the surface without propagation
    kAttachedAfterFailedStep,
  };

  struct BLFStateOnPlane {
    BLFSeedStatus status;
    double pathLength;               // signed path length of the step, cm; 0 when attached or rejected
    TrajectoryStateOnSurface state;  // valid for kOk and kAttachedAfterFailedStep only
  };

  struct BLFSeed {
    BLFSeedStatus status;
    double pathLength;                   // as BLFStateOnPlane::pathLength
    std::optional<TrajectorySeed> seed;  // for kOk and kAttachedAfterFailedStep only
  };

  // The stored state (row, nFit, the field value bField of the seed-state product, GeV/cm) on a plane:
  //  1. rejected if nFit = 0 or if lst::checkState fails;
  //  2. the curvilinear error from lst::blfToCurvilinearCovariance, rejected if a diagonal element is not positive;
  //  3. the free state at (x, y, z) with the float momentum bField / |kappa| (cos phi, sin phi, cotTheta) and the
  //     row's charge;
  //  4. propagated onto the plane with propagator (the caller's any-direction analytical propagator); if that fails,
  //     rejected, or with attachIfStepFails the free state attached to the plane as it is.
  // bField is the value LST stores, positive; a negative value reverses the momentum, and a zero or non-finite one
  // gives kCurvilinearErrorNotPositive. propagator must have propagationDirection() == anyDirection, else a LogicError:
  // the state's point can lie on either side of the plane. plane must be reference counted (a GeomDet surface or a
  // Plane::build result): the returned TrajectoryStateOnSurface keeps a pointer to it.
  BLFStateOnPlane blfStateOnPlane(BLFStateRow const& row,
                                  uint8_t nFit,
                                  float bField,
                                  Plane const& plane,
                                  MagneticField const& magneticField,
                                  Propagator const& propagator,
                                  bool attachIfStepFails);

  // The seed of the stored state: blfStateOnPlane on the surface of the last hit's detector, then the persistent state
  // on that detector, clones of the hits in their order, and alongMomentum. hits: inside out, the hit the state sits
  // on last; an empty list, a null hit or a last hit without a detector is a LogicError. bField and propagator as for
  // blfStateOnPlane.
  BLFSeed blfSeedFromState(BLFStateRow const& row,
                           uint8_t nFit,
                           float bField,
                           std::span<TrackingRecHit const* const> hits,
                           MagneticField const& magneticField,
                           Propagator const& propagator,
                           bool attachIfStepFails);

}  // namespace lst

#endif  // RecoTracker_LST_interface_BLFSeedFromState_h
