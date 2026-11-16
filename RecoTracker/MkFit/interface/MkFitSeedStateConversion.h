#ifndef RecoTracker_MkFit_interface_MkFitSeedStateConversion_h
#define RecoTracker_MkFit_interface_MkFitSeedStateConversion_h

// The state conversion of the mkFit seeds to TrajectorySeeds, for the modules that use the mkFit seeder's seeds
// (MkSeederTrajectorySeedConverter, phase2MkSeeder): an mkFit seed's state, at its last hit, as the TrajectorySeed's
// starting state on the last hit's module surface.

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include <vector>

class GeomDet;
class MagneticField;
class Propagator;
namespace mkfit {
  class Track;
  class TrackerInfo;
}  // namespace mkfit

namespace mkfitconv {

  // The seed's state at its last hit (mkFit CCS) as a curvilinear state, taken onto lastDet's surface (within 1 um
  // of it and !propagateToSurface: as on it, else propagated along, then opposite to the momentum), as a
  // PTrajectoryStateOnDet. false: the conversion failed (a covariance not positive definite, or no propagation); the
  // converter drops such a seed.
  bool seedStateOnLastHit(const mkfit::Track& seed,
                          const GeomDet& lastDet,
                          const MagneticField& mf,
                          const Propagator& along,
                          const Propagator& opposite,
                          bool propagateToSurface,
                          PTrajectoryStateOnDet& out);
  // The same state before the persistent form (the TrajectorySeed's state is persistentState(out, the last module's
  // id)): false as above.
  bool seedTsosOnLastHit(const mkfit::Track& seed,
                         const GeomDet& lastDet,
                         const MagneticField& mf,
                         const Propagator& along,
                         const Propagator& opposite,
                         bool propagateToSurface,
                         TrajectoryStateOnSurface& out);

  // The seed's hits' TrackingRecHits (inside out), appended to 'out', from the cluster-index tables of the mkFit hit
  // converters; throws as the converter does for a hit without one.
  void seedHits(const mkfit::Track& seed,
                const mkfit::TrackerInfo& ti,
                const std::vector<const TrackingRecHit*>& pixelHits,
                const std::vector<const TrackingRecHit*>& stripHits,
                std::vector<const TrackingRecHit*>& out);

}  // namespace mkfitconv

#endif
