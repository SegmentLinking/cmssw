#include "RecoTracker/MkFit/interface/MkFitSeedStateConversion.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include <cmath>

namespace mkfitconv {

  bool seedStateOnLastHit(const mkfit::Track& seed,
                          const GeomDet& lastDet,
                          const MagneticField& mf,
                          const Propagator& along,
                          const Propagator& opposite,
                          bool propagateToSurface,
                          PTrajectoryStateOnDet& out) {
    TrajectoryStateOnSurface tsos;
    if (!seedTsosOnLastHit(seed, lastDet, mf, along, opposite, propagateToSurface, tsos))
      return false;
    out = trajectoryStateTransform::persistentState(tsos, lastDet.geographicalId().rawId());
    return true;
  }

  bool seedTsosOnLastHit(const mkfit::Track& seed,
                         const GeomDet& lastDet,
                         const MagneticField& mf,
                         const Propagator& along,
                         const Propagator& opposite,
                         bool propagateToSurface,
                         TrajectoryStateOnSurface& out) {
    constexpr float kOnSurfaceTolerance = 1e-4f;  // cm
    auto state = seed.state();
    state.convertFromCCSToGlbCurvilinear();
    const auto& par = state.parameters;
    AlgebraicSymMatrix55 cov;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        cov[i][j] = state.errors.At(i, j);
    const FreeTrajectoryState fts(
        GlobalTrajectoryParameters(
            GlobalPoint(par[0], par[1], par[2]), GlobalVector(par[3], par[4], par[5]), state.charge, &mf),
        CurvilinearTrajectoryError(cov));
    if (!fts.curvilinearError().posDef())
      return false;
    const bool onSurface =
        !propagateToSurface && std::abs(lastDet.surface().toLocal(fts.position()).z()) < kOnSurfaceTolerance;
    auto tsos = onSurface ? TrajectoryStateOnSurface(fts, lastDet.surface()) : along.propagate(fts, lastDet.surface());
    if (!tsos.isValid())
      tsos = opposite.propagate(fts, lastDet.surface());
    if (!tsos.isValid())
      return false;
    out = tsos;
    return true;
  }

  void seedHits(const mkfit::Track& seed,
                const mkfit::TrackerInfo& ti,
                const std::vector<const TrackingRecHit*>& pixelHits,
                const std::vector<const TrackingRecHit*>& stripHits,
                std::vector<const TrackingRecHit*>& out) {
    for (int i = 0; i < seed.nTotalHits(); ++i) {
      const auto& hot = seed.getHitOnTrack(i);
      if (hot.index < 0 || hot.layer < 0 || hot.layer >= ti.n_layers())
        throw cms::Exception("LogicError") << "mkfitconv::seedHits: seed label " << seed.label() << " has hit " << i
                                           << " with layer " << hot.layer << " index " << hot.index;
      const auto& hits = ti[hot.layer].is_pixel() ? pixelHits : stripHits;
      if (hot.index >= (int)hits.size() || hits[hot.index] == nullptr)
        throw cms::Exception("LogicError") << "mkfitconv::seedHits: seed label " << seed.label() << " hit " << i
                                           << " has no TrackingRecHit at cluster index " << hot.index;
      out.push_back(hits[hot.index]);
    }
  }

}  // namespace mkfitconv
