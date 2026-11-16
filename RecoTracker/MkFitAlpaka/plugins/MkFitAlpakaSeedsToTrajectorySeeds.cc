// Validation only: the device mkFit seeds as building starts from them (the build module's seed rows after the pixel
// seed states and the device LST seed fit, MkFitAlpakaBuildProducer produceFittedSeeds, read as their host copy) as a
// TrajectorySeedCollection, so that the seed-level validation of the LST initial step (TrackFromSeedProducer of
// hltInitialStepTrajectorySeedsLST, MTV) works in the all-Alpaka menu, which makes no TrajectorySeeds. Not in the
// production menus (customizeHLTforMkFitAlpakaSeedValidation adds it). The seed hand-off's own rows are not used: their
// LST seeds without a pixel seed carry LSTOutputConverter's placeholder state until the build module's seed fit.
// Per seed row: the hits from their cluster keys (pixel: the MkFitClusterIndexToHit map; OT: made on demand from the
// clusters, as MkFitAlpakaOutputTrackConverter), the row state (mkFit CCS -> global curvilinear, as the output
// converter) propagated without material onto the surface of the hit it sits on (the seed hand-off states are on a
// hit: the smallest |local z| over the seed's hits), stored as the PTrajectoryStateOnDet of that det. Rows without a
// valid state (no charge, dropped fit: non-finite errors, no positive-definite error) or whose propagation fails are
// skipped (counted, LogDebug).

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2TrackerRecHitOnDemand.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitAlpaka/interface/TrackProduct.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class MkFitAlpakaSeedsToTrajectorySeeds : public edm::global::EDProducer<> {
public:
  explicit MkFitAlpakaSeedsToTrajectorySeeds(edm::ParameterSet const& iConfig)
      : seedsToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
        pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitPixelHits"))},
        otClustersToken_{consumes(iConfig.getParameter<edm::InputTag>("otClusters"))},
        otGeomToken_{esConsumes()},
        otCpeToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("Phase2StripCPE"))},
        mfToken_{esConsumes()},
        mkFitGeomToken_{esConsumes()},
        putToken_{produces()} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("seeds", edm::InputTag("hltInitialStepTrackCandidatesMkFitDevice", "fittedSeeds"))
        ->setComment("the device seed rows (MkFitAlpakaBuildProducer produceFittedSeeds), read as their host copy");
    desc.add<edm::InputTag>("mkFitPixelHits", edm::InputTag("hltMkFitEventOfHits", "pixelIndexToHit"))
        ->setComment("pixel cluster key -> legacy pixel rechit");
    desc.add<edm::InputTag>("otClusters", edm::InputTag("hltSiPhase2Clusters"))
        ->setComment("the OT clusters: OT hits made on demand from their keys");
    desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"));
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const override {
    auto const& soa = iEvent.get(seedsToken_);
    auto const v = soa.const_view();
    const int nSeeds = v.metadata().size() == 0 ? 0 : v.nTracks();
    auto const& pixelHits = iEvent.get(pixelHitsToken_).hits();
    auto const& mf = iSetup.getData(mfToken_);
    auto const& trackerInfo = iSetup.getData(mkFitGeomToken_).trackerInfo();
    const Phase2TrackerRecHitOnDemand otOnDemand(
        iEvent.getHandle(otClustersToken_), iSetup.getData(otGeomToken_), iSetup.getData(otCpeToken_));
    const AnalyticalPropagator propagator(&mf, anyDirection);

    TrajectorySeedCollection out;
    out.reserve(nSeeds);
    unsigned int nBadState = 0, nNoHit = 0, nBadProp = 0;
    for (int s = 0; s < nSeeds; ++s) {
      const auto row = v[s];
      mkfit::TrackState state;
      for (int k = 0; k < 6; ++k)
        state.parameters[k] = row.params().v[k];
      std::memcpy(state.errors.Array(), row.errors().v, sizeof(float) * 21);
      state.charge = row.charge();
      bool finite = std::isfinite(state.invpT());
      for (int k = 0; k < 21; ++k)
        finite = finite && std::isfinite(row.errors().v[k]);
      if (state.charge == 0 || !finite || !(state.invpT() > 0.f)) {
        ++nBadState;
        continue;
      }
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
      if (!fts.curvilinearError().posDef()) {
        ++nBadState;
        continue;
      }

      // the hits in seed order; the state's det = the hit surface closest to the state position
      edm::OwnVector<TrackingRecHit> hits;
      const GeomDet* stateDet = nullptr;
      float bestDz = std::numeric_limits<float>::max();
      for (int i = 0; i < row.nTotalHits(); ++i) {
        const auto hot = row.hits().hot[i];
        if (hot.index < 0)
          continue;
        std::optional<Phase2TrackerRecHit1D> otHit;
        TrackingRecHit const* hit = nullptr;
        if (trackerInfo.layer(hot.layer).is_pixel()) {
          if (static_cast<size_t>(hot.index) >= pixelHits.size() || pixelHits[hot.index] == nullptr)
            continue;
          hit = pixelHits[hot.index];
        } else {
          if (static_cast<uint32_t>(hot.index) >= otOnDemand.size())
            continue;
          otHit.emplace(otOnDemand.make(hot.index));
          hit = &*otHit;
        }
        const GeomDet* det = static_cast<BaseTrackerRecHit const*>(hit)->det();
        const float dz = std::abs(det->surface().toLocal(fts.position()).z());
        if (dz < bestDz) {
          bestDz = dz;
          stateDet = det;
        }
        hits.push_back(hit->clone());
      }
      if (hits.empty() || stateDet == nullptr) {
        ++nNoHit;
        continue;
      }
      const TrajectoryStateOnSurface tsos = propagator.propagate(fts, stateDet->surface());
      if (!tsos.isValid()) {
        ++nBadProp;
        continue;
      }
      out.emplace_back(trajectoryStateTransform::persistentState(tsos, stateDet->geographicalId().rawId()),
                       std::move(hits),
                       alongMomentum);
    }
    if (nBadState + nNoHit + nBadProp > 0)
      LogDebug("MkFitAlpakaSeedsToTrajectorySeeds") << "seed rows not converted: state " << nBadState << ", no hit "
                                                    << nNoHit << ", propagation " << nBadProp << " of " << nSeeds;
    iEvent.emplace(putToken_, std::move(out));
  }

private:
  const edm::EDGetTokenT<::mkfitdev::TrackSoAHostCollection> seedsToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelHitsToken_;
  const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otClustersToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> otGeomToken_;
  const edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> otCpeToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::EDPutTokenT<TrajectorySeedCollection> putToken_;
};

DEFINE_FWK_MODULE(MkFitAlpakaSeedsToTrajectorySeeds);
