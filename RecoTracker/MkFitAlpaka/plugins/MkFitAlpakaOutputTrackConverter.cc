// MkFitAlpakaOutputTrackConverter: host conversion of the device final fit's TrackSoA (host copy) to reco::Track,
// TrackExtra, TrackingRecHitCollection and SeedStopInfo, in place of MkFitOutputTrackConverter in the device-fit menu.
// It makes the same products as the no-hit-state branch of MkFitOutputTrackConverter, with the same CMSSW calls in the
// same order: state quality and positive-definiteness checks, the state at the beam line, the hit pattern (the hits
// the final fit removed as outliers as lost hits) and the inner / outer missing-hit navigation (outward from the fit's
// state at the outermost hit when that hit is the outermost rec hit). Differences in the implementation:
//   - the TrackSoA rows are read directly; is_pixel per layer comes from the MkFitGeometry TrackerInfo;
//   - the on-track rec hits are referenced while the track is built and cloned once, into the output collection;
//   - the hits are put in the order of the final fit (R^2 + (z - z0)^2 of the hit positions, then the HitOnTrack
//     position), with the keys computed once per hit.
// Supported: Phase-2 geometry, no per-hit states (trajectoryInEvent is refused).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitAlpaka/interface/StatusProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/StatusReport.h"
#include "RecoTracker/MkFitAlpaka/interface/TrackProduct.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace {
  // the hit order of MkFitOutputTrackConverter (the order of the final fit): the key R^2 + (z - z0)^2 of the mkFit hit
  // position (the rec hit's global position, as convertHits), then the HitOnTrack position
  struct HitKey {
    float key;
    int hot;
  };
  inline bool hitLess(const HitKey& lhs, const HitKey& rhs) {
    return lhs.key < rhs.key || (lhs.key == rhs.key && lhs.hot < rhs.hot);
  }

  // the final fit's state at the outermost hit on that hit's module (MkFitOutputTrackConverter toTSOS)
  TrajectoryStateOnSurface outerTSOS(::mkfitdev::TrackOuterState const& st,
                                     const GeomDet& det,
                                     const MagneticField& mf) {
    AlgebraicSymMatrix55 m;
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j <= i; ++j)
        m(i, j) = st.err[i * (i + 1) / 2 + j];
    return TrajectoryStateOnSurface(
        LocalTrajectoryParameters(st.par[0], st.par[1], st.par[2], st.par[3], st.par[4], st.pzSign, true),
        LocalTrajectoryError(m),
        det.surface(),
        &mf);
  }

}  // namespace

class MkFitAlpakaOutputTrackConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitAlpakaOutputTrackConverter(edm::ParameterSet const& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<::mkfitdev::TrackSoAHostCollection> tracksToken_;
  edm::EDGetTokenT<::mkfitdev::MkFitStatusHostObject> statusToken_;
  std::string statusLabel_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelClusterIndexToHitToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> stripClusterIndexToHitToken_;
  const edm::EDGetTokenT<edm::View<TrajectorySeed>> seedToken_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerEventToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorAlongToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorOppositeToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> navToken_;
  const edm::EDPutTokenT<reco::TrackCollection> putTrackToken_;
  const edm::EDPutTokenT<TrackingRecHitCollection> putHitsToken_;
  const edm::EDPutTokenT<reco::TrackExtraCollection> putExtraToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> putSeedStopInfoToken_;

  const float qualityMaxInvPt_;
  const float qualityMinTheta_;
  const float qualityMaxRsq_;
  const float qualityMaxZ_;
  const float qualityMaxPosErrSq_;
  const bool qualitySignPt_;
  const int algo_;
};

MkFitAlpakaOutputTrackConverter::MkFitAlpakaOutputTrackConverter(edm::ParameterSet const& iConfig)
    : tracksToken_{consumes(iConfig.getParameter<edm::InputTag>("tracks"))},
      pixelClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitPixelHits"))},
      stripClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitStripHits"))},
      seedToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
      measurementTrackerEventToken_{consumes(iConfig.getParameter<edm::InputTag>("measurementTrackerEvent"))},
      bsToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      propagatorAlongToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))},
      propagatorOppositeToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))},
      mfToken_{esConsumes()},
      mkFitGeomToken_{esConsumes()},
      tTopoToken_{esConsumes()},
      navToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("NavigationSchool"))},
      putTrackToken_{produces()},
      putHitsToken_{produces()},
      putExtraToken_{produces()},
      putSeedStopInfoToken_{produces()},
      qualityMaxInvPt_{float(iConfig.getParameter<double>("qualityMaxInvPt"))},
      qualityMinTheta_{float(iConfig.getParameter<double>("qualityMinTheta"))},
      qualityMaxRsq_{float(std::pow(iConfig.getParameter<double>("qualityMaxR"), 2))},
      qualityMaxZ_{float(iConfig.getParameter<double>("qualityMaxZ"))},
      qualityMaxPosErrSq_{float(std::pow(iConfig.getParameter<double>("qualityMaxPosErr"), 2))},
      qualitySignPt_{iConfig.getParameter<bool>("qualitySignPt")},
      algo_{reco::TrackBase::algoByName(
          iConfig.getParameter<std::string>("algorithmName").empty()
              ? std::string(TString(iConfig.getParameter<edm::InputTag>("seeds").label()).ReplaceAll("Seeds", "").Data())
              : iConfig.getParameter<std::string>("algorithmName"))} {
  const auto status = iConfig.getParameter<edm::InputTag>("status");
  if (!status.label().empty()) {
    statusToken_ = consumes(status);
    statusLabel_ = status.encode();
  }
  if (iConfig.getParameter<bool>("trajectoryInEvent"))
    throw cms::Exception("Configuration") << "MkFitAlpakaOutputTrackConverter: trajectoryInEvent needs per-hit states, "
                                             "which the device fit does not export; use MkFitOutputTrackConverter";
}

void MkFitAlpakaOutputTrackConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("tracks", edm::InputTag{"hltInitialStepTrackCandidatesMkFitFitDevice"})
      ->setComment("mkfitdev TrackSoA of the device final fit (host copy)");
  desc.add("status", edm::InputTag())->setComment("MkFitStatusHostObject; empty = the TrackSoA overflow counters");
  desc.add("mkFitPixelHits", edm::InputTag{"mkFitSiPixelHits"});
  desc.add("mkFitStripHits", edm::InputTag{"mkFitSiStripHits"});
  desc.add("seeds", edm::InputTag{"initialStepSeeds"});
  desc.add<std::string>("algorithmName", "")->setComment("track algorithm; if empty, the seeds label without 'Seeds'");
  desc.add("beamSpot", edm::InputTag{"offlineBeamSpot"})->setComment("beam line of the PCA");
  desc.add("propagatorAlong", edm::ESInputTag{"", "PropagatorWithMaterial"});
  desc.add("propagatorOpposite", edm::ESInputTag{"", "PropagatorWithMaterialOpposite"});
  desc.add<double>("qualityMaxInvPt", 100)->setComment("max(1/pt) for converted tracks");
  desc.add<double>("qualityMinTheta", 0.01)->setComment("lower bound on theta (or pi-theta) for converted tracks");
  desc.add<double>("qualityMaxR", 120)->setComment("max(R) for the state position for converted tracks");
  desc.add<double>("qualityMaxZ", 280)->setComment("max(|Z|) for the state position for converted tracks");
  desc.add<double>("qualityMaxPosErr", 100)->setComment("max position error for converted tracks");
  desc.add<bool>("qualitySignPt", true)->setComment("check sign of 1/pt for converted tracks");
  desc.add<edm::ESInputTag>("NavigationSchool", edm::ESInputTag{"", "SimpleNavigationSchool"});
  desc.add<edm::InputTag>("measurementTrackerEvent", edm::InputTag("MeasurementTrackerEvent"));
  desc.add<bool>("trajectoryInEvent", false)->setComment("must be False (no per-hit states from the device fit)");
  descriptions.addWithDefaultLabel(desc);
}

void MkFitAlpakaOutputTrackConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  if (!statusLabel_.empty())
    ::mkfitdev::warnIfNotClean(iEvent.get(statusToken_).value(), statusLabel_);
  const auto& soa = iEvent.get(tracksToken_);
  const auto tracks = soa.const_view();
  if (statusLabel_.empty() && tracks.metadata().size() != 0 &&
      (tracks.nOverflowTracks() != 0 || tracks.nOverflowHits() != 0))
    edm::LogWarning("MkFitAlpakaOutputTrackConverter")
        << "event " << iEvent.id().event() << ": TrackSoA overflow, " << tracks.nOverflowTracks() << " tracks dropped, "
        << tracks.nOverflowHits() << " hit lists truncated";
  // a skipped or seedless event carries a zero-capacity TrackSoA; its scalars are not read
  const int nCand = tracks.metadata().size() == 0 ? 0 : tracks.nTracks();

  const auto hseeds = iEvent.getHandle(seedToken_);
  // The TrackSoA has one row per mkFit seed and MkFitSeedConverter makes one mkFit seed per TrajectorySeed (or none):
  // the track labels index the seed collection.
  if (tracks.metadata().size() != 0 && static_cast<std::size_t>(tracks.metadata().size()) != hseeds->size())
    throw cms::Exception("Configuration")
        << "MkFitAlpakaOutputTrackConverter: the TrackSoA has " << tracks.metadata().size() << " rows but seeds has "
        << hseeds->size() << " seeds; seeds must be the TrajectorySeed collection the mkFit seeds were made from";
  const auto& pixelHits = iEvent.get(pixelClusterIndexToHitToken_).hits();
  const auto& stripHits = iEvent.get(stripClusterIndexToHitToken_).hits();
  const auto& measTk = iEvent.get(measurementTrackerEventToken_);
  const auto& bs = iEvent.get(bsToken_);
  const auto& mf = iSetup.getData(mfToken_);
  const auto& propagatorAlong = iSetup.getData(propagatorAlongToken_);
  const auto& propagatorOpposite = iSetup.getData(propagatorOppositeToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);
  const auto& tTopo = iSetup.getData(tTopoToken_);
  const auto& navSchool = iSetup.getData(navToken_);
  const auto& detLayers = mkFitGeom.detLayers();
  const auto& trackerInfo = mkFitGeom.trackerInfo();
  if (mkFitGeom.isPhase1())
    throw cms::Exception("Configuration") << "MkFitAlpakaOutputTrackConverter supports the Phase-2 geometry only";

  reco::TrackCollection trks;
  trks.reserve(nCand);
  TrackingRecHitCollection outHits;
  reco::TrackExtraCollection extras;
  extras.reserve(nCand);
  const reco::TrackExtraRefProd refExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  const TrackingRecHitRefProd refHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

  // per-candidate scratch (fixed capacity = the TrackSoA hit-list capacity)
  std::array<const BaseTrackerRecHit*, ::mkfitdev::kMaxTrkHits> hitPtr;
  std::array<HitKey, ::mkfitdev::kMaxTrkHits> keys;
  std::array<int, ::mkfitdev::kMaxTrkHits> order;
  //use negative sigma=-3.0 in order to use a more conservative definition of isInside() for Bounds classes.
  const Chi2MeasurementEstimator estimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12);  // as MkFitOutputTrackConverter
  const TSCBLBuilderNoMaterial tscblBuilder;
  for (int icand = 0; icand < nCand; ++icand) {
    const auto row = tracks[icand];
    mkfit::TrackState state;
    for (int k = 0; k < 6; ++k)
      state.parameters[k] = row.params().v[k];
    std::memcpy(state.errors.Array(), row.errors().v, sizeof(::mkfitdev::TrackErrors::v));
    state.charge = row.charge();

    // state and chi2: basic quality first (MkFitOutputTrackConverter order; a fit with a non-finite or negative chi2
    // has failed)
    if (state.invpT() > qualityMaxInvPt_ || (qualitySignPt_ && state.invpT() < 0) || state.theta() < qualityMinTheta_ ||
        (M_PI - state.theta()) < qualityMinTheta_ || state.posRsq() > qualityMaxRsq_ ||
        std::abs(state.z()) > qualityMaxZ_ ||
        (state.errors.At(0, 0) + state.errors.At(1, 1) + state.errors.At(2, 2)) > qualityMaxPosErrSq_ ||
        edm::isNotFinite(row.chi2()) || row.chi2() < 0.f)
      continue;
    state.convertFromCCSToGlbCurvilinear();
    const auto& param = state.parameters;
    const auto& err = state.errors;
    AlgebraicSymMatrix55 cov;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        cov[i][j] = err.At(i, j);
    const FreeTrajectoryState fts(
        GlobalTrajectoryParameters(
            GlobalPoint(param[0], param[1], param[2]), GlobalVector(param[3], param[4], param[5]), state.charge, &mf),
        CurvilinearTrajectoryError(cov));
    if (!fts.curvilinearError().posDef())
      continue;
    //Sylvester's criterion, start from the smaller submatrix size (as MkFitOutputTrackConverter)
    double det = 0;
    const auto& cm = fts.curvilinearError().matrix();
    if ((!cm.Sub<AlgebraicSymMatrix22>(0, 0).Det(det)) || det < 0 || (!cm.Sub<AlgebraicSymMatrix33>(0, 0).Det(det)) ||
        det < 0 || (!cm.Sub<AlgebraicSymMatrix44>(0, 0).Det(det)) || det < 0 || (!cm.Det2(det)) || det < 0)
      continue;

    // on-track hits: referenced, keyed once
    const int nTot = row.nTotalHits();
    int nHits = 0;
    for (int i = 0; i < nTot; ++i) {
      const auto hot = row.hits().hot[i];
      if (hot.index < 0) {
        if (detLayers.at(hot.layer) == nullptr)
          throw cms::Exception("LogicError") << "DetLayer for layer index " << hot.layer << " is null!";
        continue;
      }
      const bool isPixel = trackerInfo.layer(hot.layer).is_pixel();
      const auto& thit = static_cast<BaseTrackerRecHit const&>(*(isPixel ? pixelHits : stripHits)[hot.index]);
      if (!isPixel && !thit.firstClusterRef().isPhase2())
        throw cms::Exception("LogicError") << "MkFitAlpakaOutputTrackConverter: non-Phase-2 outer-tracker hit";
      hitPtr[nHits] = &thit;
      keys[nHits].hot = i;
      order[nHits] = nHits;
      ++nHits;
    }
    // the fit-order keys, once per hit: R^2 and z of the mkFit hit position (surface().toGlobal(localPosition()), as
    // convertHits), z0 = z of the first hit (in HitOnTrack order) with the smallest R^2
    {
      float minR2 = std::numeric_limits<float>::max(), z0 = 0.f;
      for (int k = 0; k < nHits; ++k) {
        const GlobalPoint pos = hitPtr[k]->globalPosition();
        const float r2 = pos.x() * pos.x() + pos.y() * pos.y();
        keys[k].key = r2;
        if (r2 < minR2) {
          minR2 = r2;
          z0 = pos.z();
        }
      }
      for (int k = 0; k < nHits; ++k) {
        const float dz = hitPtr[k]->globalPosition().z() - z0;
        keys[k].key = keys[k].key + dz * dz;
      }
    }
    // hit order: (key, HitOnTrack position), a strict total order, by insertion (nHits <= kMaxTrkHits)
    for (int i = 1; i < nHits; ++i) {
      const int current = order[i];
      int j = i;
      while (j > 0 && hitLess(keys[current], keys[order[j - 1]])) {
        order[j] = order[j - 1];
        --j;
      }
      order[j] = current;
    }
    if (nHits == 0)
      continue;  // as MkFitOutputTrackConverter (recHits[0] of an empty OwnVector); cannot happen for a fitted track

    const GeomDet* detH0 = hitPtr[order[0]]->det();
    if (detH0 == nullptr)
      continue;
    const TrajectoryStateOnSurface tsosState(fts, detH0->surface());
    if (!tsosState.isValid())
      continue;
    const TrajectoryStateClosestToBeamLine tsAtPCA = tscblBuilder(*tsosState.freeState(), bs);
    if (!tsAtPCA.isValid())
      continue;
    const auto& stateAtPCA = tsAtPCA.trackStateAtPCA();
    const GlobalPoint v0 = stateAtPCA.position();
    const GlobalVector mom = stateAtPCA.momentum();
    int ndof = -5;
    for (int k = 0; k < nHits; ++k)
      ndof += hitPtr[order[k]]->dimension();
    reco::Track trk(row.chi2(),
                    ndof,
                    math::XYZPoint(v0.x(), v0.y(), v0.z()),
                    math::XYZVector(mom.x(), mom.y(), mom.z()),
                    fts.charge(),  // = TrajectoryStateClosestToBeamLine::trackStateAtPCA().charge()
                    stateAtPCA.curvilinearError().matrix(),
                    static_cast<reco::TrackBase::TrackAlgorithm>(algo_));

    for (int k = 0; k < nHits; ++k)
      trk.appendHitPattern(*hitPtr[order[k]], tTopo);

    // the state for the navigation outward: the final fit's state at its outermost hit when that is the outermost
    // rec hit, else the innermost one (MkFitOutputTrackConverter)
    const int hotFirst = keys[order[0]].hot, hotLast = keys[order[nHits - 1]].hot;
    int hotMin = hotFirst, hotMax = hotFirst;
    for (int k = 0; k < nHits; ++k) {
      hotMin = std::min(hotMin, keys[k].hot);
      hotMax = std::max(hotMax, keys[k].hot);
    }
    const auto& outerState = row.outer();
    const bool useOuter = outerState.hot == hotLast && outerState.valid;
    const TrajectoryStateOnSurface outerTsos =
        useOuter ? outerTSOS(outerState, *hitPtr[order[nHits - 1]]->det(), mf) : tsosState;

    //extra hits (taken from TrackProducerBase<T>::setSecondHitPattern), as MkFitOutputTrackConverter
    const auto* outerLayer = detLayers.at(mkFitGeom.mkFitLayerNumber(hitPtr[order[nHits - 1]]->geographicalId()));
    const auto* innerLayer = detLayers.at(mkFitGeom.mkFitLayerNumber(hitPtr[order[0]]->geographicalId()));

    // Lost hits between the innermost and the outermost hit: the hits the final fit removed as outliers, on their own
    // modules, after the valid hits (MkFitOutputTrackConverter)
    for (int pos = hotMin + 1; pos < hotMax; ++pos) {
      const int idx = row.hits().hot[pos].index;
      if (!::mkfitdev::isFitOutlierIdx(idx))
        continue;
      const int hitIdx = ::mkfitdev::hitIdxOfFitOutlier(idx);
      const bool isPixel = trackerInfo.layer(row.hits().hot[pos].layer).is_pixel();
      const GeomDet* lostDet = isPixel ? pixelHits[hitIdx]->det() : stripHits[hitIdx]->det();
      const InvalidTrackingRecHit lostHit(*lostDet, TrackingRecHit::missing);
      trk.appendHitPattern(lostHit, tTopo);
    }

    for (int side = 0; side < 2; ++side) {  // 0: inner, 1: outer
      const auto layers = side == 0 ? navSchool.compatibleLayers(*innerLayer, fts, oppositeToMomentum)
                                    : navSchool.compatibleLayers(*outerLayer, *outerTsos.freeState(), alongMomentum);
      for (auto it : layers) {
        if (it->basicComponents().empty())
          continue;
        auto const& detWithState = side == 0 ? it->compatibleDets(tsosState, propagatorOpposite, estimator)
                                             : it->compatibleDets(outerTsos, propagatorAlong, estimator);
        if (detWithState.empty())
          continue;
        const DetId id = detWithState.front().first->geographicalId();
        MeasurementDetWithData const& measDet = measTk.idToDet(id);
        const bool missing = measDet.isActive() && !measDet.hasBadComponents(detWithState.front().second);
        const InvalidTrackingRecHit tmpHit(
            *detWithState.front().first,
            side == 0 ? (missing ? TrackingRecHit::missing_inner : TrackingRecHit::inactive_inner)
                      : (missing ? TrackingRecHit::missing_outer : TrackingRecHit::inactive_outer));
        trk.appendHitPattern(tmpHit, tTopo);
      }
    }

    // output: one clone per hit (pixel: as is; OT: the 1D re-creation of MkFitOutputTrackConverter with the yy error
    // at float max)
    const auto hidx = outHits.size();
    for (int k = 0; k < nHits; ++k) {
      const auto& thit = *hitPtr[order[k]];
      if (thit.firstClusterRef().isPixel())
        outHits.push_back(thit.clone());
      else
        outHits.push_back(std::make_unique<Phase2TrackerRecHit1D>(
            thit.localPosition(),
            LocalError(thit.localPositionError().xx(), 0.f, std::numeric_limits<float>::max()),
            *thit.det(),
            thit.firstClusterRef().cluster_phase2OT()));
    }
    reco::TrackExtra extra;
    extra.setHits(refHits, hidx, trk.numberOfValidHits());
    extra.setSeedRef(edm::RefToBase<TrajectorySeed>(hseeds, row.label()));
    const AlgebraicVector5 zero(0, 0, 0, 0, 0);
    extra.setTrajParams(reco::TrackExtra::TrajParams(trk.numberOfValidHits(), LocalTrajectoryParameters(zero, 1.)),
                        reco::TrackExtra::Chi2sFive(trk.numberOfValidHits(), 0));
    extras.push_back(std::move(extra));
    trk.setExtra(reco::TrackExtraRef(refExtras, extras.size() - 1));
    trks.push_back(std::move(trk));
  }

  const auto nSeeds = hseeds->size();
  iEvent.emplace(putTrackToken_, std::move(trks));
  iEvent.emplace(putExtraToken_, std::move(extras));
  iEvent.emplace(putHitsToken_, std::move(outHits));
  // as MkFitOutputTrackConverter: SeedStopInfo unfilled, one per seed
  iEvent.emplace(putSeedStopInfoToken_, nSeeds);
}

DEFINE_FWK_MODULE(MkFitAlpakaOutputTrackConverter);
