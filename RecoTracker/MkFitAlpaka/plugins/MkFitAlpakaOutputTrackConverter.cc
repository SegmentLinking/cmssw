// ONE host module from the device fit's TrackSoA (host copy) to reco::Track +
// TrackExtra + TrackingRecHitCollection + SeedStopInfo, replacing the MkFitOutputTrackConverter of RecoTracker/MkFit
// in the device-fit menu.
// Same operations as MkFitOutputTrackConverter's no-hit-state branch (identical products), without its waste:
//   - reads the TrackSoA rows directly (no mkfit::TrackVec / MkFitOutputWrapper copy, no MkFitEventOfHits: is_pixel
//     per layer comes from the MkFitGeometry TrackerInfo, which is what LayerOfHits::is_pixel() returns);
//   - the on-track legacy rechits are referenced, not cloned, while the track is built; each is cloned ONCE, into the
//     output collection (MkFitOutputTrackConverter: three clones per hit: candidate OwnVector, hitsVecs copy, output
//     copy);
//   - the 3D-radius hit ordering uses keys precomputed once per hit (MkFitOutputTrackConverter's comparator
//     recomputes det()->subDetector(), globalPosition() and the TOB side per comparison); same comparator.
// Unchanged physics paths (CMSSW code, same calls in the same order): CCS -> global curvilinear conversion and the
// state quality / Sylvester checks, TSCBLBuilderNoMaterial to the beam line with the CMSSW MagneticField (or the
// device PCA, pcaStates), the hit pattern (appendHitPattern per hit) and the inner/outer missing-hit navigation
// (NavigationSchool + compatibleDets with PropagatorWithMaterial / Opposite, MeasurementTrackerEvent activity; with
// navPortable the portable search on the ES flat tables, host calls where it cannot decide).
// Supported: Phase-2 geometry, no per-hit states (the device fit has none: TrajectoryInEvent is refused).

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
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
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2TrackerRecHitOnDemand.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitAlpaka/interface/OutConvProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/math/PcaToBeamLine.h"
#include "RecoTracker/MkFitAlpaka/interface/StatusProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/StatusReport.h"
#include "RecoTracker/MkFitAlpaka/interface/TrackProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/navdev/NavFlatTables.h"
#include "RecoTracker/MkFitAlpaka/interface/NavResultProduct.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace {
  // the hit ordering of MkFitOutputTrackConverter (Phase-2 branch of its recHits.sort lambda), evaluated on keys
  // computed once per hit
  struct HitKey {
    GlobalPoint pos;
    bool barrel;             // GeomDetEnumerators::isBarrel(det()->subDetector()) == DetId subdet PXB / TIB / TOB
    bool tiltedOrNotBarrel;  // (subdetId == TOB && tobSide < 3): the "barrel tilted" clause of the comparator, per hit
  };
  inline bool hitLess(const HitKey& a, const HitKey& b) {
    const bool aB = a.barrel, bB = b.barrel;
    if (aB || bB) {
      if (a.tiltedOrNotBarrel || b.tiltedOrNotBarrel || !(aB && bB))
        return a.pos.mag2() < b.pos.mag2();
      return a.pos.perp2() < b.pos.perp2();
    }
    return std::abs(a.pos.z()) < std::abs(b.pos.z());
  }
}  // namespace

class MkFitAlpakaOutputTrackConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitAlpakaOutputTrackConverter(edm::ParameterSet const& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  void endJob() override;

  const edm::EDGetTokenT<::mkfitdev::TrackSoAHostCollection> tracksToken_;
  edm::EDGetTokenT<::mkfitdev::MkFitStatusHostObject> statusToken_;
  edm::EDGetTokenT<::mkfitdev::OutConvHostCollection> pcaToken_;  // device PCA (empty tag: host TSCBL)
  std::string statusLabel_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelClusterIndexToHitToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> stripClusterIndexToHitToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed>> seedToken_;  // seed hand-off (empty tag): null seedRef
  edm::EDGetTokenT<MkFitSeedWrapper> mkFitSeedsToken_;     // hand-off: the seed count for SeedStopInfo
  edm::EDGetTokenT<std::vector<int>> seedCountToken_;      // device hand-off: one entry per seed (pixelTrackOfSeed)
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
  // otClustersOnDemand set = OT hits made on demand (mkFitStripHits may then be a size-only map)
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otClustersToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> otGeomToken_;
  edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> otCpeToken_;
  const bool dropBadChi2_;  // DEVIATION DEV-4 switch
  // the missing-hit entries from the portable search (NavSearch.h) on the ES flat tables, from the fit-row
  // start state; any call it cannot decide exactly goes to DetLayer::compatibleDets (host fallback, counted)
  const bool navPortable_;
  edm::ESGetToken<mkfitdev::navdev::NavFlatTables, TrackerRecoGeometryRecord> navFlatToken_;
  // navDevice: the same search done on the device (MkFitAlpakaNavDeviceProducer); calls it did not compute are searched
  // here as in navPortable
  edm::EDGetTokenT<::mkfitdev::NavResultHostCollection> navResultToken_;
  // host fallbacks, summed over the job: [0] inner [1] outer
  mutable std::array<std::atomic<long long>, 2> npDev_{}, nlDev_{}, nlSchool_{};
  mutable std::atomic<long long> npDevMiss_{0};
  mutable std::array<std::atomic<long long>, 2> npCalls_{}, npFound_{}, npInactive_{};
  mutable std::atomic<long long> npTracks_{0}, npHostTrack_{0}, npHostLayer_{0}, npHostSearch_{0}, npHostBad_{0},
      npHostMarginal_{0};
};

MkFitAlpakaOutputTrackConverter::MkFitAlpakaOutputTrackConverter(edm::ParameterSet const& iConfig)
    : tracksToken_{consumes(iConfig.getParameter<edm::InputTag>("tracks"))},
      pixelClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitPixelHits"))},
      stripClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitStripHits"))},
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
      // DEVIATION DEV-6: an explicit track algorithm; MkFitOutputTrackConverter derives it from the seeds label, and
      // the HLT label hltInitialStepTrajectorySeedsLST maps to undefAlgorithm, which PFAlgo treats as a 1e9-error track
      algo_{reco::TrackBase::algoByName(
          iConfig.getParameter<std::string>("algorithm").empty()
              ? std::string(TString(iConfig.getParameter<edm::InputTag>("seeds").label()).ReplaceAll("Seeds", "").Data())
              : iConfig.getParameter<std::string>("algorithm"))},
      dropBadChi2_{iConfig.getParameter<bool>("dropNegativeChi2")},
      navPortable_{iConfig.getParameter<bool>("navPortable")} {
  if (navPortable_)
    navFlatToken_ = esConsumes();
  if (auto const nr = iConfig.getParameter<edm::InputTag>("navDevice"); !nr.label().empty()) {
    if (!navPortable_)
      throw cms::Exception("Configuration") << "MkFitAlpakaOutputTrackConverter: navDevice needs navPortable";
    navResultToken_ = consumes(nr);
  }
  if (auto const& seeds = iConfig.getParameter<edm::InputTag>("seeds"); !seeds.label().empty())
    seedToken_ = consumes(seeds);
  else if (auto const& mk = iConfig.getParameter<edm::InputTag>("mkFitSeeds"); !mk.label().empty())
    mkFitSeedsToken_ = consumes(mk);
  else if (auto const& sc = iConfig.getParameter<edm::InputTag>("seedCount"); !sc.label().empty())
    seedCountToken_ = consumes(sc);
  else
    throw cms::Exception("Configuration")
        << "MkFitAlpakaOutputTrackConverter: seeds, mkFitSeeds and seedCount all empty";
  if (const auto pca = iConfig.getParameter<edm::InputTag>("pcaStates"); !pca.label().empty())
    pcaToken_ = consumes(pca);
  if (auto const ot = iConfig.getParameter<edm::InputTag>("otClustersOnDemand"); !ot.label().empty()) {
    otClustersToken_ = consumes(ot);
    otGeomToken_ = esConsumes();
    otCpeToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("Phase2StripCPE"));
  }
  const auto status = iConfig.getParameter<edm::InputTag>("status");
  if (!status.label().empty()) {
    statusToken_ = consumes(status);
    statusLabel_ = status.encode();
  }
  if (iConfig.getParameter<bool>("TrajectoryInEvent"))
    throw cms::Exception("Configuration") << "MkFitAlpakaOutputTrackConverter: TrajectoryInEvent needs per-hit states, "
                                             "which the device fit does not export; use MkFitOutputTrackConverter";
}

void MkFitAlpakaOutputTrackConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("tracks", edm::InputTag{"hltInitialStepTrackCandidatesMkFitFitDevice"})
      ->setComment("mkfitdev TrackSoA of the device final fit (host copy)");
  desc.add("status", edm::InputTag())->setComment("MkFitStatusHostObject; empty = the TrackSoA overflow counters");
  desc.add("mkFitPixelHits", edm::InputTag{"mkFitSiPixelHits"});
  desc.add("mkFitStripHits", edm::InputTag{"mkFitSiStripHits"});
  desc.add<edm::InputTag>("otClustersOnDemand", edm::InputTag(""))
      ->setComment(
          "Phase-2 OT hits of output tracks made on demand from these clusters with Phase2StripCPE "
          "(= the legacy rechits); empty = read them through mkFitStripHits");
  desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"));
  desc.add("seeds", edm::InputTag{"initialStepSeeds"})
      ->setComment("empty (seed hand-off): TrackExtra seedRef null, SeedStopInfo sized by mkFitSeeds");
  desc.add<edm::InputTag>("mkFitSeeds", edm::InputTag(""))
      ->setComment("with seeds empty: the mkFit seeds of the build");
  desc.add<edm::InputTag>("seedCount", edm::InputTag(""))
      ->setComment("K1, with seeds and mkFitSeeds empty: a per-seed vector (pixelTrackOfSeed) giving the seed count");
  desc.add<std::string>("algorithm", "")
      ->setComment(
          "DEVIATION DEV-6 (switch, MkFitOutputTrackConverter = empty): track algorithm name; empty = the "
          "MkFitOutputTrackConverter rule (seeds label "
          "without 'Seeds', undefAlgorithm for the HLT label)");
  desc.add("beamSpot", edm::InputTag{"offlineBeamSpot"})
      ->setComment("beam line of the PCA; offlineBeamSpot as MkFitOutputTrackConverter");
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
  desc.add<bool>("dropNegativeChi2", false)
      ->setComment(
          "DEVIATION DEV-4 (switch, MkFitOutputTrackConverter = false): drop candidates whose fit chi2 is negative or "
          "not finite, as "
          "the KF final fit rejects trajectories with a non-positive-definite covariance");
  desc.add("pcaStates", edm::InputTag())
      ->setComment(
          "the device PCA (MkFitAlpakaOutConvStateProducer, OutConvHostCollection) used instead of "
          "TSCBLBuilderNoMaterial; empty = the host TSCBL");
  desc.add<bool>("navPortable", false)
      ->setComment(
          "missing-hit entries from the portable compatibleDets search (NavSearch.h) on the ES flat tables "
          "(MkFitAlpakaNavFlatTablesESProducer); undecidable calls go to DetLayer::compatibleDets");
  desc.add<edm::InputTag>("navDevice", edm::InputTag())
      ->setComment("navPortable: the device search results (MkFitAlpakaNavDeviceProducer); empty = search on the host");
  desc.add<bool>("TrajectoryInEvent", false)->setComment("must be False (no per-hit states from the device fit)");
  descriptions.addWithDefaultLabel(desc);
}

void MkFitAlpakaOutputTrackConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  if (!statusLabel_.empty())
    ::mkfitdev::warnIfNotClean(iEvent.get(statusToken_).value(), statusLabel_);
  const auto& soa = iEvent.get(tracksToken_);
  const auto v = soa.const_view();
  if (statusLabel_.empty() && v.metadata().size() != 0 && (v.nOverflowTracks() != 0 || v.nOverflowHits() != 0))
    edm::LogWarning("MkFitAlpakaOutputTrackConverter")
        << "event " << iEvent.id().event() << ": TrackSoA overflow, " << v.nOverflowTracks() << " tracks dropped, "
        << v.nOverflowHits() << " hit lists truncated";
  // a skipped or seedless event carries a zero-capacity TrackSoA; its scalars are not read
  const int nCand = v.metadata().size() == 0 ? 0 : v.nTracks();
  const ::mkfitdev::OutConvHostCollection* pcaStates = pcaToken_.isUninitialized() ? nullptr : &iEvent.get(pcaToken_);
  if (pcaStates && pcaStates->const_view().metadata().size() < nCand)
    throw cms::Exception("LogicError") << "MkFitAlpakaOutputTrackConverter: PCA rows "
                                       << pcaStates->const_view().metadata().size() << " < tracks " << nCand;

  edm::Handle<edm::View<TrajectorySeed>> hseeds;
  if (!seedToken_.isUninitialized())
    iEvent.getByToken(seedToken_, hseeds);
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

  // on-demand OT hits (per-candidate store, reserved: stable addresses)
  std::optional<Phase2TrackerRecHitOnDemand> otOnDemand;
  if (!otClustersToken_.isUninitialized())
    otOnDemand.emplace(iEvent.getHandle(otClustersToken_), iSetup.getData(otGeomToken_), iSetup.getData(otCpeToken_));
  std::vector<Phase2TrackerRecHit1D> otStore;
  otStore.reserve(::mkfitdev::kMaxTrkHits);
  // per-candidate scratch (fixed capacity = the TrackSoA hit-list capacity)
  std::array<const BaseTrackerRecHit*, ::mkfitdev::kMaxTrkHits> hitPtr;
  std::array<HitKey, ::mkfitdev::kMaxTrkHits> keys;
  std::array<int, ::mkfitdev::kMaxTrkHits> order;
  //use negative sigma=-3.0 in order to use a more conservative definition of isInside() for Bounds classes.
  const Chi2MeasurementEstimator estimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12);  // as MkFitOutputTrackConverter
  const TSCBLBuilderNoMaterial tscblBuilder;
  // navPortable: the portable search on the ES flat tables from the fit-row start state. Returns 1 (entry
  // appended), 0 (no compatible det: no entry, as compatibleDets) or -1 (not decided here: the caller runs
  // compatibleDets). Exact by construction: the search is the TkDetLayers one per call, the entry type is decided only
  // where it needs no state on the det (inactive det; active det whose hasBadComponents is false for any state: probed
  // with an unbounded local error), anything else is a host call.
  namespace nd = ::mkfitdev::navdev;
  const nd::NavFlatTables* navFlat = navPortable_ ? &iSetup.getData(navFlatToken_) : nullptr;
  const nd::NavTables navTables = navFlat ? navFlat->tables() : nd::NavTables{};
  std::array<nd::NavGroups, nd::kNavArena> navArena;
  nd::DetTestStart pStart;
  std::array<long long, 2> npCalls{}, npFound{}, npInactive{};
  long long npTracks = 0, npHostTrack = 0, npHostLayer = 0, npHostSearch = 0, npHostBad = 0, npHostMarginal = 0;
  const ::mkfitdev::NavResultHostCollection* navRes =
      navResultToken_.isUninitialized() ? nullptr : &iEvent.get(navResultToken_);
  if (navRes && navRes->const_view().metadata().size() == 0)
    navRes = nullptr;  // the CPU backend's empty product (or no tracks): search on the host
  const int navNL = navFlat ? int(navFlat->allLayers.size()) : 0;
  // rows: capacity * 2 * navNL per (track, direction, layer), then 2 * capacity set headers (NavResultSoA.h)
  const int navCapacity = navRes ? navRes->const_view().metadata().size() / (2 * (navNL + 1)) : 0;
  if (navRes && (navRes->const_view().metadata().size() != navCapacity * 2 * (navNL + 1) || navCapacity < nCand))
    throw cms::Exception("LogicError") << "MkFitAlpakaOutputTrackConverter: navigation rows "
                                       << navRes->const_view().metadata().size() << " for " << nCand << " tracks and "
                                       << navNL << " layers";
  const int navHeader = navCapacity * 2 * navNL;
  std::array<long long, 2> nlDev{}, nlSchool{};
  // the compatibleLayers list of (track row, direction d) from start: the device set in the navigation school's
  // order (DetLayer pointer order, as its std::set) where exact, else the school
  auto compatibleLayers =
      [&](int row, int d, const DetLayer* start, FreeTrajectoryState const& fts, std::vector<const DetLayer*>& out) {
        const PropagationDirection dir = d == 0 ? oppositeToMomentum : alongMomentum;
        out.clear();
        if (navRes) {
          const int32_t h = navRes->const_view()[navHeader + 2 * row + d].det();
          if (h >= 0) {
            const auto ki = navFlat->layerIndex.find(start);
            if (ki != navFlat->layerIndex.end() && ki->second == h) {
              const int base = (2 * row + d) * navNL;
              for (const int l : navFlat->pointerOrder)
                if (navRes->const_view()[base + l].det() != ::mkfitdev::kNavResultNotComputed)
                  out.push_back(navFlat->allLayers[l]);
              ++nlDev[d];
              return;
            }
          }
        }
        out = navSchool.compatibleLayers(*start, fts, dir);
        ++nlSchool[d];
      };
  std::array<std::vector<const DetLayer*>, 2> compLayers;
  compLayers[0].reserve(navNL);
  compLayers[1].reserve(navNL);
  std::array<long long, 2> npDev{};
  long long npDevMiss = 0;
  auto portableEntry = [&](int row, int d, const DetLayer* layer, reco::Track& track) -> int {
    int front = ::mkfitdev::kNavResultNotComputed;  // flat det index of front(), or kNavResultEmpty
    if (navRes) {
      const auto ki = navFlat->layerIndex.find(layer);
      if (ki != navFlat->layerIndex.end()) {
        const int32_t r = navRes->const_view()[(2 * row + d) * navNL + ki->second].det();
        if (r >= ::mkfitdev::kNavResultEmpty) {
          front = r;
          ++npDev[d];
        } else
          ++npDevMiss;
      } else
        ++npDevMiss;
    }
    if (front == ::mkfitdev::kNavResultNotComputed) {
      const auto li = navFlat->layers.find(layer);
      if (li == navFlat->layers.end() || li->second[1] < 0) {
        ++npHostLayer;
        return -1;
      }
      nd::NavCtx ctx{navTables,
                     &pStart,
                     d == 1,
                     estimator.maxSagitta(),
                     estimator.minTolerance2(),
                     estimator.nSigmaCut(),
                     0.5};  // the estimator's maximal displacement
      ctx.arena = navArena.data();
      nd::NavGroups res;
      auto const& lt = li->second;
      const bool ok = lt[0] == 2   ? nd::barrelLayerSearch(lt[1], ctx, res)
                      : lt[0] == 1 ? nd::pixEndcapSearch(lt[1], lt[2], ctx, res)
                                   : nd::endcapLayerSearch(lt[1], lt[2], ctx, res);
      if (!ok || (ctx.overflow & ~nd::kNavMarginal) != 0) {
        ++npHostSearch;
        return -1;
      }
      if (ctx.overflow != 0) {
        ++npHostMarginal;
        return -1;
      }
      front = res.n == 0 ? ::mkfitdev::kNavResultEmpty : res.g[0].first;
    }
    if (front == ::mkfitdev::kNavResultEmpty) {
      ++npCalls[d];
      return 0;
    }
    const GeomDet* det = navFlat->detPtr[front];
    MeasurementDetWithData const& md = measTk.idToDet(det->geographicalId());
    const bool active = md.isActive();
    if (active) {
      const TrajectoryStateOnSurface probe(LocalTrajectoryParameters(LocalPoint(0, 0, 0), LocalVector(0, 0, 1), 1),
                                           LocalTrajectoryError(1e6f, 1e6f, 1.f, 1.f, 1.f),
                                           det->surface(),
                                           &mf);
      if (md.hasBadComponents(probe)) {
        ++npHostBad;
        return -1;
      }
    }
    ++npCalls[d];
    ++npFound[d];
    npInactive[d] += !active;
    const InvalidTrackingRecHit tmpHit(
        *det,
        active ? (d == 0 ? TrackingRecHit::missing_inner : TrackingRecHit::missing_outer)
               : (d == 0 ? TrackingRecHit::inactive_inner : TrackingRecHit::inactive_outer));
    track.appendHitPattern(tmpHit, tTopo);
    return 1;
  };

  for (int c = 0; c < nCand; ++c) {
    const auto row = v[c];
    mkfit::TrackState state;
    for (int k = 0; k < 6; ++k)
      state.parameters[k] = row.params().v[k];
    std::memcpy(state.errors.Array(), row.errors().v, sizeof(float) * 21);
    state.charge = row.charge();

    // state: basic quality first (MkFitOutputTrackConverter order)
    if (state.invpT() > qualityMaxInvPt_ || (qualitySignPt_ && state.invpT() < 0) || state.theta() < qualityMinTheta_ ||
        (M_PI - state.theta()) < qualityMinTheta_ || state.posRsq() > qualityMaxRsq_ ||
        std::abs(state.z()) > qualityMaxZ_ ||
        (state.errors.At(0, 0) + state.errors.At(1, 1) + state.errors.At(2, 2)) > qualityMaxPosErrSq_)
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
    // DEVIATION DEV-4: negative / non-finite fit chi2 = non-positive-definite covariance in the fit (bit test: -Ofast safe)
    if (dropBadChi2_ && (edm::isNotFinite(row.chi2()) || row.chi2() < 0.f))
      continue;
    //Sylvester's criterion, start from the smaller submatrix size (as MkFitOutputTrackConverter)
    double det = 0;
    const auto& cm = fts.curvilinearError().matrix();
    if ((!cm.Sub<AlgebraicSymMatrix22>(0, 0).Det(det)) || det < 0 || (!cm.Sub<AlgebraicSymMatrix33>(0, 0).Det(det)) ||
        det < 0 || (!cm.Sub<AlgebraicSymMatrix44>(0, 0).Det(det)) || det < 0 || (!cm.Det2(det)) || det < 0)
      continue;

    // on-track hits: referenced, keyed once
    const int nTot = row.nTotalHits();
    int n = 0;
    otStore.clear();
    for (int i = 0; i < nTot; ++i) {
      const auto hot = row.hits().hot[i];
      if (hot.index < 0) {
        if (detLayers.at(hot.layer) == nullptr)
          throw cms::Exception("LogicError") << "DetLayer for layer index " << hot.layer << " is null!";
        continue;
      }
      const bool isPixel = trackerInfo.layer(hot.layer).is_pixel();
      const auto& hits = isPixel ? pixelHits : stripHits;
      if (!isPixel && otOnDemand)
        otStore.push_back(otOnDemand->make(hot.index));
      const auto& thit = (!isPixel && otOnDemand) ? static_cast<BaseTrackerRecHit const&>(otStore.back())
                                                  : static_cast<BaseTrackerRecHit const&>(*hits[hot.index]);
      if (!isPixel && !thit.firstClusterRef().isPhase2())
        throw cms::Exception("LogicError") << "MkFitAlpakaOutputTrackConverter: non-Phase-2 outer-tracker hit";
      hitPtr[n] = &thit;
      order[n] = n;
      ++n;
    }
    // the comparator's keys, once per hit
    for (int k = 0; k < n; ++k) {
      const auto& thit = *hitPtr[k];
      const auto id = thit.geographicalId();
      keys[k].pos = thit.globalPosition();
      // the comparator only asks isBarrel(subDetector()): from the DetId, no GeomDet dereference
      keys[k].barrel = id.subdetId() == PixelSubdetector::PixelBarrel || id.subdetId() == StripSubdetector::TIB ||
                       id.subdetId() == StripSubdetector::TOB;
      keys[k].tiltedOrNotBarrel = (id.subdetId() == StripSubdetector::TOB && tTopo.tobSide(id) < 3);
    }
    // hit order: MkFitOutputTrackConverter's comparator on the precomputed keys, applied by a stable insertion
    // (n <= kMaxTrkHits, almost sorted input). MkFitOutputTrackConverter's OwnVector::sort is std::sort (not stable):
    // where hits are equivalent or the comparator is not a strict weak order (mixed mag2 / perp2 pairs), the two
    // orders can differ - only among such hits (same pattern words, so the hit pattern, parameters and everything else
    // are unchanged; see doc/SIMPLIFICATIONS.txt).
    for (int i = 1; i < n; ++i) {
      const int x = order[i];
      int j = i;
      while (j > 0 && hitLess(keys[x], keys[order[j - 1]])) {
        order[j] = order[j - 1];
        --j;
      }
      order[j] = x;
    }
    if (n == 0)
      continue;  // as MkFitOutputTrackConverter (recHits[0] of an empty OwnVector); cannot happen for a fitted track

    const GeomDet* detH0 = hitPtr[order[0]]->det();
    if (detH0 == nullptr)
      continue;
    const TrajectoryStateOnSurface tsosState(fts, detH0->surface());
    if (!tsosState.isValid())
      continue;
    // the PCA: the device state (pcaStates; status ok) or TSCBLBuilderNoMaterial on the host (no pcaStates, or the
    // device's host-fallback / failure rows: the host decides validity there, as MkFitOutputTrackConverter)
    const int8_t devStatus =
        pcaStates ? pcaStates->const_view()[c].pcaStatus() : int8_t(::mkfitdev::pca::kPcaHostFallback);
    const bool useDevice = devStatus == ::mkfitdev::pca::kPcaOk;
    TrajectoryStateClosestToBeamLine tsAtPCA;
    if (!useDevice) {
      tsAtPCA = tscblBuilder(*tsosState.freeState(), bs);
      if (!tsAtPCA.isValid())
        continue;
    }
    GlobalPoint v0;
    GlobalVector p;
    AlgebraicSymMatrix55 pcaCov;
    if (useDevice) {
      const ::mkfitdev::PcaState ds = pcaStates->const_view()[c].pcaState();
      const ::mkfitdev::PcaCov dc = pcaStates->const_view()[c].pcaCov();
      v0 = GlobalPoint(ds.v[0], ds.v[1], ds.v[2]);
      p = GlobalVector(ds.v[3], ds.v[4], ds.v[5]);
      for (int i = 0, k = 0; i < 5; ++i)
        for (int j = 0; j <= i; ++j)
          pcaCov(i, j) = dc.v[k++];  // float, as reco::TrackBase stores it
    } else {
      const auto& stateAtPCA = tsAtPCA.trackStateAtPCA();
      v0 = stateAtPCA.position();
      p = stateAtPCA.momentum();
      pcaCov = stateAtPCA.curvilinearError().matrix();
    }
    int ndof = -5;
    for (int k = 0; k < n; ++k)
      ndof += hitPtr[order[k]]->dimension();  // the cloned OT hit is the same class (Phase2TrackerRecHit1D)
    reco::Track trk(row.chi2(),
                    ndof,
                    math::XYZPoint(v0.x(), v0.y(), v0.z()),
                    math::XYZVector(p.x(), p.y(), p.z()),
                    fts.charge(),  // = TrajectoryStateClosestToBeamLine::trackStateAtPCA().charge()
                    pcaCov,
                    static_cast<reco::TrackBase::TrackAlgorithm>(algo_));

    for (int k = 0; k < n; ++k)
      trk.appendHitPattern(*hitPtr[order[k]], tTopo);

    //extra hits (taken from TrackProducerBase<T>::setSecondHitPattern), as MkFitOutputTrackConverter
    const auto* outerLayer = detLayers.at(mkFitGeom.mkFitLayerNumber(hitPtr[order[n - 1]]->geographicalId()));
    const auto* innerLayer = detLayers.at(mkFitGeom.mkFitLayerNumber(hitPtr[order[0]]->geographicalId()));
    bool pTrack = false;
    if (navFlat) {
      ++npTracks;
      pTrack = ::mkfitdev::navdev::startFromFitRow(row.params().v, row.errors().v, row.charge(), pStart);
      npHostTrack += !pTrack;
    }
    for (int d = 0; d < 2; ++d) {
      auto& layers = compLayers[d];
      compatibleLayers(c, d, d == 0 ? innerLayer : outerLayer, fts, layers);
      for (auto it : layers) {
        if (it->basicComponents().empty())
          continue;
        if (pTrack && portableEntry(c, d, it, trk) >= 0)
          continue;
        auto const& detWithState = d == 0 ? it->compatibleDets(tsosState, propagatorOpposite, estimator)
                                          : it->compatibleDets(tsosState, propagatorAlong, estimator);
        if (detWithState.empty())
          continue;
        const DetId id = detWithState.front().first->geographicalId();
        MeasurementDetWithData const& measDet = measTk.idToDet(id);
        const bool missing = measDet.isActive() && !measDet.hasBadComponents(detWithState.front().second);
        const InvalidTrackingRecHit tmpHit(
            *detWithState.front().first,
            d == 0 ? (missing ? TrackingRecHit::missing_inner : TrackingRecHit::inactive_inner)
                   : (missing ? TrackingRecHit::missing_outer : TrackingRecHit::inactive_outer));
        trk.appendHitPattern(tmpHit, tTopo);
      }
    }

    // output: one clone per hit (pixel: as is; OT: the 1D re-creation of MkFitOutputTrackConverter with the yy error
    // at float max)
    const auto hidx = outHits.size();
    for (int k = 0; k < n; ++k) {
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
    if (hseeds.isValid())
      extra.setSeedRef(edm::RefToBase<TrajectorySeed>(hseeds, row.label()));
    const AlgebraicVector5 zero(0, 0, 0, 0, 0);
    extra.setTrajParams(reco::TrackExtra::TrajParams(trk.numberOfValidHits(), LocalTrajectoryParameters(zero, 1.)),
                        reco::TrackExtra::Chi2sFive(trk.numberOfValidHits(), 0));
    extras.push_back(std::move(extra));
    trk.setExtra(reco::TrackExtraRef(refExtras, extras.size() - 1));
    trks.push_back(std::move(trk));
  }

  const auto nSeeds = hseeds.isValid()                      ? hseeds->size()
                      : !mkFitSeedsToken_.isUninitialized() ? iEvent.get(mkFitSeedsToken_).seeds().size()
                                                            : iEvent.get(seedCountToken_).size();
  if (navFlat) {
    for (int d = 0; d < 2; ++d) {
      npCalls_[d] += npCalls[d];
      npFound_[d] += npFound[d];
      npInactive_[d] += npInactive[d];
      npDev_[d] += npDev[d];
      nlDev_[d] += nlDev[d];
      nlSchool_[d] += nlSchool[d];
    }
    npTracks_ += npTracks;
    npHostTrack_ += npHostTrack;
    npHostLayer_ += npHostLayer;
    npHostSearch_ += npHostSearch;
    npHostBad_ += npHostBad;
    npHostMarginal_ += npHostMarginal;
    npDevMiss_ += npDevMiss;
    if (npHostSearch > 0)
      edm::LogWarning("MkFitAlpakaOutputTrackConverter")
          << "event " << iEvent.id().event() << ": " << npHostSearch
          << " portable navigation searches overflowed or hit an unsupported layout (searched on the host)";
  }
  iEvent.emplace(putTrackToken_, std::move(trks));
  iEvent.emplace(putExtraToken_, std::move(extras));
  iEvent.emplace(putHitsToken_, std::move(outHits));
  // as MkFitOutputTrackConverter: SeedStopInfo unfilled, one per seed
  iEvent.emplace(putSeedStopInfoToken_, nSeeds);
}

void MkFitAlpakaOutputTrackConverter::endJob() {
  if (navPortable_)
    edm::LogInfo("MkFitAlpakaOutputTrackConverter")
        << "portable navigation: tracks " << npTracks_ << " (out of the field box, host: " << npHostTrack_
        << ") | portable calls inner " << npCalls_[0] << " (entries " << npFound_[0] << ", inactive " << npInactive_[0]
        << "), outer " << npCalls_[1] << " (entries " << npFound_[1] << ", inactive " << npInactive_[1]
        << ") | host calls: unsupported layer " << npHostLayer_ << ", search overflow / unsupported " << npHostSearch_
        << ", front det with possible bad components " << npHostBad_ << ", a det test within rounding of its bounds "
        << npHostMarginal_ << " | device results used inner " << npDev_[0] << " outer " << npDev_[1]
        << ", not computed on the device " << npDevMiss_ << " | compatibleLayers lists from the device set: inner "
        << nlDev_[0] << " outer " << nlDev_[1] << ", from the navigation school: inner " << nlSchool_[0] << " outer "
        << nlSchool_[1];
}

DEFINE_FWK_MODULE(MkFitAlpakaOutputTrackConverter);
