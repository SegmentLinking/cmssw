#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "storeTracks.h"
// OT hits of the tracks made on demand from the clusters
#include <deque>
#include <optional>
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2TrackerRecHitOnDemand.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

/**
 * This class creates "legacy" reco::Track
 * objects from the output of SoA CA.
 */

// #define GPU_DEBUG
// struct that holds two maps for detIds of the OT modules
struct DetIdMaps {
  DetIdMaps() : detIdToOTModuleId_(), detIdIsUsedOTModule_() {}

  // map from the detId of OT modules to the moduleId among the used OT modules
  // (starting from 0 for first module of first OT layer)
  std::map<uint32_t, uint32_t> detIdToOTModuleId_;
  // map from detId to bool if used as OT extension
  std::map<uint32_t, bool> detIdIsUsedOTModule_;
  // the entries of detIdToOTModuleId_ as a flat vector sorted by
  // detId, for the per-event detset loops (a binary search in a small contiguous array instead of two std::map finds
  // per detset; same answer for every detUnit)
  std::vector<std::pair<uint32_t, uint32_t>> usedOTModules_;
  // otModuleId of a used OT module, -1 otherwise (= detIdIsUsedOTModule_ ? detIdToOTModuleId_ : -1)
  int64_t otModuleId(uint32_t detId) const {
    auto it = std::lower_bound(
        usedOTModules_.begin(), usedOTModules_.end(), detId, [](auto const &p, uint32_t d) { return p.first < d; });
    return (it != usedOTModules_.end() && it->first == detId) ? int64_t(it->second) : int64_t(-1);
  }
};

class PixelTrackProducerFromSoAAlpaka : public edm::global::EDProducer<edm::RunCache<DetIdMaps>> {
  using TrackSoAHost = reco::TracksHost;
  using HMSstorage = std::vector<uint32_t>;
  using IndToEdm = std::vector<uint32_t>;
  using TrackHitSoA = reco::TrackHitSoA;

public:
  explicit PixelTrackProducerFromSoAAlpaka(const edm::ParameterSet &iConfig);
  ~PixelTrackProducerFromSoAAlpaka() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  std::shared_ptr<DetIdMaps> globalBeginRun(edm::Run const &, edm::EventSetup const &) const override;
  void globalEndRun(edm::Run const &, edm::EventSetup const &) const override {};

private:
  void produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  // Event Data tokens
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::EDGetTokenT<TrackSoAHost> trackSoAToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollectionNew> pixelRecHitsToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitsToken_;
  const edm::EDGetTokenT<HMSstorage> pixelHMSToken_;
  edm::EDGetTokenT<HMSstorage> otHMSToken_;
  // otClustersOnDemand set = no legacy OT rechits read
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otClustersToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> otGeomToken_;
  edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> otCpeToken_;
  bool otOnDemand_ = false;
  // Event Setup tokens
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryTokenRun_;

  int32_t const minNumberOfHits_;
  pixelTrack::Quality const minQuality_;
  const bool useOTExtension_;
  const bool requireQuadsFromConsecutiveLayers_;
};

PixelTrackProducerFromSoAAlpaka::PixelTrackProducerFromSoAAlpaka(const edm::ParameterSet &iConfig)
    : beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      trackSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("trackSrc"))),
      pixelRecHitsToken_(
          consumes<SiPixelRecHitCollectionNew>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      pixelHMSToken_(consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      idealMagneticFieldToken_(esConsumes()),
      trackerTopologyToken_(esConsumes()),
      trackerGeometryTokenRun_(esConsumes<edm::Transition::BeginRun>()),
      minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
      minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
      useOTExtension_(iConfig.getParameter<bool>("useOTExtension")),
      requireQuadsFromConsecutiveLayers_(iConfig.getParameter<bool>("requireQuadsFromConsecutiveLayers")) {
  if (minQuality_ == pixelTrack::Quality::notQuality) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
  }
  if (minQuality_ < pixelTrack::Quality::dup) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " not supported";
  }
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  // TrackCollection refers to TrackingRechit and TrackExtra
  // collections, need to declare its production after them to work
  // around a rare race condition in framework scheduling
  produces<reco::TrackCollection>();
  produces<IndToEdm>();

  // if useOTExtension consume the OT RecHits
  if (useOTExtension_) {
    otOnDemand_ = !iConfig.getParameter<edm::InputTag>("otClustersOnDemand").label().empty();
    if (otOnDemand_) {
      otClustersToken_ = consumes(iConfig.getParameter<edm::InputTag>("otClustersOnDemand"));
      otGeomToken_ = esConsumes();
      otCpeToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("Phase2StripCPE"));
    } else
      otRecHitsToken_ =
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("outerTrackerRecHitSrc"));
    otHMSToken_ = consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("outerTrackerRecHitSoAConverterSrc"));
  }
}

std::shared_ptr<DetIdMaps> PixelTrackProducerFromSoAAlpaka::globalBeginRun(const edm::Run &iRun,
                                                                           const edm::EventSetup &iSetup) const {
  // make the maps object
  auto detIdMaps = std::make_shared<DetIdMaps>();

  // if OT RecHits are used in PixelTracks, fill the detIdToOTModuleId_ map
  if (useOTExtension_) {
    // get track geometry
    const auto &trackerGeometry = &iSetup.getData(trackerGeometryTokenRun_);

    // function to check if given module is used as OT for CA
    auto isPinPSinOTBarrel = [&](DetId detId) {
      // Select only P-hits from the OT barrel
      return (trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
              detId.subdetId() == StripSubdetector::TOB);
    };

    // loop over all modules and fill the map detIdToOTModuleId_
    auto const &detUnits = trackerGeometry->detUnits();
    for (uint32_t otModuleId{0}; auto &detUnit : detUnits) {
      DetId detId(detUnit->geographicalId());
      // check if the module is used for OT extension
      bool isUsedOTModule = isPinPSinOTBarrel(detId);
      detIdMaps->detIdIsUsedOTModule_[detUnit->geographicalId()] = isUsedOTModule;
      if (isUsedOTModule) {
        // save the module index among the extension modules
        detIdMaps->detIdToOTModuleId_[detUnit->geographicalId()] = otModuleId;
        otModuleId++;
      }
    }
    detIdMaps->usedOTModules_.assign(detIdMaps->detIdToOTModuleId_.begin(), detIdMaps->detIdToOTModuleId_.end());
  }

  return detIdMaps;
}

void PixelTrackProducerFromSoAAlpaka::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("trackSrc", edm::InputTag("pixelTracksAlpaka"));
  desc.add<edm::InputTag>("pixelRecHitLegacySrc", edm::InputTag("siPixelRecHitsPreSplittingLegacy"));
  desc.add<edm::InputTag>("outerTrackerRecHitSrc", edm::InputTag("hltSiPhase2RecHits"));
  desc.add<edm::InputTag>("outerTrackerRecHitSoAConverterSrc", edm::InputTag("phase2OTRecHitsSoAConverter"));
  desc.add<int>("minNumberOfHits", 0);
  desc.add<std::string>("minQuality", "loose");
  desc.add<bool>("useOTExtension", false);
  desc.add<edm::InputTag>("otClustersOnDemand", edm::InputTag(""))
      ->setComment(
          "OT hits of the tracks made on demand from these clusters with "
          "Phase2StripCPE (= the legacy rechits); empty = outerTrackerRecHitSrc");
  desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"));

  // this option for removing tracks with exactly 4 hits is a temporary solution to reduce the fake rate in Phase-2
  // and is to be replaced by a smarter inclusive track selection in the CA directly
  desc.add<bool>("requireQuadsFromConsecutiveLayers", false);

  descriptions.addWithDefaultLabel(desc);
}

void PixelTrackProducerFromSoAAlpaka::produce(edm::StreamID streamID,
                                              edm::Event &iEvent,
                                              const edm::EventSetup &iSetup) const {
  // enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity };
  reco::TrackBase::TrackQuality recoQuality[] = {reco::TrackBase::undefQuality,
                                                 reco::TrackBase::undefQuality,
                                                 reco::TrackBase::discarded,
                                                 reco::TrackBase::loose,
                                                 reco::TrackBase::tight,
                                                 reco::TrackBase::tight,
                                                 reco::TrackBase::highPurity};
  assert(reco::TrackBase::highPurity == recoQuality[int(pixelTrack::Quality::highPurity)]);

#ifdef GPU_DEBUG
  std::cout << "Converting soa helix in reco tracks" << std::endl;
#endif

  // index map: trackId(in SoA) -> trackId(in legacy edm)
  auto indToEdmP = std::make_unique<IndToEdm>();
  auto &indToEdm = *indToEdmP;

  auto const &idealField = iSetup.getData(idealMagneticFieldToken_);

  // prepare container for legacy tracks
  pixeltrackfitting::TracksWithRecHits tracks;

  // get trackerTopology
  auto const &trackerTopology = iSetup.getData(trackerTopologyToken_);

  // get the maps for the detId of the OT modules
  auto const &detIdMaps = *runCache(iEvent.getRun().index());

  // get beamspot
  const auto &bsh = iEvent.get(beamSpotToken_);
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  // get the module's starting indices in the hit collection
  auto const &pixelHitsModuleStart = iEvent.get(pixelHMSToken_);

  // get Pixel RecHits
  auto const &pixelRecHitsDSV = iEvent.get(pixelRecHitsToken_);
  auto const &pixelRecHits = pixelRecHitsDSV.data();
  auto const nPixelHits = pixelRecHits.size();

  // get OT RecHits if needed
  size_t nOTHits = 0;
  const Phase2TrackerRecHit1DCollectionNew *otRecHitsDSV = nullptr;
  edm::Handle<Phase2TrackerCluster1DCollectionNew> otClustersH;
  std::optional<Phase2TrackerRecHitOnDemand> otOnDemand;
  if (useOTExtension_ && otOnDemand_) {
    otClustersH = iEvent.getHandle(otClustersToken_);
    otOnDemand.emplace(otClustersH, iSetup.getData(otGeomToken_), iSetup.getData(otCpeToken_));
    nOTHits = otClustersH->dataSize();  // one legacy rechit per cluster
  } else if (useOTExtension_) {
    otRecHitsDSV = &iEvent.get(otRecHitsToken_);
    nOTHits = otRecHitsDSV->dataSize();
  }

  size_t nTotalHits = nPixelHits + nOTHits;

  // hitmap to go from a unique RecHit identifier to the RecHit in the legacy collection
  // (unique hit identifier is equivalent to the position of the hit in the RecHit SoA)
  std::vector<TrackingRecHit const *> hitmap;
  hitmap.resize(nTotalHits, nullptr);
  // on-demand OT hits (cluster key per hit identifier, storage with stable addresses)
  std::vector<int32_t> otKeyOfIdx;
  std::deque<Phase2TrackerRecHit1D> otHitStore;

  // loop over pixel RecHits to fill the hitmap
  for (auto const &pixelHit : pixelRecHits) {
    auto const &thit = static_cast<BaseTrackerRecHit const &>(pixelHit);
    auto const detI = thit.det()->index();
    auto const &clus = thit.firstClusterRef();
    assert(clus.isPixel());

    // get hit identifier as (hit offset of the module) + (hit index in this module)
    auto const idx = pixelHitsModuleStart[detI] + clus.pixelCluster().originalId();

    assert(nullptr == hitmap[idx]);
    hitmap[idx] = &pixelHit;
  }

  // if OT RecHits are used in PixelTracks, fill the hitmap also with those
  if (useOTExtension_) {
    // The RecHits in the SoA are ordered according to the detUnit->index()
    // of the respective OT module. For this reason, we need the map from the
    // detId to the moduleId among all used OT modules. This otModuleId corresponds
    // to the module's position in the otHitsModuleStart that we get from the event.

    // get the module's starting indices in the hit collection
    auto const &otHitsModuleStart = iEvent.get(otHMSToken_);

    // perform the exact same loop of how the SoA is initially filled with OT hits
    // and get the index by counting the hits (starting from the correpondign HitStartModule)
    if (otOnDemand) {
      // the same loop over the cluster detsets (same detIds, sizes and order as the legacy rechits): idx -> key
      otKeyOfIdx.assign(nTotalHits, -1);
      auto const *d0 = otClustersH->data().data();
      for (auto const &detSet : *otClustersH) {
        if (auto const m = detIdMaps.otModuleId(detSet.detId()); m >= 0) {
          auto otModuleId = uint32_t(m);
          for (int idx = otHitsModuleStart[otModuleId]; auto const &cluster : detSet) {
            otKeyOfIdx[idx] = int32_t(&cluster - d0);
            idx++;
          }
        }
      }
    }
    if (!otOnDemand)
      for (auto const &detSet : *otRecHitsDSV) {
        // check if module is used in extension and get the corresponding otModuleId
        if (auto const m = detIdMaps.otModuleId(detSet.detId()); m >= 0) {
          auto otModuleId = uint32_t(m);

          // loop over the RecHits of the module and fill the hitmap
          for (int idx = otHitsModuleStart[otModuleId]; auto const &recHit : detSet) {
            assert(nullptr == hitmap[idx]);
            hitmap[idx] = &recHit;
            idx++;
          }
        }
      }
  }

  // function that returns the number of skipped layers for a given pair of RecHits
  // for the case where the inner RecHit is in the pixel barrel.
  auto getNSkippedLayersInnerInBarrel = [&](const DetId &innerDetId,
                                            const DetId &outerDetId,
                                            const TrackingRecHit *innerRecHit) {
    int nSkippedLayers = 0;
    switch (outerDetId.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        nSkippedLayers = trackerTopology.pxbLayer(outerDetId) - trackerTopology.pxbLayer(innerDetId) - 1;
        break;
      case PixelSubdetector::PixelEndcap:
        nSkippedLayers = trackerTopology.pxfDisk(outerDetId) - 1;  // -1 because first disk has Id 1
        break;
      case StripSubdetector::TOB:
        // if the inner RecHit is at the edge of the barrel layer, consider the jump to the first OT layer as no skip
        if (std::abs(innerRecHit->globalPosition().z()) > 17)
          nSkippedLayers = trackerTopology.getOTLayerNumber(outerDetId) - 1;  // -1 because first barrel has Id 1
        else
          nSkippedLayers = trackerTopology.getOTLayerNumber(outerDetId) + 4 - trackerTopology.pxbLayer(innerDetId) - 1;
        break;
    }
    return nSkippedLayers;
  };

  // function that returns the number of skipped layers for a given pair of RecHits
  // for the case where the inner RecHit is in the pixel endcap.
  auto getNSkippedLayersInnerInEndcap = [&](const DetId &innerDetId, const DetId &outerDetId) {
    int nSkippedLayers = 0;
    switch (outerDetId.subdetId()) {
      case PixelSubdetector::PixelEndcap:
        nSkippedLayers = trackerTopology.pxfDisk(outerDetId) - trackerTopology.pxfDisk(innerDetId) - 1;
        break;
      case StripSubdetector::TOB:
        nSkippedLayers = trackerTopology.getOTLayerNumber(outerDetId) - 1;  // -1 because first disk has Id 1
        break;
    }
    return nSkippedLayers;
  };

  // function that returns the number of skipped layers for a given pair of RecHits
  // for the case where the inner RecHit is in the OT barrel.
  auto getNSkippedLayersInnerInOT = [&](const DetId &innerDetId, const DetId &outerDetId) {
    assert(outerDetId.subdetId() == StripSubdetector::TOB);
    int nSkippedLayers =
        trackerTopology.getOTLayerNumber(outerDetId) - trackerTopology.getOTLayerNumber(innerDetId) - 1;
    return nSkippedLayers;
  };

  // function that returns the number of skipped layers for a given pair of RecHits
  // It works only for Phase-2, as this feature does not make sense for Phase-1 due to the smaller number of layers.
  // (needed for layer-skipping quadruplet rejection)
  auto getNSkippedLayers = [&](const TrackingRecHit *innerRecHit, const TrackingRecHit *outerRecHit) {
    // get detIds and subdetectors of the hits to determine their layers
    auto innerDetId = innerRecHit->geographicalId();
    auto outerDetId = outerRecHit->geographicalId();

    int nSkippedLayers = 0;

    switch (innerDetId.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        nSkippedLayers = getNSkippedLayersInnerInBarrel(innerDetId, outerDetId, innerRecHit);
        break;
      case PixelSubdetector::PixelEndcap:
        nSkippedLayers = getNSkippedLayersInnerInEndcap(innerDetId, outerDetId);
        break;
      case StripSubdetector::TOB:
        nSkippedLayers = getNSkippedLayersInnerInOT(innerDetId, outerDetId);
        break;
    }
    return nSkippedLayers;
  };

  std::vector<const TrackingRecHit *> hits;
  hits.reserve(5);  //TODO move to a configurable parameter?

  auto const &tsoa = iEvent.get(trackSoAToken_);
  auto const quality = tsoa.view().tracks().quality();
  auto const hitOffs = tsoa.view().tracks().hitOffsets();
  auto const hitIdxs = tsoa.view().trackHits().id();
  auto nTracks = tsoa.view().tracks().nTracks();

  tracks.reserve(nTracks);

  int32_t nt = 0;

  // sort index by pt
  std::vector<int32_t> sortIdxs(nTracks);
  std::iota(sortIdxs.begin(), sortIdxs.end(), 0);
  // sort good-quality tracks by pt, keep bad-quality tracks at the bottom
  std::sort(sortIdxs.begin(), sortIdxs.end(), [&](int32_t const i1, int32_t const i2) {
    if (quality[i1] >= minQuality_ && quality[i2] >= minQuality_)
      return tsoa.view().tracks()[i1].pt() > tsoa.view().tracks()[i2].pt();
    else
      return quality[i1] > quality[i2];
  });

  indToEdm.resize(nTracks, -1);

  // loop over (sorted) tracks
  for (const auto &it : sortIdxs) {
    auto nHits = reco::nHits(tsoa.view().tracks(), it);
    assert(nHits >= 3);
    auto q = quality[it];

    // apply cuts on quality and number of hits
    if (q < minQuality_)
      // since the tracks are sorted according to quality,
      // we can break after the first track with low quality
      break;
    if (nHits < minNumberOfHits_)  //move to nLayers?
      continue;

    hits.resize(nHits);
    auto start = (it == 0) ? 0 : hitOffs[it - 1];
    auto end = hitOffs[it];
    int nRemovedHits{0};

    for (auto iHit = start; iHit < end; ++iHit) {
      // if hit in hitmap: true for pixel hits, true for OT hits if useOTExtension_
      auto hitIdx = hitIdxs[iHit];
      if (hitIdx < nTotalHits) {
        if (hitmap[hitIdx] == nullptr && !otKeyOfIdx.empty() && otKeyOfIdx[hitIdx] >= 0) {
          otHitStore.push_back(otOnDemand->make(otKeyOfIdx[hitIdx]));
          hitmap[hitIdx] = &otHitStore.back();
        }
        hits[iHit - start] = hitmap[hitIdx];
      }
      // else remove the OT hit from the track
      else
        nRemovedHits++;
    }
    hits.resize(nHits - nRemovedHits);
    end = end - nRemovedHits;

    // implement custome requirement for quadruplets coming from consecutive layers
    if (requireQuadsFromConsecutiveLayers_ && (nHits == 4)) {
      bool skipThisTrack{false};
      // loop over layer pairs and check if they skip
      for (auto iHit = start; iHit < end - 1; ++iHit) {
        // if the inner (iHit-start) to outer (iHit-start+1) hit layer-change skips 1 or more
        // layers skipt the track
        if (getNSkippedLayers(hits[iHit - start], hits[iHit - start + 1]) > 0) {
          skipThisTrack = true;
          break;
        }
      }
      if (skipThisTrack) {
        indToEdm[it] = pixelTrack::skippedTrack;  // mark as skipped
        continue;
      }
    }

#ifdef CA_DEBUG
    std::cout << "track soa : " << it << " with hits: ";
    for (auto iHit = start; iHit < end; ++iHit)
      std::cout << hitIdxs[iHit] << " - ";
    std::cout << std::endl;
#endif

    // store the index of the SoA:
    // indToEdm[index_SoAtrack] -> index_edmTrack (if it exists)
    indToEdm[it] = nt;
    ++nt;

    // mind: this values are respect the beamspot!
    float chi2 = tsoa.view().tracks()[it].chi2();
    float phi = reco::phi(tsoa.view().tracks(), it);

    riemannFit::Vector5d ipar, opar;
    riemannFit::Matrix5d icov, ocov;
    reco::copyToDense<riemannFit::Vector5d, riemannFit::Matrix5d>(tsoa.view().tracks(), ipar, icov, it);
    riemannFit::transformToPerigeePlane(ipar, icov, opar, ocov);

    LocalTrajectoryParameters lpar(opar(0), opar(1), opar(2), opar(3), opar(4), 1.);
    AlgebraicSymMatrix55 m;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        m(i, j) = ocov(i, j);

    float sp = std::sin(phi);
    float cp = std::cos(phi);
    Surface::RotationType rot(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);

    Plane impPointPlane(bs, rot);
    GlobalTrajectoryParameters gp(
        impPointPlane.toGlobal(lpar.position()), impPointPlane.toGlobal(lpar.momentum()), lpar.charge(), &idealField);
    JacobianLocalToCurvilinear jl2c(impPointPlane, lpar, idealField);

    AlgebraicSymMatrix55 mo = ROOT::Math::Similarity(jl2c.jacobian(), m);

    int ndof = 2 * hits.size() - 5;
    chi2 = chi2 * ndof;
    GlobalPoint vv = gp.position();
    math::XYZPoint pos(vv.x(), vv.y(), vv.z());
    GlobalVector pp = gp.momentum();
    math::XYZVector mom(pp.x(), pp.y(), pp.z());

    // A device fit that failed numerically can leave finite SoA parameters that still map to a
    // non-finite reco trajectory (through transformToPerigeePlane / the local->global transform).
    // Never emit such a track: drop it and free the edm index reserved above.
    if (not(std::isfinite(chi2) and std::isfinite(mom.x()) and std::isfinite(mom.y()) and std::isfinite(mom.z()))) {
      indToEdm[it] = pixelTrack::skippedTrack;
      --nt;
      continue;
    }

    auto track = std::make_unique<reco::Track>(chi2, ndof, pos, mom, gp.charge(), CurvilinearTrajectoryError(mo));

    // bad and edup not supported as fit not present or not reliable
    auto tkq = recoQuality[int(q)];
    track->setQuality(tkq);
    // loose,tight and HP are inclusive
    if (reco::TrackBase::highPurity == tkq) {
      track->setQuality(reco::TrackBase::tight);
      track->setQuality(reco::TrackBase::loose);
    } else if (reco::TrackBase::tight == tkq) {
      track->setQuality(reco::TrackBase::loose);
    }
    track->setQuality(tkq);
    // filter???
    tracks.emplace_back(track.release(), hits);
  }

#ifdef GPU_DEBUG
  std::cout << "processed " << nt << " good tuples " << tracks.size() << " out of " << indToEdm.size() << std::endl;
#endif

  // store tracks
  storeTracks(iEvent, tracks, trackerTopology);
  iEvent.put(std::move(indToEdmP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpaka);
