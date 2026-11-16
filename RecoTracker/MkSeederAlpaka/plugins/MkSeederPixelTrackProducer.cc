// MkSeederPixelTrackProducer: the "pixel tracks" of the mkFit seeder (phase2MkSeeder): per seed selected by the seed
// classifier a reco::Track from the seed's perigee (MkSeederProducer, MkSeederSeedsSoA.h: its beam-line state at the
// point of closest approach to the beam line, as TrackFromSeedProducer makes it of the beam-line seed), with the
// seed's hits, a track extra and the reference to the seed ('src': the selected seeds as TrajectorySeeds,
// MkSeederTrajectorySeedConverter, one per entry of 'selection', the seeds' rows). A seed whose closest approach did
// not converge gives the track of a failed fit (DataFormats/TrackReco/interface/trackFromSeedFitFailed.h), one whose
// beam-line state does not convert no track.
// Without 'src' (the seeds on the device, MkSeederDevicePixelSeedsProducer): no TrajectorySeeds; the selection is the
// classifier's ('scores'), the hits the seeds' TrackingRecHits by cluster index ('mkFitPixelHits', 'mkFitStripHits'),
// the extras without a seed reference; the tracks are then those of MkSeederDevicePixelSeedsProducer's rows, in order.
// With 'otClustersOnDemand' the outer-tracker hits are made from their clusters (Phase2TrackerRecHitOnDemand, equal to
// the legacy Phase2TrackerRecHits' hit at that cluster key) instead of read through 'mkFitStripHits', so the menu needs
// neither the legacy OT rechits nor the mkFit OT hit converter.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2TrackerRecHitOnDemand.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsHostCollection.h"

class MkSeederPixelTrackProducer : public edm::global::EDProducer<> {
public:
  explicit MkSeederPixelTrackProducer(edm::ParameterSet const &iConfig)
      : seedsToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
        topologyToken_{esConsumes()},
        tracksPutToken_{produces()},
        hitsPutToken_{produces()},
        extrasPutToken_{produces()} {
    if (const auto &src = iConfig.getParameter<edm::InputTag>("src"); !src.label().empty()) {
      refsToken_ = consumes(src);
      selectionToken_ = consumes(iConfig.getParameter<edm::InputTag>("selection"));
    } else {
      seedCountersToken_ = consumes(iConfig.getParameter<edm::InputTag>("seeds"));
      scoresToken_ = consumes(iConfig.getParameter<edm::InputTag>("scores"));
      pixelHitsToken_ = consumes(iConfig.getParameter<edm::InputTag>("mkFitPixelHits"));
      if (const auto &ot = iConfig.getParameter<edm::InputTag>("otClustersOnDemand"); !ot.label().empty()) {
        otClustersToken_ = consumes(ot);
        otGeometryToken_ = esConsumes();
        otCpeToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("Phase2StripCPE"));
      } else {
        stripHitsToken_ = consumes(iConfig.getParameter<edm::InputTag>("mkFitStripHits"));
      }
      mkFitGeometryToken_ = esConsumes();
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag{"hltMkFitSeederSeedsClassified"})
        ->setComment(
            "The selected seeds as TrajectorySeeds (the tracks' seed references and hits); empty: no TrajectorySeeds, "
            "the selection from 'scores' and the hits from 'mkFitPixelHits' / 'mkFitStripHits'");
    desc.add<edm::InputTag>("seeds", edm::InputTag{"mkSeederProducer"})
        ->setComment("The MkSeederProducer module (the final seeds' perigees, hits and number)");
    desc.add<edm::InputTag>("selection", edm::InputTag{"hltMkFitSeederSeedsClassified", "mkFitIndices"})
        ->setComment("The rows of the seeds in 'src'");
    desc.add<edm::InputTag>("scores", edm::InputTag{"hltMkFitSeederSeedsClassifiedDevice"})
        ->setComment("Without 'src': the MkSeederSeedClassifier module (the selection)");
    desc.add<edm::InputTag>("mkFitPixelHits", edm::InputTag{"hltMkFitSiPixelHits"})
        ->setComment("Without 'src': the pixel hits' TrackingRecHits by cluster index");
    desc.add<edm::InputTag>("mkFitStripHits", edm::InputTag{"hltMkFitSiPhase2Hits"})
        ->setComment(
            "Without 'src' and 'otClustersOnDemand': the outer-tracker hits' TrackingRecHits by cluster index");
    desc.add<edm::InputTag>("otClustersOnDemand", edm::InputTag{""})
        ->setComment(
            "Without 'src': the outer-tracker hits made on demand from these clusters with Phase2StripCPE (empty: "
            "'mkFitStripHits')");
    desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag{"phase2StripCPEESProducer", "Phase2StripCPE"})
        ->setComment("With 'otClustersOnDemand': the CPE of the outer-tracker hits (as Phase2TrackerRecHits')");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event &iEvent, edm::EventSetup const &iSetup) const override {
    const auto seeds = iEvent.get(seedsToken_).const_view();
    const auto perigee = seeds.perigee();
    const TrackerTopology &topology = iSetup.getData(topologyToken_);
    // the seeds: their rows and hits (and with 'src' their TrajectorySeeds)
    std::vector<int> rows;
    std::vector<std::vector<const TrackingRecHit *>> seedHits;
    // the outer-tracker hits made on demand (stable addresses)
    std::optional<Phase2TrackerRecHitOnDemand> otOnDemand;
    std::deque<Phase2TrackerRecHit1D> otStore;
    edm::Handle<edm::View<TrajectorySeed>> refs;
    if (!refsToken_.isUninitialized()) {
      refs = iEvent.getHandle(refsToken_);
      rows = iEvent.get(selectionToken_);
      if (rows.size() != refs->size())
        throw cms::Exception("LogicError") << "MkSeederPixelTrackProducer: " << rows.size() << " selected seeds, "
                                           << refs->size() << " seed references";
      seedHits.resize(rows.size());
      for (size_t k = 0; k < rows.size(); ++k)
        for (const auto &hit : (*refs)[k].recHits())
          seedHits[k].push_back(&hit);
    } else {
      // as MkSeederDevicePixelSeedsProducer: selected, with a perigee, with hits
      const uint32_t n = std::min<uint32_t>(iEvent.get(seedCountersToken_)->nOutput, perigee.metadata().size());
      const auto scores = iEvent.get(scoresToken_).const_view();
      const auto hits = seeds.hits();
      const auto clusterIndices = seeds.clusterIndices();
      const auto &pixelHits = iEvent.get(pixelHitsToken_).hits();
      static const std::vector<TrackingRecHit const *> kNoHits;
      const auto &stripHits = stripHitsToken_.isUninitialized() ? kNoHits : iEvent.get(stripHitsToken_).hits();
      if (!otClustersToken_.isUninitialized())
        otOnDemand.emplace(
            iEvent.getHandle(otClustersToken_), iSetup.getData(otGeometryToken_), iSetup.getData(otCpeToken_));
      const mkfit::TrackerInfo &trackerInfo = iSetup.getData(mkFitGeometryToken_).trackerInfo();
      for (uint32_t i = 0; i < n; ++i) {
        const int nHits = hits.nHits(i);
        if (!scores.isSelected(i) || perigee.status(i) == ::mkseeder::kPerigeeNone || nHits == 0)
          continue;
        rows.push_back(i);
        auto &out = seedHits.emplace_back();
        out.reserve(nHits);
        for (int k = 0; k < nHits; ++k) {
          const int layer = hits.hitLayers(i)(k);
          const int32_t index = clusterIndices.hitClusterIndex(i)(k);
          if (otOnDemand && !trackerInfo[layer].is_pixel()) {
            if (index < 0 || uint32_t(index) >= otOnDemand->size())
              throw cms::Exception("LogicError")
                  << "MkSeederPixelTrackProducer: seed " << i << " hit " << k << " has no outer-tracker cluster (layer "
                  << layer << ", cluster " << index << ")";
            out.push_back(&otStore.emplace_back(otOnDemand->make(index)));
            continue;
          }
          const auto &layerHits = trackerInfo[layer].is_pixel() ? pixelHits : stripHits;
          if (index < 0 || index >= int32_t(layerHits.size()) || layerHits[index] == nullptr)
            throw cms::Exception("LogicError")
                << "MkSeederPixelTrackProducer: seed " << i << " hit " << k << " has no TrackingRecHit (layer " << layer
                << ", cluster " << index << ")";
          out.push_back(layerHits[index]);
        }
      }
    }
    const auto refHits = iEvent.getRefBeforePut(hitsPutToken_);
    const auto refExtras = iEvent.getRefBeforePut(extrasPutToken_);
    reco::TrackCollection tracks;
    TrackingRecHitCollection recHits;
    reco::TrackExtraCollection extras;
    size_t nHits = 0;
    for (const auto &h : seedHits)
      nHits += h.size();
    tracks.reserve(rows.size());
    extras.reserve(rows.size());
    recHits.reserve(nHits);
    int nNone = 0, nFailedFit = 0;
    for (size_t k = 0; k < rows.size(); ++k) {
      const int row = rows[k];
      if (row < 0 || row >= perigee.metadata().size())
        throw cms::Exception("LogicError")
            << "MkSeederPixelTrackProducer: selection index " << row << " of " << perigee.metadata().size();
      const uint8_t status = perigee.status(row);
      if (status == ::mkseeder::kPerigeeNone) {
        ++nNone;
        continue;
      }
      if (status == ::mkseeder::kPerigeeValid) {
        reco::TrackBase::CovarianceMatrix covariance;
        for (int i = 0, c = 0; i < 5; ++i)
          for (int j = 0; j <= i; ++j, ++c)
            covariance(i, j) = perigee.covariance(row)(c);
        const auto vertex = perigee.vertex(row);
        const auto momentum = perigee.momentum(row);
        tracks.emplace_back(0.,
                            0.,
                            reco::TrackBase::Point(vertex(0), vertex(1), vertex(2)),
                            reco::TrackBase::Vector(momentum(0), momentum(1), momentum(2)),
                            int(perigee.charge(row)),
                            covariance);
      } else {
        // the magic values of a failed fit
        tracks.emplace_back(
            -1, -1, reco::TrackBase::Point(), reco::TrackBase::Vector(), 0, reco::TrackBase::CovarianceMatrix());
        ++nFailedFit;
      }
      // the hit pattern as appendHits(): up to the first hit that does not fit in it
      for (const auto *hit : seedHits[k])
        if (!tracks.back().appendHitPattern(*hit, topology))
          break;
      const size_t firstHit = recHits.size();
      for (const auto *hit : seedHits[k])
        recHits.push_back(*hit);
      extras.emplace_back();
      extras.back().setHits(refHits, firstHit, recHits.size() - firstHit);
      if (refs.isValid())
        extras.back().setSeedRef(edm::RefToBase<TrajectorySeed>(refs, k));
      tracks.back().setExtra(reco::TrackExtraRef(refExtras, extras.size() - 1));
    }
    if (nNone + nFailedFit > 0)
      edm::LogInfo("MkSeederPixelTrackProducer")
          << "of " << rows.size() << " seeds " << nFailedFit << " give the track of a failed fit, " << nNone
          << " no track (the beam-line state does not convert)";
    iEvent.emplace(tracksPutToken_, std::move(tracks));
    iEvent.emplace(hitsPutToken_, std::move(recHits));
    iEvent.emplace(extrasPutToken_, std::move(extras));
  }

private:
  const edm::EDGetTokenT<::mkseeder::MkSeederSeedsHostCollection> seedsToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed>> refsToken_;
  edm::EDGetTokenT<std::vector<int>> selectionToken_;
  edm::EDGetTokenT<::mkseeder::MkSeederDupCountersHostObject> seedCountersToken_;
  edm::EDGetTokenT<::mkseeder::MkSeederSeedScoresHostCollection> scoresToken_;
  edm::EDGetTokenT<MkFitClusterIndexToHit> pixelHitsToken_, stripHitsToken_;
  edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeometryToken_;
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otClustersToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> otGeometryToken_;
  edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> otCpeToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topologyToken_;
  const edm::EDPutTokenT<reco::TrackCollection> tracksPutToken_;
  const edm::EDPutTokenT<TrackingRecHitCollection> hitsPutToken_;
  const edm::EDPutTokenT<reco::TrackExtraCollection> extrasPutToken_;
};  // class MkSeederPixelTrackProducer

DEFINE_FWK_MODULE(MkSeederPixelTrackProducer);
