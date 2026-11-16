// MkSeederTrajectorySeedConverter: the seeds of MkSeederProducer that the seed classifier selected
// (MkSeederSeedClassifier) as TrajectorySeeds: the
// state at the last hit converted by mkfitconv::seedTsosOnLastHit(), clones of the hits' TrackingRecHits (by cluster
// index). Also put: the selected seeds' rows in the seeder's products ("mkFitIndices": their positions in the mkFit
// seeds of MkSeederSeedConverter, for the pixel tracks) and the scores of all seeds ("scores").

#include <cstdint>
#include <vector>

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitSeedStateConversion.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsHostCollection.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class MkSeederTrajectorySeedConverter : public edm::global::EDProducer<> {
public:
  explicit MkSeederTrajectorySeedConverter(edm::ParameterSet const &iConfig)
      : seedsToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
        seedCountersToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
        scoresToken_{consumes(iConfig.getParameter<edm::InputTag>("scores"))},
        pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitPixelHits"))},
        stripHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitStripHits"))},
        mkFitGeometryToken_{esConsumes()},
        magneticFieldToken_{esConsumes()},
        propagatorAlongToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))},
        propagatorOppositeToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))},
        propagateToSurface_{iConfig.getParameter<bool>("propagateToSurface")},
        seedsPutToken_{produces()},
        indicesPutToken_{produces("mkFitIndices")},
        scoresPutToken_{produces("scores")} {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("seeds", edm::InputTag{"mkSeederProducer"})
        ->setComment("The MkSeederProducer module (the final seeds and their number)");
    desc.add<edm::InputTag>("scores", edm::InputTag{"mkSeederSeedClassifier"})
        ->setComment("The MkSeederSeedClassifier module (the seeds' scores and selection)");
    desc.add<edm::InputTag>("mkFitPixelHits", edm::InputTag{"hltMkFitSiPixelHits"})
        ->setComment("The pixel hits' TrackingRecHits by cluster index");
    desc.add<edm::InputTag>("mkFitStripHits", edm::InputTag{"hltMkFitSiPhase2Hits"})
        ->setComment("The outer-tracker hits' TrackingRecHits by cluster index");
    desc.add<edm::ESInputTag>("propagatorAlong", edm::ESInputTag{"", "hltESPAnalyticalPropagator"})
        ->setComment("The propagators of the state conversion (mkfitconv::seedTsosOnLastHit)");
    desc.add<edm::ESInputTag>("propagatorOpposite", edm::ESInputTag{"", "hltESPBwdAnalyticalPropagator"})
        ->setComment("The propagators of the state conversion (mkfitconv::seedTsosOnLastHit)");
    desc.add<bool>("propagateToSurface", false)->setComment("Propagate the state onto the last hit's module surface");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event &iEvent, edm::EventSetup const &iSetup) const override {
    const auto seeds = iEvent.get(seedsToken_).const_view();
    const uint32_t n = std::min<uint32_t>(iEvent.get(seedCountersToken_)->nOutput, seeds.hits().metadata().size());
    const auto scores = iEvent.get(scoresToken_).const_view();
    const auto &pixelHits = iEvent.get(pixelHitsToken_).hits();
    const auto &stripHits = iEvent.get(stripHitsToken_).hits();
    const mkfit::TrackerInfo &trackerInfo = iSetup.getData(mkFitGeometryToken_).trackerInfo();
    const MagneticField &magneticField = iSetup.getData(magneticFieldToken_);
    const Propagator &along = iSetup.getData(propagatorAlongToken_);
    const Propagator &opposite = iSetup.getData(propagatorOppositeToken_);
    const auto hits = seeds.hits();
    const auto states = seeds.states();
    const auto clusterIndices = seeds.clusterIndices();

    std::vector<int> selected;
    std::vector<float> allScores(n);
    for (uint32_t i = 0; i < n; ++i) {
      allScores[i] = scores.score(i);
      if (scores.isSelected(i))
        selected.push_back(i);
    }
    // a seed's hit k as its TrackingRecHit (by its cluster index in its layer's hits)
    auto hitOf = [&](uint32_t i, int k) -> const TrackingRecHit * {
      const int layer = hits.hitLayers(i)(k);
      const int32_t index = clusterIndices.hitClusterIndex(i)(k);
      const auto &layerHits = trackerInfo[layer].is_pixel() ? pixelHits : stripHits;
      if (index < 0 || index >= int32_t(layerHits.size()) || layerHits[index] == nullptr)
        throw cms::Exception("LogicError") << "MkSeederTrajectorySeedConverter: seed " << i << " hit " << k
                                           << " has no TrackingRecHit (layer " << layer << ", cluster " << index << ")";
      return layerHits[index];
    };

    TrajectorySeedCollection out;
    out.reserve(selected.size());
    std::vector<int> indices;
    indices.reserve(selected.size());
    int nFailed = 0;
    for (size_t s = 0; s < selected.size(); ++s) {
      // the hits are scattered in memory: those of the seed four ahead prefetched
      if (s + 4 < selected.size())
        for (int k = 0, nh = hits.nHits(selected[s + 4]); k < nh; ++k)
          __builtin_prefetch(hitOf(selected[s + 4], k));
      const uint32_t i = selected[s];
      const int nHits = hits.nHits(i);
      if (nHits == 0)
        throw cms::Exception("LogicError") << "MkSeederTrajectorySeedConverter: seed " << i << " has no hits";
      edm::OwnVector<TrackingRecHit> recHits;
      recHits.reserve(nHits);
      for (int k = 0; k < nHits; ++k)
        recHits.push_back(hitOf(i, k)->clone());
      const GeomDet *lastDet = recHits.back().det();
      if (lastDet == nullptr)
        throw cms::Exception("LogicError")
            << "MkSeederTrajectorySeedConverter: the last hit of seed " << i << " has no GeomDet";
      // the state as an mkFit seed's (MkSeederSeedConverter)
      mkfit::TrackState state;
      for (int d = 0; d < 6; ++d)
        state.parameters[d] = states.parameters(i)(d);
      for (int d = 0; d < 21; ++d)
        state.errors.Array()[d] = states.covariance(i)(d);
      state.charge = states.charge(i);
      const mkfit::Track track(state, states.chi2(i), int(i), 0, nullptr);
      TrajectoryStateOnSurface tsos;
      if (!mkfitconv::seedTsosOnLastHit(track, *lastDet, magneticField, along, opposite, propagateToSurface_, tsos)) {
        // the device found a state on the plane where the CMSSW propagation does not (float rounding): no seed
        ++nFailed;
        continue;
      }
      out.emplace_back(trajectoryStateTransform::persistentState(tsos, lastDet->geographicalId().rawId()),
                       std::move(recHits),
                       alongMomentum);
      indices.push_back(i);
    }
    if (nFailed > 0)
      edm::LogWarning("MkSeederTrajectorySeedConverter")
          << nFailed << " of " << selected.size() << " selected seeds failed the state conversion and were dropped";
    iEvent.emplace(seedsPutToken_, std::move(out));
    iEvent.emplace(indicesPutToken_, std::move(indices));
    iEvent.emplace(scoresPutToken_, std::move(allScores));
  }

private:
  const edm::EDGetTokenT<::mkseeder::MkSeederSeedsHostCollection> seedsToken_;
  const edm::EDGetTokenT<::mkseeder::MkSeederDupCountersHostObject> seedCountersToken_;
  const edm::EDGetTokenT<::mkseeder::MkSeederSeedScoresHostCollection> scoresToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelHitsToken_, stripHitsToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeometryToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorAlongToken_, propagatorOppositeToken_;
  const bool propagateToSurface_;
  const edm::EDPutTokenT<TrajectorySeedCollection> seedsPutToken_;
  const edm::EDPutTokenT<std::vector<int>> indicesPutToken_;
  const edm::EDPutTokenT<std::vector<float>> scoresPutToken_;
};  // class MkSeederTrajectorySeedConverter

DEFINE_FWK_MODULE(MkSeederTrajectorySeedConverter);
