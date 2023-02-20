#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/LST/interface/LSTPhase2OTHitsInput.h"
#include "RecoTracker/LST/interface/LSTOutput.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

//typedef SeedingHitSet::ConstRecHitPointer Hit;
//
//struct HitLessByRadius {
//  bool operator()(const Hit& h1, const Hit& h2) { return h1->globalPosition().perp2() < h2->globalPosition().perp2(); }
//};

class LSTOutputConverter : public edm::global::EDProducer<> {
public:
  explicit LSTOutputConverter(edm::ParameterSet const& iConfig);
  ~LSTOutputConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<LSTOutput> lstOutputToken_;
  const edm::EDGetTokenT<LSTPhase2OTHitsInput> lstPhase2OTHitsInputToken_;
  const edm::EDGetTokenT<TrajectorySeedCollection> lstPixelSeedToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  //const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  //std::unique_ptr<SeedCreator> seedCreator_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatePutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> seedStopInfoPutToken_;
};

LSTOutputConverter::LSTOutputConverter(edm::ParameterSet const& iConfig)
    : lstOutputToken_(consumes<LSTOutput>(iConfig.getUntrackedParameter<edm::InputTag>("lstOutput"))),
      lstPhase2OTHitsInputToken_{consumes<LSTPhase2OTHitsInput>(iConfig.getUntrackedParameter<edm::InputTag>("phase2OTHits"))},
      lstPixelSeedToken_{consumes<TrajectorySeedCollection>(iConfig.getUntrackedParameter<edm::InputTag>("lstPixelSeeds"))},
      mfToken_(esConsumes()),
      //beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamSpot"))),
      //seedCreator_(SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", iConfig.getParameter<edm::ParameterSet>("SeedCreatorPSet"), consumesCollector())),
      trackCandidatePutToken_(produces<TrackCandidateCollection>()),
      seedStopInfoPutToken_(produces<std::vector<SeedStopInfo>>()) {}

void LSTOutputConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<edm::InputTag>("lstOutput", edm::InputTag("lstProducer"));
  desc.addUntracked<edm::InputTag>("phase2OTHits", edm::InputTag("lstPhase2OTHitsInputProducer"));
  desc.addUntracked<edm::InputTag>("lstPixelSeeds", edm::InputTag("lstPixelSeedInputProducer"));
  //desc.addUntracked<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));

  //edm::ParameterSetDescription psd0;
  //psd0.add<std::string>("ComponentName", std::string("SeedFromConsecutiveHitsCreator"));
  //psd0.add<std::string>("propagator", std::string("PropagatorWithMaterial"));
  //psd0.add<double>("SeedMomentumForBOFF", 5.0);
  //psd0.add<double>("OriginTransverseErrorMultiplier", 1.0);
  //psd0.add<double>("MinOneOverPtError", 1.0);
  //psd0.add<std::string>("magneticField", std::string(""));
  //psd0.add<std::string>("TTRHBuilder", std::string("WithTrackAngle"));
  //psd0.add<bool>("forceKinematicWithRegionDirection", false);
  //desc.add<edm::ParameterSetDescription>("SeedCreatorPSet", psd0);

  descriptions.addWithDefaultLabel(desc);
}

void LSTOutputConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Setup
  auto const& lstOutput = iEvent.get(lstOutputToken_);
  auto const& phase2OTRecHits = iEvent.get(lstPhase2OTHitsInputToken_);
  auto const& pixelSeeds = iEvent.get(lstPixelSeedToken_);
  const auto& mf = iSetup.getData(mfToken_);
  //auto const& bs = iEvent.get(beamSpotToken_);

  // Vector definitions
  std::vector<std::vector<unsigned int>> const lstTC_hitIdx = lstOutput.hitIdx();
  std::vector<unsigned int> const lstTC_len = lstOutput.len();
  std::vector<float> const lstTC_pt = lstOutput.pt();
  std::vector<float> const lstTC_eta = lstOutput.eta();
  std::vector<float> const lstTC_phi = lstOutput.phi();
  std::vector<int> const lstTC_seedIdx = lstOutput.seedIdx();

  TrackCandidateCollection output;
  output.reserve(lstTC_len.size());

  auto const& OTHits = phase2OTRecHits.hits();

  //auto seedsFromT5 = std::make_unique<TrajectorySeedCollection>();
  //GlobalPoint vtxOfBs(bs.x0(), bs.y0(), bs.z0()); 
  LogDebug("LSTOutputConverter")<<"lstTC size "<<lstTC_len.size();
  for (unsigned int i=0; i<lstTC_len.size(); i++) {
    LogDebug("LSTOutputConverter")<<" cand "<<i<<" "<<lstTC_len[i]<<" "<<lstTC_pt[i]<<" "<<lstTC_eta[i]<<" "<<lstTC_phi[i]<<" "<<lstTC_seedIdx[i];
    if (lstTC_len[i]==10) { // T5
      continue; // temporary
      //std::vector<Hit> hitsFromT5;
      //for (auto hitIdx : lstTC_hitIdx[i])
      //  hitsFromT5.push_back((Hit) OTHits[hitIdx]);
      //sort(hitsFromT5.begin(), hitsFromT5.end(), HitLessByRadius());
      //GlobalTrackingRegion region(lstTC_pt[i], vtxOfBs, 0.2, 0.2);
      //seedCreator_->init(region, iSetup, nullptr);
      //seedCreator_->makeSeed(*seedsFromT5, hitsFromT5);
    }

    //TrajectorySeed s = lstTC_len[i]==10 ? (*seedsFromT5)[0] : pixelSeeds[lstTC_seedIdx[i]];
    TrajectorySeed s = pixelSeeds[lstTC_seedIdx[i]];
    //PTrajectoryStateOnDet st = s.startingState();

    edm::OwnVector<TrackingRecHit> recHits;
    for (auto const& hit : s.recHits())
      recHits.push_back(hit.clone());

    TrajectoryStateOnSurface tsos = trajectoryStateTransform::transientState(s.startingState(), recHits.back().surface(), &mf);
    PTrajectoryStateOnDet st = trajectoryStateTransform::persistentState(tsos, recHits[0].det()->geographicalId().rawId());

    unsigned int const nPixelHits = lstTC_len[i]==10 ? 0 : recHits.size();
    for (unsigned int j=nPixelHits; j<lstTC_hitIdx[i].size(); j++)
      recHits.push_back(OTHits[lstTC_hitIdx[i][j]]->clone());

    recHits.sort([](const auto& a, const auto& b) {
      const auto asub = a.det()->subDetector();
      const auto bsub = b.det()->subDetector();
      if (GeomDetEnumerators::isInnerTracker(asub) && GeomDetEnumerators::isOuterTracker(bsub)) {
        return true;
      } else if (GeomDetEnumerators::isOuterTracker(asub) && GeomDetEnumerators::isInnerTracker(bsub)) {
        return false;
      } else {
        return asub < bsub;
      }
 
      const auto& apos = a.globalPosition();
      const auto& bpos = b.globalPosition();
      if (GeomDetEnumerators::isBarrel(asub)) {
        return apos.perp2() < bpos.perp2();
      } else {
        return std::abs(apos.z()) < std::abs(bpos.z());
      }
    });

    output.emplace_back(TrackCandidate(recHits,s,st));
  }

  LogDebug("LSTOutputConverter")<<"done with conversion: output size "<<output.size();
  iEvent.emplace(trackCandidatePutToken_, std::move(output));
  iEvent.emplace(seedStopInfoPutToken_, 0U); //dummy stop info
}

DEFINE_FWK_MODULE(LSTOutputConverter);

