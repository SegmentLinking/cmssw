#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/LST/interface/LSTPhase2OTHitsInput.h"
#include "RecoTracker/LST/interface/LSTOutput.h"

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
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatePutToken_;
};

LSTOutputConverter::LSTOutputConverter(edm::ParameterSet const& iConfig)
    : lstOutputToken_(consumes<LSTOutput>(iConfig.getUntrackedParameter<edm::InputTag>("lstOutput"))),
      lstPhase2OTHitsInputToken_{consumes<LSTPhase2OTHitsInput>(iConfig.getUntrackedParameter<edm::InputTag>("phase2OTHits"))},
      lstPixelSeedToken_{consumes<TrajectorySeedCollection>(iConfig.getUntrackedParameter<edm::InputTag>("lstPixelSeeds"))},
      trackCandidatePutToken_(produces<TrackCandidateCollection>()) {}

void LSTOutputConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<edm::InputTag>("lstOutput", edm::InputTag("lstProducer"));
  desc.addUntracked<edm::InputTag>("phase2OTHits", edm::InputTag("lstPhase2OTHitsInputProducer"));
  desc.addUntracked<edm::InputTag>("lstPixelSeeds", edm::InputTag("lstPixelSeedInputProducer"));

  descriptions.addWithDefaultLabel(desc);
}

void LSTOutputConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Setup
  auto const& lstOutput = iEvent.get(lstOutputToken_);
  auto const& phase2OTRecHits = iEvent.get(lstPhase2OTHitsInputToken_);
  auto const& pixelSeeds = iEvent.get(lstPixelSeedToken_);

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

  for (unsigned int i=0; i<lstTC_len.size(); i++) {
    if (lstTC_len[i]==10) { // T5
      continue; // temporary
    }

    TrajectorySeed s = pixelSeeds[lstTC_seedIdx[i]];
    PTrajectoryStateOnDet st = s.startingState();

    edm::OwnVector<TrackingRecHit> recHits;
    for (auto const& hit : s.recHits())
      recHits.push_back(hit.clone());

    unsigned int const nPixelHits = lstTC_len[i]==10 ? 0 : recHits.size();
    for (unsigned int j=nPixelHits; j<lstTC_hitIdx[i].size(); j++)
      recHits.push_back(OTHits[lstTC_hitIdx[i][j]]->clone());

    output.emplace_back(TrackCandidate(recHits,s,st));
  }

  iEvent.emplace(trackCandidatePutToken_, std::move(output));
}

DEFINE_FWK_MODULE(LSTOutputConverter);

