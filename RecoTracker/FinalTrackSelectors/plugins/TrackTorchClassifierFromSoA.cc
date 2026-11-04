#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackTorchClassifierFeaturesSoA.h"

// This module consumes the HOST copy of the Alpaka device scores
// The framework automatically creates host copies of device PortableCollections
class TrackTorchClassifierFromSoA : public edm::stream::EDProducer<> {
public:
  explicit TrackTorchClassifierFromSoA(const edm::ParameterSet& iConfig);
  ~TrackTorchClassifierFromSoA() override = default;

  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const edm::EDGetTokenT<reco::TrackCollection> tracks_token_;
  const edm::EDGetTokenT<PortableHostCollection<TrackTorchClassifierScoresSoA>> scores_token_;
  const edm::EDGetTokenT<PortableHostCollection<TrackTorchClassifierFeaturesSoA>> features_token_;
  // qualitycuts (loose, tight, hp) for prompt/displaced, for pT < ptThreshold and pT >= ptThreshold
  const std::vector<double> quality_cuts_prompt_;
  const float dxy_threshold_;
  const std::vector<double> quality_cuts_displaced_;
  const float pt_threshold_;
  const std::vector<double> quality_cuts_prompt_high_pt_;
  const std::vector<double> quality_cuts_displaced_high_pt_;
  const std::string name_;

  const edm::EDPutTokenT<MVACollection> scores_output_token_;
  const edm::EDPutTokenT<edm::ValueMap<float>> mva_vals_token_;
  const edm::EDPutTokenT<edm::ValueMap<int>> track_quals_token_;
  const edm::EDPutTokenT<QualityMaskCollection> quality_mask_output_token_;
};

TrackTorchClassifierFromSoA::TrackTorchClassifierFromSoA(const edm::ParameterSet& iConfig)
    : tracks_token_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      scores_token_(consumes(iConfig.getParameter<edm::InputTag>("scores"))),
      features_token_(consumes(iConfig.getParameter<edm::InputTag>("features"))),
      quality_cuts_prompt_(iConfig.getParameter<std::vector<double>>("qualityCutsPrompt")),
      dxy_threshold_(iConfig.getParameter<double>("dxyThreshold")),
      quality_cuts_displaced_(iConfig.getParameter<std::vector<double>>("qualityCutsDisplaced")),
      pt_threshold_(iConfig.getParameter<double>("ptThreshold")),
      // An empty high-pT vector means: same cuts as for low pT
      quality_cuts_prompt_high_pt_(iConfig.getParameter<std::vector<double>>("qualityCutsPromptHighPt").empty()
                                       ? quality_cuts_prompt_
                                       : iConfig.getParameter<std::vector<double>>("qualityCutsPromptHighPt")),
      quality_cuts_displaced_high_pt_(iConfig.getParameter<std::vector<double>>("qualityCutsDisplacedHighPt").empty()
                                          ? quality_cuts_displaced_
                                          : iConfig.getParameter<std::vector<double>>("qualityCutsDisplacedHighPt")),
      name_(iConfig.getParameter<std::string>("name")),
      scores_output_token_(produces<MVACollection>("MVAValues")),
      mva_vals_token_(produces<edm::ValueMap<float>>("MVAVals")),
      track_quals_token_(produces<edm::ValueMap<int>>(name_)),
      quality_mask_output_token_(produces<QualityMaskCollection>(name_)) {
  if (quality_cuts_prompt_.size() != 3 || quality_cuts_displaced_.size() != 3 ||
      quality_cuts_prompt_high_pt_.size() != 3 || quality_cuts_displaced_high_pt_.size() != 3)
    throw cms::Exception("Configuration") << "TrackTorchClassifierFromSoA: quality cuts need 3 values (loose, tight, "
                                             "highPurity); the high-pT ones may also be empty";
}

void TrackTorchClassifierFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltInitialStepTracks"));
  desc.add<edm::InputTag>("scores", edm::InputTag("hltInitialStepTrackTorchClassifier"));
  desc.add<edm::InputTag>("features", edm::InputTag("hltInitialStepTrackTorchClassifier"));
  desc.add<std::vector<double>>("qualityCutsPrompt", {0.5, 0.5, 0.5})
      ->setComment("MVA score quality cuts for prompt tracks (pT < ptThreshold)");
  desc.add<double>("dxyThreshold", 0.5)->setComment("Tracks with |dxy| > this value use the displaced quality cuts");
  desc.add<std::vector<double>>("qualityCutsDisplaced", {0.5, 0.5, 0.5})
      ->setComment("MVA score quality cuts for displaced tracks (pT < ptThreshold)");
  desc.add<double>("ptThreshold", 5.0)->setComment("Tracks with pT >= this value (GeV) use the high-pT quality cuts");
  desc.add<std::vector<double>>("qualityCutsPromptHighPt", {})
      ->setComment("MVA score quality cuts for prompt tracks with pT >= ptThreshold; empty: same as qualityCutsPrompt");
  desc.add<std::vector<double>>("qualityCutsDisplacedHighPt", {})
      ->setComment(
          "MVA score quality cuts for displaced tracks with pT >= ptThreshold; empty: same as qualityCutsDisplaced");
  desc.add<std::string>("name", "QualityMasks")->setComment("Instance label of the track quality products");
  descriptions.addWithDefaultLabel(desc);
}

void TrackTorchClassifierFromSoA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::TrackCollection> tracks_handle;
  iEvent.getByToken(tracks_token_, tracks_handle);
  const auto& tracks = *tracks_handle;
  const auto& scores_host = iEvent.get(scores_token_);
  const auto& features_host = iEvent.get(features_token_);
  const auto nTracks = tracks.size();

  // Create products: MVA score collection and quality mask collection
  auto mvas = std::make_unique<MVACollection>(tracks.size(), -99.f);
  auto mva_vals = std::make_unique<edm::ValueMap<float>>();
  std::vector<int> selTracks(nTracks, 0);
  auto selTracksValueMap = std::make_unique<edm::ValueMap<int>>();

  // Access scores and features from the host collection
  auto scores_view = scores_host.const_view();
  auto features_view = features_host.const_view();

  // Loop over tracks
  for (size_t i = 0; i < nTracks; ++i) {
    float score = scores_view[i].score();
    float dxy = features_view[i].dxyBeamSpot();
    int quality = tracks[i].qualityMask();

    (*mvas)[i] = score;
    const bool high_pt = tracks[i].pt() >= pt_threshold_;
    const auto& cuts_prompt = high_pt ? quality_cuts_prompt_high_pt_ : quality_cuts_prompt_;
    const auto& cuts_displaced = high_pt ? quality_cuts_displaced_high_pt_ : quality_cuts_displaced_;
    const bool displaced = std::abs(dxy) > dxy_threshold_;
    const bool pass_loose = score >= cuts_prompt[0] || (displaced && score >= cuts_displaced[0]);
    const bool pass_tight = score >= cuts_prompt[1] || (displaced && score >= cuts_displaced[1]);
    const bool pass_hp = score >= cuts_prompt[2] || (displaced && score >= cuts_displaced[2]);

    if (pass_loose)
      quality |= (1 << reco::TrackBase::loose);
    if (pass_loose && pass_tight)
      quality |= (1 << reco::TrackBase::tight);
    if (pass_loose && pass_tight && pass_hp)
      quality |= (1 << reco::TrackBase::highPurity);
    selTracks[i] = quality;
  }

  edm::ValueMap<float>::Filler mva_filler(*mva_vals);
  mva_filler.insert(tracks_handle, mvas->begin(), mvas->end());
  mva_filler.fill();

  edm::ValueMap<int>::Filler qual_filler(*selTracksValueMap);
  qual_filler.insert(tracks_handle, selTracks.begin(), selTracks.end());
  qual_filler.fill();

  for (auto& q : selTracks)
    q = std::max(q, 0);
  auto quals = std::make_unique<QualityMaskCollection>(selTracks.begin(), selTracks.end());
  iEvent.put(scores_output_token_, std::move(mvas));
  iEvent.put(mva_vals_token_, std::move(mva_vals));
  iEvent.put(quality_mask_output_token_, std::move(quals));
  iEvent.put(track_quals_token_, std::move(selTracksValueMap));
}

DEFINE_FWK_MODULE(TrackTorchClassifierFromSoA);
