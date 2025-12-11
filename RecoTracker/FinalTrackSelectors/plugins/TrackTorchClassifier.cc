#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoTracker/FinalTrackSelectors/interface/getBestVertex.h"
#include "PhysicsTools/PyTorch/interface/Model.h"

class TrackTorchClassifier : public edm::stream::EDProducer<> {
public:
  explicit TrackTorchClassifier(const edm::ParameterSet& iConfig);
  ~TrackTorchClassifier() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const edm::EDGetTokenT<reco::TrackCollection> src_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  const edm::EDGetTokenT<reco::VertexCollection> vertices_;
  const bool ignoreVertices_;
  
  const std::string modelPath_;
  const int batchSize_;
  const float minScore_;
  
  ::torch::Device device_;
  std::unique_ptr<cms::torch::Model> model_;
  
  const edm::EDPutTokenT<reco::TrackCollection> putToken_;
  const edm::EDPutTokenT<std::vector<float>> putScoresToken_;
};

TrackTorchClassifier::TrackTorchClassifier(const edm::ParameterSet& iConfig)
    : src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      beamspot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspot"))),
      vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      ignoreVertices_(iConfig.getParameter<bool>("ignoreVertices")),
      modelPath_(iConfig.getParameter<std::string>("modelPath")),
      batchSize_(iConfig.getParameter<int>("batchSize")),
      minScore_(iConfig.getParameter<double>("minScore")),
      device_(::torch::kCPU),
      model_(nullptr),
      putToken_(produces<reco::TrackCollection>()),
      putScoresToken_(produces<std::vector<float>>("MVAScores")) {}

void TrackTorchClassifier::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltInitialStepTracks"));
  desc.add<edm::InputTag>("beamspot", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("vertices", edm::InputTag(""));
  desc.add<bool>("ignoreVertices", true);
  desc.add<std::string>("modelPath", "RecoTracker/FinalTrackSelectors/data/best_model.pt");
  desc.add<int>("batchSize", 16);
  desc.add<double>("minScore", 0.5)->setComment("Minimum DNN score to keep track (working point)");
  descriptions.addWithDefaultLabel(desc);
}

void TrackTorchClassifier::beginStream(edm::StreamID) {
  auto fullPath = edm::FileInPath(modelPath_).fullPath();
  
  if (::torch::cuda::is_available() && ::torch::cuda::device_count() > 0) {
    device_ = ::torch::Device(::torch::kCUDA, 0);
  }
  
  model_ = std::make_unique<cms::torch::Model>(fullPath, device_);
}

void TrackTorchClassifier::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& tracks = iEvent.get(src_);
  const auto& beamSpot = iEvent.get(beamspot_);
  
  reco::VertexCollection vertices;
  if (!ignoreVertices_) {
    auto verticesHandle = iEvent.getHandle(vertices_);
    if (verticesHandle.isValid()) {
      vertices = *verticesHandle;
    }
  }
  
  int size_in = (int)tracks.size();
  std::vector<float> output(size_in, 0.0f);
  
  auto filteredTracks = std::make_unique<reco::TrackCollection>();

  if (!model_) {
    for (const auto& track : tracks) {
      filteredTracks->push_back(track);
    }
    iEvent.emplace(putToken_, std::move(*filteredTracks));
    iEvent.emplace(putScoresToken_, std::move(output));
    return;
  }

  int nbatches = (size_in + batchSize_ - 1) / batchSize_;

  for (int nb = 0; nb < nbatches; nb++) {
    int batch_start = nb * batchSize_;
    int batch_end = std::min(batch_start + batchSize_, size_in);
    int actual_batch_size = batch_end - batch_start;

    std::vector<float> inputData;
    inputData.reserve(actual_batch_size * 29);

    for (int itrack = batch_start; itrack < batch_end; itrack++) {
      const auto& trk = tracks[itrack];
      const auto& bestVertex = getBestVertex(trk, vertices);

      inputData.push_back(trk.pt());
      inputData.push_back(trk.innerMomentum().x());
      inputData.push_back(trk.innerMomentum().y());
      inputData.push_back(trk.innerMomentum().z());
      inputData.push_back(trk.innerMomentum().rho());
      inputData.push_back(trk.outerMomentum().x());
      inputData.push_back(trk.outerMomentum().y());
      inputData.push_back(trk.outerMomentum().z());
      inputData.push_back(trk.outerMomentum().rho());
      inputData.push_back(trk.ptError());
      inputData.push_back(trk.dxy(bestVertex));
      inputData.push_back(trk.dz(bestVertex));
      inputData.push_back(trk.dxy(beamSpot.position()));
      inputData.push_back(trk.dz(beamSpot.position()));
      inputData.push_back(trk.dxyError());
      inputData.push_back(trk.dzError());
      inputData.push_back(trk.normalizedChi2());
      inputData.push_back(trk.eta());
      inputData.push_back(trk.phi());
      inputData.push_back(trk.etaError());
      inputData.push_back(trk.phiError());
      inputData.push_back(trk.ndof());
      inputData.push_back(trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS));
      inputData.push_back(trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS));
      inputData.push_back(trk.hitPattern().trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_INNER_HITS));
      inputData.push_back(trk.hitPattern().trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_OUTER_HITS));
      inputData.push_back(trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS));
      inputData.push_back(trk.hitPattern().numberOfValidPixelHits());
      inputData.push_back(trk.hitPattern().numberOfValidStripHits());
    }

    auto inputTensor = ::torch::from_blob(inputData.data(), {actual_batch_size, 29}, ::torch::kFloat32).to(device_);

    std::vector<::torch::IValue> inputs;
    inputs.push_back(inputTensor);

    auto outputTensor = model_->forward(inputs).toTensor();

    auto outputCpu = outputTensor.cpu();
    auto outputAccessor = outputCpu.accessor<float, 2>();

    for (int i = 0; i < actual_batch_size; i++) {
      int itrack = batch_start + i;
      output[itrack] = outputAccessor[i][0];
    }
  }

  int n_passed = 0;
  for (int itrack = 0; itrack < size_in; itrack++) {
    if (output[itrack] >= minScore_) {
      filteredTracks->push_back(tracks[itrack]);
      n_passed++;
    }
  }

  iEvent.emplace(putToken_, std::move(*filteredTracks));
  iEvent.emplace(putScoresToken_, std::move(output));
}

DEFINE_FWK_MODULE(TrackTorchClassifier);
