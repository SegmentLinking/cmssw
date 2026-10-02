#include "RecoTracker/FinalTrackSelectors/interface/alpaka/TrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/alpaka/TrackScoresDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackTorchClassifierFeaturesSoA.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackTorchClassifierAlpaka : public stream::FixedQueueEDProducer<> {
  public:
    TrackTorchClassifierAlpaka(const edm::ParameterSet& iConfig)
        : FixedQueueEDProducer<>(iConfig),
          featuresInput_token_(consumes(iConfig.getParameter<edm::InputTag>("features"))),
          scoresPut_token_{produces()},
          model_(iConfig.getParameter<edm::FileInPath>("modelPath").fullPath()),
          useOriginalAlgo_(iConfig.getParameter<bool>("useOriginalAlgo")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("modelPath",
                                edm::FileInPath("RecoTracker/FinalTrackSelectors/data/TrackTorchClassifier/model.pt"));
      desc.add<edm::InputTag>("features", edm::InputTag("hltInitialStepTrackFeatureExtractor"));
      desc.add<bool>("useOriginalAlgo", false)->setComment("Whether the model takes the track originalAlgo as input");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      const auto& features = iEvent.get(featuresInput_token_);
      const auto batch_size = features.const_view().metadata().size();

      auto scores_device = TrackScoresDeviceCollection(iEvent.queue(), batch_size);

      auto input_records = features.const_view().records();
      auto output_records = scores_device.view().records();

      cms::torch::alpakatools::TensorCollection<Queue> inputs(batch_size);
      addInputs(inputs, input_records);

      cms::torch::alpakatools::TensorCollection<Queue> outputs(batch_size);
      outputs.add<TrackTorchClassifierScoresSoA>("scores", output_records.score());

      model_.forward(iEvent.queue(), inputs, outputs);

      iEvent.emplace(scoresPut_token_, std::move(scores_device));
    }

    void beginStream(edm::StreamID sid, Queue queue) override {
      // Warmup the model with dummy data
      const int warmupBatchSize = 4992;
      // Allocate dummy input and output tensors on the device
      auto features = TrackFeaturesDeviceCollection(queue, warmupBatchSize);
      auto scores_device = TrackScoresDeviceCollection(queue, warmupBatchSize);

      auto input_records = features.view().records();
      auto output_records = scores_device.view().records();

      for (auto it = 0; it < warmupIterations_; ++it) {
        cms::torch::alpakatools::TensorCollection<Queue> dummy_inputs(warmupBatchSize);
        cms::torch::alpakatools::TensorCollection<Queue> dummy_outputs(warmupBatchSize);

        addInputs(dummy_inputs, input_records);

        dummy_outputs.add<TrackTorchClassifierScoresSoA>("scores", output_records.score());

        model_.forward(queue, dummy_inputs, dummy_outputs);
      }
    }

  private:
    template <typename Records, typename... Extra>
    static void addFeatures(cms::torch::alpakatools::TensorCollection<Queue>& inputs,
                            const Records& records,
                            Extra... extra) {
      inputs.add<TrackTorchClassifierFeaturesSoA>("features",
                                                  records.dxyBeamSpot(),
                                                  records.dzBeamSpot(),
                                                  records.dxyError(),
                                                  records.dzError(),
                                                  records.normalizedChi2(),
                                                  records.eta(),
                                                  records.phi(),
                                                  records.etaError(),
                                                  records.phiError(),
                                                  records.ndof(),
                                                  records.lostInnerHits(),
                                                  records.lostOuterHits(),
                                                  records.layersWithoutMeas(),
                                                  records.validPixelHits(),
                                                  records.validStripHits(),
                                                  extra...);
    }

    template <typename Records>
    void addInputs(cms::torch::alpakatools::TensorCollection<Queue>& inputs, const Records& records) const {
      if (useOriginalAlgo_)
        addFeatures(inputs, records, records.originalAlgo());
      else
        addFeatures(inputs, records);
    }

    const device::EDGetToken<TrackFeaturesDeviceCollection> featuresInput_token_;
    const device::EDPutToken<TrackScoresDeviceCollection> scoresPut_token_;
    torch::AlpakaModel model_;
    const bool useOriginalAlgo_;
    const int warmupIterations_ = 3;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackTorchClassifierAlpaka);
