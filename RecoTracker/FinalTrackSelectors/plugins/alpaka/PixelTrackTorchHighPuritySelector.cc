/**
 * PixelTrackTorchHighPuritySelector
 *
 * GPU/Accelerator module performing HighPurity pixel-track selection composed of:
 *   1. CA-based quality preselection
 *   2. Feature extraction
 *   3. TorchScript DNN inference
 *   4. Score-based filtering
 *   5. Track/hit compaction and output production
 *
 * Input: TracksSoA (pixel tracks + hit associations), TrackingRecHitsSoA + OTRecHitsSoA
 * (the latter only when useHitFeatures = true).
 *
 * Torch inference: track tensor [maxPreselectedTracks (or, with dynamicBatches, the batches that can hold the
 * preselected tracks), N_track_features], padding slots 0-filled.
 */

#include <algorithm>
#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/VectorHitsSoACollection.h"
#include "RecoTracker/PixelSeeding/interface/CAHitsView.h"

#include "RecoTracker/FinalTrackSelectors/interface/PixelTrackFeaturesSoA.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/plugins/alpaka/PixelTrackTorchHighPuritySelectorKernels.h"

#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelTrackTorchHighPuritySelector : public stream::FixedQueueEDProducer<> {
    using TkSoADevice = reco::TracksSoACollection;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using VectorHitsOnDevice = reco::VectorHitsSoACollection;
    using OTHitsOnDevice = reco::OTRecHitsSoACollection;

  public:
    explicit PixelTrackTorchHighPuritySelector(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produce(device::Event&, const device::EventSetup&) override;
    void beginStream(edm::StreamID /*sid*/, Queue queue) override;

    /// Registers the "track_features" tensor, sliced in batches of `batchSize` rows: the 17 fit/cov columns,
    /// plus the first 14 hit/vector hit columns when useHitFeatures, in the model's input order. The tensor may
    /// span the two blocks: they are back to back and the fit block is a whole number of padded columns.
    template <typename TFitRecord, typename THitRecord>
    static void addTrackFeatures(cms::torch::alpakatools::BatchedTensorCollection<Queue>& tc,
                                 uint32_t batchSize,
                                 TFitRecord& r,
                                 THitRecord& h,
                                 bool useHitFeatures) {
      if (useHitFeatures) {
        tc.addBatched<PixelTrackFitFeaturesSoA>("track_features",
                                                batchSize,
                                                r.chi2(),
                                                r.dzError(),
                                                r.dxyError(),
                                                r.eta(),
                                                r.nHits(),
                                                r.phi(),
                                                r.phiError(),
                                                r.pt(),
                                                r.qOverPtError(),
                                                r.dzBS(),
                                                r.dxyBS(),
                                                r.nLayers(),
                                                r.cotThetaError(),
                                                r.covCotThetaDz(),
                                                r.covDxyQOverPt(),
                                                r.covPhiDxy(),
                                                r.covPhiQOverPt(),
                                                h.caFitChi2(),
                                                h.psFrac(),
                                                h.r0(),
                                                h.nPS(),
                                                h.spanZ(),
                                                h.nVectorHits(),
                                                h.logChi2VectorHit(),
                                                h.kErr(),
                                                h.dcaEst(),
                                                h.nBarrel(),
                                                h.rzChi2(),
                                                h.meanVectorHitKappa(),
                                                h.leverArm(),
                                                h.rMax());
      } else {
        tc.addBatched<PixelTrackFitFeaturesSoA>("track_features",
                                                batchSize,
                                                r.chi2(),
                                                r.dzError(),
                                                r.dxyError(),
                                                r.eta(),
                                                r.nHits(),
                                                r.phi(),
                                                r.phiError(),
                                                r.pt(),
                                                r.qOverPtError(),
                                                r.dzBS(),
                                                r.dxyBS(),
                                                r.nLayers(),
                                                r.cotThetaError(),
                                                r.covCotThetaDz(),
                                                r.covDxyQOverPt(),
                                                r.covPhiDxy(),
                                                r.covPhiQOverPt());
      }
    }

    /// Inference dtype, shared by the warm-up and the per-event forward passes: the first forward moves
    /// and casts the model to the device, so the two must agree.
    std::optional<::torch::Dtype> inferenceDtype() const {
      return inferenceHalf_ ? std::optional<::torch::Dtype>{::torch::kHalf} : std::nullopt;
    }

    const device::EDGetToken<TkSoADevice> pixelTrackToken_;
    const int maxNumberOfTracks_;
    const int maxPreselectedTracks_;
    const int minNumberOfHits_;
    const int avgHitsPerTrack_;
    const pixelTrack::Quality minimumTrackQuality_;
    const double scoreThreshold_;
    // dxy-aware threshold ramp: threshold goes from scoreThresholdLowDxy at |dxyBS|=0 down to
    // scoreThreshold at |dxyBS|>=dxyRampKnee. scoreThresholdLowDxy < 0 disables it (flat threshold).
    const double scoreThresholdLowDxy_;
    const double dxyRampKnee_;
    // tracks with at most shortTrackMaxLayers_ layers are cut at scoreThresholdShortTracks_ (0: no separate cut)
    const int shortTrackMaxLayers_;
    const double scoreThresholdShortTracks_;
    // tracks with |eta| below passThroughMaxAbsEta_ skip the score cut (negative: none)
    const double passThroughMaxAbsEta_;
    // run the model only on the batches that can hold preselected tracks, instead of on maxPreselectedTracks rows
    const bool dynamicBatches_;
    // When true, the 14 CA hit/vector hit features are appended to the 17 fit/cov features, for the enriched
    // 31-feature model; the merged-hits product is consumed only then.
    const bool useHitFeatures_;
    // forward(dtype) casts the whole model: kHalf is fine for the MLP but breaks a tree model, whose
    // int64 index buffers must not be cast. inferenceHalf = false leaves the model in fp32.
    const bool inferenceHalf_;
    device::EDGetToken<HitsOnDevice> pixelHitsToken_;
    device::EDGetToken<VectorHitsOnDevice> vectorHitsToken_;
    // Raw OT-rechit SoA, so the hit-feature walk can resolve the bit30-tagged raw-OT ids a track may
    // carry. Consumed only when useHitFeatures_.
    device::EDGetToken<OTHitsOnDevice> otRecHitsSoAToken_;
    torch::AlpakaModel model_;
    const int batchSize_;
    const int warmupIterations_ = 3;
    const device::EDPutToken<TkSoADevice> tokenTrackOut_;
  };

  PixelTrackTorchHighPuritySelector::PixelTrackTorchHighPuritySelector(const edm::ParameterSet& iConfig)
      : FixedQueueEDProducer(iConfig),
        pixelTrackToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))),
        maxNumberOfTracks_(iConfig.getParameter<int>("maxNumberOfTracks")),
        maxPreselectedTracks_(iConfig.getParameter<int>("maxPreselectedTracks")),
        minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
        avgHitsPerTrack_(iConfig.getParameter<int>("avgHitsPerTrack")),
        minimumTrackQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minimumTrackQuality"))),
        scoreThreshold_(iConfig.getParameter<double>("scoreThreshold")),
        scoreThresholdLowDxy_(iConfig.getParameter<double>("scoreThresholdLowDxy")),
        dxyRampKnee_(iConfig.getParameter<double>("dxyRampKnee")),
        shortTrackMaxLayers_(iConfig.getParameter<int>("shortTrackMaxLayers")),
        scoreThresholdShortTracks_(iConfig.getParameter<double>("scoreThresholdShortTracks")),
        passThroughMaxAbsEta_(iConfig.getParameter<double>("passThroughMaxAbsEta")),
        dynamicBatches_(iConfig.getParameter<bool>("dynamicBatches")),
        useHitFeatures_(iConfig.getParameter<bool>("useHitFeatures")),
        inferenceHalf_(iConfig.getParameter<bool>("inferenceHalf")),
        model_(iConfig.getParameter<edm::FileInPath>("model").fullPath()),
        batchSize_(iConfig.getParameter<int>("batchSize")),
        tokenTrackOut_(produces()) {
    if (useHitFeatures_) {
      pixelHitsToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
      vectorHitsToken_ = consumes(iConfig.getParameter<edm::InputTag>("vectorHitsSrc"));
      otRecHitsSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoASrc"));
    }
    if (minimumTrackQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " is not a pixelTrack::Quality";
    }
    if (minimumTrackQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minimumTrackQuality") + " not supported";
    }
    if (maxPreselectedTracks_ > maxNumberOfTracks_) {
      throw cms::Exception("PixelTrackConfiguration") << "maxPreselectedTracks must be <= maxNumberOfTracks";
    }
    if (batchSize_ <= 0) {
      throw cms::Exception("PixelTrackConfiguration") << "batchSize must be positive";
    }
  }

  void PixelTrackTorchHighPuritySelector::beginStream(edm::StreamID /*sid*/, Queue queue) {
    // Warmup the model with dummy data

    // Create temporary feature and score buffers used to warm up the model.
    PixelTrackFeaturesOnDevice trackFeatures(queue, batchSize_, useHitFeatures_ ? batchSize_ : 0);
    PixelTrackScoresOnDevice trackScoresOnDevice(queue, batchSize_);
    // a Metarecords keeps a reference to its view, so the views live in named locals
    auto fit_view = trackFeatures.view().fit();
    auto hit_view = trackFeatures.view().hit();
    auto fit_record = fit_view.records();
    auto hit_record = hit_view.records();
    auto score_record = trackScoresOnDevice.view().records();

    for (auto it = 0; it < warmupIterations_; ++it) {
      cms::torch::alpakatools::BatchedTensorCollection<Queue> dummy_inputs;
      cms::torch::alpakatools::BatchedTensorCollection<Queue> dummy_outputs;

      // Same column list (and therefore the same tensor width) as the per-event inference below;
      // the buffers hold batchSize_ rows, so this is a single batch.
      addTrackFeatures(dummy_inputs, batchSize_, fit_record, hit_record, useHitFeatures_);

      dummy_outputs.addBatched<PixelTrackScoresSoA>("track_scores", batchSize_, score_record.score());

      model_.forward(queue, dummy_inputs, dummy_outputs, inferenceDtype());
    }
  }

  void PixelTrackTorchHighPuritySelector::produce(device::Event& iEvent, const device::EventSetup&) {
    /*
    Processing steps:
      1. CA-based preselection of tracks
      2. Feature extraction (track SoA)
      3. DNN inference
      4. Score-based filtering
      5. Track compaction and output production
*/
    // Retrieve tokens
    auto& queue = iEvent.queue();
    const auto& tracks = iEvent.get(pixelTrackToken_).view();

    // Instantiate the necessary objects in memory
    //  - Temporary storage for filtering
    auto d_nPreselectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_nSelectedTracks = cms::alpakatools::make_device_buffer<int>(queue);
    auto d_preselectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);
    auto d_selectedTrackIndices = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_trackHitCounts = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_selectedTrackHitOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxPreselectedTracks_);
    auto d_preselectionOffsets = cms::alpakatools::make_device_buffer<int[]>(queue, maxNumberOfTracks_);

    alpaka::memset(queue, d_nPreselectedTracks, 0);
    alpaka::memset(queue, d_nSelectedTracks, 0);
    alpaka::memset(queue, d_trackHitCounts, 0);
    alpaka::memset(queue, d_selectedTrackHitOffsets, 0);
    alpaka::memset(queue, d_preselectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_selectedTrackIndices, 0xFF);
    alpaka::memset(queue, d_preselectionOffsets, 0);

    // 1. CA-based preselection of tracks
    //  Launch first kernel to look which tracks need to be filtered out
    //  based on quality criteria from the CA

    launchCAPreselection(queue,
                         maxNumberOfTracks_,
                         minNumberOfHits_,
                         minimumTrackQuality_,
                         tracks.tracks(),
                         alpaka::getPtrNative(d_preselectedTrackIndices),
                         alpaka::getPtrNative(d_preselectionOffsets),
                         alpaka::getPtrNative(d_nPreselectedTracks));

    //  - Features and scores containers, of maxPreselectedTracks_ rows, or with dynamicBatches of the whole batches
    //    that can hold the preselected tracks (the rows beyond are padding whose scores are never read): on the host
    //    the preselected count is at hand (the queue is synchronous); on a device, to avoid a synchronisation, the
    //    rows of the input track collection bound it (the event's tracks when the CA compacts its output)
    int nFeatureRows = maxPreselectedTracks_;
    if (dynamicBatches_) {
      int nBound;
      if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        auto h_nPreselectedTracks = cms::alpakatools::make_host_buffer<int>(queue);
        alpaka::memcpy(queue, h_nPreselectedTracks, d_nPreselectedTracks);
        alpaka::wait(queue);
        nBound = *h_nPreselectedTracks.data();
      } else {
        nBound = tracks.tracks().metadata().size();
      }
      nBound = std::clamp(nBound, 1, maxPreselectedTracks_);
      nFeatureRows = int(cms::alpakatools::round_up_by(uint32_t(nBound), uint32_t(batchSize_)));
    }
    //  - the hit block is sized 0 when the model does not read it
    PixelTrackFeaturesOnDevice trackFeatures(queue, nFeatureRows, useHitFeatures_ ? nFeatureRows : 0);
    PixelTrackScoresOnDevice trackScoresOnDevice(queue, nFeatureRows);

    // 2. Feature extraction. The merged TrackingRecHitsSoA is needed only for the hit/vector hit features;
    // otherwise pass a null view and nHitsTot = 0.
    caStructures::CAHitsView hitsView{};
    int nHitsTot = 0;
    // OT-rechit view for resolving tagged OT extras (empty view + 0 when not needed).
    ::reco::OTRecHitsConstView otHitsView{};
    uint32_t nOTHits = 0;
    if (useHitFeatures_) {
      const auto& pixHits = iEvent.get(pixelHitsToken_);
      const auto& vectorHits = iEvent.get(vectorHitsToken_);
      hitsView = caStructures::CAHitsView(pixHits.const_view().trackingHits(),
                                          pixHits.const_view().hitModules(),
                                          vectorHits.const_view().vectorHits(),
                                          vectorHits.const_view().vectorHitModules(),
                                          pixHits.nHits(),
                                          vectorHits.nVectorHits(),
                                          pixHits.nModules());
      nHitsTot = hitsView.size();
      const auto& otHits = iEvent.get(otRecHitsSoAToken_);
      otHitsView = otHits.const_view().otRecHits();
      nOTHits = otHitsView.metadata().size();
    }

    launchFeaturesExtractor(queue,
                            nFeatureRows,
                            tracks.tracks(),
                            tracks.trackHits(),
                            hitsView,
                            nHitsTot,
                            otHitsView,
                            nOTHits,
                            alpaka::getPtrNative(d_preselectedTrackIndices),
                            alpaka::getPtrNative(d_nPreselectedTracks),
                            trackFeatures.view().fit(),
                            trackFeatures.view().hit(),
                            alpaka::getPtrNative(d_trackHitCounts));

    // 3. DNN inference
    //  Prepare TensorCollection inputs and outputs for the model
    // a Metarecords keeps a reference to its view, so the views live in named locals
    auto fit_view = trackFeatures.view().fit();
    auto hit_view = trackFeatures.view().hit();
    auto fit_record = fit_view.records();
    auto hit_record = hit_view.records();
    auto score_record = trackScoresOnDevice.view().records();

    auto inputs = cms::torch::alpakatools::BatchedTensorCollection<Queue>();
    auto outputs = cms::torch::alpakatools::BatchedTensorCollection<Queue>();

    // Order must match the TorchScript model input schema
    addTrackFeatures(inputs, batchSize_, fit_record, hit_record, useHitFeatures_);

    outputs.addBatched<PixelTrackScoresSoA>("track_scores", batchSize_, score_record.score());

    model_.forward(queue, inputs, outputs, inferenceDtype());

    // 4. Score-based filtering
    launchScoreFilter(queue,
                      maxPreselectedTracks_,
                      scoreThreshold_,
                      scoreThresholdLowDxy_,
                      dxyRampKnee_,
                      shortTrackMaxLayers_,
                      scoreThresholdShortTracks_,
                      passThroughMaxAbsEta_,
                      trackFeatures.const_view().fit(),
                      trackScoresOnDevice.view(),
                      alpaka::getPtrNative(d_preselectedTrackIndices),
                      alpaka::getPtrNative(d_nPreselectedTracks),
                      alpaka::getPtrNative(d_trackHitCounts),
                      alpaka::getPtrNative(d_selectedTrackIndices),
                      alpaka::getPtrNative(d_nSelectedTracks),
                      alpaka::getPtrNative(d_selectedTrackHitOffsets));

    auto tracks_out = launchProduceOutputTracks(queue,
                                                maxPreselectedTracks_,
                                                avgHitsPerTrack_,
                                                tracks.tracks(),
                                                tracks.trackHits(),
                                                alpaka::getPtrNative(d_selectedTrackIndices),
                                                alpaka::getPtrNative(d_nSelectedTracks),
                                                alpaka::getPtrNative(d_selectedTrackHitOffsets));
    iEvent.emplace(tokenTrackOut_, std::move(tracks_out));
  }

  void PixelTrackTorchHighPuritySelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pixelTrackSrc", {"hltPhase2PixelTracksSoA"});
    desc.add<int>("maxNumberOfTracks", 2 * 60 * 1024);
    desc.add<int>("maxPreselectedTracks", 9984);
    desc.add<int>("minNumberOfHits", 0);
    desc.add<int>("avgHitsPerTrack", 8);
    desc.add<std::string>("minimumTrackQuality", "tight");
    desc.add<edm::FileInPath>("model");
    desc.add<double>("scoreThreshold", 0.4);
    desc.add<int>("batchSize", 4992);
    // dxy-aware threshold ramp: the cut goes from scoreThresholdLowDxy at |dxyBS| = 0 to
    // scoreThreshold at |dxyBS| >= dxyRampKnee [cm]. scoreThresholdLowDxy < 0 disables it.
    desc.add<double>("scoreThresholdLowDxy", -1.0);
    desc.add<double>("dxyRampKnee", 2.0);
    desc.add<int>("shortTrackMaxLayers", 0)
        ->setComment(
            "Tracks with at most this many layers are cut at scoreThresholdShortTracks instead of scoreThreshold "
            "(0: no separate threshold).");
    desc.add<double>("scoreThresholdShortTracks", 0.);
    desc.add<double>("passThroughMaxAbsEta", -1.)
        ->setComment("Tracks with |eta| below this value skip the score cut (negative: none).");
    desc.add<bool>("dynamicBatches", false)
        ->setComment(
            "Run the model only on the batches that can hold preselected tracks (identical scores): on the host "
            "the preselected count is read back, on a device the input track rows bound it (useful with an "
            "event-sized input, see compactOutput of the CA).");
    // Hit/vector hit features (enriched model); the hit features read the pixel rechits + vector hits the CA indexed,
    // consumed only when useHitFeatures is true.
    desc.add<bool>("useHitFeatures", false);
    desc.add<edm::InputTag>("pixelRecHitSrc", {"siPixelRecHitsPreSplittingAlpaka"});
    desc.add<edm::InputTag>("vectorHitsSrc", {"phase2OTVectorHitsSoA"});
    // Raw OT-rechit SoA, consumed only when useHitFeatures is true: resolves the bit30-tagged raw-OT hit
    // ids so OT-extended tracks are scored on their full hit content.
    desc.add<edm::InputTag>("otRecHitsSoASrc", {"caOTRecHitsSoA"});
    // inferenceHalf = false keeps the model in fp32, required for a tree model whose int index buffers
    // must not be cast to half; true selects the fp16 path used by the MLP models.
    desc.add<bool>("inferenceHalf", true);
    descriptions.addWithDefaultLabel(desc);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTrackTorchHighPuritySelector);
