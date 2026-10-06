#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/LSTCore/interface/alpaka/T3Features.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Builds the per-T3 transformer input features from the T3 collections of LSTProducer
  // (which must run with produceT3Collections = True)
  class LSTT3FeatureProducer : public global::EDProducer<> {
  public:
    LSTT3FeatureProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          maxT3Pt_(config.getParameter<double>("maxT3Pt")),
          lstInputToken_{consumes(config.getParameter<edm::InputTag>("lstInput"))},
          rangesToken_{consumes(config.getParameter<edm::InputTag>("lst"))},
          miniDoubletsToken_{consumes(config.getParameter<edm::InputTag>("lst"))},
          segmentsToken_{consumes(config.getParameter<edm::InputTag>("lst"))},
          tripletsToken_{consumes(config.getParameter<edm::InputTag>("lst"))},
          featuresToken_{produces()} {}

    void produce(edm::StreamID sid, device::Event& iEvent, const device::EventSetup& iSetup) const override {
      iEvent.emplace(featuresToken_,
                     lst::makeT3Features(iEvent.queue(),
                                         iEvent.get(lstInputToken_),
                                         iEvent.get(rangesToken_),
                                         iEvent.get(miniDoubletsToken_),
                                         iEvent.get(segmentsToken_),
                                         iEvent.get(tripletsToken_),
                                         maxT3Pt_));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("lstInput", edm::InputTag{"lstInputProducer"});
      desc.add<edm::InputTag>("lst", edm::InputTag{"lstProducer"})
          ->setComment("LSTProducer with produceT3Collections = True");
      desc.add<double>("maxT3Pt", 2000.)->setComment("Keep T3s with pt < maxT3Pt (MAX_T3_PT in transformer-oc)");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const float maxT3Pt_;
    const device::EDGetToken<lst::LSTInputDeviceCollection> lstInputToken_;
    const device::EDGetToken<lst::ObjectRangesDeviceCollection> rangesToken_;
    const device::EDGetToken<lst::MiniDoubletsDeviceCollection> miniDoubletsToken_;
    const device::EDGetToken<lst::SegmentsDeviceCollection> segmentsToken_;
    const device::EDGetToken<lst::TripletsDeviceCollection> tripletsToken_;
    const device::EDPutToken<lst::T3FeaturesDeviceCollection> featuresToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTT3FeatureProducer);
