#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/Product.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/ScopedContext.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/LST/interface/LSTPixelSeedInput.h"
#include "RecoTracker/LST/interface/LSTPhase2OTHitsInput.h"
#include "RecoTracker/LST/interface/LSTOutput.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include "SDL/LST.h"
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTProducer : public edm::stream::EDProducer<edm::ExternalWork> {
  public:
    LSTProducer(edm::ParameterSet const& config)
        : lstPixelSeedInputToken_{consumes<LSTPixelSeedInput>(config.getParameter<edm::InputTag>("pixelSeedInput"))},
          lstPhase2OTHitsInputToken_{consumes<LSTPhase2OTHitsInput>(config.getParameter<edm::InputTag>("phase2OTHitsInput"))},
          lstOutputToken_{produces<LSTOutput>()} {}

    void acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder task) override {
      // create a context based on the EDM stream number
      cms::alpakatools::ScopedContextAcquire<Queue> ctx{event.streamID(), std::move(task)};

      // Inputs
      auto const& pixelSeeds = event.get(lstPixelSeedInputToken_);
      auto const& phase2OTHits = event.get(lstPhase2OTHitsInputToken_);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      lst_.eventSetup();
      lst_.run(ctx.queue().getNativeHandle(),
               pixelSeeds.px(),
               pixelSeeds.py(),
               pixelSeeds.pz(),
               pixelSeeds.dxy(),
               pixelSeeds.dz(),
               pixelSeeds.ptErr(),
               pixelSeeds.etaErr(),
               pixelSeeds.stateTrajGlbX(),
               pixelSeeds.stateTrajGlbY(),
               pixelSeeds.stateTrajGlbZ(),
               pixelSeeds.stateTrajGlbPx(),
               pixelSeeds.stateTrajGlbPy(),
               pixelSeeds.stateTrajGlbPz(),
               pixelSeeds.q(),
               pixelSeeds.algo(),
               pixelSeeds.hitIdx(),
               phase2OTHits.detId(),
               phase2OTHits.x(),
               phase2OTHits.y(),
               phase2OTHits.z());
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
    }

    void produce(edm::Event& event, edm::EventSetup const&) override {
      // Output
      LSTOutput lstOutput;
      lstOutput.setLSTOutputTraits(lst_.hits(), lst_.len(), lst_.pt(), lst_.eta(), lst_.phi(), lst_.seedIdx());
 
      event.emplace(lstOutputToken_, std::move(lstOutput));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("pixelSeedInput", edm::InputTag{"lstPixelSeedInputProducer"});
      desc.add<edm::InputTag>("phase2OTHitsInput", edm::InputTag{"lstPhase2OTHitsInputProducer"});
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<LSTPixelSeedInput> lstPixelSeedInputToken_;
    edm::EDGetTokenT<LSTPhase2OTHitsInput> lstPhase2OTHitsInputToken_;
    edm::EDPutTokenT<LSTOutput> lstOutputToken_;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    SDL::LST lst_;
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTProducer);
