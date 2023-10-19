#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/LST/interface/LSTOutput.h"
#include "RecoTracker/LST/interface/LSTPhase2OTHitsInput.h"
#include "RecoTracker/LST/interface/LSTPixelSeedInput.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "SDL/LST.h"
#include "SDL/ModuleConnectionMap.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTProducer : public stream::SynchronizingEDProducer<> {
  public:
    LSTProducer(edm::ParameterSet const& config)
        : lstPixelSeedInputToken_{consumes<LSTPixelSeedInput>(config.getParameter<edm::InputTag>("pixelSeedInput"))},
          lstPhase2OTHitsInputToken_{
              consumes<LSTPhase2OTHitsInput>(config.getParameter<edm::InputTag>("phase2OTHitsInput"))},
          mmapToken_(esConsumes(config.getParameter<edm::ESInputTag>("mmap"))),
          verbose_(config.getParameter<int>("verbose")),
          lstOutputToken_{produces()} {
      mmap_pLSTokens_ = edm::vector_transform(
          config.getUntrackedParameter<std::vector<edm::ESInputTag>>("mmap_pLS"), [&](const edm::ESInputTag& tag) {
            return (edm::ESGetToken<SDL::ModuleConnectionMap, TrackerRecoGeometryRecord>)esConsumes(tag);
          });
      mmap_pLS_posTokens_ = edm::vector_transform(
          config.getUntrackedParameter<std::vector<edm::ESInputTag>>("mmap_pLS_pos"), [&](const edm::ESInputTag& tag) {
            return (edm::ESGetToken<SDL::ModuleConnectionMap, TrackerRecoGeometryRecord>)esConsumes(tag);
          });
      mmap_pLS_negTokens_ = edm::vector_transform(
          config.getUntrackedParameter<std::vector<edm::ESInputTag>>("mmap_pLS_neg"), [&](const edm::ESInputTag& tag) {
            return (edm::ESGetToken<SDL::ModuleConnectionMap, TrackerRecoGeometryRecord>)esConsumes(tag);
          });
    }

    void acquire(device::Event const& event, device::EventSetup const& setup) override {
      // Inputs
      auto const& pixelSeeds = event.get(lstPixelSeedInputToken_);
      auto const& phase2OTHits = event.get(lstPhase2OTHitsInputToken_);

      auto const& mmap = setup.getData(mmapToken_);
      std::vector<SDL::ModuleConnectionMap> moduleConnectionMap_pLStoLayer, moduleConnectionMap_pLStoLayer_pos,
          moduleConnectionMap_pLStoLayer_neg;
      for (size_t iMCM = 0; iMCM < mmap_pLSTokens_.size(); ++iMCM) {
        moduleConnectionMap_pLStoLayer.push_back(setup.getData(mmap_pLSTokens_[iMCM]));
        moduleConnectionMap_pLStoLayer_pos.push_back(setup.getData(mmap_pLS_posTokens_[iMCM]));
        moduleConnectionMap_pLStoLayer_neg.push_back(setup.getData(mmap_pLS_negTokens_[iMCM]));
      }
      lst_.eventSetup(
          mmap, moduleConnectionMap_pLStoLayer, moduleConnectionMap_pLStoLayer_pos, moduleConnectionMap_pLStoLayer_neg);
      lst_.run(event.queue(),
               verbose_,
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
               pixelSeeds.hitIdx(),
               phase2OTHits.detId(),
               phase2OTHits.x(),
               phase2OTHits.y(),
               phase2OTHits.z());
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      // Output
      LSTOutput lstOutput;
      lstOutput.setLSTOutputTraits(lst_.hits(), lst_.len(), lst_.seedIdx(), lst_.trackCandidateType());

      event.emplace(lstOutputToken_, std::move(lstOutput));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("pixelSeedInput", edm::InputTag{"lstPixelSeedInputProducer"});
      desc.add<edm::InputTag>("phase2OTHitsInput", edm::InputTag{"lstPhase2OTHitsInputProducer"});
      desc.add<edm::ESInputTag>("mmap", edm::ESInputTag("", ""));
      desc.addUntracked<std::vector<edm::ESInputTag>>(
          "mmap_pLS",
          std::vector<edm::ESInputTag>{edm::ESInputTag("", "pLS_layer1_subdet5"),
                                       edm::ESInputTag("", "pLS_layer2_subdet5"),
                                       edm::ESInputTag("", "pLS_layer1_subdet4"),
                                       edm::ESInputTag("", "pLS_layer2_subdet4")});
      desc.addUntracked<std::vector<edm::ESInputTag>>(
          "mmap_pLS_pos",
          std::vector<edm::ESInputTag>{edm::ESInputTag("", "pLS_layer1_subdet5_pos"),
                                       edm::ESInputTag("", "pLS_layer2_subdet5_pos"),
                                       edm::ESInputTag("", "pLS_layer1_subdet4_pos"),
                                       edm::ESInputTag("", "pLS_layer2_subdet4_pos")});
      desc.addUntracked<std::vector<edm::ESInputTag>>(
          "mmap_pLS_neg",
          std::vector<edm::ESInputTag>{edm::ESInputTag("", "pLS_layer1_subdet5_neg"),
                                       edm::ESInputTag("", "pLS_layer2_subdet5_neg"),
                                       edm::ESInputTag("", "pLS_layer1_subdet4_neg"),
                                       edm::ESInputTag("", "pLS_layer2_subdet4_neg")});
      desc.add<int>("verbose", 0);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<LSTPixelSeedInput> lstPixelSeedInputToken_;
    edm::EDGetTokenT<LSTPhase2OTHitsInput> lstPhase2OTHitsInputToken_;
    const edm::ESGetToken<SDL::ModuleConnectionMap, TrackerRecoGeometryRecord> mmapToken_;
    std::vector<edm::ESGetToken<SDL::ModuleConnectionMap, TrackerRecoGeometryRecord>> mmap_pLSTokens_,
        mmap_pLS_posTokens_, mmap_pLS_negTokens_;
    const int verbose_;
    edm::EDPutTokenT<LSTOutput> lstOutputToken_;

    SDL::LST lst_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTProducer);
