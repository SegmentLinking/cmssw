// The pixel-track counts of the device pixel tracks (Patatrack TracksSoA) as a host product, without the legacy pixel
// tracks: the SoA's nTracks and the number of tracks that PixelTrackProducerFromSoAAlpaka converts (quality >=
// minQuality, a finite converted track; its minNumberOfHits <= 3 and no requireQuadsFromConsecutiveLayers).
// Product: std::vector<unsigned int> {nTracks, nSelected}. The device LST input (MkFitAlpakaLstInputProducer,
// pixelTrackCount) sizes its per-track buffers and its pixel seed states with them and numbers the selected tracks on the
// device as the legacy collection does (descending pT).
#include <Eigen/Core>  // before any SoA header (Eigen columns of TracksSoA)
#include <optional>
#include <string>
#include <vector>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "MkFitAlpakaLstInputKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkFitAlpakaPixelTrackCountProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit MkFitAlpakaPixelTrackCountProducer(edm::ParameterSet const& cfg)
        : SynchronizingEDProducer<>(cfg),
          tracksToken_(consumes(cfg.getParameter<edm::InputTag>("pixelTracksSoA"))),
          minQuality_(static_cast<int>(pixelTrack::qualityByName(cfg.getParameter<std::string>("minQuality")))),
          putToken_(produces()) {
      if (minQuality_ == static_cast<int>(pixelTrack::Quality::notQuality))
        throw cms::Exception("Configuration")
            << cfg.getParameter<std::string>("minQuality") << " is not a pixelTrack::Quality";
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("pixelTracksSoA", edm::InputTag("hltPhase2PixelTrackTorchHighPuritySelector"));
      desc.add<std::string>("minQuality", "tight")
          ->setComment("as PixelTrackProducerFromSoAAlpaka (and the LST input's pixel-track quality cut)");
      descriptions.addWithDefaultLabel(desc);
    }

    void acquire(device::Event const& iEvent, device::EventSetup const&) override {
      auto& queue = iEvent.queue();
      auto const view = iEvent.get(tracksToken_).const_view();
      dCnt_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2));
      alpaka::memset(queue, *dCnt_, 0);
      ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin::launchPixelTrackCount(
          queue, view, uint32_t(view.tracks().metadata().size()), minQuality_, dCnt_->data());
      hCnt_.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 2));
      alpaka::memcpy(queue, *hCnt_, *dCnt_);
    }

    void produce(device::Event& iEvent, device::EventSetup const&) override {
      iEvent.emplace(putToken_, std::vector<unsigned int>{hCnt_->data()[0], hCnt_->data()[1]});
      dCnt_.reset();
      hCnt_.reset();
    }

  private:
    const device::EDGetToken<reco::TracksSoACollection> tracksToken_;
    const int minQuality_;
    const edm::EDPutTokenT<std::vector<unsigned int>> putToken_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> dCnt_;
    std::optional<cms::alpakatools::host_buffer<uint32_t[]>> hCnt_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaPixelTrackCountProducer);
