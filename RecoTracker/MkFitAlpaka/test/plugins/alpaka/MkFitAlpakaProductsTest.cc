// integ: test producer of the MkFitAlpaka event products (device EventOfHits, TrackSoA, candidate storage).
// Fills them with known values on the device (and runs the MPlex <-> SoA packers there); the framework copies them
// to the host for MkFitAlpakaProductsCheck (test/integ_products_cfg.py). Exercises dictionaries + CopyToHost.

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/test/plugins/alpaka/MkFitProductsTestKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  class MkFitAlpakaProductsTest : public global::EDProducer<> {
  public:
    explicit MkFitAlpakaProductsTest(edm::ParameterSet const& ps)
        : EDProducer<>(ps), eohToken_{produces()}, trkToken_{produces()}, candToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const&) const override {
      using namespace ::mkfitdev::productstest;
      auto& queue = iEvent.queue();
      EventOfHitsDeviceCollection eoh(queue, kHits, kLayers, kHits, kBins);
      TrackSoADeviceCollection trk(queue, kTracks);
      CandStoreDeviceCollection cand(queue, ::mkfitdev::candStoreSizes(kSeeds, kHotsPerSeed));
      fillProductsTest(queue, eoh, trk, cand, static_cast<uint32_t>(iEvent.id().event()));
      iEvent.emplace(eohToken_, std::move(eoh));
      iEvent.emplace(trkToken_, std::move(trk));
      iEvent.emplace(candToken_, std::move(cand));
    }

  private:
    const device::EDPutToken<EventOfHitsDeviceCollection> eohToken_;
    const device::EDPutToken<TrackSoADeviceCollection> trkToken_;
    const device::EDPutToken<CandStoreDeviceCollection> candToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

DEFINE_FWK_ALPAKA_MODULE(mkfitdev::MkFitAlpakaProductsTest);
