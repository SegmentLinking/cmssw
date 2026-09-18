#include <alpaka/alpaka.hpp>

#include "RecoTracker/LSTCore/interface/alpaka/LST.h"
#include "RecoTracker/LSTGeometry/interface/Common.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
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

#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTProducer : public global::EDProducer<> {
  public:
    LSTProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          verbose_(config.getParameter<bool>("verbose")),
          ptCut_(config.getParameter<double>("ptCut")),
          ptCutStr_(lst::floatToStr(ptCut_, 1)),
          clustSizeCut_(static_cast<uint16_t>(config.getParameter<uint32_t>("clustSizeCut"))),
          nopLSDupClean_(config.getParameter<bool>("nopLSDupClean")),
          tcpLSTriplets_(config.getParameter<bool>("tcpLSTriplets")),
          reduceMemByFullPrecompute_(config.getParameter<bool>("reduceMemByFullPrecompute")),
          produceBLFFit_(config.getParameter<bool>("produceBLFFit")),
          lstInputToken_{consumes(config.getParameter<edm::InputTag>("lstInput"))},
          lstESToken_{esConsumes(edm::ESInputTag("", ptCutStr_))},
          bFieldToken_(consumes<float>(config.getParameter<edm::InputTag>("bField"))),
          pixelHitsLoadedToken_(produceBLFFit_ ? consumes<bool>(config.getParameter<edm::InputTag>("pixelHitsLoaded"))
                                               : edm::EDGetTokenT<bool>{}),
          lstOutputToken_{produces()},
          lstBLFFitOutputToken_(produceBLFFit_
                                    ? device::EDPutToken<lst::TrackCandidatesBLFFitDeviceCollection>(produces())
                                    : device::EDPutToken<lst::TrackCandidatesBLFFitDeviceCollection>()) {}

    void produce(edm::StreamID sid, device::Event& iEvent, const device::EventSetup& iSetup) const override {
      lst::LST lst;
      // Inputs
      auto const& lstInputDC = iEvent.get(lstInputToken_);
      auto const& lstESDeviceData = iSetup.getData(lstESToken_);
      const float bField = iEvent.get(bFieldToken_);

#if LST_BLF_PIXEL_HITS
      // In a build that carries the pixel seed hits, running the fit without them
      // is a silent downgrade to the outer-tracker-only ==> Warn
      if (produceBLFFit_ && !iEvent.get(pixelHitsLoadedToken_)) {
        edm::LogWarning("LSTProducer")
            << "produceBLFFit is true but the pixel seed hit positions were not loaded into LSTInputSoA "
               "(produceSeedHitPositions is false on the LSTInputProducer feeding this module). The Broken Line "
               "Fit will run over the outer tracker hits only.";
      }
#endif

      lst.run(iEvent.queue(),
              verbose_,
              ptCut_,
              clustSizeCut_,
              &lstESDeviceData,
              &lstInputDC,
              nopLSDupClean_,
              tcpLSTriplets_,
              reduceMemByFullPrecompute_,
              produceBLFFit_,
              bField);

      // Output
      auto lstTrackCandidates = lst.getTrackCandidates();
      iEvent.emplace(lstOutputToken_, std::move(*lstTrackCandidates));
      if (produceBLFFit_) {
        auto lstBLFFit = lst.getTrackCandidatesBLFFit();
        iEvent.emplace(lstBLFFitOutputToken_, std::move(*lstBLFFit));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("lstInput", edm::InputTag{"lstInputProducer"});
      desc.add<bool>("verbose", false);
      desc.add<double>("ptCut", 0.8);
      desc.add<uint32_t>("clustSizeCut", 16);
      desc.add<bool>("nopLSDupClean", false);
      desc.add<bool>("tcpLSTriplets", false);
      desc.add<bool>("reduceMemByFullPrecompute", false)
          ->setComment(
              "If true, run extra counting kernels that exactly size the MD/LS/T3/T5/T4 "
              "buffers, reducing average per-event memory at a small CPU/GPU runtime cost. "
              "If false (default), buffers use cheaper, looser occupancy estimates.");
      desc.add<edm::InputTag>("bField", edm::InputTag{"lstInputProducer"});
      desc.add<bool>("produceBLFFit", false)
          ->setComment(
              "If true, run the Broken Line Fit over the LST track candidates and produce "
              "TrackCandidatesBLFFitDeviceCollection. If false (default), the fit is not run, none of its "
              "kernels are launched and the product is not produced. Setting this true while the "
              "LSTInputProducer named by 'pixelHitsLoaded' has produceSeedHitPositions false is allowed but "
              "makes the fit use the outer tracker hits only; a warning is issued per event in that case.");
      desc.add<edm::InputTag>("pixelHitsLoaded", edm::InputTag{"lstInputProducer"});
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const bool verbose_;
    const double ptCut_;
    const std::string ptCutStr_;
    const uint16_t clustSizeCut_;
    const bool nopLSDupClean_;
    const bool tcpLSTriplets_;
    const bool reduceMemByFullPrecompute_;
    const bool produceBLFFit_;
    const device::EDGetToken<lst::LSTInputDeviceCollection> lstInputToken_;
    const device::ESGetToken<lst::LSTESData<Device>, TrackerRecoGeometryRecord> lstESToken_;
    const edm::EDGetTokenT<float> bFieldToken_;
    const edm::EDGetTokenT<bool> pixelHitsLoadedToken_;
    const device::EDPutToken<lst::TrackCandidatesBaseDeviceCollection> lstOutputToken_;
    const device::EDPutToken<lst::TrackCandidatesBLFFitDeviceCollection> lstBLFFitOutputToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTProducer);
