#include <alpaka/alpaka.hpp>

#include "RecoTracker/LSTCore/interface/alpaka/LST.h"
#include "RecoTracker/LSTGeometry/interface/Common.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStatesDeviceCollection.h"
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
          produceTCSeedStates_(config.getParameter<bool>("produceTCSeedStates")),
          lstInputToken_{consumes(config.getParameter<edm::InputTag>("lstInput"))},
          lstESToken_{esConsumes(edm::ESInputTag("", ptCutStr_))},
          lstOutputToken_{produces()},
          tcSeedStatesToken_(produceTCSeedStates_ ? device::EDPutToken<lst::TCSeedStatesDeviceCollection>{produces()}
                                                  : device::EDPutToken<lst::TCSeedStatesDeviceCollection>{}) {}

    void produce(edm::StreamID sid, device::Event& iEvent, const device::EventSetup& iSetup) const override {
      lst::LST lst;
      // Inputs
      auto const& lstInputDC = iEvent.get(lstInputToken_);
      auto const& lstESDeviceData = iSetup.getData(lstESToken_);

      lst.run(iEvent.queue(),
              verbose_,
              ptCut_,
              clustSizeCut_,
              &lstESDeviceData,
              &lstInputDC,
              nopLSDupClean_,
              tcpLSTriplets_,
              produceTCSeedStates_);

      // Output
      auto lstTrackCandidates = lst.getTrackCandidates();
      iEvent.emplace(lstOutputToken_, std::move(*lstTrackCandidates.release()));
      if (produceTCSeedStates_) {
        auto tcSeedStates = lst.getTCSeedStates();
        if (!tcSeedStates) {
          throw edm::Exception(edm::errors::LogicError)
              << "LST did not return the track candidate seed states although they were "
                 "requested (produceTCSeedStates is true)";
        }
        iEvent.emplace(tcSeedStatesToken_, std::move(*tcSeedStates));
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
      // Also produce the Broken Line Fit seed state of every track candidate (lst::TCSeedStatesDeviceCollection).
      // Needs the LST input filled on the device (hltInputLST), not the host lstInputProducer.
      desc.add<bool>("produceTCSeedStates", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const bool verbose_;
    const double ptCut_;
    const std::string ptCutStr_;
    const uint16_t clustSizeCut_;
    const bool nopLSDupClean_;
    const bool tcpLSTriplets_;
    const bool produceTCSeedStates_;
    const device::EDGetToken<lst::LSTInputDeviceCollection> lstInputToken_;
    const device::ESGetToken<lst::LSTESData<Device>, TrackerRecoGeometryRecord> lstESToken_;
    const device::EDPutToken<lst::TrackCandidatesBaseDeviceCollection> lstOutputToken_;
    const device::EDPutToken<lst::TCSeedStatesDeviceCollection> tcSeedStatesToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LSTProducer);
