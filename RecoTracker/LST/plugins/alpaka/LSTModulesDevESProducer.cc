#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// LST includes
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTModulesDevESProducer : public ESProducer {
  public:
    LSTModulesDevESProducer(const edm::ParameterSet& iConfig);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    std::unique_ptr<SDL::LSTESHostData<SDL::Dev>> produceHost(TrackerRecoGeometryRecord const& iRecord);
    std::unique_ptr<SDL::LSTESDeviceData<SDL::Dev>> produceDevice(
        device::Record<TrackerRecoGeometryRecord> const& iRecord);

  private:
    edm::ESGetToken<SDL::LSTESHostData<SDL::Dev>, TrackerRecoGeometryRecord> lstESHostToken_;
    std::string ptCutLabel_;
  };

  LSTModulesDevESProducer::LSTModulesDevESProducer(const edm::ParameterSet& iConfig)
      : ESProducer(iConfig), ptCutLabel_(iConfig.getParameter<std::string>("ptCutLabel")) {
    setWhatProduced(this, &LSTModulesDevESProducer::produceHost, ptCutLabel_);
    auto cc = setWhatProduced(this, &LSTModulesDevESProducer::produceDevice, ptCutLabel_);
    lstESHostToken_ = cc.consumes(edm::ESInputTag("", ptCutLabel_));
  }

  void LSTModulesDevESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("ptCutLabel", "0.8");
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<SDL::LSTESHostData<SDL::Dev>> LSTModulesDevESProducer::produceHost(
      TrackerRecoGeometryRecord const& iRecord) {
    return SDL::loadAndFillESHost(ptCutLabel_);
  }

  std::unique_ptr<SDL::LSTESDeviceData<SDL::Dev>> LSTModulesDevESProducer::produceDevice(
      device::Record<TrackerRecoGeometryRecord> const& iRecord) {
    auto const& lstESHostData = iRecord.get(lstESHostToken_);
    SDL::QueueAcc& queue = iRecord.queue();
    return SDL::loadAndFillESDevice(queue, &lstESHostData, ptCutLabel_);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(LSTModulesDevESProducer);
