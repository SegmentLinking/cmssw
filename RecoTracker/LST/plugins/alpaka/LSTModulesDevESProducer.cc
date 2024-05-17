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
    LSTModulesDevESProducer(const edm::ParameterSet &iConfig);

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::optional<SDL::modulesBuffer<SDL::Dev>> produce(device::Record<TrackerRecoGeometryRecord> const& iRecord);

};

  LSTModulesDevESProducer::LSTModulesDevESProducer(const edm::ParameterSet &iConfig) : ESProducer(iConfig) {
    setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
  }

  void LSTModulesDevESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("ComponentName", "")->setComment("Product label");
    descriptions.addWithDefaultLabel(desc);
  }


  std::optional<SDL::modulesBuffer<SDL::Dev>> LSTModulesDevESProducer::produce(device::Record<TrackerRecoGeometryRecord> const& iRecord) {
    SDL::QueueAcc& queue = iRecord.queue();
    SDL::modulesBuffer<SDL::Dev> modules(alpaka::getDev(queue));
    SDL::LST<SDL::Acc>::loadAndFillES(queue, &modules);
    return modules;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(LSTModulesDevESProducer);
