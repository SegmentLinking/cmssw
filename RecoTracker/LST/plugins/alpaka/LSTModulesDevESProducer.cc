#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// LST includes
#include <RecoTracker/LSTCore/interface/Module.h>
#include <RecoTracker/LSTCore/interface/ModuleMethods.h>
#include <RecoTracker/LSTCore/interface/LST.h>

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<SDL::modulesBuffer<alpaka_common::DevHost>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, SDL::modulesBuffer<alpaka_common::DevHost> const& srcData) {
      return SDL::modulesBuffer<alpaka::Dev<TQueue>>(queue, srcData);
    }
  };
} //namespace cms::alpakatools
namespace ALPAKA_ACCELERATOR_NAMESPACE {

class LSTModulesDevESProducer : public ESProducer {
public:
  LSTModulesDevESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::optional<SDL::modulesBuffer<alpaka_common::DevHost>> produce(const TrackerRecoGeometryRecord &iRecord);

};

LSTModulesDevESProducer::LSTModulesDevESProducer(const edm::ParameterSet &iConfig)
  : ESProducer(iConfig)
{
  setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
}

void LSTModulesDevESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "")->setComment("Product label");
  descriptions.addWithDefaultLabel(desc);
}

  std::optional<SDL::modulesBuffer<alpaka_common::DevHost>> LSTModulesDevESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
    SDL::modulesBuffer<alpaka_common::DevHost> modules(cms::alpakatools::host());
    alpaka::QueueCpuBlocking queue(cms::alpakatools::host());
    SDL::LST::loadAndFillES(queue, &modules);
    return modules;
}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(LSTModulesDevESProducer);
