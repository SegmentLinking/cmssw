#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// LST includes
#include "SDL/ModuleConnectionMap.h"

class LSTModuleConnectionMapESProducer : public edm::ESProducer {
public:
  LSTModuleConnectionMapESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<SDL::ModuleConnectionMap> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  const std::string txtFile_;
};

LSTModuleConnectionMapESProducer::LSTModuleConnectionMapESProducer(const edm::ParameterSet &iConfig)
    : txtFile_{iConfig.getParameter<edm::FileInPath>("txt").fullPath()} {
  setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
}

void LSTModuleConnectionMapESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "")->setComment("Product label");
  desc.add<edm::FileInPath>("txt", edm::FileInPath())->setComment("Path to the txt file for the module map parameters");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<SDL::ModuleConnectionMap> LSTModuleConnectionMapESProducer::produce(
    const TrackerRecoGeometryRecord &iRecord) {
  return std::make_unique<SDL::ModuleConnectionMap>(txtFile_);
}

DEFINE_FWK_EVENTSETUP_MODULE(LSTModuleConnectionMapESProducer);
