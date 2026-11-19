#include <memory>
#include <string>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederConfig.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// The seeder's configuration (mkseeder::MkSeederConfig) from a JSON file
class MkSeederConfigESProducer : public edm::ESProducer {
public:
  explicit MkSeederConfigESProducer(const edm::ParameterSet &iConfig)
      : configFile_{iConfig.getParameter<edm::FileInPath>("config").fullPath()} {
    setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("ComponentName", "")->setComment("Product label");
    desc.add<edm::FileInPath>("config", edm::FileInPath())
        ->setComment("The JSON file of the seeder's configuration (mkseeder::MkSeederConfig)");
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<mkseeder::MkSeederConfig> produce(const TrackerRecoGeometryRecord &) {
    return std::make_unique<mkseeder::MkSeederConfig>(mkseeder::readMkSeederConfig(configFile_));
  }

private:
  const std::string configFile_;
};

DEFINE_FWK_EVENTSETUP_MODULE(MkSeederConfigESProducer);
