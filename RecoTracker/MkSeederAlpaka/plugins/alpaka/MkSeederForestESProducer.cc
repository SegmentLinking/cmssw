#include <memory>
#include <string>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederForest.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederForestDeviceCollection.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedFeaturesSoA.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The seed classifier's trees (MkSeederForestSoA.h) from the xgboost model's JSON file, once per IOV (the framework
  // copies them to the device once); the model must take the seeder's features (MkSeederSeedFeaturesSoA.h).
  class MkSeederForestESProducer : public ESProducer {
  public:
    MkSeederForestESProducer(edm::ParameterSet const &iConfig)
        : ESProducer(iConfig), path_{iConfig.getParameter<edm::FileInPath>("model").fullPath()} {
      setWhatProduced(this);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model",
                                edm::FileInPath("RecoTracker/MkFit/data/mkfit-phase2-seed-classifier-6l-v3-n100.json"))
          ->setComment("The xgboost model (Booster.save_model() JSON)");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<::mkseeder::MkSeederForestHostCollection> produce(TrackerRecoGeometryRecord const &) {
      auto forest = std::make_unique<::mkseeder::MkSeederForestHostCollection>(::mkseeder::readMkSeederForest(path_));
      // the model takes all the features or only the base ones (without the PS-module features)
      if (const int n = forest->const_view().nodes().nFeatures();
          n != ::mkseeder::seedfeature::kNFeatures && n != ::mkseeder::seedfeature::kNBaseFeatures)
        throw cms::Exception("Configuration")
            << "MkSeederForestESProducer: the model " << path_ << " takes " << n << " features, the seeds have "
            << ::mkseeder::seedfeature::kNFeatures << " (or " << ::mkseeder::seedfeature::kNBaseFeatures
            << " without the PS ones)";
      return forest;
    }

  private:
    const std::string path_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(MkSeederForestESProducer);
