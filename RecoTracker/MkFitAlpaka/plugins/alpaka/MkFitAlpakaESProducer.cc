// ESProducer of the MkFitAlpaka device-resident ES data (geometry, material, iteration configuration),
// filled from MkFitCore host ES products (MkFitGeometry, IterationConfig). The framework copies the
// host product to each device.

#include <string>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkFitAlpakaESProducer : public ESProducer {
  public:
    MkFitAlpakaESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
      geomToken_ = cc.consumes();
      iterConfigToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("iterationConfig"));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName", "")->setComment("Product label");
      desc.add<edm::ESInputTag>("iterationConfig", edm::ESInputTag("", "hltInitialStepTrackCandidatesMkFitConfig"))
          ->setComment("mkfit::IterationConfig (MkFitIterationConfigESProducer ComponentName)");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<mkfitdev::ESDataHost> produce(TrackerRecoGeometryRecord const& iRecord) {
      // MkFitGeometryESProducer sets the runtime mkfit::Config flags (usePropToPlane, usePtMultScat) when it
      // runs, i.e. before this get returns.
      auto const& geom = iRecord.get(geomToken_);
      auto const& iterConfig = iRecord.get(iterConfigToken_);
      auto data = mkfitdev::fillESDataHost(geom.trackerInfo(), iterConfig);
      // throw outside the configuration envelope the device chain implements
      mkfitdev::checkSupportedConfig(data->hostConfigValue(), data->layers->const_view(), data->sizes.nLayers);
      return data;
    }

  private:
    edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> geomToken_;
    edm::ESGetToken<mkfit::IterationConfig, TrackerRecoGeometryRecord> iterConfigToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(MkFitAlpakaESProducer);
