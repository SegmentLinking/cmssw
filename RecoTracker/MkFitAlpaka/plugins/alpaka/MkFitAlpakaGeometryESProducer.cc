// ESProducer of the MkFitAlpaka geometry-only ES product (layers, module planes and shapes, detId map, material map,
// field constants; no iteration configuration), filled from the MkFitCore MkFitGeometry. The iteration's ES product
// (MkFitAlpakaESProducer) is built on it; other device consumers of the mkFit geometry can read it directly. The
// framework copies the host product to each device.

#include <string>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFitAlpaka/interface/es/ESGeometryData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkFitAlpakaGeometryESProducer : public ESProducer {
  public:
    MkFitAlpakaGeometryESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
      geomToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName", "")->setComment("Product label");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<mkfitdev::ESGeometryDataHost> produce(TrackerRecoGeometryRecord const& iRecord) {
      // MkFitGeometryESProducer sets the runtime mkfit::Config field constants when it runs, i.e. before this get
      // returns
      return mkfitdev::fillESGeometryHost(iRecord.get(geomToken_).trackerInfo());
    }

  private:
    edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> geomToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(MkFitAlpakaGeometryESProducer);
