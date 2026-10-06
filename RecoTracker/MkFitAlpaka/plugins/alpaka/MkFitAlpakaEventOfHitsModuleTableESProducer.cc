// ESProducer of the device hit input's per-module table:
// built once per IOV of TrackerRecoGeometryRecord (TrackerGeometry + MkFitGeometry), host product + one device copy per
// device. MkFitAlpakaEventOfHitsProducer reads it when its 'moduleTable' parameter names it.

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFitAlpaka/interface/hits/HitModuleTableESData.h"
#include "RecoTracker/MkFitAlpaka/plugins/alpaka/MkFitAlpakaEventOfHitsModuleTable.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkFitAlpakaEventOfHitsModuleTableESProducer : public ESProducer {
  public:
    MkFitAlpakaEventOfHitsModuleTableESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
      geomToken_ = cc.consumes();
      mkFitGeomToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName", "MkFitAlpakaHitModuleTable");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<::mkfitdev::HitModuleTableESDataHost> produce(TrackerRecoGeometryRecord const& iRecord) {
      return ::mkfitdev::makeHitModuleTableESDataHost(
          ::mkfitdev::buildHitModuleTable(iRecord.get(geomToken_), iRecord.get(mkFitGeomToken_)));
    }

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(MkFitAlpakaEventOfHitsModuleTableESProducer);
