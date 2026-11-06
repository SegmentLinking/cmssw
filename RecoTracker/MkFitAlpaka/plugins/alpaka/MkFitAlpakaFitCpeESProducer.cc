// ESProducer of the device PixelCPEGeneric tables of the mkFit final fit: built once per IOV of
// PixelCPEFastParamsRecord (module constants of PixelCPEFastParamsPhase2, GenError DB object, geometry, B field), host
// product + one device copy per device (CopyToDevice in interface/fit/CpeESData.h): no per-module mutable tables (no
// IOV-change race, no 260 kB copy per event).

#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"

#include "RecoTracker/MkFitAlpaka/interface/fit/CpeESData.h"
#include "RecoTracker/MkFitAlpaka/plugins/alpaka/MkFitAlpakaFitCpeTables.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkFitAlpakaFitCpeESProducer : public ESProducer {
  public:
    MkFitAlpakaFitCpeESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
      paramsToken_ = cc.consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("cpeFastParams")));
      genErrToken_ = cc.consumes();
      geomToken_ = cc.consumes();
      mfToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName", "MkFitAlpakaFitCpe");
      desc.add<std::string>("cpeFastParams", "PixelCPEFastParamsPhase2")
          ->setComment("ComponentName of the PixelCPEFastParams ES product (per-module constants)");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<::mkfitdev::cpe::CpeESDataHost> produce(PixelCPEFastParamsRecord const& iRecord) {
      auto t = std::make_shared<::mkfitdev::cpe::CpeTablesHost>();
      ::mkfitdev::cpe::buildCpeTables(
          iRecord.get(paramsToken_), iRecord.get(genErrToken_), iRecord.get(geomToken_), iRecord.get(mfToken_), *t);
      return ::mkfitdev::cpe::makeCpeESDataHost(std::move(t));
    }

  private:
    edm::ESGetToken<PixelCPEFastParamsHost<pixelTopology::Phase2>, PixelCPEFastParamsRecord> paramsToken_;
    edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> genErrToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(MkFitAlpakaFitCpeESProducer);
