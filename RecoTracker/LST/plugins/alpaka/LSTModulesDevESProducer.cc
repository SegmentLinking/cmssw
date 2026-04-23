// LST includes
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"
#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Geometry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTModulesDevESProducer : public ESProducer {
  private:
    double ptCut_;
    edm::ESGetToken<lstgeometry::Geometry, TrackerRecoGeometryRecord> lstGeoToken_;

  public:
    LSTModulesDevESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig), ptCut_(iConfig.getParameter<double>("ptCut")) {
      std::string ptCutStr = lst::floatToStr(ptCut_, 1);

      auto cc = setWhatProduced(this, ptCutStr);
      lstGeoToken_ = cc.consumes<lstgeometry::Geometry>(edm::ESInputTag("", ptCutStr));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<double>("ptCut", 0.8);
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<lst::LSTESData<DevHost>> produce(TrackerRecoGeometryRecord const& iRecord) {
      // const auto& lstg = iRecord.get(lstGeoToken_);
      // auto lstESData = lst::fillESDataHost(lstg);
      // std::cout << "Using LSTGeometry ES product" << std::endl;

      auto ptLabel = lst::floatToStr(ptCut_, 1);
      auto lstESData = lst::loadAndFillESDataHost(ptLabel);
      std::cout << "Using LSTESData from file with ptCut " << ptLabel << std::endl;
      
      float modulesSize = alpaka::getExtentProduct(lstESData->modules->buffer()) / 1e6;
      float endcapSize = alpaka::getExtentProduct(lstESData->endcapGeometry->buffer()) / 1e6;
      std::cout
          << "LSTESData for ptCut " << ptCut_ << " will use " << (modulesSize + endcapSize) << " MB of VRAM\n"
          << "(Modules: " << modulesSize << " MB, EndcapGeometry: " << endcapSize << " MB)";
      
      return lstESData;
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(LSTModulesDevESProducer);
