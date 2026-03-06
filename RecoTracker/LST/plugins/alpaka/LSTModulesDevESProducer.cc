#include <sstream>

// LST includes
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"
#include "RecoTracker/LSTGeometry/interface/Geometry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class LSTModulesDevESProducer : public ESProducer {
  private:
    double ptCut_;
    edm::ESGetToken<lstgeometry::Geometry, TrackerRecoGeometryRecord> lstGeoToken_;

  public:
    LSTModulesDevESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig), ptCut_(iConfig.getParameter<double>("ptCut")) {
      std::ostringstream ptCutOSS;
      ptCutOSS << std::setprecision(1) << ptCut_;
      std::string ptCutStr = ptCutOSS.str();

      auto cc = setWhatProduced(this, "LSTModuleMaps_" + ptCutStr);
      lstGeoToken_ = cc.consumes<lstgeometry::Geometry>(edm::ESInputTag("", "LSTGeometry_" + ptCutStr));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<double>("ptCut", 0.8);
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<lst::LSTESData<DevHost>> produce(TrackerRecoGeometryRecord const& iRecord) {
      const auto& lstg = iRecord.get(lstGeoToken_);
      return lst::fillESDataHost(lstg);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(LSTModulesDevESProducer);
