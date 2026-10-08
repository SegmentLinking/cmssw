#include <optional>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEParamsHost.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/alpaka/Phase2StripCPEParamsCollection.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The per-module parameters of a Phase2StripCPE, copied from the CPE itself, and the module surfaces, for the device
  class Phase2StripCPEParamsESProducerAlpaka : public ESProducer {
  public:
    Phase2StripCPEParamsESProducerAlpaka(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      cpeToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("Phase2StripCPE"));
      geometryToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"))
          ->setComment("the Phase2StripCPE whose parameters are copied");
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<Phase2StripCPEParamsHost> produce(TkPhase2OTCPERecord const& iRecord) {
      auto const* cpe = dynamic_cast<Phase2StripCPE const*>(&iRecord.get(cpeToken_));
      if (cpe == nullptr)
        throw cms::Exception("Configuration") << "Phase2StripCPEParamsESProducer: the CPE is not a Phase2StripCPE";
      auto const& detUnits = iRecord.get(geometryToken_).detUnits();
      const unsigned int firstModuleIndex = cpe->firstModuleIndex();

      Phase2StripCPEParamsHost product(cms::alpakatools::host(), detUnits.size() - firstModuleIndex);
      auto view = product.view();
      view.firstModuleIndex() = firstModuleIndex;
      for (unsigned int index = firstModuleIndex; index < detUnits.size(); ++index) {
        auto const& param = cpe->moduleParam(index);
        auto const& surface = detUnits[index]->surface();
        auto module = view[index - firstModuleIndex];
        module.position() = param.position;
        module.xerrLocal() = param.localErr.xx();
        module.yerrLocal() = param.localErr.yy();
        module.frame() = SOAFrame<float>(surface.position().x(),
                                         surface.position().y(),
                                         surface.position().z(),
                                         SOARotation<float>(surface.rotation()));
        module.detId() = detUnits[index]->geographicalId().rawId();
      }
      return product;
    }

  private:
    edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> cpeToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(Phase2StripCPEParamsESProducerAlpaka);
