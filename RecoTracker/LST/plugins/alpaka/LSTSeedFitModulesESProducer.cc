#include <optional>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/LST/interface/LSTSeedFitModulesHost.h"
#include "RecoTracker/LST/interface/alpaka/LSTSeedFitModulesCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The surface of every tracker module takes the local hit errors of the pixel tracks
  // to the global frame in the device LST input
  class LSTSeedFitModulesESProducer : public ESProducer {
  public:
    LSTSeedFitModulesESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      geometryToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<::lst::LSTSeedFitModulesHost> produce(TrackerDigiGeometryRecord const& iRecord) {
      auto const& detUnits = iRecord.get(geometryToken_).detUnits();
      ::lst::LSTSeedFitModulesHost product(cms::alpakatools::host(), detUnits.size());
      auto view = product.view();
      for (auto const* detUnit : detUnits) {
        auto const& surface = detUnit->surface();
        auto module = view[detUnit->index()];
        module.frame() = SOAFrame<float>(surface.position().x(),
                                         surface.position().y(),
                                         surface.position().z(),
                                         SOARotation<float>(surface.rotation()));
      }
      return product;
    }

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(LSTSeedFitModulesESProducer);
