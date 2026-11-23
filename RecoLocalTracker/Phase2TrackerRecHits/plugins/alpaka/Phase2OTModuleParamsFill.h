#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTModuleParamsFill_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTModuleParamsFill_h

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEModuleParams.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTHits {

  // The Phase2StripCPE parameters, copied from the CPE itself, the local position errors and the surface of the OT
  // module detUnit, in a row (cpe, xerrLocal, yerrLocal, frame) of a module parameter SoA
  template <typename ModuleRow>
  void fillModuleParams(ModuleRow&& module, GeomDetUnit const& detUnit, Phase2StripCPE const& cpe) {
    auto const& det = static_cast<PixelGeomDetUnit const&>(detUnit);
    // the module's own topology, not the proxy that Phase2StripCPE::Param::topology points to
    auto const* topology = dynamic_cast<RectangularPixelPhase2Topology const*>(&det.specificType().specificTopology());
    if (topology == nullptr)
      throw cms::Exception("BadGeometry")
          << "module " << det.geographicalId().rawId() << " does not have a RectangularPixelPhase2Topology";
    auto const& param = cpe.moduleParam(det.index());
    auto const& surface = det.surface();
    module.cpe() = phase2StripCPE::moduleParams(*topology, param.coveredStrips);
    module.xerrLocal() = param.localErr.xx();
    module.yerrLocal() = param.localErr.yy();
    module.frame() = SOAFrame<float>(
        surface.position().x(), surface.position().y(), surface.position().z(), SOARotation<float>(surface.rotation()));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTModuleParamsFill_h
