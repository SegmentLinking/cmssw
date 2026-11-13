#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaEventOfHitsModuleTable_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaEventOfHitsModuleTable_h

// Host builder of the device hit input's per-module table (rotation, position, mkFit layer, uniqueIdInLayer per
// GeomDet index), for MkFitAlpakaEventOfHitsModuleTableESProducer.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitInput.h"

namespace mkfitdev {

  inline std::vector<HitModuleDev> buildHitModuleTable(TrackerGeometry const& geom, MkFitGeometry const& mkFitGeom) {
    uint32_t nModules = 0;
    for (auto const* det : geom.detUnits())
      nModules = std::max<uint32_t>(nModules, det->index() + 1);
    std::vector<HitModuleDev> table(nModules, HitModuleDev{{0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0}, -1, 0});
    uint32_t nOutside = 0;
    for (auto const* det : geom.detUnits()) {
      const auto& surface = det->surface();
      const auto& rot = surface.rotation();
      HitModuleDev entry{{rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz()},
                         {surface.position().x(), surface.position().y(), surface.position().z()},
                         -1,
                         0};
      // modules outside the mkFit layers (layer -1, or the MkFitGeometry lookups throw) keep layer -1: their hits
      // are not registered, where the host converter would throw, so they are reported
      try {
        entry.layer = mkFitGeom.mkFitLayerNumber(det->geographicalId());
        if (entry.layer >= 0)
          entry.detIdInLayer = mkFitGeom.uniqueIdInLayer(entry.layer, det->geographicalId().rawId());
      } catch (std::out_of_range const&) {
        entry.layer = -1;
      } catch (std::runtime_error const&) {
        entry.layer = -1;
      }
      if (entry.layer < 0) {
        entry.detIdInLayer = 0;
        ++nOutside;
      }
      table[det->index()] = entry;
    }
    if (nOutside > 0)
      edm::LogWarning("MkFitAlpakaEventOfHits")
          << nOutside << " tracker modules are outside the mkFit layers: their hits are dropped";
    return table;
  }

}  // namespace mkfitdev

#endif
