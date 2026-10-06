#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaEventOfHitsModuleTable_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaEventOfHitsModuleTable_h

// Host builder of the device hit input's per-module table (rotation, position, mkFit layer, uniqueIdInLayer per
// GeomDet index), shared by MkFitAlpakaEventOfHitsProducer (module-local per-IOV table) and
// MkFitAlpakaEventOfHitsModuleTableESProducer.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitInput.h"

namespace mkfitdev {

  inline std::vector<HitModuleDev> buildHitModuleTable(TrackerGeometry const& geom, MkFitGeometry const& mkg) {
    uint32_t n = 0;
    for (auto const* d : geom.detUnits())
      n = std::max<uint32_t>(n, d->index() + 1);
    std::vector<HitModuleDev> t(n, HitModuleDev{{0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0}, -1, 0});
    for (auto const* d : geom.detUnits()) {
      const auto& sf = d->surface();
      const auto& R = sf.rotation();
      HitModuleDev m{{R.xx(), R.xy(), R.xz(), R.yx(), R.yy(), R.yz(), R.zx(), R.zy(), R.zz()},
                     {sf.position().x(), sf.position().y(), sf.position().z()},
                     -1,
                     0};
      try {
        m.layer = mkg.mkFitLayerNumber(d->geographicalId());
        m.detIdInLayer = mkg.uniqueIdInLayer(m.layer, d->geographicalId().rawId());
      } catch (std::out_of_range const&) {
        m.layer = -1;
        m.detIdInLayer = 0;
      }
      t[d->index()] = m;
    }
    return t;
  }

}  // namespace mkfitdev

#endif
