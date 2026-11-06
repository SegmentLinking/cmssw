#ifndef RecoTracker_MkFitAlpaka_interface_hits_DeviceHitRawSoA_h
#define RecoTracker_MkFitAlpaka_interface_hits_DeviceHitRawSoA_h

// Device hit input: per HitSoA row, what the host still has to provide (module-internal
// scratch, not an event product). Rows [0, nPixel): pixel, indexed by the legacy cluster key like the HitVec;
// srcRow = row of the pixel rechit SoA (moduleStart[det] + originalId), kNoHit for a cluster index without a hit;
// spans = legacy cluster sizes (sizeX | sizeY << 16). Rows [nPixel, nPixel + nStrip): OT, indexed by the OT cluster
// key; local position / error copied from the rechit, module = GeomDet index (-1: no hit), spans = cluster size.
#include <cstdint>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkfitdev {
  GENERATE_SOA_LAYOUT(DeviceHitRawSoALayout,
                      SOA_COLUMN(uint32_t, srcRow),
                      SOA_COLUMN(int32_t, module),
                      SOA_COLUMN(uint32_t, spans),
                      SOA_COLUMN(float, lx),
                      SOA_COLUMN(float, ly),
                      SOA_COLUMN(float, exx),
                      SOA_COLUMN(float, exy),
                      SOA_COLUMN(float, eyy))
  using DeviceHitRawSoA = DeviceHitRawSoALayout<>;
  using DeviceHitRawHostCollection = PortableHostCollection<DeviceHitRawSoA>;
  constexpr uint32_t kNoHit = 0xffffffffu;
}  // namespace mkfitdev

#endif
