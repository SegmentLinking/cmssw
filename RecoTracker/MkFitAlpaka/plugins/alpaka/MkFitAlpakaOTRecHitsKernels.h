#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaOTRecHitsKernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaOTRecHitsKernels_h

// Device Phase2StripCPE kernel, one thread per OT cluster.
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTRecHitSoA.h"

namespace mkfitdev::othits {

  // one non-empty OT cluster detset: first cluster key, number of clusters, table index (GeomDet index - firstIndex)
  struct OTDetSetSpan {
    uint32_t first, size;
    int32_t mi;
  };

  // A Phase2TrackerCluster1D as stored (Phase2TrackerDigi channel, then size | threshold << 15) -> the SoA's clustSize
  // and strip = firstStrip() | column() << 16 (Phase2TrackerDigi::channelToRow / channelToColumn). The producer checks
  // this decode against the accessors on its first event.
  ALPAKA_FN_HOST_ACC inline void decodeCluster(uint16_t channel, uint16_t data, uint16_t& size, uint32_t& strip) {
    size = data & 0x7fffu;
    strip = (channel & 0x03ffu) | (((channel >> 10) & 0x1fu) << 16);
  }

}  // namespace mkfitdev::othits

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits {

  // fills nHits and module, detId, clustSize, strip of rows [0, n) from the detset spans and the raw clusters (two
  // uint16 per cluster key), one thread per span; detId from the per-module table
  void runOTExpand(Queue& queue,
                   ::mkfitdev::OTRecHitSoA::View view,
                   ::mkfitdev::othits::OTDetSetSpan const* spans,
                   uint32_t nSpans,
                   uint16_t const* raw,
                   ::mkfitdev::OTCpeModule const* table,
                   int32_t firstIndex,
                   uint32_t n);

  // fills lx, ly, exx, eyy, gx, gy, gz of rows [0, n) from module, clustSize, strip and the per-module table
  // (index = module - firstIndex)
  void runOTCpe(Queue& queue,
                ::mkfitdev::OTRecHitSoA::View view,
                ::mkfitdev::OTCpeModule const* table,
                int32_t firstIndex,
                uint32_t n);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits

#endif
