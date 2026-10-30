#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaOTRecHitsKernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaOTRecHitsKernels_h

// Device kernels of MkFitAlpakaOTRecHitsProducer: expansion of the stored OT clusters (one thread per detset) and
// Phase2StripCPE (one thread per cluster).
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTRecHitSoA.h"

namespace mkfitdev::othits {

  // one non-empty OT cluster detset
  struct OTDetSetSpan {
    uint32_t first;      // first cluster key
    uint32_t size;       // number of clusters
    int32_t tableIndex;  // index in the per-module table (GeomDet index - firstIndex)
  };

  // A Phase2TrackerCluster1D as stored (Phase2TrackerDigi channel, then size | threshold << 15) -> the SoA's clustSize
  // and strip = firstStrip() | column() << 16 (Phase2TrackerDigi::channelToRow / channelToColumn). The producer checks
  // this decode against the accessors on its first event.
  ALPAKA_FN_HOST_ACC inline void decodeCluster(uint16_t channel, uint16_t data, uint16_t& size, uint32_t& strip) {
    size = data & 0x7fffu;
    strip = (channel & 0x03ffu) | (((channel >> 10) & 0x1fu) << ::mkfitdev::kOTColumnShift);
  }

}  // namespace mkfitdev::othits

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits {

  // fills nHits and module, clustSize, strip of rows [0, nClusters) from the detset spans and the raw clusters (two
  // uint16 per cluster key), one thread per span
  void runOTExpand(Queue& queue,
                   ::mkfitdev::OTRecHitSoA::View view,
                   ::mkfitdev::othits::OTDetSetSpan const* spans,
                   uint32_t nSpans,
                   uint16_t const* raw,
                   ::mkfitdev::OTCpeModule const* table,
                   int32_t firstIndex,
                   uint32_t nClusters);

  // fills lx, ly, exx, eyy of rows [0, nClusters) from module, clustSize, strip and the per-module table
  // (index = module - firstIndex)
  void runOTCpe(Queue& queue,
                ::mkfitdev::OTRecHitSoA::View view,
                ::mkfitdev::OTCpeModule const* table,
                int32_t firstIndex,
                uint32_t nClusters);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits

#endif
