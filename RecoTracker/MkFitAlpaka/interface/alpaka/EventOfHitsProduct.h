#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_EventOfHitsProduct_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_EventOfHitsProduct_h

// Event product "device EventOfHits" (device side); see interface/EventOfHitsProduct.h.
// CPU backends: the pooled host collection (same interface, memory from the producer's per-stream
// pool); GPU backends: the PortableCollection (caching allocator; deleted early after the device fit in the menu).

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/EventOfHitsProduct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  using EventOfHitsDeviceCollection = ::mkfitdev::EventOfHitsPooledHostCollection;
#else
  using EventOfHitsDeviceCollection = PortableCollection< ::mkfitdev::EventOfHitsBlocks>;
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
