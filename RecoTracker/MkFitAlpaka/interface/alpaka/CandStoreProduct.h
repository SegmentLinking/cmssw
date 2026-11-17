#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_CandStoreProduct_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_CandStoreProduct_h

// Event product "candidate storage" (device side); see interface/CandStoreProduct.h.

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/CandStoreProduct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using CandStoreDeviceCollection = PortableCollection<::mkfitdev::CandStoreBlocks>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
