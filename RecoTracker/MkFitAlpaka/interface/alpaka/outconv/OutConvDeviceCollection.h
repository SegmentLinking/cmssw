#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_outconv_OutConvDeviceCollection_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_outconv_OutConvDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/outconv/OutConvHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/outconv/OutConvSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using ::mkfitdev::OutConvSoA;
  using ::mkfitdev::OutConvSoAConstView;
  using ::mkfitdev::OutConvSoAView;
  using OutConvDeviceCollection = PortableCollection<OutConvSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
