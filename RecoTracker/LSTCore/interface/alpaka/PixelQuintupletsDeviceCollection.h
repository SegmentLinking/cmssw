#ifndef RecoTracker_LSTCore_interface_PixelQuintupletsDeviceCollection_h
#define RecoTracker_LSTCore_interface_PixelQuintupletsDeviceCollection_h

#include "RecoTracker/LSTCore/interface/PixelQuintupletsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using PixelQuintupletsDeviceCollection = PortableCollection<PixelQuintupletsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
