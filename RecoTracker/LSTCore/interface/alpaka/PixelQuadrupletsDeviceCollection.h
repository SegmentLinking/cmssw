#ifndef RecoTracker_LSTCore_interface_PixelQuadrupletsDeviceCollection_h
#define RecoTracker_LSTCore_interface_PixelQuadrupletsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/PixelQuadrupletsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using PixelQuadrupletsDeviceCollection = PortableCollection<PixelQuadrupletsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
