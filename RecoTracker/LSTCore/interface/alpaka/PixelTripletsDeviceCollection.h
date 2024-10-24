#ifndef RecoTracker_LSTCore_interface_PixelTripletsDeviceCollection_h
#define RecoTracker_LSTCore_interface_PixelTripletsDeviceCollection_h

#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using PixelTripletsDeviceCollection = PortableCollection<PixelTripletsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
