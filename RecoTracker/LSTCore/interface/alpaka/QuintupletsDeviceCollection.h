#ifndef RecoTracker_LSTCore_interface_QuintupletsDeviceCollection_h
#define RecoTracker_LSTCore_interface_QuintupletsDeviceCollection_h

#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using QuintupletsDeviceCollection = PortableCollection2<QuintupletsSoA, QuintupletsOccupancySoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
