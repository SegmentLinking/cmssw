#ifndef RecoTracker_LSTCore_interface_TripletsDeviceCollection_h
#define RecoTracker_LSTCore_interface_TripletsDeviceCollection_h

#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using TripletsDeviceCollection = PortableCollection2<TripletsSoA, TripletsOccupancySoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
