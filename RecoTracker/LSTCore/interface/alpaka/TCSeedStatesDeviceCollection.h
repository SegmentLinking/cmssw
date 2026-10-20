#ifndef RecoTracker_LSTCore_interface_alpaka_TCSeedStatesDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_TCSeedStatesDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using TCSeedStatesDeviceCollection = PortableCollection<TCSeedStatesSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
