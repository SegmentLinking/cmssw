#ifndef RecoTracker_LSTCore_interface_alpaka_PT2sDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_PT2sDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/PT2sSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using PT2sDeviceCollection = PortableCollection<PT2sSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
