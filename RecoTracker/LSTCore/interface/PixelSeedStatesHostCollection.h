#ifndef RecoTracker_LSTCore_interface_PixelSeedStatesHostCollection_h
#define RecoTracker_LSTCore_interface_PixelSeedStatesHostCollection_h

#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using PixelSeedStatesHostCollection = PortableHostCollection<PixelSeedStatesSoA>;
}  // namespace lst

#endif
