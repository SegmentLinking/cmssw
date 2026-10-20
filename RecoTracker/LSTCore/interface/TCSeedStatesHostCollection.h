#ifndef RecoTracker_LSTCore_interface_TCSeedStatesHostCollection_h
#define RecoTracker_LSTCore_interface_TCSeedStatesHostCollection_h

#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using TCSeedStatesHostCollection = PortableHostCollection<TCSeedStatesSoA>;
}  // namespace lst

#endif
