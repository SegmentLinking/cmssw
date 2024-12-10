#ifndef RecoTracker_LSTCore_interface_PT2sHostCollection_h
#define RecoTracker_LSTCore_interface_PT2sHostCollection_h

#include "RecoTracker/LSTCore/interface/PT2sSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using PT2sHostCollection = PortableHostCollection<PT2sSoA>;
}  // namespace lst
#endif
