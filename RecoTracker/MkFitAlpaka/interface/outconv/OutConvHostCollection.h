#ifndef RecoTracker_MkFitAlpaka_interface_outconv_OutConvHostCollection_h
#define RecoTracker_MkFitAlpaka_interface_outconv_OutConvHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/outconv/OutConvSoA.h"

namespace mkfitdev {
  using OutConvHostCollection = PortableHostCollection<OutConvSoA>;
}  // namespace mkfitdev

#endif
