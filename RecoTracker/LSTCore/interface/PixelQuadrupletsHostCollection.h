#ifndef RecoTracker_LSTCore_interface_PixelQuadrupletsHostCollection_h
#define RecoTracker_LSTCore_interface_PixelQuadrupletsHostCollection_h

#include "RecoTracker/LSTCore/interface/PixelQuadrupletsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using PixelQuadrupletsHostCollection = PortableHostCollection<PixelQuadrupletsSoA>;
}  // namespace lst
#endif
