#ifndef RecoTracker_MkFitAlpaka_interface_seeds_SeedsHostCollection_h
#define RecoTracker_MkFitAlpaka_interface_seeds_SeedsHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"

namespace mkfitdev {
  using SeedsHostCollection = PortableHostCollection<SeedSoA>;
}  // namespace mkfitdev

#endif
