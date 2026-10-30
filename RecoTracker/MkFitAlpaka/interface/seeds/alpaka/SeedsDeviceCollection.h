#ifndef RecoTracker_MkFitAlpaka_interface_seeds_alpaka_SeedsDeviceCollection_h
#define RecoTracker_MkFitAlpaka_interface_seeds_alpaka_SeedsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedsHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using ::mkfitdev::SeedSoA;
  using SeedsDeviceCollection = PortableCollection<SeedSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
