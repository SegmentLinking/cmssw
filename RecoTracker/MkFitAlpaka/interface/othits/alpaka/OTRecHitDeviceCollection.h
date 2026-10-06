#ifndef RecoTracker_MkFitAlpaka_interface_othits_alpaka_OTRecHitDeviceCollection_h
#define RecoTracker_MkFitAlpaka_interface_othits_alpaka_OTRecHitDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTRecHitSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using OTRecHitDeviceCollection = PortableCollection<::mkfitdev::OTRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(mkfitdev::OTRecHitDeviceCollection, ::mkfitdev::OTRecHitHostCollection);

#endif
