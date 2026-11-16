#ifndef RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederSeedsDeviceCollection_h
#define RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederSeedsDeviceCollection_h

// The seeder's and the seed classifier's event products on the device (MkSeederSeedsHostCollection.h)

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  using MkSeederSeedsDeviceCollection = PortableCollection<::mkseeder::MkSeederSeedsSoA>;
  using MkSeederSeedFeaturesDeviceCollection = PortableCollection<::mkseeder::MkSeederSeedFeaturesSoA>;
  using MkSeederSeedScoresDeviceCollection = PortableCollection<::mkseeder::MkSeederSeedScoresSoA>;
  using MkSeederSeedPixelStatesDeviceCollection = PortableCollection<::mkseeder::MkSeederSeedPixelStatesSoA>;
  using MkSeederDupCountersDeviceObject = PortableObject<::mkseeder::MkSeederDupCounters>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(mkseeder::MkSeederSeedsDeviceCollection, ::mkseeder::MkSeederSeedsHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(mkseeder::MkSeederSeedFeaturesDeviceCollection,
                                      ::mkseeder::MkSeederSeedFeaturesHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(mkseeder::MkSeederSeedScoresDeviceCollection,
                                      ::mkseeder::MkSeederSeedScoresHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(mkseeder::MkSeederSeedPixelStatesDeviceCollection,
                                      ::mkseeder::MkSeederSeedPixelStatesHostCollection);

#endif
