#ifndef RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederEventDeviceCollection_h
#define RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederEventDeviceCollection_h

// The device collections of the seeder's per-event data

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederEventHostCollection.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederSeedsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // the hits in bin order (the binning), the binning's input (the hits in cluster order, filled on the host)
  using MkSeederEventHitsDeviceCollection = PortableCollection<::mkseeder::MkSeederEventHitsSoA>;
  using MkSeederUnbinnedHitsDeviceCollection = PortableCollection<::mkseeder::MkSeederUnbinnedHitsSoA>;
  using MkSeederEventTablesDeviceCollection = PortableCollection<::mkseeder::MkSeederEventTablesSoA>;
  using MkSeederClustersDeviceCollection = PortableCollection<::mkseeder::MkSeederClustersSoA>;
  // the CPU backends only
  using MkSeederCpuCacheDeviceCollection = PortableCollection<::mkseeder::MkSeederCpuCacheSoA>;

  // the intermediate results: the chain finder's quads; the cleaning's kept seeds; the seeds, their states and
  // helices; the counters (the event products: MkSeederSeedsDeviceCollection.h)
  using MkSeederQuadsDeviceCollection = PortableCollection<::mkseeder::MkSeederQuadsSoA>;
  using MkSeederKeptSeedsDeviceCollection = PortableCollection<::mkseeder::MkSeederKeptSeedsSoA>;
  using MkSeederSeedHitsDeviceCollection = PortableCollection<::mkseeder::MkSeederSeedHitsSoA>;
  using MkSeederSeedStatesDeviceCollection = PortableCollection<::mkseeder::MkSeederSeedStatesSoA>;
  using MkSeederSeedHelicesDeviceCollection = PortableCollection<::mkseeder::MkSeederSeedHelicesSoA>;
  using MkSeederFinderCountersDeviceObject = PortableObject<::mkseeder::MkSeederFinderCounters>;
  using MkSeederCleanCountersDeviceObject = PortableObject<::mkseeder::MkSeederCleanCounters>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
