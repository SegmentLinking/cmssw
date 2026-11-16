#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederEventHostCollection_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederEventHostCollection_h

// The host collections of the seeder's per-event data (filled on the host, copied to the device) and the host
// copies of its intermediate results (the event products: MkSeederSeedsHostCollection.h)

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederCounters.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederEventHitsSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederEventTablesSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederQuadsSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsHostCollection.h"

namespace mkseeder {

  using MkSeederUnbinnedHitsHostCollection = PortableHostCollection<MkSeederUnbinnedHitsSoA>;
  using MkSeederEventTablesHostCollection = PortableHostCollection<MkSeederEventTablesSoA>;
  using MkSeederClustersHostCollection = PortableHostCollection<MkSeederClustersSoA>;
  using MkSeederQuadsHostCollection = PortableHostCollection<MkSeederQuadsSoA>;
  using MkSeederKeptSeedsHostCollection = PortableHostCollection<MkSeederKeptSeedsSoA>;
  using MkSeederSeedHitsHostCollection = PortableHostCollection<MkSeederSeedHitsSoA>;
  using MkSeederSeedStatesHostCollection = PortableHostCollection<MkSeederSeedStatesSoA>;
  using MkSeederSeedHelicesHostCollection = PortableHostCollection<MkSeederSeedHelicesSoA>;
  using MkSeederFinderCountersHostObject = PortableHostObject<MkSeederFinderCounters>;
  using MkSeederCleanCountersHostObject = PortableHostObject<MkSeederCleanCounters>;

}  // namespace mkseeder

#endif
