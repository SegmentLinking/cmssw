#ifndef RecoTracker_MkFitAlpaka_interface_hits_EventOfHitsHostCollections_h
#define RecoTracker_MkFitAlpaka_interface_hits_EventOfHitsHostCollections_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"

namespace mkfitdev {
  using HitsHostCollection = PortableHostCollection<HitSoA>;
  using LayersHostCollection = PortableHostCollection<LayerSoA>;
  using BinnedHitsHostCollection = PortableHostCollection<BinnedHitSoA>;
  using BinsHostCollection = PortableHostCollection<BinSoA>;
}  // namespace mkfitdev

#endif
