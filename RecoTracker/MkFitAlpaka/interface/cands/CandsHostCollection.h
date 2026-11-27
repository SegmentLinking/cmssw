#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandsHostCollection_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandsHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"

namespace mkfitdev {
  using SeedCandsHostCollection = PortableHostCollection<SeedCandsSoA>;
  using CandSlotsHostCollection = PortableHostCollection<CandSlotsSoA>;
  using CandHotsHostCollection = PortableHostCollection<CandHotsSoA>;
  using CandOptionsHostCollection = PortableHostCollection<CandOptionsSoA>;
  using CandExtrasHostCollection = PortableHostCollection<CandExtrasSoA>;
  using CandUpdatesHostCollection = PortableHostCollection<CandUpdatesSoA>;
  using CandSelHitsHostCollection = PortableHostCollection<CandSelHitsSoA>;
  using CandHitChi2HostCollection = PortableHostCollection<CandHitChi2SoA>;
}  // namespace mkfitdev

#endif
