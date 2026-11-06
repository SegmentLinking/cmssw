#ifndef RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandsDeviceCollection_h
#define RecoTracker_MkFitAlpaka_interface_cands_alpaka_CandsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using ::mkfitdev::CandExtrasSoA;
  using ::mkfitdev::CandHitChi2SoA;
  using ::mkfitdev::CandHotsSoA;
  using ::mkfitdev::CandOptionsSoA;
  using ::mkfitdev::CandSelHitsSoA;
  using ::mkfitdev::CandSlotsSoA;
  using ::mkfitdev::CandUpdatesSoA;
  using ::mkfitdev::SeedCandsSoA;

  using SeedCandsDeviceCollection = PortableCollection<SeedCandsSoA>;
  using CandSlotsDeviceCollection = PortableCollection<CandSlotsSoA>;
  using CandHotsDeviceCollection = PortableCollection<CandHotsSoA>;
  using CandOptionsDeviceCollection = PortableCollection<CandOptionsSoA>;
  using CandExtrasDeviceCollection = PortableCollection<CandExtrasSoA>;
  using CandUpdatesDeviceCollection = PortableCollection<CandUpdatesSoA>;
  using CandSelHitsDeviceCollection = PortableCollection<CandSelHitsSoA>;
  using CandHitChi2DeviceCollection = PortableCollection<CandHitChi2SoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
