#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_tracks_TrackSoADeviceCollection_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_tracks_TrackSoADeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoAHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using ::mkfitdev::TrackSoA;
  using ::mkfitdev::TrackSoAConstView;
  using ::mkfitdev::TrackSoAView;
  using TrackSoADeviceCollection = PortableCollection<TrackSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
