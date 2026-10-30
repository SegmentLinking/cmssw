#ifndef RecoTracker_MkFitAlpaka_interface_tracks_TrackSoAHostCollection_h
#define RecoTracker_MkFitAlpaka_interface_tracks_TrackSoAHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace mkfitdev {
  using TrackSoAHostCollection = PortableHostCollection<TrackSoA>;
}  // namespace mkfitdev

#endif
