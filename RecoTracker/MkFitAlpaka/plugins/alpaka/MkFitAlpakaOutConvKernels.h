#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaOutConvKernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaOutConvKernels_h
// Device precomputation for the host output conversion, one thread per fitted track
// (MkFitAlpakaOutConvStateProducer). See interface/outconv/OutConvSoA.h.
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/PcaToBeamLine.h"
#include "RecoTracker/MkFitAlpaka/interface/outconv/OutConvSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::outconv {
  // rows [0, trk.nTracks()) of trk -> the same rows of out; capacity = trk's (the row count stays on the device)
  void launchOutConvStates(Queue& queue,
                           ::mkfitdev::TrackSoAConstView trk,
                           int capacity,
                           ::mkfitdev::pca::BeamIn const& beamLine,
                           ::mkfitdev::OutConvSoAView out);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::outconv

#endif
