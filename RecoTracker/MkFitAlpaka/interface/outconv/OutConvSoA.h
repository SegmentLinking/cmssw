#ifndef RecoTracker_MkFitAlpaka_interface_outconv_OutConvSoA_h
#define RecoTracker_MkFitAlpaka_interface_outconv_OutConvSoA_h

// Per fitted track, what the device precomputes for the host output conversion
// (MkFitAlpakaOutConvStateProducer -> MkFitAlpakaOutputTrackConverter pcaStates). Row t belongs to row t of the fitted
// TrackSoA (rows [0, nTracks) are written).
//   pcaState / pcaCov: the state at the point of closest approach to the beam line, as TSCBLBuilderNoMaterial on the
//     converter's first-hit FreeTrajectoryState (interface/math/PcaToBeamLine.h): position and momentum (GlobalPoint /
//     GlobalVector, float) and the curvilinear covariance rounded to float in reco::TrackBase's packed order
//     (i >= j at i * (i + 1) / 2 + j; reco::Track stores it as float, so the rounding is the same as the host's).
//   pcaStatus: pca::PcaStatus (0 ok, 1 failed = invalid TSCBL on the host, 2 host fallback: the first-hit state is
//   outside
//     the closed-form field volume).

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkfitdev {

  struct PcaState {
    float v[6];  // x, y, z, px, py, pz
  };
  struct PcaCov {
    float v[15];
  };

  GENERATE_SOA_LAYOUT(OutConvLayout,
                      SOA_COLUMN(PcaState, pcaState),
                      SOA_COLUMN(PcaCov, pcaCov),
                      SOA_COLUMN(int8_t, pcaStatus))

  using OutConvSoA = OutConvLayout<>;
  using OutConvSoAView = OutConvSoA::View;
  using OutConvSoAConstView = OutConvSoA::ConstView;

}  // namespace mkfitdev

#endif
