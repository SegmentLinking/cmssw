#ifndef RecoTracker_MkFitAlpaka_interface_fit_HitStateDev_h
#define RecoTracker_MkFitAlpaka_interface_fit_HitStateDev_h

#include <cstdint>

namespace mkfitdev::fit {

  // Device mirror of mkfit::HitStateOnTrack (MkFitCore/interface/HitStateOnTrack.h): the smoothed state
  // of the final fit on the module plane of one hit, local frame (q/p, dx/dz, dy/dz, x, y), err = lower triangle
  // (00, 10, 11, 20, ...), chi2 = the hit's chi2 increment, kind 0 Combined / 1 ForwardOnly / 2 BackwardOnly.
  // Stored per (track, HitOnTrack position): element t * kMaxTrkHits + position.
  struct HitStateDev {
    float par[5];
    float err[15];
    float chi2;
    int8_t pzSign;
    int8_t kind;
    int8_t valid;
  };

}  // namespace mkfitdev::fit

#endif
