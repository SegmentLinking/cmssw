#ifndef RecoTracker_MkFitCore_interface_HitStateOnTrack_h
#define RecoTracker_MkFitCore_interface_HitStateOnTrack_h

#include <vector>

namespace mkfit {

  // State of the final fit on the module plane of one hit, in that module's local frame as CMSSW's
  // LocalTrajectoryParameters / LocalTrajectoryError: par = (q/p, dx/dz, dy/dz, x, y), err = lower triangle of the
  // 5x5 covariance (00, 10, 11, 20, 21, 22, ...), pzSign = sign of the local z momentum.  A track has one entry per
  // HitOnTrack position; valid is false where the final fit did not use that position (missing or removed hit).
  struct HitStateOnTrack {
    enum Kind : signed char { Combined = 0, ForwardOnly = 1, BackwardOnly = 2 };

    float par[5];
    float err[15];
    float chi2;  // the hit's chi2 increment in the final fit
    signed char pzSign;
    signed char kind;  // how the smoothed state was obtained (Kind)
    bool valid = false;
  };
  using HitStatesOnTrack = std::vector<HitStateOnTrack>;

}  // namespace mkfit

#endif
