#ifndef RecoTracker_MkFit_MkFitOutputWrapper_h
#define RecoTracker_MkFit_MkFitOutputWrapper_h

#include <vector>

#include "RecoTracker/MkFitCore/interface/HitStateOnTrack.h"

namespace mkfit {
  class Track;
  using TrackVec = std::vector<Track>;
}  // namespace mkfit

class MkFitOutputWrapper {
public:
  MkFitOutputWrapper();
  MkFitOutputWrapper(mkfit::TrackVec tracks, bool propagatedToFirstLayer);
  // with the final fit's per-hit states, one HitStatesOnTrack per track (see mkfit::HitStateOnTrack); fwd and bwd
  // (validation only) hold the two states each smoothed state combines
  MkFitOutputWrapper(mkfit::TrackVec tracks,
                     bool propagatedToFirstLayer,
                     std::vector<mkfit::HitStatesOnTrack> hitStates,
                     std::vector<mkfit::HitStatesOnTrack> hitStatesFwd = {},
                     std::vector<mkfit::HitStatesOnTrack> hitStatesBwd = {});
  ~MkFitOutputWrapper();

  MkFitOutputWrapper(MkFitOutputWrapper const&) = delete;
  MkFitOutputWrapper& operator=(MkFitOutputWrapper const&) = delete;
  MkFitOutputWrapper(MkFitOutputWrapper&&);
  MkFitOutputWrapper& operator=(MkFitOutputWrapper&&);

  mkfit::TrackVec const& tracks() const { return tracks_; }
  bool propagatedToFirstLayer() const { return propagatedToFirstLayer_; }
  std::vector<mkfit::HitStatesOnTrack> const& hitStates() const { return hitStates_; }
  std::vector<mkfit::HitStatesOnTrack> const& hitStatesFwd() const { return hitStatesFwd_; }
  std::vector<mkfit::HitStatesOnTrack> const& hitStatesBwd() const { return hitStatesBwd_; }

private:
  mkfit::TrackVec tracks_;
  bool propagatedToFirstLayer_;
  std::vector<mkfit::HitStatesOnTrack> hitStates_;
  std::vector<mkfit::HitStatesOnTrack> hitStatesFwd_;
  std::vector<mkfit::HitStatesOnTrack> hitStatesBwd_;
};

#endif
