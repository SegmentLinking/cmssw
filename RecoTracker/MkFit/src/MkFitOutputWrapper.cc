#include "RecoTracker/MkFit/interface/MkFitOutputWrapper.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/Track.h"

MkFitOutputWrapper::MkFitOutputWrapper() = default;

MkFitOutputWrapper::MkFitOutputWrapper(mkfit::TrackVec tracks, bool propagatedToFirstLayer)
    : tracks_{std::move(tracks)}, propagatedToFirstLayer_{propagatedToFirstLayer} {}

MkFitOutputWrapper::MkFitOutputWrapper(mkfit::TrackVec tracks,
                                       bool propagatedToFirstLayer,
                                       std::vector<mkfit::HitStatesOnTrack> hitStates,
                                       std::vector<mkfit::HitStatesOnTrack> hitStatesFwd,
                                       std::vector<mkfit::HitStatesOnTrack> hitStatesBwd)
    : tracks_{std::move(tracks)},
      propagatedToFirstLayer_{propagatedToFirstLayer},
      hitStates_{std::move(hitStates)},
      hitStatesFwd_{std::move(hitStatesFwd)},
      hitStatesBwd_{std::move(hitStatesBwd)} {}

MkFitOutputWrapper::~MkFitOutputWrapper() = default;

MkFitOutputWrapper::MkFitOutputWrapper(MkFitOutputWrapper&&) = default;
MkFitOutputWrapper& MkFitOutputWrapper::operator=(MkFitOutputWrapper&&) = default;
