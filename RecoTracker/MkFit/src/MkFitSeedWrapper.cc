#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/Track.h"

MkFitSeedWrapper::MkFitSeedWrapper() = default;

MkFitSeedWrapper::MkFitSeedWrapper(mkfit::TrackVec seeds)
    : seeds_{std::make_unique<mkfit::TrackVec>(std::move(seeds))} {}

MkFitSeedWrapper::MkFitSeedWrapper(mkfit::TrackVec seeds, std::vector<mkfit::SeedQuality> quality)
    : seeds_{std::make_unique<mkfit::TrackVec>(std::move(seeds))},
      quality_{std::make_unique<std::vector<mkfit::SeedQuality>>(std::move(quality))} {}

std::vector<mkfit::SeedQuality> const& MkFitSeedWrapper::quality() const {
  static const std::vector<mkfit::SeedQuality> empty;
  return quality_ ? *quality_ : empty;
}

MkFitSeedWrapper::~MkFitSeedWrapper() = default;

MkFitSeedWrapper::MkFitSeedWrapper(MkFitSeedWrapper&&) = default;
MkFitSeedWrapper& MkFitSeedWrapper::operator=(MkFitSeedWrapper&&) = default;
