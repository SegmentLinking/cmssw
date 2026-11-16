#ifndef RecoTracker_MkFit_MkFitSeedWrapper_h
#define RecoTracker_MkFit_MkFitSeedWrapper_h

#include <memory>
#include <vector>

namespace mkfit {
  class Track;
  using TrackVec = std::vector<Track>;
  struct SeedQuality;
}  // namespace mkfit

class MkFitSeedWrapper {
public:
  MkFitSeedWrapper();
  MkFitSeedWrapper(mkfit::TrackVec seeds);
  // With the mkFit seeder's quality field, by seed label (mkfit::SeedQuality, Track.h)
  MkFitSeedWrapper(mkfit::TrackVec seeds, std::vector<mkfit::SeedQuality> quality);
  ~MkFitSeedWrapper();

  MkFitSeedWrapper(MkFitSeedWrapper const&) = delete;
  MkFitSeedWrapper& operator=(MkFitSeedWrapper const&) = delete;
  MkFitSeedWrapper(MkFitSeedWrapper&&);
  MkFitSeedWrapper& operator=(MkFitSeedWrapper&&);

  mkfit::TrackVec const& seeds() const { return *seeds_; }
  // empty unless the seeds come from the mkFit seeder
  std::vector<mkfit::SeedQuality> const& quality() const;

private:
  std::unique_ptr<mkfit::TrackVec> seeds_;  // for pimpl pattern
  std::unique_ptr<std::vector<mkfit::SeedQuality>> quality_;
};

#endif
