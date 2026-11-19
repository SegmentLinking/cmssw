#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederSeedsHostCollection_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederSeedsHostCollection_h

// The seeder's event products on the host (the device products, copied by the framework; on a CPU backend the same
// objects): MkSeederProducer's final seeds (their hits with the hits' cluster indices, states, beam-line states,
// quality and perigees), their number (the second duplicate cleaning's counters: nOutput) and the seed classifier's
// features, optionally the seeds' states at their last pixel hit; MkSeederSeedClassifier's scores

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederCounters.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedFeaturesSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedScoresSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsSoA.h"

namespace mkseeder {

  using MkSeederSeedsHostCollection = PortableHostCollection<MkSeederSeedsSoA>;
  using MkSeederSeedFeaturesHostCollection = PortableHostCollection<MkSeederSeedFeaturesSoA>;
  using MkSeederSeedScoresHostCollection = PortableHostCollection<MkSeederSeedScoresSoA>;
  using MkSeederSeedPixelStatesHostCollection = PortableHostCollection<MkSeederSeedPixelStatesSoA>;
  using MkSeederDupCountersHostObject = PortableHostObject<MkSeederDupCounters>;

}  // namespace mkseeder

#endif
