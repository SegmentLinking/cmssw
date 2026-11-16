#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederSeedScoresSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederSeedScoresSoA_h

// The seed classifier's output (MkSeederSeedClassifier), one row per final seed: the model's score (the probability
// of a real seed) and whether the seed passes its threshold.

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  GENERATE_SOA_LAYOUT(MkSeederSeedScoresLayout, SOA_COLUMN(float, score), SOA_COLUMN(uint8_t, isSelected))

  using MkSeederSeedScoresSoA = MkSeederSeedScoresLayout<>;

}  // namespace mkseeder

#endif
