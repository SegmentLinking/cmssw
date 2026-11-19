#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederClassifier_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederClassifier_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederForestSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedFeaturesSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedScoresSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // The seed selection's thresholds on the score (MkSeederSeedClassifier's parameters): per |eta| bin (nEtaEdges
  // ascending edges, nEtaEdges + 1 thresholds; none: 'threshold' everywhere), a separate one for the 3-hit seeds
  // (tripletThreshold >= 0)
  struct SeedClassifierThresholds {
    static constexpr int kMaxEtaEdges = 8;
    double threshold;
    double tripletThreshold;
    int nEtaEdges;
    double etaEdges[kMaxEtaEdges];
    double etaThresholds[kMaxEtaEdges + 1];
    // zFold: the model sees each seed as its z mirror image with last z >= first z (dz, first z and last z times the
    // sign of last z - first z)
    int zFold = 0;
  };

  // per seed of the first *count rows of features its score (MkSeederForestSoA.h: the same comparisons and the same
  // double sums in the same order as xgboost's evaluation) and whether it passes its threshold (a dropped seed
  // never does); the rows after the first *count: score 0, not selected
  void classifySeeds(Queue &queue,
                     ::mkseeder::MkSeederSeedFeaturesSoA::ConstView features,
                     const uint32_t *count,
                     ::mkseeder::MkSeederForestSoA::ConstView forest,
                     const SeedClassifierThresholds &thresholds,
                     ::mkseeder::MkSeederSeedScoresSoA::View scores);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
