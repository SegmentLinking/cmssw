#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederQuadsSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederQuadsSoA_h

// The chain finder's output (the quads and triplet seeds) and the cleaning's kept seeds.

#include <cstdint>

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // a quad's four mkFit layers (a triplet seed: layer 3 = -1) and hits
  using MkSeederQuadLayers = Eigen::Matrix<int16_t, 4, 1>;
  using MkSeederQuadHits = Eigen::Matrix<uint32_t, 4, 1>;

  // One row per quad or triplet seed: its hits as their cluster indices (the index in the layer's HitVec), its
  // score, its fake score; a triplet that failed the triplet MLP and is kept for the extension's probe (SeederConfig
  // trip_dnn_ext) has tripletFailedMlp set and fakeScore = score
  GENERATE_SOA_LAYOUT(MkSeederQuadsLayout,
                      SOA_EIGEN_COLUMN(MkSeederQuadLayers, layers),
                      SOA_EIGEN_COLUMN(MkSeederQuadHits, hits),
                      SOA_COLUMN(float, score),
                      SOA_COLUMN(float, fakeScore),
                      SOA_COLUMN(uint8_t, tripletFailedMlp))

  // One row per kept seed in the cleaning's output order (its label): the quad's columns with its hits as their
  // bin-order rows (the fit's input), the number of dropped quads charged to it (n_amb)
  GENERATE_SOA_LAYOUT(MkSeederKeptSeedsLayout,
                      SOA_COLUMN(uint8_t, nHits),
                      SOA_EIGEN_COLUMN(MkSeederQuadLayers, layers),
                      SOA_EIGEN_COLUMN(MkSeederQuadHits, hitRows),
                      SOA_COLUMN(float, score),
                      SOA_COLUMN(float, fakeScore),
                      SOA_COLUMN(int32_t, nAmbiguous),
                      SOA_COLUMN(uint8_t, tripletFailedMlp))

  using MkSeederQuadsSoA = MkSeederQuadsLayout<>;
  using MkSeederKeptSeedsSoA = MkSeederKeptSeedsLayout<>;

}  // namespace mkseeder

#endif
