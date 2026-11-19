#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederSeedsSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederSeedsSoA_h

// The seeds after the cleaning: their hits and the post step's bookkeeping (the OT extension, the pixel fill), their
// states (the fits, the refits, the beam-line states) and their helices (the duplicate cleanings).

#include <cstdint>

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // the hits of a seed (mkfit::Track::Status::kMaxSeedHits) and of the OT extension
  constexpr int kMaxSeedHits = 15;
  constexpr int kMaxExtensionHits = 3;
  using MkSeederSeedHitLayers = Eigen::Matrix<int16_t, kMaxSeedHits, 1>;
  using MkSeederSeedHitRows = Eigen::Matrix<uint32_t, kMaxSeedHits, 1>;
  using MkSeederExtensionHitLayers = Eigen::Matrix<int16_t, kMaxExtensionHits, 1>;
  using MkSeederExtensionHitRows = Eigen::Matrix<uint32_t, kMaxExtensionHits, 1>;
  // a state's parameters (mkFit's x, y, z, 1/pT, phi, theta) and the lower triangle of their covariance
  using MkSeederStateParameters = Eigen::Matrix<float, 6, 1>;
  using MkSeederStateCovariance = Eigen::Matrix<float, 21, 1>;

  // One row per seed: its hits (mkFit layer, bin-order row of MkSeederHitsSoA), the post step's flags and the
  // extension's hits (for the refit of the extension alone when the refit with the fill fails), its label (the
  // cleaning's index of the seed it comes from)
  GENERATE_SOA_LAYOUT(MkSeederSeedHitsLayout,
                      SOA_COLUMN(uint8_t, nHits),
                      SOA_EIGEN_COLUMN(MkSeederSeedHitLayers, hitLayers),
                      SOA_EIGEN_COLUMN(MkSeederSeedHitRows, hitRows),
                      // the r-z slope the extension's windows start from
                      SOA_COLUMN(float, cot),
                      SOA_COLUMN(uint8_t, isKept),
                      SOA_COLUMN(uint8_t, isExtended),
                      SOA_COLUMN(uint8_t, isFilled),
                      // the post step's input: a triplet that failed the triplet MLP (dropped unless the extension
                      // gives it SeederConfig::trip_dnn_ext outer-tracker hits)
                      SOA_COLUMN(uint8_t, tripletFailedMlp),
                      SOA_COLUMN(uint8_t, nExtensionHits),
                      SOA_EIGEN_COLUMN(MkSeederExtensionHitLayers, extensionLayers),
                      SOA_EIGEN_COLUMN(MkSeederExtensionHitRows, extensionHits),
                      SOA_COLUMN(uint32_t, label))

  // One row per seed: its state at its last hit (fitOk: the fit worked; wasRefitted: the post step refitted it
  // because its hits changed), the state at the beam line (hasBeamlineState)
  GENERATE_SOA_LAYOUT(MkSeederSeedStatesLayout,
                      SOA_COLUMN(uint8_t, fitOk),
                      SOA_COLUMN(uint8_t, wasRefitted),
                      SOA_COLUMN(uint8_t, hasBeamlineState),
                      SOA_COLUMN(int32_t, charge),
                      SOA_COLUMN(float, chi2),
                      SOA_EIGEN_COLUMN(MkSeederStateParameters, parameters),
                      SOA_EIGEN_COLUMN(MkSeederStateCovariance, covariance),
                      SOA_COLUMN(int32_t, beamlineCharge),
                      SOA_EIGEN_COLUMN(MkSeederStateParameters, beamlineParameters),
                      SOA_EIGEN_COLUMN(MkSeederStateCovariance, beamlineCovariance))

  // One row per seed: the helix of its state at the beam line for the duplicate cleaning: q /
  // pT, cot(theta), phi0, z0, eta and the variances of q / pT and cot(theta)
  GENERATE_SOA_LAYOUT(MkSeederSeedHelicesLayout,
                      SOA_COLUMN(float, qOverPt),
                      SOA_COLUMN(float, cot),
                      SOA_COLUMN(float, phi0),
                      SOA_COLUMN(float, z0),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, varQOverPt),
                      SOA_COLUMN(float, varCot))

  // One row per seed: its quality (the cleaning's: the kept seed's score, fake score and n_amb)
  GENERATE_SOA_LAYOUT(MkSeederSeedQualityLayout,
                      SOA_COLUMN(float, score),
                      SOA_COLUMN(float, fakeScore),
                      SOA_COLUMN(int32_t, nAmbiguous))

  // One row per final seed: its hits' cluster indices (the index in the layer's HitVec; the hits' clusterIndex of
  // their hitRows): the seeds' hits on the host (MkSeederSeedConverter)
  using MkSeederSeedHitClusterIndices = Eigen::Matrix<int32_t, kMaxSeedHits, 1>;
  GENERATE_SOA_LAYOUT(MkSeederSeedClusterIndicesLayout,
                      SOA_EIGEN_COLUMN(MkSeederSeedHitClusterIndices, hitClusterIndex))

  // One row per final seed: its pixel track (MkSeederPcaKernels.h: the beam-line state at its point of closest approach
  // to the beam line, as TrackFromSeedProducer makes it of the beam-line seed): the status (kPerigeeValid; a closest
  // approach that did not converge: kPerigeeFailedFit, the track of a failed fit; a beam-line state that does not
  // convert: kPerigeeNone, no track), the charge, the reference point, the momentum and the perigee covariance
  // (reco::TrackBase's: the lower triangle, row by row)
  enum MkSeederPerigeeStatus : uint8_t { kPerigeeNone = 0, kPerigeeValid = 1, kPerigeeFailedFit = 2 };
  using MkSeederPerigeeVector = Eigen::Matrix<float, 3, 1>;
  using MkSeederPerigeeCovariance = Eigen::Matrix<float, 15, 1>;
  GENERATE_SOA_LAYOUT(MkSeederSeedPerigeeLayout,
                      SOA_COLUMN(uint8_t, status),
                      SOA_COLUMN(int8_t, charge),
                      SOA_EIGEN_COLUMN(MkSeederPerigeeVector, vertex),
                      SOA_EIGEN_COLUMN(MkSeederPerigeeVector, momentum),
                      SOA_EIGEN_COLUMN(MkSeederPerigeeCovariance, covariance))

  // One row per final seed (MkSeederProducer's optional 'lastPixelStates'): for a seed whose last hit is an
  // outer-tracker hit after pixel hits, its state at its last hit taken back onto its last pixel hit's module without
  // material (the AnalyticalPropagator oppositeToMomentum of LSTInputProducer's stateAtLastPixelHit), as the
  // parameters (mkFit's x, y, z, 1/pT, phi, theta) and covariance of MkSeederSeedStatesLayout; the seed's number of
  // pixel hits. status kPixelStateNone: no such state (the last hit is a pixel hit, no pixel hit, the state does not
  // convert, or no crossing)
  enum MkSeederPixelStateStatus : uint8_t { kPixelStateNone = 0, kPixelStateValid = 1 };
  GENERATE_SOA_LAYOUT(MkSeederSeedPixelStatesLayout,
                      SOA_COLUMN(uint8_t, status),
                      SOA_COLUMN(uint8_t, nPixelHits),
                      SOA_COLUMN(int8_t, charge),
                      SOA_EIGEN_COLUMN(MkSeederStateParameters, parameters),
                      SOA_EIGEN_COLUMN(MkSeederStateCovariance, covariance))

  // One row per seed: its hits only (the post step's copy of its input hits, the refit's fallback)
  GENERATE_SOA_LAYOUT(MkSeederSeedHitListLayout,
                      SOA_COLUMN(uint8_t, nHits),
                      SOA_EIGEN_COLUMN(MkSeederSeedHitLayers, hitLayers),
                      SOA_EIGEN_COLUMN(MkSeederSeedHitRows, hitRows))

  // the seeds with their states and their quality (the duplicate cleanings' outputs; 1 row: not set), row by row; the
  // final seeds (the seeder's event product, MkSeederSeedsHostCollection.h) also with their hits' cluster indices and
  // their perigees
  GENERATE_SOA_BLOCKS(MkSeederSeedsLayout,
                      SOA_BLOCK(hits, MkSeederSeedHitsLayout),
                      SOA_BLOCK(states, MkSeederSeedStatesLayout),
                      SOA_BLOCK(quality, MkSeederSeedQualityLayout),
                      SOA_BLOCK(clusterIndices, MkSeederSeedClusterIndicesLayout),
                      SOA_BLOCK(perigee, MkSeederSeedPerigeeLayout))

  using MkSeederSeedHitsSoA = MkSeederSeedHitsLayout<>;
  using MkSeederSeedStatesSoA = MkSeederSeedStatesLayout<>;
  using MkSeederSeedHitListSoA = MkSeederSeedHitListLayout<>;
  using MkSeederSeedHelicesSoA = MkSeederSeedHelicesLayout<>;
  using MkSeederSeedQualitySoA = MkSeederSeedQualityLayout<>;
  using MkSeederSeedClusterIndicesSoA = MkSeederSeedClusterIndicesLayout<>;
  using MkSeederSeedPerigeeSoA = MkSeederSeedPerigeeLayout<>;
  using MkSeederSeedPixelStatesSoA = MkSeederSeedPixelStatesLayout<>;
  using MkSeederSeedsSoA = MkSeederSeedsLayout<>;

}  // namespace mkseeder

#endif
