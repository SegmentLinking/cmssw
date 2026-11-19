#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederSeedFeaturesSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederSeedFeaturesSoA_h

// The seed classifier's inputs, one row per final seed (MkSeederProducer, MkSeederPcaKernels.h; read by
// MkSeederSeedClassifier): the trained model's features in the order of its training, one column per feature (the
// classifier's threads read a feature of consecutive seeds), and the seed's flags.

#include <cstdint>

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // The model's features (the training's names in the comments; RecoTracker/MkFit/interface/MkFitSeedFeatureCalc.h):
  // the forward state's at the point of closest approach to the beam line as a reco::Track, the hits', the helix
  // consistency (the circle through the first, the middle and the last hit; the line in (s, z)), the pixel clusters'
  // sizes minus those expected from the circle's direction and their charges, the hit sharing with the other seeds,
  // the seeder's own (the seed fit's chi2, the cleaning's fake score, score and ambiguous quads). Missing values are
  // kSeedFeatureNone, as in the training.
  namespace seedfeature {
    enum Index : int {
      kPt = 0,               // pt
      kAbsEta,               // eta (as |eta|)
      kPtErrorRelative,      // pt_err_rel
      kCotThetaError,        // cot_err
      kEtaError,             // eta_err
      kPhiError,             // phi_err
      kDxy,                  // dxy
      kDxyError,             // dxy_err
      kDz,                   // dz
      kDzError,              // dz_err
      kNHits,                // nhits
      kNPixelHits,           // npix
      kNPixelBarrelHits,     // npxb
      kNPixelDiscHits,       // npxf
      kNOuterHits,           // not
      kNOuterBarrelHits,     // notb
      kNOuterDiscHits,       // note
      kNLayers,              // nlay
      kFirstHitR,            // first_r
      kFirstHitZ,            // first_z
      kLastHitR,             // last_r
      kLastHitZ,             // last_z
      kCircleResidualMax,    // circ_res_max
      kCircleResidualRms,    // circ_res_rms
      kRzResidualMax,        // rz_res_max
      kRzResidualRms,        // rz_res_rms
      kPtOverCirclePt,       // pt_ratio
      kPixelSizeXExcessMin,  // pix_dsx_min
      kPixelSizeXExcessMax,  // pix_dsx_max
      kPixelSizeYExcessMin,  // pix_dsy_min
      kPixelSizeYExcessMax,  // pix_dsy_max
      kPixelChargeMin,       // pix_q_min
      kPixelChargeMax,       // pix_q_max
      kOuterClusterSizeMax,  // ot_size_max
      kNSharingSeeds,        // nshare1
      kNSharingSeeds2Hits,   // nshare2
      kMaxSharedHits,        // share_max
      kSeedChi2,             // mk_chi2
      kSeedFakeScore,        // mk_fake
      kSeedCleanScore,       // mk_clean
      kSeedNAmbiguous,       // mk_namb
      // the PS-module confirmation features (copies of seedpsfeature's kNOTP, kNMarked, kPullMax, kPullMin,
      // kNRangeMatch, kNOutOfRange; kSeedFeatureNone without ps_mark: a model that uses them needs ps_mark 1). A model
      // may also take only the first kNBaseFeatures (the 6L v1 classifier)
      kPSNOTP,         // ps_notp
      kPSNMarked,      // ps_nmk
      kPSPullMax,      // ps_pmax
      kPSPullMin,      // ps_pmin
      kPSNRangeMatch,  // ps_c1
      kPSNOutOfRange,  // ps_cout
      // the seed's pixel hits that are merged-cluster candidates (MkSeederHitsSoA::mergedTag != 0; 0 without mrg_tag)
      kNMergedBarrel,  // mrg_nb: on barrel layers
      kNMergedDisc,    // mrg_nd: on discs
      kNFeatures,
      kNBaseFeatures = kPSNOTP
    };
  }  // namespace seedfeature
  constexpr float kSeedFeatureNone = -999.f;
  using MkSeederSeedModelInputs = Eigen::Matrix<float, seedfeature::kNFeatures, 1>;

  // The PS-module confirmation features of a seed (not model inputs yet: for a retraining; kSeedFeatureNone when
  // ps_mark is off; the q/pT-dependent ones, ps_pmax, ps_pmin, ps_c1, ps_c2, ps_cout, also when the seed has no valid
  // perigee): from the marks of its OT1-3 P hits (MkSeederOTHitsSoA psMatches, psQOverPt, psCounts; MkSeederPSMark.h),
  // the seed's q/pT from its perigee
  namespace seedpsfeature {
    enum Index : int {
      kNOTP = 0,      // ps_notp: the seed's OT1-3 P hits
      kNMarked,       // ps_nmk: those marked (>= 1 S hit in the ps_pt window)
      kPullMax,       // ps_pmax: the largest |q/pT(mark) - q/pT(seed)| / (ps_curv_res x 23 cm / r) of the marked ones
      kPullMin,       // ps_pmin: the smallest (kSeedFeatureNone: none marked)
      kNRangeMatch,   // ps_c1: hits with >= 1 S match (count window) in the seed's q/pT range
      kNRangeMatch2,  // ps_c2: hits with >= 2 there
      kNOutOfRange,   // ps_cout: hits with S matches only outside the seed's range
      kMaxTotal,      // ps_ctot: the largest total count (over the ranges) of the hits
      kNBestMerged,   // ps_mrgb: hits whose best S hit is merged
      kNAnyMerged,    // ps_mrga: hits with a merged S hit in the count window
      kNPSFeatures
    };
  }  // namespace seedpsfeature
  using MkSeederSeedPSInputs = Eigen::Matrix<float, seedpsfeature::kNPSFeatures, 1>;

  GENERATE_SOA_LAYOUT(MkSeederSeedFeaturesLayout,
                      SOA_EIGEN_COLUMN(MkSeederSeedModelInputs, inputs),
                      // the forward state does not convert to a TrajectorySeed state (its curvilinear covariance is not
                      // positive on the diagonal, or it has no crossing with its module's plane): the seed is not
                      // selected and takes no part in the other seeds' sharing features
                      SOA_COLUMN(uint8_t, isDropped),
                      // the state features are at the point of closest approach to the beam line (0: at the state's
                      // own position, the closest approach did not converge)
                      SOA_COLUMN(uint8_t, hasStateAtBeamline),
                      SOA_EIGEN_COLUMN(MkSeederSeedPSInputs, psInputs),
                      // the seed's pixel hits that are merged-cluster candidates (MkSeederHitsSoA::mergedTag != 0; 0
                      // when the tag is off): not a model input, for the classifier's training
                      SOA_COLUMN(uint8_t, nMergedCandidates))

  using MkSeederSeedFeaturesSoA = MkSeederSeedFeaturesLayout<>;

}  // namespace mkseeder

#endif
