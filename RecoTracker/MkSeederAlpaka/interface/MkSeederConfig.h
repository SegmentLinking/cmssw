#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederConfig_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederConfig_h

// The configuration of the seeder (host only): the seeding windows and the layer patterns, the chain of crossed
// layers, the fake cuts of the chain finder, the cleanings, the outer-tracker extension, the pixel fill and the seed
// fit. Read from a JSON file (readMkSeederConfig(), the JSON keys are given at each member) by MkSeederConfigESProducer;
// MkSeederESProducer builds the static tables from it and MkSeederProducer takes the per-event settings from it.
//
// A file may set the options that were removed from the seeder only to their
// "off" value (readMkSeederConfig() throws otherwise); unknown keys are ignored.

#include <array>
#include <string>
#include <vector>

#include "RecoTracker/MkSeederAlpaka/interface/MkSeederPSMark.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTripletMlpWeights.h"

namespace mkseeder {

  // trip_net_c's "off" cut (MkSeederConfig::stageCNetCuts)
  inline constexpr float kStageCNetOff = -1e30f;
  inline constexpr std::array<float, 5> kStageCNetCutsOff = {
      kStageCNetOff, kStageCNetOff, kStageCNetOff, kStageCNetOff, kStageCNetOff};

  struct MkSeederConfig {
    // the stage d windows of a pattern for the triplets in an |eta| slice (JSON: lo, hi, aphi, bphi, aq, bq)
    struct EtaWindow {
      float absEtaLow = 0, absEtaHigh = 0, phiWindowA = 0, phiWindowB = 0, qWindowA = 0, qWindowB = 0;
    };
    // A layer combination (mkFit layer ids, crossing order) with its windows. A pattern with +z pixel discs (16-27)
    // is also used mirrored to -z (+22), with the same windows.
    struct Pattern {
      std::array<int, 4> layers = {-1, -1, -1, -1};
      // phi_c, q_c, phi_d, q_d (JSON: win); < 0: the global ones
      std::array<float, 4> windows = {-1, -1, -1, -1};
      // the 1/pT terms of the stage d windows: w = a + b / pT (JSON: bwin)
      std::array<float, 2> dWindowSlopes = {0, 0};
      // > 0: the 1/pT terms scale with the c-d path length over this length (JSON: sref)
      float referencePathLength = 0;
      std::vector<EtaWindow> etaWindows;  // JSON: eta_win
    };
    // the kept band of the cluster length along z per |cot theta| bin of barrel pixel layer 0-3 (JSON: layer,
    // bin_width, lo, hi)
    struct ShapeWindow {
      int layer = -1;
      float binWidth = 0;
      std::vector<int> low, high;
    };

    // ---- the seeding windows
    float ptMin = 0.9f;              // pt_min, GeV
    float d0Max = 0.1f;              // d0_max, cm
    float zVertexHalfWidth = 25.0f;  // zv, cm: the start doublets' line reaches the beam line within this
    // zv_outer, cm (> 0): zv for the start doublets whose inner hit is at r >= zv_outer_rmin (displaced tracks)
    float zVertexHalfWidthOuter = 0;
    float zVertexOuterMinRadius = 3.5f;  // zv_outer_rmin, cm
    // zv_outer_cotmin (> 0): zv_outer only for the doublets with |cot| >= this (the inner hits that can make one)
    float zVertexOuterMinAbsCot = 0;
    int zVertexOuterDiscOnly = 0;  // zv_outer_disk (1): zv_outer only for the start pairs whose inner layer is a disk
    float doubletPhiMargin = 0.002f;  // marg_b, rad
    float phiWindowC = 0.004f;        // phi_c, rad
    float qWindowC = 0.10f;           // q_c, cm
    float phiWindowD = 0.004f;        // phi_d, rad
    float qWindowD = 0.10f;           // q_d, cm
    double windowScale = 1.0;         // win_scale: multiplies every c and d window
    std::vector<Pattern> patterns;    // patterns

    // ---- the chain
    int maxHoles = 0;                 // max_holes
    int maxHolesForOuterTracker = 0;  // max_holes_ot
    int alwaysForwardMisses = 0;      // hole_always
    float missGapDz = 0.f;   // miss_gap_dz (cm): > 0, an L1-L2 doublet's TBPX L3 miss goes on only near an L3 z gap
    int anyCombination = 0;  // any_combination: also layer combinations without a window table
    int maxStartHoles = -1;  // start_holes (< 0: max_holes)
    int leadOnly = 0;        // lead_only
    int innerOuterTrackerOnly = 0;               // inner_ot_only
    int startGapMode = 0;                        // start_gap
    float crossingMargin = 0.2f;                 // crossing_margin, cm
    float gapMapMargin = -1;                     // gap_map_margin, cm (>= 0: the gap map)
    std::vector<int> chainOuterTrackerLayers;    // chain_ot
    int allowAnyOuterTracker = 0;                // ot_any
    int maxPixelLayersForExtraOuterTracker = 0;  // ot_max_pix
    // prune: layer combinations (mkFit layer ids, crossing order) the finder does not search, with or without a
    // window table: 3 layers = stage c's third hit after the doublet, 4 = stage d's fourth hit after the triplet (a
    // candidate is forwarded as for a miss). With +z pixel discs also mirrored to -z (+22).
    std::vector<std::vector<int>> prunedCombinations;

    // ---- the chain finder's fake cuts and triplet seeds
    float fakeScoreCut = 0;                                                    // fk_score
    float fakeScoreCutForward = 0;                                             // fk_score_fwd
    float forwardMinAbsEta = 99;                                               // fk_eta_fwd
    int useShapeCut = 0;                                                       // fk_shape
    std::vector<ShapeWindow> shapeWindows;                                     // shape_win
    float ot2WindowScale = 0;                                                  // fk_ot2
    std::array<float, 4> ot2Windows = {-8.1e-4f, 5.92e-3f, 0.3084f, 0.0909f};  // ot2_win
    float ot2PhiWindowMin = 1.31e-3f;                                          // ot2_phimin
    float ot1WindowScale = 0;                                                  // fk_ot1
    float ot1MaxAbsEta = 99;                                                   // fk_ot1_eta
    float ot1ConfirmationGate = 0;                                             // fk_ot1_gate
    float ot2ConfirmationSigma = 0;                                            // fk_ot2_sig
    float ot2ConfirmationGate = 0;                                             // fk_ot2_gate
    // > 0: the OT2-P confirmation of a quad ending on OT1-P with three pixel hits takes an OT2-P hit within the
    // extension's local windows (ext_wx, ext_wy) times this (instead of ot2_win, fk_ot2 and fk_ot2_sig)
    float ot2LocalWindowScale = 0;  // fk_ot2_local
    // the OT hit confirmation (each off at 0): fk_ot1 / fk_ot2 with a chi2 of the local residual in the hit's module
    // frame (the helix onto the sensor plane) instead of dphi, dz at the hit's radius; > 0: the cut in sigma
    // (fk_ot1_lsig, fk_ot2_lsig), sigma_x = (a + b / pT)(1 + c |cot|) (fk_ot_lsx = [a, b, c], cm), sigma_y (fk_ot1_lsy,
    // fk_ot2_lsy, cm); the gates fk_ot1_gate / fk_ot2_gate as before
    float ot1LocalSigma = 0;                                   // fk_ot1_lsig
    float ot2LocalSigma = 0;                                   // fk_ot2_lsig
    std::vector<float> otLocalSigmaX = {0.02f, 0.03f, 0.75f};  // fk_ot_lsx
    float ot1LocalSigmaY = 0.08f;                              // fk_ot1_lsy
    float ot2LocalSigmaY = 0.11f;                              // fk_ot2_lsy
    // fk_ot2_lsig: > 0: the quads exempted by fk_ot2_gate still need a hit within this many sigma (as the global rule's
    // ot2_win box applies to them); 0: no hit needed
    float ot2LocalGateSigma = 0;   // fk_ot2_lgate
    int emitTriplets = 0;          // triplet_seeds
    float tripletMinAbsCot = 0.f;  // trip_cot_lo
    float tripletMaxAbsCot = 0.f;  // trip_cot_hi

    // ---- the cleaning of the quads and triplets
    int quadMinSharedHitsToDrop = 0;     // dedup (0: no quad cleaning)
    int innermostFirst = 0;              // dedup_inner
    int tripletMinSharedHitsToDrop = 2;  // trip_clean_min
    int tripletVetoBySingleQuad = 0;     // trip_veto_quad: 1: a single kept quad must hold trip_clean_min hits
    // the triplet veto (off at 0): only a kept quad with a fake score below trip_veto_fk and
    // a kept better triplet with a score below trip_veto_tsc hold hits for the triplet veto (0: all)
    float tripletVetoMaxQuadFakeScore = 0;  // trip_veto_fk
    float tripletVetoMaxTripletScore = 0;   // trip_veto_tsc
    int minPixelHits = 0;                   // min_pix
    // > 0: a quad whose third hit is on an outer-tracker layer needs only this many pixel hits (instead of min_pix);
    // a doublet goes on to an outer-tracker third hit only if it found no third hit at its current position, these
    // quads (>= 2 outer-tracker hits) do not veto triplets in the cleaning, and the final seeds with fewer than min_pix
    // pixel hits do not count in the classifier's sharing features (JSON: min_pix_ot)
    int minPixelHitsOuterTrackerThird = 0;
    // clf_lorentz, pixels: the classifier's expected cluster size x on the barrel pixel layers 2-4 gets this Lorentz
    // drift width added (pix_dsx features; 0: none, as the first 6L classifier was trained)
    float classifierLorentzSizeX = 0;
    float d0MaxOuterTrackerThird = 0;          // d0_max_ot, cm (> 0): stage c's d0 term on an outer-tracker position
    float tripletMlpCut = 0;                   // trip_dnn
    int tripletMlpRescueOuterTrackerHits = 0;  // trip_dnn_ext
    float tripletMlpRescueMinScore = 0;        // trip_dnn_ext_min

    // ---- the stage-c triplet network (MkSeederStageCNet.h): its logit cuts per |eta| bin (0-0.8-1.6-2.4-3.2-); JSON
    // trip_net_c, absent: off
    std::array<float, 5> stageCNetCuts = kStageCNetCutsOff;

    // ---- the duplicate cleanings of the fitted seeds (the second one after the pixel fill)
    float dupMaxSigma = 0;            // dup_sig (> 0 required)
    int dupMinSharedHits = 1;         // dup_shared
    int dupMinSharedHitsTriplet = 0;  // dup_shared_trip (> 0): dup_shared for a seed of 3 hits (the removed one)
    int dupNoShareTriplets = 1;       // dup_ns_trip: 0: a seed of 3 hits is not removed by a no-share neighbour
    float dupMaxDPhi = 0.05f;         // dup_dphi, rad
    float dupMaxDZ = 1.0f;            // dup_dz, cm
    float dupHighPtThreshold = 0;     // dup_hpt_pt, GeV
    float dupHighPtDPhi0 = 0.0008f;   // dup_hpt_dphi0
    float dupHighPtDPhi1 = 0.03f;     // dup_hpt_dphi1
    float dupHighPtDZ = 0.05f;        // dup_hpt_dz
    int dupMergeMode = 0;             // dup_merge
    // optional (off by default): the merge's OT hits only when (bits of dup_merge_ot) 1: within
    // dup_merge_ot_sig sigma of the owner's circle and r-z line (the chain finder's OT confirmation sigmas), 2: a marked
    // OT1-3 P hit with a q/pT estimate compatible with the owner's (ps_curv), 4: the removed seed's score is at most
    // dup_merge_ot_score, 8: never; in the passes of dup_merge_ot_pass (bit 1 the first cleaning, bit 2 the second)
    int dupMergeOuterTracker = 0;              // dup_merge_ot
    float dupMergeOuterTrackerSigma = 3;       // dup_merge_ot_sig
    float dupMergeOuterTrackerMaxScore = 1e9;  // dup_merge_ot_score
    int dupMergeOuterTrackerPasses = 3;        // dup_merge_ot_pass
    float dupNoShareMinAbsEta = 99;            // dup_noshare_eta
    float dupNoShareMaxDEta = 0.02f;           // dup_deta
    float dupNoShareMaxSigma = 0;              // dup_ns_sig
    float dupNoShareMaxDPhi = 0;               // dup_ns_dphi
    float dupNoShareMaxDZ = 0;                 // dup_ns_dz
    int dupNoShareMerge = 1;                   // dup_ns_merge
    float dupNoShareMaxResidual = 0;           // dup_ns_res, cm

    // ---- the outer-tracker extension
    int extendIntoOuterTracker = 0;                       // ext_ot
    float extensionSigma = 3;                             // ext_sig
    int extensionNLayers = 3;                             // ext_nlay
    int extensionAnyLayer = 0;                            // ext_any
    int extensionMinHits = 0;                             // ext_min_hits
    int extensionLocalWindows = 0;                        // ext_local
    std::vector<float> extensionWindowX = {0.02f, 0.1f};  // ext_wx
    std::vector<float> extensionWindowY = {0.3f, 0.2f};   // ext_wy
    int keepTripletsWithoutOuterTracker = 0;              // trip_keep_noot
    int extensionBranches = 0;                            // ext_branch
    int extensionRedo = 0;                                // ext_redo
    int extensionBranchMaxPixelHits = 0;                  // ext_branch_maxpix
    int extensionMinOuterTrackerHits = 0;                 // ext_min_ot
    // optional (off by default): ext_min_ot counts only the marked OT1-3 P hits; the extension
    // path choice by the sum of (score - ext_path_bonus) over the added hits instead of the most hits first
    int extensionMinOuterTrackerHitsMarked = 0;  // ext_min_ot_mk
    float extensionPathBonus = 0;                // ext_path_bonus

    // ---- the pixel fill
    int pixelFillMode = 0;         // pix_fill
    int pixelFillOverlaps = 0;     // pix_fill_ov
    float pixelFillSigma = 3;      // pix_fill_sig
    float pixelFillSigmaDisc = 0;  // pix_fill_sig_disc: the score cut of the disc layers (0: pix_fill_sig)
    std::vector<float> pixelFillWeights = {0.002f, 0.004f, 0.004f, 0.006f};  // pix_fill_w

    // ---- merged-cluster candidates (each off at 0): a pixel hit whose cluster has more charge per
    // path length, or is longer, than one particle crossing the module along the line from the beam spot gives; the
    // tag is a column of the hits (MkSeederHitsSoA::mergedTag), counted per seed (MkSeederSeedFeaturesSoA)
    int mergedTagMode = 0;              // mrg_tag: 1 = tag the pixel hits
    float mergedChargeMpv = 8.1667e5f;  // mrg_q_mpv: the most probable charge per path length of one particle, e/cm
    float mergedChargeCut = 2.0f;       // mrg_q: the charge per path length over mrg_q_mpv at or above: bit 0
    float mergedSizeXCut = 3.f;         // mrg_dsx: the size along the local x minus the expected at or above: bit 1
    float mergedSizeYCut = 4.f;         // mrg_dsy: the same along the local y: bit 2
    // > 0: a tagged hit's covariance gets (mrg_err pitch max(size excess, 1) / sqrt(12))^2 added in its local x and y
    float mergedErrorScale = 0;  // mrg_err
    // > 1: the stage d windows (and the fake-score normalisation) times this for a triplet with a hit tagged for both
    // charge and size (isStrongMergedCandidate)
    float mergedWindowScale = 0;  // mrg_win
    // 1: a strong candidate hit does not count as a shared hit in the duplicate cleanings (two particles may share it)
    int mergedNoShare = 0;  // mrg_share
    // 1: a hit tagged for its charge (bit 0) is exempt from the upper limit of the barrel cluster-length band (fk_shape)
    int mergedShapeExempt = 0;  // mrg_shape

    // ---- the seed fit (JSON: fit)
    std::array<float, 6> fitPriorSigma = {1.0f, 1.0f, 1.0f, 1.0f, 0.03f, 0.03f};  // fit.prior_sigma
    float fitPriorScale = 1.0f;                                                   // fit.prior_scale
    float fitBeamlineErrorRescale = 100.f;                                        // fit.bl_err_rescale

    // ---- the PS-module confirmation of the OT1-3 P hits (each off at 0)
    PSMarkParameters psMark;          // ps_mark, ps_pt, ps_d0, ps_w0, ps_hl, ps_dv, ps_cnt_pt, ps_mrg_n
    int psRequireOT1 = 0;             // ps_ot1: the OT1-P confirmation (fk_ot1) takes marked hits only
    int psRequireOT2 = 0;             // ps_ot2: the OT2-P confirmation (fk_ot2) takes marked hits only
    int psRequireChain = 0;           // ps_chain: the chain finder's OT1-3 P hits must be marked
    int psRequireExtension = 0;       // ps_ext: 1 the extension's OT1-3 P candidates must be marked; 2 (3): have an S
                                      // match in the helix's q/pT range (or a neighbouring one; psCountInRange())
    float psCurvatureSigma = 0;       // ps_curv: > 0: a marked hit's q/pT estimate within this many sigma of the seed's
    float psCurvatureSigmaChain = 0;  // ps_curv_chain: the same for the OT1-3 P hits of the chain finder's stage d
    int psAlternatives = 0;           // ps_alt: bits, a candidate failing ps_curv is still taken (psExtensionReject())
    float psCurvatureResolution = 0.25f;  // ps_curv_res, 1/GeV: the q/pT estimate's resolution at OT1 (scaled by r)
    // ps_seed: the chain finder's quads with OT1-3 P hits and a helix pT >= ps_seed_pt are dropped unless one of their
    // OT1-3 P hits is marked (and its q/pT estimate within ps_seed_curv sigma of the helix's; 0: the mark only):
    // 1 all such quads, 2 only the quads with two pixel hits (min_pix_ot)
    int psSeed = 0;            // ps_seed
    float psSeedPtMin = 1.0f;  // ps_seed_pt, GeV
    float psSeedSigma = 3.0f;  // ps_seed_curv
    // ps_seed_rng: 1 (2) a quad's OT1-3 P hit confirms it if it has an S match in the q/pT range of the helix (or a
    // neighbouring one; MkSeederPSMark.h psCountInRange(), with the low-pT counts of ps_low_pt), instead of the mark
    // and ps_seed_curv
    int psSeedRange = 0;
    // ps_c2: stage c takes an OT1-3 P third hit of a two-pixel start only if marked, when the three hits' circle has
    // pT >= ps_seed_pt
    int psStageC2 = 0;
  };  // struct MkSeederConfig

  // The configuration from a JSON file (the keys as at the members above); throws for a removed option set to
  // another value than its "off" value, and for values the seeder does not support
  MkSeederConfig readMkSeederConfig(const std::string &jsonFile);

  // The weights of the triplet MLP from a JSON file (featureMean, featureScale, hidden1Weights, hidden1Bias,
  // hidden2Weights, hidden2Bias, outputWeights, outputBias; nFeatures, nHidden1, nHidden2 as checks)
  MkSeederTripletMlpWeights readMkSeederTripletMlpWeights(const std::string &jsonFile);

}  // namespace mkseeder

#endif
