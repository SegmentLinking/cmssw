#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDeviceTypes_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDeviceTypes_h

// The seeder's chain finder and the steps after it as trivially copyable kernel
// arguments: the views of the static tables (MkSeederESData), of the event's hits and tables (MkSeederEventHitsSoA,
// MkSeederEventTablesSoA) and of the intermediate results (MkSeederQuadsSoA, MkSeederSeedsSoA; the chain finder's
// candidates: MkSeederQueues.h; the steps' scratch: MkSeederScratchSoA.h), with the kernels' working copies of a
// candidate, a quad, a seed and its state. Host and device code; no alpaka, no mkFit types.

#include <cstddef>
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederChainSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederCounters.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederEventHitsSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederEventTablesSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederGapMapSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederLayerGeometrySoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederModulesSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederPSMark.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederPatternsSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederQuadsSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedFeaturesSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederShapeSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTripletMlpWeights.h"
// the per-seed math of the triplet MLP and the post step (FillCircle)
#include "MkSeederPostMath.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // the views of the static tables (MkSeederESData) and of the event's data the kernels read: without the range
  // checks (the indices come from the tables themselves; a check per access costs a branch in the finder's inner
  // loops)
  template <typename TLayout>
  using TableView = typename TLayout::template ConstViewTemplate<cms::soa::RestrictQualify::Default,
                                                                 cms::soa::RangeChecking::disabled>;
  template <typename TLayout>
  using MutableTableView =
      typename TLayout::template ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::disabled>;
  using LayerGeometryView = TableView<::mkseeder::MkSeederLayerGeometrySoA>;
  using ChainPositionsView = TableView<::mkseeder::MkSeederChainPositionsSoA>;
  using ChainStartsView = TableView<::mkseeder::MkSeederChainStartsSoA>;
  using ChainHolesView = TableView<::mkseeder::MkSeederChainHolesSoA>;
  using WindowTablesView = TableView<::mkseeder::MkSeederWindowTablesSoA>;
  using EtaWindowsView = TableView<::mkseeder::MkSeederEtaWindowsSoA>;
  using ComboIndexView = TableView<::mkseeder::MkSeederComboIndexSoA>;
  using ShapeLayersView = TableView<::mkseeder::MkSeederShapeLayersSoA>;
  using ShapeBandsView = TableView<::mkseeder::MkSeederShapeBandsSoA>;
  using GapMapLayersView = TableView<::mkseeder::MkSeederGapMapLayersSoA>;
  using GapMapLaddersView = TableView<::mkseeder::MkSeederGapMapLaddersSoA>;
  using GapMapZGapsView = TableView<::mkseeder::MkSeederGapMapZGapsSoA>;
  using GapMapBitsView = TableView<::mkseeder::MkSeederGapMapBitsSoA>;
  using GapMapPhiStartsView = TableView<::mkseeder::MkSeederGapMapPhiStartsSoA>;
  using GapMapLadderIndexView = TableView<::mkseeder::MkSeederGapMapLadderIndexSoA>;
  using ModulesView = TableView<::mkseeder::MkSeederModulesSoA>;
  using MaterialView = TableView<::mkseeder::MkSeederMaterialSoA>;
  using PatternsView = TableView<::mkseeder::MkSeederPatternsSoA>;
  // the event's
  using HitsView = TableView<::mkseeder::MkSeederHitsSoA>;
  using ClusterOrderView = TableView<::mkseeder::MkSeederClusterOrderSoA>;
  using OTHitsView = TableView<::mkseeder::MkSeederOTHitsSoA>;
  using BinStartsView = TableView<::mkseeder::MkSeederBinStartsSoA>;
  using EventLayersView = TableView<::mkseeder::MkSeederLayersSoA>;
  using EventLayer = EventLayersView::const_element;
  using GapRaysView = TableView<::mkseeder::MkSeederGapRaysSoA>;
  using StartPairWorkView = TableView<::mkseeder::MkSeederStartPairWorkSoA>;
  using PixelClustersView = TableView<::mkseeder::MkSeederPixelClustersSoA>;
  using OTClustersView = TableView<::mkseeder::MkSeederOTClustersSoA>;
  using HitsCpuCacheView = TableView<::mkseeder::MkSeederHitsCpuCacheSoA>;
  using OTHitFramesView = TableView<::mkseeder::MkSeederOTHitFramesSoA>;
  using LayersCpuCacheView = TableView<::mkseeder::MkSeederLayersCpuCacheSoA>;
  using PSStripHitsView = TableView<::mkseeder::MkSeederPSStripHitsSoA>;
  using PSStripStartsView = TableView<::mkseeder::MkSeederPSStripStartsSoA>;
  using ::mkseeder::MkSeederChainParameters;
  using ::mkseeder::MkSeederTripletMlpWeights;

  // the views of the intermediate results (written and read by the kernels, also within one kernel by other
  // threads): neither restrict-qualified nor range checked, as plain pointers
  template <typename TLayout>
  using ScratchView =
      typename TLayout::template ViewTemplate<cms::soa::RestrictQualify::disabled, cms::soa::RangeChecking::disabled>;
  template <typename TLayout>
  using ScratchConstView = typename TLayout::template ConstViewTemplate<cms::soa::RestrictQualify::disabled,
                                                                        cms::soa::RangeChecking::disabled>;
  using QuadsView = ScratchView<::mkseeder::MkSeederQuadsSoA>;
  using QuadsConstView = ScratchConstView<::mkseeder::MkSeederQuadsSoA>;
  using KeptSeedsView = ScratchView<::mkseeder::MkSeederKeptSeedsSoA>;
  using KeptSeedsConstView = ScratchConstView<::mkseeder::MkSeederKeptSeedsSoA>;
  using SeedHitsView = ScratchView<::mkseeder::MkSeederSeedHitsSoA>;
  using SeedHitsConstView = ScratchConstView<::mkseeder::MkSeederSeedHitsSoA>;
  using SeedStatesView = ScratchView<::mkseeder::MkSeederSeedStatesSoA>;
  using SeedStatesConstView = ScratchConstView<::mkseeder::MkSeederSeedStatesSoA>;
  using SeedHelicesView = ScratchView<::mkseeder::MkSeederSeedHelicesSoA>;
  using SeedHitListView = ScratchView<::mkseeder::MkSeederSeedHitListSoA>;
  using SeedHelicesConstView = ScratchConstView<::mkseeder::MkSeederSeedHelicesSoA>;
  using SeedQualityView = ScratchView<::mkseeder::MkSeederSeedQualitySoA>;
  using SeedQualityConstView = ScratchConstView<::mkseeder::MkSeederSeedQualitySoA>;
  using SeedClusterIndicesView = ScratchView<::mkseeder::MkSeederSeedClusterIndicesSoA>;
  using SeedPerigeeView = ScratchView<::mkseeder::MkSeederSeedPerigeeSoA>;
  using SeedPixelStatesView = ScratchView<::mkseeder::MkSeederSeedPixelStatesSoA>;
  using SeedFeaturesView = ScratchView<::mkseeder::MkSeederSeedFeaturesSoA>;
  using ::mkseeder::MkSeederCleanCounters;
  using ::mkseeder::MkSeederDupCounters;
  using ::mkseeder::MkSeederFinderCounters;

  // a hit's covariance column (MkSeederHitsSoA, MkSeederOTHitsSoA) as 6 floats
  template <typename TView>
  ALPAKA_FN_HOST_ACC inline void hitCovariance(const TView &view, uint32_t i, float *cov) {
    const auto c = view.covariance(i);
    for (int k = 0; k < 6; ++k)
      cov[k] = c(k);
  }

  // module m's plane as mkfit::ModuleInfo's 9 floats: the position, the z direction, the x direction
  ALPAKA_FN_HOST_ACC inline void modulePlane(const ModulesView &modules, uint32_t m, float *plane) {
    const auto row = modules[m];
    for (int d = 0; d < 3; ++d)
      plane[d] = row.position()(d), plane[3 + d] = row.zDirection()(d), plane[6 + d] = row.xDirection()(d);
  }
  // pixel module m's frame in 12 floats: its rotation's rows x, y, z, the pitches x, y and the thickness
  ALPAKA_FN_HOST_ACC inline void pixelModuleFrame(const ModulesView &modules, uint32_t m, float *frame) {
    const auto row = modules[m];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        frame[3 * i + j] = row.rotation()(i, j);
    frame[9] = row.pitchX(), frame[10] = row.pitchY(), frame[11] = row.thickness();
  }

  // The hits' tags the binning computes (KernelScatterBinnedHits): a pixel hit's merged-cluster candidate bits
  // (mrg_tag; MkSeederConfig mrg_*) and an OT1-3 P hit's PS-module confirmation (ps_mark; MkSeederPSMark.h)
  struct HitTagInputs {
    int mergedTag = 0;
    float mergedChargeMpv = 0, mergedChargeCut = 0, mergedSizeXCut = 0, mergedSizeYCut = 0, mergedErrorScale = 0;
    uint32_t nPixelClusters = 0;  // the pixel clusters with a charge and sizes (MkSeederPixelClustersSoA rows)
    int psMark = 0;
    ::mkseeder::PSMarkParameters psParameters;
    float bz = 0;  // the field at the origin (T)
    // per P layer (mkFit 4, 6, 8) its S layer's first row of psStripStarts
    uint32_t psFirstStart[3] = {0, 0, 0};
    float beamX = 0, beamY = 0, beamZ = 0;
    ModulesView modules;
    PSStripHitsView psStripHits;
    PSStripStartsView psStripStarts;
  };

  // a found quad in the kernels (a row of MkSeederQuadsSoA; layers[3] = -1: a triplet seed), hits as the layers'
  // external (cluster) indices
  struct Quad {
    int16_t layers[4];
    uint32_t hits[4];
    float score, fakeScore;
    uint8_t tripletFailedMlp = 0;
  };
  ALPAKA_FN_HOST_ACC inline void storeQuad(QuadsView quads, uint32_t i, const Quad &q) {
    auto row = quads[i];
    for (int k = 0; k < 4; ++k)
      row.layers()(k) = q.layers[k], row.hits()(k) = q.hits[k];
    row.score() = q.score, row.fakeScore() = q.fakeScore, row.tripletFailedMlp() = q.tripletFailedMlp;
  }

  // The chain finder's constants that depend on the event (its beam spot and hit counts)
  struct ChainEventParameters {
    int nWorkItems;                 // stage b's work items: the a hits of the start pairs whose two layers are binned
    float zVertexLow, zVertexHigh;  // the beam spot's z -+ MkSeederChainParameters::zVertexHalfWidth
    float zVertexLowOuter, zVertexHighOuter;  // -+ zVertexHalfWidthOuter (zv_outer; its inner hits only)
    // the beam spot (mkfit::BeamSpot): the hits' xBeam, yBeam are relative to its line at the hit's z
    float beamSpotX, beamSpotY, beamSpotZ, beamSpotDxDz, beamSpotDyDz;
  };

  // The device view of one side: the static tables (MkSeederESData) and the event's hits.
  struct ChainFinderInputs {
    MkSeederChainParameters chain;
    ChainEventParameters chainEvent;
    EventLayersView layers;  // by mkFit layer id, nLayers
    int nLayers;
    LayerGeometryView layerGeometry;
    ChainPositionsView chainPositions;
    ChainStartsView chainStarts;
    StartPairWorkView startPairs;  // per start pair: its first work item (the prefix of the a layers' hits)
    ChainHolesView chainHoles;
    WindowTablesView windowTables;
    EtaWindowsView etaWindows;
    ComboIndexView windowTableByTriplet, windowTableByQuad;
    ShapeLayersView shapeLayers;
    ShapeBandsView shapeBands;
    GapMapLayersView gapMapLayers;
    GapMapLaddersView gapMapLadders;
    GapMapZGapsView gapMapZGaps;
    GapMapBitsView gapMapBits;
    GapMapPhiStartsView gapMapPhiStarts;
    GapMapLadderIndexView gapMapLadderIndex;
    // the event's gap rays (the beam spot's): per gap map layer kGapRayNPhiBins rows (SensorGapMap::ray_table)
    GapRaysView gapRays;
    // the hits in bin order, the layers' bin starts
    HitsView hits;
    BinStartsView binStarts;
    const MkSeederTripletMlpWeights *tripletMlpWeights;
    // CPU backends with the triplet MLP: per hit its azimuth around the beam spot (phiBeam) for the MLP's features
    HitsCpuCacheView cpuHits;
    bool hasPhiBeam;
    // the PS-module confirmation (ps_mark): the OT1-3 P hits' marks (otHits rows: layers.firstOTHit + the hit's index in
    // its layer; also their covariances for fk_ot2_local) and the selections that use them, bits 1 ps_ot1 (the fk_ot1 confirmation), 2 ps_ot2 (fk_ot2), 4 ps_chain
    // (the chain's OT1-3 P hits); 0: off
    OTHitsView otHits;
    int psMode = 0;
    // ps_curv_chain > 0: a marked OT1-3 P hit taken by stage d needs its q/pT estimate within psCurvatureSigma x
    // (psCurvatureResolution x 23 cm / r) of the stage's helix
    float psCurvatureSigma = 0, psCurvatureResolution = 0;
    // ps_seed, ps_seed_pt, ps_seed_curv (stage d's quads), ps_c2 (stage c's OT third hits of two-pixel starts)
    int psSeed = 0, psStageC2 = 0, psSeedRange = 0, psLow = 0;
    float psSeedPtMin = 0, psSeedSigma = 0;
  };

  // ---- the steps after the seed fit on the device (MkSeederPostKernels.h)
  constexpr int kMaxLayerIds = 80;  // mkFit layer ids
  // the SeederConfig of the OT extension and the pixel fill
  struct PostParameters {
    int extendIntoOT, nExtensionLayers, extensionAnyLayer, extensionMinHits, extensionLocal, keepTripletsWithoutOT;
    // the extension options (SeederConfig; 0 = off), described at extendWithBranching()
    int extensionBranching, extensionRedo, extensionMinOTHits, tripletMlpExtensionHits,
        extensionBranchingMaxPixelLayers, extensionMinOTMarked;
    float extensionPathBonus;
    float extensionSigma, extensionWindowX[2], extensionWindowY[2];
    int pixelFillMode, pixelFillOverlaps;
    float pixelFillSigma, pixelFillWeights[4], pixelFillSigmaDisc;
  };
  struct PostInputs {
    PostParameters parameters;
    LayerGeometryView layerGeometry;  // by mkFit layer id
    int nLayers;
    EventLayersView layers;  // by mkFit layer id: firstOTHit; the pixel layers with hits (pixelLayerWithHits)
    HitsView hits;           // the hits in bin order: x, y, detIdInLayer
    OTHitsView otHits;       // OT1-3 P: the covariances, layers.firstOTHit + the hit's index in its layer
    // the CPU backends, made once per event (else computed per use; hasCpuCache): per hit atan2(y, x) and
    // sqrt(x^2 + y^2); per mkFit layer id the windows' phi pad (layerPhiPad); per OT1-3 P hit (otHits' rows) the
    // module frame of extensionLocalResidual (hasOTFrames: there are such hits)
    HitsCpuCacheView cpuHits;
    LayersCpuCacheView cpuLayers;
    OTHitFramesView cpuOTHitFrames;
    bool hasCpuCache, hasOTFrames;
    float beamSpotRadius;      // |beam spot (x, y)|: the bins are beam-relative, the cuts global
    int nPixelLayersWithHits;  // the pixel layers with hits (layers.nPixelLayersWithHits())
    // the PS-module confirmation (ps_ext, ps_curv): the extension's OT1-3 P candidates must be marked; > 0: a marked
    // candidate's q/pT estimate within psCurvatureSigma x (psCurvatureResolution at OT1-P, scaled by 23 cm / r) of the
    // helix's
    int psExtension = 0;
    float psCurvatureSigma = 0, psCurvatureResolution = 0;
    int psAlternatives = 0;  // ps_alt (psExtensionReject())
    int psLow = 0;           // ps_low_pt > 0: the OT hits' low-pT counts are filled (psCountsLow)
  };
  // a seed in the post step's kernels (a row of MkSeederSeedHitsSoA): its hits (mkFit layer, bin-order hit row), cot;
  // keep / extended / filled (out); in: tfail, a triplet that failed the triplet MLP (trip_dnn_ext: dropped unless
  // the extension gives it trip_dnn_ext OT hits); the hits the extension added (next, elay, edh)
  struct WorkingSeed {
    int nHits;
    float cot;
    int16_t layers[::mkseeder::kMaxSeedHits];
    uint32_t hitRows[::mkseeder::kMaxSeedHits];
    int8_t isKept, isExtended, isFilled;
    int8_t tripletFailedMlp;
    int8_t nExtensionHits;
    int16_t extensionLayers[3];
    uint32_t extensionHits[3];
  };
  static_assert(::mkseeder::kMaxExtensionHits == 3);
  ALPAKA_FN_HOST_ACC inline void loadSeed(const SeedHitsConstView &seeds, uint32_t i, WorkingSeed &s) {
    const auto row = seeds[i];
    s.nHits = row.nHits(), s.cot = row.cot();
    for (int k = 0; k < s.nHits; ++k)
      s.layers[k] = row.hitLayers()(k), s.hitRows[k] = row.hitRows()(k);
    s.isKept = row.isKept(), s.isExtended = row.isExtended(), s.isFilled = row.isFilled();
    s.tripletFailedMlp = row.tripletFailedMlp(), s.nExtensionHits = row.nExtensionHits();
    for (int k = 0; k < s.nExtensionHits; ++k)
      s.extensionLayers[k] = row.extensionLayers()(k), s.extensionHits[k] = row.extensionHits()(k);
  }
  // the hits and the flags (the label is not touched)
  ALPAKA_FN_HOST_ACC inline void storeSeed(SeedHitsView seeds, uint32_t i, const WorkingSeed &s) {
    auto row = seeds[i];
    row.nHits() = s.nHits, row.cot() = s.cot;
    for (int k = 0; k < s.nHits; ++k)
      row.hitLayers()(k) = s.layers[k], row.hitRows()(k) = s.hitRows[k];
    row.isKept() = s.isKept, row.isExtended() = s.isExtended, row.isFilled() = s.isFilled;
    row.tripletFailedMlp() = s.tripletFailedMlp, row.nExtensionHits() = s.nExtensionHits;
    for (int k = 0; k < s.nExtensionHits; ++k)
      row.extensionLayers()(k) = s.extensionLayers[k], row.extensionHits()(k) = s.extensionHits[k];
  }
  // a seed's hits (n, mkFit layers, bin-order rows) into arrays: the fits' input
  template <typename TSeedHits>
  ALPAKA_FN_HOST_ACC inline int loadSeedHits(const TSeedHits &seeds, uint32_t i, int16_t *lay, uint32_t *dh) {
    const auto row = seeds[i];
    const int n = row.nHits();
    for (int k = 0; k < n; ++k)
      lay[k] = row.hitLayers()(k), dh[k] = row.hitRows()(k);
    return n;
  }
  // a seed's state in the kernels (a row of MkSeederSeedStatesSoA): in, its state at its last hit; out, refit 1 if
  // its hits changed and their refit worked (the state is then the refit's, else the input's), bl 1 with the
  // beam-line state of its final hits and state
  ALPAKA_FN_HOST_ACC inline void loadState(
      const SeedStatesConstView &states, uint32_t i, float *par, float *err, int &chg, float &chi2) {
    const auto row = states[i];
    for (int d = 0; d < 6; ++d)
      par[d] = row.parameters()(d);
    for (int d = 0; d < 21; ++d)
      err[d] = row.covariance()(d);
    chg = row.charge(), chi2 = row.chi2();
  }
  ALPAKA_FN_HOST_ACC inline void storeState(
      SeedStatesView states, uint32_t i, const float *par, const float *err, int chg, float chi2) {
    auto row = states[i];
    for (int d = 0; d < 6; ++d)
      row.parameters()(d) = par[d];
    for (int d = 0; d < 21; ++d)
      row.covariance()(d) = err[d];
    row.charge() = chg, row.chi2() = chi2;
  }
  ALPAKA_FN_HOST_ACC inline void storeBeamlineState(
      SeedStatesView states, uint32_t i, const float *par, const float *err, int chg) {
    auto row = states[i];
    for (int d = 0; d < 6; ++d)
      row.beamlineParameters()(d) = par[d];
    for (int d = 0; d < 21; ++d)
      row.beamlineCovariance()(d) = err[d];
    row.beamlineCharge() = chg;
  }

  // the pixel fill in steps (MkSeederPostKernels.h): per seed, its circle and line; per seed and pixel layer, the
  // layer's best hits
  struct FillPreparation {
    int hasWork;  // 0: nothing to fill
    int nPixelHits, nOuterHits;
    int16_t pixelLayers[::mkseeder::kMaxSeedHits], outerLayers[::mkseeder::kMaxSeedHits];
    uint32_t pixelHits[::mkseeder::kMaxSeedHits], outerHits[::mkseeder::kMaxSeedHits];
    FillCircle circle;
    float zAtFirstHit, cot, s_max, rA, srp, sq, sig2;
  };
  // the best hit of a layer (kb, at s = sb) and the best on another module (kb2, sb2); -1: none
  struct FillCandidate {
    int bestHit, secondBestHit;
    float bestPathLength, secondBestPathLength;
  };

  // ---- the seed fits on the device (MkSeederFitKernels.h): the fit, the refit and the beam-line state, with the
  // propagation to a plane and the Kalman update on it of RecoTracker/MkFitAlpaka
  struct FitParameters {
    // mkfit::Config's field parametrisation (bFieldFromZR) and the backward-fit propagation flags
    float fieldC1, fieldB0, fieldB1, fieldA;
    float bField;  // mkfit::Config::Bfield (T): the field of the helix <-> pT conversions of the seed fits and fallbacks
    int useParametrizedField, applyMaterial;
    // the seed fit's prior (MkSeederConfig fitPriorSigma, fitPriorScale) and the beam-line refit's error rescale
    float priorSigma[6], priorScale, beamlineErrorRescale;
  };
  struct FitInputs {
    int refitEnabled;
    FitParameters parameters;
    MaterialView material;            // TrackerInfo's material grid
    ModulesView modules;              // the module planes (mkfit::ModuleInfo)
    LayerGeometryView layerGeometry;  // by mkFit layer id: the layer's first module
    HitsView hits;                    // the hits in bin order: x, y, z, covariance, detIdInLayer
  };
  // the final seeds' states at the beam line (MkSeederPcaKernels.h): the inputs (the final seeds, their states and
  // quality: the second duplicate cleaning's output, *count of them; the module frames; the beam spot, double as
  // reco::BeamSpot) and the outputs (the classifier's state and seeder features, the seeds' perigees)
  struct PcaInputs {
    SeedHitsConstView seeds;
    SeedStatesConstView states;
    SeedQualityConstView quality;
    const uint32_t *count;
    uint32_t cap;
    ModulesView modules;
    LayerGeometryView layerGeometry;
    HitsView hits;       // detIdInLayer
    double beamSpot[5];  // x0, y0, z0, dxdz, dydz
    SeedFeaturesView features;
    SeedPerigeeView perigee;
  };

  // the inputs of the classifier's other features (hits, circle, clusters, sharing) of the final seeds (after
  // PcaInputs's: isDropped, pt), their output
  struct ClassifierFeatureInputs {
    SeedHitsConstView seeds;
    const uint32_t *count;
    uint32_t cap, nTot;
    LayerGeometryView layerGeometry;  // by mkFit layer id: isPixel
    // the hits in bin order: x, y, z, clusterIndex (the index in the layer's HitVec, the cluster index),
    // detIdInLayer, clusterRows (the OT cluster size, up to 32)
    HitsView hits;
    PixelClustersView pixelClusters;  // charge, sizeX, sizeY of the first nPixelClusters
    uint32_t nPixelClusters;
    OTClustersView otClusters;  // size of the first nOTClusters
    uint32_t nOTClusters;
    ModulesView modules;                         // per module (firstModule + detid): the pixel frames
    double fieldAtOrigin, beamSpotX, beamSpotY;  // the field at the origin (Tesla), the beam spot
    int shareMinHits;
    int shareMinPixelHits;
    float lorentzSizeX;  // clf_lorentz: added to the expected cluster size x on the barrel pixel layers 2-4  // > 0: also only the seeds with this many pixel hits count in the sharing (min_pix_ot)
    SeedFeaturesView features;
    // the PS-module confirmation features (psInputs; psMark 0: kSeedFeatureNone): the OT1-3 P hits' marks (otHits rows:
    // layers.firstOTHit + the hit's index in its layer), the seeds' perigees (charge, momentum), ps_curv_res
    int psMark = 0;
    EventLayersView layers;
    OTHitsView otHits;
    SeedPerigeeView perigee;
    float psCurvatureResolution = 0.25f;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
