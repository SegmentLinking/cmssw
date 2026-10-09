#ifndef RecoTracker_MkFitAlpaka_interface_es_ESConfig_h
#define RecoTracker_MkFitAlpaka_interface_es_ESConfig_h

// Flat, trivially copyable copy of the scalar ES configuration of one mkFit iteration (MkFitCore
// mkfit::IterationConfig + SteeringParams + PropagationConfig + the runtime mkfit::Config values).
// Member names follow the MkFitCore classes (m_ prefix dropped). Filled by mkfitdev::fillESDataHost.

#include <cstdint>

namespace mkfitdev {

  namespace es {
    constexpr int kMaxRegions = 8;      // LST step: 5 (m_n_regions); fill throws above this
    constexpr int kMaxPlanLayers = 64;  // LST step: 34 max per region; fill throws above this
    constexpr int kNoModule = -1;       // findModule() result for an unknown detid
  }  // namespace es

  // mkfit::LayerInfo::LayerType_e
  enum class LayerType : int { Undef = -1, Barrel = 0, EndCapPos = 1, EndCapNeg = 2 };

  // Standard functions of IterationConfig, resolved from their registered names (MkStdSeqs.cc,
  // MkSeedPartitioners-phase*.cc). None = empty name (MkFitCore: unset std::function).
  enum class SeedCleaner : int { None = 0, Phase1Default };  // "phase1:default"
  enum class SeedPartitioner : int { None = 0, Phase2_1, Phase2_1_debug, Phase1_0, Phase1_1, Phase1_1_debug };
  enum class CandFilter : int {
    None = 0,
    NHits,         // "phase1:qfilter_n_hits"
    NHitsPixSeed,  // "phase1:qfilter_n_hits_pixseed"
    NLayers,       // "phase1:qfilter_n_layers"
    PixelLessFwd,  // "phase1:qfilter_pixelLessFwd"
    PixelLessBkwd  // "phase1:qfilter_pixelLessBkwd"
  };
  enum class DuplicateCleaner : int {
    None = 0,
    CleanDuplicates,         // "phase1:clean_duplicates"
    SharedHits,              // "phase1:clean_duplicates_sharedhits"
    SharedHitsPixelSeed,     // "phase1:clean_duplicates_sharedhits_pixelseed"
    SharedHitsPixelPriority  // "phase2:clean_duplicates_sharedhits_pixelpriority"
  };
  enum class TrackScorer : int { None = 0, Default };  // "default" and "phase1:default" = trackScoreDefault

  // mkfit::SteeringParams::IterationType_e
  enum IterationType : int { IT_FwdSearch = 0, IT_BkwFit, IT_BkwSearch };

  // mkfit::PropagationFlags (tracker_info back-pointer dropped)
  struct PropFlags {
    bool use_param_b_field;
    bool apply_material;
    bool copy_input_state_on_fail;
  };

  // mkfit::PropagationConfig
  struct PropConfig {
    bool backward_fit_to_pca;
    bool finding_requires_propagation_to_hit_pos;
    PropFlags finding_inter_layer_pflags;
    PropFlags finding_intra_layer_pflags;
    PropFlags backward_fit_pflags;
    PropFlags forward_fit_pflags;
    PropFlags seed_fit_pflags;
    PropFlags pca_prop_pflags;
  };

  // mkfit::IterationParams
  struct IterParams {
    int nlayers_per_seed;
    int maxCandsPerSeed;
    int maxHolesPerCand;
    int maxConsecHoles;
    float chi2Cut_min;
    float chi2CutOverlap;
    float pTCutOverlap;
    bool recheckOverlap;
    bool useHitSelectionV2;
    int minHitsQF;
    float minPtCut;
    unsigned int maxClusterSize;
  };

  // mkfit::SteeringParams of one eta region. Plan index semantics as SteeringParams::make_iterator:
  //   forward search : index fwd_search_pickup .. n_plan-1 (ascending), pickup-only at index == fwd_search_pickup
  //   backward fit   : index n_plan-1 .. bkw_fit_last (descending)
  //   backward search: index bkw_search_pickup .. 0 (descending), pickup-only at index == bkw_search_pickup;
  //                    only if bkw_search_pickup != -1 (has_bksearch_plan)
  struct SteeringRegion {
    int region;
    int n_plan;
    int fwd_search_pickup;
    int bkw_fit_last;
    int bkw_search_pickup;
    TrackScorer track_scorer;  // resolved as MkFitCore: empty name -> the iteration's default scorer
    int layer[es::kMaxPlanLayers];

    constexpr bool has_bksearch_plan() const { return bkw_search_pickup != -1; }

    // SteeringParams::make_iterator / iterator::operator++ as index arithmetic:
    //   for (int i = begin_index(t); i != end_index(t); i += step(t)) { layer[i] ... }
    constexpr int begin_index(IterationType t) const {
      return t == IT_FwdSearch ? fwd_search_pickup : (t == IT_BkwFit ? n_plan - 1 : bkw_search_pickup);
    }
    constexpr int end_index(IterationType t) const {
      return t == IT_FwdSearch ? n_plan : (t == IT_BkwFit ? bkw_fit_last - 1 : -1);
    }
    constexpr int step(IterationType t) const { return t == IT_FwdSearch ? 1 : -1; }
    // MkFitCore throws for IT_BkwFit; here false
    constexpr bool is_pickup_only(IterationType t, int i) const {
      return t == IT_FwdSearch ? i == fwd_search_pickup : (t == IT_BkwSearch ? i == bkw_search_pickup : false);
    }
  };

  // The final fit's field-model and material switches (mkfit::Config::refit*, set by
  // MkFitGeometryESProducer from its refit* parameters; MkBuilder::fit_tracks builds the refit flags from them).
  struct RefitConfig {
    bool bFieldAtMid;
    bool radialFieldCorr;
    bool elossSignFromPass;
    bool bkwMsFixedMomentum;
    int bkwSubSteps;
    bool materialPerModule;
  };

  struct ESConfig {
    // ---- TrackerInfo globals
    int n_layers;
    int n_barrel_layers;
    int n_ecap_pos_layers;
    int n_ecap_neg_layers;
    int outer_barrel_layer;  // TrackerInfo::outer_barrel_layer().layer_id()
    int n_total_modules;
    int n_total_shapes;

    // ---- mkfit::Config values that are runtime (extern) in MkFitCore and so not usable on device
    bool usePropToPlane;  // set true by MkFitGeometryESProducer for Phase-2
    bool usePtMultScat;   // set true by MkFitGeometryESProducer for Phase-2
    float maxdPt, maxdPhi, maxdEta, maxdR, minFracHitsShared;
    float maxd1pt, maxdphi, maxdcth, maxcth_ob, maxcth_fw;

    // ---- TrackerInfo::prop_config()
    PropConfig prop_config;

    // ---- mkfit::Config::refit* and mag_* (runtime, from the MkFitGeometry ES producer)
    RefitConfig refit;
    float mag_c1, mag_b0, mag_b1, mag_a;

    // ---- IterationConfig
    int iteration_index;
    int track_algorithm;
    bool requires_seed_hit_sorting;
    bool backward_search;
    bool backward_drop_seed_hits;
    int backward_fit_min_hits;
    // backward-fit outlier rejection (JSON keys; 0 = off) and the backward-search pixel-layer gate
    // (MkFitIterationConfigESProducer parameters backwardSearchMinPixelLayers / backwardSearchPromptMaxD0; 0 = off)
    float backward_fit_outlier_chi2;
    int backward_fit_max_outliers;
    float backward_fit_outlier_min_pt;
    int backward_search_min_pixel_layers;
    float backward_search_prompt_max_d0;
    // TrackerInfo is_pixel() per layer, bit l of word l / 64 (pixel-priority cleaner, backward-search gate)
    uint64_t pixel_layer_mask[4];

    float sc_ptthr_hpt;
    float sc_drmax_bh, sc_dzmax_bh;
    float sc_drmax_eh, sc_dzmax_eh;
    float sc_drmax_bl, sc_dzmax_bl;
    float sc_drmax_el, sc_dzmax_el;

    float dc_fracSharedHits;
    float dc_drth_central;
    float dc_drth_obarrel;
    float dc_drth_forward;

    IterParams params;
    IterParams backward_params;

    int n_regions;
    int region_order[es::kMaxRegions];
    SteeringRegion steering_params[es::kMaxRegions];

    SeedCleaner seed_cleaner;
    SeedPartitioner seed_partitioner;
    CandFilter pre_bkfit_filter;
    CandFilter post_bkfit_filter;
    DuplicateCleaner duplicate_cleaner;
    TrackScorer default_track_scorer;

    constexpr bool merge_seed_hits_during_cleaning() const { return backward_search && backward_drop_seed_hits; }
  };

}  // namespace mkfitdev

#endif
