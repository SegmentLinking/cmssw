#ifndef RecoTracker_MkFitAlpaka_interface_es_ESConfig_h
#define RecoTracker_MkFitAlpaka_interface_es_ESConfig_h

// Flat, trivially copyable copy of the scalar ES configuration of one mkFit iteration (MkFitCore
// mkfit::IterationConfig + SteeringParams + PropagationConfig + the runtime mkfit::Config values).
// Member names follow the MkFitCore classes (m_ prefix dropped). Filled by mkfitdev::fillESDataHost.

#include <cstdint>

namespace mkfitdev {

  namespace es {
    constexpr int kMaxRegions = 8;      // eta regions (m_n_regions) per iteration; the fill throws above this
    constexpr int kMaxPlanLayers = 64;  // layer-plan entries per region; the fill throws above this
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

  // mkfit::PropagationFlags (tracker_info back-pointer dropped)
  struct PropFlags {
    bool use_param_b_field;
    bool apply_material;
    bool copy_input_state_on_fail;
    bool use_fit_b_field;  // the parametrised field of the final fit (Config::bFieldFromZRFit)
  };

  // mkfit::PropagationConfig
  struct PropConfig {
    bool backward_fit_to_pca;
    bool finding_requires_propagation_to_hit_pos;
    PropFlags finding_inter_layer_pflags;
    PropFlags finding_intra_layer_pflags;
    PropFlags backward_fit_pflags;
    PropFlags final_fit_pflags;  // the final fit, MkBuilder::fit_tracks(), both passes
  };

  // mkfit::IterationParams
  struct IterParams {
    int maxCandsPerSeed;
    int maxHolesPerCand;
    int maxConsecHoles;
    float chi2Cut_min;
    float pTCutOverlap;
    bool recheckOverlap;
    bool useHitSelectionV2;
    int minHitsQF;
    float minPtCut;
    unsigned int maxClusterSize;
  };

  // mkfit::SteeringParams of one eta region: the layer plan, the forward-search pickup index and the backward-search
  // pickup index (-1: no backward-search plan)
  struct SteeringRegion {
    int region;
    int n_plan;
    int fwd_search_pickup;
    int bkw_search_pickup;
    TrackScorer track_scorer;  // resolved as MkFitCore: empty name -> the iteration's default scorer
    int layer[es::kMaxPlanLayers];
  };

  // The final fit's field-model and material switches: mkfit::FinalFitFlags (PropagationConfig::final_fit_ffflags, set
  // by MkFitGeometryESProducer from its refit* parameters): b_field_at_mid, radial_field_corr, eloss_sign_from_pass,
  // bkw_ms_fixed_momentum, bkw_sub_steps, material_per_module, in that order.
  struct RefitConfig {
    bool bFieldAtMid;
    bool radialFieldCorr;
    bool elossSignFromPass;
    bool bkwMsFixedMomentum;
    int bkwSubSteps;
    bool materialPerModule;
  };

  struct ESConfig {
    // ---- mkfit::Config flags that are runtime in MkFitCore (the device code has them as constexpr: checked)
    bool usePropToPlane;  // set true by MkFitGeometryESProducer for Phase-2
    bool usePtMultScat;   // set true by MkFitGeometryESProducer for Phase-2

    // ---- TrackerInfo::prop_config()
    PropConfig prop_config;

    // ---- PropagationConfig::final_fit_ffflags (the field constants are in ESSizes / MaterialView)
    RefitConfig refit;

    // ---- IterationConfig
    bool requires_seed_hit_sorting;
    bool backward_search;
    bool backward_drop_seed_hits;
    // backward-fit outlier rejection (JSON keys; 0 = off) and the backward-search pixel-layer gate
    // (MkFitIterationConfigESProducer parameters backwardSearchMinPixelLayers / backwardSearchPromptMaxD0; 0 = off)
    float backward_fit_outlier_chi2;
    int backward_fit_max_outliers;
    float backward_fit_outlier_min_pt;
    int backward_search_min_pixel_layers;
    float backward_search_prompt_max_d0;
    // TrackerInfo is_pixel() per layer, bit l of word l / 64 (pixel-priority cleaner, backward-search gate)
    uint64_t pixel_layer_mask[4];

    float dc_fracSharedHits;
    float dc_drth_central;
    float dc_drth_obarrel;
    float dc_drth_forward;

    IterParams params;
    IterParams backward_params;

    int n_regions;
    SteeringRegion steering_params[es::kMaxRegions];

    SeedCleaner seed_cleaner;
    SeedPartitioner seed_partitioner;
    CandFilter pre_bkfit_filter;
    CandFilter post_bkfit_filter;
    DuplicateCleaner duplicate_cleaner;
    TrackScorer default_track_scorer;
  };

}  // namespace mkfitdev

#endif
