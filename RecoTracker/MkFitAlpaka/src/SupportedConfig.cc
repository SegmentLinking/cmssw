// The validated configuration envelope of the device chain. See interface/SupportedConfig.h and
// doc/SIMPLIFICATIONS.txt. Each check names what the device assumes.

#include <initializer_list>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Params.h"

namespace mkfitdev {

  namespace {

    // Collects every violation, then throws once with all of them. Nothing is allocated while all checks pass
    // (the build module calls the checks every event).
    class Violations {
    public:
      explicit Violations(const char* where) : where_(where) {}
      void require(bool ok, const char* what) {
        if (!ok)
          add(what);
      }
      void add(std::string const& what) {
        list_ += "\n  - ";
        list_ += what;
        ++nViolations_;
      }
      void throwIfAny() const {
        if (nViolations_ > 0)
          throw cms::Exception("MkFitAlpakaUnsupportedConfig")
              << where_ << ": " << nViolations_ << " setting(s) outside the configuration the device chain implements"
              << " (RecoTracker/MkFitAlpaka/interface/SupportedConfig.h):" << list_;
      }

    private:
      const char* where_;
      std::string list_;
      int nViolations_ = 0;
    };

    // the duplicate cleaners of MkStdSeqs.cc that the device implements (src/alpaka/clean)
    bool isPortedDuplicateCleaner(ESConfig const& cfg) {
      return cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelSeed ||
             cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelPriority ||
             cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelPriorityPtAdaptive;
    }

    void checkESLevel(Violations& violations, ESConfig const& cfg) {
      // standard functions: the device implements exactly these ones
      violations.require(cfg.pre_bkfit_filter == CandFilter::NHitsPixSeed,
                         "pre_bkfit_filter must be phase1:qfilter_n_hits_pixseed (device: passLstStepFilter)");
      violations.require(cfg.post_bkfit_filter == CandFilter::NHitsPixSeed,
                         "post_bkfit_filter must be phase1:qfilter_n_hits_pixseed (device: passLstStepFilter)");
      violations.require(cfg.duplicate_cleaner == DuplicateCleaner::None || isPortedDuplicateCleaner(cfg),
                         "duplicate_cleaner must be phase1:clean_duplicates_sharedhits_pixelseed, "
                         "phase2:clean_duplicates_sharedhits_pixelpriority, "
                         "phase2:clean_duplicates_sharedhits_pixelpriority_ptadaptive or empty");
      // knobs: the device implements them for any value; reject only values MkFitCore cannot mean
      violations.require(cfg.backward_fit_max_outliers >= 0, "m_backward_fit_max_outliers must be >= 0");
      violations.require(cfg.backward_search_min_pixel_layers >= 0, "backwardSearchMinPixelLayers must be >= 0");
      violations.require(cfg.seed_partitioner == SeedPartitioner::Phase2_1, "seed_partitioner must be phase2:1");
      // track scorers: the search and the post-backward-fit scores are trackScoreDefault; the final pick also takes
      // trackScoreV2p2Llh, in the final sort of either finder as in MkBuilder
      violations.require(cfg.default_track_scorer == TrackScorer::Default, "default_track_scorer must be 'default'");
      for (int region = 0; region < cfg.n_regions && region < es::kMaxRegions; ++region) {
        SteeringRegion const& steering = cfg.steering_params[region];
        if (steering.track_scorer != TrackScorer::Default)
          violations.add("track_scorer of region " + std::to_string(region) + " must be 'default' or empty");
        if (steering.final_pick_track_scorer != TrackScorer::Default &&
            steering.final_pick_track_scorer != TrackScorer::V2p2Llh)
          violations.add("final_pick_track_scorer of region " + std::to_string(region) +
                         " must be 'default', 'v2p2:llh' or empty (m_final_pick_track_scorer_name)");
        if (steering.post_bkfit_track_scorer != TrackScorer::Default)
          violations.add("post_bkfit_track_scorer of region " + std::to_string(region) +
                         " must be 'default' or empty (m_post_bkfit_track_scorer_name)");
      }
      // "keep" re-scores every track with the score it carries, i.e. changes nothing
      violations.require(
          cfg.duplicate_cleaner_track_scorer == TrackScorer::None ||
              cfg.duplicate_cleaner_track_scorer == TrackScorer::Keep,
          "duplicate_cleaner_track_scorer must be empty or 'keep' (m_duplicate_cleaner_track_scorer_name):"
          " the device does not re-score before the duplicate cleaner");
      // compile-time capacities
      for (IterParams const* params : {&cfg.params, &cfg.backward_params})
        if (params->maxCandsPerSeed < 1 || params->maxCandsPerSeed > kMaxCandsPerSeed)
          violations.add(std::string(params == &cfg.params ? "params" : "backward_params") +
                         ".maxCandsPerSeed = " + std::to_string(params->maxCandsPerSeed) +
                         " outside [1, kMaxCandsPerSeed = " + std::to_string(kMaxCandsPerSeed) +
                         "]: the device candidate slots are a compile-time capacity");
      // one minHitsQF for the pre- and the post-backward-search filter (MkFitCore qfilter_n_hits_pixseed: params_cur())
      if (cfg.params.minHitsQF != cfg.backward_params.minHitsQF)
        violations.add("params.minHitsQF (" + std::to_string(cfg.params.minHitsQF) +
                       ") != backward_params.minHitsQF (" + std::to_string(cfg.backward_params.minHitsQF) +
                       "): the device filters use one value");
      // v2p2 tunings: the device candidate storage is sized by the V2p2Params limits (MkFitIterationConfigESProducer
      // checks the same ranges)
      if (cfg.v2p2.max_presel_hits < 1 || cfg.v2p2.max_presel_hits > V2p2Params::max_presel_hits_limit)
        violations.add("v2p2.max_presel_hits = " + std::to_string(cfg.v2p2.max_presel_hits) +
                       " outside [1, max_presel_hits_limit = " + std::to_string(V2p2Params::max_presel_hits_limit) +
                       "]");
      if (cfg.v2p2.max_hits_per_layer < 1 || cfg.v2p2.max_hits_per_layer > V2p2Params::max_hits_per_layer_limit)
        violations.add(
            "v2p2.max_hits_per_layer = " + std::to_string(cfg.v2p2.max_hits_per_layer) +
            " outside [1, max_hits_per_layer_limit = " + std::to_string(V2p2Params::max_hits_per_layer_limit) + "]");
      violations.require(cfg.params.useHitSelectionV2 && cfg.backward_params.useHitSelectionV2,
                         "useHitSelectionV2 must be true (only selectHitIndicesV2 is ported)");
      // iteration flags and runtime mkfit::Config values the device code fixes
      violations.require(!cfg.requires_seed_hit_sorting, "requires_seed_hit_sorting must be false");
      violations.require(!cfg.backward_drop_seed_hits, "backward_drop_seed_hits must be false");
      violations.require(!cfg.params.recheckOverlap && !cfg.backward_params.recheckOverlap,
                         "recheckOverlap must be false (the overlap re-check is not ported)");
      violations.require(cfg.backward_search,
                         "backward_search must be true (the engine always runs the backward search)");
      violations.require(cfg.usePropToPlane == Config::usePropToPlane,
                         "mkfit::Config::usePropToPlane differs from the device constexpr (Phase-2 only)");
      violations.require(cfg.usePtMultScat == Config::usePtMultScat,
                         "mkfit::Config::usePtMultScat differs from the device constexpr (Phase-2 only)");
      violations.require(!cfg.prop_config.backward_fit_to_pca,
                         "prop_config.backward_fit_to_pca must be false (no PCA step ported)");
      violations.require(cfg.prop_config.finding_requires_propagation_to_hit_pos,
                         "prop_config.finding_requires_propagation_to_hit_pos must be true");
    }

  }  // namespace

  void checkSupportedConfig(ESConfig const& cfg, LayerInfoSoA::ConstView layers, int nLayers) {
    Violations violations("MkFitAlpaka ES configuration");
    checkESLevel(violations, cfg);
    // passStripChargePCMfromTrack is not ported: no STRIP layer may carry a charge cut. MkFitCore calls it only for
    // !is_pixel() layers (MkFinder::findCandidates / findCandidatesCloneEngine); pixel layers keep the default
    // has_charge = true (MkFitGeometryESProducer clears it on Phase-2 strip layers only).
    for (int layer = 0; layer < nLayers; ++layer)
      if (!layers[layer].is_pixel() && layers[layer].has_charge())
        violations.add("strip layer " + std::to_string(layer) +
                       " has_charge() is true: strip charge PCM is not ported");
    violations.throwIfAny();
  }

  void checkSupportedBuildConfig(ESConfig const& cfg, BuildModuleConfig const& moduleCfg) {
    Violations violations("MkFitAlpaka build module");
    checkESLevel(violations, cfg);
    // no hit mask column on the device (MkFitProducer fills m_iteration_hit_mask from clustersToSkip)
    violations.require(moduleCfg.clustersToSkip.empty(), "clustersToSkip must be empty: no hit mask on the device");
    // MkFitProducer's clone engine and MkFinderV2p2 are ported, its bestHit and standard routines are not
    const bool v2p2 = moduleCfg.buildingRoutine == "v2p2";
    violations.require(moduleCfg.buildingRoutine == "cloneEngine" || v2p2,
                       "buildingRoutine must be cloneEngine or v2p2 (bestHit and standard are not ported)");
    violations.require(!moduleCfg.backwardFitInCMSSW,
                       "backwardFitInCMSSW must be false (the device runs mkFit's backward fit)");
    // MkFitCore runs the seed cleaner only if seedCleaning && itconf.m_seed_cleaner; none is ported
    violations.require(!(moduleCfg.seedCleaning && cfg.seed_cleaner != SeedCleaner::None),
                       "seedCleaning with a seed cleaner in the IterationConfig: no seed cleaner is ported");
    // MkFitCore runs the duplicate cleaner only if removeDuplicates && itconf.m_duplicate_cleaner; the device cleaner
    // runs whenever removeDuplicates is set
    violations.require(!moduleCfg.removeDuplicates || isPortedDuplicateCleaner(cfg),
                       "removeDuplicates needs duplicate_cleaner = phase1:clean_duplicates_sharedhits_pixelseed, "
                       "phase2:clean_duplicates_sharedhits_pixelpriority or "
                       "phase2:clean_duplicates_sharedhits_pixelpriority_ptadaptive");
    violations.throwIfAny();
  }

  void checkSupportedFitConfig(ESConfig const& cfg) {
    Violations violations("MkFitAlpaka fit module");
    // the fit's propagation flags come from the ES (final_fit_pflags / final_fit_ffflags, RefitConfig); the
    // propagation itself depends on the Phase-2 constexprs
    violations.require(cfg.usePropToPlane == Config::usePropToPlane,
                       "mkfit::Config::usePropToPlane differs from the device constexpr (Phase-2 only)");
    violations.require(cfg.usePtMultScat == Config::usePtMultScat,
                       "mkfit::Config::usePtMultScat differs from the device constexpr (Phase-2 only)");
    violations.throwIfAny();
  }

}  // namespace mkfitdev
