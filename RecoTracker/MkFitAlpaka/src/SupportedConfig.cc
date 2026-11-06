// The validated configuration envelope of the device chain. See interface/SupportedConfig.h and
// doc/SIMPLIFICATIONS.txt. Each check names what the device assumes.

#include <initializer_list>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"

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

    void checkESLevel(Violations& violations, ESConfig const& cfg) {
      // standard functions: the device implements exactly these ones
      violations.require(cfg.pre_bkfit_filter == CandFilter::NHitsPixSeed,
                         "pre_bkfit_filter must be phase1:qfilter_n_hits_pixseed (device: passLstStepFilter)");
      violations.require(cfg.post_bkfit_filter == CandFilter::NHitsPixSeed,
                         "post_bkfit_filter must be phase1:qfilter_n_hits_pixseed (device: passLstStepFilter)");
      violations.require(cfg.duplicate_cleaner == DuplicateCleaner::None ||
                             cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelSeed ||
                             cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelPriority,
                         "duplicate_cleaner must be phase1:clean_duplicates_sharedhits_pixelseed, "
                         "phase2:clean_duplicates_sharedhits_pixelpriority or empty");
      // knobs: the device implements them for any value; reject only values MkFitCore cannot mean
      violations.require(cfg.backward_fit_max_outliers >= 0, "m_backward_fit_max_outliers must be >= 0");
      violations.require(cfg.backward_search_min_pixel_layers >= 0, "backwardSearchMinPixelLayers must be >= 0");
      violations.require(cfg.seed_partitioner == SeedPartitioner::Phase2_1, "seed_partitioner must be phase2:1");
      violations.require(cfg.default_track_scorer == TrackScorer::Default, "default_track_scorer must be 'default'");
      for (int region = 0; region < cfg.n_regions && region < es::kMaxRegions; ++region)
        if (cfg.steering_params[region].track_scorer != TrackScorer::Default)
          violations.add("track_scorer of region " + std::to_string(region) + " must be 'default' or empty");
      // compile-time capacities
      for (IterParams const* params : {&cfg.params, &cfg.backward_params})
        if (params->maxCandsPerSeed < 1 || params->maxCandsPerSeed > kMaxCandsPerSeed)
          violations.add(std::string(params == &cfg.params ? "params" : "backward_params") +
                         ".maxCandsPerSeed = " + std::to_string(params->maxCandsPerSeed) +
                         " outside [1, kMaxCandsPerSeed = " + std::to_string(kMaxCandsPerSeed) +
                         "]: the device candidate slots are a compile-time capacity");
      // one minHitsQF for the pre- and the post-backward-search filter (MkFitCore: params_cur(), MkStdSeqs.cc:625)
      if (cfg.params.minHitsQF != cfg.backward_params.minHitsQF)
        violations.add("params.minHitsQF (" + std::to_string(cfg.params.minHitsQF) +
                       ") != backward_params.minHitsQF (" + std::to_string(cfg.backward_params.minHitsQF) +
                       "): the device filters use one value");
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
    // !is_pixel() layers (MkFinder.cc:1742-1756); pixel layers keep the default has_charge = true
    // (MkFitGeometryESProducer.cc:339-340 clears it on Phase-2 strip layers only).
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
    violations.require(moduleCfg.buildingRoutine == "cloneEngine", "buildingRoutine must be cloneEngine");
    violations.require(!moduleCfg.backwardFitInCMSSW,
                       "backwardFitInCMSSW must be false (the device runs mkFit's backward fit)");
    // MkFitCore runs the seed cleaner only if seedCleaning && itconf.m_seed_cleaner; none is ported
    violations.require(!(moduleCfg.seedCleaning && cfg.seed_cleaner != SeedCleaner::None),
                       "seedCleaning with a seed cleaner in the IterationConfig: no seed cleaner is ported");
    // MkFitCore runs the duplicate cleaner only if removeDuplicates && itconf.m_duplicate_cleaner; the device cleaner
    // is the sharedhits_pixelseed one and runs whenever removeDuplicates is set
    violations.require(!moduleCfg.removeDuplicates || cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelSeed ||
                           cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelPriority,
                       "removeDuplicates needs duplicate_cleaner = phase1:clean_duplicates_sharedhits_pixelseed or "
                       "phase2:clean_duplicates_sharedhits_pixelpriority");
    violations.throwIfAny();
  }

  void checkSupportedFitConfig(ESConfig const& cfg) {
    Violations violations("MkFitAlpaka fit module");
    // MkFitter fit: flags PF_use_param_b_field | PF_apply_material hard-coded (MkBuilder.cc:1440), as the port;
    // the propagation itself depends on the Phase-2 constexprs
    violations.require(cfg.usePropToPlane == Config::usePropToPlane,
                       "mkfit::Config::usePropToPlane differs from the device constexpr (Phase-2 only)");
    violations.require(cfg.usePtMultScat == Config::usePtMultScat,
                       "mkfit::Config::usePtMultScat differs from the device constexpr (Phase-2 only)");
    violations.throwIfAny();
  }

}  // namespace mkfitdev
