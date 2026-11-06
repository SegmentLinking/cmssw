// The validated configuration envelope of the device chain. See interface/SupportedConfig.h.
// Each check names what the device assumes and where (doc/SIMPLIFICATIONS.txt "Guard" lines).

#include <initializer_list>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"

namespace mkfitdev {

  namespace {

    // collects every violation, then throws once with all of them. Nothing is allocated while all checks pass
    // (the build module may call the checks every event).
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
        ++n_;
      }
      void throwIfAny() const {
        if (n_ > 0)
          throw cms::Exception("MkFitAlpakaUnsupportedConfig")
              << where_ << ": " << n_ << " setting(s) outside the configuration the device chain implements"
              << " (RecoTracker/MkFitAlpaka/interface/SupportedConfig.h):" << list_;
      }

    private:
      const char* where_;
      std::string list_;
      int n_ = 0;
    };

    void checkESLevel(Violations& v, ESConfig const& c) {
      // (a) standard functions: the device implements exactly these ones
      v.require(c.pre_bkfit_filter == CandFilter::NHitsPixSeed,
                "pre_bkfit_filter must be phase1:qfilter_n_hits_pixseed (device: passLstStepFilter)");
      v.require(c.post_bkfit_filter == CandFilter::NHitsPixSeed,
                "post_bkfit_filter must be phase1:qfilter_n_hits_pixseed (device: passLstStepFilter)");
      v.require(c.duplicate_cleaner == DuplicateCleaner::None ||
                    c.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelSeed ||
                    c.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelPriority,
                "duplicate_cleaner must be phase1:clean_duplicates_sharedhits_pixelseed, "
                "phase2:clean_duplicates_sharedhits_pixelpriority or empty");
      // knobs: the device implements them for any value; reject only values MkFitCore cannot mean
      v.require(c.backward_fit_max_outliers >= 0, "m_backward_fit_max_outliers must be >= 0");
      v.require(c.backward_search_min_pixel_layers >= 0, "backwardSearchMinPixelLayers must be >= 0");
      v.require(c.seed_partitioner == SeedPartitioner::Phase2_1, "seed_partitioner must be phase2:1");
      v.require(c.default_track_scorer == TrackScorer::Default, "default_track_scorer must be 'default'");
      for (int r = 0; r < c.n_regions && r < es::kMaxRegions; ++r)
        if (c.steering_params[r].track_scorer != TrackScorer::Default)
          v.add("track_scorer of region " + std::to_string(r) + " must be 'default' or empty");
      // (b) compile-time capacities
      for (IterParams const* p : {&c.params, &c.backward_params})
        if (p->maxCandsPerSeed < 1 || p->maxCandsPerSeed > kMaxCandsPerSeed)
          v.add(std::string(p == &c.params ? "params" : "backward_params") +
                ".maxCandsPerSeed = " + std::to_string(p->maxCandsPerSeed) + " outside [1, kMaxCandsPerSeed = " +
                std::to_string(kMaxCandsPerSeed) + "]: the device candidate slots are a compile-time capacity");
      // (e) one minHitsQF for the pre- and the post-backward-search filter (MkFitCore: params_cur(), MkStdSeqs.cc:625)
      if (c.params.minHitsQF != c.backward_params.minHitsQF)
        v.add("params.minHitsQF (" + std::to_string(c.params.minHitsQF) + ") != backward_params.minHitsQF (" +
              std::to_string(c.backward_params.minHitsQF) + "): the device filters use one value");
      v.require(c.params.useHitSelectionV2 && c.backward_params.useHitSelectionV2,
                "useHitSelectionV2 must be true (only selectHitIndicesV2 is ported)");
      // (f) iteration flags and runtime mkfit::Config values the device code fixes
      v.require(!c.requires_seed_hit_sorting, "requires_seed_hit_sorting must be false");
      v.require(!c.backward_drop_seed_hits, "backward_drop_seed_hits must be false");
      v.require(c.backward_search, "backward_search must be true (the engine always runs the backward search)");
      v.require(c.usePropToPlane == Config::usePropToPlane,
                "mkfit::Config::usePropToPlane differs from the device constexpr (Phase-2 only)");
      v.require(c.usePtMultScat == Config::usePtMultScat,
                "mkfit::Config::usePtMultScat differs from the device constexpr (Phase-2 only)");
      v.require(!c.prop_config.backward_fit_to_pca,
                "prop_config.backward_fit_to_pca must be false (no PCA step ported)");
      v.require(c.prop_config.finding_requires_propagation_to_hit_pos,
                "prop_config.finding_requires_propagation_to_hit_pos must be true");
    }

  }  // namespace

  void checkSupportedConfig(ESConfig const& c, LayerInfoSoA::ConstView layers, int nLayers) {
    Violations v("MkFitAlpaka ES configuration");
    checkESLevel(v, c);
    // (d) passStripChargePCMfromTrack is not ported: no STRIP layer may carry a charge cut. MkFitCore calls it only for
    //     !is_pixel() layers (MkFinder.cc:1742-1756); pixel layers keep the default has_charge = true
    //     (MkFitGeometryESProducer.cc:339-340 clears it on Phase-2 strip layers only).
    for (int l = 0; l < nLayers; ++l)
      if (!layers[l].is_pixel() && layers[l].has_charge())
        v.add("strip layer " + std::to_string(l) + " has_charge() is true: strip charge PCM is not ported");
    v.throwIfAny();
  }

  void checkSupportedBuildConfig(ESConfig const& c, BuildModuleConfig const& m) {
    Violations v("MkFitAlpaka build module");
    checkESLevel(v, c);
    // (c) no hit mask column on the device (MkFitProducer fills m_iteration_hit_mask from clustersToSkip)
    v.require(m.clustersToSkip.empty(), "clustersToSkip must be empty: no hit mask on the device");
    v.require(m.buildingRoutine == "cloneEngine", "buildingRoutine must be cloneEngine");
    v.require(!m.backwardFitInCMSSW, "backwardFitInCMSSW must be false (the device runs mkFit's backward fit)");
    // MkFitCore runs the seed cleaner only if seedCleaning && itconf.m_seed_cleaner; none is ported
    v.require(!(m.seedCleaning && c.seed_cleaner != SeedCleaner::None),
              "seedCleaning with a seed cleaner in the IterationConfig: no seed cleaner is ported");
    // MkFitCore runs the duplicate cleaner only if removeDuplicates && itconf.m_duplicate_cleaner; the device cleaner
    // is the sharedhits_pixelseed one and runs whenever removeDuplicates is set
    v.require(!m.removeDuplicates || c.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelSeed ||
                  c.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelPriority,
              "removeDuplicates needs duplicate_cleaner = phase1:clean_duplicates_sharedhits_pixelseed or "
              "phase2:clean_duplicates_sharedhits_pixelpriority");
    v.throwIfAny();
  }

  void checkSupportedFitConfig(ESConfig const& c) {
    Violations v("MkFitAlpaka fit module");
    // MkFitter fit: flags PF_use_param_b_field | PF_apply_material hard-coded (MkBuilder.cc:1440), as the port;
    // the propagation itself depends on the Phase-2 constexprs
    v.require(c.usePropToPlane == Config::usePropToPlane,
              "mkfit::Config::usePropToPlane differs from the device constexpr (Phase-2 only)");
    v.require(c.usePtMultScat == Config::usePtMultScat,
              "mkfit::Config::usePtMultScat differs from the device constexpr (Phase-2 only)");
    v.throwIfAny();
  }

}  // namespace mkfitdev
