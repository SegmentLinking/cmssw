// Host unit test of mkfitdev::fillESDataHost on a small synthetic TrackerInfo / IterationConfig: covers what the
// Phase-2 cmsRun test cannot (an r-hole layer, which Phase-2 does not have, the per-task track scorers per building
// routine, the v2p2 tunings and the double-layer partners) and the error paths (capacities, hit-window parameters,
// unknown function names, paired plan entries). Run: testMkFitAlpakaESFill

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/V2p2Params.h"
// package-private MkFitCore header on purpose: angle brackets keep scram's private-header check off this test
#include <RecoTracker/MkFitCore/src/V2p2Config.h>

#include "RecoTracker/MkFitAlpaka/interface/SupportedConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"

namespace {
  // the device copies of the MkFinderV2p2 knobs (interface/v2p2/V2p2Config.h), checked at compile time
  static_assert(mkfitdev::Config::V2p2::Window::mini_prop_param_b == mkfit::Config::V2p2::Window::mini_prop_param_b &&
                mkfitdev::Config::V2p2::Window::dq_trk_fac == mkfit::Config::V2p2::Window::dq_trk_fac &&
                mkfitdev::Config::V2p2::Window::dq_hit_fac == mkfit::Config::V2p2::Window::dq_hit_fac &&
                mkfitdev::Config::V2p2::Window::surface_q == mkfit::Config::V2p2::Window::surface_q &&
                mkfitdev::Config::V2p2::Window::q_extra_bins == mkfit::Config::V2p2::Window::q_extra_bins &&
                mkfitdev::Config::V2p2::Window::dphi_trk_fac == mkfit::Config::V2p2::Window::dphi_trk_fac &&
                mkfitdev::Config::V2p2::Window::phi_per_hit == mkfit::Config::V2p2::Window::phi_per_hit &&
                mkfitdev::Config::V2p2::Window::dphi_hit_fac == mkfit::Config::V2p2::Window::dphi_hit_fac &&
                mkfitdev::Config::V2p2::Window::dphi_flat_rad == mkfit::Config::V2p2::Window::dphi_flat_rad &&
                mkfitdev::Config::V2p2::Window::phi_extra_bins == mkfit::Config::V2p2::Window::phi_extra_bins &&
                mkfitdev::Config::V2p2::Window::dphi_max_rad == mkfit::Config::V2p2::Window::dphi_max_rad &&
                mkfitdev::Config::V2p2::Window::min_cos_incidence == mkfit::Config::V2p2::Window::min_cos_incidence);
  static_assert(mkfitdev::Config::V2p2::PreCut::q == mkfit::Config::V2p2::PreCut::q &&
                mkfitdev::Config::V2p2::PreCut::phi == mkfit::Config::V2p2::PreCut::phi &&
                mkfitdev::Config::V2p2::PreCut::dq_slack == mkfit::Config::V2p2::PreCut::dq_slack &&
                mkfitdev::Config::V2p2::PreCut::dphi_slack == mkfit::Config::V2p2::PreCut::dphi_slack &&
                mkfitdev::Config::V2p2::PreCut::qbar_fac == mkfit::Config::V2p2::PreCut::qbar_fac);
  static_assert(
      mkfitdev::Config::V2p2::InLayer::max_presel_hits_limit == mkfit::Config::V2p2::InLayer::max_presel_hits_limit &&
      mkfitdev::Config::V2p2::InLayer::max_hits_per_layer_limit ==
          mkfit::Config::V2p2::InLayer::max_hits_per_layer_limit &&
      mkfitdev::Config::V2p2::InLayer::reserve_hole_slot == mkfit::Config::V2p2::InLayer::reserve_hole_slot &&
      mkfitdev::Config::V2p2::InLayer::best_short == mkfit::Config::V2p2::InLayer::best_short &&
      mkfitdev::Config::V2p2::InLayer::early_select == mkfit::Config::V2p2::InLayer::early_select &&
      mkfitdev::Config::V2p2::InLayer::late_max_cands == mkfit::Config::V2p2::InLayer::late_max_cands &&
      mkfitdev::Config::V2p2::InLayer::late_max_cands_hits == mkfit::Config::V2p2::InLayer::late_max_cands_hits &&
      mkfitdev::Config::V2p2::InLayer::late_max_cands_step_cm == mkfit::Config::V2p2::InLayer::late_max_cands_step_cm &&
      mkfitdev::Config::V2p2::InLayer::sister_hole_drop == mkfit::Config::V2p2::InLayer::sister_hole_drop &&
      mkfitdev::Config::V2p2::InLayer::sister_hole_chi2 == mkfit::Config::V2p2::InLayer::sister_hole_chi2);
  static_assert(mkfitdev::Config::V2p2::Policy::use_wsr == mkfit::Config::V2p2::Policy::use_wsr &&
                mkfitdev::Config::V2p2::Policy::wsr_n_sigma == mkfit::Config::V2p2::Policy::wsr_n_sigma &&
                mkfitdev::Config::V2p2::Policy::use_hole_limits == mkfit::Config::V2p2::Policy::use_hole_limits &&
                mkfitdev::Config::V2p2::Policy::use_stop_cuts == mkfit::Config::V2p2::Policy::use_stop_cuts &&
                mkfitdev::Config::V2p2::Policy::final_pick_hole_penalty ==
                    mkfit::Config::V2p2::Policy::final_pick_hole_penalty &&
                mkfitdev::Config::V2p2::Policy::final_pick_tail_penalty ==
                    mkfit::Config::V2p2::Policy::final_pick_tail_penalty &&
                mkfitdev::Config::V2p2::Policy::looper_max_angle == mkfit::Config::V2p2::Policy::looper_max_angle &&
                mkfitdev::Config::V2p2::Policy::looper_max_pt == mkfit::Config::V2p2::Policy::looper_max_pt &&
                mkfitdev::Config::V2p2::Policy::looper_min_r == mkfit::Config::V2p2::Policy::looper_min_r);
  static_assert(mkfitdev::Config::V2p2::Score::mode == mkfit::Config::V2p2::Score::mode &&
                mkfitdev::Config::V2p2::Score::use_rho == mkfit::Config::V2p2::Score::use_rho &&
                mkfitdev::Config::V2p2::Score::rho_const == mkfit::Config::V2p2::Score::rho_const &&
                mkfitdev::Config::V2p2::Score::use_detv == mkfit::Config::V2p2::Score::use_detv &&
                mkfitdev::Config::V2p2::Score::detv_const == mkfit::Config::V2p2::Score::detv_const);

  constexpr int kNLayers = 3, kModulesPerLayer = 50, kNModules = kNLayers * kModulesPerLayer;
  constexpr int kMatBinsZ = 30, kMatBinsR = 12;

  int nFail = 0;
  void check(bool ok, const char* what) {
    if (!ok) {
      ++nFail;
      std::printf("FAIL: %s\n", what);
    }
  }

  void makeTracker(mkfit::TrackerInfo& ti) {
    ti.create_layers(1, 1, 1);
    ti.create_material(kMatBinsZ, 300.f, kMatBinsR, 120.f);
    ti.layer_nc(0).set_limits(20.f, 25.f, -100.f, 100.f);
    ti.layer_nc(0).set_q_bin(2.f);
    ti.layer_nc(0).set_propagate_to(22.5f);
    ti.layer_nc(1).set_limits(20.f, 110.f, 120.f, 125.f);
    ti.layer_nc(1).set_q_bin(5.6f);
    ti.layer_nc(1).set_r_hole_range(30.5f, 42.25f);
    ti.layer_nc(2).set_limits(20.f, 110.f, -125.f, -120.f);
    ti.layer_nc(2).set_q_bin(5.6f);
    unsigned int detid = 0x14000000;
    for (int l = 0; l < kNLayers; ++l) {
      for (int m = 0; m < kModulesPerLayer; ++m)
        ti.layer_nc(l).register_module(mkfit::ModuleInfo({1.f * m, 2.f, 3.f}, {0, 0, 1}, {1, 0, 0}, detid += 4, 0));
      ti.layer_nc(l).resize_shapes(1);
      mkfit::ModuleShape ms;
      ms.round_assign(1.f, 0.f, 2.f, 0.01f);
      ti.layer_nc(l).register_shape(ms, 0);
    }
    for (int iz = 0; iz < kMatBinsZ; ++iz)
      for (int ir = 0; ir < kMatBinsR; ++ir) {
        ti.material_bbxi(iz, ir) = 1e-4f * (iz + 1) * (ir + 1);
        ti.material_radl(iz, ir) = 1e-3f * (iz + 1) + ir;
      }
  }

  void makeConfig(mkfit::IterationConfig& ic, int nRegions, int nLayers) {
    ic.set_num_regions_layers(nRegions, nLayers);
    for (int r = 0; r < nRegions; ++r) {
      ic.m_region_order[r] = nRegions - 1 - r;
      ic.m_steering_params[r].fill_plan(0, nLayers - 1);
      ic.m_steering_params[r].set_iterator_limits(1, 0, 2);
    }
    for (int l = 0; l < nLayers; ++l)
      ic.m_layer_configs[l].set_selection_limits(0.01f, 0.02f, 1.f, 2.f);
    ic.m_duplicate_cleaner_name = "phase1:clean_duplicates_sharedhits_pixelseed";
    ic.m_default_track_scorer_name = "phase1:default";
  }

  template <typename TFunc>
  bool throws(TFunc&& func) {
    try {
      func();
    } catch (cms::Exception const&) {
      return true;
    }
    return false;
  }

  // the violations checkSupportedConfig reports for a filled configuration ("" if none); the synthetic geometry and
  // configuration are outside the envelope anyway, so the tests below look for one item in the list
  std::string supportedConfigViolations(mkfitdev::ESDataHost const& es) {
    try {
      mkfitdev::checkSupportedConfig(es.hostConfigValue(), es.layers->const_view(), es.sizes.nLayers);
    } catch (cms::Exception const& e) {
      return e.what();
    }
    return "";
  }

  // the same for the build module with the given MkFitProducer buildingRoutine
  std::string buildConfigViolations(mkfitdev::ESDataHost const& es, std::string const& routine) {
    mkfitdev::BuildModuleConfig moduleCfg;
    moduleCfg.buildingRoutine = routine;
    try {
      mkfitdev::checkSupportedBuildConfig(es.hostConfigValue(), moduleCfg);
    } catch (cms::Exception const& e) {
      return e.what();
    }
    return "";
  }

  bool sameScoreParams(mkfitdev::V2p2ScoreParams const& dev, mkfit::V2p2ScoreParams const& ref) {
    bool same = dev.hit_bonus == ref.hit_bonus && dev.overlap_bonus == ref.overlap_bonus &&
                dev.chi2_weight == ref.chi2_weight && dev.miss_penalty == ref.miss_penalty &&
                dev.edge_penalty == ref.edge_penalty && dev.gap_penalty == ref.gap_penalty &&
                dev.stop_penalty == ref.stop_penalty && dev.hit_eff == ref.hit_eff;
    for (int group = 0; group < mkfit::V2p2ScoreParams::n_hit_eff_grp; ++group)
      same = same && dev.hit_eff_grp[group] == ref.hit_eff_grp[group];
    return same;
  }
}  // namespace

int main() {
  mkfit::TrackerInfo ti;
  makeTracker(ti);
  // the final fit's flags as MkFitGeometryESProducer sets them; bkw_sub_steps and radial_field_corr are set away from
  // their defaults so the checks below show that they are read from final_fit_ffflags
  ti.prop_config_nc().final_fit_pflags =
      mkfit::PropagationFlags(mkfit::PF_use_param_b_field | mkfit::PF_apply_material | mkfit::PF_use_fit_b_field);
  ti.prop_config_nc().final_fit_ffflags.bkw_sub_steps = 3;
  ti.prop_config_nc().final_fit_ffflags.radial_field_corr = false;
  mkfit::IterationConfig ic;
  makeConfig(ic, 2, kNLayers);

  auto es = mkfitdev::fillESDataHost(ti, ic);
  auto layers = es->layers->const_view();
  check(es->sizes.nLayers == kNLayers && es->sizes.nModules == kNModules, "sizes");
  {
    // the phi-margin cap of the v2p2 window, bitwise as MkFitCore's Window::capped_dphi, NaN and infinity included
    bool sameCap = true;
    const float margins[] = {0.f,
                             0.01f,
                             0.0499f,
                             0.05f,
                             0.0501f,
                             1.34f,
                             3.14f,
                             -0.01f,
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()};
    for (float m : margins) {
      const float dev = mkfitdev::Config::V2p2::Window::capped_dphi(m),
                  ref = mkfit::Config::V2p2::Window::capped_dphi(m);
      sameCap = sameCap && std::memcmp(&dev, &ref, sizeof(float)) == 0;
    }
    check(sameCap, "phi-margin cap as MkFitCore");
  }
  check(!layers[0].has_r_range_hole() && !layers[2].has_r_range_hole(), "no hole on layers 0, 2");
  check(layers[1].has_r_range_hole(), "hole on layer 1");
  check(layers[1].hole_r_min() == 30.5f && layers[1].hole_r_max() == 42.25f, "hole range copied");
  std::printf("hole: [%.9g, %.9g]\n", layers[1].hole_r_min(), layers[1].hole_r_max());
  const mkfitdev::ESView esView = es->view();
  for (float dr : {0.f, 0.5f, 3.f})
    for (float r = 15.f; r < 115.f; r += 0.0137f) {
      auto got = esView.isWithinRSensitiveRegion(1, r, dr);
      auto expected = ti.layer(1).is_within_r_sensitive_region(r, dr);
      check(got.wsr == int(expected.m_wsr) && got.in_gap == bool(expected.m_in_gap),
            "isWithinRSensitiveRegion vs MkFitCore");
    }
  for (int m = 0; m < kNModules; ++m) {
    const uint32_t detid = es->modules->const_view()[m].detid();
    check(esView.findModule(detid) == m, "findModule");
    check(esView.findModule(detid + 1) == mkfitdev::es::kNoModule, "findModule unknown");
  }
  for (float z = -310.f; z < 310.f; z += 0.77f)
    for (float r = -3.f; r < 125.f; r += 0.61f) {
      float bbxi, radl;
      esView.material.materialChecked(z, r, bbxi, radl);
      const auto expected = ti.material_checked(z, r);
      check(bbxi == expected.bbxi && radl == expected.radl, "material vs MkFitCore");
    }
  auto const& config = es->hostConfigValue();
  check(config.prop_config.final_fit_pflags.use_fit_b_field && config.prop_config.final_fit_pflags.use_param_b_field &&
            !config.prop_config.backward_fit_pflags.use_fit_b_field,
        "final-fit propagation flags");
  check(config.refit.bkwSubSteps == 3 && !config.refit.radialFieldCorr && config.refit.bFieldAtMid &&
            config.refit.materialPerModule,
        "final-fit switches from final_fit_ffflags");
  check(esView.material.bField.c1 == mkfit::Config::mag_c1 && esView.material.bField.a == mkfit::Config::mag_a &&
            esView.material.bFieldFit.c1 == mkfit::Config::mag_fit_c1 &&
            esView.material.bFieldFit.a == mkfit::Config::mag_fit_a,
        "field constants of building and final fit");
  check(config.n_regions == 2, "regions");
  check(config.steering_params[1].n_plan == 3 && config.steering_params[1].layer[2] == 2, "layer plan");
  check(config.steering_params[0].bkw_search_pickup == 2 && config.steering_params[0].fwd_search_pickup == 1,
        "pickups");
  check(config.duplicate_cleaner == mkfitdev::DuplicateCleaner::SharedHitsPixelSeed, "duplicate cleaner enum");
  check(config.steering_params[0].track_scorer == mkfitdev::TrackScorer::Default, "scorer from default");
  check(config.steering_params[1].final_pick_track_scorer == mkfitdev::TrackScorer::Default &&
            config.steering_params[1].post_bkfit_track_scorer == mkfitdev::TrackScorer::Default,
        "per-task scorers default to the region's scorer");
  check(config.duplicate_cleaner_track_scorer == mkfitdev::TrackScorer::None, "no re-scoring before the cleaner");
  check(supportedConfigViolations(*es).find("track_scorer") == std::string::npos, "default scorers supported");
  {
    // a region without a backward-search plan, which the backward search skips as MkFitCore does
    mkfit::IterationConfig noPlan;
    makeConfig(noPlan, 2, kNLayers);
    noPlan.m_steering_params[1].set_iterator_limits(1, 0);
    const auto noPlanES = mkfitdev::fillESDataHost(ti, noPlan);
    check(noPlanES->hostConfigValue().steering_params[1].bkw_search_pickup == -1, "no backward-search plan filled");
    check(supportedConfigViolations(*noPlanES) == supportedConfigViolations(*es),
          "region without a backward-search plan supported");
  }
  check(config.seed_partitioner == mkfitdev::SeedPartitioner::None, "empty partitioner");
  {
    // the pT-adaptive cleaner, its parameters set away from the IterationConfig defaults
    mkfit::IterationConfig adaptive;
    makeConfig(adaptive, 2, kNLayers);
    adaptive.m_duplicate_cleaner_name = "phase2:clean_duplicates_sharedhits_pixelpriority_ptadaptive";
    adaptive.dc_fracSharedHitsLowPt = 0.15f;
    adaptive.dc_lowPtRampStart = 4.f;
    adaptive.dc_lowPtRampEnd = 8.f;
    adaptive.dc_lowPtMaxRelDiffInvPt = 0.2f;
    adaptive.dc_lowPtRequireSameCharge = false;
    adaptive.dc_lowPtMaxD0 = 1.5f;
    adaptive.dc_lowPtMaxDRTimesPt = 0.05f;
    adaptive.dc_minUniqueHitsToKeep = 7;
    adaptive.dc_minPtUniqueHitsToKeep = 12.f;
    auto esAdaptive = mkfitdev::fillESDataHost(ti, adaptive);
    auto const& cfg = esAdaptive->hostConfigValue();
    check(cfg.duplicate_cleaner == mkfitdev::DuplicateCleaner::SharedHitsPixelPriorityPtAdaptive,
          "pT-adaptive cleaner enum");
    check(cfg.dc_fracSharedHitsLowPt == 0.15f && cfg.dc_lowPtRampStart == 4.f && cfg.dc_lowPtRampEnd == 8.f &&
              cfg.dc_lowPtMaxRelDiffInvPt == 0.2f && !cfg.dc_lowPtRequireSameCharge && cfg.dc_lowPtMaxD0 == 1.5f &&
              cfg.dc_lowPtMaxDRTimesPt == 0.05f && cfg.dc_minUniqueHitsToKeep == 7 &&
              cfg.dc_minPtUniqueHitsToKeep == 12.f,
          "pT-adaptive cleaner parameters");
  }

  {
    // per-task scorers as IterationConfig::setupStandardFunctionsFromNames resolves them; the device supports
    // trackScoreDefault for every task and "keep" (no change) before the duplicate cleaner
    mkfit::IterationConfig scorers;
    makeConfig(scorers, 2, kNLayers);
    scorers.m_final_pick_track_scorer_name = "v2p2:llh";
    scorers.m_post_bkfit_track_scorer_name = "keep";
    scorers.m_duplicate_cleaner_track_scorer_name = "keep";
    auto esScorers = mkfitdev::fillESDataHost(ti, scorers);
    auto const& cfg = esScorers->hostConfigValue();
    check(cfg.steering_params[1].final_pick_track_scorer == mkfitdev::TrackScorer::V2p2Llh &&
              cfg.steering_params[1].post_bkfit_track_scorer == mkfitdev::TrackScorer::Keep &&
              cfg.steering_params[1].track_scorer == mkfitdev::TrackScorer::Default,
          "per-task scorers from their names");
    check(cfg.duplicate_cleaner_track_scorer == mkfitdev::TrackScorer::Keep, "duplicate-cleaner scorer from its name");
    const std::string violations = supportedConfigViolations(*esScorers);
    check(violations.find("final_pick_track_scorer of region 1") == std::string::npos,
          "'v2p2:llh' final-pick scorer accepted by the ES");
    check(violations.find("post_bkfit_track_scorer of region 1") != std::string::npos,
          "non-default post-backward-fit scorer refused");
    // the final sort of either finder takes it, as in MkBuilder
    check(buildConfigViolations(*esScorers, "cloneEngine").find("'v2p2:llh'") == std::string::npos,
          "'v2p2:llh' final-pick scorer accepted with the clone engine");
    check(buildConfigViolations(*esScorers, "v2p2").find("'v2p2:llh'") == std::string::npos,
          "'v2p2:llh' final-pick scorer accepted with v2p2");
    check(violations.find("duplicate_cleaner_track_scorer") == std::string::npos,
          "'keep' before the duplicate cleaner supported");
    scorers.m_final_pick_track_scorer_name = "phase1:default";
    scorers.m_post_bkfit_track_scorer_name = "";
    scorers.m_duplicate_cleaner_track_scorer_name = "default";
    const std::string rescore = supportedConfigViolations(*mkfitdev::fillESDataHost(ti, scorers));
    check(rescore.find("final_pick_track_scorer") == std::string::npos, "'phase1:default' final-pick scorer supported");
    check(rescore.find("duplicate_cleaner_track_scorer") != std::string::npos,
          "re-scoring before the duplicate cleaner refused");
  }

  {
    // building routines: the clone engine and MkFinderV2p2 are ported, the others are refused
    check(buildConfigViolations(*es, "cloneEngine").find("buildingRoutine") == std::string::npos,
          "buildingRoutine cloneEngine accepted");
    check(buildConfigViolations(*es, "v2p2").find("buildingRoutine") == std::string::npos,
          "buildingRoutine v2p2 accepted");
    check(
        buildConfigViolations(*es, "standard").find("buildingRoutine must be cloneEngine or v2p2") != std::string::npos,
        "buildingRoutine standard refused");
  }
  {
    // the v2p2 tunings (IterationConfig::m_v2p2), every member set away from its default
    mkfit::IterationConfig tuned;
    makeConfig(tuned, 2, kNLayers);
    mkfit::V2p2Params& vp = tuned.m_v2p2;
    for (mkfit::V2p2ScoreParams* sp : {&vp.score_outward, &vp.score_inward}) {
      const float offset = sp == &vp.score_outward ? 0.5f : 0.25f;
      sp->hit_bonus += offset;
      sp->overlap_bonus += offset;
      sp->chi2_weight += offset;
      sp->miss_penalty += offset;
      sp->edge_penalty += offset;
      sp->gap_penalty += offset;
      sp->stop_penalty += offset;
      sp->hit_eff = 0.5f * offset;
      for (int group = 0; group < mkfit::V2p2ScoreParams::n_hit_eff_grp; ++group)
        sp->hit_eff_grp[group] = 0.01f * (group + 1) + offset;
    }
    vp.chi2_trk_fac = 2.5f;
    vp.hit_chi2_cut = 11.34f;
    vp.max_presel_hits = 7;
    vp.max_hits_per_layer = 3;
    vp.window_dphi_trk_fac = 2.5f;
    vp.window_dq_trk_fac = 1.25f;
    vp.outside_hole_max_seed_hits = 3;
    auto esTuned = mkfitdev::fillESDataHost(ti, tuned);
    auto const& cfg = esTuned->hostConfigValue().v2p2;
    check(sameScoreParams(cfg.score_outward, vp.score_outward) && sameScoreParams(cfg.score_inward, vp.score_inward),
          "v2p2 score parameters copied");
    check(cfg.chi2_trk_fac == vp.chi2_trk_fac && cfg.hit_chi2_cut == vp.hit_chi2_cut &&
              cfg.max_presel_hits == vp.max_presel_hits && cfg.max_hits_per_layer == vp.max_hits_per_layer,
          "v2p2 hit acceptance and capacities copied");
    check(cfg.window_dphi_trk_fac == vp.window_dphi_trk_fac && cfg.window_dq_trk_fac == vp.window_dq_trk_fac &&
              cfg.outside_hole_max_seed_hits == vp.outside_hole_max_seed_hits,
          "v2p2 window factors and short-seed hole rule copied");
    check(supportedConfigViolations(*esTuned).find("v2p2.") == std::string::npos, "v2p2 tunings in range supported");
    // the defaults, as MkFitIterationConfigESProducer leaves them when its PSet is not set
    auto const& defaults = es->hostConfigValue().v2p2;
    const mkfit::V2p2Params ref;
    check(sameScoreParams(defaults.score_outward, ref.score_outward) &&
              sameScoreParams(defaults.score_inward, ref.score_inward) && defaults.chi2_trk_fac == ref.chi2_trk_fac &&
              defaults.hit_chi2_cut == ref.hit_chi2_cut && defaults.max_presel_hits == ref.max_presel_hits &&
              defaults.max_hits_per_layer == ref.max_hits_per_layer &&
              defaults.window_dphi_trk_fac == ref.window_dphi_trk_fac &&
              defaults.window_dq_trk_fac == ref.window_dq_trk_fac &&
              defaults.outside_hole_max_seed_hits == ref.outside_hole_max_seed_hits,
          "v2p2 default tunings copied");
    // beyond the device capacities (MkFitIterationConfigESProducer refuses them too)
    vp.max_presel_hits = mkfit::V2p2Params::max_presel_hits_limit + 1;
    vp.max_hits_per_layer = 0;
    const std::string outside = supportedConfigViolations(*mkfitdev::fillESDataHost(ti, tuned));
    check(outside.find("v2p2.max_presel_hits") != std::string::npos &&
              outside.find("v2p2.max_hits_per_layer") != std::string::npos,
          "v2p2 tunings beyond the device capacities refused");
  }
  {
    // double layers: the test's own partner function pairs layers 1 and 2 (TrackerInfo::setup_double_layers)
    mkfit::TrackerInfo paired;
    makeTracker(paired);
    paired.setup_double_layers([](int l) { return l == 1 ? 2 : l == 2 ? 1 : -1; });
    auto esPaired = mkfitdev::fillESDataHost(paired, ic);
    auto pairedLayers = esPaired->layers->const_view();
    bool samePartners = true;
    for (int l = 0; l < kNLayers; ++l)
      samePartners = samePartners && pairedLayers[l].double_layer_partner() == paired.layer(l).double_layer_partner();
    check(samePartners && pairedLayers[0].double_layer_partner() == -1 && pairedLayers[1].double_layer_partner() == 2 &&
              pairedLayers[2].double_layer_partner() == 1,
          "double-layer partners");
    bool noPartners = true;
    for (int l = 0; l < kNLayers; ++l)
      noPartners = noPartners && layers[l].double_layer_partner() == -1;
    check(noPartners, "no double layers in the unpaired tracker");
  }

  // error paths: never truncate silently
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, mkfitdev::es::kMaxRegions + 1, kNLayers);
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "too many regions throws");
  }
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, 2, kNLayers);
    bad.m_layer_configs[1].m_winpars_fwd = {1.f, 2.f};
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "winpars throws");
  }
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, 2, kNLayers);
    bad.m_pre_bkfit_filter_name = "phase9:nonsense";
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "unknown filter name throws");
  }
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, 2, kNLayers);
    bad.m_duplicate_cleaner_track_scorer_name = "phase9:nonsense";
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "unknown scorer name throws");
  }
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, 1, kNLayers);
    for (int i = 0; i < mkfitdev::es::kMaxPlanLayers; ++i)
      bad.m_steering_params[0].append_plan(0);
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "too long layer plan throws");
  }
  {
    // a plan entry with a second layer (LayerControl::m_layer_sec): the device plan has one layer per entry
    mkfit::IterationConfig bad;
    makeConfig(bad, 1, kNLayers);
    bad.m_steering_params[0].m_layer_plan[1].m_layer_sec = 2;
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "paired plan entry throws");
  }

  std::printf("testMkFitAlpakaESFill: %s (%d failures)\n", nFail == 0 ? "PASS" : "FAIL", nFail);
  return nFail == 0 ? 0 : 1;
}
