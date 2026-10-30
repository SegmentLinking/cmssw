// Host unit test of mkfitdev::fillESDataHost on a small synthetic TrackerInfo / IterationConfig: covers what the
// Phase-2 cmsRun test cannot (an r-hole layer, which Phase-2 does not have) and the error paths (capacities,
// hit-window parameters, unknown function names). Run: testMkFitAlpakaESFill

#include <cstdio>
#include <stdexcept>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"

namespace {
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
  check(!layers[0].has_r_range_hole() && !layers[2].has_r_range_hole(), "no hole on layers 0, 2");
  check(layers[1].has_r_range_hole(), "hole on layer 1");
  check(layers[1].hole_r_min() == 30.5f && layers[1].hole_r_max() == 42.25f, "hole range recovered exactly");
  std::printf("hole recovered: [%.9g, %.9g]\n", layers[1].hole_r_min(), layers[1].hole_r_max());
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
  check(config.seed_partitioner == mkfitdev::SeedPartitioner::None, "empty partitioner");

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
    makeConfig(bad, 1, kNLayers);
    for (int i = 0; i < mkfitdev::es::kMaxPlanLayers; ++i)
      bad.m_steering_params[0].append_plan(0);
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "too long layer plan throws");
  }

  std::printf("testMkFitAlpakaESFill: %s (%d failures)\n", nFail == 0 ? "PASS" : "FAIL", nFail);
  return nFail == 0 ? 0 : 1;
}
