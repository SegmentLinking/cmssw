// Host unit test of mkfitdev::fillESDataHost on a small synthetic TrackerInfo / IterationConfig: covers what the
// Phase-2 cmsRun test cannot (an r-hole layer, which Phase-2 does not have) and the error paths (capacities,
// hit-window parameters, unknown function names). Run: testMkFitAlpakaESFill

#include <cstdio>
#include <stdexcept>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"

namespace {
  int nFail = 0;
  void check(bool ok, const char* what) {
    if (!ok) {
      ++nFail;
      std::printf("FAIL: %s\n", what);
    }
  }

  void makeTracker(mkfit::TrackerInfo& ti) {
    ti.create_layers(1, 1, 1);
    ti.create_material(30, 300.f, 12, 120.f);
    ti.layer_nc(0).set_limits(20.f, 25.f, -100.f, 100.f);
    ti.layer_nc(0).set_q_bin(2.f);
    ti.layer_nc(0).set_propagate_to(22.5f);
    ti.layer_nc(1).set_limits(20.f, 110.f, 120.f, 125.f);
    ti.layer_nc(1).set_q_bin(5.6f);
    ti.layer_nc(1).set_r_hole_range(30.5f, 42.25f);
    ti.layer_nc(2).set_limits(20.f, 110.f, -125.f, -120.f);
    ti.layer_nc(2).set_q_bin(5.6f);
    unsigned int detid = 0x14000000;
    for (int l = 0; l < 3; ++l) {
      for (int m = 0; m < 50; ++m)
        ti.layer_nc(l).register_module(mkfit::ModuleInfo({1.f * m, 2.f, 3.f}, {0, 0, 1}, {1, 0, 0}, detid += 4, 0));
      ti.layer_nc(l).resize_shapes(1);
      mkfit::ModuleShape ms;
      ms.round_assign(1.f, 0.f, 2.f, 0.01f);
      ti.layer_nc(l).register_shape(ms, 0);
    }
    for (int iz = 0; iz < 30; ++iz)
      for (int ir = 0; ir < 12; ++ir) {
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

  template <typename F>
  bool throws(F&& f) {
    try {
      f();
    } catch (cms::Exception const&) {
      return true;
    }
    return false;
  }
}  // namespace

int main() {
  mkfit::TrackerInfo ti;
  makeTracker(ti);
  mkfit::IterationConfig ic;
  makeConfig(ic, 2, 3);

  auto es = mkfitdev::fillESDataHost(ti, ic);
  auto L = es->layers->const_view();
  check(es->sizes.nLayers == 3 && es->sizes.nModules == 150 && es->sizes.nShapes == 3, "sizes");
  check(!L[0].has_r_range_hole() && !L[2].has_r_range_hole(), "no hole on layers 0, 2");
  check(L[1].has_r_range_hole(), "hole on layer 1");
  check(L[1].hole_r_min() == 30.5f && L[1].hole_r_max() == 42.25f, "hole range recovered exactly");
  std::printf("hole recovered: [%.9g, %.9g]\n", L[1].hole_r_min(), L[1].hole_r_max());
  const mkfitdev::ESView v = es->view();
  for (float r = 15.f; r < 115.f; r += 0.0137f)
    check(v.isInRHole(1, r) == ti.layer(1).is_in_r_hole(r), "isInRHole vs MkFitCore");
  for (float dr : {0.f, 0.5f, 3.f})
    for (float r = 15.f; r < 115.f; r += 0.0137f) {
      auto a = v.isWithinRSensitiveRegion(1, r, dr);
      auto b = ti.layer(1).is_within_r_sensitive_region(r, dr);
      check(a.wsr == int(b.m_wsr) && a.in_gap == bool(b.m_in_gap), "isWithinRSensitiveRegion vs MkFitCore");
    }
  for (int m = 0; m < 150; ++m) {
    const uint32_t d = es->modules->const_view()[m].detid();
    check(v.findModule(d) == m, "findModule");
    check(v.findModule(d + 1) == mkfitdev::es::kNoModule, "findModule unknown");
  }
  for (float z = -310.f; z < 310.f; z += 0.77f)
    for (float r = -3.f; r < 125.f; r += 0.61f) {
      float b, x;
      v.material.materialChecked(z, r, b, x);
      const auto m = ti.material_checked(z, r);
      check(b == m.bbxi && x == m.radl, "material vs MkFitCore");
    }
  auto const& C = es->hostConfigValue();
  check(C.n_regions == 2 && C.region_order[0] == 1 && C.region_order[1] == 0, "regions");
  check(C.steering_params[1].n_plan == 3 && C.steering_params[1].layer[2] == 2, "layer plan");
  check(C.steering_params[0].bkw_search_pickup == 2 && C.steering_params[0].fwd_search_pickup == 1, "pickups");
  check(C.duplicate_cleaner == mkfitdev::DuplicateCleaner::SharedHitsPixelSeed, "duplicate cleaner enum");
  check(C.steering_params[0].track_scorer == mkfitdev::TrackScorer::Default, "scorer from default");
  check(C.seed_partitioner == mkfitdev::SeedPartitioner::None, "empty partitioner");

  // error paths: never truncate silently
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, mkfitdev::es::kMaxRegions + 1, 3);
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "too many regions throws");
  }
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, 2, 3);
    bad.m_layer_configs[1].m_winpars_fwd = {1.f, 2.f};
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "winpars throws");
  }
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, 2, 3);
    bad.m_pre_bkfit_filter_name = "phase9:nonsense";
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "unknown filter name throws");
  }
  {
    mkfit::IterationConfig bad;
    makeConfig(bad, 1, 3);
    for (int i = 0; i < mkfitdev::es::kMaxPlanLayers; ++i)
      bad.m_steering_params[0].append_plan(0);
    check(throws([&] { mkfitdev::fillESDataHost(ti, bad); }), "too long layer plan throws");
  }

  std::printf("testMkFitAlpakaESFill: %s (%d failures)\n", nFail == 0 ? "PASS" : "FAIL", nFail);
  return nFail == 0 ? 0 : 1;
}
