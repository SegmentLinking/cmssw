// Host fill of the MkFitAlpaka ES product from MkFitCore host ES objects.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"

namespace mkfitdev {

  namespace {

    SeedCleaner seedCleanerFromName(std::string const& name) {
      if (name.empty())
        return SeedCleaner::None;
      if (name == "phase1:default")
        return SeedCleaner::Phase1Default;
      throw cms::Exception("MkFitAlpakaES") << "unknown seed cleaner '" << name << "'";
    }

    SeedPartitioner seedPartitionerFromName(std::string const& name) {
      if (name.empty())
        return SeedPartitioner::None;
      if (name == "phase2:1")
        return SeedPartitioner::Phase2_1;
      if (name == "phase2:1:debug")
        return SeedPartitioner::Phase2_1_debug;
      if (name == "phase1:0")
        return SeedPartitioner::Phase1_0;
      if (name == "phase1:1")
        return SeedPartitioner::Phase1_1;
      if (name == "phase1:1:debug")
        return SeedPartitioner::Phase1_1_debug;
      throw cms::Exception("MkFitAlpakaES") << "unknown seed partitioner '" << name << "'";
    }

    CandFilter candFilterFromName(std::string const& name) {
      if (name.empty())
        return CandFilter::None;
      if (name == "phase1:qfilter_n_hits")
        return CandFilter::NHits;
      if (name == "phase1:qfilter_n_hits_pixseed")
        return CandFilter::NHitsPixSeed;
      if (name == "phase1:qfilter_n_layers")
        return CandFilter::NLayers;
      if (name == "phase1:qfilter_pixelLessFwd")
        return CandFilter::PixelLessFwd;
      if (name == "phase1:qfilter_pixelLessBkwd")
        return CandFilter::PixelLessBkwd;
      throw cms::Exception("MkFitAlpakaES") << "unknown candidate filter '" << name << "'";
    }

    DuplicateCleaner duplicateCleanerFromName(std::string const& name) {
      if (name.empty())
        return DuplicateCleaner::None;
      if (name == "phase1:clean_duplicates")
        return DuplicateCleaner::CleanDuplicates;
      if (name == "phase1:clean_duplicates_sharedhits")
        return DuplicateCleaner::SharedHits;
      if (name == "phase1:clean_duplicates_sharedhits_pixelseed")
        return DuplicateCleaner::SharedHitsPixelSeed;
      if (name == "phase2:clean_duplicates_sharedhits_pixelpriority")
        return DuplicateCleaner::SharedHitsPixelPriority;
      throw cms::Exception("MkFitAlpakaES") << "unknown duplicate cleaner '" << name << "'";
    }

    TrackScorer trackScorerFromName(std::string const& name) {
      if (name.empty())
        return TrackScorer::None;
      if (name == "default" || name == "phase1:default")
        return TrackScorer::Default;
      throw cms::Exception("MkFitAlpakaES") << "unknown track scorer '" << name << "'";
    }

    PropFlags convert(mkfit::PropagationFlags const& flags) {
      return PropFlags{
          flags.use_param_b_field, flags.apply_material, flags.copy_input_state_on_fail, flags.use_fit_b_field};
    }

    IterParams convert(mkfit::IterationParams const& params) {
      IterParams out;
      out.maxCandsPerSeed = params.maxCandsPerSeed;
      out.maxHolesPerCand = params.maxHolesPerCand;
      out.maxConsecHoles = params.maxConsecHoles;
      out.chi2Cut_min = params.chi2Cut_min;
      out.pTCutOverlap = params.pTCutOverlap;
      out.recheckOverlap = params.recheckOverlap;
      out.useHitSelectionV2 = params.useHitSelectionV2;
      out.minHitsQF = params.minHitsQF;
      out.minPtCut = params.minPtCut;
      out.maxClusterSize = params.maxClusterSize;
      return out;
    }

    // LayerInfo keeps the r-hole range private (only is_in_r_hole() is public). Recover it exactly: the hole is
    // the open interval (min, max), so min = the float just below the smallest r inside, max = the float just
    // above the largest r inside. A point inside is found by a 10 um scan of [rin, rout] (holes are cm-scale).
    bool recoverRHole(mkfit::LayerInfo const& li, float& hmin, float& hmax) {
      const float lo = std::min(li.rin(), li.rout()), hi = std::max(li.rin(), li.rout());
      float inside = std::numeric_limits<float>::quiet_NaN();
      for (double radius = lo; radius <= hi; radius += 1e-3) {
        if (li.is_in_r_hole(static_cast<float>(radius))) {
          inside = static_cast<float>(radius);
          break;
        }
      }
      if (std::isnan(inside))
        return false;
      // positive floats are ordered like their bit patterns: bisect on the bits
      auto bits = [](float value) { return std::bit_cast<uint32_t>(value); };
      auto flt = [](uint32_t pattern) { return std::bit_cast<float>(pattern); };
      // smallest float r with is_in_r_hole(r) in (0, inside]
      uint32_t lower = 0, upper = bits(inside);  // invariant: !hole(lower) (r = 0 is never inside), hole(upper)
      while (upper - lower > 1) {
        uint32_t mid = lower + (upper - lower) / 2;
        (li.is_in_r_hole(flt(mid)) ? upper : lower) = mid;
      }
      hmin = flt(lower);
      // largest float r with is_in_r_hole(r)
      lower = bits(inside);
      upper = bits(std::numeric_limits<float>::max());  // invariant: hole(lower), !hole(upper)
      while (upper - lower > 1) {
        uint32_t mid = lower + (upper - lower) / 2;
        (li.is_in_r_hole(flt(mid)) ? lower : upper) = mid;
      }
      hmax = flt(upper);
      return true;
    }

  }  // namespace

  std::unique_ptr<ESDataHost> fillESDataHost(mkfit::TrackerInfo const& ti, mkfit::IterationConfig const& ic) {
    auto const& host = cms::alpakatools::host();

    ESSizes sizes;
    sizes.nLayers = ti.n_layers();
    for (int layer = 0; layer < sizes.nLayers; ++layer)
      sizes.nModules += ti.layer(layer).n_modules();
    sizes.nMaterialBins = ti.mat_nbins_z() * ti.mat_nbins_r();
    sizes.matNBinsZ = ti.mat_nbins_z();
    sizes.matNBinsR = ti.mat_nbins_r();
    // as TrackerInfo::create_material: m_mat_fac_z = nBinZ / m_mat_range_z
    sizes.matFacZ = ti.mat_nbins_z() / ti.mat_range_z();
    sizes.matFacR = ti.mat_nbins_r() / ti.mat_range_r();
    // field constants, set by MkFitGeometryESProducer::produce: mkfit::Config::mag_* (track building, CMSSW's
    // ParabolicMf) and mag_fit_* (the final fit, from bFieldParams)
    sizes.bFieldFit = Config::BFieldParams{
        mkfit::Config::mag_fit_c1, mkfit::Config::mag_fit_b0, mkfit::Config::mag_fit_b1, mkfit::Config::mag_fit_a};
    sizes.bField =
        Config::BFieldParams{mkfit::Config::mag_c1, mkfit::Config::mag_b0, mkfit::Config::mag_b1, mkfit::Config::mag_a};
    // detid table: power of two >= 2 x modules
    {
      uint32_t cap = 1;
      int log2cap = 0;
      while (cap < 2u * static_cast<uint32_t>(sizes.nModules)) {
        cap <<= 1;
        ++log2cap;
      }
      if (log2cap < 1) {
        cap = 2;
        log2cap = 1;
      }
      sizes.detIdMapCapacity = cap;
    }

    if (static_cast<int>(ic.m_layer_configs.size()) != sizes.nLayers)
      throw cms::Exception("MkFitAlpakaES") << "IterationConfig has " << ic.m_layer_configs.size()
                                            << " layer configs, TrackerInfo has " << sizes.nLayers << " layers";

    // ---------------- layers + modules
    auto layers = std::make_shared<PortableHostCollection<LayerInfoSoA>>(host, sizes.nLayers);
    auto modules = std::make_shared<PortableHostCollection<ModuleInfoSoA>>(host, sizes.nModules);
    auto lv = layers->view();
    auto mv = modules->view();
    int moduleBegin = 0;
    for (int layer = 0; layer < sizes.nLayers; ++layer) {
      mkfit::LayerInfo const& li = ti.layer(layer);
      auto row = lv[layer];
      row.layer_type() = static_cast<int>(li.layer_type());
      row.rin() = li.rin();
      row.rout() = li.rout();
      row.zmin() = li.zmin();
      row.zmax() = li.zmax();
      row.propagate_to() = li.propagate_to();
      row.q_bin() = li.q_bin();
      float hmin = 0.f, hmax = 0.f;
      const bool hasHole = recoverRHole(li, hmin, hmax);
      row.has_r_range_hole() = hasHole;
      row.hole_r_min() = hmin;
      row.hole_r_max() = hmax;
      row.is_pixel() = li.is_pixel();
      row.has_charge() = li.has_charge();
      row.module_begin() = moduleBegin;
      // LayerOfHits binning: use the MkFitCore Initializator itself
      mkfit::LayerOfHits::Initializator init(li);
      row.q_min() = init.m_qmin;
      row.q_max() = init.m_qmax;
      row.n_q() = init.m_nq;
      // IterationLayerConfig: the dynamic hit-window parameters are not ported
      mkfit::IterationLayerConfig const& lc = ic.m_layer_configs[layer];
      if (lc.m_layer != layer)
        throw cms::Exception("MkFitAlpakaES") << "layer config " << layer << " has m_layer " << lc.m_layer;
      if (!lc.m_winpars_fwd.empty() || !lc.m_winpars_bkw.empty())
        throw cms::Exception("MkFitAlpakaES")
            << "layer " << layer << " has hit-window parameters (m_winpars_*), not supported by MkFitAlpaka ES yet";

      for (int sid = 0; sid < li.n_modules(); ++sid) {
        mkfit::ModuleInfo const& mi = li.module_info(sid);
        auto moduleRow = mv[moduleBegin + sid];
        moduleRow.pos_x() = mi.pos[0];
        moduleRow.pos_y() = mi.pos[1];
        moduleRow.pos_z() = mi.pos[2];
        moduleRow.zdir_x() = mi.zdir[0];
        moduleRow.zdir_y() = mi.zdir[1];
        moduleRow.zdir_z() = mi.zdir[2];
        moduleRow.xdir_x() = mi.xdir[0];
        moduleRow.xdir_y() = mi.xdir[1];
        moduleRow.xdir_z() = mi.xdir[2];
        moduleRow.detid() = mi.detid;
        moduleRow.layer() = layer;
        moduleRow.sid() = sid;
        moduleRow.radl() = mi.radl;
        moduleRow.bbxi() = mi.bbxi;
      }
      moduleBegin += li.n_modules();
    }

    // ---------------- detid -> module table (open addressing, filled in module order)
    auto detIdMap = std::make_shared<PortableHostCollection<DetIdMapSoA>>(host, sizes.detIdMapCapacity);
    {
      auto dv = detIdMap->view();
      const uint32_t cap = sizes.detIdMapCapacity;
      const uint32_t shift = 32 - std::countr_zero(cap);
      const uint32_t mask = cap - 1;
      for (uint32_t i = 0; i < cap; ++i) {
        dv[i].key() = 0;
        dv[i].module() = es::kNoModule;
      }
      int maxProbe = 0;
      for (int imodule = 0; imodule < sizes.nModules; ++imodule) {
        const uint32_t detid = mv[imodule].detid();
        if (detid == 0)
          throw cms::Exception("MkFitAlpakaES") << "module " << imodule << " has detid 0";
        uint32_t slot = detIdHashSlot(detid, shift);
        int probe = 0;
        while (dv[slot].key() != 0) {
          if (dv[slot].key() == detid)
            throw cms::Exception("MkFitAlpakaES") << "duplicate detid " << detid << " (module " << imodule << ")";
          slot = (slot + 1) & mask;
          ++probe;
        }
        dv[slot].key() = detid;
        dv[slot].module() = imodule;
        maxProbe = std::max(maxProbe, probe);
      }
      dv.hash_shift() = shift;
      dv.max_probe() = maxProbe;
    }

    // ---------------- material map
    auto material = std::make_shared<PortableHostCollection<MaterialSoA>>(host, sizes.nMaterialBins);
    {
      auto matv = material->view();
      for (int iz = 0; iz < sizes.matNBinsZ; ++iz)
        for (int ir = 0; ir < sizes.matNBinsR; ++ir) {
          matv[iz * sizes.matNBinsR + ir].bbxi() = ti.material_bbxi(iz, ir);
          matv[iz * sizes.matNBinsR + ir].radl() = ti.material_radl(iz, ir);
        }
    }

    // ---------------- scalar configuration
    auto config = std::make_shared<PortableHostObject<ESConfig>>(host);
    ESConfig& cfg = config->value();
    std::memset(&cfg, 0, sizeof(ESConfig));
    cfg.usePropToPlane = mkfit::Config::usePropToPlane;
    cfg.usePtMultScat = mkfit::Config::usePtMultScat;

    mkfit::PropagationConfig const& pc = ti.prop_config();
    cfg.prop_config.backward_fit_to_pca = pc.backward_fit_to_pca;
    cfg.prop_config.finding_requires_propagation_to_hit_pos = pc.finding_requires_propagation_to_hit_pos;
    cfg.prop_config.finding_inter_layer_pflags = convert(pc.finding_inter_layer_pflags);
    cfg.prop_config.finding_intra_layer_pflags = convert(pc.finding_intra_layer_pflags);
    cfg.prop_config.backward_fit_pflags = convert(pc.backward_fit_pflags);
    cfg.prop_config.final_fit_pflags = convert(pc.final_fit_pflags);

    // the final fit's switches (PropagationConfig::final_fit_ffflags), set by MkFitGeometryESProducer::produce
    mkfit::FinalFitFlags const& ff = pc.final_fit_ffflags;
    cfg.refit.bFieldAtMid = ff.b_field_at_mid;
    cfg.refit.radialFieldCorr = ff.radial_field_corr;
    cfg.refit.elossSignFromPass = ff.eloss_sign_from_pass;
    cfg.refit.bkwMsFixedMomentum = ff.bkw_ms_fixed_momentum;
    cfg.refit.bkwSubSteps = ff.bkw_sub_steps;
    cfg.refit.materialPerModule = ff.material_per_module;

    cfg.requires_seed_hit_sorting = ic.m_requires_seed_hit_sorting;
    cfg.backward_search = ic.m_backward_search;
    cfg.backward_drop_seed_hits = ic.m_backward_drop_seed_hits;
    cfg.backward_fit_outlier_chi2 = ic.m_backward_fit_outlier_chi2;
    cfg.backward_fit_max_outliers = ic.m_backward_fit_max_outliers;
    cfg.backward_fit_outlier_min_pt = ic.m_backward_fit_outlier_min_pt;
    cfg.backward_search_min_pixel_layers = ic.m_backward_search_min_pixel_layers;
    cfg.backward_search_prompt_max_d0 = ic.m_backward_search_prompt_max_d0;
    for (int word = 0; word < 4; ++word)
      cfg.pixel_layer_mask[word] = 0;
    if (ti.n_layers() > 256)
      throw cms::Exception("MkFitAlpakaES") << "pixel layer mask: " << ti.n_layers() << " layers > 256";
    for (int layer = 0; layer < ti.n_layers(); ++layer)
      if (ti.layer(layer).is_pixel())
        cfg.pixel_layer_mask[layer >> 6] |= 1ull << (layer & 63);
    cfg.dc_fracSharedHits = ic.dc_fracSharedHits;
    cfg.dc_drth_central = ic.dc_drth_central;
    cfg.dc_drth_obarrel = ic.dc_drth_obarrel;
    cfg.dc_drth_forward = ic.dc_drth_forward;
    cfg.params = convert(ic.m_params);
    cfg.backward_params = convert(ic.m_backward_params);

    const int nRegions = ic.m_steering_params.size();
    if (nRegions > es::kMaxRegions || ic.m_n_regions != nRegions ||
        static_cast<int>(ic.m_region_order.size()) != nRegions)
      throw cms::Exception("MkFitAlpakaES")
          << "regions: m_n_regions " << ic.m_n_regions << ", steering params " << nRegions << ", region order "
          << ic.m_region_order.size() << ", capacity " << es::kMaxRegions;
    cfg.n_regions = nRegions;

    cfg.seed_cleaner = seedCleanerFromName(ic.m_seed_cleaner_name);
    cfg.seed_partitioner = seedPartitionerFromName(ic.m_seed_partitioner_name);
    cfg.pre_bkfit_filter = candFilterFromName(ic.m_pre_bkfit_filter_name);
    cfg.post_bkfit_filter = candFilterFromName(ic.m_post_bkfit_filter_name);
    cfg.duplicate_cleaner = duplicateCleanerFromName(ic.m_duplicate_cleaner_name);
    cfg.default_track_scorer = trackScorerFromName(ic.m_default_track_scorer_name);

    for (int region = 0; region < es::kMaxRegions; ++region) {
      SteeringRegion& steering = cfg.steering_params[region];
      for (int i = 0; i < es::kMaxPlanLayers; ++i)
        steering.layer[i] = -1;
      if (region >= nRegions) {
        steering.region = -1;
        steering.bkw_search_pickup = -1;
        continue;
      }
      mkfit::SteeringParams const& sp = ic.m_steering_params[region];
      const int nPlan = sp.m_layer_plan.size();
      if (nPlan > es::kMaxPlanLayers)
        throw cms::Exception("MkFitAlpakaES")
            << "region " << region << " layer plan has " << nPlan << " entries, capacity " << es::kMaxPlanLayers;
      steering.region = sp.m_region;
      steering.n_plan = nPlan;
      steering.fwd_search_pickup = sp.m_fwd_search_pickup;
      steering.bkw_search_pickup = sp.m_bkw_search_pickup;
      // as IterationConfig::setupStandardFunctionsFromNames
      steering.track_scorer =
          sp.m_track_scorer_name.empty() ? cfg.default_track_scorer : trackScorerFromName(sp.m_track_scorer_name);
      for (int i = 0; i < nPlan; ++i)
        steering.layer[i] = sp.m_layer_plan[i].m_layer;
    }

    return std::make_unique<ESDataHost>(
        std::move(layers), std::move(modules), std::move(detIdMap), std::move(material), config, config, sizes);
  }

}  // namespace mkfitdev
