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

    SeedCleaner seedCleanerFromName(std::string const& n) {
      if (n.empty())
        return SeedCleaner::None;
      if (n == "phase1:default")
        return SeedCleaner::Phase1Default;
      throw cms::Exception("MkFitAlpakaES") << "unknown seed cleaner '" << n << "'";
    }

    SeedPartitioner seedPartitionerFromName(std::string const& n) {
      if (n.empty())
        return SeedPartitioner::None;
      if (n == "phase2:1")
        return SeedPartitioner::Phase2_1;
      if (n == "phase2:1:debug")
        return SeedPartitioner::Phase2_1_debug;
      if (n == "phase1:0")
        return SeedPartitioner::Phase1_0;
      if (n == "phase1:1")
        return SeedPartitioner::Phase1_1;
      if (n == "phase1:1:debug")
        return SeedPartitioner::Phase1_1_debug;
      throw cms::Exception("MkFitAlpakaES") << "unknown seed partitioner '" << n << "'";
    }

    CandFilter candFilterFromName(std::string const& n) {
      if (n.empty())
        return CandFilter::None;
      if (n == "phase1:qfilter_n_hits")
        return CandFilter::NHits;
      if (n == "phase1:qfilter_n_hits_pixseed")
        return CandFilter::NHitsPixSeed;
      if (n == "phase1:qfilter_n_layers")
        return CandFilter::NLayers;
      if (n == "phase1:qfilter_pixelLessFwd")
        return CandFilter::PixelLessFwd;
      if (n == "phase1:qfilter_pixelLessBkwd")
        return CandFilter::PixelLessBkwd;
      throw cms::Exception("MkFitAlpakaES") << "unknown candidate filter '" << n << "'";
    }

    DuplicateCleaner duplicateCleanerFromName(std::string const& n) {
      if (n.empty())
        return DuplicateCleaner::None;
      if (n == "phase1:clean_duplicates")
        return DuplicateCleaner::CleanDuplicates;
      if (n == "phase1:clean_duplicates_sharedhits")
        return DuplicateCleaner::SharedHits;
      if (n == "phase1:clean_duplicates_sharedhits_pixelseed")
        return DuplicateCleaner::SharedHitsPixelSeed;
      if (n == "phase2:clean_duplicates_sharedhits_pixelpriority")
        return DuplicateCleaner::SharedHitsPixelPriority;
      throw cms::Exception("MkFitAlpakaES") << "unknown duplicate cleaner '" << n << "'";
    }

    TrackScorer trackScorerFromName(std::string const& n) {
      if (n.empty())
        return TrackScorer::None;
      if (n == "default" || n == "phase1:default")
        return TrackScorer::Default;
      throw cms::Exception("MkFitAlpakaES") << "unknown track scorer '" << n << "'";
    }

    PropFlags convert(mkfit::PropagationFlags const& f) {
      return PropFlags{f.use_param_b_field, f.apply_material, f.copy_input_state_on_fail};
    }

    IterParams convert(mkfit::IterationParams const& p) {
      IterParams o;
      o.nlayers_per_seed = p.nlayers_per_seed;
      o.maxCandsPerSeed = p.maxCandsPerSeed;
      o.maxHolesPerCand = p.maxHolesPerCand;
      o.maxConsecHoles = p.maxConsecHoles;
      o.chi2Cut_min = p.chi2Cut_min;
      o.chi2CutOverlap = p.chi2CutOverlap;
      o.pTCutOverlap = p.pTCutOverlap;
      o.recheckOverlap = p.recheckOverlap;
      o.useHitSelectionV2 = p.useHitSelectionV2;
      o.minHitsQF = p.minHitsQF;
      o.minPtCut = p.minPtCut;
      o.maxClusterSize = p.maxClusterSize;
      return o;
    }

    // LayerInfo keeps the r-hole range private (only is_in_r_hole() is public). Recover it exactly: the hole is
    // the open interval (min, max), so min = the float just below the smallest r inside, max = the float just
    // above the largest r inside. A point inside is found by a 10 um scan of [rin, rout] (holes are cm-scale).
    bool recoverRHole(mkfit::LayerInfo const& li, float& hmin, float& hmax) {
      const float lo = std::min(li.rin(), li.rout()), hi = std::max(li.rin(), li.rout());
      float inside = std::numeric_limits<float>::quiet_NaN();
      for (double r = lo; r <= hi; r += 1e-3) {
        if (li.is_in_r_hole(static_cast<float>(r))) {
          inside = static_cast<float>(r);
          break;
        }
      }
      if (std::isnan(inside))
        return false;
      // positive floats are ordered like their bit patterns: bisect on the bits
      auto bits = [](float x) { return std::bit_cast<uint32_t>(x); };
      auto flt = [](uint32_t b) { return std::bit_cast<float>(b); };
      // smallest float r with is_in_r_hole(r) in (0, inside]
      uint32_t a = 0, b = bits(inside);  // invariant: !hole(a) (r = 0 is never inside), hole(b)
      while (b - a > 1) {
        uint32_t m = a + (b - a) / 2;
        (li.is_in_r_hole(flt(m)) ? b : a) = m;
      }
      hmin = flt(a);
      // largest float r with is_in_r_hole(r)
      a = bits(inside);
      b = bits(std::numeric_limits<float>::max());  // invariant: hole(a), !hole(b)
      while (b - a > 1) {
        uint32_t m = a + (b - a) / 2;
        (li.is_in_r_hole(flt(m)) ? a : b) = m;
      }
      hmax = flt(b);
      return true;
    }

  }  // namespace

  std::unique_ptr<ESDataHost> fillESDataHost(mkfit::TrackerInfo const& ti, mkfit::IterationConfig const& ic) {
    auto const& host = cms::alpakatools::host();

    ESSizes sizes;
    sizes.nLayers = ti.n_layers();
    for (int l = 0; l < sizes.nLayers; ++l) {
      sizes.nModules += ti.layer(l).n_modules();
      sizes.nShapes += ti.layer(l).n_shapes();
    }
    sizes.nMaterialBins = ti.mat_nbins_z() * ti.mat_nbins_r();
    sizes.matNBinsZ = ti.mat_nbins_z();
    sizes.matNBinsR = ti.mat_nbins_r();
    // as TrackerInfo::create_material: m_mat_fac_z = nBinZ / m_mat_range_z
    sizes.matFacZ = ti.mat_nbins_z() / ti.mat_range_z();
    sizes.matFacR = ti.mat_nbins_r() / ti.mat_range_r();
    // field constants: MkFitGeometryESProducer::produce sets mkfit::Config::mag_* from bFieldParams
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

    // ---------------- layers + modules + shapes
    auto layers = std::make_shared<PortableHostCollection<LayerInfoSoA>>(host, sizes.nLayers);
    auto modules = std::make_shared<PortableHostCollection<ModuleInfoSoA>>(host, sizes.nModules);
    auto shapes = std::make_shared<PortableHostCollection<ModuleShapeSoA>>(host, sizes.nShapes);
    auto lv = layers->view();
    auto mv = modules->view();
    auto sv = shapes->view();
    int moduleBegin = 0, shapeBegin = 0;
    for (int l = 0; l < sizes.nLayers; ++l) {
      mkfit::LayerInfo const& li = ti.layer(l);
      auto row = lv[l];
      row.layer_id() = li.layer_id();
      row.layer_type() = static_cast<int>(li.layer_type());
      row.subdet() = li.subdet();
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
      row.is_stereo() = li.is_stereo();
      row.is_pixel() = li.is_pixel();
      row.has_charge() = li.has_charge();
      row.module_begin() = moduleBegin;
      row.n_modules() = li.n_modules();
      row.shape_begin() = shapeBegin;
      row.n_shapes() = li.n_shapes();
      // LayerOfHits binning: use the MkFitCore Initializator itself
      mkfit::LayerOfHits::Initializator init(li);
      row.q_min() = init.m_qmin;
      row.q_max() = init.m_qmax;
      row.n_q() = init.m_nq;
      // IterationLayerConfig
      mkfit::IterationLayerConfig const& lc = ic.m_layer_configs[l];
      if (lc.m_layer != l)
        throw cms::Exception("MkFitAlpakaES") << "layer config " << l << " has m_layer " << lc.m_layer;
      if (!lc.m_winpars_fwd.empty() || !lc.m_winpars_bkw.empty())
        throw cms::Exception("MkFitAlpakaES")
            << "layer " << l << " has hit-window parameters (m_winpars_*), not supported by MkFitAlpaka ES yet";
      row.select_min_dphi() = lc.min_dphi();
      row.select_max_dphi() = lc.max_dphi();
      row.select_min_dq() = lc.min_dq();
      row.select_max_dq() = lc.max_dq();

      for (int s = 0; s < li.n_modules(); ++s) {
        mkfit::ModuleInfo const& mi = li.module_info(s);
        auto m = mv[moduleBegin + s];
        m.pos_x() = mi.pos[0];
        m.pos_y() = mi.pos[1];
        m.pos_z() = mi.pos[2];
        m.zdir_x() = mi.zdir[0];
        m.zdir_y() = mi.zdir[1];
        m.zdir_z() = mi.zdir[2];
        m.xdir_x() = mi.xdir[0];
        m.xdir_y() = mi.xdir[1];
        m.xdir_z() = mi.xdir[2];
        m.detid() = mi.detid;
        m.shapeid() = mi.shapeid;
        m.layer() = l;
        m.sid() = s;
        m.radl() = mi.radl;
        m.bbxi() = mi.bbxi;
      }
      for (int s = 0; s < li.n_shapes(); ++s) {
        mkfit::ModuleShape const& ms = li.module_shape(s);
        auto r = sv[shapeBegin + s];
        r.dx1() = ms.dx1;
        r.dx2() = ms.dx2;
        r.dy() = ms.dy;
        r.dz() = ms.dz;
      }
      moduleBegin += li.n_modules();
      shapeBegin += li.n_shapes();
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
      for (int m = 0; m < sizes.nModules; ++m) {
        const uint32_t detid = mv[m].detid();
        if (detid == 0)
          throw cms::Exception("MkFitAlpakaES") << "module " << m << " has detid 0";
        uint32_t slot = detIdHashSlot(detid, shift);
        int probe = 0;
        while (dv[slot].key() != 0) {
          if (dv[slot].key() == detid)
            throw cms::Exception("MkFitAlpakaES") << "duplicate detid " << detid << " (module " << m << ")";
          slot = (slot + 1) & mask;
          ++probe;
        }
        dv[slot].key() = detid;
        dv[slot].module() = m;
        maxProbe = std::max(maxProbe, probe);
      }
      dv.hash_shift() = shift;
      dv.max_probe() = maxProbe;
      dv.n_entries() = sizes.nModules;
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
      matv.nbins_z() = sizes.matNBinsZ;
      matv.nbins_r() = sizes.matNBinsR;
      matv.range_z() = ti.mat_range_z();
      matv.range_r() = ti.mat_range_r();
      matv.fac_z() = sizes.matFacZ;
      matv.fac_r() = sizes.matFacR;
    }

    // ---------------- scalar configuration
    auto config = std::make_shared<PortableHostObject<ESConfig>>(host);
    ESConfig& c = config->value();
    std::memset(&c, 0, sizeof(ESConfig));
    c.n_layers = sizes.nLayers;
    c.n_barrel_layers = ti.barrel_layers().size();
    c.n_ecap_pos_layers = ti.endcap_pos_layers().size();
    c.n_ecap_neg_layers = ti.endcap_neg_layers().size();
    c.outer_barrel_layer = ti.barrel_layers().empty() ? -1 : ti.outer_barrel_layer().layer_id();
    c.n_total_modules = sizes.nModules;
    c.n_total_shapes = sizes.nShapes;

    c.usePropToPlane = mkfit::Config::usePropToPlane;
    c.usePtMultScat = mkfit::Config::usePtMultScat;
    c.maxdPt = mkfit::Config::maxdPt;
    c.maxdPhi = mkfit::Config::maxdPhi;
    c.maxdEta = mkfit::Config::maxdEta;
    c.maxdR = mkfit::Config::maxdR;
    c.minFracHitsShared = mkfit::Config::minFracHitsShared;
    c.maxd1pt = mkfit::Config::maxd1pt;
    c.maxdphi = mkfit::Config::maxdphi;
    c.maxdcth = mkfit::Config::maxdcth;
    c.maxcth_ob = mkfit::Config::maxcth_ob;
    c.maxcth_fw = mkfit::Config::maxcth_fw;

    mkfit::PropagationConfig const& pc = ti.prop_config();
    c.prop_config.backward_fit_to_pca = pc.backward_fit_to_pca;
    c.prop_config.finding_requires_propagation_to_hit_pos = pc.finding_requires_propagation_to_hit_pos;
    c.prop_config.finding_inter_layer_pflags = convert(pc.finding_inter_layer_pflags);
    c.prop_config.finding_intra_layer_pflags = convert(pc.finding_intra_layer_pflags);
    c.prop_config.backward_fit_pflags = convert(pc.backward_fit_pflags);
    c.prop_config.forward_fit_pflags = convert(pc.forward_fit_pflags);
    c.prop_config.seed_fit_pflags = convert(pc.seed_fit_pflags);
    c.prop_config.pca_prop_pflags = convert(pc.pca_prop_pflags);

    // runtime globals, set by MkFitGeometryESProducer::produce (the MkFitGeometry is made before this)
    c.refit.bFieldAtMid = mkfit::Config::refitBFieldAtMid;
    c.refit.radialFieldCorr = mkfit::Config::refitRadialFieldCorr;
    c.refit.elossSignFromPass = mkfit::Config::refitElossSignFromPass;
    c.refit.bkwMsFixedMomentum = mkfit::Config::refitBkwMsFixedMomentum;
    c.refit.bkwSubSteps = mkfit::Config::refitBkwSubSteps;
    c.refit.materialPerModule = mkfit::Config::refitMaterialPerModule;
    c.mag_c1 = mkfit::Config::mag_c1;
    c.mag_b0 = mkfit::Config::mag_b0;
    c.mag_b1 = mkfit::Config::mag_b1;
    c.mag_a = mkfit::Config::mag_a;

    c.iteration_index = ic.m_iteration_index;
    c.track_algorithm = ic.m_track_algorithm;
    c.requires_seed_hit_sorting = ic.m_requires_seed_hit_sorting;
    c.backward_search = ic.m_backward_search;
    c.backward_drop_seed_hits = ic.m_backward_drop_seed_hits;
    c.backward_fit_min_hits = ic.m_backward_fit_min_hits;
    c.backward_fit_outlier_chi2 = ic.m_backward_fit_outlier_chi2;
    c.backward_fit_max_outliers = ic.m_backward_fit_max_outliers;
    c.backward_fit_outlier_min_pt = ic.m_backward_fit_outlier_min_pt;
    c.backward_search_min_pixel_layers = ic.m_backward_search_min_pixel_layers;
    c.backward_search_prompt_max_d0 = ic.m_backward_search_prompt_max_d0;
    for (int w = 0; w < 4; ++w)
      c.pixel_layer_mask[w] = 0;
    if (ti.n_layers() > 256)
      throw cms::Exception("MkFitAlpakaES") << "pixel layer mask: " << ti.n_layers() << " layers > 256";
    for (int l = 0; l < ti.n_layers(); ++l)
      if (ti.layer(l).is_pixel())
        c.pixel_layer_mask[l >> 6] |= uint64_t(1) << (l & 63);
    c.sc_ptthr_hpt = ic.sc_ptthr_hpt;
    c.sc_drmax_bh = ic.sc_drmax_bh;
    c.sc_dzmax_bh = ic.sc_dzmax_bh;
    c.sc_drmax_eh = ic.sc_drmax_eh;
    c.sc_dzmax_eh = ic.sc_dzmax_eh;
    c.sc_drmax_bl = ic.sc_drmax_bl;
    c.sc_dzmax_bl = ic.sc_dzmax_bl;
    c.sc_drmax_el = ic.sc_drmax_el;
    c.sc_dzmax_el = ic.sc_dzmax_el;
    c.dc_fracSharedHits = ic.dc_fracSharedHits;
    c.dc_drth_central = ic.dc_drth_central;
    c.dc_drth_obarrel = ic.dc_drth_obarrel;
    c.dc_drth_forward = ic.dc_drth_forward;
    c.params = convert(ic.m_params);
    c.backward_params = convert(ic.m_backward_params);

    const int nRegions = ic.m_steering_params.size();
    if (nRegions > es::kMaxRegions || ic.m_n_regions != nRegions ||
        static_cast<int>(ic.m_region_order.size()) != nRegions)
      throw cms::Exception("MkFitAlpakaES")
          << "regions: m_n_regions " << ic.m_n_regions << ", steering params " << nRegions << ", region order "
          << ic.m_region_order.size() << ", capacity " << es::kMaxRegions;
    c.n_regions = nRegions;
    for (int r = 0; r < es::kMaxRegions; ++r)
      c.region_order[r] = r < nRegions ? ic.m_region_order[r] : -1;

    c.seed_cleaner = seedCleanerFromName(ic.m_seed_cleaner_name);
    c.seed_partitioner = seedPartitionerFromName(ic.m_seed_partitioner_name);
    c.pre_bkfit_filter = candFilterFromName(ic.m_pre_bkfit_filter_name);
    c.post_bkfit_filter = candFilterFromName(ic.m_post_bkfit_filter_name);
    c.duplicate_cleaner = duplicateCleanerFromName(ic.m_duplicate_cleaner_name);
    c.default_track_scorer = trackScorerFromName(ic.m_default_track_scorer_name);

    for (int r = 0; r < es::kMaxRegions; ++r) {
      SteeringRegion& s = c.steering_params[r];
      for (int i = 0; i < es::kMaxPlanLayers; ++i)
        s.layer[i] = -1;
      if (r >= nRegions) {
        s.region = -1;
        s.bkw_search_pickup = -1;
        continue;
      }
      mkfit::SteeringParams const& sp = ic.m_steering_params[r];
      const int nPlan = sp.m_layer_plan.size();
      if (nPlan > es::kMaxPlanLayers)
        throw cms::Exception("MkFitAlpakaES")
            << "region " << r << " layer plan has " << nPlan << " entries, capacity " << es::kMaxPlanLayers;
      s.region = sp.m_region;
      s.n_plan = nPlan;
      s.fwd_search_pickup = sp.m_fwd_search_pickup;
      s.bkw_fit_last = sp.m_bkw_fit_last;
      s.bkw_search_pickup = sp.m_bkw_search_pickup;
      // as IterationConfig::setupStandardFunctionsFromNames
      s.track_scorer =
          sp.m_track_scorer_name.empty() ? c.default_track_scorer : trackScorerFromName(sp.m_track_scorer_name);
      for (int i = 0; i < nPlan; ++i)
        s.layer[i] = sp.m_layer_plan[i].m_layer;
    }

    return std::make_unique<ESDataHost>(std::move(layers),
                                        std::move(modules),
                                        std::move(shapes),
                                        std::move(detIdMap),
                                        std::move(material),
                                        config,
                                        config,
                                        sizes);
  }

}  // namespace mkfitdev
