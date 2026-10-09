#ifndef RecoTracker_MkFitAlpaka_interface_es_ESLayouts_h
#define RecoTracker_MkFitAlpaka_interface_es_ESLayouts_h

// SoA layouts of the device-resident mkFit geometry (mkfit::TrackerInfo / LayerInfo / ModuleInfo /
// ModuleShape / material map) plus the per-layer part of mkfit::IterationConfig (IterationLayerConfig).

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkfitdev {

  // One row per mkFit layer (60 in Phase-2), row index = layer id.
  GENERATE_SOA_LAYOUT(LayerInfoLayout,
                      // LayerInfo members
                      SOA_COLUMN(int, layer_id),
                      SOA_COLUMN(int, layer_type),  // mkfitdev::LayerType / LayerInfo::LayerType_e
                      SOA_COLUMN(int, subdet),
                      SOA_COLUMN(float, rin),
                      SOA_COLUMN(float, rout),
                      SOA_COLUMN(float, zmin),
                      SOA_COLUMN(float, zmax),
                      SOA_COLUMN(float, propagate_to),
                      SOA_COLUMN(float, q_bin),
                      SOA_COLUMN(float, hole_r_min),
                      SOA_COLUMN(float, hole_r_max),
                      SOA_COLUMN(bool, has_r_range_hole),
                      SOA_COLUMN(bool, is_stereo),
                      SOA_COLUMN(bool, is_pixel),
                      SOA_COLUMN(bool, has_charge),
                      // modules of the layer: global module rows [module_begin, module_begin + n_modules),
                      // global row = module_begin + short id (LayerInfo::short_id)
                      SOA_COLUMN(int, module_begin),
                      SOA_COLUMN(int, n_modules),
                      // shapes of the layer: global shape rows [shape_begin, shape_begin + n_shapes),
                      // global row = shape_begin + ModuleInfo::shapeid
                      SOA_COLUMN(int, shape_begin),
                      SOA_COLUMN(int, n_shapes),
                      // q binning of LayerOfHits, as LayerOfHits::Initializator(const LayerInfo&) computes it
                      SOA_COLUMN(float, q_min),
                      SOA_COLUMN(float, q_max),
                      SOA_COLUMN(uint32_t, n_q),
                      // IterationLayerConfig of the iteration (m_winpars_* are empty for the LST step;
                      // fillESDataHost throws if they are not)
                      SOA_COLUMN(float, select_min_dphi),
                      SOA_COLUMN(float, select_max_dphi),
                      SOA_COLUMN(float, select_min_dq),
                      SOA_COLUMN(float, select_max_dq))

  // One row per module, ordered by (layer, short id).
  GENERATE_SOA_LAYOUT(ModuleInfoLayout,
                      SOA_COLUMN(float, pos_x),
                      SOA_COLUMN(float, pos_y),
                      SOA_COLUMN(float, pos_z),
                      SOA_COLUMN(float, zdir_x),  // normal to the module plane
                      SOA_COLUMN(float, zdir_y),
                      SOA_COLUMN(float, zdir_z),
                      SOA_COLUMN(float, xdir_x),  // precise / "phi" direction
                      SOA_COLUMN(float, xdir_y),
                      SOA_COLUMN(float, xdir_z),
                      SOA_COLUMN(uint32_t, detid),
                      SOA_COLUMN(uint16_t, shapeid),  // within the layer
                      SOA_COLUMN(int16_t, layer),
                      SOA_COLUMN(int32_t, sid),  // short id within the layer
                      // the module's own material (ModuleInfo::radl / bbxi from MediumProperties
                      // radLen and xi), read by the final fit with refitMaterialPerModule
                      SOA_COLUMN(float, radl),
                      SOA_COLUMN(float, bbxi))

  // One row per module shape, grouped by layer (LayerInfo::m_shapes).
  GENERATE_SOA_LAYOUT(
      ModuleShapeLayout, SOA_COLUMN(float, dx1), SOA_COLUMN(float, dx2), SOA_COLUMN(float, dy), SOA_COLUMN(float, dz))

  // detid -> global module row: open-addressing table (linear probing), capacity = power of two >= 2 x modules,
  // slot = detIdHashSlot(detid, hash_shift) (ESView.h), empty slot key = 0 (never a valid DetId). Filled in module order,
  // so the content is deterministic. max_probe = longest probe distance of any key (lookups stop after it).
  GENERATE_SOA_LAYOUT(DetIdMapLayout,
                      SOA_COLUMN(uint32_t, key),
                      SOA_COLUMN(int32_t, module),
                      SOA_SCALAR(uint32_t, hash_shift),
                      SOA_SCALAR(int32_t, max_probe),
                      SOA_SCALAR(int32_t, n_entries))

  // (z, r) material map of TrackerInfo: row = binZ * nbins_r + binR (rectvec order).
  GENERATE_SOA_LAYOUT(MaterialLayout,
                      SOA_COLUMN(float, bbxi),
                      SOA_COLUMN(float, radl),
                      SOA_SCALAR(int32_t, nbins_z),
                      SOA_SCALAR(int32_t, nbins_r),
                      SOA_SCALAR(float, range_z),
                      SOA_SCALAR(float, range_r),
                      SOA_SCALAR(float, fac_z),
                      SOA_SCALAR(float, fac_r))

  using LayerInfoSoA = LayerInfoLayout<>;
  using ModuleInfoSoA = ModuleInfoLayout<>;
  using ModuleShapeSoA = ModuleShapeLayout<>;
  using DetIdMapSoA = DetIdMapLayout<>;
  using MaterialSoA = MaterialLayout<>;

}  // namespace mkfitdev

#endif
