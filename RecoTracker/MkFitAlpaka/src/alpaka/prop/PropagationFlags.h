#ifndef RecoTracker_MkFitAlpaka_src_alpaka_prop_PropagationFlags_h
#define RecoTracker_MkFitAlpaka_src_alpaka_prop_PropagationFlags_h

// Device mirror of mkfit::PropagationFlags (interface/PropagationConfig.h). The TrackerInfo back-pointer
// is replaced by the flat material map view.
// Phase-2 LST step values (MkFitGeometryESProducer.cc:625-635):
//   finding_inter_layer, finding_intra_layer, backward_fit, forward_fit: use_param_b_field | apply_material
//   seed_fit, pca_prop: none;  finding_requires_propagation_to_hit_pos = true;  backward_fit_to_pca = false.

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/es/MaterialView.h"

namespace mkfitdev::prop {

  enum PropagationFlagsEnum {
    PF_none = 0,
    PF_use_param_b_field = 0x1,
    PF_apply_material = 0x2,
    PF_copy_input_state_on_fail = 0x4,
    PF_b_field_at_mid = 0x8,
    PF_radial_field_corr = 0x10,
    PF_eloss_by_pass = 0x20,
    PF_eloss_outward = 0x40
  };

  struct PropagationFlags {
    ::mkfitdev::MaterialView material{nullptr, nullptr, 0, 0, 0.f, 0.f, {}};
    bool use_param_b_field = false;
    bool apply_material = false;
    bool copy_input_state_on_fail = false;
    // final fit only ( MkBuilder::fit_tracks sets them from mkfit::Config::refit*):
    // propagation to plane with use_param_b_field: B at the chord midpoint of the step / radial-field half-kicks
    bool b_field_at_mid = false;
    bool radial_field_corr = false;
    // energy-loss sign from the pass (eloss_outward: forward loses) instead of the path-length sign
    bool eloss_by_pass = false;
    bool eloss_outward = false;
    // PropagationFlags::ms_ref_p is passed to the device
    // propagation as an explicit MPlexQF argument (msRefP), not through the flags.

    PropagationFlags() = default;
    ALPAKA_FN_HOST_ACC PropagationFlags(int pfe, const ::mkfitdev::MaterialView& mv)
        : material(mv),
          use_param_b_field(pfe & PF_use_param_b_field),
          apply_material(pfe & PF_apply_material),
          copy_input_state_on_fail(pfe & PF_copy_input_state_on_fail),
          b_field_at_mid(pfe & PF_b_field_at_mid),
          radial_field_corr(pfe & PF_radial_field_corr),
          eloss_by_pass(pfe & PF_eloss_by_pass),
          eloss_outward(pfe & PF_eloss_outward) {}
  };

  struct Material {
    float bbxi{0}, radl{0};
  };

  // mkfit::TrackerInfo::material_checked(z, r)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Material material_checked(const ::mkfitdev::MaterialView& mv, float z, float r) {
    const int zbin = z * mv.facZ, rbin = r * mv.facR;
    Material m;
    if (zbin >= 0 && zbin < mv.nBinsZ && rbin >= 0 && rbin < mv.nBinsR) {
      m.bbxi = mv.bbxi[zbin * mv.nBinsR + rbin];
      m.radl = mv.radl[zbin * mv.nBinsR + rbin];
    }
    return m;
  }

}  // namespace mkfitdev::prop

#endif
