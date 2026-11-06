#ifndef RecoTracker_MkFitCore_interface_PropagationConfig_h
#define RecoTracker_MkFitCore_interface_PropagationConfig_h

#include "RecoTracker/MkFitCore/interface/portable/Macros.h"

namespace mkfit {

  class TrackerInfo;

  // Runtime settings and material map read by the propagation, from Config:: and TrackerInfo (propagation_env());
  // GPU code fills its own copy, with the material map in device memory.
  struct PropagationEnv {
    struct Material {
      float bbxi{0}, radl{0};
    };

    // Bz = (mag_b0 z^2 + mag_b1 z + mag_c1) (mag_a r^2 + 1), as Config::bFieldFromZR()
    float mag_c1 = 0.f, mag_b0 = 0.f, mag_b1 = 0.f, mag_a = 0.f;
    bool use_pt_mult_scat = false;  // Config::usePtMultScat
    // (|z|, r) material map of TrackerInfo, bin (iz, ir) at material[iz * mat_nbins_r + ir]
    const Material *material = nullptr;
    int mat_nbins_z = 0, mat_nbins_r = 0;
    float mat_fac_z = 0.f, mat_fac_r = 0.f;

    MKFIT_HOST_DEVICE float bFieldFromZR(const float z, const float r) const {
      return (mag_b0 * z * z + mag_b1 * z + mag_c1) * (mag_a * r * r + 1.f);
    }

    // as TrackerInfo::material_checked()
    MKFIT_HOST_DEVICE Material material_checked(float z, float r) const {
      const int zbin = z * mat_fac_z, rbin = r * mat_fac_r;
      return (zbin >= 0 && zbin < mat_nbins_z && rbin >= 0 && rbin < mat_nbins_r) ? material[zbin * mat_nbins_r + rbin]
                                                                                  : Material();
    }
  };

  enum PropagationFlagsEnum {
    PF_none = 0,
    PF_use_param_b_field = 0x1,
    PF_apply_material = 0x2,
    PF_copy_input_state_on_fail = 0x4
  };

  class PropagationFlags {
  public:
    bool use_param_b_field : 1;
    bool apply_material : 1;
    bool copy_input_state_on_fail : 1;
    // Could add: bool use_trig_approx       -- now Config::useTrigApprox = true
    // Could add: int  n_prop_to_r_iters : 8 -- now Config::Niter = 5
    PropagationEnv env;

    MKFIT_HOST_DEVICE PropagationFlags()
        : use_param_b_field(false), apply_material(false), copy_input_state_on_fail(false) {}

    MKFIT_HOST_DEVICE PropagationFlags(int pfe)
        : use_param_b_field(pfe & PF_use_param_b_field),
          apply_material(pfe & PF_apply_material),
          copy_input_state_on_fail(pfe & PF_copy_input_state_on_fail) {}
  };

  class PropagationConfig {
  public:
    bool backward_fit_to_pca = false;
    bool finding_requires_propagation_to_hit_pos = false;
    PropagationFlags finding_inter_layer_pflags;
    PropagationFlags finding_intra_layer_pflags;
    PropagationFlags backward_fit_pflags;
    PropagationFlags forward_fit_pflags;
    PropagationFlags seed_fit_pflags;
    PropagationFlags pca_prop_pflags;

    void apply_tracker_info(const TrackerInfo *ti);
  };
}  // namespace mkfit

#endif
