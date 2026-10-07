#ifndef RecoTracker_MkFitCore_interface_PropagationConfig_h
#define RecoTracker_MkFitCore_interface_PropagationConfig_h

namespace mkfit {

  class TrackerInfo;

  enum PropagationFlagsEnum {
    PF_none = 0,
    PF_use_param_b_field = 0x1,
    PF_apply_material = 0x2,
    PF_copy_input_state_on_fail = 0x4
  };

  class PropagationFlags {
  public:
    const TrackerInfo *tracker_info = nullptr;  // back-pointer for easy passing into low-level funcs
    bool use_param_b_field : 1;
    bool apply_material : 1;
    bool copy_input_state_on_fail : 1;
    // Could add: bool use_trig_approx       -- now Config::useTrigApprox = true
    // Could add: int  n_prop_to_r_iters : 8 -- now Config::Niter = 5

    PropagationFlags() : use_param_b_field(false), apply_material(false), copy_input_state_on_fail(false) {}

    PropagationFlags(int pfe)
        : use_param_b_field(pfe & PF_use_param_b_field),
          apply_material(pfe & PF_apply_material),
          copy_input_state_on_fail(pfe & PF_copy_input_state_on_fail) {}
  };

  // Choices of the final fit, MkBuilder::fit_tracks(), read by its own sequences (FinalFit.cc).
  struct FinalFitFlags {
    // Sample B at the chord midpoint of each propagation to a plane instead of at its start, so that the
    // outward and inward propagations are inverses of each other.  With the parametrised field only.
    bool b_field_at_mid = true;
    // Correct each propagation to a plane for the radial field component Br = -(r/2) dBz/dz, which the
    // constant-Bz helix neglects, antisymmetrically: half of the change in r*p_phi at each end of the step.
    // With the parametrised field only.
    bool radial_field_corr = true;
  };

  class PropagationConfig {
  public:
    bool backward_fit_to_pca = false;
    bool finding_requires_propagation_to_hit_pos = false;
    PropagationFlags finding_inter_layer_pflags;
    PropagationFlags finding_intra_layer_pflags;
    PropagationFlags backward_fit_pflags;
    // The final fit, MkBuilder::fit_tracks(), both passes.
    PropagationFlags final_fit_pflags;
    FinalFitFlags final_fit_ffflags;
    PropagationFlags seed_fit_pflags;
    PropagationFlags pca_prop_pflags;

    void apply_tracker_info(const TrackerInfo *ti);
  };
}  // namespace mkfit

#endif
