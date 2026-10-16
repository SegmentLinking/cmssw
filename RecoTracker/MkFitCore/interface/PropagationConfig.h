#ifndef RecoTracker_MkFitCore_interface_PropagationConfig_h
#define RecoTracker_MkFitCore_interface_PropagationConfig_h

namespace mkfit {

  class TrackerInfo;

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

  class PropagationFlags {
  public:
    const TrackerInfo *tracker_info = nullptr;  // back-pointer for easy passing into low-level funcs
    // Optional per-lane |p| [GeV] at which the multiple-scattering noise (theta0 and beta) is evaluated in
    // propagation to plane, instead of at the propagated state's own momentum.  nullptr = running estimate.
    const float *ms_ref_p = nullptr;
    bool use_param_b_field : 1;
    bool apply_material : 1;
    bool copy_input_state_on_fail : 1;
    // Propagation to plane only, with use_param_b_field:
    // sample B at the chord midpoint of each step instead of at its start, so that the outward and
    // inward propagations are inverses of each other (see PropagationMPlexPlane.cc)
    bool b_field_at_mid : 1;
    // correct each step for the radial field component, antisymmetrically: half of the change in
    // r*p_phi at each end of the step (see PropagationMPlexPlane.cc)
    bool radial_field_corr : 1;
    // Sign of the energy loss in propagation to plane.  Unset: from the sign of the step's path length, i.e.
    // whether the step moves along the momentum.  Set: from the fit pass -- eloss_outward (forward pass)
    // loses energy on every step, otherwise gains it.  The particle crosses every module whatever order a
    // fit visits them in, so the pass gives the right sign; the path-length sign is wrong on every step the
    // fit takes backwards, e.g. between the two sensors of a PS module visited in reverse order.
    bool eloss_by_pass : 1;
    bool eloss_outward : 1;
    // Could add: bool use_trig_approx       -- now Config::useTrigApprox = true
    // Could add: int  n_prop_to_r_iters : 8 -- now Config::Niter = 5

    PropagationFlags()
        : use_param_b_field(false),
          apply_material(false),
          copy_input_state_on_fail(false),
          b_field_at_mid(false),
          radial_field_corr(false),
          eloss_by_pass(false),
          eloss_outward(false) {}

    PropagationFlags(int pfe)
        : use_param_b_field(pfe & PF_use_param_b_field),
          apply_material(pfe & PF_apply_material),
          copy_input_state_on_fail(pfe & PF_copy_input_state_on_fail),
          b_field_at_mid(pfe & PF_b_field_at_mid),
          radial_field_corr(pfe & PF_radial_field_corr),
          eloss_by_pass(pfe & PF_eloss_by_pass),
          eloss_outward(pfe & PF_eloss_outward) {}
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
