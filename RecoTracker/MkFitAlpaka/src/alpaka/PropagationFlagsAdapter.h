#ifndef RecoTracker_MkFitAlpaka_src_alpaka_PropagationFlagsAdapter_h
#define RecoTracker_MkFitAlpaka_src_alpaka_PropagationFlagsAdapter_h

// ES -> device adapter for the propagation flags: the prop code takes mkfitdev::prop::PropagationFlags,
// the ES product carries mkfitdev::PropFlags inside ESConfig::prop_config plus the material map in ESView.
// MkFitCore: PropagationFlags are members of TrackerInfo::prop_config() (PropagationConfig.h) and carry a TrackerInfo
// back-pointer for material_checked(); here the MaterialView replaces the back-pointer.
// Use: const auto pf = mkfitdev::prop::propagationFlags(es, mkfitdev::prop::PropStage::FindingInterLayer);
// This is the ONLY place that turns ES flags into prop::PropagationFlags: engine (EnginePropConfig),
// select (K2), bkfit and the producers call it. The final fit keeps MkFitCore's hard-coded flags (MkBuilder.cc:1440).

#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/es/MaterialView.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationFlags.h"

namespace mkfitdev::prop {

  // the six PropagationFlags members of PropagationConfig
  enum class PropStage : int { FindingInterLayer, FindingIntraLayer, BackwardFit, ForwardFit, SeedFit, PcaProp };

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PropagationFlags makePropagationFlags(::mkfitdev::PropFlags const& f,
                                                                            ::mkfitdev::MaterialView const& mv) {
    PropagationFlags pf;
    pf.material = mv;
    pf.use_param_b_field = f.use_param_b_field;
    pf.apply_material = f.apply_material;
    pf.copy_input_state_on_fail = f.copy_input_state_on_fail;
    return pf;
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE ::mkfitdev::PropFlags const& propFlagsOf(::mkfitdev::PropConfig const& pc,
                                                                               PropStage s) {
    switch (s) {
      case PropStage::FindingInterLayer:
        return pc.finding_inter_layer_pflags;
      case PropStage::FindingIntraLayer:
        return pc.finding_intra_layer_pflags;
      case PropStage::BackwardFit:
        return pc.backward_fit_pflags;
      case PropStage::ForwardFit:
        return pc.forward_fit_pflags;
      case PropStage::SeedFit:
        return pc.seed_fit_pflags;
      default:
        return pc.pca_prop_pflags;
    }
  }

  // host steering: the host copy of the ES config + the material view of the target device (kernel arguments)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PropagationFlags propagationFlags(::mkfitdev::ESConfig const& c,
                                                                        ::mkfitdev::MaterialView const& mv,
                                                                        PropStage s) {
    return makePropagationFlags(propFlagsOf(c.prop_config, s), mv);
  }

  // es.config must point into the same memory space as the caller (device ESView on device, host ESView on host)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PropagationFlags propagationFlags(::mkfitdev::ESView const& es, PropStage s) {
    return makePropagationFlags(propFlagsOf(es.config->prop_config, s), es.material);
  }

}  // namespace mkfitdev::prop

#endif
