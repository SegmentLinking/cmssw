#ifndef RecoTracker_MkFitAlpaka_src_alpaka_PropagationFlagsAdapter_h
#define RecoTracker_MkFitAlpaka_src_alpaka_PropagationFlagsAdapter_h

// ES -> device adapter for the propagation flags: the ES product carries mkfitdev::PropFlags inside
// ESConfig::prop_config; the MaterialView takes the place of MkFitCore's TrackerInfo back-pointer.

#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/es/MaterialView.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationFlags.h"

namespace mkfitdev::prop {

  enum class PropStage : int { FindingInterLayer, FinalFit };

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PropagationFlags makePropagationFlags(::mkfitdev::PropFlags const& flags,
                                                                            ::mkfitdev::MaterialView const& mv) {
    PropagationFlags pf;
    pf.material = mv;
    pf.use_param_b_field = flags.use_param_b_field;
    pf.apply_material = flags.apply_material;
    pf.copy_input_state_on_fail = flags.copy_input_state_on_fail;
    pf.use_fit_b_field = flags.use_fit_b_field;
    return pf;
  }

  // es.config must point into the same memory space as the caller (device ESView on device, host ESView on host).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PropagationFlags propagationFlags(::mkfitdev::ESView const& es, PropStage stage) {
    const auto& pc = es.config->prop_config;
    return makePropagationFlags(stage == PropStage::FinalFit ? pc.final_fit_pflags : pc.finding_inter_layer_pflags,
                                es.material);
  }

}  // namespace mkfitdev::prop

#endif
