#ifndef RecoTracker_MkFitAlpaka_interface_SupportedConfig_h
#define RecoTracker_MkFitAlpaka_interface_SupportedConfig_h

// The validated configuration envelope of the device chain. Host only.
// The port implements the LST step of the Phase-2 HLT (mkfit-phase2-lstStep.json + the HLT_75e33 module configs).
// Several MkFitCore code paths are not ported because that configuration never reaches them (doc/SIMPLIFICATIONS.txt),
// and several capacities are compile-time constants. Outside that envelope the device would run DIFFERENT code
// without any error, so every entry point checks it and throws cms::Exception("MkFitAlpakaUnsupportedConfig"):
//   - the ES producer: checkSupportedConfig(ESConfig, layers)                     (JSON, runtime mkfit::Config, geometry)
//   - the build producer: checkSupportedBuildConfig(ESConfig, BuildModuleConfig)  (+ MkFitProducer parameters)
//   - the fit producer: checkSupportedFitConfig(ESConfig)
// A new configuration is supported by porting the missing path AND widening the check, in the same commit.

#include <string>

#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"

namespace mkfitdev {

  // The MkFitProducer parameters (hltInitialStepTrackCandidatesMkFit) the device build depends on.
  // Defaults = the HLT_75e33 values.
  struct BuildModuleConfig {
    std::string clustersToSkip;  // InputTag label; the device has no hit mask (m_iteration_hit_mask)
    std::string buildingRoutine = "cloneEngine";
    bool seedCleaning = true;         // runs the IterationConfig seed cleaner if one is set (none ported)
    bool removeDuplicates = true;     // runs the IterationConfig duplicate cleaner if one is set
    bool backwardFitInCMSSW = false;  // the device runs mkFit's backward fit + backward search
  };

  // ES level. layers = the ES layer table (host view), for has_charge.
  void checkSupportedConfig(ESConfig const& c, LayerInfoSoA::ConstView layers, int nLayers);
  // Build module: the ES-level checks plus the module parameters.
  void checkSupportedBuildConfig(ESConfig const& c, BuildModuleConfig const& m);
  // Fit module (MkFitFitProducer path): the ES-level items the fit depends on.
  void checkSupportedFitConfig(ESConfig const& c);

}  // namespace mkfitdev

#endif
