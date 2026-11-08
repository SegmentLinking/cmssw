#ifndef RecoTracker_LST_interface_LSTSeedFitModulesHost_h
#define RecoTracker_LST_interface_LSTSeedFitModulesHost_h

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace lst {

  // The tracker modules for the device fit of the pixel tracks, one row per GeomDetUnit::index().
  // The module surface (position and rotation), which takes the local hit errors to the global frame.
  GENERATE_SOA_LAYOUT(LSTSeedFitModulesLayout, SOA_COLUMN(SOAFrame<float>, frame))

  using LSTSeedFitModulesSoA = LSTSeedFitModulesLayout<>;
  using LSTSeedFitModulesConstView = LSTSeedFitModulesSoA::ConstView;
  using LSTSeedFitModulesHost = PortableHostCollection<LSTSeedFitModulesSoA>;

}  // namespace lst

#endif
