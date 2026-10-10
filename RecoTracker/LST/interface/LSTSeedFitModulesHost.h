#ifndef RecoTracker_LST_interface_LSTSeedFitModulesHost_h
#define RecoTracker_LST_interface_LSTSeedFitModulesHost_h

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace lst {

  // the tracker modules for the device fit of the pixel seeds, one row per GeomDetUnit::index(): the module surface
  // (position and rotation) and its material (radiation length fraction and energy-loss xi, as PropagatorWithMaterial)
  GENERATE_SOA_LAYOUT(LSTSeedFitModulesLayout,
                      SOA_COLUMN(SOAFrame<float>, frame),
                      SOA_COLUMN(float, radLen),
                      SOA_COLUMN(float, xi))

  using LSTSeedFitModulesSoA = LSTSeedFitModulesLayout<>;
  using LSTSeedFitModulesConstView = LSTSeedFitModulesSoA::ConstView;
  using LSTSeedFitModulesHost = PortableHostCollection<LSTSeedFitModulesSoA>;

}  // namespace lst

#endif
