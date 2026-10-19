#ifndef RecoTracker_LST_interface_alpaka_LSTSeedFitModulesCollection_h
#define RecoTracker_LST_interface_alpaka_LSTSeedFitModulesCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/LST/interface/LSTSeedFitModulesHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  using LSTSeedFitModulesCollection = PortableCollection<::lst::LSTSeedFitModulesSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
