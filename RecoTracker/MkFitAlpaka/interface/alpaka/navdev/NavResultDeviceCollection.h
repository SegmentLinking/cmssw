#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_navdev_NavResultDeviceCollection_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_navdev_NavResultDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/navdev/NavResultHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/navdev/NavResultSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using ::mkfitdev::NavResultSoA;
  using ::mkfitdev::NavResultSoAConstView;
  using ::mkfitdev::NavResultSoAView;
  using NavResultDeviceCollection = PortableCollection<NavResultSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
