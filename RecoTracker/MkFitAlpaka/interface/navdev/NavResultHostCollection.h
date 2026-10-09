#ifndef RecoTracker_MkFitAlpaka_interface_navdev_NavResultHostCollection_h
#define RecoTracker_MkFitAlpaka_interface_navdev_NavResultHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/navdev/NavResultSoA.h"

namespace mkfitdev {
  using NavResultHostCollection = PortableHostCollection<NavResultSoA>;
}  // namespace mkfitdev

#endif
