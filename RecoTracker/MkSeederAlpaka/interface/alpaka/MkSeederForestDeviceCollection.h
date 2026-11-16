#ifndef RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederForestDeviceCollection_h
#define RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederForestDeviceCollection_h

// The seed classifier's trees on the device (the ES product of MkSeederForestESProducer, MkSeederForest.h)

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederForest.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  using MkSeederForestDeviceCollection = PortableCollection<::mkseeder::MkSeederForestSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
