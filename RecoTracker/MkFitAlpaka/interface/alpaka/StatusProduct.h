#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_StatusProduct_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_StatusProduct_h

// Event product "status" (device side): MkFitStatusDeviceObject; see interface/StatusProduct.h.

#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/StatusProduct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using namespace ::mkfitdev;

  using MkFitStatusDeviceObject = PortableObject<::mkfitdev::MkFitStatus>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(mkfitdev::MkFitStatusDeviceObject, ::mkfitdev::MkFitStatusHostObject);

#endif
