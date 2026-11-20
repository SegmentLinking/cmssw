#ifndef RecoLocalTracker_SiPixelRecHits_interface_alpaka_PixelGenErrorTablesDevice_h
#define RecoLocalTracker_SiPixelRecHits_interface_alpaka_PixelGenErrorTablesDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelGenErrorTablesHost.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelGenErrorTablesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::PixelGenErrorTablesHost;
  using PixelGenErrorTablesDevice = PortableCollection<PixelGenErrorTablesSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(PixelGenErrorTablesDevice, PixelGenErrorTablesHost);

#endif  // RecoLocalTracker_SiPixelRecHits_interface_alpaka_PixelGenErrorTablesDevice_h
