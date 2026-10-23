#ifndef RecoLocalTracker_SiPixelRecHits_test_alpaka_PixelCPETrackAnglesTestKernel_h
#define RecoLocalTracker_SiPixelRecHits_test_alpaka_PixelCPETrackAnglesTestKernel_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDeviceTrackAngles.h"

namespace pixelCPEforDevice::test {

  struct TrackAnglesCase {
    int module;
    GenericClusterParams cluster;
    float cotAlpha, cotBeta;
  };

  struct TrackAnglesResult {
    HitParameters hit;
    bool valid;
  };

}  // namespace pixelCPEforDevice::test

namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelCPEforDeviceTest {

  // hitParametersTrackAngles for every case
  void runTrackAngles(Queue& queue,
                      pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::Phase2> const* params,
                      PixelGenErrorTablesSoAConstView tables,
                      pixelCPEforDevice::test::TrackAnglesCase const* cases,
                      pixelCPEforDevice::test::TrackAnglesResult* results,
                      int32_t size);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelCPEforDeviceTest

#endif  // RecoLocalTracker_SiPixelRecHits_test_alpaka_PixelCPETrackAnglesTestKernel_h
