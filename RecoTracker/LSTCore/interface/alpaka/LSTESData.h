#ifndef RecoTracker_LSTCore_interface_alpaka_LSTESData_h
#define RecoTracker_LSTCore_interface_alpaka_LSTESData_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/EndcapGeometryBuffer.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#include "RecoTracker/LSTCore/interface/PixelMap.h"

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include <filesystem>
#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE::SDL {

  template <typename TDev>
  struct LSTESData {
    uint16_t nModules;
    uint16_t nLowerModules;
    unsigned int nPixels;
    unsigned int nEndCapMap;
    std::shared_ptr<const modulesBuffer<TDev>> modulesBuffers;
    std::shared_ptr<const EndcapGeometryBuffer<TDev>> endcapGeometryBuffers;
    std::shared_ptr<const ::SDL::pixelMap> pixelMapping;

    LSTESData(uint16_t const& nModulesIn,
              uint16_t const& nLowerModulesIn,
              unsigned int const& nPixelsIn,
              unsigned int const& nEndCapMapIn,
              std::shared_ptr<const modulesBuffer<TDev>> const& modulesBuffersIn,
              std::shared_ptr<const EndcapGeometryBuffer<TDev>> const& endcapGeometryBuffersIn,
              std::shared_ptr<const ::SDL::pixelMap> const& pixelMappingIn)
        : nModules(nModulesIn),
          nLowerModules(nLowerModulesIn),
          nPixels(nPixelsIn),
          nEndCapMap(nEndCapMapIn),
          modulesBuffers(modulesBuffersIn),
          endcapGeometryBuffers(endcapGeometryBuffersIn),
          pixelMapping(pixelMappingIn) {}
  };

  std::unique_ptr<LSTESData<alpaka_common::DevHost>> loadAndFillESHost();

}  // namespace SDL

namespace cms::alpakatools {
  template <>
  struct CopyToDevice<ALPAKA_ACCELERATOR_NAMESPACE::SDL::LSTESData<alpaka_common::DevHost>> {
    template <typename TQueue>
    static ALPAKA_ACCELERATOR_NAMESPACE::SDL::LSTESData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue,
                                                         ALPAKA_ACCELERATOR_NAMESPACE::SDL::LSTESData<alpaka_common::DevHost> const& srcData) {
      auto deviceModulesBuffers = std::make_shared<ALPAKA_ACCELERATOR_NAMESPACE::SDL::modulesBuffer<alpaka::Dev<TQueue>>>(
          alpaka::getDev(queue), srcData.nModules, srcData.nPixels);
      deviceModulesBuffers->copyFromSrc(queue, *srcData.modulesBuffers);
      auto deviceEndcapGeometryBuffers =
          std::make_shared<ALPAKA_ACCELERATOR_NAMESPACE::SDL::EndcapGeometryBuffer<alpaka::Dev<TQueue>>>(alpaka::getDev(queue), srcData.nEndCapMap);
      deviceEndcapGeometryBuffers->copyFromSrc(queue, *srcData.endcapGeometryBuffers);

      return ALPAKA_ACCELERATOR_NAMESPACE::SDL::LSTESData<alpaka::Dev<TQueue>>(srcData.nModules,
                                                 srcData.nLowerModules,
                                                 srcData.nPixels,
                                                 srcData.nEndCapMap,
                                                 deviceModulesBuffers,
                                                 deviceEndcapGeometryBuffers,
                                                 srcData.pixelMapping);
    }
  };
}  // namespace cms::alpakatools

#endif
