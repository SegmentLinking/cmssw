#ifndef LSTESData_H
#define LSTESData_H

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/EndcapGeometryBuffer.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include <filesystem>
#include <memory>

namespace SDL {

  struct pixelMap;

  template <typename TDev>
  struct LSTESData {
    uint16_t nModules;
    uint16_t nLowerModules;
    unsigned int nPixels;
    unsigned int nEndCapMap;
    std::shared_ptr<const modulesBuffer<TDev>> modulesBuffers;
    std::shared_ptr<const endcapGeometryBuffer<TDev>> endcapGeometryBuffers;
    std::shared_ptr<const pixelMap> pixelMapping;

    LSTESData(uint16_t const& nModulesIn,
              uint16_t const& nLowerModulesIn,
              unsigned int const& nPixelsIn,
              unsigned int const& nEndCapMapIn,
              std::shared_ptr<const modulesBuffer<TDev>> const& modulesBuffersIn,
              std::shared_ptr<const endcapGeometryBuffer<TDev>> const& endcapGeometryBuffersIn,
              std::shared_ptr<const pixelMap> const& pixelMappingIn)
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
  struct CopyToDevice<SDL::LSTESData<alpaka_common::DevHost>> {
    template <typename TQueue>
    static SDL::LSTESData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue,
                                                         SDL::LSTESData<alpaka_common::DevHost> const& srcData) {
      auto deviceModulesBuffers = std::make_shared<SDL::modulesBuffer<alpaka::Dev<TQueue>>>(
          alpaka::getDev(queue), srcData.nModules, srcData.nPixels);
      deviceModulesBuffers->copyFromSrc(queue, *srcData.modulesBuffers);
      auto deviceEndcapGeometryBuffers =
          std::make_shared<SDL::endcapGeometryBuffer<alpaka::Dev<TQueue>>>(alpaka::getDev(queue), srcData.nEndCapMap);
      deviceEndcapGeometryBuffers->copyFromSrc(queue, *srcData.endcapGeometryBuffers);

      return SDL::LSTESData<alpaka::Dev<TQueue>>(srcData.nModules,
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
