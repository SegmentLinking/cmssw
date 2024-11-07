#ifndef RecoTracker_LSTCore_interface_LSTESData_h
#define RecoTracker_LSTCore_interface_LSTESData_h

#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometryDevHostCollection.h"
#include "RecoTracker/LSTCore/interface/ModulesHostCollection.h"
#include "RecoTracker/LSTCore/interface/PixelMap.h"

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include <filesystem>
#include <memory>

namespace lst {

  template <typename TDev>
  struct LSTESData {
    uint16_t nModules;
    uint16_t nLowerModules;
    unsigned int nPixels;
    unsigned int nEndCapMap;
    std::unique_ptr<const PortableMultiCollection<TDev, ModulesSoA, ModulesPixelSoA>> modules;
    std::unique_ptr<const PortableCollection<EndcapGeometryDevSoA, TDev>> endcapGeometry;
    // Host-side object that is shared between the LSTESData<TDev> objects for different devices
    std::shared_ptr<const PixelMap> pixelMapping;

    LSTESData(uint16_t const& nModulesIn,
              uint16_t const& nLowerModulesIn,
              unsigned int const& nPixelsIn,
              unsigned int const& nEndCapMapIn,
              std::unique_ptr<const PortableMultiCollection<TDev, ModulesSoA, ModulesPixelSoA>> modulesIn,
              std::unique_ptr<const PortableCollection<EndcapGeometryDevSoA, TDev>> endcapGeometryIn,
              std::shared_ptr<const PixelMap> const& pixelMappingIn)
        : nModules(nModulesIn),
          nLowerModules(nLowerModulesIn),
          nPixels(nPixelsIn),
          nEndCapMap(nEndCapMapIn),
          modules(std::move(modulesIn)),
          endcapGeometry(std::move(endcapGeometryIn)),
          pixelMapping(pixelMappingIn) {}
  };

  std::unique_ptr<LSTESData<alpaka_common::DevHost>> loadAndFillESHost();

}  // namespace lst

namespace cms::alpakatools {

  // The templated definition in CMSSW doesn't work when using CPU as the device
  template <>
  struct CopyToDevice<PortableHostCollection<lst::EndcapGeometryDevSoA>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableHostCollection<lst::EndcapGeometryDevSoA> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableCollection<lst::EndcapGeometryDevSoA, TDevice> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <>
  struct CopyToDevice<PortableHostMultiCollection<lst::ModulesSoA, lst::ModulesPixelSoA>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue,
                          PortableHostMultiCollection<lst::ModulesSoA, lst::ModulesPixelSoA> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableMultiCollection<TDevice, lst::ModulesSoA, lst::ModulesPixelSoA> dstData(srcData.sizes(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <>
  struct CopyToDevice<lst::LSTESData<alpaka_common::DevHost>> {
    template <typename TQueue>
    static lst::LSTESData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue,
                                                         lst::LSTESData<alpaka_common::DevHost> const& srcData) {
      auto deviceModules =
          std::make_unique<PortableMultiCollection<alpaka::Dev<TQueue>, lst::ModulesSoA, lst::ModulesPixelSoA>>(
              CopyToDevice<PortableHostMultiCollection<lst::ModulesSoA, lst::ModulesPixelSoA>>::copyAsync(
                  queue, *srcData.modules));
      auto deviceEndcapGeometry = std::make_unique<PortableCollection<lst::EndcapGeometryDevSoA, alpaka::Dev<TQueue>>>(
          CopyToDevice<PortableHostCollection<lst::EndcapGeometryDevSoA>>::copyAsync(queue, *srcData.endcapGeometry));

      return lst::LSTESData<alpaka::Dev<TQueue>>(srcData.nModules,
                                                 srcData.nLowerModules,
                                                 srcData.nPixels,
                                                 srcData.nEndCapMap,
                                                 std::move(deviceModules),
                                                 std::move(deviceEndcapGeometry),
                                                 srcData.pixelMapping);
    }
  };
}  // namespace cms::alpakatools

#endif
