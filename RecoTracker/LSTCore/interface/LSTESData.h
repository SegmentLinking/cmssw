#ifndef RecoTracker_LSTCore_interface_LSTESData_h
#define RecoTracker_LSTCore_interface_LSTESData_h

#include "RecoTracker/LSTCore/interface/Constants.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometryDevSoA.h"
#include "RecoTracker/LSTCore/interface/Module.h"
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
    ModulesBuffer<TDev> modulesBuffers;
    std::unique_ptr<const PortableCollection<EndcapGeometryDevSoA, TDev>> endcapGeometry;
    std::shared_ptr<const PixelMap> pixelMapping;

    LSTESData(uint16_t const& nModulesIn,
              uint16_t const& nLowerModulesIn,
              unsigned int const& nPixelsIn,
              unsigned int const& nEndCapMapIn,
              ModulesBuffer<TDev> const& modulesBuffersIn,
              std::unique_ptr<const PortableCollection<EndcapGeometryDevSoA, TDev>> endcapGeometryIn,
              std::shared_ptr<const PixelMap> const& pixelMappingIn)
        : nModules(nModulesIn),
          nLowerModules(nLowerModulesIn),
          nPixels(nPixelsIn),
          nEndCapMap(nEndCapMapIn),
          modulesBuffers(modulesBuffersIn),
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
  struct CopyToDevice<lst::LSTESData<alpaka_common::DevHost>> {
    template <typename TQueue>
    static lst::LSTESData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue,
                                                         lst::LSTESData<alpaka_common::DevHost> const& srcData) {
      auto deviceModulesBuffers =
          lst::ModulesBuffer<alpaka::Dev<TQueue>>(alpaka::getDev(queue), srcData.nModules, srcData.nPixels);
      deviceModulesBuffers.copyFromSrc(queue, srcData.modulesBuffers);
      auto deviceEndcapGeometry = std::make_unique<PortableCollection<lst::EndcapGeometryDevSoA, alpaka::Dev<TQueue>>>(
          CopyToDevice<PortableHostCollection<lst::EndcapGeometryDevSoA>>::copyAsync(queue, *srcData.endcapGeometry));

      return lst::LSTESData<alpaka::Dev<TQueue>>(srcData.nModules,
                                                 srcData.nLowerModules,
                                                 srcData.nPixels,
                                                 srcData.nEndCapMap,
                                                 std::move(deviceModulesBuffers),
                                                 std::move(deviceEndcapGeometry),
                                                 srcData.pixelMapping);
    }
  };
}  // namespace cms::alpakatools

#endif
