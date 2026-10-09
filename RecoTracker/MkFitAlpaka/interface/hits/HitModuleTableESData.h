#ifndef RecoTracker_MkFitAlpaka_interface_hits_HitModuleTableESData_h
#define RecoTracker_MkFitAlpaka_interface_hits_HitModuleTableESData_h

// ES product: the per-module table of the device hit input (rotation, position, mkFit layer,
// uniqueIdInLayer per GeomDet index; interface/hits/DeviceHitInput.h HitModuleDev), built once per IOV of
// TrackerRecoGeometryRecord by MkFitAlpakaEventOfHitsModuleTableESProducer; host product + one device copy per device
// (CopyToDevice below). Replaces a per-event 1.7 MB copy.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitInput.h"

namespace mkfitdev {

  template <typename TDev>
  struct HitModuleTableESData {
    using Buf = alpaka::Buf<TDev, HitModuleDev, alpaka_common::Dim1D, alpaka_common::Idx>;
    Buf modules;
    uint32_t nModules;

    HitModuleTableESData(Buf b, uint32_t n) : modules(std::move(b)), nModules(n) {}
    const HitModuleDev* data() const { return alpaka::getPtrNative(modules); }
  };

  using HitModuleTableESDataHost = HitModuleTableESData<alpaka_common::DevHost>;

  inline std::unique_ptr<HitModuleTableESDataHost> makeHitModuleTableESDataHost(std::vector<HitModuleDev> const& v) {
    auto b = cms::alpakatools::make_host_buffer<HitModuleDev[]>(std::max<std::size_t>(v.size(), 1));
    std::copy(v.begin(), v.end(), b.data());
    return std::make_unique<HitModuleTableESDataHost>(std::move(b), uint32_t(v.size()));
  }

}  // namespace mkfitdev

namespace cms::alpakatools {

  template <>
  struct CopyToDevice<mkfitdev::HitModuleTableESDataHost> {
    template <typename TQueue>
    static mkfitdev::HitModuleTableESData<alpaka::Dev<TQueue>> copyAsync(
        TQueue& queue, mkfitdev::HitModuleTableESDataHost const& src) {
      using TDev = alpaka::Dev<TQueue>;
      if constexpr (std::is_same_v<TDev, alpaka_common::DevHost>) {
        return src;
      } else {
        auto d = make_device_buffer<mkfitdev::HitModuleDev[]>(queue, alpaka::getExtentProduct(src.modules));
        alpaka::memcpy(queue, d, src.modules);
        return mkfitdev::HitModuleTableESData<TDev>(std::move(d), src.nModules);
      }
    }
  };

}  // namespace cms::alpakatools

#endif
