#ifndef RecoTracker_MkFitAlpaka_interface_fit_CpeESData_h
#define RecoTracker_MkFitAlpaka_interface_fit_CpeESData_h

// ES product of the device PixelCPEGeneric of the mkFit final fit: the per-module constants, the
// flattened GenError templates and their float pool (interface/fit/CpeGeneric.h), one copy per memory space
// (LSTESData / mkfitdev::ESData style). Made per IOV by MkFitAlpakaFitCpeESProducer (PixelCPEFastParamsRecord);
// the framework copies it to each device with CopyToDevice below. The host tables (incl. rawId -> module index for
// the host cluster packing and the per-IOV cross-check against the menu's 'PixelCPEGeneric') are shared by all copies.

#include <cstdint>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeGeneric.h"

namespace mkfitdev::cpe {

  struct CpeTablesHost {
    std::vector<ModuleCpe> modules;  // PixelCPEFastParamsPhase2 detParams order
    std::vector<uint32_t> rawId;     // per module
    std::vector<GenErrTemplate> templ;
    std::vector<float> pool;
    std::unordered_map<uint32_t, int> rawToModule;

    CpeTables view() const {
      return CpeTables{modules.data(), int(modules.size()), templ.data(), int(templ.size()), pool.data()};
    }
  };

  template <typename TDev>
  struct CpeESData {
    using ModBuf = alpaka::Buf<TDev, ModuleCpe, alpaka_common::Dim1D, alpaka_common::Idx>;
    using TemplBuf = alpaka::Buf<TDev, GenErrTemplate, alpaka_common::Dim1D, alpaka_common::Idx>;
    using PoolBuf = alpaka::Buf<TDev, float, alpaka_common::Dim1D, alpaka_common::Idx>;

    ModBuf modules;
    TemplBuf templ;
    PoolBuf pool;
    std::shared_ptr<const CpeTablesHost> host;  // same object in every memory space

    CpeESData(ModBuf m, TemplBuf t, PoolBuf p, std::shared_ptr<const CpeTablesHost> h)
        : modules(std::move(m)), templ(std::move(t)), pool(std::move(p)), host(std::move(h)) {}

    // tables in this memory space; pass by value to kernels
    CpeTables view() const {
      return CpeTables{alpaka::getPtrNative(modules),
                       int(host->modules.size()),
                       alpaka::getPtrNative(templ),
                       int(host->templ.size()),
                       alpaka::getPtrNative(pool)};
    }
  };

  using CpeESDataHost = CpeESData<alpaka_common::DevHost>;

  // host product from host tables (one copy of the three arrays into host buffers, once per IOV)
  inline std::unique_ptr<CpeESDataHost> makeCpeESDataHost(std::shared_ptr<const CpeTablesHost> t) {
    using cms::alpakatools::make_host_buffer;
    auto copyIn = [](auto const& v) {
      using T = typename std::remove_cvref_t<decltype(v)>::value_type;
      auto b = make_host_buffer<T[]>(std::max<std::size_t>(v.size(), 1));
      std::copy(v.begin(), v.end(), b.data());
      return b;
    };
    return std::make_unique<CpeESDataHost>(copyIn(t->modules), copyIn(t->templ), copyIn(t->pool), std::move(t));
  }

}  // namespace mkfitdev::cpe

namespace cms::alpakatools {

  template <>
  struct CopyToDevice<mkfitdev::cpe::CpeESDataHost> {
    template <typename TQueue>
    static mkfitdev::cpe::CpeESData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue,
                                                                   mkfitdev::cpe::CpeESDataHost const& src) {
      using TDev = alpaka::Dev<TQueue>;
      if constexpr (std::is_same_v<TDev, alpaka_common::DevHost>) {
        return src;
      } else {
        auto copyBuf = [&queue](auto const& hostBuf) {
          using T = std::remove_const_t<std::remove_pointer_t<decltype(alpaka::getPtrNative(hostBuf))>>;
          const auto n = alpaka::getExtentProduct(hostBuf);
          auto d = make_device_buffer<T[]>(queue, n);
          alpaka::memcpy(queue, d, hostBuf);
          return d;
        };
        return mkfitdev::cpe::CpeESData<TDev>(copyBuf(src.modules), copyBuf(src.templ), copyBuf(src.pool), src.host);
      }
    }
  };

}  // namespace cms::alpakatools

#endif
