#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_Backend_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_Backend_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // CPU backends (serial, TBB): blocking queues, device memory is host memory
  constexpr bool kCpuBackend = std::is_same_v<Device, alpaka::DevCpu>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
