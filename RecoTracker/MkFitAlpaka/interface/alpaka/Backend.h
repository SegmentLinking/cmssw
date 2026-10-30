#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_Backend_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_Backend_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // CPU backends (serial, TBB): device memory is host memory
  constexpr bool kCpuBackend = std::is_same_v<Device, alpaka::DevCpu>;
  // Blocking queue (serial; the TBB queue is non-blocking): the work is done when produce() returns
  constexpr bool kBlockingQueue = std::is_same_v<Queue, alpaka::QueueCpuBlocking>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
