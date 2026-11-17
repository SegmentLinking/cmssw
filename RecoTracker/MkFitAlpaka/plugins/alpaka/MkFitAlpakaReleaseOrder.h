#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaReleaseOrder_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaReleaseOrder_h

// A device product deleted early (canDeleteEarly) returns its block to the caching allocator with a marker recorded on
// the queue it was allocated on: the block is reused at once on that queue, and on any other queue once the marker is
// reached. A reader running on another queue makes the allocation queue wait (on the device) for its work so far, so
// the block cannot be handed out while the reader's kernels still use it.

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // allocationQueue: native handle of the allocation queue (the producer's "queue" product, 0 on CPU backends).
  // Returns true if the order was added, false if this reader runs on the allocation queue or on a blocking queue.
  inline bool orderReleaseAfterReads(Queue& queue, unsigned long long allocationQueue, char const* category) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    if (allocationQueue == reinterpret_cast<unsigned long long>(alpaka::getNativeHandle(queue)))
      return false;
    alpaka::Event<Queue> done(alpaka::getDev(queue));
    alpaka::enqueue(queue, done);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    const auto rc =
        cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(allocationQueue), alpaka::getNativeHandle(done), 0);
    if (rc != cudaSuccess)
      throw cms::Exception(category) << "cudaStreamWaitEvent: " << cudaGetErrorString(rc);
#else
    const auto rc =
        hipStreamWaitEvent(reinterpret_cast<hipStream_t>(allocationQueue), alpaka::getNativeHandle(done), 0);
    if (rc != hipSuccess)
      throw cms::Exception(category) << "hipStreamWaitEvent: " << hipGetErrorString(rc);
#endif
    return true;
#else
    (void)queue;
    (void)allocationQueue;
    (void)category;
    return false;
#endif
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
