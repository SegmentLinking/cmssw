#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaReleaseOrder_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaReleaseOrder_h

// A device product deleted early (canDeleteEarly) returns its block to the caching allocator with a marker on the
// queue it was allocated on. Its last reader, if it runs on another queue, makes that queue wait (on the device) for
// its work, so the block cannot be handed out while the reader's kernels still use it.

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // native handle of the queue (0 on CPU backends: blocking queues, nothing to order)
  inline unsigned long long queueHandle(Queue& queue) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return reinterpret_cast<unsigned long long>(alpaka::getNativeHandle(queue));
#else
    (void)queue;
    return 0;
#endif
  }

  // called by the last reader after enqueueing its reads; allocationQueue = queueHandle() of the producer's queue
  inline void orderReleaseAfterReads(Queue& queue, unsigned long long allocationQueue) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    if (allocationQueue == queueHandle(queue))
      return;
    alpaka::Event<Queue> done(alpaka::getDev(queue));
    alpaka::enqueue(queue, done);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    if (cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(allocationQueue), alpaka::getNativeHandle(done), 0) !=
        cudaSuccess)
      throw cms::Exception("MkFitAlpakaReleaseOrder") << "cudaStreamWaitEvent failed";
#else
    if (hipStreamWaitEvent(reinterpret_cast<hipStream_t>(allocationQueue), alpaka::getNativeHandle(done), 0) !=
        hipSuccess)
      throw cms::Exception("MkFitAlpakaReleaseOrder") << "hipStreamWaitEvent failed";
#endif
#else
    (void)queue;
    (void)allocationQueue;
#endif
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
