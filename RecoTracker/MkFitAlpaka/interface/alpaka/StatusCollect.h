#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_StatusCollect_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_StatusCollect_h

// Device-side filling of the per-event status product. Entry points defined in src/alpaka/Status.dev.cc.
// Use (build module, once per event, all asynchronous in the event queue):
//   MkFitStatusDeviceObject status(queue);
//   zeroStatus(queue, status);
//   StatusSources src;
//   src.add(seedsD.view().metadata().addressOf_nOverflowHits(), kSeedHitsTruncated);   // any device scalar counter
//   ...
//   collectStatus(queue, status, src);   // status.counter[index] += *address, for every source
//   iEvent.emplace(statusToken_, std::move(status));
// Kernels may also atomicAdd on status.data()->counter[i] directly (hierarchy::Blocks = device scope).

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusProduct.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // A device scalar counter (int32 or uint32) and the status slot it is added to.
  struct StatusSource {
    const uint32_t* u32 = nullptr;
    const int32_t* i32 = nullptr;
    int index = 0;
  };

  struct StatusSources {
    static constexpr int kMax = ::mkfitdev::kMaxStatusCounters;
    StatusSource sources[kMax];
    int nSources = 0;
    // returns false (and adds nothing) when full
    bool add(const uint32_t* counter, int index) { return push(StatusSource{counter, nullptr, index}); }
    bool add(const int32_t* counter, int index) { return push(StatusSource{nullptr, counter, index}); }

  private:
    bool push(StatusSource const& source) {
      if (nSources >= kMax || source.index < 0 || source.index >= ::mkfitdev::kMaxStatusCounters)
        return false;
      sources[nSources++] = source;
      return true;
    }
  };

  void zeroStatus(Queue& queue, MkFitStatusDeviceObject& status);
  void collectStatus(Queue& queue, MkFitStatusDeviceObject& status, StatusSources const& sources);
  // single-slot increment from the host side of the module (e.g. the event-skipped flag)
  void addStatus(Queue& queue, MkFitStatusDeviceObject& status, int index, uint32_t value);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
