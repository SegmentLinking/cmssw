#ifndef RecoTracker_MkFitAlpaka_interface_seeds_alpaka_SeedsPackDevice_h
#define RecoTracker_MkFitAlpaka_interface_seeds_alpaka_SeedsPackDevice_h

// HOST-ONLY (plugins; never from a .dev.cc): MkFitCore seeds (MkFitSeedWrapper::seeds()) -> device seed table, for
// seeds::importSeeds. CPU backends pack straight into the "device" collection (host memory, no staging and no copy);
// GPU backends pack into a pinned host collection and enqueue one copy (the staging buffer is kept alive by the
// queue-ordered caching allocator until the copy completes).
// Last-hit positions are not packed: importSeeds reads them from the device EventOfHits (DeviceHitPositions).

#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedsHostCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedsHostPack.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/SeedsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::seeds {

  inline SeedsDeviceCollection packSeedsToDevice(Queue& queue, mkfit::TrackVec const& in) {
    const int n = in.size();
    SeedsDeviceCollection d(queue, n);
    auto noPos = [](int, int, float*) {};
    if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
      ::mkfitdev::packSeeds(in, noPos, d.view());
    } else {
      ::mkfitdev::SeedsHostCollection h(queue, n);
      ::mkfitdev::packSeeds(in, noPos, h.view());
      alpaka::memcpy(queue, d.buffer(), h.buffer());
    }
    return d;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::seeds

#endif
