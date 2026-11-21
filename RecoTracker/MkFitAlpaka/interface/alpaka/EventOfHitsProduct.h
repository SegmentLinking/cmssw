#ifndef RecoTracker_MkFitAlpaka_interface_alpaka_EventOfHitsProduct_h
#define RecoTracker_MkFitAlpaka_interface_alpaka_EventOfHitsProduct_h

// Event product "device EventOfHits" (device side); see interface/EventOfHitsProduct.h.
// CPU backends: the pooled host collection (same interface, memory from the producer's pool); GPU backends: the
// four blocks in two allocations of the caching allocator (same interface, the host copy is one EventOfHitsBlocks).

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/EventOfHitsProduct.h"

namespace mkfitdev {
  // GPU backends: the binned hits get their own allocation, the hits, layers and bins blocks share one: the caching
  // allocator rounds each allocation up to a power of two, and one allocation of all four blocks doubles for the
  // largest events.
  template <typename TDev>
  class EventOfHitsSplitDeviceCollection {
  public:
    using Layout = EventOfHitsBlocks;
    using View = Layout::View;
    using ConstView = Layout::ConstView;
    using Sizes = EventOfHitsSizes;
    using Buffer = cms::alpakatools::device_buffer<TDev, std::byte[]>;

    EventOfHitsSplitDeviceCollection() = delete;
    explicit EventOfHitsSplitDeviceCollection(edm::Uninitialized) noexcept {}

    template <typename TQueue>
    EventOfHitsSplitDeviceCollection(TQueue const& queue, Sizes const& sizes)
        : sizes_{sizes},
          main_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, hitBytes() + layerBytes() + binBytes())},
          binned_{cms::alpakatools::make_device_buffer<std::byte[]>(queue, binnedBytes())} {
      HitSoA hits(main_->data(), sizes[kHitsBlock]);
      LayerSoA layers(main_->data() + hitBytes(), sizes[kLayersBlock]);
      BinnedHitSoA binned(binned_->data(), sizes[kBinnedHitsBlock]);
      BinSoA bins(main_->data() + hitBytes() + layerBytes(), sizes[kBinsBlock]);
      view_ = View(HitSoA::View(hits), LayerSoA::View(layers), BinnedHitSoA::View(binned), BinSoA::View(bins));
    }

    template <typename TQueue, std::integral... Ints>
      requires(sizeof...(Ints) == kNumBlocks)
    explicit EventOfHitsSplitDeviceCollection(TQueue const& queue, const Ints... sizes)
        : EventOfHitsSplitDeviceCollection(queue, Sizes{{static_cast<int32_t>(sizes)...}}) {}

    EventOfHitsSplitDeviceCollection(EventOfHitsSplitDeviceCollection const&) = delete;
    EventOfHitsSplitDeviceCollection& operator=(EventOfHitsSplitDeviceCollection const&) = delete;
    EventOfHitsSplitDeviceCollection(EventOfHitsSplitDeviceCollection&&) = default;
    EventOfHitsSplitDeviceCollection& operator=(EventOfHitsSplitDeviceCollection&&) = default;
    ~EventOfHitsSplitDeviceCollection() = default;

    View& view() { return view_; }
    ConstView const& view() const { return view_; }
    ConstView const& const_view() const { return view_; }

    // into a host EventOfHitsBlocks collection of the same sizes (blocks in declaration order)
    template <typename TQueue>
    EventOfHitsHostCollection copyToHost(TQueue& queue) const {
      EventOfHitsHostCollection dst(queue, sizes_);
      std::byte* hostBegin = dst.buffer().data();
      std::byte const* mainBegin = main_->data();
      const auto device = alpaka::getDev(*main_);
      const std::size_t hitsAndLayersBytes = hitBytes() + layerBytes(), binnedHitsBytes = binnedBytes();
      using cms::alpakatools::make_device_view;
      using cms::alpakatools::make_host_view;
      alpaka::memcpy(queue,
                     make_host_view(hostBegin, hitsAndLayersBytes),
                     make_device_view(device, mainBegin, hitsAndLayersBytes));
      alpaka::memcpy(queue,
                     make_host_view(hostBegin + hitsAndLayersBytes, binnedHitsBytes),
                     make_device_view(device, binned_->data(), binnedHitsBytes));
      alpaka::memcpy(queue,
                     make_host_view(hostBegin + hitsAndLayersBytes + binnedHitsBytes, binBytes()),
                     make_device_view(device, mainBegin + hitsAndLayersBytes, binBytes()));
      return dst;
    }

  private:
    std::size_t hitBytes() const { return HitSoA::computeDataSize(sizes_[kHitsBlock]); }
    std::size_t layerBytes() const { return LayerSoA::computeDataSize(sizes_[kLayersBlock]); }
    std::size_t binnedBytes() const { return BinnedHitSoA::computeDataSize(sizes_[kBinnedHitsBlock]); }
    std::size_t binBytes() const { return BinSoA::computeDataSize(sizes_[kBinsBlock]); }

    Sizes sizes_;                   //!
    std::optional<Buffer> main_;    //!
    std::optional<Buffer> binned_;  //!
    View view_;                     //!
  };
}  // namespace mkfitdev

namespace cms::alpakatools {
  template <typename TDev>
  struct CopyToHost<::mkfitdev::EventOfHitsSplitDeviceCollection<TDev>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ::mkfitdev::EventOfHitsSplitDeviceCollection<TDev> const& src) {
      return src.copyToHost(queue);
    }
  };
}  // namespace cms::alpakatools

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  using EventOfHitsDeviceCollection = ::mkfitdev::EventOfHitsPooledHostCollection;
#else
  using EventOfHitsDeviceCollection = ::mkfitdev::EventOfHitsSplitDeviceCollection<Device>;
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
