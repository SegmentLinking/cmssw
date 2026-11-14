#ifndef RecoTracker_MkFitAlpaka_interface_EventOfHitsProduct_h
#define RecoTracker_MkFitAlpaka_interface_EventOfHitsProduct_h

// Event product "device EventOfHits" (host side). One PortableCollection over a SoABlocks layout, so the framework
// copies it to the host automatically (CopyToHost of PortableCollection) and one token gives the whole binning.
// Each block view has exactly the type of the standalone layout view (HitSoA::View, LayerSoA::View, ...), so the
// hits kernels take product.view().hits() etc. unchanged. Block sizes: hits nHits, layers nLayers,
// binnedHits nHits, bins nBinsTotal (= layers.nBinsTotal()).

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"

namespace mkfitdev {
  GENERATE_SOA_BLOCKS(EventOfHitsBlocksLayout,
                      SOA_BLOCK(hits, HitSoALayout),
                      SOA_BLOCK(layers, LayerSoALayout),
                      SOA_BLOCK(binnedHits, BinnedHitSoALayout),
                      SOA_BLOCK(bins, BinSoALayout))

  using EventOfHitsBlocks = EventOfHitsBlocksLayout<>;
  using EventOfHitsHostCollection = PortableHostCollection<EventOfHitsBlocks>;

  // index of each block in the product's sizes array (declaration order of EventOfHitsBlocksLayout)
  constexpr std::size_t kHitsBlock = 0, kLayersBlock = 1, kBinnedHitsBlock = 2, kBinsBlock = 3, kNumBlocks = 4;
  using EventOfHitsSizes = std::array<int32_t, kNumBlocks>;

  // The device EventOfHits product of the CPU backends: the PortableHostCollection interface (view, const_view,
  // buffer), but the memory can be a block of the producer's pool. The buffer is then an exact-size alpaka::BufCpu into
  // the block whose deleter only drops a reference to it (and to the keeper, the pool entry), so the block stays
  // resident between events instead of being re-faulted every event (jemalloc purges large allocations on free). The
  // entry goes back to the pool only when the product is gone, so a block is never in two events. Every row a consumer
  // reads is rewritten every event.
  class EventOfHitsPooledHostCollection {
  public:
    using Layout = EventOfHitsBlocks;
    using View = Layout::View;
    using ConstView = Layout::ConstView;
    using Buffer = cms::alpakatools::host_buffer<std::byte[]>;
    using ConstBuffer = cms::alpakatools::const_host_buffer<std::byte[]>;
    using Sizes = EventOfHitsSizes;

    EventOfHitsPooledHostCollection() = delete;
    explicit EventOfHitsPooledHostCollection(edm::Uninitialized) noexcept {}

    // own allocation, as PortableHostCollection(queue, sizes)
    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    EventOfHitsPooledHostCollection(TQueue const& queue, Sizes const& sizes)
        : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, Layout::computeDataSize(sizes))},
          layout_{buffer_->data(), sizes},
          view_{layout_} {
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    template <typename TQueue, std::integral... Ints>
      requires(alpaka::isQueue<TQueue> && sizeof...(Ints) == kNumBlocks)
    explicit EventOfHitsPooledHostCollection(TQueue const& queue, const Ints... sizes)
        : EventOfHitsPooledHostCollection(queue, Sizes{{static_cast<int32_t>(sizes)...}}) {}

    // in a pool block of at least bytes(sizes) bytes; keeper: released with the last reference to the buffer
    EventOfHitsPooledHostCollection(Buffer const& block, Sizes const& sizes, std::shared_ptr<void> keeper = nullptr)
        : buffer_{inBlock(block, Layout::computeDataSize(sizes), std::move(keeper))},
          layout_{buffer_->data(), sizes},
          view_{layout_} {
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    static std::size_t bytes(Sizes const& sizes) { return Layout::computeDataSize(sizes); }

    EventOfHitsPooledHostCollection(EventOfHitsPooledHostCollection const&) = delete;
    EventOfHitsPooledHostCollection& operator=(EventOfHitsPooledHostCollection const&) = delete;
    EventOfHitsPooledHostCollection(EventOfHitsPooledHostCollection&&) = default;
    EventOfHitsPooledHostCollection& operator=(EventOfHitsPooledHostCollection&&) = default;
    ~EventOfHitsPooledHostCollection() = default;

    View& view() { return view_; }
    ConstView const& view() const { return view_; }
    ConstView const& const_view() const { return view_; }

    // exact size (= a PortableHostCollection of the same sizes)
    Buffer buffer() { return *buffer_; }
    ConstBuffer buffer() const { return *buffer_; }
    ConstBuffer const_buffer() const { return *buffer_; }

  private:
    static Buffer inBlock(Buffer block, std::size_t bytes, std::shared_ptr<void> keeper) {
      assert(bytes <= static_cast<std::size_t>(alpaka::getExtentProduct(block)));
      std::byte* begin = block.data();
      return Buffer(
          alpaka::getDev(block),
          begin,
          [keep = block, keeper = std::move(keeper)](std::byte*) {},  // the block is freed with its last reference
          alpaka_common::Vec1D{static_cast<alpaka_common::Idx>(bytes)});
    }

    std::optional<Buffer> buffer_;  //!
    Layout layout_;                 //!
    View view_;                     //!
  };
}  // namespace mkfitdev

#endif
