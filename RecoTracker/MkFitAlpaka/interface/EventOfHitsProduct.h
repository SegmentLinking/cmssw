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

  // the device EventOfHits product of the CPU backends. The PortableHostCollection
  // interface (view, const_view, buffer), but the memory can be a block of the producer's per-EDM-stream pool: the
  // buffer is then an exact-size alpaka::BufCpu into the block whose deleter only drops a reference to it, so the
  // ~31 MB stay resident between the events of a stream instead of being re-faulted every event (jemalloc purges
  // allocations >= 8 MiB on free). An EDM stream starts its next event only after this one's products are gone, so a
  // block is never in two events. Every row a consumer reads is rewritten every event.
  class EventOfHitsPooledHostCollection {
  public:
    using Layout = EventOfHitsBlocks;
    using View = Layout::View;
    using ConstView = Layout::ConstView;
    using Buffer = cms::alpakatools::host_buffer<std::byte[]>;
    using ConstBuffer = cms::alpakatools::const_host_buffer<std::byte[]>;
    static constexpr std::size_t kBlocks = 4;
    using Sizes = std::array<int32_t, kBlocks>;

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
      requires(alpaka::isQueue<TQueue> && sizeof...(Ints) == kBlocks)
    explicit EventOfHitsPooledHostCollection(TQueue const& queue, const Ints... sizes)
        : EventOfHitsPooledHostCollection(queue, Sizes{{static_cast<int32_t>(sizes)...}}) {}

    // in a pool block of at least bytes(sizes) bytes
    EventOfHitsPooledHostCollection(Buffer const& block, Sizes const& sizes)
        : buffer_{inBlock(block, Layout::computeDataSize(sizes))}, layout_{buffer_->data(), sizes}, view_{layout_} {
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
    static Buffer inBlock(Buffer block, std::size_t n) {
      assert(n <= static_cast<std::size_t>(alpaka::getExtentProduct(block)));
      std::byte* p = block.data();
      return Buffer(
          alpaka::getDev(block),
          p,
          [keep = block](std::byte*) {},  // the block is freed with its last reference
          alpaka_common::Vec1D{static_cast<alpaka_common::Idx>(n)});
    }

    std::optional<Buffer> buffer_;  //!
    Layout layout_;                 //!
    View view_;                     //!
  };
}  // namespace mkfitdev

#endif
