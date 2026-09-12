#ifndef RecoTracker_LSTCore_src_alpaka_MemoryProfiler_h
#define RecoTracker_LSTCore_src_alpaka_MemoryProfiler_h

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// ALPAKA_ACCELERATOR_NAMESPACE is a macro (alpaka_serial_sync, alpaka_cuda_async,
// ...); two levels are needed to stringify what it expands to.
#define LST_MEMPROFILE_STRINGIFY_(x) #x
#define LST_MEMPROFILE_STRINGIFY(x) LST_MEMPROFILE_STRINGIFY_(x)
#define LST_MEMPROFILE_BACKEND LST_MEMPROFILE_STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE)

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Per-event memory accounting for the LST collections, active only when the
  // memoryProfile flag is set. Tracks live bytes as a running quantity and keeps
  // a per-event high, separately for the device and host budgets.
  // The allocation sum is kept alongside it only so the two can be compared.
  class MemoryProfiler {
  public:
    enum class Domain : unsigned { Device = 0, Host = 1 };
    static constexpr unsigned kNDomains = 2;

    // Whether the CMSSW caching allocator is actually in the allocation path.
    static constexpr bool kAllocatorInPath = not std::is_same_v<Device, alpaka::DevCpu>;

    struct Entry {
      std::string_view name;
      Domain domain;
      std::size_t slots = 0;      // element count, 0 when not meaningful
      std::size_t requested = 0;  // bytes asked for
      std::size_t reserved = 0;   // bytes the process actually holds
    };

    // Allocated versus actually-used slots for one stage.
    struct Usage {
      std::string stage;
      std::size_t allocated = 0;  // slots the collection was sized for
      std::size_t used = 0;       // slots actually filled
    };

    struct Column {
      std::string collection;
      std::string block;  // empty for a flat (non-blocks) layout
      std::string name;
      std::string kind;
      std::size_t bytes = 0;
      std::size_t padding = 0;
    };

    // Mirror of CachingAllocator::findBin, which is private.
    static constexpr std::size_t reservedFor(std::size_t bytes, cms::alpakatools::AllocatorConfig const& cfg = {}) {
      if constexpr (not kAllocatorInPath) {
        return bytes;
      } else {
        std::size_t minBinBytes = ipow(cfg.binGrowth, cfg.minBin);
        std::size_t maxBinBytes = ipow(cfg.binGrowth, cfg.maxBin);
        if (bytes < minBinBytes) {
          return minBinBytes;
        }
        if (bytes > maxBinBytes) {
          return bytes;  // findBin would throw; report the raw size instead
        }
        std::size_t binBytes = minBinBytes;
        while (binBytes < bytes) {
          binBytes *= cfg.binGrowth;
        }
        return binBytes;
      }
    }

    // Enumerate one layout's columns.
    template <typename TLayoutPart>
    void recordColumns(std::string_view collection, std::string_view block, TLayoutPart const& part) {
      std::ostringstream os;
      part.soaToStreamInternal(os);
      std::istringstream in(os.str());
      std::string line;
      while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tag, name, w_at, w_offset, w_has, w_size, w_and, w_padding;
        std::size_t offset = 0, size = 0, padding = 0;
        ls >> tag;
        // Both kinds must be taken. Reading only "Column" leaves the scalars
        // unattributed, and the per-collection bytes then fail to reconcile
        // against the allocation by 128 bytes per scalar.
        if (tag != "Column" && tag != "Scalar") {
          continue;
        }
        ls >> name >> w_at >> w_offset >> offset >> w_has >> w_size >> size >> w_and >> w_padding >> padding;
        if (ls.fail()) {
          continue;
        }
        columns_.push_back(Column{std::string(collection), std::string(block), name, tag, size, padding});
      }
    }

    void recordUsage(std::string_view stage, std::size_t allocated, std::size_t used) {
      usage_.push_back(Usage{std::string(stage), allocated, used});
    }

    void recordAlloc(std::string_view name, Domain domain, std::size_t slots, std::size_t requested) {
      std::size_t reserved = reservedFor(requested);
      unsigned d = static_cast<unsigned>(domain);
      liveRequested_[d] += requested;
      liveReserved_[d] += reserved;
      sumRequested_[d] += requested;
      peakRequested_[d] = std::max(peakRequested_[d], liveRequested_[d]);
      peakReserved_[d] = std::max(peakReserved_[d], liveReserved_[d]);
      entries_.push_back(Entry{name, domain, slots, requested, reserved});
    }

    // All collections released together, as resetEventSync() does.
    void releaseAll() {
      liveRequested_.fill(0);
      liveReserved_.fill(0);
    }

    // Close the current event: the peaks stay readable until this is called.
    void endEvent() {
      releaseAll();
      peakRequested_.fill(0);
      peakReserved_.fill(0);
      sumRequested_.fill(0);
      entries_.clear();
      columns_.clear();
      usage_.clear();
    }

    std::size_t peakRequested(Domain d) const { return peakRequested_[static_cast<unsigned>(d)]; }
    std::size_t peakReserved(Domain d) const { return peakReserved_[static_cast<unsigned>(d)]; }
    std::size_t sumRequested(Domain d) const { return sumRequested_[static_cast<unsigned>(d)]; }
    std::size_t liveRequested(Domain d) const { return liveRequested_[static_cast<unsigned>(d)]; }
    std::vector<Entry> const& entries() const { return entries_; }
    std::vector<Column> const& columns() const { return columns_; }
    std::vector<Usage> const& usage() const { return usage_; }

    // What a record needs beyond the accounting itself.
    struct RecordMeta {
      std::uint64_t event = 0;
      unsigned int stream = 0;
      std::size_t nHits = 0;        // input scale, so records can be compared
      std::size_t nPixelSeeds = 0;  // across events of different difficulty
      bool reduceMemByFullPrecompute = false;
    };

    // One JSON-lines record. Shared by the standalone driver and the CMSSW
    // producer so the two paths cannot drift: a field added here appears in both.
    void writeRecord(std::ostream& out, RecordMeta const& meta) const {
      out << "{";
      out << "\"event\":" << meta.event;
      out << ",\"stream\":" << meta.stream;
      out << ",\"backend\":\"" << LST_MEMPROFILE_BACKEND << "\"";
      out << ",\"allocator_in_path\":" << (kAllocatorInPath ? "true" : "false");
      cms::alpakatools::AllocatorConfig cfg;
      out << ",\"allocator_bin_growth\":" << cfg.binGrowth;
      out << ",\"allocator_min_bin\":" << cfg.minBin;
      out << ",\"allocator_max_bin\":" << cfg.maxBin;
      out << ",\"reduce_mem_by_full_precompute\":" << (meta.reduceMemByFullPrecompute ? "true" : "false");
      out << ",\"memory_profile\":true";
      out << ",\"n_hits\":" << meta.nHits;
      out << ",\"n_pixel_seeds\":" << meta.nPixelSeeds;
      out << ",\"peak_device_requested\":" << peakRequested(Domain::Device);
      out << ",\"peak_device_reserved\":" << peakReserved(Domain::Device);
      out << ",\"peak_host_requested\":" << peakRequested(Domain::Host);
      out << ",\"peak_host_reserved\":" << peakReserved(Domain::Host);
      out << ",\"sum_device_requested\":" << sumRequested(Domain::Device);
      out << ",\"sum_host_requested\":" << sumRequested(Domain::Host);

      out << ",\"collections\":[";
      bool first = true;
      for (auto const& e : entries_) {
        if (!first)
          out << ",";
        first = false;
        out << "{\"name\":\"" << e.name << "\"";
        out << ",\"domain\":\"" << (e.domain == Domain::Device ? "device" : "host") << "\"";
        out << ",\"slots\":" << e.slots;
        out << ",\"requested\":" << e.requested;
        out << ",\"reserved\":" << e.reserved;
        out << "}";
      }
      out << "]";

      out << ",\"columns\":[";
      first = true;
      for (auto const& c : columns_) {
        if (!first)
          out << ",";
        first = false;
        out << "{\"collection\":\"" << c.collection << "\"";
        out << ",\"block\":\"" << c.block << "\"";
        out << ",\"name\":\"" << c.name << "\"";
        out << ",\"kind\":\"" << c.kind << "\"";
        out << ",\"bytes\":" << c.bytes;
        out << ",\"padding\":" << c.padding;
        out << "}";
      }
      out << "]";

      out << ",\"usage\":[";
      first = true;
      for (auto const& u : usage_) {
        if (!first)
          out << ",";
        first = false;
        out << "{\"stage\":\"" << u.stage << "\"";
        out << ",\"allocated\":" << u.allocated;
        out << ",\"used\":" << u.used;
        out << "}";
      }
      out << "]}" << std::endl;
    }

  private:
    static constexpr std::size_t ipow(unsigned base, unsigned exp) {
      std::size_t r = 1;
      for (unsigned i = 0; i < exp; ++i) {
        r *= base;
      }
      return r;
    }

    std::array<std::size_t, kNDomains> liveRequested_{};
    std::array<std::size_t, kNDomains> liveReserved_{};
    std::array<std::size_t, kNDomains> peakRequested_{};
    std::array<std::size_t, kNDomains> peakReserved_{};
    std::array<std::size_t, kNDomains> sumRequested_{};
    std::vector<Entry> entries_;
    std::vector<Column> columns_;
    std::vector<Usage> usage_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
