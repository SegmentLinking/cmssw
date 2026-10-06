// Component "clean": the only translation unit that instantiates the duplicate-cleaner / filter kernels
// (src/alpaka/clean/CleanKernels.h). Exported entry point: CleanAlgo (src/alpaka/clean/CleanAlgo.h).
#include <stdexcept>
#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/clean/CleanAlgo.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/clean/CleanKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::clean {

  namespace detail {
    constexpr int kThreads = 256;      // threads per block on GPUs (elements per thread on CPUs)
    constexpr int kScanThreads = 256;  // DESIGN: blocks <= 256 threads
    constexpr int kPairBlocks = 1024;  // grid-stride over the pair list, whose length is known only on device

    inline WorkDiv1D perTrack(int n) {
      return cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(n > 0 ? n : 1, kThreads), kThreads);
    }
    inline WorkDiv1D oneBlock() {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
      return cms::alpakatools::make_workdiv<Acc1D>(1, 1);
#else
      return cms::alpakatools::make_workdiv<Acc1D>(1, kScanThreads);
#endif
    }
  }  // namespace detail

  CleanAlgo::CleanAlgo(Queue& queue, int capacity)
      : capacity_(capacity),
        ct_{cms::alpakatools::make_device_buffer<float[]>(queue, capacity)},
        hasPix_{cms::alpakatools::make_device_buffer<int8_t[]>(queue, capacity)},
        cell_{cms::alpakatools::make_device_buffer<int[]>(queue, capacity)},
        flags_{cms::alpakatools::make_device_buffer<int[]>(queue, capacity)},
        keep_{cms::alpakatools::make_device_buffer<int[]>(queue, capacity)},
        dest_{cms::alpakatools::make_device_buffer<int[]>(queue, capacity + 1)},
        pairCount_{cms::alpakatools::make_device_buffer<int[]>(queue, capacity)},
        pairOffset_{cms::alpakatools::make_device_buffer<int[]>(queue, capacity + 1)},
        wild_{cms::alpakatools::make_device_buffer<int[]>(queue, capacity)},
        cellContent_{cms::alpakatools::make_device_buffer<int[]>(queue, capacity)},
        cellCount_{cms::alpakatools::make_device_buffer<int[]>(queue, kNCells)},
        cellStart_{cms::alpakatools::make_device_buffer<int[]>(queue, kNCells + 1)},
        cellCursor_{cms::alpakatools::make_device_buffer<int[]>(queue, kNCells)},
        counters_{cms::alpakatools::make_device_buffer<int[]>(queue, kNCounters)} {}

  void CleanAlgo::checkRows(int rows) const {
    if (rows > capacity_)
      throw std::runtime_error("mkfitdev::clean::CleanAlgo: TrackSoA with " + std::to_string(rows) +
                               " rows exceeds the cleaner capacity " + std::to_string(capacity_));
  }

  void CleanAlgo::flagDuplicates(Queue& queue, ::mkfitdev::TrackSoAView tracks, DupCleanParams const& p) {
    checkRows(tracks.metadata().size());
    alpaka::memset(queue, cellCount_, 0);
    alpaka::memset(queue, counters_, 0);
    const auto wdTrk = detail::perTrack(capacity_);
    alpaka::exec<Acc1D>(queue,
                        wdTrk,
                        KernelPrepare{},
                        tracks,
                        p,
                        ct_.data(),
                        hasPix_.data(),
                        cell_.data(),
                        flags_.data(),
                        cellCount_.data(),
                        wild_.data(),
                        counters_.data());
    alpaka::exec<Acc1D>(queue,
                        detail::oneBlock(),
                        KernelScanOneBlock{},
                        cellCount_.data(),
                        cellStart_.data(),
                        (const int32_t*)nullptr,
                        kNCells);
    alpaka::memset(queue, cellCursor_, 0);
    alpaka::exec<Acc1D>(queue,
                        wdTrk,
                        KernelFill{},
                        tracks,
                        cell_.data(),
                        cellStart_.data(),
                        cellCursor_.data(),
                        cellContent_.data(),
                        counters_.data());
    alpaka::exec<Acc1D>(queue, wdTrk, KernelCountPairs{}, tracks, cell_.data(), cellStart_.data(), pairCount_.data());
    alpaka::exec<Acc1D>(queue,
                        detail::oneBlock(),
                        KernelScanOneBlock{},
                        pairCount_.data(),
                        pairOffset_.data(),
                        tracks.metadata().addressOf_nTracks(),
                        0);
    const auto wdPairs = cms::alpakatools::make_workdiv<Acc1D>(detail::kPairBlocks, detail::kThreads);
    alpaka::exec<Acc1D>(queue,
                        wdPairs,
                        KernelPairs{},
                        tracks,
                        ct_.data(),
                        hasPix_.data(),
                        cell_.data(),
                        cellStart_.data(),
                        cellContent_.data(),
                        pairOffset_.data(),
                        p,
                        flags_.data(),
                        counters_.data());
    alpaka::exec<Acc1D>(queue,
                        wdPairs,
                        KernelWildcardPairs{},
                        tracks,
                        ct_.data(),
                        hasPix_.data(),
                        wild_.data(),
                        p,
                        flags_.data(),
                        counters_.data());
    alpaka::exec<Acc1D>(queue, wdTrk, KernelWriteDupFlags{}, tracks, flags_.data(), keep_.data());
  }

  void CleanAlgo::compact(Queue& queue, ::mkfitdev::TrackSoAConstView in, ::mkfitdev::TrackSoAView out) {
    checkRows(in.metadata().size());
    if (out.metadata().size() < in.metadata().size())
      throw std::runtime_error("mkfitdev::clean::CleanAlgo: output TrackSoA smaller than the input");
    alpaka::exec<Acc1D>(queue,
                        detail::oneBlock(),
                        KernelScanOneBlock{},
                        keep_.data(),
                        dest_.data(),
                        in.metadata().addressOf_nTracks(),
                        0);
    alpaka::exec<Acc1D>(queue, detail::perTrack(capacity_), KernelCompact{}, in, out, keep_.data(), dest_.data());
  }

  void CleanAlgo::removeDuplicates(Queue& queue, ::mkfitdev::TrackSoAConstView in, ::mkfitdev::TrackSoAView out) {
    compact(queue, in, out);
  }

  void CleanAlgo::filterTracks(Queue& queue,
                               ::mkfitdev::TrackSoAConstView in,
                               ::mkfitdev::TrackSoAView out,
                               int minHitsQF) {
    alpaka::exec<Acc1D>(queue, detail::perTrack(capacity_), KernelFilterTracks{}, in, minHitsQF, keep_.data());
    compact(queue, in, out);
  }

  void CleanAlgo::seedCompactionMap(
      Queue& queue, const int* pass, const int32_t* nSeedsPtr, int* dest, const int* oldSep, int* newSep, int nRegions) {
    alpaka::exec<Acc1D>(queue, detail::oneBlock(), KernelScanOneBlock{}, pass, dest, nSeedsPtr, 0);
    alpaka::exec<Acc1D>(
        queue, detail::perTrack(nRegions), KernelNewSeparators{}, (const int*)dest, oldSep, newSep, nRegions);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::clean
