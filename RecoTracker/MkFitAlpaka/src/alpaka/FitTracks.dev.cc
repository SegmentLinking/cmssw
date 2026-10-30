// Device mkFit final fit: kernel instantiation and launch. See src/alpaka/fit/FitKernels.h.

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/MkFitAlpaka/src/alpaka/fit/FitKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/fit/FitTracks.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::fit {

  using namespace cms::alpakatools;

  void runFinalFit(Queue& queue,
                   ESView const& es,
                   ::mkfitdev::HitSoAConstView hits,
                   uint32_t nPixel,
                   ::mkfitdev::TrackSoAView tracks,
                   int nTracks,
                   FitCounters* counters,
                   bool useCpe,
                   ::mkfitdev::cpe::CpeTables cpe,
                   const ::mkfitdev::cpe::ClusterCpe* clusters,
                   const int32_t* nTracksDev) {
    if (nTracks <= 0)
      return;
    const int stride = nTracks;
    const auto nHitSlots = static_cast<uint32_t>(stride) * ::mkfitdev::kMaxTrkHits;
    auto key = make_device_buffer<float[]>(queue, nHitSlots);
    auto chi2f = make_device_buffer<float[]>(queue, nHitSlots);
    auto chi2b = make_device_buffer<float[]>(queue, nHitSlots);
    auto order = make_device_buffer<int16_t[]>(queue, nHitSlots);
    auto meta = make_device_buffer<int32_t[]>(queue, 3 * stride);
    const FitScratch sc{key.data(), chi2f.data(), chi2b.data(), order.data(), meta.data(), stride};
    const FitHitAccess ha{es, hits, nPixel, cpe, clusters};
    if constexpr (kNN == 1) {
      // GPU: one thread per track, the fit split into stage kernels; 64 threads per block (<= 128)
      constexpr uint32_t kThreads = 64;
      const uint32_t blocks = divide_up_by(static_cast<uint32_t>(nTracks), kThreads);
      const auto wd = make_workdiv<Acc1D>(blocks, kThreads);
      auto stages = [&](auto cpeTag) {
        constexpr bool kCpe = decltype(cpeTag)::value;
        alpaka::exec<Acc1D>(queue, wd, KernelFinalFitStage<0, kCpe>{}, ha, tracks, nTracks, nTracksDev, sc, counters);
        alpaka::exec<Acc1D>(queue, wd, KernelFinalFitStage<1, kCpe>{}, ha, tracks, nTracks, nTracksDev, sc, counters);
        alpaka::exec<Acc1D>(queue, wd, KernelFinalFitStage<2, kCpe>{}, ha, tracks, nTracks, nTracksDev, sc, counters);
        // the outlier rounds (MkBuilder::fittracks remap loop, until a round removes no hit): the first
        // kStagedOutlierRounds as stage kernels (removal, forward and backward refit), the rest in one tail kernel
        for (int round = 0; round < kStagedOutlierRounds; ++round) {
          alpaka::exec<Acc1D>(
              queue, wd, KernelFinalFitStage<3, kCpe>{}, ha, tracks, nTracks, nTracksDev, sc, counters, round);
          alpaka::exec<Acc1D>(
              queue, wd, KernelFinalFitStage<4, kCpe>{}, ha, tracks, nTracks, nTracksDev, sc, counters, round);
          alpaka::exec<Acc1D>(
              queue, wd, KernelFinalFitStage<5, kCpe>{}, ha, tracks, nTracks, nTracksDev, sc, counters, round);
        }
        alpaka::exec<Acc1D>(queue, wd, KernelFinalFitTail<kCpe>{}, ha, tracks, nTracks, nTracksDev, sc, counters);
      };
      if (useCpe)
        stages(std::true_type{});
      else
        stages(std::false_type{});
    } else {
      // CPU backends only: else the GPU library compiles the never-launched KernelFinalFitGrouped<1>.
      // Not a template dispatch: that also changes GCC's inlining of the CPU fit.
#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED))
      // CPU: MkFitCore grouping by the number of hits with index >= 0 into kNN-wide Matriplex groups
      const int maxGroups = nTracks / kNN + kMaxGroupKey + 2;
      auto perm = make_device_buffer<int32_t[]>(queue, nTracks);
      auto gStart = make_device_buffer<int32_t[]>(queue, maxGroups);
      auto gCount = make_device_buffer<int32_t[]>(queue, maxGroups);
      auto nGroups = make_device_buffer<int32_t>(queue);
      FitGroups groups{perm.data(), gStart.data(), gCount.data(), nGroups.data()};
      alpaka::exec<Acc1D>(
          queue, make_workdiv<Acc1D>(1, 1), KernelGroupTracks<kNN>{}, tracks, nTracks, nTracksDev, groups);
      constexpr uint32_t kElements = 16;
      const uint32_t blocks = divide_up_by(static_cast<uint32_t>(maxGroups), kElements);
      auto grouped = [&](auto cpeTag) {
        constexpr bool kCpe = decltype(cpeTag)::value;
        alpaka::exec<Acc1D>(queue,
                            make_workdiv<Acc1D>(blocks, kElements),
                            KernelFinalFitGrouped<kNN, kCpe>{},
                            ha,
                            tracks,
                            maxGroups,
                            groups,
                            sc,
                            counters);
      };
      if (useCpe)
        grouped(std::true_type{});
      else
        grouped(std::false_type{});
#endif
    }
    // the scratch buffers are queue-ordered (caching allocator): freed after the kernel completes
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::fit
