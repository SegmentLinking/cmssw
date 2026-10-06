// Top-level translation unit of the seeds component: kernels in src/alpaka/seeds/*.h.
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/SeedsAlgo.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandSeedOpsLaunch.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/clean/CleanAlgo.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/seeds/SeedKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::seeds {

  void importSeeds(Queue& queue,
                   ::mkfitdev::SeedSoAView seeds,
                   int32_t capacity,
                   ::mkfitdev::SeedPartitionLimits const& limits,
                   ::mkfitdev::SeedCandsSoA::View seedCands,
                   ::mkfitdev::CandSlotsSoA::View slots,
                   ::mkfitdev::CandHotsSoA::View hots,
                   int32_t hotsPerSeed,
                   DeviceHitPositions const& hitPos) {
    if (capacity <= 0)
      return;
    constexpr int kThreads = 128;
    const int blocks = (capacity + kThreads - 1) / kThreads;
    auto binCount = cms::alpakatools::make_device_buffer<int32_t[]>(queue, kSeedNBins);
    auto binFill = cms::alpakatools::make_device_buffer<int32_t[]>(queue, kSeedNBins);
    auto binStart = cms::alpakatools::make_device_buffer<int32_t[]>(queue, kSeedNBins + 1);
    auto members = cms::alpakatools::make_device_buffer<int32_t[]>(queue, capacity);
    alpaka::memset(queue, binCount, 0);
    alpaka::memset(queue, binFill, 0);
    const auto wd = cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads);
    const auto wd1 = cms::alpakatools::make_workdiv<Acc1D>(1, kScanBlock);
    alpaka::exec<Acc1D>(queue,
                        wd,
                        KernelSeedPrepare{},
                        seeds,
                        limits,
                        binCount.data(),
                        hitPos.x,
                        hitPos.y,
                        hitPos.z,
                        hitPos.layerHitBase);
    alpaka::exec<Acc1D>(queue, wd1, KernelSeedScan{}, seeds, binCount.data(), binStart.data(), seedCands, hotsPerSeed);
    alpaka::exec<Acc1D>(queue,
                        wd,
                        KernelSeedFill{},
                        ::mkfitdev::SeedSoAConstView(seeds),
                        binStart.data(),
                        binFill.data(),
                        members.data());
    alpaka::exec<Acc1D>(queue, wd, KernelSeedRank{}, seeds, binStart.data(), members.data());
    alpaka::exec<Acc1D>(
        queue, wd, KernelSeedInitCands{}, ::mkfitdev::SeedSoAConstView(seeds), seedCands, slots, hots, hotsPerSeed);
  }

  void exportBestCands(Queue& queue,
                       ::mkfitdev::SeedCandsSoA::ConstView seedCands,
                       ::mkfitdev::CandSlotsSoA::ConstView slots,
                       ::mkfitdev::CandHotsSoA::ConstView hots,
                       int32_t hotsPerSeed,
                       int32_t const* nSeedsDev,
                       int32_t capacity,
                       int8_t const* passed,
                       ::mkfitdev::TrackSoAView out,
                       uint32_t* brokenChains) {
    if (capacity <= 0)
      return;
    constexpr int kThreads = 128;
    const int blocks = (capacity + kThreads - 1) / kThreads;
    auto dest = cms::alpakatools::make_device_buffer<int32_t[]>(queue, capacity);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, kScanBlock),
                        KernelExportScan{},
                        seedCands,
                        nSeedsDev,
                        passed,
                        dest.data(),
                        out);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads),
                        KernelExportBestCands{},
                        seedCands,
                        slots,
                        hots,
                        hotsPerSeed,
                        nSeedsDev,
                        dest.data(),
                        out,
                        brokenChains);
  }

  void exportAndClean(Queue& queue,
                      ::mkfitdev::SeedCandsSoA::ConstView seedCands,
                      ::mkfitdev::CandSlotsSoA::ConstView slots,
                      ::mkfitdev::CandHotsSoA::ConstView hots,
                      int32_t hotsPerSeed,
                      int32_t const* nSeedsDev,
                      int32_t capacity,
                      bool removeDuplicates,
                      const float dc[4],
                      const uint64_t* pixelPriorityLayers,
                      ::mkfitdev::TrackSoAView exported,
                      ::mkfitdev::TrackSoAView out,
                      uint32_t* brokenChains) {
    if (capacity <= 0)
      return;
    // no filter here: every row s < *nSeedsDev with candidates is a survivor of the engine's post-filter
    exportBestCands(queue, seedCands, slots, hots, hotsPerSeed, nSeedsDev, capacity, nullptr, exported, brokenChains);
    clean::CleanAlgo algo(queue, capacity);
    if (removeDuplicates) {
      algo.flagDuplicates(queue, exported, ::mkfitdev::clean::makeDupCleanParams(dc, pixelPriorityLayers));
      algo.removeDuplicates(queue, exported, out);
    } else {
      algo.filterTracks(queue, exported, out, 0);  // stable copy (n-hits >= 0 and not silly: the engine filtered)
    }
  }

  void runChainTail(Queue& queue,
                    ::mkfitdev::SeedCandsSoA::View seedCands,
                    ::mkfitdev::CandSlotsSoA::View slots,
                    ::mkfitdev::CandHotsSoA::View hots,
                    int32_t hotsPerSeed,
                    int32_t const* nSeedsDev,
                    int32_t capacity,
                    bool bkwRep,
                    int minHitsQF,
                    bool removeDuplicates,
                    const float dc[4],
                    const uint64_t* pixelPriorityLayers,
                    ::mkfitdev::TrackSoAView exported,
                    ::mkfitdev::TrackSoAView out) {
    if (capacity <= 0)
      return;
    auto passed = cms::alpakatools::make_device_buffer<int8_t[]>(queue, capacity);
    // rows [nSeeds, capacity) are allocated; the filter kernel loops over capacity rows, export reads s < *nSeedsDev
    ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::filterSeeds(
        queue, seedCands, slots, hots, passed.data(), bkwRep, true, minHitsQF, capacity);
    exportBestCands(queue, seedCands, slots, hots, hotsPerSeed, nSeedsDev, capacity, passed.data(), exported);
    clean::CleanAlgo algo(queue, capacity);
    if (removeDuplicates) {
      algo.flagDuplicates(queue, exported, ::mkfitdev::clean::makeDupCleanParams(dc, pixelPriorityLayers));
      algo.removeDuplicates(queue, exported, out);
    } else {
      algo.filterTracks(queue, exported, out, 0);  // stable copy (n-hits >= 0 and not silly: export already filtered)
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::seeds
