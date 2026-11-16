
#include <array>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederBinKernels.h"
#include "MkSeederClean.h"
#include "MkSeederCleanKernels.h"
#include "MkSeederCleanStep.h"
#include "MkSeederDevice.h"
#include "MkSeederKernels.h"
#include "MkSeederPostKernels.h"
#include "MkSeederQueues.h"
#include "MkSeederRadixSort.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  using namespace cms::alpakatools;

  namespace {
    void findQuadsOneSide(Queue &queue,
                          mkseeder::ChainFinderInputs const &view,
                          mkseeder::Queues const &queues,
                          std::vector<int> const &binnedLayerIds,
                          mkseeder::WorkPieces const &pieces,
                          mkseeder::WorkPieces const &quadPieces) {
      constexpr uint32_t kThreads = 128;
      // the queue kernels stride over a count only the device knows: a fixed grid
      constexpr uint32_t kQueueBlocks = std::is_same_v<Device, alpaka::DevCpu> ? 1 : 256;
      const auto wq = make_workdiv<Acc1D>(kQueueBlocks, kThreads);
      if (view.chainEvent.nWorkItems > 0) {
        const auto wd = make_workdiv<Acc1D>(divide_up_by((uint32_t)view.chainEvent.nWorkItems, kThreads), kThreads);
        alpaka::exec<Acc1D>(queue, wd, KernelFindStartDoublets{}, view, queues, pieces);
        if constexpr (!std::is_same_v<Device, alpaka::DevCpu>)
          alpaka::exec<Acc1D>(
              queue, make_workdiv<Acc1D>(1024, kThreads), KernelFindStartDoubletPieces{}, view, queues, pieces);
      }
      // trip_net_c on a GPU: the stage-c network's pass over a position's triplets (from the fourth position on: no
      // triplet reaches the first three) runs in the miss kernel of the stage c before it, the kept triplets onto the
      // spare queue (2 n), which then replaces the triplet queue (stageCNetFilterQueue; a CPU backend's stage c evaluates
      // the network before its pushes)
      const int spare = 2 * view.chain.nPositions;
      auto nextFiltered = [&](int p) {
        if (!view.chain.useStageCNet)
          return -1;
        for (int q = p + 1; q < view.chain.nPositions; ++q)
          if (binnedLayerIds[q] >= 0)
            return q >= 3 ? q : -1;
        return -1;
      };
      for (int p = 0; p < view.chain.nPositions; ++p) {
        if (binnedLayerIds[p] < 0)
          continue;
        alpaka::exec<Acc1D>(queue, wq, KernelExtendToTriplets{}, view, queues, p);
        // on a CPU backend, KernelExtendToTriplets forwards the misses itself (MkSeederKernelsCpu.h)
        if constexpr (!std::is_same_v<Device, alpaka::DevCpu>)
          alpaka::exec<Acc1D>(queue, wq, KernelExtendToTripletsMiss{}, view, queues, p, nextFiltered(p), spare);
        if (std::is_same_v<Device, alpaka::DevCpu> || !KernelExtendToQuads::splitAt(binnedLayerIds[p])) {
          alpaka::exec<Acc1D>(
              queue, wq, KernelExtendToQuads{}, view, queues, p, quadPieces, KernelExtendToQuads::kWhole);
        } else {
          // a GPU at the positions with long windows: the hits that can pass (a light kernel, many blocks), their
          // tests, the forwarding (a light kernel)
          alpaka::exec<Acc1D>(
              queue, make_workdiv<Acc1D>(1024, kThreads), KernelScanQuadWindows{}, view, queues, p, quadPieces);
          alpaka::exec<Acc1D>(
              queue, wq, KernelExtendToQuads{}, view, queues, p, quadPieces, KernelExtendToQuads::kPairs);
          alpaka::exec<Acc1D>(
              queue, make_workdiv<Acc1D>(1024, kThreads), KernelForwardQuadCandidates{}, view, queues, p, quadPieces);
        }
        alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(4, kThreads), KernelReleaseConsumedChunks{}, view, queues, p);
      }
    }

    // the first n values of a device column set to the byte value
    template <typename T>
    void memsetColumn(Queue &queue, T *device, size_t n, uint8_t value) {
      auto view = make_device_view<T>(alpaka::getDev(queue), device, n);
      alpaka::memset(queue, view, value);
    }
  }  // namespace

  size_t findQuads(Queue &queue,
                   mkseeder::ChainFinderInputs const (&views)[2],
                   std::vector<int> const (&binnedLayerIds)[2],
                   uint32_t nQueues,
                   uint32_t &poolChunks,
                   mkseeder::QuadsView quads,
                   uint32_t quadCapacity,
                   ::mkseeder::MkSeederFinderCounters *counters) {
    // the queues, reused by the two sides in turn; a queue can hold the whole pool
    constexpr bool kSerial = alpaka::accMatchesTags<Acc1D, alpaka::TagCpuSerial>;
    constexpr bool kGpu = !std::is_same_v<Device, alpaka::DevCpu>;
    constexpr uint32_t kCandidateChunkSize = mkseeder::Queues::kCandidateChunkSize;
    auto poolBytes = [](uint32_t chunks) {
      return ::mkseeder::MkSeederCandidatePoolSoA::computeDataSize(std::array<int32_t, 2>{
          int32_t(chunks * kCandidateChunkSize), int32_t(kSerial ? chunks * kCandidateChunkSize : 1)});
    };
    if constexpr (kGpu) {
      // the caching allocator hands out a power of two: the chunks that fill it (the largest count in that bin)
      size_t bin = 256;
      while (bin < poolBytes(poolChunks))
        bin <<= 1;
      uint32_t lo = poolChunks, hi = uint32_t(bin / (kCandidateChunkSize * sizeof(uint32_t))) + 1;
      while (lo + 1 < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        (poolBytes(mid) <= bin ? lo : hi) = mid;
      }
      poolChunks = lo;
    }
    const uint32_t nq = nQueues, mcpq = poolChunks;
    const uint32_t nCand = poolChunks * mkseeder::Queues::kCandidateChunkSize;
    auto dev = alpaka::getDev(queue);
    // the candidate pool (the serial backend: with the candidates' lines) and the queues' tables (a GPU: with stage
    // c's flags), released once both sides are enqueued
    MkSeederCandidatePoolDeviceCollection pool(queue, int32_t(nCand), int32_t(kSerial ? nCand : 1));
    MkSeederQueueTablesDeviceCollection tables(
        queue, int32_t(nq), int32_t(nq * mcpq), int32_t(poolChunks), int32_t(kGpu ? nCand : 1));
    auto poolView = pool.view();
    auto tablesView = tables.view();
    mkseeder::Queues Q;
    Q.pool = mkseeder::CandidatesView(poolView.candidates());
    Q.lines = mkseeder::CandidateLinesView(poolView.lines());
    Q.tripletStage = mkseeder::TripletStageView(tablesView.tripletStage());
    Q.hasLines = kSerial;
    Q.poolChunks = poolChunks, Q.maxChunksPerQueue = mcpq;
    Q.queues = mkseeder::QueuesView(tablesView.queues());
    Q.chunkTable = mkseeder::ChunksView(tablesView.chunkTable());
    Q.freeChunks = mkseeder::ChunksView(tablesView.freeChunks());
    Q.counters = counters;
    Q.quads = quads;
    Q.quadCapacity = quadCapacity;
    // stage c's found flags, one per candidate a queue can hold (cleared by KernelExtendToTripletsMiss as it reads them; a
    // CPU backend's KernelExtendToTriplets forwards the misses itself)
    if constexpr (kGpu)
      memsetColumn(queue, Q.tripletStage.foundTriplet().data(), nCand, 0);
    // a GPU: the piece list of stage b (KernelFindStartDoublets), 256k (work item, piece) pairs = 2 MB, and stage d's
    // (candidate, hit) pairs (KernelScanQuadWindows), 512k = 4 MB
    constexpr uint32_t kPieceCapacity = kGpu ? 262144 : 0, kPairCapacity = kGpu ? 524288 : 0;
    auto pieceBuffer = make_device_buffer<uint32_t[]>(queue, 2 + 2 * (kPieceCapacity + kPairCapacity));
    const mkseeder::WorkPieces pieces{pieceBuffer.data(), kPieceCapacity},
        quadPieces{pieceBuffer.data() + 1 + 2 * kPieceCapacity, kPairCapacity};
    for (int sd = 0; sd < 2; ++sd) {
      // each side starts with empty queues and the whole pool
      memsetColumn(queue, Q.queues.count().data(), nq, 0);
      if constexpr (kGpu) {
        memsetColumn(queue, pieces.data, 1, 0);
        memsetColumn(queue, quadPieces.data, 1, 0);
        memsetColumn(queue, &counters->stageCItems, 2, 0);  // and stageCNetBlocks
      }
      memsetColumn(queue, &counters->nextFreshChunk, 2, 0);
      memsetColumn(queue, Q.chunkTable.chunk().data(), size_t(nq) * mcpq, 0xff);
      findQuadsOneSide(queue, views[sd], Q, binnedLayerIds[sd], pieces, quadPieces);
      auto used = make_device_view<uint32_t>(dev, &counters->chunksUsed[sd], 1);
      alpaka::memcpy(queue, used, make_device_view<const uint32_t>(dev, &counters->nextFreshChunk, 1));
    }
    return alpaka::getExtentProduct(pool.buffer());
  }

  void extendAndFillSeeds(Queue &queue,
                          mkseeder::ChainFinderInputs const &view,
                          mkseeder::PostInputs const &post,
                          mkseeder::FitInputs const &fit,
                          mkseeder::SeedHitsView seeds,
                          mkseeder::SeedStatesView states,
                          uint32_t n,
                          const uint32_t *count) {
    if (n == 0)
      return;
    // a GPU: the extension with a block per seed, the pixel fill's layers in parallel (on a CPU backend the seed's
    // loop over its layers is as fast)
    constexpr bool kGpu = !std::is_same_v<Device, alpaka::DevCpu>;
    const bool fill = kGpu && post.nPixelLayersWithHits > 0;
    // the scratch: the input hits (the refits' fallback), the fill's per seed and per seed and layer steps
    PortableCollection<MkSeederPostScratchSoA> scratch(queue,
                                                       int32_t(fit.refitEnabled ? n : 1),
                                                       int32_t(kGpu ? n : 1),
                                                       int32_t(fill ? n * uint32_t(post.nPixelLayersWithHits) : 1));
    const mkseeder::SeedHitListView backup(scratch.view().hitsBackup());
    const mkseeder::FillPrepsView preps(scratch.view().preps());
    const mkseeder::FillCandidatesView cands(scratch.view().fillCandidates());
    constexpr uint32_t kThreads = 64;
    const auto wd = make_workdiv<Acc1D>(divide_up_by(n, kThreads), kThreads);
    if (fit.refitEnabled)
      alpaka::exec<Acc1D>(queue, wd, KernelBackupSeedHits{}, mkseeder::SeedHitsConstView(seeds), backup, n, count);
    if constexpr (kGpu)
      alpaka::exec<Acc1D>(
          queue, make_workdiv<Acc1D>(n, kExtLanes<Acc1D>), KernelExtendBlock{}, view, post, seeds, n, count);
    else
      alpaka::exec<Acc1D>(queue, wd, KernelExtendAndFillSeeds{}, view, post, seeds, n, count);
    if (fill) {
      alpaka::exec<Acc1D>(
          queue, wd, KernelPrepareFills{}, view, post, mkseeder::SeedHitsConstView(seeds), preps, n, count);
      const uint32_t nt = n * (uint32_t)post.nPixelLayersWithHits;
      const auto wl = make_workdiv<Acc1D>(divide_up_by(nt, kThreads), kThreads);
      alpaka::exec<Acc1D>(queue, wl, KernelFillLayers{}, view, post, preps, cands, n, count);
      alpaka::exec<Acc1D>(queue, wd, KernelFillFinish{}, view, post, seeds, preps, cands, n, count);
    }
    if (fit.refitEnabled)
      refitSeeds(queue, fit, seeds, backup, states, n, count);
  }

  // thread t (0..n) writes the starts of the bins (bin[t - 1], bin[t]] (bin[-1] = -1, bin[n] = nb) of the layer's n
  // hits from firstHit: t, at firstBinStart + the bin
  class KernelComputeBinStarts {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  mkseeder::HitsView hits,
                                  mkseeder::MutableTableView<::mkseeder::MkSeederBinStartsSoA> binStarts,
                                  uint32_t firstHit,
                                  uint32_t n,
                                  uint32_t nb,
                                  uint32_t firstBinStart) const {
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, n + 1)) {
        const uint32_t lo = t == 0 ? 0 : hits.binIndex(firstHit + t - 1) + 1;
        const uint32_t hi = t == n ? nb : hits.binIndex(firstHit + t);
        for (uint32_t b = lo; b <= hi; ++b)
          binStarts.start(firstBinStart + b) = t;
      }
    }
  };

  class KernelComputePolarCoordinates {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  mkseeder::HitsView hits,
                                  mkseeder::MutableTableView<::mkseeder::MkSeederHitsCpuCacheSoA> cache,
                                  uint32_t n) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        cache.phiGlobal(i) = libm::atan2(acc, hits.y(i), hits.x(i));
        cache.rGlobal(i) = alpaka::math::sqrt(acc, hits.x(i) * hits.x(i) + hits.y(i) * hits.y(i));
      }
    }
  };

  void computeGlobalPolarCoordinates(Queue &queue,
                                     MkSeederEventHitsDeviceCollection const &hits,
                                     MkSeederCpuCacheDeviceCollection &cache,
                                     uint32_t n) {
    if (n == 0)
      return;
    constexpr uint32_t kThreads = 128;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(n, kThreads), kThreads),
                        KernelComputePolarCoordinates{},
                        mkseeder::HitsView(hits.const_view().hits()),
                        mkseeder::MutableTableView<::mkseeder::MkSeederHitsCpuCacheSoA>(cache.view().hits()),
                        n);
  }

  // one thread per layer with a covariance
  class KernelComputeOTModuleFrames {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  mkseeder::ChainFinderInputs v,
                                  mkseeder::PostInputs pv,
                                  mkseeder::MutableTableView<::mkseeder::MkSeederOTHitFramesSoA> frames,
                                  mkseeder::MutableTableView<::mkseeder::MkSeederLayersCpuCacheSoA> layersCache) const {
      constexpr int kLay[3] = {4, 6, 8};
      if (cms::alpakatools::once_per_grid(acc))
        for (int l = 0; l < pv.nLayers; ++l)
          layersCache.phiWindowPad(l) = layerPhiPadExpression(acc, pv, l);
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, 3u)) {
        const int l = kLay[j];
        if (l >= v.nLayers || !v.layerGeometry[l].isBinned())
          continue;
        const mkseeder::EventLayer L = v.layers[l];
        for (uint32_t i = 0; i < L.nHits(); ++i) {
          const uint32_t c = pv.layers.firstOTHit(l) + i, ih = L.firstHit() + i;
          float E[6], F[9];
          mkseeder::hitCovariance(pv.otHits, c, E);
          extensionLocalFrame(acc, pv.hits.x(ih), pv.hits.y(ih), E, F);
          auto f = frames[c];
          f.hitR() = F[0], f.localXDirectionX() = F[1], f.localXDirectionY() = F[2], f.localYDirectionX() = F[3];
          f.localYDirectionY() = F[4], f.localYDirectionZ() = F[5], f.normalX() = F[6], f.normalY() = F[7];
          f.normalZ() = F[8];
        }
      }
    }
  };

  // a GPU: one thread per OT1-3 P hit row c (of the layers 4, 6, 8 in turn): its frame
  class KernelComputeOTModuleFramesParallel {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  mkseeder::ChainFinderInputs v,
                                  mkseeder::PostInputs pv,
                                  mkseeder::MutableTableView<::mkseeder::MkSeederOTHitFramesSoA> frames,
                                  uint32_t nCov) const {
      constexpr int kLay[3] = {4, 6, 8};
      for (uint32_t c : cms::alpakatools::uniform_elements(acc, nCov)) {
        for (int j = 0; j < 3; ++j) {
          const int l = kLay[j];
          if (l >= v.nLayers || !v.layerGeometry[l].isBinned())
            continue;
          const mkseeder::EventLayer L = v.layers[l];
          const int32_t i = int32_t(c) - pv.layers.firstOTHit(l);
          if (i < 0 || i >= int32_t(L.nHits()))
            continue;
          const uint32_t ih = L.firstHit() + uint32_t(i);
          float E[6], F[9];
          mkseeder::hitCovariance(pv.otHits, c, E);
          extensionLocalFrame(acc, pv.hits.x(ih), pv.hits.y(ih), E, F);
          auto f = frames[c];
          f.hitR() = F[0], f.localXDirectionX() = F[1], f.localXDirectionY() = F[2], f.localYDirectionX() = F[3];
          f.localYDirectionY() = F[4], f.localYDirectionZ() = F[5], f.normalX() = F[6], f.normalY() = F[7];
          f.normalZ() = F[8];
        }
      }
    }
  };

  void computeOTModuleFrames(Queue &queue,
                             mkseeder::ChainFinderInputs const &view,
                             mkseeder::PostInputs const &post,
                             MkSeederCpuCacheDeviceCollection &cache,
                             uint32_t nCov) {
    if constexpr (!std::is_same_v<Device, alpaka::DevCpu>) {
      // a GPU: the frames only (the hits' polar coordinates and the layers' pads are computed where they are used)
      constexpr uint32_t kThreads = 128;
      if (nCov > 0)
        alpaka::exec<Acc1D>(queue,
                            make_workdiv<Acc1D>(divide_up_by(nCov, kThreads), kThreads),
                            KernelComputeOTModuleFramesParallel{},
                            view,
                            post,
                            mkseeder::MutableTableView<::mkseeder::MkSeederOTHitFramesSoA>(cache.view().otHitFrames()),
                            nCov);
      return;
    }
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(1u, 4u),
                        KernelComputeOTModuleFrames{},
                        view,
                        post,
                        mkseeder::MutableTableView<::mkseeder::MkSeederOTHitFramesSoA>(cache.view().otHitFrames()),
                        mkseeder::MutableTableView<::mkseeder::MkSeederLayersCpuCacheSoA>(cache.view().layers()));
  }

  void computeBinStarts(Queue &queue,
                        MkSeederEventHitsDeviceCollection &hits,
                        uint32_t firstHit,
                        uint32_t n,
                        uint32_t nb,
                        uint32_t firstBinStart) {
    constexpr uint32_t kThreads = 128;
    const auto wd = make_workdiv<Acc1D>(divide_up_by(n + 1, kThreads), kThreads);
    alpaka::exec<Acc1D>(queue,
                        wd,
                        KernelComputeBinStarts{},
                        mkseeder::HitsView(hits.const_view().hits()),
                        mkseeder::MutableTableView<::mkseeder::MkSeederBinStartsSoA>(hits.view().binStarts()),
                        firstHit,
                        n,
                        nb,
                        firstBinStart);
  }

  namespace {
    using SortKeysView = ::mkseeder::MkSeederSortKeysSoA::View;
    using SortKeysConstView = ::mkseeder::MkSeederSortKeysSoA::ConstView;
    // the 128-bit keys of the sort order (by layers, then hits): layers + 1 (8 bits each) and hit 0 in the high
    // word; hits 1-3 (21 bits each) in the low word. A hit is its position among its layer's hits in cluster order
    // (the same order as its cluster index among the layer's: the layers are compared first), below 2^21
    // (MkSeederProducer checks the layers' hit counts)
    class KernelQuadSortKeys {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    mkseeder::QuadsConstView q,
                                    const uint32_t *count,
                                    uint32_t cap,
                                    mkseeder::CleanInputs clean,
                                    SortKeysView keys) const {
        const uint32_t n = *count < cap ? *count : cap;
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
          const auto x = q[i];
          uint64_t h = 0;
          uint32_t hit[4];
          for (int k = 0; k < 4; ++k) {
            const int layer = x.layers()(k);
            h = (h << 8) | uint64_t(uint8_t(layer + 1));
            hit[k] = x.hits()(k);
            if (layer >= 0)
              hit[k] = clean::cpos(clean, layer, hit[k]) - clean.layers.firstHit(layer);
          }
          keys.keyHigh(i) = (h << 32) | hit[0];
          keys.keyLow(i) = (uint64_t(hit[1]) << 42) | (uint64_t(hit[2]) << 21) | hit[3];
          keys.index(i) = i;
        }
      }
    };
    // the sorted quads' rows (the cleaning reads the quads through them)
    class KernelSortedQuadRows {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t cap,
                                    SortKeysConstView sorted,
                                    ::mkseeder::MkSeederCleanItemsSoA::View items) const {
        const uint32_t n = *count < cap ? *count : cap;
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, n))
          items.quadOfSortedIndex(i) = sorted.index(i);
      }
    };
  }  // namespace

  namespace {
    // the padding block's rows are rounded up to the layout's alignment: rows that make an allocation larger than
    // half a bin by at least one byte
    constexpr size_t kPaddingSlack = 256;
    // the caching allocator's bin of n bytes (a power of two, at least 256)
    size_t allocatorBin(size_t n) {
      size_t bin = 256;
      while (bin < n)
        bin <<= 1;
      return bin;
    }
  }  // namespace

  void sortAndCleanQuads(Queue &queue, mkseeder::CleanInputs const &clean, size_t minBytes) {
    const uint32_t capacity = clean.cap;
    if (capacity == 0)
      return;
    const auto wn = make_workdiv<Acc1D>(divide_up_by(capacity, 128u), 128u);
    // the sort's keys and counts, the cleaning's state (per quad, per hit, the per-hit lists: nTot + 1 offsets, 4
    // per quad) and the padding up to minBytes
    constexpr bool kGpu = !std::is_same_v<Device, alpaka::DevCpu>;
    const uint32_t nTot = clean.nTot;
    const auto sortRows = GlobalRadixSort::scratchRows(capacity);
    std::array<int32_t, 8> rows{sortRows[0],
                                sortRows[1],
                                sortRows[2],
                                int32_t(capacity),
                                int32_t(std::max(nTot, 1u)),
                                int32_t(nTot + 1),
                                int32_t(4 * capacity),
                                1};
    // in the caching allocator's bin of minBytes (a power of two) when it fits there: more than half of it
    const size_t bytes = ::mkseeder::MkSeederSortCleanScratchSoA::computeDataSize(rows);
    const size_t bin = allocatorBin(minBytes);
    if (kGpu && bytes <= bin / 2)
      rows[7] = int32_t(bin / 2 - bytes + 1 + kPaddingSlack);
    PortableCollection<::mkseeder::MkSeederSortCleanScratchSoA> scratch(queue, rows);
    auto sv = scratch.view();
    GlobalRadixSort sort(sv.keys0(), sv.keys1(), sv.counts(), capacity);
    // the quads stay where they are: the cleaning reads them in the sorted order through their rows
    alpaka::exec<Acc1D>(queue, wn, KernelQuadSortKeys{}, clean.quads, clean.count, capacity, clean, sort.keys());
    // byte 11 (bits 24-31 of hit 0) is 0: no pass
    const auto sorted = sort.sort(queue, clean.count, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15});
    alpaka::exec<Acc1D>(queue, wn, KernelSortedQuadRows{}, clean.count, capacity, sorted, sv.items());
    cleanQuadsAndTriplets(
        queue, clean, sort, CleanScratch{sv.items(), sv.hitUse(), sv.hitItemOffsets(), sv.hitItems()});
  }

  void scatterHitsToBins(Queue &queue,
                         mkseeder::HitsView in,
                         mkseeder::OTHitsView inOT,
                         const uint32_t *count,
                         GlobalRadixSort &sort,
                         uint32_t n,
                         MkSeederEventHitsDeviceCollection &hits,
                         MkSeederEventTablesDeviceCollection const &tables,
                         mkseeder::LayerGeometryView layerGeometry,
                         MkSeederClustersDeviceCollection &clusters,
                         mkseeder::HitTagInputs const &tags) {
    if (n == 0)
      return;
    // the keys: the layer and the bin in the low word; a stable sort by its 4 low bytes (slot < 128)
    const auto wn = make_workdiv<Acc1D>(divide_up_by(n, 128u), 128u);
    auto keys = sort.keys();
    alpaka::exec<Acc1D>(
        queue, wn, bins::KernelFillBinKeys{}, in, n, keys.keyHigh().data(), keys.keyLow().data(), keys.index().data());
    const auto sorted = sort.sort(queue, count, {0, 1, 2, 3});
    auto out = hits.view();
    auto outClusters = clusters.view();
    alpaka::exec<Acc1D>(queue,
                        wn,
                        bins::KernelScatterBinnedHits{},
                        in,
                        inOT,
                        mkseeder::MutableTableView<::mkseeder::MkSeederHitsSoA>(out.hits()),
                        mkseeder::MutableTableView<::mkseeder::MkSeederClusterOrderSoA>(out.clusterOrder()),
                        mkseeder::MutableTableView<::mkseeder::MkSeederOTHitsSoA>(out.otHits()),
                        mkseeder::EventLayersView(tables.const_view().layers()),
                        layerGeometry,
                        mkseeder::MutableTableView<::mkseeder::MkSeederPixelClustersSoA>(outClusters.pixelClusters()),
                        mkseeder::MutableTableView<::mkseeder::MkSeederOTClustersSoA>(outClusters.otClusters()),
                        sorted.keyLow().data(),
                        sorted.index().data(),
                        n,
                        tags);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder
