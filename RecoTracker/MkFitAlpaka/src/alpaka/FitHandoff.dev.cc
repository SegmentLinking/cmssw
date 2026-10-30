// Building -> fit device handoff: entry points of src/alpaka/engine/FitHandoff.h.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/FitHandoff.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/FitHandoffKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff {

  using namespace cms::alpakatools;

  namespace {
    constexpr uint32_t kThreadsPerBlock = 128;
    constexpr uint32_t kClusScratchArrays = 10;  // the int32 arrays of ClusScratch, cap entries each
  }  // namespace

  void selectFitInput(Queue& queue,
                      ::mkfitdev::TrackSoAConstView in,
                      int capacity,
                      CandCutSel const& sel,
                      ::mkfitdev::TrackSoAView out) {
    if (capacity <= 0)
      return;
    // destination row of every input row; queue-ordered scratch (caching allocator)
    auto dst = make_device_buffer<int32_t[]>(queue, capacity);
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(1, kSelThreads), KernelCandSelIndex{}, in, capacity, sel, dst.data(), out);
    const uint32_t blocks = divide_up_by(static_cast<uint32_t>(capacity), kThreadsPerBlock);
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(blocks, kThreadsPerBlock), KernelCandSelCopy{}, in, capacity, dst.data(), out);
  }

  void buildClusterCpe(Queue& queue,
                       SiPixelDigisSoAConstView digis,
                       uint32_t nDigis,
                       SiPixelClustersSoAConstView clusters,
                       uint32_t nClusters,
                       const ::mkfitdev::handoff::ClusterRef* refs,
                       uint32_t nRows,
                       ::mkfitdev::cpe::ClusterCpe* out,
                       ::mkfitdev::handoff::ClusterCpeCounters* counters) {
    if (nRows == 0)
      return;
    const uint32_t cap = nClusters > 0 ? nClusters : 1;
    auto buf = make_device_buffer<int32_t[]>(queue, kClusScratchArrays * cap);  // queue-ordered scratch
    int32_t* base = buf.data();
    ClusScratch scratch{base,
                        base + cap,
                        base + 2 * cap,
                        base + 3 * cap,
                        base + 4 * cap,
                        base + 5 * cap,
                        base + 6 * cap,
                        base + 7 * cap,
                        base + 8 * cap,
                        base + 9 * cap,
                        cap};
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(divide_up_by(cap, kThreadsPerBlock), kThreadsPerBlock), KernelClusInit{}, scratch);
    if (nDigis > 0) {
      const auto wd = make_workdiv<Acc1D>(divide_up_by(nDigis, kThreadsPerBlock), kThreadsPerBlock);
      alpaka::exec<Acc1D>(queue, wd, KernelClusCount{}, digis, nDigis, clusters, scratch, counters);
      alpaka::exec<Acc1D>(queue, wd, KernelClusMinMax{}, digis, nDigis, clusters, scratch);
      alpaka::exec<Acc1D>(queue, wd, KernelClusEdges{}, digis, nDigis, clusters, scratch);
    }
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nRows, kThreadsPerBlock), kThreadsPerBlock),
                        KernelClusGather{},
                        clusters,
                        scratch,
                        refs,
                        nRows,
                        out,
                        counters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff
