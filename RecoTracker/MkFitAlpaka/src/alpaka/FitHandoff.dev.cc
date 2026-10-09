// Building -> fit device handoff: entry points of src/alpaka/engine/FitHandoff.h.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/FitHandoff.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/FitHandoffKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff {

  using namespace cms::alpakatools;

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
    constexpr uint32_t kCopyThreads = 128;
    const uint32_t blocks = divide_up_by(static_cast<uint32_t>(capacity), kCopyThreads);
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(blocks, kCopyThreads), KernelCandSelCopy{}, in, capacity, dst.data(), out);
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
    auto buf = make_device_buffer<int32_t[]>(queue, 10 * cap);  // queue-ordered scratch
    int32_t* p = buf.data();
    ClusScratch s{p,
                  p + cap,
                  p + 2 * cap,
                  p + 3 * cap,
                  p + 4 * cap,
                  p + 5 * cap,
                  p + 6 * cap,
                  p + 7 * cap,
                  p + 8 * cap,
                  p + 9 * cap,
                  cap};
    constexpr uint32_t kThreads = 128;
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(divide_up_by(cap, kThreads), kThreads), KernelClusInit{}, s);
    if (nDigis > 0) {
      const auto wd = make_workdiv<Acc1D>(divide_up_by(nDigis, kThreads), kThreads);
      alpaka::exec<Acc1D>(queue, wd, KernelClusCount{}, digis, nDigis, clusters, s, counters);
      alpaka::exec<Acc1D>(queue, wd, KernelClusMinMax{}, digis, nDigis, clusters, s);
      alpaka::exec<Acc1D>(queue, wd, KernelClusEdges{}, digis, nDigis, clusters, s);
    }
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nRows, kThreads), kThreads),
                        KernelClusGather{},
                        clusters,
                        s,
                        refs,
                        nRows,
                        out,
                        counters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff
