// See MkFitAlpakaOTRecHitsKernels.h.
#include <algorithm>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTCpe.h"

#include "MkFitAlpakaOTRecHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits {

  using namespace cms::alpakatools;
  using ::mkfitdev::Contract;

  namespace {
    constexpr uint32_t kThreadsPerBlock = 128;

    template <Contract ML>
    struct KernelOTCpe {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::OTRecHitSoA::View view,
                                    ::mkfitdev::OTCpeModule const* table,
                                    int32_t firstIndex,
                                    uint32_t nClusters) const {
        for (uint32_t i : uniform_elements(acc, nClusters)) {
          ::mkfitdev::OTCpeModule const& cpeModule = table[view[i].module() - firstIndex];
          const uint32_t strip = view[i].strip();
          float lx, ly;
          ::mkfitdev::otcpe::localPosition<ML>(cpeModule,
                                               strip & ::mkfitdev::kOTFirstStripMask,
                                               strip >> ::mkfitdev::kOTColumnShift,
                                               view[i].clustSize(),
                                               lx,
                                               ly);
          view[i].lx() = lx;
          view[i].ly() = ly;
          view[i].exx() = cpeModule.exx;
          view[i].eyy() = cpeModule.eyy;
        }
      }
    };

    struct KernelOTExpand {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::OTRecHitSoA::View view,
                                    ::mkfitdev::othits::OTDetSetSpan const* spans,
                                    uint32_t nSpans,
                                    uint16_t const* raw,
                                    ::mkfitdev::OTCpeModule const* table,
                                    int32_t firstIndex,
                                    uint32_t nClusters) const {
        if (once_per_grid(acc))
          view.nHits() = nClusters;
        for (uint32_t i : uniform_elements(acc, nSpans)) {
          const ::mkfitdev::othits::OTDetSetSpan span = spans[i];
          const int32_t gind = span.tableIndex + firstIndex;
          for (uint32_t k = span.first; k < span.first + span.size; ++k) {
            uint16_t size;
            uint32_t strip;
            ::mkfitdev::othits::decodeCluster(raw[2 * k], raw[2 * k + 1], size, strip);
            view[k].module() = gind;
            view[k].clustSize() = size;
            view[k].strip() = strip;
          }
        }
      }
    };

  }  // namespace

  void runOTExpand(Queue& queue,
                   ::mkfitdev::OTRecHitSoA::View view,
                   ::mkfitdev::othits::OTDetSetSpan const* spans,
                   uint32_t nSpans,
                   uint16_t const* raw,
                   ::mkfitdev::OTCpeModule const* table,
                   int32_t firstIndex,
                   uint32_t nClusters) {
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(std::max(nSpans, 1u), kThreadsPerBlock), kThreadsPerBlock);
    alpaka::exec<Acc1D>(queue, workDiv, KernelOTExpand{}, view, spans, nSpans, raw, table, firstIndex, nClusters);
  }

  void runOTCpe(Queue& queue,
                ::mkfitdev::OTRecHitSoA::View view,
                ::mkfitdev::OTCpeModule const* table,
                int32_t firstIndex,
                uint32_t nClusters) {
    if (nClusters == 0)
      return;
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(nClusters, kThreadsPerBlock), kThreadsPerBlock);
    // the contraction the host Phase2StripCPE reproduces bitwise (first product fused)
    alpaka::exec<Acc1D>(queue, workDiv, KernelOTCpe<Contract::kFuseFirst>{}, view, table, firstIndex, nClusters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits
