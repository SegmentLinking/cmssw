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
    template <Contract ML, Contract MG>
    struct KernelOTCpe {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::OTRecHitSoA::View v,
                                    ::mkfitdev::OTCpeModule const* table,
                                    int32_t firstIndex,
                                    uint32_t n) const {
        for (uint32_t i : uniform_elements(acc, n)) {
          ::mkfitdev::OTCpeModule const& m = table[v[i].module() - firstIndex];
          const uint32_t s = v[i].strip();
          float lx, ly;
          ::mkfitdev::otcpe::localPosition<ML>(m, s & 0xffffu, s >> 16, v[i].clustSize(), lx, ly);
          float g[3];
          ::mkfitdev::otcpe::toGlobal<MG>(m, lx, ly, g);
          v[i].lx() = lx;
          v[i].ly() = ly;
          v[i].exx() = m.exx;
          v[i].eyy() = m.eyy;
          v[i].gx() = g[0];
          v[i].gy() = g[1];
          v[i].gz() = g[2];
        }
      }
    };

    struct KernelOTExpand {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::OTRecHitSoA::View v,
                                    ::mkfitdev::othits::OTDetSetSpan const* spans,
                                    uint32_t nSpans,
                                    uint16_t const* raw,
                                    ::mkfitdev::OTCpeModule const* table,
                                    int32_t firstIndex,
                                    uint32_t n) const {
        if (once_per_grid(acc))
          v.nHits() = n;
        for (uint32_t s : uniform_elements(acc, nSpans)) {
          const ::mkfitdev::othits::OTDetSetSpan sp = spans[s];
          const int32_t module = sp.mi + firstIndex;
          const uint32_t detId = table[sp.mi].detId;
          for (uint32_t k = sp.first; k < sp.first + sp.size; ++k) {
            uint16_t size;
            uint32_t strip;
            ::mkfitdev::othits::decodeCluster(raw[2 * k], raw[2 * k + 1], size, strip);
            v[k].module() = module;
            v[k].detId() = detId;
            v[k].clustSize() = size;
            v[k].strip() = strip;
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
                   uint32_t n) {
    constexpr uint32_t kBlock = 128;
    const auto wd = make_workdiv<Acc1D>(divide_up_by(std::max(nSpans, 1u), kBlock), kBlock);
    alpaka::exec<Acc1D>(queue, wd, KernelOTExpand{}, view, spans, nSpans, raw, table, firstIndex, n);
  }

  void runOTCpe(Queue& queue,
                ::mkfitdev::OTRecHitSoA::View view,
                ::mkfitdev::OTCpeModule const* table,
                int32_t firstIndex,
                uint32_t n) {
    if (n == 0)
      return;
    constexpr uint32_t kBlock = 128;
    const auto wd = make_workdiv<Acc1D>(divide_up_by(n, kBlock), kBlock);
    // the contraction the host Phase2StripCPE and Surface::toGlobal reproduce bitwise (first product fused)
    alpaka::exec<Acc1D>(
        queue, wd, KernelOTCpe<Contract::kFuseFirst, Contract::kFuseFirst>{}, view, table, firstIndex, n);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits
