// Per-event status product filling: see interface/alpaka/StatusCollect.h.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/StatusCollect.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  namespace {
    struct KernelCollectStatus {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::mkfitdev::MkFitStatus* status, StatusSources src) const {
        // one thread: a handful of scalar loads, fixed order (deterministic)
        if (cms::alpakatools::once_per_grid(acc)) {
          for (int i = 0; i < src.n; ++i) {
            const StatusSource& x = src.s[i];
            const uint32_t v = x.u32 ? *x.u32 : (x.i32 ? static_cast<uint32_t>(*x.i32) : 0u);
            status->counter[x.index] += v;
          }
        }
      }
    };

    // alpaka::memset does not take a 0-dim (PortableObject) buffer on CUDA: zero with a one-thread kernel
    struct KernelZeroStatus {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::mkfitdev::MkFitStatus* status) const {
        if (cms::alpakatools::once_per_grid(acc))
          for (int i = 0; i < ::mkfitdev::kMaxStatusCounters; ++i)
            status->counter[i] = 0;
      }
    };

    struct KernelAddStatus {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::mkfitdev::MkFitStatus* status, int index, uint32_t v) const {
        if (cms::alpakatools::once_per_grid(acc))
          status->counter[index] += v;
      }
    };
  }  // namespace

  void zeroStatus(Queue& queue, MkFitStatusDeviceObject& status) {
    const auto wd = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, wd, KernelZeroStatus{}, status.data());
  }

  void collectStatus(Queue& queue, MkFitStatusDeviceObject& status, StatusSources const& sources) {
    if (sources.n == 0)
      return;
    const auto wd = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, wd, KernelCollectStatus{}, status.data(), sources);
  }

  void addStatus(Queue& queue, MkFitStatusDeviceObject& status, int index, uint32_t value) {
    if (index < 0 || index >= ::mkfitdev::kMaxStatusCounters || value == 0)
      return;
    const auto wd = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, wd, KernelAddStatus{}, status.data(), index, value);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev
