#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederDevice.h"
#include "MkSeederFitKernels.h"

// The launches of the seed fits (MkSeederFitKernels.h): the quads' fit, the refits and the beam-line states.

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  using namespace cms::alpakatools;

  namespace {
    constexpr uint32_t kThreads = 64;
    // fit::kLanes seeds per work item
    WorkDiv1D fitWorkDiv(uint32_t n) {
      return make_workdiv<Acc1D>(divide_up_by(fit::laneGroups(n), kThreads), kThreads);
    }
  }  // namespace

  void fitSeedHits(Queue &queue,
                   mkseeder::FitInputs const &fit,
                   mkseeder::KeptSeedsConstView seeds,
                   mkseeder::SeedStatesView out,
                   uint32_t n,
                   const uint32_t *skip) {
    if (n == 0)
      return;
    if (!skip) {
      const uint32_t *all = nullptr;
      alpaka::exec<Acc1D>(queue, fitWorkDiv(n), fit::KernelFitQuads{}, fit, seeds, out, n, all, all);
      return;
    }
    // the seeds to fit as a list (the skipped ones out of the lane groups)
    auto flags = make_device_buffer<uint32_t[]>(queue, n);
    auto pos = make_device_buffer<uint32_t[]>(queue, n);
    auto rows = make_device_buffer<uint32_t[]>(queue, n);
    auto nRows = make_device_buffer<uint32_t>(queue);
    const auto wn = make_workdiv<Acc1D>(divide_up_by(n, 128u), 128u);
    alpaka::exec<Acc1D>(queue, wn, fit::KernelFitRowFlags{}, skip, n, flags.data(), out);
    prefixScan<Acc1D>(queue, flags.data(), pos.data(), n);
    alpaka::exec<Acc1D>(queue, wn, fit::KernelFitRowList{}, flags.data(), pos.data(), n, rows.data(), nRows.data());
    alpaka::exec<Acc1D>(queue,
                        fitWorkDiv(n),
                        fit::KernelFitQuads{},
                        fit,
                        seeds,
                        out,
                        n,
                        static_cast<const uint32_t *>(rows.data()),
                        static_cast<const uint32_t *>(nRows.data()));
  }

  void markDroppedProbes(
      Queue &queue, mkseeder::SeedHitsConstView probe, const uint32_t *nProbes, uint32_t cap, uint32_t *skip) {
    if (cap == 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(cap, 128u), 128u),
                        fit::KernelMarkDroppedProbes{},
                        probe,
                        nProbes,
                        cap,
                        skip);
  }

  void refitSeeds(Queue &queue,
                  mkseeder::FitInputs const &fit,
                  mkseeder::SeedHitsView seeds,
                  mkseeder::SeedHitListView backup,
                  mkseeder::SeedStatesView states,
                  uint32_t n,
                  const uint32_t *count) {
    const auto wd = fitWorkDiv(n);
    alpaka::exec<Acc1D>(queue, wd, fit::KernelRefitSeeds{}, fit, seeds, backup, states, n, count);
    alpaka::exec<Acc1D>(
        queue, wd, fit::KernelComputeBeamlineStates{}, fit, mkseeder::SeedHitsConstView(seeds), states, n, count);
  }

  void fitMergedSeeds(Queue &queue,
                      mkseeder::FitInputs const &fit,
                      mkseeder::SeedHitsConstView seeds,
                      mkseeder::SeedStatesView out,
                      uint32_t n,
                      bool beamline) {
    if (beamline)
      alpaka::exec<Acc1D>(queue, fitWorkDiv(n), fit::KernelRefitBl{}, fit, seeds, out, n);
    else
      alpaka::exec<Acc1D>(queue, fitWorkDiv(n), fit::KernelFitSeeds{}, fit, seeds, out, n);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder
