#ifndef RecoTracker_LST_plugins_alpaka_LSTInputDeviceFitKernel_h
#define RecoTracker_LST_plugins_alpaka_LSTInputDeviceFitKernel_h

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/SeedStateFit.h"

#include "LSTInputDeviceKernels.h"
#include "LSTInputDeviceTrackFit.h"

// The fit kernel of the device LST input for one number of fitted hits; each compilation unit that includes this
// header instantiates fitTracks for one fit size, or the inner-tracker-only fits, so that the fits of different sizes
// are compiled separately.
namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  // one thread per track of list `list` (its entries at `tracks`): the full fit of N hits, or with kITOnly the
  // inner-tracker-only fit, and the count of its rejections by reason. The fit's work array is in block-shared memory,
  // one slice per thread of the block (on the CPU backends a block is one thread, which reuses its slice for each of
  // its tracks): it keeps the per-thread stack small, as the CUDA driver reserves the largest stack frame of the
  // launched kernels for every resident thread.
  template <uint32_t N, bool kITOnly>
  struct FitTracksKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  FitInput const* inputs,
                                  float bField,
                                  uint32_t list,
                                  uint32_t const* tracks,
                                  Counters* counters,
                                  FittedState* states) const {
      using ::ALPAKA_ACCELERATOR_NAMESPACE::lst::fitThreadsPerBlock;
      using ::ALPAKA_ACCELERATOR_NAMESPACE::lst::fitWorkSize;
      using ::ALPAKA_ACCELERATOR_NAMESPACE::lst::kFitSharedBytesCPU;
      using ::ALPAKA_ACCELERATOR_NAMESPACE::lst::kFitSharedBytesGPU;
      constexpr uint32_t kWork = fitWorkSize(N);
      constexpr uint32_t kSlots =
          cms::alpakatools::requires_single_thread_per_block_v<Acc1D> ? 1 : fitThreadsPerBlock(N);
      static_assert(kSlots * kWork * sizeof(double) <= kFitSharedBytesGPU &&
                        kSlots * kWork * sizeof(double) <= kFitSharedBytesCPU,
                    "the work arrays of a block must fit in its static shared memory on every backend (alpaka checks "
                    "the CPU capacity only with an assertion that a release build may compile out)");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] <= kSlots));
      auto& workArrays = alpaka::declareSharedVar<double[kSlots * kWork], __COUNTER__>(acc);
      double* work = workArrays + alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] * kWork;
      const uint32_t size = counters->listSize[list];
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, size)) {
        const uint32_t track = tracks[i];
        FittedState& state = states[track];
        ::lstInputDevice::FitHit const* hits = kITOnly ? &inputs[track].itHits[0] : &inputs[track].hits[0];
        fitState<N>(acc, hits, bField, state, work);
        uint32_t* counter = nullptr;
        switch (static_cast<::lst::SeedStateStatus>(state.status)) {
          case ::lst::SeedStateStatus::kNonFinite:
            counter = kITOnly ? &counters->nITNonFinite : &counters->nNonFinite;
            break;
          case ::lst::SeedStateStatus::kChargeMismatch:
            counter = kITOnly ? &counters->nITChargeMismatch : &counters->nChargeMismatch;
            break;
          case ::lst::SeedStateStatus::kNonPositiveVariance:
            counter = kITOnly ? &counters->nITNonPositiveVariance : &counters->nNonPositiveVariance;
            break;
          default:
            break;
        }
        if (counter != nullptr)
          alpaka::atomicAdd(acc, counter, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // the fits of list `list` into states
  template <uint32_t N, bool kITOnly>
  void launchFits(Queue& queue,
                  uint32_t nTracks,
                  FitInput const* inputs,
                  float bField,
                  uint32_t list,
                  uint32_t const* lists,
                  Counters* counters,
                  FittedState* states) {
    if (nTracks == 0)
      return;
    // The list sizes are known on the device only: the grid covers every track, and the threads beyond the size of
    // the list return at once. One track per thread: the fit keeps its matrices in registers and local memory, and
    // its work array in block-shared memory, which sets the number of threads per block.
    constexpr uint32_t kThreads = ::ALPAKA_ACCELERATOR_NAMESPACE::lst::fitThreadsPerBlock(N);
    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTracks, kThreads), kThreads),
        FitTracksKernel<N, kITOnly>{},
        inputs,
        bField,
        list,
        lists + list * nTracks,
        counters,
        states);
  }

  template <uint32_t N>
  void fitTracks(Queue& queue,
                 uint32_t nTracks,
                 FitInput const* inputs,
                 float bField,
                 uint32_t const* lists,
                 Counters* counters,
                 FittedState* states) {
    static_assert(nFitHits(N) == N, "a fit size");
    launchFits<N, false>(queue, nTracks, inputs, bField, ::lstInputDevice::fitList(N), lists, counters, states);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice

#endif
