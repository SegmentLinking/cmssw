#ifndef RecoTracker_LSTCore_src_alpaka_TCSeedStateFitKernel_h
#define RecoTracker_LSTCore_src_alpaka_TCSeedStateFitKernel_h

// The fit kernel of the seed states of the track candidates for one number of fitted hits; each compilation unit that
// includes this header instantiates fitCandidates for one N, so that the fits of different sizes are compiled
// separately. Compiled only in CMSSW, as the fit (TCSeedStateFit.h).
#ifndef LST_STANDALONE

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "RecoTracker/LSTCore/interface/SeedStateFitTypes.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/SeedStateFit.h"
#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStateFit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState {

  // One thread per track candidate of the list of N hits: the Broken Line Fit of its prepared hits, its checks, and
  // the count of its rejections by reason. The hit selection ran before, in the prepare kernel, so that this kernel
  // keeps only the fit's matrices in registers and local memory, and the fit's work array in block-shared memory, one
  // slice per thread of the block (on the CPU backends a block is one thread, which reuses its slice for each of its
  // candidates): it keeps the per-thread stack small, as the CUDA driver reserves the largest stack frame of the
  // launched kernels for every resident thread.
  template <uint32_t N>
  struct FitCandidatesKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::lst::HitsBaseConst hits,
                                  FitInput const* inputs,
                                  uint32_t const* list,
                                  Counters* counters,
                                  ::lst::FittedState* states) const {
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
      const uint32_t size = counters->listSize[fitList(N)];
      const float bField = hits.B();
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, size)) {
        const uint32_t candidate = list[i];
        ::lst::FittedState& state = states[candidate];
        fitState<N>(acc, inputs[candidate].hits, bField, state, work);
        uint32_t* counter = nullptr;
        switch (static_cast<::lst::SeedStateStatus>(state.status)) {
          case ::lst::SeedStateStatus::kNonFinite:
            counter = &counters->nNonFinite;
            break;
          case ::lst::SeedStateStatus::kChargeMismatch:
            counter = &counters->nChargeMismatch;
            break;
          case ::lst::SeedStateStatus::kNonPositiveVariance:
            counter = &counters->nNonPositiveVariance;
            break;
          default:
            break;
        }
        if (counter != nullptr)
          alpaka::atomicAdd(acc, counter, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <uint32_t N>
  void fitCandidates(Queue& queue, ::lst::HitsBaseConst hits, uint32_t nTCs, Scratch const& scratch) {
    static_assert(fitSize(N) == N, "a fit size");
    if (nTCs == 0)
      return;
    // The list sizes are known on the device only: the grid covers every track candidate, and the threads beyond the
    // size of the list return at once. One track candidate per thread: the fit keeps its matrices in registers and
    // local memory, and its work array in block-shared memory, which sets the number of threads per block.
    constexpr uint32_t kThreads = fitThreadsPerBlock(N);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTCs, kThreads), kThreads),
                        FitCandidatesKernel<N>{},
                        hits,
                        scratch.inputs,
                        scratch.lists + fitList(N) * nTCs,
                        scratch.counters,
                        scratch.states);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState

#endif  // LST_STANDALONE

#endif
