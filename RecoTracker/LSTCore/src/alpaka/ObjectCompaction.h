#ifndef RecoTracker_LSTCore_src_alpaka_ObjectCompaction_h
#define RecoTracker_LSTCore_src_alpaka_ObjectCompaction_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"

// Shrink a collection allocated at the counting-kernel size to the produced objects. Modules are laid out in module
// order and rows keep their order within a module (on the serial CPU backend the global order is unchanged).

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Single block. exactIndices[m] = exclusive prefix sum (in module order) of nObjects over the modules that had a
  // loose allocation (looseIndices[m] != -1), -1 otherwise; exactIndices[nLowerModules] = total.
  struct ExactModuleOffsetsKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  unsigned int nLowerModules,
                                  unsigned int const* __restrict__ nObjects,
                                  int const* __restrict__ looseIndices,
                                  int* __restrict__ exactIndices) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));
      constexpr unsigned int kMaxThreads = 1024;
      auto& partial = alpaka::declareSharedVar<unsigned int[kMaxThreads], __COUNTER__>(acc);
      const unsigned int nThreads = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const unsigned int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      ALPAKA_ASSERT_ACC(nThreads <= kMaxThreads);

      const unsigned int chunk = (nLowerModules + nThreads - 1) / nThreads;
      const unsigned int begin = tid * chunk < nLowerModules ? tid * chunk : nLowerModules;
      const unsigned int end = begin + chunk < nLowerModules ? begin + chunk : nLowerModules;
      unsigned int sum = 0;
      for (unsigned int m = begin; m < end; ++m) {
        if (looseIndices[m] != -1)
          sum += nObjects[m];
      }
      partial[tid] = sum;
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        unsigned int running = 0;
        for (unsigned int t = 0; t < nThreads; ++t) {
          const unsigned int v = partial[t];
          partial[t] = running;
          running += v;
        }
        exactIndices[nLowerModules] = static_cast<int>(running);
      }
      alpaka::syncBlockThreads(acc);
      unsigned int offset = partial[tid];
      for (unsigned int m = begin; m < end; ++m) {
        if (looseIndices[m] == -1) {
          exactIndices[m] = -1;
        } else {
          exactIndices[m] = static_cast<int>(offset);
          offset += nObjects[m];
        }
      }
    }
  };

  // Copy the produced rows of every eligible module from the loose collection to the exact one, and copy the
  // per-module occupancy block unchanged.
  struct CompactModuleObjects {
    template <typename TConst, typename TView, typename TOccConst, typename TOccView>
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  unsigned int nLowerModules,
                                  uint16_t const* __restrict__ eligibleModules,
                                  unsigned int nEligibleModules,
                                  unsigned int const* __restrict__ nObjects,
                                  int const* __restrict__ looseIndices,
                                  int const* __restrict__ exactIndices,
                                  TConst src,
                                  TView dst,
                                  TOccConst srcOcc,
                                  TOccView dstOcc) const {
      for (unsigned int m : cms::alpakatools::uniform_elements(acc, nLowerModules)) {
        dstOcc[m] = srcOcc[m];
      }
      for (unsigned int iter : cms::alpakatools::independent_groups(acc, nEligibleModules)) {
        const uint16_t m = eligibleModules[iter];
        const int loose = looseIndices[m];
        if (loose == -1)
          continue;
        const int exact = exactIndices[m];
        for (unsigned int k : cms::alpakatools::independent_group_elements(acc, nObjects[m])) {
          dst[exact + k] = src[loose + k];
        }
      }
    }
  };

  // After CompactModuleObjects: point the module ranges at the exact rows.
  struct SetExactModuleIndices {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  unsigned int nLowerModules,
                                  unsigned int const* __restrict__ nObjects,
                                  int const* __restrict__ exactIndices,
                                  int* __restrict__ moduleIndices,
                                  int* __restrict__ moduleOccupancy) const {
      for (unsigned int m : cms::alpakatools::uniform_elements(acc, nLowerModules)) {
        moduleIndices[m] = exactIndices[m];
        if (exactIndices[m] != -1)
          moduleOccupancy[m] = static_cast<int>(nObjects[m]);
      }
    }
  };

  // Copy the first n rows (objects filled through one global atomic counter) to a collection of size n.
  struct CompactPrefixObjects {
    template <typename TConst, typename TView>
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, unsigned int n, TConst src, TView dst) const {
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, n)) {
        dst[i] = src[i];
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
