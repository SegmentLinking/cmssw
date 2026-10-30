#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexCommon_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexCommon_h

// Portable (host + Alpaka device) port of RecoTracker/MkFitCore/src/Matriplex/MatriplexCommon.h
// (CMSSW_20_1_0_pre2). Intrinsics, aligned allocation and exceptions are dropped; everything is ALPAKA_FN_HOST_ACC.

#include <alpaka/core/Common.hpp>

// MPLEX_SIMD: MkFitCore writes "#pragma omp simd" (built with -fopenmp-simd) in front of the lane loops.
// Here: the same pragma for the host compiler (the CPU backends build with -fopenmp-simd), nothing for device compilers.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__clang__) || !defined(__GNUC__)
#define MPLEX_SIMD
#else
#define MPLEX_SIMD _Pragma("omp simd")
#endif

// MPLEX_ALIGN for x86-64-v3 (AVX2) is 32 bytes.
#ifndef MPLEX_ALIGN
#define MPLEX_ALIGN 32
#endif

namespace mkfitdev {
  namespace Matriplex {
    typedef int idx_t;

    // Alignment of a Matriplex object: MkFitCore alignment when the lane width fills whole vectors,
    // natural alignment otherwise (N = 1 on GPU backends: plain registers / local memory).
    template <typename T, idx_t N>
    constexpr int mplexAlign() {
      return (N * sizeof(T)) % MPLEX_ALIGN == 0 ? MPLEX_ALIGN : alignof(T);
    }

    // Offset of element (i, j) in the packed lower triangle of a symmetric D x D matrix; equal to
    // gSymOffsets[D][i * D + j] (closed form, so that no global table is needed on device).
    ALPAKA_FN_HOST_ACC constexpr idx_t symOffset(idx_t i, idx_t j) {
      return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
    }
  }  // namespace Matriplex
}  // namespace mkfitdev

#endif
