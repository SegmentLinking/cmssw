#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexCommon_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexCommon_h

// Portable (host + Alpaka device) port of RecoTracker/MkFitCore/src/Matriplex/MatriplexCommon.h
// (CMSSW_20_1_0_pre2). Intrinsics, aligned allocation and exceptions are dropped; everything is ALPAKA_FN_HOST_ACC.

#include <algorithm>
#include <cmath>

#include <alpaka/core/Common.hpp>

// MPLEX_SIMD: MkFitCore writes "#pragma omp simd" (built with -fopenmp-simd) in front of the lane loops.
// Here: the same pragma for the host compiler (the CPU backends build with -fopenmp-simd), nothing for device compilers.
// MPLEX_UNROLL: complete unrolling of a short fixed loop inside a lane loop (up to the trip count in the pragma; a
// longer loop is unrolled only partly), so that the host compiler vectorizes the lane loop as one body; nothing for
// device compilers, which unroll such loops by themselves.
#define MPLEX_PRAGMA(x) _Pragma(#x)
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__clang__) || !defined(__GNUC__)
#define MPLEX_SIMD
#define MPLEX_UNROLL
#else
#define MPLEX_SIMD _Pragma("omp simd")
#define MPLEX_UNROLL MPLEX_PRAGMA(GCC unroll 16)
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

    namespace internal {
      // MkFitCore's Matriplex::internal::atan2_poly: atan2 without branches (one polynomial for atan on the unit
      // interval, the octant by selects), so that a loop over it vectorises, unlike vdt::fast_atan2f.
      ALPAKA_FN_HOST_ACC inline float atan2_poly(float y, float x) {
        constexpr float kPio2Hi = 0x1.921fb6p0f, kPio2Lo = -0x1.777a5cp-25f, kPiHi = 0x1.921fb6p1f,
                        kPiLo = -0x1.777a5cp-24f;
        const float ax = std::fabs(x), ay = std::fabs(y);
        const float mx = std::max(ax, ay), mn = std::min(ax, ay);
        const float t = mn / (mx == 0 ? 1.f : mx), t2 = t * t;  // (0, 0): 0
        const float p =
            -0x1.5554dcp-2f +
            t2 * (0x1.9978f4p-3f +
                  t2 * (-0x1.230adcp-3f +
                        t2 * (0x1.b4e12ap-4f +
                              t2 * (-0x1.3556b6p-4f +
                                    t2 * (0x1.61fddap-5f + t2 * (-0x1.0c2c14p-6f + t2 * 0x1.7ed232p-9f))))));
        float a = t + t * t2 * p;
        const bool swap = ay > ax, xneg = std::signbit(x);
        const float bhi = swap ? kPio2Hi : (xneg ? kPiHi : 0.f), blo = swap ? kPio2Lo : (xneg ? kPiLo : 0.f);
        a = (blo + (swap != xneg ? -a : a)) + bhi;
        return std::copysign(a, y);
      }
    }  // namespace internal
  }  // namespace Matriplex
}  // namespace mkfitdev

#endif
