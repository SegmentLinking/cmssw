#ifndef RecoTracker_MkFitAlpaka_interface_math_MathUtils_h
#define RecoTracker_MkFitAlpaka_interface_math_MathUtils_h

// Portable copies of the small inline helpers of MkFitCore (CMSSW_20_1_0_pre2):
//   RecoTracker/MkFitCore/src/Matrix.h      hipo, sincos4
//   RecoTracker/MkFitCore/interface/Hit.h   squashPhiMinimal, getPhi, getTheta, getEta
// Same formulas (vdt calls go to the portable mkfitdev::vdt).

#include <cmath>
#include <cstdint>
#include <cstring>

#include <alpaka/core/Common.hpp>

#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/vdtMath.h"

namespace mkfitdev {

  // ---- Matrix.h
  ALPAKA_FN_HOST_ACC inline float hipo(float x, float y) { return std::sqrt(x * x + y * y); }

  ALPAKA_FN_HOST_ACC inline void sincos4(const float x, float& sin, float& cos) {
    const float x2 = x * x;
    cos = 1.f - 0.5f * x2 + 0.04166667f * x2 * x2;
    sin = x - 0.16666667f * x * x2;
  }

  // ---- cms_common_macros.h: mkfit::isFinite in CMSSW builds = edm::isFinite, a bit-pattern test immune to
  // fast-math. Forced inline: plain inline changes the CPU vectorization of the propagation at rounding level.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isFinite(float x) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, x);  // no union pun, no memcpy
    return (bits & 0x7f800000u) != 0x7f800000u;
  }

  // ---- Hit.h
  ALPAKA_FN_HOST_ACC inline float squashPhiMinimal(float phi) {
    return phi >= Const::PI ? phi - Const::TwoPI : (phi < -Const::PI ? phi + Const::TwoPI : phi);
  }

  ALPAKA_FN_HOST_ACC inline float getPhi(float x, float y) { return vdt::fast_atan2f(y, x); }

  ALPAKA_FN_HOST_ACC inline float getTheta(float r, float z) { return vdt::fast_atan2f(r, z); }

  ALPAKA_FN_HOST_ACC inline float getEta(float theta) { return -1.0f * vdt::fast_logf(vdt::fast_tanf(theta / 2.0f)); }

  ALPAKA_FN_HOST_ACC inline float getEta(float x, float y, float z) {
    const float theta = vdt::fast_atan2f(std::sqrt(x * x + y * y), z);
    return -1.0f * vdt::fast_logf(vdt::fast_tanf(theta / 2.0f));
  }
}  // namespace mkfitdev

#endif
