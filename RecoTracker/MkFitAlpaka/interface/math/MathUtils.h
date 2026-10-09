#ifndef RecoTracker_MkFitAlpaka_interface_math_MathUtils_h
#define RecoTracker_MkFitAlpaka_interface_math_MathUtils_h

// Portable copies of the small inline helpers of MkFitCore (CMSSW_20_1_0_pre2):
//   RecoTracker/MkFitCore/src/Matrix.h      hipo, hipo_sqr, sincos4
//   RecoTracker/MkFitCore/interface/Hit.h   sqr, cube, squashPhi*, getRad2, getPhi, getTheta, getEta, ...
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

  ALPAKA_FN_HOST_ACC inline float hipo_sqr(float x, float y) { return x * x + y * y; }

  ALPAKA_FN_HOST_ACC inline void sincos4(const float x, float& sin, float& cos) {
    // Had this writen with explicit division by factorial.
    // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

    const float x2 = x * x;
    cos = 1.f - 0.5f * x2 + 0.04166667f * x2 * x2;
    sin = x - 0.16666667f * x * x2;
  }

  // ---- cms_common_macros.h: mkfit::isFinite in CMSSW builds = edm::isFinite, a bit-pattern test immune to
  // fast-math. One copy for the package (prop, clean, ...).
  // ALPAKA_FN_INLINE (forced) as prop had it: plain inline changed CPU vectorization of a prop op at rounding level
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isFinite(float x) {
    const uint32_t l = __builtin_bit_cast(uint32_t, x);  // no union pun, no memcpy
    return (l & 0x7f800000u) != 0x7f800000u;
  }

  // ---- Hit.h
  template <typename T>
  ALPAKA_FN_HOST_ACC inline T sqr(T x) {
    return x * x;
  }
  template <typename T>
  ALPAKA_FN_HOST_ACC inline T cube(T x) {
    return x * x * x;
  }

  ALPAKA_FN_HOST_ACC inline float squashPhiGeneral(float phi) {
    // MkFitCore mixes in double here (0.5 * ...); kept.
    return phi - std::floor(0.5 * Const::InvPI * (phi + Const::PI)) * Const::TwoPI;
  }

  ALPAKA_FN_HOST_ACC inline float squashPhiMinimal(float phi) {
    return phi >= Const::PI ? phi - Const::TwoPI : (phi < -Const::PI ? phi + Const::TwoPI : phi);
  }

  ALPAKA_FN_HOST_ACC inline float getRad2(float x, float y) { return x * x + y * y; }

  ALPAKA_FN_HOST_ACC inline float getInvRad2(float x, float y) { return 1.0f / (x * x + y * y); }

  ALPAKA_FN_HOST_ACC inline float getPhi(float x, float y) { return vdt::fast_atan2f(y, x); }

  ALPAKA_FN_HOST_ACC inline float getTheta(float r, float z) { return vdt::fast_atan2f(r, z); }

  ALPAKA_FN_HOST_ACC inline float getEta(float r, float z) {
    return -1.0f * vdt::fast_logf(vdt::fast_tanf(getTheta(r, z) / 2.0f));
  }

  ALPAKA_FN_HOST_ACC inline float getEta(float theta) { return -1.0f * vdt::fast_logf(vdt::fast_tanf(theta / 2.0f)); }

  ALPAKA_FN_HOST_ACC inline float getEta(float x, float y, float z) {
    const float theta = vdt::fast_atan2f(std::sqrt(x * x + y * y), z);
    return -1.0f * vdt::fast_logf(vdt::fast_tanf(theta / 2.0f));
  }

  ALPAKA_FN_HOST_ACC inline float getHypot(float x, float y) { return std::sqrt(x * x + y * y); }

  ALPAKA_FN_HOST_ACC inline float getRadErr2(float x, float y, float exx, float eyy, float exy) {
    return (x * x * exx + y * y * eyy + 2.0f * x * y * exy) / getRad2(x, y);
  }

  ALPAKA_FN_HOST_ACC inline float getInvRadErr2(float x, float y, float exx, float eyy, float exy) {
    return (x * x * exx + y * y * eyy + 2.0f * x * y * exy) / cube(getRad2(x, y));
  }

  ALPAKA_FN_HOST_ACC inline float getPhiErr2(float x, float y, float exx, float eyy, float exy) {
    const float rad2 = getRad2(x, y);
    return (y * y * exx + x * x * eyy - 2.0f * x * y * exy) / (rad2 * rad2);
  }

}  // namespace mkfitdev

#endif
