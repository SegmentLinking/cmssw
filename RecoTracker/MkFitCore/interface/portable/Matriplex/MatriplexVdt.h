#ifndef RecoTracker_MkFitCore_interface_portable_Matriplex_MatriplexVdt_h
#define RecoTracker_MkFitCore_interface_portable_Matriplex_MatriplexVdt_h

// vdt for Matriplex and the portable mkFit physics: vdt itself in host code; in device code a transliteration of
// the single-precision vdt 0.4.3 functions (same constants, same operations in the same order).

#include "RecoTracker/MkFitCore/interface/portable/Macros.h"

#ifndef MKFIT_DEVICE_COMPILATION

#include "vdt/atan2.h"
#include "vdt/cos.h"
#include "vdt/log.h"
#include "vdt/sin.h"
#include "vdt/sincos.h"
#include "vdt/sqrt.h"
#include "vdt/tan.h"

namespace Matriplex {
  namespace vdt = ::vdt;
}

#else

#include <cmath>
#include <cstdint>
#include <limits>

namespace Matriplex {
  namespace vdt {
    namespace details {

      // vdtcore_common.h
      constexpr double kPiD = 3.14159265358979323846;
      constexpr float TWOPIF = 2. * kPiD;
      constexpr float PIF = kPiD;
      constexpr float PIO2F = kPiD / 2.;
      constexpr float PIO4F = kPiD / 4.;
      constexpr float ONEOPIO4F = 4. / kPiD;
      constexpr float MAXNUMF = 3.4028234663852885981170418348451692544e38f;

      MKFIT_HOST_DEVICE inline uint32_t sp2uint32(float x) { return __float_as_uint(x); }

      MKFIT_HOST_DEVICE inline float uint322sp(uint32_t x) { return __uint_as_float(x); }

      MKFIT_HOST_DEVICE inline float spANDuint32(const float x, const uint32_t i) {
        return uint322sp(sp2uint32(x) & i);
      }
      MKFIT_HOST_DEVICE inline float spORuint32(const float x, const uint32_t i) { return uint322sp(sp2uint32(x) | i); }
      MKFIT_HOST_DEVICE inline float spXORuint32(const float x, const uint32_t i) {
        return uint322sp(sp2uint32(x) ^ i);
      }

      MKFIT_HOST_DEVICE inline uint32_t getSignMask(const float x) {
        const uint32_t mask = 0x80000000;
        return sp2uint32(x) & mask;
      }

      /// Like frexp but vectorising and the exponent is a float.
      MKFIT_HOST_DEVICE inline float getMantExponentf(const float x, float& fe) {
        uint32_t n = sp2uint32(x);
        int32_t e = (n >> 23) - 127;
        fe = e;

        // fractional part
        const uint32_t p05f = 0x3f000000;  // //sp2uint32(0.5);
        n &= 0x807fffff;                   // ~0x7f800000;
        n |= p05f;

        return uint322sp(n);
      }

      // sincos.h (float part)
      constexpr float DP1F = 0.78515625;
      constexpr float DP2F = 2.4187564849853515625e-4;
      constexpr float DP3F = 3.77489497744594108e-8;

      MKFIT_HOST_DEVICE inline float reduce2quadrant(float x, int& quad) {
        /* make argument positive */
        x = std::fabs(x);

        quad = int(ONEOPIO4F * x); /* integer part of x/PIO4 */

        quad = (quad + 1) & (~1);
        const float y = float(quad);
        // quad &=4;
        // Extended precision modular arithmetic
        return ((x - y * DP1F) - y * DP2F) - y * DP3F;
      }

      MKFIT_HOST_DEVICE inline void fast_sincosf_m45_45(const float x, float& s, float& c) {
        float z = x * x;

        s = (((-1.9515295891E-4f * z + 8.3321608736E-3f) * z - 1.6666654611E-1f) * z * x) + x;

        c = ((2.443315711809948E-005f * z - 1.388731625493765E-003f) * z + 4.166664568298827E-002f) * z * z - 0.5f * z +
            1.0f;
      }

      // tan.h (float part)
      constexpr float DP1Ftan = 0.78515625;
      constexpr float DP2Ftan = 2.4187564849853515625e-4;
      constexpr float DP3Ftan = 3.77489497744594108e-8;

      MKFIT_HOST_DEVICE inline float reduce2quadranttan(float x, int32_t& quad) {
        x = std::fabs(x);
        quad = int(ONEOPIO4F * x);  // always positive, so (int) == std::floor
        quad = (quad + 1) & (~1);
        const float y = quad;
        // Extended precision modular arithmetic
        return ((x - y * DP1Ftan) - y * DP2Ftan) - y * DP3Ftan;
      }

      // log.h (float part)
      constexpr float LOGF_UPPER_LIMIT = MAXNUMF;
      constexpr float LOGF_LOWER_LIMIT = 0;

      constexpr float PX1logf = 7.0376836292E-2f;
      constexpr float PX2logf = -1.1514610310E-1f;
      constexpr float PX3logf = 1.1676998740E-1f;
      constexpr float PX4logf = -1.2420140846E-1f;
      constexpr float PX5logf = 1.4249322787E-1f;
      constexpr float PX6logf = -1.6668057665E-1f;
      constexpr float PX7logf = 2.0000714765E-1f;
      constexpr float PX8logf = -2.4999993993E-1f;
      constexpr float PX9logf = 3.3333331174E-1f;

      MKFIT_HOST_DEVICE inline float get_log_poly(const float x) {
        float y = x * PX1logf;
        y += PX2logf;
        y *= x;
        y += PX3logf;
        y *= x;
        y += PX4logf;
        y *= x;
        y += PX5logf;
        y *= x;
        y += PX6logf;
        y *= x;
        y += PX7logf;
        y *= x;
        y += PX8logf;
        y *= x;
        y += PX9logf;
        return y;
      }

      constexpr float SQRTHF = 0.707106781186547524f;

    }  // namespace details

    //------------------------------------------------------------------------------
    // sincos.h
    MKFIT_HOST_DEVICE inline void fast_sincosf(const float xx, float& s, float& c) {
      int j;
      const float x = details::reduce2quadrant(xx, j);
      int signS = (j & 4);

      j -= 2;

      const int signC = (j & 4);
      const int poly = j & 2;

      float ls, lc;
      details::fast_sincosf_m45_45(x, ls, lc);

      //swap
      if (poly == 0) {
        const float tmp = lc;
        lc = ls;
        ls = tmp;
      }

      if (signC == 0)
        lc = -lc;
      if (signS != 0)
        ls = -ls;
      if (xx < 0)
        ls = -ls;
      c = lc;
      s = ls;
    }

    // sin.h, cos.h
    MKFIT_HOST_DEVICE inline float fast_sinf(float x) {
      float s, c;
      fast_sincosf(x, s, c);
      return s;
    }

    MKFIT_HOST_DEVICE inline float fast_cosf(float x) {
      float s, c;
      fast_sincosf(x, s, c);
      return c;
    }

    //------------------------------------------------------------------------------
    // tan.h
    MKFIT_HOST_DEVICE inline float fast_tanf(float x) {
      const uint32_t sign_mask = details::getSignMask(x);

      int32_t quad = 0;
      const float z = details::reduce2quadranttan(x, quad);

      const float zz = z * z;

      float res = z;

      if (zz > 1.0e-14f) {
        res =
            (((((9.38540185543E-3f * zz + 3.11992232697E-3f) * zz + 2.44301354525E-2f) * zz + 5.34112807005E-2f) * zz +
              1.33387994085E-1f) *
                 zz +
             3.33331568548E-1f) *
                zz * z +
            z;
      }

      // A no branching way to say: if j&2 res = -1/res. You can!!!
      quad &= 2;
      quad >>= 1;
      const int32_t alt = quad ^ 1;

      const float zeroIfXNonZero = (x == 0.f);
      res += zeroIfXNonZero;
      res = quad * (-1.f / res) + alt * res;  // one coeff is one and one is 0!

      // Again, return 0 if the input is 0
      return details::spXORuint32(res, sign_mask) * (1.f - zeroIfXNonZero);
    }

    //------------------------------------------------------------------------------
    // atan.h
    MKFIT_HOST_DEVICE inline float fast_atanf(float xx) {
      const uint32_t sign_mask = details::getSignMask(xx);

      float x = std::fabs(xx);
      const float x0 = x;
      float y = 0.0f;

      /* range reduction */
      if (x0 > 0.4142135623730950f) {  // * tan pi/8
        x = (x0 - 1.0f) / (x0 + 1.0f);
        y = details::PIO4F;
      }
      if (x0 > 2.414213562373095f) {  // tan 3pi/8
        x = -(1.0f / x0);
        y = details::PIO2F;
      }

      const float x2 = x * x;
      y += (((8.05374449538e-2f * x2 - 1.38776856032E-1f) * x2 + 1.99777106478E-1f) * x2 - 3.33329491539E-1f) * x2 * x +
           x;

      return details::spORuint32(y, sign_mask);
    }

    //------------------------------------------------------------------------------
    // atan2.h
    MKFIT_HOST_DEVICE inline float fast_atan2f(float y, float x) {
      // move in first octant
      float xx = std::fabs(x);
      float yy = std::fabs(y);
      float tmp(0.0f);
      if (yy > xx) {
        tmp = yy;
        yy = xx;
        xx = tmp;
        tmp = 1.f;
      }

      // To avoid the fpe, we protect against /0.
      const float oneIfXXZero = (xx == 0.f);

      float t = yy / (xx /*+oneIfXXZero*/);
      float z = t;
      if (t > 0.4142135623730950f)  // * tan pi/8
        z = (t - 1.0f) / (t + 1.0f);

      //printf("%e %e %e %e\n",yy,xx,t,z);
      float z2 = z * z;

      float ret =
          ((((8.05374449538e-2f * z2 - 1.38776856032E-1f) * z2 + 1.99777106478E-1f) * z2 - 3.33329491539E-1f) * z2 * z +
           z);

      // Here we put the result to 0 if xx was 0, if not nothing happens!
      ret *= (1.f - oneIfXXZero);

      // move back in place
      if (y == 0.f)
        ret = 0.f;
      if (t > 0.4142135623730950f)
        ret += details::PIO4F;
      if (tmp != 0)
        ret = details::PIO2F - ret;
      if (x < 0.f)
        ret = details::PIF - ret;
      if (y < 0.f)
        ret = -ret;

      return ret;
    }

    //------------------------------------------------------------------------------
    // log.h
    MKFIT_HOST_DEVICE inline float fast_logf(float x) {
      const float original_x = x;

      float fe;
      x = details::getMantExponentf(x, fe);

      x > details::SQRTHF ? fe += 1.f : x += x;
      x -= 1.0f;

      const float x2 = x * x;

      float res = details::get_log_poly(x);
      res *= x2 * x;

      res += -2.12194440e-4f * fe;
      res += -0.5f * x2;

      res = x + res;

      res += 0.693359375f * fe;

      if (original_x > details::LOGF_UPPER_LIMIT)
        res = std::numeric_limits<float>::infinity();
      if (original_x < details::LOGF_LOWER_LIMIT)
        res = -std::numeric_limits<float>::quiet_NaN();

      return res;
    }

    //------------------------------------------------------------------------------
    // sqrt.h
    MKFIT_HOST_DEVICE inline float fast_isqrtf_general(float x, const uint32_t ISQRT_ITERATIONS) {
      const float threehalfs = 1.5f;
      const float x2 = x * 0.5f;
      float y = x;
      uint32_t i = details::sp2uint32(y);
      i = 0x5f3759df - (i >> 1);
      y = details::uint322sp(i);
      for (uint32_t j = 0; j < ISQRT_ITERATIONS; ++j)
        y *= (threehalfs - (x2 * y * y));

      return y;
    }

    MKFIT_HOST_DEVICE inline float fast_isqrtf(float x) { return fast_isqrtf_general(x, 2); }
    MKFIT_HOST_DEVICE inline float fast_approx_isqrtf(float x) { return fast_isqrtf_general(x, 1); }
    MKFIT_HOST_DEVICE inline float isqrtf(float x) { return 1.f / std::sqrt(x); }

    // Double precision is not provided in device code (declared for the generic Matriplex members only).
    MKFIT_HOST_DEVICE double fast_isqrt(double x);
    MKFIT_HOST_DEVICE double fast_sin(double x);
    MKFIT_HOST_DEVICE double fast_cos(double x);
    MKFIT_HOST_DEVICE void fast_sincos(double x, double& s, double& c);
    MKFIT_HOST_DEVICE double fast_tan(double x);
    MKFIT_HOST_DEVICE double fast_atan2(double y, double x);

  }  // namespace vdt
}  // namespace Matriplex

#endif  // MKFIT_DEVICE_COMPILATION

#endif
