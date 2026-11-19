#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederLibm_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederLibm_h

// libm::atan2 / asin / tan: the seeder's arc functions with the same values on every backend.
//  - CPU backends: atan2 and asin are glibc's float atan2f / asinf (the fdlibm algorithms of glibc 2.28, libm-2.28:
//    sysdeps/ieee754/flt-32 e_atan2f.c, s_atanf.c, e_asinf.c, with the constants of that libm) inline, operation by
//    operation, with every product rounded on its own (rounding barrier, see nc()): the same results as the library
//    calls (checked on every float for atanf and asinf, and on 2e8 random pairs for atan2f) without the call; tan is
//    the library's tanf;
//  - GPU backends: alpaka::math::atan2 / asin (the device library), and tan as the port of glibc's tanf below with
//    every operation rounded on its own, so that it equals the host's for |x| <= 2^7 pi/2.
// The ports are portable C++ (no instruction-set assembly beyond the rounding barrier of nc()).

#include <bit>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "MkSeederBackend.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::libm {

  // A value the compiler must not fuse into a multiply-add with its consumer: the glibc ports are written with
  // each product and sum rounded on its own. Only a target with a hardware fused multiply-add can contract
  // (__FP_FAST_FMAF); on the others the barrier is not needed.
  inline float nc(float p) {
#if defined(__FP_FAST_FMAF)
#if defined(MKSEEDER_PORTABLE_BARRIER)  // the portable fallback on any target (a test option)
    volatile float v = p;
    p = v;
#elif defined(__x86_64__) || defined(__i386__)
    asm("" : "+x"(p));
#elif defined(__aarch64__)
    asm("" : "+w"(p));
#elif defined(__powerpc64__) || defined(__riscv)
    asm("" : "+f"(p));
#else
    volatile float v = p;  // portable, slower
    p = v;
#endif
#endif
    return p;
  }
  inline uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }
  inline float fromb(uint32_t u) { return std::bit_cast<float>(u); }

  namespace glibc {
    inline float atanf(float x) {
      static const float atanhi[] = {4.6364760399e-01f, 7.8539812565e-01f, 9.8279368877e-01f, 1.5707962513e+00f};
      static const float atanlo[] = {5.0121582440e-09f, 3.7748947079e-08f, 3.4473217170e-08f, 7.5497894159e-08f};
      static const float aT[] = {3.3333334327e-01f,
                                 -2.0000000298e-01f,
                                 1.4285714924e-01f,
                                 -1.1111110449e-01f,
                                 9.0908870101e-02f,
                                 -7.6918758452e-02f,
                                 6.6610731184e-02f,
                                 -5.8335702866e-02f,
                                 4.9768779427e-02f,
                                 -3.6531571299e-02f,
                                 1.6285819933e-02f};
      const int32_t hx = (int32_t)bits(x), ix = hx & 0x7fffffff;
      int id;
      if (ix >= 0x4c000000) {
        if (ix > 0x7f800000)
          return x + x;
        return hx > 0 ? atanhi[3] + atanlo[3] : -atanhi[3] - atanlo[3];
      }
      if (ix < 0x3ee00000) {
        if (ix < 0x31000000)
          return x;
        id = -1;
      } else {
        x = std::fabs(x);
        if (ix < 0x3f980000) {
          if (ix < 0x3f300000)
            id = 0, x = (nc(2.0f * x) - 1.0f) / (2.0f + x);
          else
            id = 1, x = (x - 1.0f) / (x + 1.0f);
        } else {
          if (ix < 0x401c0000)
            id = 2, x = (x - 1.5f) / (1.0f + nc(1.5f * x));
          else
            id = 3, x = -1.0f / x;
        }
      }
      const float z = nc(x * x), w = nc(z * z);
      const float s1 =
          nc(z * (aT[0] + nc(w * (aT[2] + nc(w * (aT[4] + nc(w * (aT[6] + nc(w * (aT[8] + nc(w * aT[10])))))))))));
      const float s2 = nc(w * (aT[1] + nc(w * (aT[3] + nc(w * (aT[5] + nc(w * (aT[7] + nc(w * aT[9])))))))));
      if (id < 0)
        return x - nc(x * (s1 + s2));
      const float zz = atanhi[id] - ((nc(x * (s1 + s2)) - atanlo[id]) - x);
      return hx < 0 ? -zz : zz;
    }
    inline float atan2f(float y, float x) {
      const float tiny = 1.0e-30f, pi_o_2 = 1.5707963705e+00f, pi = 3.1415927410e+00f, pi_lo = -8.7422776573e-08f;
      const int32_t hx = (int32_t)bits(x), ix = hx & 0x7fffffff, hy = (int32_t)bits(y), iy = hy & 0x7fffffff;
      if (ix > 0x7f800000 || iy > 0x7f800000)
        return x + y;
      if (hx == 0x3f800000)
        return atanf(y);
      const int m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
      if (iy == 0) {
        switch (m) {
          case 0:
          case 1:
            return y;
          case 2:
            return pi + tiny;
          default:
            return -pi - tiny;
        }
      }
      if (ix == 0)
        return hy < 0 ? -pi_o_2 - tiny : pi_o_2 + tiny;
      if (ix == 0x7f800000) {
        const float pi_o_4 = 7.8539818525e-01f;
        if (iy == 0x7f800000) {
          switch (m) {
            case 0:
              return pi_o_4 + tiny;
            case 1:
              return -pi_o_4 - tiny;
            case 2:
              return 3.0f * pi_o_4 + tiny;
            default:
              return -3.0f * pi_o_4 - tiny;
          }
        } else {
          switch (m) {
            case 0:
              return 0.0f;
            case 1:
              return -0.0f;
            case 2:
              return pi + tiny;
            default:
              return -pi - tiny;
          }
        }
      }
      if (iy == 0x7f800000)
        return hy < 0 ? -pi_o_2 - tiny : pi_o_2 + tiny;
      const int k = (iy - ix) >> 23;
      float z;
      if (k > 26)
        z = pi_o_2 + nc(0.5f * pi_lo);
      else if (hx < 0 && k < -26)
        z = 0.0f;
      else
        z = atanf(std::fabs(y / x));
      switch (m) {
        case 0:
          return z;
        case 1:
          return fromb(bits(z) ^ 0x80000000u);
        case 2:
          return pi - (z - pi_lo);
        default:
          return (z - pi_lo) - pi;
      }
    }
    inline float asinf(float x) {
      const float pio2_hi = 0x1.921fb6p+0f, pio2_lo = -0x1.777a5cp-25f, pio4_hi = 0x1.921fb6p-1f;
      const float p0 = 0x1.5555c8p-3f, p1 = 0x1.3301e4p-4f, p2 = 0x1.747e4ap-5f, p3 = 0x1.8c283cp-6f,
                  p4 = 0x1.596d28p-5f;
      const int32_t hx = (int32_t)bits(x), ix = hx & 0x7fffffff;
      if (ix == 0x3f800000)
        return nc(x * pio2_hi) + nc(x * pio2_lo);
      if (ix > 0x3f800000)
        return (x - x) / (x - x);
      if (ix < 0x3f000000) {
        if (ix < 0x32000000)
          return x;
        const float t = nc(x * x);
        const float w = nc(t * (p0 + nc(t * (p1 + nc(t * (p2 + nc(t * (p3 + nc(t * p4)))))))));
        return x + nc(x * w);
      }
      float w = 1.0f - std::fabs(x);
      float t = nc(w * 0.5f);
      float p = nc(t * (p0 + nc(t * (p1 + nc(t * (p2 + nc(t * (p3 + nc(t * p4)))))))));
      const float s = std::sqrt(t);
      if (ix >= 0x3F79999A) {
        t = pio2_hi - (nc(2.0f * (s + nc(s * p))) - pio2_lo);
      } else {
        w = fromb(bits(s) & 0xfffff000u);
        const float c = (t - nc(w * w)) / (s + w);
        const float r = p;
        p = nc(nc(2.0f * s) * r) - (pio2_lo - nc(2.0f * c));
        const float q = pio4_hi - nc(2.0f * w);
        t = pio4_hi - (p - q);
      }
      return hx > 0 ? t : -t;
    }
  }  // namespace glibc

  // tan: glibc's float tanf (libm-2.28: sysdeps/ieee754/flt-32 s_tanf.c, k_tanf.c and the |x| <= 2^7 pi/2
  // part of e_rem_pio2f.c), operation by operation with each product and sum rounded on its own (no fused
  // multiply-add): on the device the same values as the host's tanf (checked on every float of |x| <= 2^7 pi/2 on the
  // GPU against the host's libm).
  template <typename TOps>
  ALPAKA_FN_HOST_ACC inline float kernelTanf(float x, float y, int iy) {
    constexpr float one = 1.0f, pio4 = 7.8539812565e-01f, pio4lo = 3.7748947079e-08f;
    constexpr float T[] = {3.3333334327e-01f,
                           1.3333334029e-01f,
                           5.3968254477e-02f,
                           2.1869488060e-02f,
                           8.8632395491e-03f,
                           3.5920790397e-03f,
                           1.4562094584e-03f,
                           5.8804126456e-04f,
                           2.4646313977e-04f,
                           7.8179444245e-05f,
                           7.1407252108e-05f,
                           -1.8558637748e-05f,
                           2.5907305826e-05f};
    using O = TOps;
    const int32_t hx = (int32_t)O::bits(x), ix = hx & 0x7fffffff;
    if (ix < 0x39000000) {  // |x| < 2^-13
      if ((int)x == 0) {
        if ((ix | (iy + 1)) == 0)
          return one / std::fabs(x);
        else if (iy == 1)
          return x;
        else
          return -one / x;
      }
    }
    if (ix >= 0x3f2ca140) {  // |x| >= 0.6744
      if (hx < 0)
        x = -x, y = -y;
      const float z = O::sub(pio4, x), w = O::sub(pio4lo, y);
      x = O::add(z, w), y = 0.0f;
      if (std::fabs(x) < 0x1p-13f)
        return O::mul(float((1 - ((hx >> 30) & 2)) * iy), O::sub(1.0f, O::mul(float(2 * iy), x)));
    }
    const float z = O::mul(x, x), w = O::mul(z, z);
    float r = O::add(
        T[1],
        O::mul(
            w,
            O::add(T[3], O::mul(w, O::add(T[5], O::mul(w, O::add(T[7], O::mul(w, O::add(T[9], O::mul(w, T[11]))))))))));
    float v = O::mul(
        z,
        O::add(
            T[2],
            O::mul(
                w,
                O::add(T[4],
                       O::mul(w, O::add(T[6], O::mul(w, O::add(T[8], O::mul(w, O::add(T[10], O::mul(w, T[12])))))))))));
    float s = O::mul(z, x);
    r = O::add(y, O::mul(z, O::add(O::mul(s, O::add(r, v)), y)));
    r = O::add(r, O::mul(T[0], s));
    const float ww = O::add(x, r);
    if (ix >= 0x3f2ca140) {
      v = (float)iy;
      return O::mul((float)(1 - ((hx >> 30) & 2)),
                    O::sub(v, O::mul(2.0f, O::sub(x, O::sub(O::mul(ww, ww) / O::add(ww, v), r)))));
    }
    if (iy == 1)
      return ww;
    // -1 / (x + r) accurately
    const float zz = O::fromb(O::bits(ww) & 0xfffff000u);
    v = O::sub(r, O::sub(zz, x));
    const float a = -1.0f / ww;
    const float t = O::fromb(O::bits(a) & 0xfffff000u);
    s = O::add(1.0f, O::mul(t, zz));
    return O::add(t, O::mul(a, O::add(s, O::mul(t, v))));
  }
  // tanf for |x| <= 2^7 pi/2 (else NaN; the caller checks)
  template <typename TOps>
  ALPAKA_FN_HOST_ACC inline float tanfMedium(float x) {
    constexpr float half = 5.0000000000e-01f, invpio2 = 6.3661980629e-01f, pio2_1 = 1.5707855225e+00f,
                    pio2_1t = 1.0804334124e-05f, pio2_2 = 1.0804273188e-05f, pio2_2t = 6.0770999344e-11f,
                    pio2_3 = 6.0770943833e-11f, pio2_3t = 6.1232342629e-17f;
    constexpr int32_t npio2_hw[] = {0x3fc90f00, 0x40490f00, 0x4096cb00, 0x40c90f00, 0x40fb5300, 0x4116cb00, 0x412fed00,
                                    0x41490f00, 0x41623100, 0x417b5300, 0x418a3a00, 0x4196cb00, 0x41a35c00, 0x41afed00,
                                    0x41bc7e00, 0x41c90f00, 0x41d5a000, 0x41e23100, 0x41eec200, 0x41fb5300, 0x4203f200,
                                    0x420a3a00, 0x42108300, 0x4216cb00, 0x421d1400, 0x42235c00, 0x4229a500, 0x422fed00,
                                    0x42363600, 0x423c7e00, 0x4242c700, 0x42490f00};
    using O = TOps;
    const int32_t hx = (int32_t)O::bits(x), ix = hx & 0x7fffffff;
    if (ix <= 0x3f490fda)  // |x| ~< pi/4
      return kernelTanf<O>(x, 0.0f, 1);
    if (ix >= 0x7f800000)
      return x - x;
    // __ieee754_rem_pio2f
    float y0, y1;
    int32_t n;
    if (ix <= 0x3f490fd8) {
      y0 = x, y1 = 0, n = 0;
    } else if (ix < 0x4016cbe4) {  // |x| < 3pi/4
      if (hx > 0) {
        float z = O::sub(x, pio2_1);
        if ((ix & 0xffffffc0) != 0x3fc90fc0) {
          y0 = O::sub(z, pio2_1t), y1 = O::sub(O::sub(z, y0), pio2_1t);
        } else {
          z = O::sub(z, pio2_2);
          y0 = O::sub(z, pio2_2t), y1 = O::sub(O::sub(z, y0), pio2_2t);
        }
        n = 1;
      } else {
        float z = O::add(x, pio2_1);
        if ((ix & 0xffffffc0) != 0x3fc90fc0) {
          y0 = O::add(z, pio2_1t), y1 = O::add(O::sub(z, y0), pio2_1t);
        } else {
          z = O::add(z, pio2_2);
          y0 = O::add(z, pio2_2t), y1 = O::add(O::sub(z, y0), pio2_2t);
        }
        n = -1;
      }
    } else if (ix <= 0x43490f80) {  // |x| ~<= 2^7 pi/2
      float t = std::fabs(x);
      n = (int32_t)O::add(O::mul(t, invpio2), half);
      const float fn = (float)n;
      float r = O::sub(t, O::mul(fn, pio2_1));
      float w = O::mul(fn, pio2_1t);
      if (n < 32 && (int32_t)(ix & 0xffffff00) != npio2_hw[n - 1]) {
        y0 = O::sub(r, w);
      } else {
        const int32_t j = ix >> 23;
        y0 = O::sub(r, w);
        int32_t i = j - ((O::bits(y0) >> 23) & 0xff);
        if (i > 8) {
          t = r;
          w = O::mul(fn, pio2_2);
          r = O::sub(t, w);
          w = O::sub(O::mul(fn, pio2_2t), O::sub(O::sub(t, r), w));
          y0 = O::sub(r, w);
          i = j - ((O::bits(y0) >> 23) & 0xff);
          if (i > 25) {
            t = r;
            w = O::mul(fn, pio2_3);
            r = O::sub(t, w);
            w = O::sub(O::mul(fn, pio2_3t), O::sub(O::sub(t, r), w));
            y0 = O::sub(r, w);
          }
        }
      }
      y1 = O::sub(O::sub(r, y0), w);
      if (hx < 0)
        y0 = -y0, y1 = -y1, n = -n;
    } else
      return NAN;
    return kernelTanf<O>(y0, y1, 1 - ((n & 1) << 1));
  }

  // each operation rounded on its own, on the host: the barrier of nc()
  struct RoundedOps {
    static float mul(float a, float b) { return nc(a * b); }
    static float add(float a, float b) { return nc(a + b); }
    static float sub(float a, float b) { return nc(a - b); }
    static float div(float a, float b) { return a / b; }
    static float sqrt(float a) { return std::sqrt(a); }
    static float fma(float a, float b, float c) { return std::fma(a, b, c); }
    static uint32_t bits(float x) { return libm::bits(x); }
    static float fromb(uint32_t u) { return libm::fromb(u); }
  };
#if defined(__CUDACC__) || defined(__HIPCC__)
  // each operation rounded on its own: the intrinsics are never contracted into fused multiply-adds
  struct DeviceOps {
    ALPAKA_FN_ACC static float mul(float a, float b) { return __fmul_rn(a, b); }
    ALPAKA_FN_ACC static float add(float a, float b) { return __fadd_rn(a, b); }
    ALPAKA_FN_ACC static float sub(float a, float b) { return __fsub_rn(a, b); }
    ALPAKA_FN_ACC static float div(float a, float b) { return __fdiv_rn(a, b); }
    ALPAKA_FN_ACC static float sqrt(float a) { return __fsqrt_rn(a); }
    ALPAKA_FN_ACC static float fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
    ALPAKA_FN_ACC static uint32_t bits(float x) { return __float_as_uint(x); }
    ALPAKA_FN_ACC static float fromb(uint32_t u) { return __uint_as_float(u); }
  };
#else
  using DeviceOps = RoundedOps;  // not used: only a CUDA / HIP compilation instantiates the GPU branch
#endif

  // glibc's atanf / atan2f (glibc::atanf / atan2f above) with the operations of TOps (each product, sum and quotient
  // rounded on its own): with DeviceOps on a GPU the same values as glibc::atan2f on the host (the stage-c network's
  // azimuths, MkSeederStageCNet.h)
  template <typename TOps>
  ALPAKA_FN_HOST_ACC inline float atanfOps(float x) {
    constexpr float atanhi[] = {4.6364760399e-01f, 7.8539812565e-01f, 9.8279368877e-01f, 1.5707962513e+00f};
    constexpr float atanlo[] = {5.0121582440e-09f, 3.7748947079e-08f, 3.4473217170e-08f, 7.5497894159e-08f};
    constexpr float aT[] = {3.3333334327e-01f,
                            -2.0000000298e-01f,
                            1.4285714924e-01f,
                            -1.1111110449e-01f,
                            9.0908870101e-02f,
                            -7.6918758452e-02f,
                            6.6610731184e-02f,
                            -5.8335702866e-02f,
                            4.9768779427e-02f,
                            -3.6531571299e-02f,
                            1.6285819933e-02f};
    using O = TOps;
    const int32_t hx = (int32_t)O::bits(x), ix = hx & 0x7fffffff;
    int id;
    if (ix >= 0x4c000000) {
      if (ix > 0x7f800000)
        return O::add(x, x);
      return hx > 0 ? O::add(atanhi[3], atanlo[3]) : O::sub(-atanhi[3], atanlo[3]);
    }
    if (ix < 0x3ee00000) {
      if (ix < 0x31000000)
        return x;
      id = -1;
    } else {
      x = std::fabs(x);
      if (ix < 0x3f980000) {
        if (ix < 0x3f300000)
          id = 0, x = O::div(O::sub(O::mul(2.0f, x), 1.0f), O::add(2.0f, x));
        else
          id = 1, x = O::div(O::sub(x, 1.0f), O::add(x, 1.0f));
      } else {
        if (ix < 0x401c0000)
          id = 2, x = O::div(O::sub(x, 1.5f), O::add(1.0f, O::mul(1.5f, x)));
        else
          id = 3, x = O::div(-1.0f, x);
      }
    }
    const float z = O::mul(x, x), w = O::mul(z, z);
    const float s1 = O::mul(
        z,
        O::add(
            aT[0],
            O::mul(
                w,
                O::add(
                    aT[2],
                    O::mul(w, O::add(aT[4], O::mul(w, O::add(aT[6], O::mul(w, O::add(aT[8], O::mul(w, aT[10])))))))))));
    const float s2 = O::mul(
        w,
        O::add(aT[1], O::mul(w, O::add(aT[3], O::mul(w, O::add(aT[5], O::mul(w, O::add(aT[7], O::mul(w, aT[9])))))))));
    if (id < 0)
      return O::sub(x, O::mul(x, O::add(s1, s2)));
    const float zz = O::sub(atanhi[id], O::sub(O::sub(O::mul(x, O::add(s1, s2)), atanlo[id]), x));
    return hx < 0 ? -zz : zz;
  }
  template <typename TOps>
  ALPAKA_FN_HOST_ACC inline float atan2fOps(float y, float x) {
    constexpr float tiny = 1.0e-30f, pi_o_2 = 1.5707963705e+00f, pi = 3.1415927410e+00f, pi_lo = -8.7422776573e-08f,
                    pi_o_4 = 7.8539818525e-01f;
    using O = TOps;
    const int32_t hx = (int32_t)O::bits(x), ix = hx & 0x7fffffff, hy = (int32_t)O::bits(y), iy = hy & 0x7fffffff;
    if (ix > 0x7f800000 || iy > 0x7f800000)
      return O::add(x, y);
    if (hx == 0x3f800000)
      return atanfOps<O>(y);
    const int m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
    if (iy == 0) {
      if (m < 2)
        return y;
      return m == 2 ? O::add(pi, tiny) : O::sub(-pi, tiny);
    }
    if (ix == 0)
      return hy < 0 ? O::sub(-pi_o_2, tiny) : O::add(pi_o_2, tiny);
    if (ix == 0x7f800000) {
      if (iy == 0x7f800000) {
        switch (m) {
          case 0:
            return O::add(pi_o_4, tiny);
          case 1:
            return O::sub(-pi_o_4, tiny);
          case 2:
            return O::add(O::mul(3.0f, pi_o_4), tiny);
          default:
            return O::sub(O::mul(-3.0f, pi_o_4), tiny);
        }
      }
      switch (m) {
        case 0:
          return 0.0f;
        case 1:
          return -0.0f;
        case 2:
          return O::add(pi, tiny);
        default:
          return O::sub(-pi, tiny);
      }
    }
    if (iy == 0x7f800000)
      return hy < 0 ? O::sub(-pi_o_2, tiny) : O::add(pi_o_2, tiny);
    const int k = (iy - ix) >> 23;
    float z;
    if (k > 26)
      z = O::add(pi_o_2, O::mul(0.5f, pi_lo));
    else if (hx < 0 && k < -26)
      z = 0.0f;
    else
      z = atanfOps<O>(std::fabs(O::div(y, x)));
    switch (m) {
      case 0:
        return z;
      case 1:
        return O::fromb(O::bits(z) ^ 0x80000000u);
      case 2:
        return O::sub(pi, O::sub(z, pi_lo));
      default:
        return O::sub(O::sub(z, pi_lo), pi);
    }
  }

  // ---- the functions the kernels call
  template <typename TAcc>
  ALPAKA_FN_ACC inline float tan(const TAcc &acc, float x) {
    if constexpr (mkseeder::kGpuAcc<TAcc>) {
      if (alpaka::math::abs(acc, x) <= 0x1.921f00p+7f)  // 0x43490f80
        return tanfMedium<DeviceOps>(x);
    }
    return alpaka::math::tan(acc, x);  // the host's tanf (a GPU beyond 2^7 pi/2: not a seed's theta)
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline float atan2(const TAcc &acc, float y, float x) {
    if constexpr (mkseeder::kCpuAcc<TAcc>)
      return glibc::atan2f(y, x);
    else
      return alpaka::math::atan2(acc, y, x);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline float asin(const TAcc &acc, float x) {
    if constexpr (mkseeder::kCpuAcc<TAcc>)
      return glibc::asinf(x);
    else
      return alpaka::math::asin(acc, x);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::libm

#endif
