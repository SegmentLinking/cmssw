#ifndef RecoTracker_MkFitAlpaka_interface_math_TkBfield_h
#define RecoTracker_MkFitAlpaka_interface_math_TkBfield_h

// The menu's tracker magnetic field, portable (host + device), float as the release.
// The Phase-2 HLT menu's MagneticField (grid_160812_3_8t with the parametrised slave OAE_1103l_071212) returns, wherever r < 115 cm and |z| < 280 cm,
//   OAEParametrizedMagneticField::inTeslaUnchecked = magfieldparam::TkBfield("3_8T").getBxyz(x / 100)
// (MagneticField/ParametrizedEngine/src/{OAEParametrizedMagneticField.cc, TkBfield.cc, BCyl.h} of CMSSW_20_1_0_pre2).
// This file transliterates exactly that chain: same float operations in the same order, the 3.8 T parameter set,
// the derived constants as BCylParam's constructor computes them, and unsafe_expf<3> (DataFormats/Math approx_exp.h,
// ESTRIN polynomial, fpfloor = std::floor as in the x86-64 SSE4.1 build). Outside the validity volume the menu's
// field is the volume-based grid: callers must not rely on these values there (tkBfieldValid()).
// Consumers: the field-corrected pLS pT, the LST seed fit, the device PCA, the OT Lorentz shift.

#include <cmath>
#include <cstdint>

#ifndef ALPAKA_FN_HOST_ACC  // a host-only test may define it
#include <alpaka/alpaka.hpp>
#endif

namespace mkfitdev::field {

  namespace tkbfield_detail {
    // approx_exp.h unsafe_expf_impl<3> (ESTRIN form)
    ALPAKA_FN_HOST_ACC inline float unsafeExpf3(float x) {
      constexpr float inv_log2f = float(0x1.715476p0);
      constexpr float log2H = float(0xb.172p-4);
      constexpr float log2L = float(0x1.7f7d1cp-20);
      float y = x;
      float z = std::floor((x * inv_log2f) + 0.5f);
      y -= z * log2H;
      y -= z * log2L;
      int32_t e = z;
      e -= 1;
      const float p23 = (float(0x1.02249p0) + y * float(0x5.62042p-4));
      const float p01 = float(0x2.p0) + y * float(0x1.fff798p0);
      const float p = p01 + y * y * p23;
      union {
        uint32_t ui32;
        float f;
      } ef;
      const uint32_t biased_exponent = e + 127;
      ef.ui32 = (biased_exponent << 23);
      return p * ef.f;
    }

    // BCyl.h bcylDetails::ffunkti<float>
    ALPAKA_FN_HOST_ACC inline void ffunkti(float u, float* ff) {
      float a, b, a2, u2;
      u2 = u * u;
      a = 1.f / (1.f + u2);
      a2 = -3.f * a * a;
      b = std::sqrt(a);
      ff[0] = u * b;
      ff[1] = a * b;
      ff[2] = a2 * ff[0];
      ff[3] = a2 * ff[1] * (1.f - 4 * u2);
    }
  }  // namespace tkbfield_detail

  // True where the menu's field is the closed form (OAEParametrizedMagneticField::isDefined).
  ALPAKA_FN_HOST_ACC inline bool tkBfieldValid(float x, float y, float z) {
    return (x * x + y * y) < (115.f * 115.f) && std::fabs(z) < 280.f;
  }

  // B (tesla) at the global point (x, y, z) in cm: OAEParametrizedMagneticField::inTeslaUnchecked.
  ALPAKA_FN_HOST_ACC inline void tkBfield(float xcm, float ycm, float zcm, float& bx, float& by, float& bz) {
    using namespace tkbfield_detail;
    // TkBfield.cc fpar4 ("3_8T", 3.8T-2G); function-local (a namespace-scope constexpr array is host-only in CUDA)
    constexpr float kPrm[9] = {
        4.24326f, 15.0201f, 3.81492f, 0.0178712f, 0.000656527f, 2.45818f, 0.00778695f, 2.12500f, 1.77436f};
    constexpr float ooh = 1. / 100;
    const float x0 = xcm * ooh, x1 = ycm * ooh, x2 = zcm * ooh;
    // BCylParam constructor (float members; hb0 evaluated in double as there)
    const float ap2 = 4 * kPrm[0] * kPrm[0] / (kPrm[1] * kPrm[1]);
    const float hb0 = 0.5 * kPrm[2] * std::sqrt(1.0 + ap2);
    const float hlova = 1 / std::sqrt(ap2);
    const float ainv = 2 * hlova / kPrm[1];
    const float coeff = 1 / (kPrm[8] * kPrm[8]);
    // TkBfield::getBxyz -> BCycl<float>::compute(r2, z, Br, Bz)
    const float r2 = x0 * x0 + x1 * x1;
    float z = x2;
    z -= kPrm[3];
    const float az = std::abs(z);
    const float zainv = z * ainv;
    const float u = hlova - zainv;
    const float v = hlova + zainv;
    float fu[4], gv[4];
    ffunkti(u, fu);
    ffunkti(v, gv);
    const float rat = 0.5f * ainv;
    const float rat2 = rat * rat * r2;
    float br = hb0 * rat * (fu[1] - gv[1] - (fu[3] - gv[3]) * rat2 * 0.5f);
    float bzz = hb0 * (fu[0] + gv[0] - (fu[2] + gv[2]) * rat2);
    const float corBr = kPrm[4] * z * (az - kPrm[5]) * (az - kPrm[5]);
    const float corBz = -kPrm[6] * (unsafeExpf3(-(z - kPrm[7]) * (z - kPrm[7]) * coeff) +
                                    unsafeExpf3(-(z + kPrm[7]) * (z + kPrm[7]) * coeff));
    br += corBr;
    bzz += corBz;
    bx = br * x0;
    by = br * x1;
    bz = bzz;
  }

  // Bz only (tesla), same evaluation.
  ALPAKA_FN_HOST_ACC inline float tkBz(float xcm, float ycm, float zcm) {
    float bx, by, bz;
    tkBfield(xcm, ycm, zcm, bx, by, bz);
    return bz;
  }

}  // namespace mkfitdev::field

#endif
