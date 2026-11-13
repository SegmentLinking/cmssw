#ifndef RecoTracker_MkFitAlpaka_interface_hits_DeviceHitInput_h
#define RecoTracker_MkFitAlpaka_interface_hits_DeviceHitInput_h

// Device hit input: the per-hit work of the RecoTracker/MkFit hit converters
// (RecoTracker/MkFit/plugins/convertHits.h + MkFitSiPixelHitConverter / MkFitPhase2HitConverter traits) as portable
// functions of the hit's local quantities and a per-module table, so the HitSoA row can be made on the device from the
// pixel rechit SoA (xLocal, yLocal, xerrLocal, yerrLocal, detectorIndex; legacy cluster sizes from the host) and from
// the device OT rechit SoA.
//   position = Surface::toGlobal(LocalPoint(x, y))            (float, ext-vector TkRotation: rotateBack + position)
//   error    = ErrorFrameTransformer::transform(LocalError, Surface) (float expression, stored via GlobalError(double)
//              and converted back with float(): exact)
//   packed   = Hit::setupAsPixel(shortId, sizeX, sizeY) / setupAsStrip(shortId, 255, size)
// Bitwise identity with the host converters depends on where the host compiler contracted a*b + c*d into an fma.
// The variants below make that explicit (fmaf is correctly rounded on every backend, so a variant that matches the
// host converters is bitwise on the GPU backends too); a comparison with the converters' output picked the variant.

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"

namespace mkfitdev {

  // Per tracker module (index = GeomDet::index(), = the pixel rechit SoA detectorIndex): the Surface rotation rows
  // (TkRotation<float> xx..zz) and position, the mkFit layer (-1: not an mkFit module) and uniqueIdInLayer.
  struct HitModuleDev {
    float rotation[9];  // xx xy xz | yx yy yz | zx zy zz
    float position[3];
    int32_t layer;
    uint32_t detIdInLayer;
  };

  // a*b + c*d as compiled: kFuseSecond = fma(c, d, a*b), kFuseFirst = fma(a, b, c*d), kNoFuse = two roundings
  enum class Contract : int { kFuseSecond = 0, kFuseFirst = 1, kNoFuse = 2 };

  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float sumOfProducts(float a, float b, float c, float d) {
    if constexpr (M == Contract::kFuseSecond)
      return std::fma(c, d, a * b);
    else if constexpr (M == Contract::kFuseFirst)
      return std::fma(a, b, c * d);
    else {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return __fadd_rn(__fmul_rn(a, b), __fmul_rn(c, d));
#else
      volatile float ab = a * b;
      volatile float cd = c * d;
      return ab + cd;
#endif
    }
  }

  // Surface::toGlobal(LocalPoint(lx, ly)): rotateBack(v) = v0*axis0 + v1*axis1 + v2*axis2 (v2 = 0), + position.
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void localToGlobal(HitModuleDev const& module, float lx, float ly, float* xyz) {
    for (int j = 0; j < 3; ++j) {
      const float sum2 = sumOfProducts<M>(lx, module.rotation[j], ly, module.rotation[3 + j]);
      // third term: lz * zx.. with lz = 0 (fused or not, sum2 + (+-0) == sum2 for sum2 != 0; kept for the sign of zero)
      const float sum3 =
          M == Contract::kNoFuse ? sum2 + 0.f * module.rotation[6 + j] : std::fma(0.f, module.rotation[6 + j], sum2);
      xyz[j] = sum3 + module.position[j];
    }
  }

  // ErrorFrameTransformer::transform(LocalError(cxx, cxy, cyy), surface) -> (cxx, cyx, cyy, czx, czy, czz) as float
  // (the SMatrixSym33 packed order of mkfit::Hit: 00, 10, 11, 20, 21, 22).
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void localToGlobalError(
      HitModuleDev const& module, float cxx, float cxy, float cyy, float* err) {
    const float xx = module.rotation[0], xy = module.rotation[1], xz = module.rotation[2];
    const float yx = module.rotation[3], yy = module.rotation[4], yz = module.rotation[5];
    auto sop = [](float a, float b, float c, float d) { return sumOfProducts<M>(a, b, c, d); };
    err[0] = sop(xx, sop(xx, cxx, yx, cxy), yx, sop(xx, cxy, yx, cyy));
    err[1] = sop(xx, sop(xy, cxx, yy, cxy), yx, sop(xy, cxy, yy, cyy));
    err[2] = sop(xy, sop(xy, cxx, yy, cxy), yy, sop(xy, cxy, yy, cyy));
    err[3] = sop(xx, sop(xz, cxx, yz, cxy), yx, sop(xz, cxy, yz, cyy));
    err[4] = sop(xy, sop(xz, cxx, yz, cxy), yy, sop(xz, cxy, yz, cyy));
    err[5] = sop(xz, sop(xz, cxx, yz, cxy), yz, sop(xz, cxy, yz, cyy));
  }

  // Hit::setupAsPixel(shortId, rows, cols): charge_pcm = 255 (raw), spans min(31, n - 1) into 5-bit fields
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t packPixel(uint32_t shortId, int rows, int cols) {
    const int rowsRaw = rows - 1 < 31 ? rows - 1 : 31, colsRaw = cols - 1 < 31 ? cols - 1 : 31;
    return hitpack::pack(shortId, 255u, static_cast<uint32_t>(rowsRaw), static_cast<uint32_t>(colsRaw));
  }
  // Hit::setupAsStrip(shortId, 255, rows): set_charge_pcm(255) = 0 (255 < kMinChargePerCM), span_cols stays 0
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t packStrip(uint32_t shortId, int rows) {
    const int rowsRaw = rows - 1 < 31 ? rows - 1 : 31;
    return hitpack::pack(shortId, 0u, static_cast<uint32_t>(rowsRaw), 0u);
  }

}  // namespace mkfitdev

#endif
