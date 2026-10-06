#ifndef RecoTracker_MkFitAlpaka_interface_hits_DeviceHitInput_h
#define RecoTracker_MkFitAlpaka_interface_hits_DeviceHitInput_h

// Device hit input: the per-hit work of the RecoTracker/MkFit hit converters
// (RecoTracker/MkFit/plugins/convertHits.h + MkFitSiPixelHitConverter / MkFitPhase2HitConverter traits) as portable
// functions of the hit's LOCAL quantities and a per-module table, so the HitSoA row can be made on the device from the
// pixel rechit SoA (xLocal, yLocal, xerrLocal, yerrLocal, clusterSizeX/Y, detectorIndex) and from raw OT rechits.
//   position = Surface::toGlobal(LocalPoint(x, y))            (float, ext-vector TkRotation: rotateBack + position)
//   error    = ErrorFrameTransformer::transform(LocalError, Surface) (float expression, stored via GlobalError(double)
//              and converted back with float(): exact)
//   packed   = Hit::setupAsPixel(shortId, sizeX, sizeY) / setupAsStrip(shortId, 255, size)
// Bitwise identity with the host converters depends on WHERE the host compiler contracted a*b + c*d into an fma.
// The variants below make that explicit (fmaf is correctly rounded on every backend, so a variant that matches the
// host converters is bitwise on CUDA too); a host comparison with the converters' output
// picked the one they use.

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"

namespace mkfitdev {

  // Per tracker module (index = GeomDet::index(), = the pixel rechit SoA detectorIndex): the Surface rotation rows
  // (TkRotation<float> xx..zz) and position, the mkFit layer (-1: not an mkFit module) and uniqueIdInLayer.
  struct HitModuleDev {
    float r[9];  // xx xy xz | yx yy yz | zx zy zz
    float p[3];
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
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void localToGlobal(HitModuleDev const& m, float lx, float ly, float* g) {
    for (int j = 0; j < 3; ++j) {
      const float t = sumOfProducts<M>(lx, m.r[j], ly, m.r[3 + j]);
      // third term: lz * zx.. with lz = 0 (fused or not, t + (+-0) == t for t != 0; kept for the sign of zero)
      const float t3 = M == Contract::kNoFuse ? t + 0.f * m.r[6 + j] : std::fma(0.f, m.r[6 + j], t);
      g[j] = t3 + m.p[j];
    }
  }

  // ErrorFrameTransformer::transform(LocalError(cxx, cxy, cyy), surface) -> (cxx, cyx, cyy, czx, czy, czz) as float
  // (the SMatrixSym33 packed order of mkfit::Hit: 00, 10, 11, 20, 21, 22).
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void localToGlobalError(
      HitModuleDev const& m, float cxx, float cxy, float cyy, float* e) {
    const float xx = m.r[0], xy = m.r[1], xz = m.r[2], yx = m.r[3], yy = m.r[4], yz = m.r[5];
    auto S = [](float a, float b, float c, float d) { return sumOfProducts<M>(a, b, c, d); };
    e[0] = S(xx, S(xx, cxx, yx, cxy), yx, S(xx, cxy, yx, cyy));
    e[1] = S(xx, S(xy, cxx, yy, cxy), yx, S(xy, cxy, yy, cyy));
    e[2] = S(xy, S(xy, cxx, yy, cxy), yy, S(xy, cxy, yy, cyy));
    e[3] = S(xx, S(xz, cxx, yz, cxy), yx, S(xz, cxy, yz, cyy));
    e[4] = S(xy, S(xz, cxx, yz, cxy), yy, S(xz, cxy, yz, cyy));
    e[5] = S(xz, S(xz, cxx, yz, cxy), yz, S(xz, cxy, yz, cyy));
  }

  // Hit::setupAsPixel(shortId, rows, cols): charge_pcm = 255 (raw), spans min(31, n - 1) into 5-bit fields
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t packPixel(uint32_t shortId, int rows, int cols) {
    const int r = rows - 1 < 31 ? rows - 1 : 31, c = cols - 1 < 31 ? cols - 1 : 31;
    return hitpack::pack(shortId, 255u, static_cast<uint32_t>(r), static_cast<uint32_t>(c));
  }
  // Legacy SiPixelCluster sizeX()/sizeY() from the pixel rechit SoA clusterSizeX/Y (pixelCPEforDevice.h position():
  // 8 * (max - min + 1) - unbalance, unbalance = int(8 |q_first - q_last| / (q_first + q_last)) in [0, 7] since both
  // edge charges are > 0; negated at the module edges; capped at maxSizeCluster): ceil(|size| / 8).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int pixelSpanFromSoA(int16_t soaSize) {
    const int s = soaSize < 0 ? -int(soaSize) : int(soaSize);
    return (s + 7) / 8;
  }

  // Hit::setupAsStrip(shortId, 255, rows): set_charge_pcm(255) = 0 (255 < kMinChargePerCM), span_cols stays 0
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t packStrip(uint32_t shortId, int rows) {
    const int r = rows - 1 < 31 ? rows - 1 : 31;
    return hitpack::pack(shortId, 0u, static_cast<uint32_t>(r), 0u);
  }

}  // namespace mkfitdev

#endif
