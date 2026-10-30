#ifndef RecoTracker_MkFitAlpaka_interface_othits_OTCpe_h
#define RecoTracker_MkFitAlpaka_interface_othits_OTCpe_h

// Portable Phase2StripCPE::localParameters (Phase2StripCPE.cc) + RectangularPixelPhase2Topology::localX/localY
// (RectangularPixelPhase2Topology.cc), same operations in the same order. The one a*b + c*d of localX/localY
// (float(binoff * pitch) + fraction * local_pitch) goes through mkfitdev::sumOfProducts<M>, as the host compiler may
// contract it: MkFitAlpakaOTRecHitsProducer uses kFuseFirst, the variant that reproduces the host rechits bitwise.
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitInput.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTRecHitSoA.h"

namespace mkfitdev::otcpe {

  // a*b + c*d + e + f with the a*b + c*d part per variant (then plain float adds, left to right)
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float topoLocal(
      float binoff, float pitch, float frac, float lpitch, float half, float off) {
    const float products = sumOfProducts<M>(binoff, pitch, frac, lpitch);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fadd_rn(__fadd_rn(products, half), off);
#else
    return (products + half) + off;
#endif
  }

  // RectangularPixelPhase2Topology::localX(mpx)
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float localX(OTCpeModule const& module, float mpx) {
    int binoffx = int(mpx);
    float fractionX = mpx - float(binoffx);
    float local_pitchx = module.pitchX;
    float half = 0.f;  // ispix_secondhalf_x * 2 * BIG_PIX_PITCH_X * nrows / ROWS_PER_ROC with ispix = 0: +0
    if (binoffx >= module.xB) {
      binoffx = binoffx - module.xShift;
      half = module.xHalfTerm;
    } else if (module.xA <= binoffx && binoffx < module.xB) {
      binoffx = module.xA;
      fractionX = mpx - float(module.xA);
      local_pitchx = module.bigPitchX;
    }
    return topoLocal<M>(float(binoffx), module.pitchX, fractionX, local_pitchx, half, module.xOffset);
  }

  // RectangularPixelPhase2Topology::localY(mpy)
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float localY(OTCpeModule const& module, float mpy) {
    int binoffy = int(mpy);
    float fractionY = mpy - float(binoffy);
    float local_pitchy = module.pitchY;
    float half = 0.f;
    if (binoffy >= module.yB) {
      binoffy = binoffy - module.yShift;
      half = module.yHalfTerm;
    } else if (module.yA <= binoffy && binoffy < module.yB) {
      binoffy = module.yA;
      fractionY = mpy - float(module.yA);
      local_pitchy = module.bigPitchY;
    }
    return topoLocal<M>(float(binoffy), module.pitchY, fractionY, local_pitchy, half, module.yOffset);
  }

  // Phase2StripCPE::localParameters: ix = center() - 0.5 * coveredStrips, iy = column + 0.5
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void localPosition(
      OTCpeModule const& module, uint32_t firstStrip, uint32_t column, uint32_t size, float& lx, float& ly) {
    const float center = float(firstStrip) + 0.5f * float(size);  // Phase2TrackerCluster1D::center()
    const float ix = center - module.halfCovered;
    const float iy = float(column) + 0.5f;
    lx = localX<M>(module, ix);
    ly = localY<M>(module, iy);
  }

}  // namespace mkfitdev::otcpe

#endif
