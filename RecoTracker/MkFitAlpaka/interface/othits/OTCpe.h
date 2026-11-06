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
    const float s = sumOfProducts<M>(binoff, pitch, frac, lpitch);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fadd_rn(__fadd_rn(s, half), off);
#else
    return (s + half) + off;
#endif
  }

  // RectangularPixelPhase2Topology::localX(mpx)
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float localX(OTCpeModule const& m, float mpx) {
    int binoffx = int(mpx);
    float fractionX = mpx - float(binoffx);
    float local_pitchx = m.pitchX;
    float half = 0.f;  // ispix_secondhalf_x * 2 * BIG_PIX_PITCH_X * nrows / ROWS_PER_ROC with ispix = 0: +0
    if (binoffx >= m.xB) {
      binoffx = binoffx - m.xShift;
      half = m.xHalfTerm;
    } else if (m.xA <= binoffx && binoffx < m.xB) {
      binoffx = m.xA;
      fractionX = mpx - float(m.xA);
      local_pitchx = m.bigPitchX;
    }
    return topoLocal<M>(float(binoffx), m.pitchX, fractionX, local_pitchx, half, m.xOffset);
  }

  // RectangularPixelPhase2Topology::localY(mpy)
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float localY(OTCpeModule const& m, float mpy) {
    int binoffy = int(mpy);
    float fractionY = mpy - float(binoffy);
    float local_pitchy = m.pitchY;
    float half = 0.f;
    if (binoffy >= m.yB) {
      binoffy = binoffy - m.yShift;
      half = m.yHalfTerm;
    } else if (m.yA <= binoffy && binoffy < m.yB) {
      binoffy = m.yA;
      fractionY = mpy - float(m.yA);
      local_pitchy = m.bigPitchY;
    }
    return topoLocal<M>(float(binoffy), m.pitchY, fractionY, local_pitchy, half, m.yOffset);
  }

  // Phase2StripCPE::localParameters: ix = center() - 0.5 * coveredStrips, iy = column + 0.5
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void localPosition(
      OTCpeModule const& m, uint32_t firstStrip, uint32_t column, uint32_t size, float& lx, float& ly) {
    const float center = float(firstStrip) + 0.5f * float(size);  // Phase2TrackerCluster1D::center()
    const float ix = center - m.halfCovered;
    const float iy = float(column) + 0.5f;
    lx = localX<M>(m, ix);
    ly = localY<M>(m, iy);
  }

  // Surface::toGlobal(LocalPoint(lx, ly, 0)) with the rotation/position of the OT module (HitModuleDev layout)
  template <Contract M>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void toGlobal(OTCpeModule const& m, float lx, float ly, float* g) {
    HitModuleDev h;
    for (int j = 0; j < 9; ++j)
      h.r[j] = m.r[j];
    for (int j = 0; j < 3; ++j)
      h.p[j] = m.p[j];
    localToGlobal<M>(h, lx, ly, g);
  }

}  // namespace mkfitdev::otcpe

#endif
