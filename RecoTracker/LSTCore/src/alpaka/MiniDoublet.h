#ifndef RecoTracker_LSTCore_src_alpaka_MiniDoublet_h
#define RecoTracker_LSTCore_src_alpaka_MiniDoublet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/HitsSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/MiniDoubletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Pre-computed module-constant data for MiniDoublet kernels.
  // Populated once per module to avoid redundant SoA loads in the inner hit-pair loop.
  struct ModuleMDData {
    uint16_t lowerModuleIndex;
    short subdet;
    short side;
    short moduleType;
    short moduleLayerType;
    unsigned int iL;  // layer - 1
    bool isTilted;
    bool isEndcapTwoS;
    bool isGloballyInner;

    // shiftStripHits constants
    float slope;      // dxdys
    float drdz;       // drdzs[lowerModuleIndex]
    float moduleSep;  // moduleGapSize result

    // dPhiThreshold pre-computed constants
    float miniPVoff;
    float miniMuls;
    float miniTilt2;     // 0 for non-tilted and endcap
    float sqrtTermBase;  // miniMuls^2 + miniPVoff^2
    float sqrtTermFlat;  // sqrt(sqrtTermBase), valid for barrel flat
  };

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addMDToMemory(TAcc const& acc,
                                                    MiniDoublets mds,
                                                    HitsBaseConst hitsBase,
                                                    HitsExtendedConst hitsExtended,
                                                    ModuleMDData const& mod,
                                                    unsigned int lowerHitIdx,
                                                    unsigned int upperHitIdx,
                                                    float dz,
                                                    float dPhi,
                                                    float dPhiChange,
                                                    float shiftedX,
                                                    float shiftedY,
                                                    float shiftedZ,
                                                    float noShiftedDphi,
                                                    float noShiftedDPhiChange,
                                                    unsigned int idx) {
    //the index into which this MD needs to be written will be computed in the kernel
    //nMDs variable will be incremented in the kernel, no need to worry about that here

    mds.moduleIndices()[idx] = mod.lowerModuleIndex;
    unsigned int anchorHitIndex, outerHitIndex;
    if (mod.moduleType == PS and mod.moduleLayerType == Strip) {
      mds.anchorHitIndices()[idx] = upperHitIdx;
      mds.outerHitIndices()[idx] = lowerHitIdx;

      anchorHitIndex = upperHitIdx;
      outerHitIndex = lowerHitIdx;
    } else {
      mds.anchorHitIndices()[idx] = lowerHitIdx;
      mds.outerHitIndices()[idx] = upperHitIdx;

      anchorHitIndex = lowerHitIdx;
      outerHitIndex = upperHitIdx;
    }

    mds.dphichanges()[idx] = dPhiChange;
    mds.dphis()[idx] = dPhi;
    mds.dzs()[idx] = dz;
#ifdef CUT_VALUE_DEBUG
    mds.shiftedXs()[idx] = shiftedX;
    mds.shiftedYs()[idx] = shiftedY;
    mds.shiftedZs()[idx] = shiftedZ;

    mds.noShiftedDphis()[idx] = noShiftedDphi;
    mds.noShiftedDphiChanges()[idx] = noShiftedDPhiChange;
#endif

    mds.anchorX()[idx] = hitsBase.xs()[anchorHitIndex];
    mds.anchorY()[idx] = hitsBase.ys()[anchorHitIndex];
    mds.anchorZ()[idx] = hitsBase.zs()[anchorHitIndex];
    mds.anchorRt()[idx] = hitsExtended.rts()[anchorHitIndex];
    mds.anchorPhi()[idx] = hitsExtended.phis()[anchorHitIndex];
    mds.anchorEta()[idx] = hitsExtended.etas()[anchorHitIndex];
    mds.anchorHighEdgeX()[idx] = hitsExtended.highEdgeXs()[anchorHitIndex];
    mds.anchorHighEdgeY()[idx] = hitsExtended.highEdgeYs()[anchorHitIndex];
    mds.anchorLowEdgeX()[idx] = hitsExtended.lowEdgeXs()[anchorHitIndex];
    mds.anchorLowEdgeY()[idx] = hitsExtended.lowEdgeYs()[anchorHitIndex];
    // Edge phi values are only meaningful for TwoS endcap modules (where highEdgeX/Y are set in HitLoopKernel).
    // Downstream readers (Segment.h, Quintuplet.h) only access these when outerLayerEndcapTwoS is true.
    if (mod.isEndcapTwoS) {
      mds.anchorHighEdgePhi()[idx] = alpaka::math::atan2(acc, mds.anchorHighEdgeY()[idx], mds.anchorHighEdgeX()[idx]);
      mds.anchorLowEdgePhi()[idx] = alpaka::math::atan2(acc, mds.anchorLowEdgeY()[idx], mds.anchorLowEdgeX()[idx]);
    }

    mds.outerX()[idx] = hitsBase.xs()[outerHitIndex];
    mds.outerY()[idx] = hitsBase.ys()[outerHitIndex];
    mds.outerZ()[idx] = hitsBase.zs()[outerHitIndex];
#ifdef CUT_VALUE_DEBUG
    mds.outerRt()[idx] = hitsExtended.rts()[outerHitIndex];
    mds.outerPhi()[idx] = hitsExtended.phis()[outerHitIndex];
    mds.outerEta()[idx] = hitsExtended.etas()[outerHitIndex];
    mds.outerHighEdgeX()[idx] = hitsExtended.highEdgeXs()[outerHitIndex];
    mds.outerHighEdgeY()[idx] = hitsExtended.highEdgeYs()[outerHitIndex];
    mds.outerLowEdgeX()[idx] = hitsExtended.lowEdgeXs()[outerHitIndex];
    mds.outerLowEdgeY()[idx] = hitsExtended.lowEdgeYs()[outerHitIndex];
#endif
  }

  // Overload for callers (Segment.h) that still pass ModulesConst + moduleIndex.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addMDToMemory(TAcc const& acc,
                                                    MiniDoublets mds,
                                                    HitsBaseConst hitsBase,
                                                    HitsExtendedConst hitsExtended,
                                                    ModulesConst modules,
                                                    unsigned int lowerHitIdx,
                                                    unsigned int upperHitIdx,
                                                    uint16_t lowerModuleIdx,
                                                    float dz,
                                                    float dPhi,
                                                    float dPhiChange,
                                                    float shiftedX,
                                                    float shiftedY,
                                                    float shiftedZ,
                                                    float noShiftedDphi,
                                                    float noShiftedDPhiChange,
                                                    unsigned int idx) {
    ModuleMDData mod;
    mod.lowerModuleIndex = lowerModuleIdx;
    mod.moduleType = modules.moduleType()[lowerModuleIdx];
    mod.moduleLayerType = modules.moduleLayerType()[lowerModuleIdx];
    mod.subdet = modules.subdets()[lowerModuleIdx];
    mod.isEndcapTwoS = (mod.subdet == Endcap && mod.moduleType == TwoS);
    addMDToMemory(acc,
                  mds,
                  hitsBase,
                  hitsExtended,
                  mod,
                  lowerHitIdx,
                  upperHitIdx,
                  dz,
                  dPhi,
                  dPhiChange,
                  shiftedX,
                  shiftedY,
                  shiftedZ,
                  noShiftedDphi,
                  noShiftedDPhiChange,
                  idx);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isTighterTiltedModules(ModulesConst modules, uint16_t moduleIndex) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modules.subdets()[moduleIndex];
    short layer = modules.layers()[moduleIndex];
    short side = modules.sides()[moduleIndex];
    short rod = modules.rods()[moduleIndex];

    if (subdet == Barrel) {
      if ((side != Center and layer == 3) or (side == NegZ and layer == 2 and rod > 5) or
          (side == PosZ and layer == 2 and rod < 8) or (side == NegZ and layer == 1 and rod > 9) or
          (side == PosZ and layer == 1 and rod < 4))
        return true;
      else
        return false;
    } else
      return false;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize(ModulesConst modules, uint16_t moduleIndex) {
    unsigned int iL = modules.layers()[moduleIndex] - 1;
    unsigned int iR = modules.rings()[moduleIndex] - 1;
    short subdet = modules.subdets()[moduleIndex];
    short side = modules.sides()[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = kMiniDeltaFlat[iL];
    } else if (isTighterTiltedModules(modules, moduleIndex)) {
      moduleSeparation = kMiniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = kMiniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = kMiniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float dPhiThreshold(
      TAcc const& acc, float rt, ModuleMDData const& mod, const float ptCut, float dPhi = 0, float dz = 0) {
    const float miniSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rt * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    if (mod.subdet == Barrel and mod.side == Center) {
      return miniSlope + mod.sqrtTermFlat;
    } else if (mod.subdet == Barrel) {
      return miniSlope + alpaka::math::sqrt(acc, mod.sqrtTermBase + mod.miniTilt2 * miniSlope * miniSlope);
    } else {
      const float miniLum = alpaka::math::abs(acc, dPhi * kDeltaZLum / dz);
      return miniSlope + alpaka::math::sqrt(acc, mod.sqrtTermBase + miniLum * miniLum);
    }
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_INLINE ALPAKA_FN_ACC void shiftStripHits(TAcc const& acc,
                                                     ModuleMDData const& mod,
                                                     float* shiftedCoords,
                                                     float xLower,
                                                     float yLower,
                                                     float zLower,
                                                     float rtLower,
                                                     float xUpper,
                                                     float yUpper,
                                                     float zUpper,
                                                     float rtUpper) {
    float xp, yp, zp, rtp;
    float xo, yo;
    bool pHitInverted = false;
    if (mod.moduleType == PS) {
      if (mod.moduleLayerType == Pixel) {
        xo = xUpper;
        yo = yUpper;
        xp = xLower;
        yp = yLower;
        zp = zLower;
        rtp = rtLower;
      } else {
        xo = xLower;
        yo = yLower;
        xp = xUpper;
        yp = yUpper;
        zp = zUpper;
        rtp = rtUpper;
        pHitInverted = true;
      }
    } else {
      xo = xUpper;
      yo = yUpper;
      xp = xLower;
      yp = yLower;
      zp = zLower;
      rtp = rtLower;
    }

    const bool isEndcap = (mod.subdet == Endcap);

    const float hypot_rz = alpaka::math::sqrt(acc, rtp * rtp + zp * zp);
    const float sinA = rtp / hypot_rz;
    const float cosA = alpaka::math::abs(acc, zp) / hypot_rz;

    float sinApB;
    if (isEndcap) {
      sinApB = cosA;
    } else {
      const float inv_hypot_drdz = 1.f / alpaka::math::sqrt(acc, 1.f + mod.drdz * mod.drdz);
      sinApB = sinA * inv_hypot_drdz + cosA * mod.drdz * inv_hypot_drdz;
    }

    float moduleSeparation = mod.moduleSep;

    if (mod.isGloballyInner == pHitInverted) {
      moduleSeparation *= -1;
    }

    float drprime = moduleSeparation * sinA / sinApB;

    float drprime_x, drprime_y;
    const float slope = mod.slope;
    if (slope != kVerticalModuleSlope && edm::isFinite(slope)) {
      const float inv_hypot_slope = 1.f / alpaka::math::sqrt(acc, 1.f + slope * slope);
      drprime_x = drprime * ((xp > 0.f) - (xp < 0.f)) * alpaka::math::abs(acc, slope) * inv_hypot_slope;
      drprime_y = drprime * ((yp > 0.f) - (yp < 0.f)) * inv_hypot_slope;
    } else {
      drprime_x = drprime * ((xp > 0.f) - (xp < 0.f));
      drprime_y = 0.f;
    }

    float xa = xp + drprime_x;
    float ya = yp + drprime_y;

    float xn, yn;
    if (slope == kVerticalModuleSlope || edm::isNotFinite(slope)) {
      xn = xa;
      yn = yo;
    } else if (slope == 0) {
      xn = xo;
      yn = ya;
    } else {
      xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope));
      yn = (xn - xa) * slope + ya;
    }

    float absdzprime = alpaka::math::abs(acc, moduleSeparation * cosA / sinApB);

    float abszn;
    if (mod.moduleLayerType == Pixel) {
      abszn = alpaka::math::abs(acc, zp) + absdzprime;
    } else {
      abszn = alpaka::math::abs(acc, zp) - absdzprime;
    }

    float zn = abszn * ((zp > 0) ? 1 : -1);

    shiftedCoords[0] = xn;
    shiftedCoords[1] = yn;
    shiftedCoords[2] = zn;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runMiniDoubletDefaultAlgoBarrel(TAcc const& acc,
                                                                      ModuleMDData const& mod,
                                                                      float& dz,
                                                                      float& dPhi,
                                                                      float& dPhiChange,
                                                                      float& shiftedX,
                                                                      float& shiftedY,
                                                                      float& shiftedZ,
                                                                      float& noShiftedDphi,
                                                                      float& noShiftedDphiChange,
                                                                      float xLower,
                                                                      float yLower,
                                                                      float zLower,
                                                                      float rtLower,
                                                                      float xUpper,
                                                                      float yUpper,
                                                                      float zUpper,
                                                                      float rtUpper,
                                                                      const float ptCut) {
    dz = zLower - zUpper;
    const float dzCut = mod.moduleType == PS ? 2.f : 10.f;
    const float sign = ((dz > 0) - (dz < 0)) * ((zLower > 0) - (zLower < 0));
    const float invertedcrossercut = (alpaka::math::abs(acc, dz) > 2) * sign;

    if ((alpaka::math::abs(acc, dz) >= dzCut) || (invertedcrossercut > 0)) {
      return false;
    }

    float miniCut = mod.moduleLayerType == Pixel ? dPhiThreshold(acc, rtLower, mod, ptCut)
                                                 : dPhiThreshold(acc, rtUpper, mod, ptCut);
    const float miniCutSq = miniCut * miniCut;
    const float tanMiniCutSq = miniCutSq * (1.f + (2.f / 3.f) * miniCutSq);

    float x1, y1, x2, y2, r1sq, r2sq;
    float shiftedRt2 = 0.f;

    if (mod.isTilted) {
      float shiftedCoords[3];
      shiftStripHits(acc, mod, shiftedCoords, xLower, yLower, zLower, rtLower, xUpper, yUpper, zUpper, rtUpper);
      float xn = shiftedCoords[0];
      float yn = shiftedCoords[1];
      shiftedRt2 = xn * xn + yn * yn;

      if (mod.moduleLayerType == Pixel) {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zUpper;
        x1 = xLower;
        y1 = yLower;
        x2 = xn;
        y2 = yn;
        r1sq = rtLower * rtLower;
        r2sq = shiftedRt2;
      } else {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zLower;
        x1 = xn;
        y1 = yn;
        x2 = xUpper;
        y2 = yUpper;
        r1sq = shiftedRt2;
        r2sq = rtUpper * rtUpper;
      }
    } else {
      shiftedX = 0.f;
      shiftedY = 0.f;
      shiftedZ = 0.f;
      x1 = xLower;
      y1 = yLower;
      x2 = xUpper;
      y2 = yUpper;
      r1sq = rtLower * rtLower;
      r2sq = rtUpper * rtUpper;
    }

    const float crossDP = x1 * y2 - x2 * y1;
    const float dotDP = x1 * x2 + y1 * y2;
    if (dotDP <= 0.f || crossDP * crossDP >= tanMiniCutSq * dotDP * dotDP) {
      return false;
    }

    const float rInnerSq = alpaka::math::min(acc, r1sq, r2sq);
    const float dotDC = dotDP - rInnerSq;
    if (dotDC <= 0.f || crossDP * crossDP >= tanMiniCutSq * dotDC * dotDC) {
      return false;
    }

    dPhi = alpaka::math::atan2(acc, crossDP, dotDP);
    if (r1sq < r2sq) {
      dPhiChange = alpaka::math::atan2(acc, crossDP, dotDP - r1sq);
    } else {
      dPhiChange = alpaka::math::atan2(acc, -crossDP, dotDP - r2sq);
    }

#ifdef CUT_VALUE_DEBUG
    if (mod.isTilted) {
      const float crossNS = xLower * yUpper - xUpper * yLower;
      const float dotNS = xLower * xUpper + yLower * yUpper;
      noShiftedDphi = alpaka::math::atan2(acc, crossNS, dotNS);
      if (rtLower < rtUpper) {
        noShiftedDphiChange = alpaka::math::atan2(acc, crossNS, dotNS - rtLower * rtLower);
      } else {
        noShiftedDphiChange = alpaka::math::atan2(acc, -crossNS, dotNS - rtUpper * rtUpper);
      }
    } else {
      noShiftedDphi = dPhi;
      noShiftedDphiChange = dPhiChange;
    }
#endif

    return true;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runMiniDoubletDefaultAlgoEndcap(TAcc const& acc,
                                                                      ModuleMDData const& mod,
                                                                      float& drt,
                                                                      float& dPhi,
                                                                      float& dPhiChange,
                                                                      float& shiftedX,
                                                                      float& shiftedY,
                                                                      float& shiftedZ,
                                                                      float& noShiftedDphi,
                                                                      float& noShiftedDphichange,
                                                                      float xLower,
                                                                      float yLower,
                                                                      float zLower,
                                                                      float rtLower,
                                                                      float xUpper,
                                                                      float yUpper,
                                                                      float zUpper,
                                                                      float rtUpper,
                                                                      const float ptCut) {
    float dz = zLower - zUpper;

    const float dzCut = 1.f;

    if (alpaka::math::abs(acc, dz) >= dzCut) {
      return false;
    }
    const float drtCut = mod.moduleType == PS ? 2.f : 10.f;
    drt = rtLower - rtUpper;
    if (alpaka::math::abs(acc, drt) >= drtCut) {
      return false;
    }
    float xn = 0, yn = 0, zn = 0;

    float shiftedCoords[3];
    shiftStripHits(acc, mod, shiftedCoords, xLower, yLower, zLower, rtLower, xUpper, yUpper, zUpper, rtUpper);

    xn = shiftedCoords[0];
    yn = shiftedCoords[1];
    zn = shiftedCoords[2];

    // Compute dPhi using cross/dot product with single atan2 (instead of 2 atan2 from deltaPhi).
    // deltaPhi(x1,y1,x2,y2) = atan2(x1*y2 - x2*y1, x1*x2 + y1*y2).
    float dpX1, dpY1, dpX2, dpY2;  // the two points for dPhi
    if (mod.moduleType == PS) {
      if (mod.moduleLayerType == Pixel) {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zUpper;
        dpX1 = xLower;
        dpY1 = yLower;
        dpX2 = xn;
        dpY2 = yn;
      } else {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zLower;
        dpX1 = xn;
        dpY1 = yn;
        dpX2 = xUpper;
        dpY2 = yUpper;
      }
    } else {
      shiftedX = xn;
      shiftedY = yn;
      shiftedZ = zUpper;
      dpX1 = xLower;
      dpY1 = yLower;
      dpX2 = xn;
      dpY2 = yn;
    }

    const float crossDP = dpX1 * dpY2 - dpX2 * dpY1;
    const float dotDP = dpX1 * dpX2 + dpY1 * dpY2;

    // Algebraic dPhiChange pre-check: reject pairs guaranteed to fail the dPhiChange cut
    // without computing atan2.  Uses |tan(dPhi)| = |crossDP/dotDP| as an overestimate of
    // |dPhi| for the miniLum term (since tan(x) >= x for x in [0, pi/2)), making this
    // strictly looser than the actual cut -- physics-safe (never rejects a pair that would pass).
    {
      float rt = mod.moduleLayerType == Pixel ? rtLower : rtUpper;
      float miniSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rt * k2Rinv1GeVf / ptCut, kSinAlphaMax));
      if (dotDP > 0.f) {
        // Use |tan(dPhi)| = |crossDP|/dotDP as overestimate of |dPhi| for miniLum
        float absCrossDP = alpaka::math::abs(acc, crossDP);
        float tanDPhiOverDz = absCrossDP / (dotDP * alpaka::math::abs(acc, dz));
        float miniLum = tanDPhiOverDz * kDeltaZLum;  // overestimates actual miniLum
        float miniCut = miniSlope + alpaka::math::sqrt(acc, mod.sqrtTermBase + miniLum * miniLum);
        // dPhiChange = dPhi * (|zLower| + |dz|) / |dz|, so the dPhiChange cut
        // |dPhiChange| >= miniCut becomes |dPhi| >= miniCut * |dz| / (|zLower| + |dz|)
        // Using crossDP/dotDP: |crossDP/dotDP| >= miniCut * |dz| / (|zLower| + |dz|)
        // => |crossDP| * (|zLower| + |dz|) >= miniCut * dotDP * |dz|
        float absDz = alpaka::math::abs(acc, dz);
        float lhs = absCrossDP * (alpaka::math::abs(acc, zLower) + absDz);
        float rhs = miniCut * dotDP * absDz;
        if (lhs >= rhs)
          return false;
      } else {
        // dotDP <= 0 means |dPhi| > pi/2 -- guaranteed to fail any reasonable cut
        return false;
      }
    }

    dPhi = alpaka::math::atan2(acc, crossDP, dotDP);

    // dz needs to change if it is a PS module where the strip hits are shifted in order to properly account for the case when a tilted module falls under "endcap logic"
    // if it was an endcap it will have zero effect
    if (mod.moduleType == PS) {
      dz = mod.moduleLayerType == Pixel ? zLower - zn : zUpper - zn;
    }

    float miniCut = mod.moduleLayerType == Pixel ? dPhiThreshold(acc, rtLower, mod, ptCut, dPhi, dz)
                                                 : dPhiThreshold(acc, rtUpper, mod, ptCut, dPhi, dz);

    if (alpaka::math::abs(acc, dPhi) >= miniCut) {
      return false;
    }

    // Cut #4: Another cut on the dphi after some modification
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3119-L3124

    float dzFrac = alpaka::math::abs(acc, dz) / alpaka::math::abs(acc, zLower);
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);

    if (alpaka::math::abs(acc, dPhiChange) >= miniCut) {
      return false;
    }

    // Passed all cuts -- compute noShifted values for storage (only ~2% of pairs reach here).
#ifdef CUT_VALUE_DEBUG
    const float crossNS = xLower * yUpper - xUpper * yLower;
    const float dotNS = xLower * xUpper + yLower * yUpper;
    noShiftedDphi = alpaka::math::atan2(acc, crossNS, dotNS);
    noShiftedDphichange = noShiftedDphi / dzFrac * (1.f + dzFrac);
#endif

    return true;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runMiniDoubletDefaultAlgo(TAcc const& acc,
                                                                ModuleMDData const& mod,
                                                                float& dz,
                                                                float& dPhi,
                                                                float& dPhiChange,
                                                                float& shiftedX,
                                                                float& shiftedY,
                                                                float& shiftedZ,
                                                                float& noShiftedDphi,
                                                                float& noShiftedDphiChange,
                                                                float xLower,
                                                                float yLower,
                                                                float zLower,
                                                                float rtLower,
                                                                float xUpper,
                                                                float yUpper,
                                                                float zUpper,
                                                                float rtUpper,
                                                                const float ptCut,
                                                                uint16_t clustSizeLower,
                                                                uint16_t clustSizeUpper,
                                                                const uint16_t clustSizeCut) {
    if (clustSizeLower > clustSizeCut or clustSizeUpper > clustSizeCut) {
      return false;
    }
    if (mod.subdet == Barrel) {
      return runMiniDoubletDefaultAlgoBarrel(acc,
                                             mod,
                                             dz,
                                             dPhi,
                                             dPhiChange,
                                             shiftedX,
                                             shiftedY,
                                             shiftedZ,
                                             noShiftedDphi,
                                             noShiftedDphiChange,
                                             xLower,
                                             yLower,
                                             zLower,
                                             rtLower,
                                             xUpper,
                                             yUpper,
                                             zUpper,
                                             rtUpper,
                                             ptCut);
    } else {
      return runMiniDoubletDefaultAlgoEndcap(acc,
                                             mod,
                                             dz,
                                             dPhi,
                                             dPhiChange,
                                             shiftedX,
                                             shiftedY,
                                             shiftedZ,
                                             noShiftedDphi,
                                             noShiftedDphiChange,
                                             xLower,
                                             yLower,
                                             zLower,
                                             rtLower,
                                             xUpper,
                                             yUpper,
                                             zUpper,
                                             rtUpper,
                                             ptCut);
    }
  }

  struct CreateMiniDoublets {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  HitsBaseConst hitsBase,
                                  HitsExtendedConst hitsExtended,
                                  HitsRangesConst hitsRanges,
                                  MiniDoublets mds,
                                  MiniDoubletsOccupancy mdsOccupancy,
                                  ObjectRangesConst ranges,
                                  const float ptCut,
                                  const uint16_t clustSizeCut) const {
      for (uint16_t lowerModuleIndex : cms::alpakatools::uniform_elements_y(acc, modules.nLowerModules())) {
        int nLowerHits = hitsRanges.hitRangesnLower()[lowerModuleIndex];
        int nUpperHits = hitsRanges.hitRangesnUpper()[lowerModuleIndex];
        if (hitsRanges.hitRangesLower()[lowerModuleIndex] == -1)
          continue;
        unsigned int upHitArrayIndex = hitsRanges.hitRangesUpper()[lowerModuleIndex];
        unsigned int loHitArrayIndex = hitsRanges.hitRangesLower()[lowerModuleIndex];
        int limit = nUpperHits * nLowerHits;

        // Hoist module-constant data once per module to avoid redundant SoA loads per hit pair.
        ModuleMDData mod;
        mod.lowerModuleIndex = lowerModuleIndex;
        mod.subdet = modules.subdets()[lowerModuleIndex];
        mod.side = modules.sides()[lowerModuleIndex];
        mod.moduleType = modules.moduleType()[lowerModuleIndex];
        mod.moduleLayerType = modules.moduleLayerType()[lowerModuleIndex];
        mod.iL = modules.layers()[lowerModuleIndex] - 1;
        mod.isTilted = (mod.subdet == Barrel && mod.side != Center);
        mod.isEndcapTwoS = (mod.subdet == Endcap && mod.moduleType == TwoS);
        mod.isGloballyInner = modules.isGloballyInner()[lowerModuleIndex];
        mod.slope = modules.dxdys()[lowerModuleIndex];
        mod.drdz = modules.drdzs()[lowerModuleIndex];
        mod.moduleSep = moduleGapSize(modules, lowerModuleIndex);

        // Pre-compute dPhiThreshold module-constant parts
        float rLayNominal = (mod.subdet == Barrel) ? kMiniRminMeanBarrel[mod.iL] : kMiniRminMeanEndcap[mod.iL];
        mod.miniPVoff = 0.1f / rLayNominal;
        mod.miniMuls = (mod.subdet == Barrel) ? kMiniMulsPtScaleBarrel[mod.iL] * 3.f / ptCut
                                              : kMiniMulsPtScaleEndcap[mod.iL] * 3.f / ptCut;
        mod.sqrtTermBase = mod.miniMuls * mod.miniMuls + mod.miniPVoff * mod.miniPVoff;
        mod.sqrtTermFlat = alpaka::math::sqrt(acc, mod.sqrtTermBase);

        if (mod.isTilted) {
          float drdzThresh;
          if (mod.moduleType == PS and mod.moduleLayerType == Strip) {
            drdzThresh = modules.drdzs()[lowerModuleIndex];
          } else {
            drdzThresh = modules.drdzs()[modules.partnerModuleIndices()[lowerModuleIndex]];
          }
          mod.miniTilt2 = 0.25f * (kPixelPSZpitch * kPixelPSZpitch) * (drdzThresh * drdzThresh) /
                          (1.f + drdzThresh * drdzThresh) / mod.moduleSep;
        } else {
          mod.miniTilt2 = 0.f;
        }

        for (int hitIndex : cms::alpakatools::uniform_elements_x(acc, limit)) {
          int lowerHitIndex = hitIndex / nUpperHits;
          int upperHitIndex = hitIndex % nUpperHits;
          if (upperHitIndex >= nUpperHits)
            continue;
          if (lowerHitIndex >= nLowerHits)
            continue;
          unsigned int lowerHitArrayIndex = loHitArrayIndex + lowerHitIndex;
          float xLower = hitsBase.xs()[lowerHitArrayIndex];
          float yLower = hitsBase.ys()[lowerHitArrayIndex];
          float zLower = hitsBase.zs()[lowerHitArrayIndex];
          float rtLower = hitsExtended.rts()[lowerHitArrayIndex];
          unsigned int upperHitArrayIndex = upHitArrayIndex + upperHitIndex;
          float xUpper = hitsBase.xs()[upperHitArrayIndex];
          float yUpper = hitsBase.ys()[upperHitArrayIndex];
          float zUpper = hitsBase.zs()[upperHitArrayIndex];
          float rtUpper = hitsExtended.rts()[upperHitArrayIndex];
          uint16_t clustSizeLower = hitsBase.clustsize()[lowerHitArrayIndex];
          uint16_t clustSizeUpper = hitsBase.clustsize()[upperHitArrayIndex];

          float dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDphi, noShiftedDphiChange;
          bool success = runMiniDoubletDefaultAlgo(acc,
                                                   mod,
                                                   dz,
                                                   dphi,
                                                   dphichange,
                                                   shiftedX,
                                                   shiftedY,
                                                   shiftedZ,
                                                   noShiftedDphi,
                                                   noShiftedDphiChange,
                                                   xLower,
                                                   yLower,
                                                   zLower,
                                                   rtLower,
                                                   xUpper,
                                                   yUpper,
                                                   zUpper,
                                                   rtUpper,
                                                   ptCut,
                                                   clustSizeLower,
                                                   clustSizeUpper,
                                                   clustSizeCut);
          if (success) {
            int totOccupancyMDs = alpaka::atomicAdd(
                acc, &mdsOccupancy.totOccupancyMDs()[lowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
            if (totOccupancyMDs >= (ranges.miniDoubletModuleOccupancy()[lowerModuleIndex])) {
#ifdef WARNINGS
              printf(
                  "Mini-doublet excess alert! Module index = %d, Occupancy = %d\n", lowerModuleIndex, totOccupancyMDs);
#endif
            } else {
              int mdModuleIndex =
                  alpaka::atomicAdd(acc, &mdsOccupancy.nMDs()[lowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
              unsigned int mdIndex = ranges.miniDoubletModuleIndices()[lowerModuleIndex] + mdModuleIndex;

              addMDToMemory(acc,
                            mds,
                            hitsBase,
                            hitsExtended,
                            mod,
                            lowerHitArrayIndex,
                            upperHitArrayIndex,
                            dz,
                            dphi,
                            dphichange,
                            shiftedX,
                            shiftedY,
                            shiftedZ,
                            noShiftedDphi,
                            noShiftedDphiChange,
                            mdIndex);
            }
          }
        }
      }
    }
  };

  // Helper function to determine eta bin for occupancies
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int getEtaBin(const float module_eta) {
    if (module_eta < 0.75f)
      return 0;
    else if (module_eta < 1.5f)
      return 1;
    else if (module_eta < 2.25f)
      return 2;
    else if (module_eta < 3.0f)
      return 3;
    return -1;
  }

  // Helper function to determine category number for occupancies
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int getCategoryNumber(const short module_layers,
                                                       const short module_subdets,
                                                       const short module_rings) {
    if (module_subdets == Barrel) {
      return (module_layers <= 3) ? 0 : 1;
    } else if (module_subdets == Endcap) {
      if (module_layers <= 2) {
        return (module_rings >= 11) ? 2 : 3;
      } else {
        return (module_rings >= 8) ? 2 : 3;
      }
    }
    return -1;
  }

  struct CreateMDArrayRangesGPU {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  HitsRangesConst hitsRanges,
                                  ObjectRanges ranges,
                                  const float ptCut) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      // Declare variables in shared memory and set to 0
      int& nTotalMDs = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalMDs = 0;
      }
      alpaka::syncBlockThreads(acc);

      // Occupancy matrix for 0.8 GeV pT Cut
      constexpr int p08_occupancy_matrix[4][4] = {
          {49, 42, 37, 41},  // category 0
          {100, 100, 0, 0},  // category 1
          {0, 16, 19, 0},    // category 2
          {0, 14, 20, 25}    // category 3
      };

      // Occupancy matrix for 0.6 GeV pT Cut, 99.99%
      constexpr int p06_occupancy_matrix[4][4] = {
          {60, 57, 54, 48},  // category 0
          {259, 195, 0, 0},  // category 1
          {0, 23, 28, 0},    // category 2
          {0, 25, 25, 33}    // category 3
      };

      // Select the appropriate occupancy matrix based on ptCut
      const auto& occupancy_matrix = (ptCut < 0.8f) ? p06_occupancy_matrix : p08_occupancy_matrix;

      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        const int nLower = hitsRanges.hitRangesnLower()[i];
        const int nUpper = hitsRanges.hitRangesnUpper()[i];
        const int dynamicMDs = nLower * nUpper;

        // Matrix-based cap
        short module_layers = modules.layers()[i];
        short module_subdets = modules.subdets()[i];
        short module_rings = modules.rings()[i];
        float module_eta = alpaka::math::abs(acc, modules.eta()[i]);

        int category_number = getCategoryNumber(module_layers, module_subdets, module_rings);
        int eta_number = getEtaBin(module_eta);

#ifdef WARNINGS
        if (category_number == -1 || eta_number == -1) {
          printf("Unhandled case in createMDArrayRangesGPU! Module index = %i\n", i);
        }
#endif

        int occupancy = (category_number != -1 && eta_number != -1)
                            ? alpaka::math::min(acc, dynamicMDs, occupancy_matrix[category_number][eta_number])
                            : 0;
        unsigned int nTotMDs = alpaka::atomicAdd(acc, &nTotalMDs, occupancy, alpaka::hierarchy::Threads{});

        ranges.miniDoubletModuleIndices()[i] = nTotMDs;
        ranges.miniDoubletModuleOccupancy()[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.miniDoubletModuleIndices()[modules.nLowerModules()] = nTotalMDs;
        ranges.nTotalMDs() = nTotalMDs;
      }
    }
  };

  struct AddMiniDoubletRangesToEventExplicit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsOccupancy mdsOccupancy,
                                  ObjectRanges ranges,
                                  HitsRangesConst hitsRanges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (mdsOccupancy.nMDs()[i] == 0 or hitsRanges.hitRanges()[i][0] == -1) {
          ranges.mdRanges()[i][0] = -1;
          ranges.mdRanges()[i][1] = -1;
        } else {
          ranges.mdRanges()[i][0] = ranges.miniDoubletModuleIndices()[i];
          ranges.mdRanges()[i][1] = ranges.miniDoubletModuleIndices()[i] + mdsOccupancy.nMDs()[i] - 1;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
