#ifndef RecoTracker_LSTCore_src_alpaka_MiniDoublet_h
#define RecoTracker_LSTCore_src_alpaka_MiniDoublet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <limits>

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
    float slope;      // dxdys
    float drdz;       // drdzs[lowerModuleIndex]
    float moduleSep;  // moduleGapSize result
    float miniPVoff;
    float miniMuls;
    float miniTilt2;             // 0 for non-tilted and endcap
    float miniMulsAndPVoff;      // miniMuls^2 + miniPVoff^2
    float sqrtMiniMulsAndPVoff;  // sqrt(miniMulsAndPVoff), valid for barrel flat

    unsigned int iL;  // layer - 1

    uint16_t lowerModuleIndex;
    short subdet;
    short side;
    short moduleType;
    short moduleLayerType;

    bool isTilted;
    bool isGloballyInner;
  };

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addMDToMemory(TAcc const& acc,
                                                    MiniDoublets mds,
                                                    MiniDoubletsBuild mdsBuild,
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

    mdsBuild.dphichanges()[idx] = dPhiChange;
    mdsBuild.dphis()[idx] = dPhi;
    mdsBuild.dzs()[idx] = dz;
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
    // hit phi, computed only for the MD anchor hits (the Hits collection does not store it)
    mds.anchorPhi()[idx] = cms::alpakatools::phi(acc, hitsBase.xs()[anchorHitIndex], hitsBase.ys()[anchorHitIndex]);
    // hit eta, computed only for the MD anchor hits (the Hits collection does not store it)
    float const anchorX = hitsBase.xs()[anchorHitIndex];
    float const anchorY = hitsBase.ys()[anchorHitIndex];
    float const anchorZ = hitsBase.zs()[anchorHitIndex];
    mds.anchorEta()[idx] =
        ((anchorZ > 0) - (anchorZ < 0)) *
        alpaka::math::acosh(acc,
                            alpaka::math::sqrt(acc, anchorX * anchorX + anchorY * anchorY + anchorZ * anchorZ) /
                                hitsExtended.rts()[anchorHitIndex]);

    mds.outerX()[idx] = hitsBase.xs()[outerHitIndex];
    mds.outerY()[idx] = hitsBase.ys()[outerHitIndex];
    mds.outerZ()[idx] = hitsBase.zs()[outerHitIndex];
#ifdef CUT_VALUE_DEBUG
    mds.outerRt()[idx] = hitsExtended.rts()[outerHitIndex];
    mds.outerPhi()[idx] = cms::alpakatools::phi(acc, hitsBase.xs()[outerHitIndex], hitsBase.ys()[outerHitIndex]);
    {
      float const outerX = hitsBase.xs()[outerHitIndex];
      float const outerY = hitsBase.ys()[outerHitIndex];
      float const outerZ = hitsBase.zs()[outerHitIndex];
      mds.outerEta()[idx] =
          ((outerZ > 0) - (outerZ < 0)) *
          alpaka::math::acosh(acc,
                              alpaka::math::sqrt(acc, outerX * outerX + outerY * outerY + outerZ * outerZ) /
                                  hitsExtended.rts()[outerHitIndex]);
    }
#endif
  }

  // Overload for callers (Segment.h) that still pass ModulesConst + moduleIndex.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addMDToMemory(TAcc const& acc,
                                                    MiniDoublets mds,
                                                    MiniDoubletsBuild mdsBuild,
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
    addMDToMemory(acc,
                  mds,
                  mdsBuild,
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
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float miniSlopeOf(TAcc const& acc, float rt, const float ptCut) {
    return alpaka::math::asin(acc, alpaka::math::min(acc, rt * k2Rinv1GeVf / ptCut, kSinAlphaMax));
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float dPhiThreshold(
      TAcc const& acc, float miniSlope, ModuleMDData const& mod, float dPhi = 0, float dz = 0) {
    // Barrel flat: no tilt or luminous region correction
    if (mod.subdet == Barrel and mod.side == Center) {
      return miniSlope + mod.sqrtMiniMulsAndPVoff;
    }
    // Barrel tilted
    else if (mod.subdet == Barrel) {
      return miniSlope + alpaka::math::sqrt(acc, mod.miniMulsAndPVoff + mod.miniTilt2 * miniSlope * miniSlope);
    }
    // Endcap: luminous region correction
    else {
      const float miniLum = alpaka::math::abs(acc, dPhi * kDeltaZLum / dz);
      return miniSlope + alpaka::math::sqrt(acc, mod.miniMulsAndPVoff + miniLum * miniLum);
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
    // This is the strip shift scheme that is explained in http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20190607_SDL_Update.pdf (see backup slides)
    // The main feature of this shifting is that the strip hits are shifted to be "aligned" in the line of sight from interaction point to the the pixel hit.
    // (since pixel hit is well defined in 3-d)
    // The strip hit is shifted along the strip detector to be placed in a guessed position where we think they would have actually crossed
    // The size of the radial direction shift due to module separation gap is computed in "radial" size, while the shift is done along the actual strip orientation
    // This means that there may be very very subtle edge effects coming from whether the strip hit is center of the module or the at the edge of the module
    // But this should be relatively minor effect

    float xp;   // pixel x (pixel hit x)
    float yp;   // pixel y (pixel hit y)
    float zp;   // pixel z
    float rtp;  // pixel rt
    float xo;   // old x (before the strip hit is moved up or down)
    float yo;   // old y (before the strip hit is moved up or down)
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

    // Algebraic trig: sin(atan(r/z)) = r/hypot, cos(atan(r/z)) = |z|/hypot
    const float hypot_rz = alpaka::math::sqrt(acc, rtp * rtp + zp * zp);
    const float sinA = rtp / hypot_rz;
    const float cosA = alpaka::math::abs(acc, zp) / hypot_rz;

    // sin(A+B) via angle-addition identity; endcap: B=pi/2 so sin(A+pi/2)=cosA
    // The tilt module on the positive z-axis has negative drdz slope in r-z plane and vice versa
    float sinApB;
    if (isEndcap) {
      sinApB = cosA;
    } else {
      const float inv_hypot_drdz = 1.f / alpaka::math::sqrt(acc, 1.f + mod.drdz * mod.drdz);
      sinApB = sinA * inv_hypot_drdz + cosA * mod.drdz * inv_hypot_drdz;
    }

    float moduleSeparation = mod.moduleSep;

    // Sign flips if the pixel is later layer
    if (mod.isGloballyInner == pHitInverted) {
      moduleSeparation *= -1;
    }

    float drprime = moduleSeparation * sinA / sinApB;

    float drprime_x, drprime_y;  // drprime * {sin,cos}(atan(slope))
    // Algebraic: sin(atan(slope)) = |slope|/sqrt(1+slope^2), cos(atan(slope)) = 1/sqrt(1+slope^2)
    const float slope = mod.slope;
    if (edm::isFinite(slope)) {
      const float inv_hypot_slope = 1.f / alpaka::math::sqrt(acc, 1.f + slope * slope);
      drprime_x = drprime * ((xp > 0.f) - (xp < 0.f)) * alpaka::math::abs(acc, slope) * inv_hypot_slope;
      drprime_y = drprime * ((yp > 0.f) - (yp < 0.f)) * inv_hypot_slope;
    } else {
      drprime_x = drprime * ((xp > 0.f) - (xp < 0.f));
      drprime_y = 0.f;
    }

    float xa = xp + drprime_x;  // anchor x (the guessed position on the strip module plane)
    float ya = yp + drprime_y;  // anchor y

    // Compute the new strip hit position (handle slope = infinity and slope = 0 cases)
    float xn, yn;
    if (edm::isNotFinite(slope)) {
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

  // Per-hit terms of the MD selection (functions of the hit rt and its lower module only), computed once per hit by
  // FillMDHitTerms. Barrel: cut = dPhiThreshold, tanCut = its Pade tangent bound; endcap: cut = miniSlope (asin).
  struct MDHitTerms {
    float cut;
    float tanCut;
  };

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE MDHitTerms
  mdHitTerms(TAcc const& acc, ModuleMDData const& mod, float rt, const float ptCut) {
    MDHitTerms terms;
    if (mod.subdet == Barrel) {
      terms.cut = dPhiThreshold(acc, miniSlopeOf(acc, rt, ptCut), mod);
      const float miniCutSq = terms.cut * terms.cut;
      terms.tanCut = alpaka::math::sqrt(acc, miniCutSq / (1.f - (2.f / 3.f) * miniCutSq));
    } else {
      terms.cut = miniSlopeOf(acc, rt, ptCut);
      terms.tanCut = 0.f;
    }
    return terms;
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
                                                                      float miniCut,
                                                                      float tanMiniCut) {
    dz = zLower - zUpper;
    const float dzCut = mod.moduleType == PS ? 2.f : 10.f;
    const float sign = ((dz > 0) - (dz < 0)) * ((zLower > 0) - (zLower < 0));
    const float invertedcrossercut = (alpaka::math::abs(acc, dz) > 2) * sign;

    if ((alpaka::math::abs(acc, dz) >= dzCut) || (invertedcrossercut > 0)) {
      return false;
    }

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

    // Cross-product pre-checks: Pade [2,2] approximant overestimates tan(miniCut)
    const float crossDPhi = x1 * y2 - x2 * y1;
    const float dotDPhi = x1 * x2 + y1 * y2;
    const float absCrossDPhi = alpaka::math::abs(acc, crossDPhi);
    if (dotDPhi <= 0.f || absCrossDPhi >= tanMiniCut * dotDPhi)
      return false;

    const float rInnerSq = alpaka::math::min(acc, r1sq, r2sq);
    const float dotDPhiChange = dotDPhi - rInnerSq;
    if (dotDPhiChange <= 0.f || absCrossDPhi >= tanMiniCut * dotDPhiChange)
      return false;

    // Cut #2: dphi difference
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3085
    dPhi = alpaka::math::atan2(acc, crossDPhi, dotDPhi);
    noShiftedDphi = mod.isTilted ? cms::alpakatools::deltaPhi(acc, xLower, yLower, xUpper, yUpper) : dPhi;

    if (alpaka::math::abs(acc, dPhi) >= miniCut)
      return false;

    // Cut #3: The dphi change going from lower Hit to upper Hit
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3076
    // dPhiChange should be calculated so that the upper hit has higher rt.
    // The strip hit shifting should guarantee rt ordering, but we check explicitly for safety.
    dPhiChange = alpaka::math::atan2(acc, (r1sq < r2sq) ? crossDPhi : -crossDPhi, dotDPhiChange);
    if (mod.isTilted) {
      noShiftedDphiChange = rtLower < rtUpper ? deltaPhiChange(acc, xLower, yLower, xUpper, yUpper)
                                              : deltaPhiChange(acc, xUpper, yUpper, xLower, yLower);
    } else {
      noShiftedDphiChange = dPhiChange;
    }

    return alpaka::math::abs(acc, dPhiChange) < miniCut;
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
                                                                      const float ptCut,
                                                                      float miniSlope) {
    // Cut #1: dz cut. The dz difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3093
    // For PS module in case when it is tilted a different dz (after the strip hit shift) is calculated later.
    float dz = zLower - zUpper;  // Not const since later it might change depending on the type of module

    const float dzCut = 1.f;

    if (alpaka::math::abs(acc, dz) >= dzCut) {
      return false;
    }
    // Cut #2: drt cut. The drt difference can't be larger than 1cm. (max separation is 4mm for modules in the endcap)
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3100
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

    float x1, y1, x2, y2;
    if (mod.moduleType == PS) {
      if (mod.moduleLayerType == Pixel) {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zUpper;
        x1 = xLower;
        y1 = yLower;
        x2 = xn;
        y2 = yn;
      } else {
        shiftedX = xn;
        shiftedY = yn;
        shiftedZ = zLower;
        x1 = xn;
        y1 = yn;
        x2 = xUpper;
        y2 = yUpper;
      }
    } else {
      shiftedX = xn;
      shiftedY = yn;
      shiftedZ = zUpper;
      x1 = xLower;
      y1 = yLower;
      x2 = xn;
      y2 = yn;
    }

    const float crossDPhi = x1 * y2 - x2 * y1;
    const float dotDPhi = x1 * x2 + y1 * y2;

    // |dPhi| < pi/2
    if (dotDPhi <= 0.f)
      return false;

    // |dPhi| < pi/4 (since dotDPhi > 0, equivalent to |tan(dPhi)| < 1)
    if (alpaka::math::abs(acc, crossDPhi) >= dotDPhi)
      return false;

    // dz needs to change if it is a PS module where the strip hits are shifted in order to properly account for the case when a tilted module falls under "endcap logic"
    // if it was an endcap it will have zero effect
    if (mod.moduleType == PS) {
      dz = mod.moduleLayerType == Pixel ? zLower - zn : zUpper - zn;
    }

    const float absDz = alpaka::math::abs(acc, dz);
    const float tanDPhi = alpaka::math::abs(acc, crossDPhi) / dotDPhi;
    const float miniLum = tanDPhi / absDz * kDeltaZLum;

    const float rt = mod.moduleLayerType == Pixel ? rtLower : rtUpper;
    const float sdSlopeSin = alpaka::math::min(acc, rt * k2Rinv1GeVf / ptCut, kSinAlphaMax);
    const float looseCutDPhi = sdSlopeSin + alpaka::math::sqrt(acc, mod.miniMulsAndPVoff + miniLum * miniLum);

    // Algebraic dPhi pre-check: |sin(dPhi)| < looseCutDPhi.
    // looseCutDPhi = sdSlopeSin + sqrt(mulsAndPVoff + miniLum^2) >= sin(exact_cut)
    // via sin(A+B) <= sin(A) + B, with A = asin(sdSlopeSin), B = sqrt(...).
    // Lagrange identity: cross^2 + dot^2 = |r1|^2*|r2|^2, so sin^2(dPhi) = cross^2/(cross^2+dot^2).
    const float crossSq = crossDPhi * crossDPhi;
    const float r1r2sq = crossSq + dotDPhi * dotDPhi;

    if (crossSq >= looseCutDPhi * looseCutDPhi * r1r2sq)
      return false;

    // dPhiChange pre-check: in endcap, dPhiChange = dPhi * (1+dzFrac)/dzFrac.
    // So |dPhiChange| >= cut implies |dPhi| >= cut * dzFrac/(1+dzFrac).
    // Padding looseCutDPhi with 0.5*s^3 gives an upper bound on the exact angle.
    const float dzFrac = absDz / alpaka::math::abs(acc, zLower);
    const float looseCutDPhiChange =
        (looseCutDPhi + 0.5f * sdSlopeSin * sdSlopeSin * sdSlopeSin) * dzFrac / (1.f + dzFrac);

    if (crossSq >= looseCutDPhiChange * looseCutDPhiChange * r1r2sq)
      return false;

    // Cut #3: dphi
    dPhi = alpaka::math::atan2(acc, crossDPhi, dotDPhi);

    const float miniCut = dPhiThreshold(acc, miniSlope, mod, dPhi, dz);

    if (alpaka::math::abs(acc, dPhi) >= miniCut) {
      return false;
    }

    // Cut #4: dPhiChange
    // Ref to original code: https://github.com/slava77/cms-tkph2-ntuple/blob/184d2325147e6930030d3d1f780136bc2dd29ce6/doubletAnalysis.C#L3119-L3124

    // dzFrac already computed above for dPhiChange pre-check
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    noShiftedDphi = cms::alpakatools::deltaPhi(acc, xLower, yLower, xUpper, yUpper);
    noShiftedDphichange = noShiftedDphi / dzFrac * (1.f + dzFrac);

    return alpaka::math::abs(acc, dPhiChange) < miniCut;
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
                                                                const uint16_t clustSizeCut,
                                                                MDHitTerms const& hitTerms) {
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
                                             hitTerms.cut,
                                             hitTerms.tanCut);
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
                                             ptCut,
                                             hitTerms.cut);
    }
  }

  // Hoist module-constant data once per module to avoid redundant SoA loads per hit pair.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE ModuleMDData
  loadModuleMDData(TAcc const& acc, ModulesConst modules, uint16_t lowerModuleIndex, const float ptCut) {
    ModuleMDData mod;
    mod.lowerModuleIndex = lowerModuleIndex;
    mod.subdet = modules.subdets()[lowerModuleIndex];
    mod.side = modules.sides()[lowerModuleIndex];
    mod.moduleType = modules.moduleType()[lowerModuleIndex];
    mod.moduleLayerType = modules.moduleLayerType()[lowerModuleIndex];
    mod.iL = modules.layers()[lowerModuleIndex] - 1;
    mod.isTilted = (mod.subdet == Barrel && mod.side != Center);
    mod.isGloballyInner = modules.isGloballyInner()[lowerModuleIndex];
    mod.slope = modules.dxdys()[lowerModuleIndex];
    mod.drdz = modules.drdzs()[lowerModuleIndex];
    mod.moduleSep = moduleGapSize(modules, lowerModuleIndex);

    // Pre-compute dPhiThreshold module-constant parts
    float rLayNominal = (mod.subdet == Barrel) ? kMiniRminMeanBarrel[mod.iL] : kMiniRminMeanEndcap[mod.iL];
    mod.miniPVoff = 0.1f / rLayNominal;
    mod.miniMuls = (mod.subdet == Barrel) ? kMiniMulsPtScaleBarrel[mod.iL] * 3.f / ptCut
                                          : kMiniMulsPtScaleEndcap[mod.iL] * 3.f / ptCut;
    mod.miniMulsAndPVoff = mod.miniMuls * mod.miniMuls + mod.miniPVoff * mod.miniPVoff;
    mod.sqrtMiniMulsAndPVoff = alpaka::math::sqrt(acc, mod.miniMulsAndPVoff);

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
    return mod;
  }

  // CountMiniDoublets writes, per lower hit, the decisions of its pairs with upper hit j < kMDPassMaskBits as a bit
  // mask; CreateMiniDoublets evaluates only the accepted pairs and every pair with a larger j.
  constexpr int kMDPassMaskBits = 64;

  // MDHitTerms of the hits that set the MD cut (lower hits if the lower sensor is the pixel one, else upper hits).
  struct FillMDHitTerms {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  HitsExtendedConst hitsExtended,
                                  HitsRangesConst hitsRanges,
                                  MDHitTerms* hitTerms,
                                  const float ptCut) const {
      for (uint16_t lowerModuleIndex : cms::alpakatools::uniform_elements_y(acc, modules.nLowerModules())) {
        if (hitsRanges.hitRangesLower()[lowerModuleIndex] == -1)
          continue;
        ModuleMDData mod = loadModuleMDData(acc, modules, lowerModuleIndex, ptCut);
        const bool onLower = (mod.moduleLayerType == Pixel);
        const int nHits =
            onLower ? hitsRanges.hitRangesnLower()[lowerModuleIndex] : hitsRanges.hitRangesnUpper()[lowerModuleIndex];
        const unsigned int hit0 =
            onLower ? hitsRanges.hitRangesLower()[lowerModuleIndex] : hitsRanges.hitRangesUpper()[lowerModuleIndex];
        for (int hitIndex : cms::alpakatools::uniform_elements_x(acc, nHits))
          hitTerms[hit0 + hitIndex] = mdHitTerms(acc, mod, hitsExtended.rts()[hit0 + hitIndex], ptCut);
      }
    }
  };

  // Upper hits of a lower module in bins of u = x dx + y dy, the coordinate along the module direction (dx, dy) in the
  // transverse plane (which the strip shift keeps); v = y dx - x dy is the coordinate across it. nBins = 0: not binned.
  struct MDUpperBins {
    float dx;
    float dy;
    float uMin;
    float uMax;
    float invW;  // bins per cm
    float vMin;  // v of the upper hits
    float vMax;
    float dvAnchorMin;  // v(anchor) - v(hit) of the upper hits when they are the pixel hits of a strip shift
    float dvAnchorMax;
    float cutMax;  // largest cut term of the upper hits when they set the cut (barrel tanCut, endcap miniSlope)
    int nBins;
  };
  // Binned MD count on the CPU only: on GPUs the serial per-module fill costs more than the count saves.
  constexpr bool kMDBinUpperHits = cms::alpakatools::requires_single_thread_per_block_v<Acc2D>;
  constexpr int kMDBinMinHits = 6;
  constexpr int kMDBinMax = 64;
  constexpr float kMDBinMargin = 0.05f;  // cm on both window edges

  // v of the strip-shift anchor of a pixel hit: the v that the shifted strip hit takes.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float mdAnchorV(
      TAcc const& acc, ModuleMDData const& mod, float dx, float dy, float xHit, float yHit, float zHit, float rtHit) {
    float shifted[3];
    shiftStripHits(acc, mod, shifted, xHit, yHit, zHit, rtHit, xHit, yHit, zHit, rtHit);
    return shifted[1] * dx - shifted[0] * dy;
  }

  // Monotone in u, clamped to [0, nBins - 1].
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int mdBinOf(MDUpperBins const& upperBins, float uHit) {
    float position = (uHit - upperBins.uMin) * upperBins.invW;
    position = position > 0.f ? position : 0.f;
    position = position < float(upperBins.nBins - 1) ? position : float(upperBins.nBins - 1);
    return int(position);
  }

  // Bins the upper hits of each lower module (count, prefix, scatter): binStart holds the nBins + 1 <= nUpper bin offsets
  // and binPerm the upper hit indices in bin order, both at the upper hits' positions.
  struct FillMDUpperBins {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  HitsBaseConst hitsBase,
                                  HitsExtendedConst hitsExtended,
                                  HitsRangesConst hitsRanges,
                                  const MDHitTerms* hitTerms,
                                  MDUpperBins* bins,
                                  uint16_t* binStart,
                                  uint16_t* binPerm,
                                  const float ptCut) const {
      for (uint16_t lowerModuleIndex : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        MDUpperBins upperBins{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0};
        const int nU = hitsRanges.hitRangesnUpper()[lowerModuleIndex];
        if (hitsRanges.hitRangesLower()[lowerModuleIndex] == -1 || nU < kMDBinMinHits) {
          bins[lowerModuleIndex] = upperBins;
          continue;
        }
        const unsigned int h0 = hitsRanges.hitRangesUpper()[lowerModuleIndex];
        const ModuleMDData mod = loadModuleMDData(acc, modules, lowerModuleIndex, ptCut);
        // Shifted modules: the direction (1, slope) of the strip shift. Flat barrel: perpendicular to the module's phi.
        if (mod.isTilted || mod.subdet == Endcap) {
          upperBins.dx = edm::isFinite(mod.slope) ? 1.f / alpaka::math::sqrt(acc, 1.f + mod.slope * mod.slope) : 0.f;
          upperBins.dy = edm::isFinite(mod.slope) ? mod.slope * upperBins.dx : 1.f;
        } else {
          upperBins.dx = -alpaka::math::sin(acc, modules.phi()[lowerModuleIndex]);
          upperBins.dy = alpaka::math::cos(acc, modules.phi()[lowerModuleIndex]);
        }
        const float dx = upperBins.dx;
        const float dy = upperBins.dy;
        const bool cutFromUpper = mod.moduleLayerType != Pixel;
        const bool upperAnchors =
            mod.moduleType == PS && mod.moduleLayerType != Pixel && (mod.isTilted || mod.subdet == Endcap);
        constexpr float inf = std::numeric_limits<float>::infinity();
        upperBins.uMin = upperBins.vMin = upperBins.dvAnchorMin = inf;
        upperBins.uMax = upperBins.vMax = upperBins.dvAnchorMax = -inf;
        bool finite = true;
        for (int j = 0; j < nU; ++j) {
          const unsigned int hitIndex = h0 + j;
          const float xHit = hitsBase.xs()[hitIndex];
          const float yHit = hitsBase.ys()[hitIndex];
          const float uHit = xHit * dx + yHit * dy;
          const float vHit = yHit * dx - xHit * dy;
          upperBins.uMin = alpaka::math::min(acc, upperBins.uMin, uHit);
          upperBins.uMax = alpaka::math::max(acc, upperBins.uMax, uHit);
          upperBins.vMin = alpaka::math::min(acc, upperBins.vMin, vHit);
          upperBins.vMax = alpaka::math::max(acc, upperBins.vMax, vHit);
          if (cutFromUpper) {
            const float cut = mod.subdet == Barrel ? hitTerms[hitIndex].tanCut : hitTerms[hitIndex].cut;
            finite = finite && edm::isFinite(cut);
            upperBins.cutMax = alpaka::math::max(acc, upperBins.cutMax, cut);
          }
          if (upperAnchors) {
            const float dv =
                mdAnchorV(acc, mod, dx, dy, xHit, yHit, hitsBase.zs()[hitIndex], hitsExtended.rts()[hitIndex]) - vHit;
            finite = finite && edm::isFinite(dv);
            upperBins.dvAnchorMin = alpaka::math::min(acc, upperBins.dvAnchorMin, dv);
            upperBins.dvAnchorMax = alpaka::math::max(acc, upperBins.dvAnchorMax, dv);
          }
        }
        const float width = upperBins.uMax - upperBins.uMin;
        if (!finite || !(width > 0.f) || !edm::isFinite(width) || !edm::isFinite(upperBins.vMax - upperBins.vMin)) {
          upperBins.nBins = 0;
          bins[lowerModuleIndex] = upperBins;
          continue;
        }
        upperBins.nBins = nU - 1 < kMDBinMax ? nU - 1 : kMDBinMax;
        upperBins.invW = float(upperBins.nBins) / width;
        uint16_t start[kMDBinMax + 1];
        for (int k = 0; k <= upperBins.nBins; ++k)
          start[k] = 0;
        for (int j = 0; j < nU; ++j)
          ++start[mdBinOf(upperBins, hitsBase.xs()[h0 + j] * dx + hitsBase.ys()[h0 + j] * dy) + 1];
        for (int k = 1; k <= upperBins.nBins; ++k)
          start[k] += start[k - 1];
        for (int k = 0; k <= upperBins.nBins; ++k)
          binStart[h0 + k] = start[k];
        for (int j = 0; j < nU; ++j)
          binPerm[h0 + start[mdBinOf(upperBins, hitsBase.xs()[h0 + j] * dx + hitsBase.ys()[h0 + j] * dy)]++] = j;
        bins[lowerModuleIndex] = upperBins;
      }
    }
  };

  // u window holding every upper hit that can pass the MD angular pre-checks with this lower hit (false: no window), for
  // P1 = (u1, V1), P2 = (u2, V2) the selection's lower/upper-side points (a shifted strip hit takes v of the anchor).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool mdUpperWindow(TAcc const& acc,
                                                    ModuleMDData const& mod,
                                                    MDUpperBins const& ub,
                                                    float lowerCut,
                                                    float lowerTanCut,
                                                    float xLower,
                                                    float yLower,
                                                    float zLower,
                                                    float rtLower,
                                                    float& uLo,
                                                    float& uHi) {
    const float dx = ub.dx;
    const float dy = ub.dy;
    const float u1 = xLower * dx + yLower * dy;
    const float v1 = yLower * dx - xLower * dy;
    const bool shifted = mod.isTilted || mod.subdet == Endcap;
    const bool upperAnchors = shifted && mod.moduleType == PS && mod.moduleLayerType != Pixel;
    // v -> -v keeps |cross| and dot: orient so that V1 > 0.
    const float sgn = (upperAnchors ? ub.vMin : v1) < 0.f ? -1.f : 1.f;
    const float v2Lo = sgn > 0.f ? ub.vMin : -ub.vMax;  // oriented v range of the upper hits
    const float v2Hi = sgn > 0.f ? ub.vMax : -ub.vMin;
    // V1 = V2 + a: ranges of V1, V2 and a.
    float V1Lo, V1Hi, V2Lo, V2Hi, aLo, aHi;
    if (upperAnchors) {
      aLo = sgn > 0.f ? ub.dvAnchorMin : -ub.dvAnchorMax;
      aHi = sgn > 0.f ? ub.dvAnchorMax : -ub.dvAnchorMin;
      V2Lo = v2Lo;
      V2Hi = v2Hi;
      V1Lo = v2Lo + aLo;
      V1Hi = v2Hi + aHi;
    } else {
      V1Lo = V1Hi = sgn * v1;
      if (shifted) {
        V2Lo = V2Hi = sgn * mdAnchorV(acc, mod, dx, dy, xLower, yLower, zLower, rtLower);
      } else {
        V2Lo = v2Lo;
        V2Hi = v2Hi;
      }
      aLo = V1Lo - V2Hi;
      aHi = V1Lo - V2Lo;
    }
    if (!(V1Lo > 1.f) || !(V2Lo > 1.f))
      return false;
    const float uAbsMax = alpaka::math::max(acc, alpaka::math::abs(acc, ub.uMin), alpaka::math::abs(acc, ub.uMax));
    if (mod.subdet == Barrel) {
      // T = tanCut: |P1 x P2| < T (P1.P2 - min(r1^2, r2^2)). Dv > 0: angle(P2 - P1, P1) < atan T (as |t2| T < 1);
      // Dv < 0: angle(P1 - P2, P2) < atan T and angle(P1, P2) < atan T, so angle(P1 - P2, P1) < 2 atan T.
      float tanCut = mod.moduleLayerType == Pixel ? lowerTanCut : ub.cutMax;
      const bool outward = -aHi > 0.01f;  // Dv = -a > 0
      if (outward) {
        if (!(tanCut * uAbsMax < 0.8f * V2Lo))
          return false;
      } else if (aLo > 0.01f) {
        tanCut = 2.f * tanCut / (1.f - tanCut * tanCut);  // tan(2 atan T), valid for T < 1
        if (!(tanCut > 0.f))
          return false;
      } else {
        return false;
      }
      const float tLo = alpaka::math::min(acc, u1 / V1Lo, u1 / V1Hi);
      const float tHi = alpaka::math::max(acc, u1 / V1Lo, u1 / V1Hi);
      if (!(alpaka::math::max(acc, alpaka::math::abs(acc, tLo), alpaka::math::abs(acc, tHi)) * tanCut < 0.8f))
        return false;
      const float kLo = (tLo - tanCut) / (1.f + tLo * tanCut);
      const float kHi = (tHi + tanCut) / (1.f - tHi * tanCut);
      if (outward) {  // D = (u2 - u1, -a)
        uLo = u1 + alpaka::math::min(acc, -aHi * kLo, -aLo * kLo);
        uHi = u1 + alpaka::math::max(acc, -aHi * kHi, -aLo * kHi);
      } else {  // -D = (u1 - u2, a)
        uLo = u1 - alpaka::math::max(acc, aLo * kHi, aHi * kHi);
        uHi = u1 - alpaka::math::min(acc, aLo * kLo, aHi * kLo);
      }
    } else {
      // |sin dPhi| < LC <= f B + g |tan dPhi| and |tan dPhi| < sqrt2 |sin dPhi| (|dPhi| < pi/4): |sin dPhi| < sinMax.
      const float absZ = alpaka::math::abs(acc, zLower);
      // PS: |dz| = |z - z(anchor)| = the module separation; 2S: the dz cut.
      const float dzFrac = (mod.moduleType == PS ? 1.001f * mod.moduleSep : 1.f) / absZ;
      const float den = 1.f - 1.4143f * kDeltaZLum / absZ;
      const float slopeSin = mod.moduleLayerType == Pixel ? lowerCut : ub.cutMax;  // miniSlope >= sdSlopeSin
      const float sinMax = 1.001f * dzFrac / (1.f + dzFrac) *
                           (slopeSin + 0.5f * slopeSin * slopeSin * slopeSin + mod.sqrtMiniMulsAndPVoff) / den;
      if (!(den > 0.1f) || !(sinMax < 0.5f))
        return false;
      const float crossMax = sinMax * alpaka::math::sqrt(acc, u1 * u1 + V1Hi * V1Hi) *
                             alpaka::math::sqrt(acc, uAbsMax * uAbsMax + V2Hi * V2Hi);
      // u2 = (u1 V2 -+ crossMax) / (V2 + a) is monotone in V2 and in a: extremes at the corners.
      const float V2s[2] = {V2Lo, V2Hi};
      const float aRange[2] = {upperAnchors ? aLo : V1Lo - V2Lo, upperAnchors ? aHi : V1Lo - V2Lo};
      uLo = std::numeric_limits<float>::infinity();
      uHi = -uLo;
      for (float V2 : V2s) {
        for (float dvAnchor : aRange) {
          uLo = alpaka::math::min(acc, uLo, (u1 * V2 - crossMax) / (V2 + dvAnchor));
          uHi = alpaka::math::max(acc, uHi, (u1 * V2 + crossMax) / (V2 + dvAnchor));
        }
      }
    }
    uLo -= kMDBinMargin;
    uHi += kMDBinMargin;
    return edm::isFinite(uLo) && edm::isFinite(uHi);
  }

  struct CreateMiniDoublets {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  HitsBaseConst hitsBase,
                                  HitsExtendedConst hitsExtended,
                                  HitsRangesConst hitsRanges,
                                  MiniDoublets mds,
                                  MiniDoubletsBuild mdsBuild,
                                  MiniDoubletsOccupancy mdsOccupancy,
                                  ObjectRangesConst ranges,
                                  const uint64_t* mdPassMask,
                                  const MDHitTerms* hitTerms,
                                  const float ptCut,
                                  const uint16_t clustSizeCut) const {
      for (uint16_t lowerModuleIndex : cms::alpakatools::uniform_elements_y(acc, modules.nLowerModules())) {
        int nLowerHits = hitsRanges.hitRangesnLower()[lowerModuleIndex];
        int nUpperHits = hitsRanges.hitRangesnUpper()[lowerModuleIndex];
        if (hitsRanges.hitRangesLower()[lowerModuleIndex] == -1)
          continue;
        unsigned int upHitArrayIndex = hitsRanges.hitRangesUpper()[lowerModuleIndex];
        unsigned int loHitArrayIndex = hitsRanges.hitRangesLower()[lowerModuleIndex];

        ModuleMDData mod = loadModuleMDData(acc, modules, lowerModuleIndex, ptCut);

        for (int lowerHitIndex : cms::alpakatools::uniform_elements_x(acc, nLowerHits)) {
          unsigned int lowerHitArrayIndex = loHitArrayIndex + lowerHitIndex;
          float xLower = hitsBase.xs()[lowerHitArrayIndex];
          float yLower = hitsBase.ys()[lowerHitArrayIndex];
          float zLower = hitsBase.zs()[lowerHitArrayIndex];
          float rtLower = hitsExtended.rts()[lowerHitArrayIndex];
          uint16_t clustSizeLower = hitsBase.clustsize()[lowerHitArrayIndex];

          auto tryAddMD = [&](int upperHitIndex) {
            unsigned int upperHitArrayIndex = upHitArrayIndex + upperHitIndex;
            float xUpper = hitsBase.xs()[upperHitArrayIndex];
            float yUpper = hitsBase.ys()[upperHitArrayIndex];
            float zUpper = hitsBase.zs()[upperHitArrayIndex];
            float rtUpper = hitsExtended.rts()[upperHitArrayIndex];
            uint16_t clustSizeUpper = hitsBase.clustsize()[upperHitArrayIndex];

            const MDHitTerms cutHitTerms =
                hitTerms[mod.moduleLayerType == Pixel ? lowerHitArrayIndex : upperHitArrayIndex];
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
                                                     clustSizeCut,
                                                     cutHitTerms);
            if (!success)
              return;
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
                            mdsBuild,
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
          };

          // Ascending upper hit index, as in the full pair loop: the recorded pairs, then the unrecorded tail.
          for (uint64_t passBits = mdPassMask[lowerHitArrayIndex]; passBits != 0; passBits &= passBits - 1)
            tryAddMD(alpaka::ffs(acc, static_cast<std::int64_t>(passBits)) - 1);
          for (int upperHitIndex = kMDPassMaskBits; upperHitIndex < nUpperHits; ++upperHitIndex)
            tryAddMD(upperHitIndex);
        }
      }
    }
  };

  struct CountMiniDoublets {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  HitsBaseConst hitsBase,
                                  HitsExtendedConst hitsExtended,
                                  HitsRangesConst hitsRanges,
                                  ObjectRanges ranges,
                                  uint64_t* mdPassMask,
                                  const MDHitTerms* hitTerms,
                                  const MDUpperBins* bins,
                                  const uint16_t* binStart,
                                  const uint16_t* binPerm,
                                  const float ptCut,
                                  const uint16_t clustSizeCut) const {
      for (uint16_t lowerModuleIndex : cms::alpakatools::uniform_elements_y(acc, modules.nLowerModules())) {
        int nLowerHits = hitsRanges.hitRangesnLower()[lowerModuleIndex];
        int nUpperHits = hitsRanges.hitRangesnUpper()[lowerModuleIndex];
        if (hitsRanges.hitRangesLower()[lowerModuleIndex] == -1)
          continue;
        unsigned int upHitArrayIndex = hitsRanges.hitRangesUpper()[lowerModuleIndex];
        unsigned int loHitArrayIndex = hitsRanges.hitRangesLower()[lowerModuleIndex];

        ModuleMDData mod = loadModuleMDData(acc, modules, lowerModuleIndex, ptCut);
        const MDUpperBins ub = kMDBinUpperHits ? bins[lowerModuleIndex] : MDUpperBins{};

        for (int lowerHitIndex : cms::alpakatools::uniform_elements_x(acc, nLowerHits)) {
          unsigned int lowerHitArrayIndex = loHitArrayIndex + lowerHitIndex;
          float xLower = hitsBase.xs()[lowerHitArrayIndex];
          float yLower = hitsBase.ys()[lowerHitArrayIndex];
          float zLower = hitsBase.zs()[lowerHitArrayIndex];
          float rtLower = hitsExtended.rts()[lowerHitArrayIndex];
          uint16_t clustSizeLower = hitsBase.clustsize()[lowerHitArrayIndex];

          // Visit only the upper hits in the bins of the u window (all of them without a window).
          int kBegin = 0, kEnd = nUpperHits;
          const uint16_t* order = nullptr;
          if (kMDBinUpperHits && ub.nBins > 0) {
            const MDHitTerms lowerTerms =
                mod.moduleLayerType == Pixel ? hitTerms[lowerHitArrayIndex] : MDHitTerms{0.f, 0.f};
            float uLo, uHi;
            if (mdUpperWindow(
                    acc, mod, ub, lowerTerms.cut, lowerTerms.tanCut, xLower, yLower, zLower, rtLower, uLo, uHi)) {
              kBegin = binStart[upHitArrayIndex + mdBinOf(ub, uLo)];
              kEnd = binStart[upHitArrayIndex + mdBinOf(ub, uHi) + 1];
              order = binPerm + upHitArrayIndex;
            }
          }

          uint64_t passBits = 0;
          int nPass = 0;
          for (int k = kBegin; k < kEnd; ++k) {
            const int upperHitIndex = order ? order[k] : k;
            unsigned int upperHitArrayIndex = upHitArrayIndex + upperHitIndex;
            float xUpper = hitsBase.xs()[upperHitArrayIndex];
            float yUpper = hitsBase.ys()[upperHitArrayIndex];
            float zUpper = hitsBase.zs()[upperHitArrayIndex];
            float rtUpper = hitsExtended.rts()[upperHitArrayIndex];
            uint16_t clustSizeUpper = hitsBase.clustsize()[upperHitArrayIndex];

            const MDHitTerms cutHitTerms =
                hitTerms[mod.moduleLayerType == Pixel ? lowerHitArrayIndex : upperHitArrayIndex];
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
                                                     clustSizeCut,
                                                     cutHitTerms);
            if (success) {
              ++nPass;
              if (upperHitIndex < kMDPassMaskBits)
                passBits |= uint64_t(1) << upperHitIndex;
            }
          }
          mdPassMask[lowerHitArrayIndex] = passBits;
          if (nPass > 0)
            alpaka::atomicAdd(
                acc, &ranges.miniDoubletModuleOccupancy()[lowerModuleIndex], nPass, alpaka::hierarchy::Threads{});
        }
      }
    }
  };

  struct CreateMDArrayRangesGPU {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ModulesConst modules, ObjectRanges ranges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      // Declare variables in shared memory and set to 0
      int& nTotalMDs = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalMDs = 0;
      }
      alpaka::syncBlockThreads(acc);

      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        const int occupancy = ranges.miniDoubletModuleOccupancy()[i];
        const unsigned int nTotMDs = alpaka::atomicAdd(acc, &nTotalMDs, occupancy, alpaka::hierarchy::Threads{});
        ranges.miniDoubletModuleIndices()[i] = nTotMDs;
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
