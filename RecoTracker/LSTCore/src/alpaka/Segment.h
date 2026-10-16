#ifndef RecoTracker_LSTCore_src_alpaka_Segment_h
#define RecoTracker_LSTCore_src_alpaka_Segment_h

#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/SegmentsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/HitsSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"

#include "NeuralNetwork.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isTighterTiltedModules_seg(ModulesConst modules, unsigned int moduleIndex) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modules.subdets()[moduleIndex];
    short layer = modules.layers()[moduleIndex];
    short side = modules.sides()[moduleIndex];
    short rod = modules.rods()[moduleIndex];

    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isTighterTiltedModules_seg(short subdet, short layer, short side, short rod) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(short layer, short ring, short subdet, short side, short rod) {
    unsigned int iL = layer - 1;
    unsigned int iR = ring - 1;

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = kMiniDeltaFlat[iL];
    } else if (isTighterTiltedModules_seg(subdet, layer, side, rod)) {
      moduleSeparation = kMiniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = kMiniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = kMiniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(ModulesConst modules, unsigned int moduleIndex) {
    unsigned int iL = modules.layers()[moduleIndex] - 1;
    unsigned int iR = modules.rings()[moduleIndex] - 1;
    short subdet = modules.subdets()[moduleIndex];
    short side = modules.sides()[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = kMiniDeltaFlat[iL];
    } else if (isTighterTiltedModules_seg(modules, moduleIndex)) {
      moduleSeparation = kMiniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = kMiniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = kMiniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  }

  // Pre-loaded module data for segment creation, eliminating redundant SoA lookups
  // in inner loops. Populated once per module (outer/middle loop level).
  struct ModuleSegData {
    float drdz;
    float moduleGapSize;
    float segMiniTilt2;  // 0.25 * kPixelPSZpitch^2 * drdz^2 / (1+drdz^2) / gap^2; 0 if not tilted
    float sdMuls;        // kMiniMulsPtScale[iL] * 3 / ptCut
    float edgeDx;        // 2S endcap strip half-vector (0 elsewhere)
    float edgeDy;

    unsigned int iL;  // layer - 1

    short subdet;
    short side;
    short layer;
    short moduleType;

    bool isTilted;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE ModuleSegData loadModuleSegData(ModulesConst modules,
                                                                 uint16_t moduleIndex,
                                                                 const float ptCut) {
    ModuleSegData mod;
    mod.subdet = modules.subdets()[moduleIndex];
    mod.side = modules.sides()[moduleIndex];
    mod.layer = modules.layers()[moduleIndex];
    mod.iL = mod.layer - 1;
    mod.moduleType = modules.moduleType()[moduleIndex];
    mod.drdz = modules.drdzs()[moduleIndex];
    mod.edgeDx = modules.edgeDx()[moduleIndex];
    mod.edgeDy = modules.edgeDy()[moduleIndex];
    mod.moduleGapSize = moduleGapSize_seg(modules, moduleIndex);
    mod.isTilted = (mod.subdet == Barrel and mod.side != Center);
    mod.segMiniTilt2 = mod.isTilted ? (0.25f * (kPixelPSZpitch * kPixelPSZpitch) * (mod.drdz * mod.drdz) /
                                       (1.f + mod.drdz * mod.drdz) / (mod.moduleGapSize * mod.moduleGapSize))
                                    : 0.f;
    mod.sdMuls = (mod.subdet == Barrel) ? kMiniMulsPtScaleBarrel[mod.iL] * 3.f / ptCut
                                        : kMiniMulsPtScaleEndcap[mod.iL] * 3.f / ptCut;
    return mod;
  }

  // Terms of the segment selection that depend only on the outer MD's anchor rt: computed once per MD.
  struct SegOuterMDTerms {
    float sdSlopeSin;
    float sdSlope;
    float dzDrtScale;  // tan(asin(s))/asin(s), barrel
    float drtDzScale;  // asin(s)/tan(asin(s)), endcap
  };

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE SegOuterMDTerms segOuterMDTerms(TAcc const& acc, float rtOut, const float ptCut) {
    SegOuterMDTerms terms;
    terms.sdSlopeSin = alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax);
    terms.sdSlope = alpaka::math::asin(acc, terms.sdSlopeSin);
    // Exact: tan(asin(s))/asin(s) = s/(asin(s)*sqrt(1-s^2)), eliminates tan call
    terms.dzDrtScale =
        terms.sdSlopeSin / (terms.sdSlope * alpaka::math::sqrt(acc, 1.f - terms.sdSlopeSin * terms.sdSlopeSin));
    // Exact: asin(s)/tan(asin(s)) = asin(s)*sqrt(1-s^2)/s, eliminates tan call
    terms.drtDzScale =
        terms.sdSlope * alpaka::math::sqrt(acc, 1.f - terms.sdSlopeSin * terms.sdSlopeSin) / terms.sdSlopeSin;
    return terms;
  }

  // SegOuterMDTerms of every OT MD, read by the count kernel (and CreateSegments' unrecorded tail) per MD pair.
  struct FillSegOuterMDTerms {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, MiniDoubletsConst mds, unsigned int nMDs, SegOuterMDTerms* terms, const float ptCut) const {
      for (unsigned int mdIndex : cms::alpakatools::uniform_elements(acc, nMDs))
        terms[mdIndex] = segOuterMDTerms(acc, mds.anchorRt()[mdIndex], ptCut);
    }
  };

  // Returns false (thresholds not filled) when the MD-MD cut |dAlphaInnerMDOuterMD| < dAlphaThresholdValues[2]
  // is already failed with asin(s) replaced by its upper bound s/sqrt(1-s^2): the asin is then skipped.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool dAlphaThreshold(TAcc const& acc,
                                                      float* dAlphaThresholdValues,
                                                      ModuleSegData const& innerMod,
                                                      ModuleSegData const& outerMod,
                                                      MiniDoubletsBuildConst mdsBuild,
                                                      float xIn,
                                                      float yIn,
                                                      float zIn,
                                                      float rtIn,
                                                      float xOut,
                                                      float yOut,
                                                      float zOut,
                                                      float rtOut,
                                                      unsigned int innerMDIndex,
                                                      unsigned int outerMDIndex,
                                                      const float ptCut,
                                                      float dAlphaInnerMDOuterMD,
                                                      float& dAlphaBfieldOut,
                                                      float& dAlphaResMulsOut) {
    const float sdMuls = innerMod.sdMuls;

    //more accurate then outer rt - inner rt
    float segmentDr = alpaka::math::sqrt(acc, (yOut - yIn) * (yOut - yIn) + (xOut - xIn) * (xOut - xIn));
    const float sinBfield = alpaka::math::min(acc, segmentDr * k2Rinv1GeVf / ptCut, kSinAlphaMax);

    // Unique stuff for the segment dudes alone
    const float miniDelta = innerMod.moduleGapSize;
    float dAlpha_res_inner =
        0.02f / miniDelta * (innerMod.subdet == Barrel ? 1.0f : alpaka::math::abs(acc, zIn) / rtIn);
    float dAlpha_res_outer =
        0.02f / miniDelta * (outerMod.subdet == Barrel ? 1.0f : alpaka::math::abs(acc, zOut) / rtOut);

    float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;
    const float dAlphaResMuls = alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);

    // 1e-5 relative margin: covers the float rounding of the bound and of asin (a few 1e-7).
    const float asinUpper = sinBfield / alpaka::math::sqrt(acc, 1.f - sinBfield * sinBfield) * 1.00001f;
    if (alpaka::math::abs(acc, dAlphaInnerMDOuterMD) >= asinUpper + dAlphaResMuls)
      return false;

    const float dAlpha_Bfield = alpaka::math::asin(acc, sinBfield);

    float sdLumForInnerMini2;
    float sdLumForOuterMini2;

    if (innerMod.subdet == Barrel) {
      sdLumForInnerMini2 = innerMod.segMiniTilt2 * (dAlpha_Bfield * dAlpha_Bfield);
    } else {
      sdLumForInnerMini2 = (mdsBuild.dphis()[innerMDIndex] * mdsBuild.dphis()[innerMDIndex]) *
                           (kDeltaZLum * kDeltaZLum) / (mdsBuild.dzs()[innerMDIndex] * mdsBuild.dzs()[innerMDIndex]);
    }

    if (outerMod.subdet == Barrel) {
      sdLumForOuterMini2 = outerMod.segMiniTilt2 * (dAlpha_Bfield * dAlpha_Bfield);
    } else {
      sdLumForOuterMini2 = (mdsBuild.dphis()[outerMDIndex] * mdsBuild.dphis()[outerMDIndex]) *
                           (kDeltaZLum * kDeltaZLum) / (mdsBuild.dzs()[outerMDIndex] * mdsBuild.dzs()[outerMDIndex]);
    }

    if (innerMod.subdet == Barrel and innerMod.side == Center) {
      dAlphaThresholdValues[0] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[0] =
          dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForInnerMini2);
    }

    if (outerMod.subdet == Barrel and outerMod.side == Center) {
      dAlphaThresholdValues[1] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[1] =
          dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForOuterMini2);
    }

    //Inner to outer
    dAlphaThresholdValues[2] = dAlpha_Bfield + dAlphaResMuls;

    // Returned for the line residual's resolution.
    dAlphaBfieldOut = dAlpha_Bfield;
    dAlphaResMulsOut = dAlphaResMuls;
    return true;
  }

  // Line residual cut in units of its resolution (99.4% of true above-cut segments pass).
  HOST_DEVICE_CONSTANT float kLsLineResidCut = 0.75f;

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addSegmentToMemory(Segments segments,
                                                         unsigned int lowerMDIndex,
                                                         unsigned int upperMDIndex,
                                                         uint16_t outerLowerModuleIndex,
                                                         float dPhiChange,
                                                         float dPhiChangeMin,
                                                         float dPhiChangeMax,
                                                         float dPhiChangeOut,
#ifdef CUT_VALUE_DEBUG
                                                         float dPhi,
                                                         float dPhiMin,
                                                         float dPhiMax,
                                                         float zHi,
                                                         float zLo,
                                                         float rtHi,
                                                         float rtLo,
                                                         float dAlphaInner,
                                                         float dAlphaOuter,
                                                         float dAlphaInnerOuter,
#endif
                                                         unsigned int idx) {
    segments.mdIndices()[idx][0] = lowerMDIndex;
    segments.mdIndices()[idx][1] = upperMDIndex;
    segments.outerLowerModuleIndices()[idx] = outerLowerModuleIndex;

    segments.dPhiChanges()[idx] = __F2H(dPhiChange);
#ifdef CUT_VALUE_DEBUG
    segments.dPhis()[idx] = __F2H(dPhi);
    segments.dPhiMins()[idx] = __F2H(dPhiMin);
    segments.dPhiMaxs()[idx] = __F2H(dPhiMax);
#endif
    segments.dPhiChangeMins()[idx] = __F2H(dPhiChangeMin);
    segments.dPhiChangeMaxs()[idx] = __F2H(dPhiChangeMax);
    segments.dPhiChangeOuts()[idx] = dPhiChangeOut;

#ifdef CUT_VALUE_DEBUG
    segments.zHis()[idx] = __F2H(zHi);
    segments.zLos()[idx] = __F2H(zLo);
    segments.rtHis()[idx] = __F2H(rtHi);
    segments.rtLos()[idx] = __F2H(rtLo);
    segments.dAlphaInners()[idx] = __F2H(dAlphaInner);
    segments.dAlphaOuters()[idx] = __F2H(dAlphaOuter);
    segments.dAlphaInnerOuters()[idx] = __F2H(dAlphaInnerOuter);
#endif
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelSegmentToMemory(TAcc const& acc,
                                                              Segments segments,
                                                              PixelSegments pixelSegments,
                                                              PixelSeedsConst pixelSeeds,
                                                              MiniDoubletsConst mds,
                                                              unsigned int innerMDIndex,
                                                              unsigned int outerMDIndex,
                                                              uint16_t pixelModuleIndex,
                                                              const Params_pLS::ArrayUxHits& hitIdxs,
                                                              float dPhiChange,
                                                              unsigned int idx,
                                                              unsigned int pixelSegmentArrayIndex,
                                                              float score) {
    segments.mdIndices()[idx][0] = innerMDIndex;
    segments.mdIndices()[idx][1] = outerMDIndex;
    segments.outerLowerModuleIndices()[idx] = pixelModuleIndex;
    segments.dPhiChanges()[idx] = __F2H(dPhiChange);

    pixelSegments.isDup()[pixelSegmentArrayIndex] = false;
    pixelSegments.partOfPT5()[pixelSegmentArrayIndex] = false;
    pixelSegments.score()[pixelSegmentArrayIndex] = score;
    pixelSegments.pLSHitsIdxs()[pixelSegmentArrayIndex] = hitIdxs;

    //computing circle parameters
    /*
    The two anchor hits are r3PCA and r3LH. p3PCA pt, eta, phi is hitIndex1 x, y, z
    */
    float circleRadius = mds.outerX()[innerMDIndex] / (2 * k2Rinv1GeVf);
    float circlePhi = mds.outerZ()[innerMDIndex];
    float candidateCenterXs[] = {mds.anchorX()[innerMDIndex] + circleRadius * alpaka::math::sin(acc, circlePhi),
                                 mds.anchorX()[innerMDIndex] - circleRadius * alpaka::math::sin(acc, circlePhi)};
    float candidateCenterYs[] = {mds.anchorY()[innerMDIndex] - circleRadius * alpaka::math::cos(acc, circlePhi),
                                 mds.anchorY()[innerMDIndex] + circleRadius * alpaka::math::cos(acc, circlePhi)};

    //check which of the circles can accommodate r3LH better (we won't get perfect agreement)
    float bestChiSquared = std::numeric_limits<float>::infinity();
    float chiSquared;
    size_t bestIndex;
    for (size_t i = 0; i < 2; i++) {
      chiSquared = alpaka::math::abs(acc,
                                     alpaka::math::sqrt(acc,
                                                        (mds.anchorX()[outerMDIndex] - candidateCenterXs[i]) *
                                                                (mds.anchorX()[outerMDIndex] - candidateCenterXs[i]) +
                                                            (mds.anchorY()[outerMDIndex] - candidateCenterYs[i]) *
                                                                (mds.anchorY()[outerMDIndex] - candidateCenterYs[i])) -
                                         circleRadius);
      if (chiSquared < bestChiSquared) {
        bestChiSquared = chiSquared;
        bestIndex = i;
      }
    }
    pixelSegments.circleCenterX()[pixelSegmentArrayIndex] = candidateCenterXs[bestIndex];
    pixelSegments.circleCenterY()[pixelSegmentArrayIndex] = candidateCenterYs[bestIndex];
    pixelSegments.circleRadius()[pixelSegmentArrayIndex] = circleRadius;

    float plsEmbed[Params_pLS::kEmbed];
    plsembdnn::runEmbed(acc,
                        pixelSeeds.eta()[pixelSegmentArrayIndex],
                        pixelSeeds.etaErr()[pixelSegmentArrayIndex],
                        pixelSeeds.phi()[pixelSegmentArrayIndex],
                        pixelSegments.circleCenterX()[pixelSegmentArrayIndex],
                        pixelSegments.circleCenterY()[pixelSegmentArrayIndex],
                        pixelSegments.circleRadius()[pixelSegmentArrayIndex],
                        pixelSeeds.ptIn()[pixelSegmentArrayIndex],
                        pixelSeeds.ptErr()[pixelSegmentArrayIndex],
                        static_cast<bool>(pixelSeeds.isQuad()[pixelSegmentArrayIndex]),
                        plsEmbed);

    CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_pLS::kEmbed; ++k) {
      pixelSegments.plsEmbed()[pixelSegmentArrayIndex][k] = plsEmbed[k];
    }
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passDeltaPhiCutsBarrel(TAcc const& acc,
                                                             MiniDoubletsConst mds,
                                                             unsigned int innerMD,
                                                             unsigned int outerMD,
                                                             const float xIn,
                                                             const float yIn,
                                                             const float xOut,
                                                             const float yOut,
                                                             const float rtIn,
                                                             const float rtOut,
                                                             const float sdSlopeSin,
                                                             const float sdMulsAndPVoff,
                                                             const float sdCut,
                                                             float& dPhi) {
    // Loose sin^2-based pre-check for dPhi using x/y coordinates directly,
    // avoiding anchorPhi SoA reads + reducePhiRange for pairs that clearly fail.
    //
    // Check: |sin(dPhi)| < L where L = sdSlopeSin + sdMulsAndPVoff (looseCutDPhi).
    // This is strictly looser than |dPhi| < sdCut because L = s + M >= sin(asin(s) + M)
    // = sin(sdCut), provable via f(M) = s+M - sin(asin(s)+M), f(0)=0, f'(M)=1-cos(...)>=0.
    // Using Lagrange identity (cross^2+dot^2 = rtIn^2*rtOut^2): |cross| >= L*rtIn*rtOut.
    const float crossDPhi = xIn * yOut - xOut * yIn;
    const float dotDPhi = xIn * xOut + yIn * yOut;
    if (dotDPhi <= 0.f)
      return false;
    // Lagrange identity: crossDPhi^2 + dotDPhi^2 = rtIn^2 * rtOut^2
    const float looseCutDPhi = sdSlopeSin + sdMulsAndPVoff;
    if (alpaka::math::abs(acc, crossDPhi) >= looseCutDPhi * rtIn * rtOut)
      return false;

    dPhi = cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[outerMD] - mds.anchorPhi()[innerMD]);

    return alpaka::math::abs(acc, dPhi) <= sdCut;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passDeltaPhiCutsEndcap(TAcc const& acc,
                                                             MiniDoubletsConst mds,
                                                             unsigned int innerMD,
                                                             unsigned int outerMD,
                                                             const float xIn,
                                                             const float yIn,
                                                             const float xOut,
                                                             const float yOut,
                                                             const float rtIn,
                                                             const float rtOut,
                                                             const float sdSlopeSin,
                                                             float& dPhi,
                                                             const float sdSlope) {
    // Phi pre-check: tan^2(dPhi) > tan^2(sdSlope) implies |dPhi| > sdSlope.
    // Using Lagrange identity: cross^2 + dot^2 = rtIn^2 * rtOut^2, so
    // |cross|/sqrt(cross^2+dot^2) > sdSlopeSin simplifies to |cross| > sdSlopeSin * rtIn * rtOut.
    const float crossDPhi = xIn * yOut - xOut * yIn;
    const float dotDPhi = xIn * xOut + yIn * yOut;
    if (dotDPhi <= 0.f || alpaka::math::abs(acc, crossDPhi) > sdSlopeSin * rtIn * rtOut)
      return false;

    dPhi = cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[outerMD] - mds.anchorPhi()[innerMD]);

    return alpaka::math::abs(acc, dPhi) <= sdSlope;
  }

  // Stored payload of a barrel-barrel segment; shared by the algorithm and FillCompactSegments.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float segmentDPhiChangeBarrel(
      TAcc const& acc, MiniDoubletsConst mds, unsigned int innerMDIndex, float xIn, float yIn, float xOut, float yOut) {
    return cms::alpakatools::reducePhiRange(
        acc, cms::alpakatools::phi(acc, xOut - xIn, yOut - yIn) - mds.anchorPhi()[innerMDIndex]);
  }

  // Same angle measured at the outer anchor, in the form runQuintupletdBetaCutBBBB uses for its outer segment.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float segmentDPhiChangeOut(TAcc const& acc,
                                                            MiniDoubletsConst mds,
                                                            unsigned int innerMDIndex,
                                                            unsigned int outerMDIndex) {
    return cms::alpakatools::reducePhiRange(
        acc,
        cms::alpakatools::phi(acc,
                              mds.anchorX()[outerMDIndex] - mds.anchorX()[innerMDIndex],
                              mds.anchorY()[outerMDIndex] - mds.anchorY()[innerMDIndex]) -
            mds.anchorPhi()[outerMDIndex]);
  }

  // Stored payload of an endcap segment from its dPhi; shared by the algorithm and FillCompactSegments.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void segmentDPhiChangesEndcap(TAcc const& acc,
                                                               ModuleSegData const& outerMod,
                                                               MiniDoubletsConst mds,
                                                               unsigned int innerMDIndex,
                                                               float xOut,
                                                               float yOut,
                                                               float zIn,
                                                               float zOut,
                                                               float dPhi,
                                                               float& dPhiMin,
                                                               float& dPhiMax,
                                                               float& dPhiChange,
                                                               float& dPhiChangeMin,
                                                               float& dPhiChangeMax) {
    if ((outerMod.subdet == Endcap) && (outerMod.moduleType == TwoS)) {
      float dPhiPosHigh = cms::alpakatools::reducePhiRange(
          acc,
          alpaka::math::atan2(acc, yOut + outerMod.edgeDy, xOut + outerMod.edgeDx) - mds.anchorPhi()[innerMDIndex]);
      float dPhiPosLow = cms::alpakatools::reducePhiRange(
          acc,
          alpaka::math::atan2(acc, yOut - outerMod.edgeDy, xOut - outerMod.edgeDx) - mds.anchorPhi()[innerMDIndex]);

      dPhiMax = alpaka::math::abs(acc, dPhiPosHigh) > alpaka::math::abs(acc, dPhiPosLow) ? dPhiPosHigh : dPhiPosLow;
      dPhiMin = alpaka::math::abs(acc, dPhiPosHigh) > alpaka::math::abs(acc, dPhiPosLow) ? dPhiPosLow : dPhiPosHigh;
    } else {
      dPhiMax = dPhi;
      dPhiMin = dPhi;
    }

    float dzFrac = (zOut - zIn) / zIn;
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin / dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax / dzFrac * (1.f + dzFrac);
  }

  // Margin on the chord angle for the float rounding of the atan2 and of the stored anchor phis (about 1e-6 rad each).
  HOST_DEVICE_CONSTANT float kChordAngleMargin = 1e-5f;

  // Angle from r_in to the chord, arctan(t) with t = cross/dot (dot > 0), bounded without the atan2 by the odd series
  // bounds [t - t^3/3 + t^5/5 - t^7/7, t - t^3/3 + t^5/5] (t < 1) or [pi/4, pi/2); posErr adds posErr * rtIn / dot.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool chordAngleBounds(TAcc const& acc,
                                                       float xIn,
                                                       float yIn,
                                                       float rtIn,
                                                       float xOut,
                                                       float yOut,
                                                       float posErr,
                                                       float& angleLo,
                                                       float& angleHi) {
    const float chordX = xOut - xIn;
    const float chordY = yOut - yIn;
    const float dot = xIn * chordX + yIn * chordY;
    if (not(dot > 0.f))
      return false;
    const float invDot = 1.f / dot;
    const float margin = kChordAngleMargin + posErr * rtIn * invDot;
    if (not(margin < 0.1f))  // small-angle regime: asin(posErr / |chord|) within 0.2% of its argument
      return false;
    const float tanAngle = (xIn * chordY - yIn * chordX) * invDot;
    const float absTan = alpaka::math::abs(acc, tanAngle);
    const float tanAngle2 = tanAngle * tanAngle;
    const float seriesHi = absTan * (1.f + tanAngle2 * (-1.f / 3.f + tanAngle2 * 0.2f));
    const float seriesLo = seriesHi - absTan * tanAngle2 * tanAngle2 * tanAngle2 * (1.f / 7.f);
    const float absAngleLo = absTan >= 1.f ? 0.25f * kPi : seriesLo;
    const float absAngleHi = absTan >= 1.f ? 0.5f * kPi : seriesHi;
    angleLo = (tanAngle >= 0.f ? absAngleLo : -absAngleHi) - margin;
    angleHi = (tanAngle >= 0.f ? absAngleHi : -absAngleLo) + margin;
    return true;
  }

  // Cut |center - angle| < halfWidth for every angle in [angleLo, angleHi]: 1 if it passes for all, -1 if it fails
  // for all, 0 otherwise (NaN included). Bitwise operators keep the comparisons free of branches.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int chordCutDecision(float angleLo, float angleHi, float center, float halfWidth) {
    const float windowLo = center - halfWidth;
    const float windowHi = center + halfWidth;
    const bool fails = (angleHi <= windowLo) | (angleLo >= windowHi);
    const bool passes = (angleLo > windowLo) & (angleHi < windowHi);
    return int(passes) - int(fails);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoBarrel(TAcc const& acc,
                                                                  ModuleSegData const& innerMod,
                                                                  ModuleSegData const& outerMod,
                                                                  MiniDoubletsConst mds,
                                                                  MiniDoubletsBuildConst mdsBuild,
                                                                  unsigned int innerMDIndex,
                                                                  unsigned int outerMDIndex,
                                                                  float& dPhi,
                                                                  float& dPhiMin,
                                                                  float& dPhiMax,
                                                                  float& dPhiChange,
                                                                  float& dPhiChangeMin,
                                                                  float& dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                                                  float& dAlphaInnerMDSegment,
                                                                  float& dAlphaOuterMDSegment,
                                                                  float& dAlphaInnerMDOuterMD,
                                                                  float& zLo,
                                                                  float& zHi,
#endif
                                                                  SegOuterMDTerms const& outerTerms,
                                                                  const float ptCut) {
#ifndef CUT_VALUE_DEBUG
    float dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;
    float zLo, zHi;
#endif
    float xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut;

    xIn = mds.anchorX()[innerMDIndex];
    yIn = mds.anchorY()[innerMDIndex];
    zIn = mds.anchorZ()[innerMDIndex];
    rtIn = mds.anchorRt()[innerMDIndex];

    xOut = mds.anchorX()[outerMDIndex];
    yOut = mds.anchorY()[outerMDIndex];
    zOut = mds.anchorZ()[outerMDIndex];
    rtOut = mds.anchorRt()[outerMDIndex];

    const float sdSlopeSin = outerTerms.sdSlopeSin;
    const float sdSlope = outerTerms.sdSlope;
    const float dzDrtScale = outerTerms.dzDrtScale;

    const float zGeom = innerMod.layer <= 2 ? 2.f * kPixelPSZpitch : 2.f * kStrip2SZpitch;

    //slope-correction only on outer end
    zLo = zIn + (zIn - kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) - zGeom;
    zHi = zIn + (zIn + kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    if ((zOut < zLo) || (zOut > zHi))
      return false;

    const float sdPVoff = 0.1f / rtOut;
    const float sdMulsAndPVoff = alpaka::math::sqrt(acc, innerMod.sdMuls * innerMod.sdMuls + sdPVoff * sdPVoff);
    const float sdCut = sdSlope + sdMulsAndPVoff;

    if (!passDeltaPhiCutsBarrel(acc,
                                mds,
                                innerMDIndex,
                                outerMDIndex,
                                xIn,
                                yIn,
                                xOut,
                                yOut,
                                rtIn,
                                rtOut,
                                sdSlopeSin,
                                sdMulsAndPVoff,
                                sdCut,
                                dPhi))
      return false;

    float innerMDAlpha = mdsBuild.dphichanges()[innerMDIndex];
    float outerMDAlpha = mdsBuild.dphichanges()[outerMDIndex];
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    float dAlphaBfield = 0.f;
    float dAlphaResMuls = 0.f;
    float dAlphaThresholdValues[3];
    if (!dAlphaThreshold(acc,
                         dAlphaThresholdValues,
                         innerMod,
                         outerMod,
                         mdsBuild,
                         xIn,
                         yIn,
                         zIn,
                         rtIn,
                         xOut,
                         yOut,
                         zOut,
                         rtOut,
                         innerMDIndex,
                         outerMDIndex,
                         ptCut,
                         dAlphaInnerMDOuterMD,
                         dAlphaBfield,
                         dAlphaResMuls))
      return false;

    float dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    float dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    float dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    // The MD-MD cut needs no dPhiChange: test it before the atan2 (same decision, NaN included).
    if (not(alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold))
      return false;

    // Origin-free line residual: the chord makes equal angles with the tangents at its ends for any radius and d0.
    const float lineResidualSigma =
        (dAlphaInnerMDSegmentThreshold - dAlphaBfield) + (dAlphaOuterMDSegmentThreshold - dAlphaBfield);

    // Serial backends decide the three cuts on dPhiChange below from its atan2-free interval when that suffices (same
    // decisions); on a GPU the undecided lanes of a warp still run the atan2, so the check does not pay there.
    if constexpr (cms::alpakatools::requires_single_thread_per_block_v<TAcc>) {
      float angleLo, angleHi;
      if (chordAngleBounds(acc, xIn, yIn, rtIn, xOut, yOut, 0.f, angleLo, angleHi)) {
        const int lineDecision = chordCutDecision(angleLo,
                                                  angleHi,
                                                  0.5f * (innerMDAlpha + outerMDAlpha + dPhi),
                                                  0.5f * (kLsLineResidCut * lineResidualSigma));
        const int innerDecision = chordCutDecision(angleLo, angleHi, innerMDAlpha, dAlphaInnerMDSegmentThreshold);
        const int outerDecision = chordCutDecision(angleLo, angleHi, outerMDAlpha, dAlphaOuterMDSegmentThreshold);
        if ((lineDecision < 0) | (innerDecision < 0) | (outerDecision < 0))
          return false;
#ifndef CUT_VALUE_DEBUG
        // dPhiChange is left unset: outside CUT_VALUE_DEBUG no caller reads it (FillCompactSegments recomputes it).
        if (lineDecision + innerDecision + outerDecision == 3)
          return true;
#endif
      }
    }

    dPhiChange = segmentDPhiChangeBarrel(acc, mds, innerMDIndex, xIn, yIn, xOut, yOut);
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;

    const float lineResidual = innerMDAlpha + outerMDAlpha + dPhi - 2.f * dPhiChange;
    if (alpaka::math::abs(acc, lineResidual) >= kLsLineResidCut * lineResidualSigma)
      return false;

    if (alpaka::math::abs(acc, dAlphaInnerMDSegment) >= dAlphaInnerMDSegmentThreshold)
      return false;
    if (alpaka::math::abs(acc, dAlphaOuterMDSegment) >= dAlphaOuterMDSegmentThreshold)
      return false;
    return true;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoEndcap(TAcc const& acc,
                                                                  ModuleSegData const& innerMod,
                                                                  ModuleSegData const& outerMod,
                                                                  MiniDoubletsConst mds,
                                                                  MiniDoubletsBuildConst mdsBuild,
                                                                  unsigned int innerMDIndex,
                                                                  unsigned int outerMDIndex,
                                                                  float& dPhi,
                                                                  float& dPhiMin,
                                                                  float& dPhiMax,
                                                                  float& dPhiChange,
                                                                  float& dPhiChangeMin,
                                                                  float& dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                                                  float& dAlphaInnerMDSegment,
                                                                  float& dAlphaOuterMDSegment,
                                                                  float& dAlphaInnerMDOuterMD,
                                                                  float& rtLo,
                                                                  float& rtHi,
#endif
                                                                  SegOuterMDTerms const& outerTerms,
                                                                  const float ptCut) {
#ifndef CUT_VALUE_DEBUG
    float dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;
    float rtLo, rtHi;
#endif
    float xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut;

    xIn = mds.anchorX()[innerMDIndex];
    yIn = mds.anchorY()[innerMDIndex];
    zIn = mds.anchorZ()[innerMDIndex];
    rtIn = mds.anchorRt()[innerMDIndex];

    xOut = mds.anchorX()[outerMDIndex];
    yOut = mds.anchorY()[outerMDIndex];
    zOut = mds.anchorZ()[outerMDIndex];
    rtOut = mds.anchorRt()[outerMDIndex];

    const float sdSlopeSin = outerTerms.sdSlopeSin;
    float rtGeom = ((rtIn < kDisks2SMinRadius && rtOut < kDisks2SMinRadius)
                        ? (2.f * kPixelPSZpitch)
                        : ((rtIn < kDisks2SMinRadius || rtOut < kDisks2SMinRadius) ? (kPixelPSZpitch + kStrip2SZpitch)
                                                                                   : (2.f * kStrip2SZpitch)));

    //cut 0 - z compatibility
    if (zIn * zOut < 0)
      return false;

    float dz = zOut - zIn;
    float dLum = alpaka::math::copysign(acc, kDeltaZLum, zIn);
    const float sdSlope = outerTerms.sdSlope;
    const float drtDzScale = outerTerms.drtDzScale;

    //rt should increase
    rtLo = alpaka::math::max(acc, rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom, rtIn - 0.5f * rtGeom);
    //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction
    rtHi = rtIn * (zOut - dLum) / (zIn - dLum) + rtGeom;

    // Completeness
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    if (!passDeltaPhiCutsEndcap(
            acc, mds, innerMDIndex, outerMDIndex, xIn, yIn, xOut, yOut, rtIn, rtOut, sdSlopeSin, dPhi, sdSlope))
      return false;

    float innerMDAlpha = mdsBuild.dphichanges()[innerMDIndex];
    float outerMDAlpha = mdsBuild.dphichanges()[outerMDIndex];
    dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    float dAlphaBfield = 0.f;
    float dAlphaResMuls = 0.f;
    float dAlphaThresholdValues[3];
    if (!dAlphaThreshold(acc,
                         dAlphaThresholdValues,
                         innerMod,
                         outerMod,
                         mdsBuild,
                         xIn,
                         yIn,
                         zIn,
                         rtIn,
                         xOut,
                         yOut,
                         zOut,
                         rtOut,
                         innerMDIndex,
                         outerMDIndex,
                         ptCut,
                         dAlphaInnerMDOuterMD,
                         dAlphaBfield,
                         dAlphaResMuls))
      return false;

    float dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    float dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    float dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    // Cheapest cuts first (same decision, NaN included): MD-MD, then MD-segment, then the chord residual.
    if (not(alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold))
      return false;

    segmentDPhiChangesEndcap(acc,
                             outerMod,
                             mds,
                             innerMDIndex,
                             xOut,
                             yOut,
                             zIn,
                             zOut,
                             dPhi,
                             dPhiMin,
                             dPhiMax,
                             dPhiChange,
                             dPhiChangeMin,
                             dPhiChangeMax);
    dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    if (alpaka::math::abs(acc, dAlphaInnerMDSegment) >= dAlphaInnerMDSegmentThreshold)
      return false;
    if (alpaka::math::abs(acc, dAlphaOuterMDSegment) >= dAlphaOuterMDSegmentThreshold)
      return false;

    // Endcap dPhiChange is a z-extrapolation, so the chord turn is rebuilt; the resolution is the symmetric one.
    // Serial backends decide it first from the atan2-free interval, as in the barrel; this atan2 uses the stored rt and
    // phi (position margin 1e-5 of rt).
    if constexpr (cms::alpakatools::requires_single_thread_per_block_v<TAcc>) {
      float angleLo, angleHi;
      if (chordAngleBounds(acc, xIn, yIn, rtIn, xOut, yOut, 1e-5f * (rtIn + rtOut), angleLo, angleHi)) {
        const int lineDecision = chordCutDecision(
            angleLo, angleHi, 0.5f * (innerMDAlpha + outerMDAlpha + dPhi), kLsLineResidCut * dAlphaResMuls);
        if (lineDecision != 0)
          return lineDecision > 0;
      }
    }
    const float chord =
        alpaka::math::atan2(acc, rtOut * alpaka::math::sin(acc, dPhi), rtOut * alpaka::math::cos(acc, dPhi) - rtIn);
    const float lineResidual = innerMDAlpha + outerMDAlpha + dPhi - 2.f * chord;
    if (alpaka::math::abs(acc, lineResidual) >= kLsLineResidCut * 2.f * dAlphaResMuls)
      return false;
    return true;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgo(TAcc const& acc,
                                                            ModuleSegData const& innerMod,
                                                            ModuleSegData const& outerMod,
                                                            MiniDoubletsConst mds,
                                                            MiniDoubletsBuildConst mdsBuild,
                                                            unsigned int innerMDIndex,
                                                            unsigned int outerMDIndex,
                                                            float& dPhi,
                                                            float& dPhiMin,
                                                            float& dPhiMax,
                                                            float& dPhiChange,
                                                            float& dPhiChangeMin,
                                                            float& dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                                            float& dAlphaInnerMDSegment,
                                                            float& dAlphaOuterMDSegment,
                                                            float& dAlphaInnerMDOuterMD,
                                                            float& zLo,
                                                            float& zHi,
                                                            float& rtLo,
                                                            float& rtHi,
#endif
                                                            SegOuterMDTerms const& outerTerms,
                                                            const float ptCut) {
    if (innerMod.subdet == Barrel and outerMod.subdet == Barrel) {
#ifdef CUT_VALUE_DEBUG
      rtLo = -999.f;
      rtHi = -999.f;
#endif
      return runSegmentDefaultAlgoBarrel(acc,
                                         innerMod,
                                         outerMod,
                                         mds,
                                         mdsBuild,
                                         innerMDIndex,
                                         outerMDIndex,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                         dAlphaInnerMDSegment,
                                         dAlphaOuterMDSegment,
                                         dAlphaInnerMDOuterMD,
                                         zLo,
                                         zHi,
#endif
                                         outerTerms,
                                         ptCut);
    } else {
#ifdef CUT_VALUE_DEBUG
      zLo = -999.f;
      zHi = -999.f;
#endif
      return runSegmentDefaultAlgoEndcap(acc,
                                         innerMod,
                                         outerMod,
                                         mds,
                                         mdsBuild,
                                         innerMDIndex,
                                         outerMDIndex,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                         dAlphaInnerMDSegment,
                                         dAlphaOuterMDSegment,
                                         dAlphaInnerMDOuterMD,
                                         rtLo,
                                         rtHi,
#endif
                                         outerTerms,
                                         ptCut);
    }
  }

  // Per-inner-MD pass mask written by CountMiniDoubletConnections: bit j = j-th outer MD over the connected modules
  // in module-map order. CreateSegments reads it back and evaluates only the pairs with j >= kSegPassMaskBits,
  // which the count reserves a slot for without evaluating them.
  constexpr unsigned int kSegPassMaskWords = 4;
  constexpr unsigned int kSegPassMaskBits = 64 * kSegPassMaskWords;

  // slotOffsets[s] = first mask bit of the outer MDs of connected module slot s; one inner module per block.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fillSegPassMaskSlotOffsets(TAcc const& acc,
                                                                 ModulesConst modules,
                                                                 MiniDoubletsOccupancyConst mdsOccupancy,
                                                                 uint16_t innerLowerModuleIndex,
                                                                 unsigned int nConnectedModules,
                                                                 unsigned int* slotOffsets) {
    alpaka::syncBlockThreads(acc);  // the previous module's readers are done
    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2] == 0) {
      for (unsigned int slot : cms::alpakatools::uniform_elements_y(acc, nConnectedModules))
        slotOffsets[slot] = mdsOccupancy.nMDs()[modules.moduleMap()[innerLowerModuleIndex][slot]];
    }
    alpaka::syncBlockThreads(acc);
    if (cms::alpakatools::once_per_block(acc)) {
      unsigned int offset = 0;
      for (unsigned int slot = 0; slot < nConnectedModules; ++slot) {
        const unsigned int nOuterMDs = slotOffsets[slot];
        slotOffsets[slot] = offset;
        offset += nOuterMDs;
      }
    }
    alpaka::syncBlockThreads(acc);
  }

  struct CreateSegments {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsBuildConst mdsBuild,
                                  MiniDoubletsOccupancyConst mdsOccupancy,
                                  SegmentCandidates candidates,
                                  SegmentsOccupancy segmentsOccupancy,
                                  ObjectRanges ranges,
                                  const uint64_t* segPassMask,
                                  const SegOuterMDTerms* outerMDTerms,
                                  const float ptCut) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));
      auto& slotOffsets = alpaka::declareSharedVar<unsigned int[max_connected_modules], __COUNTER__>(acc);
      for (uint16_t innerLowerModuleIndex : cms::alpakatools::uniform_groups_z(acc, modules.nLowerModules())) {
        unsigned int nInnerMDs = mdsOccupancy.nMDs()[innerLowerModuleIndex];
        if (nInnerMDs == 0)
          continue;

        unsigned int nConnectedModules = modules.nConnectedModules()[innerLowerModuleIndex];
        fillSegPassMaskSlotOffsets(acc, modules, mdsOccupancy, innerLowerModuleIndex, nConnectedModules, slotOffsets);

        for (uint16_t outerLowerModuleArrayIdx : cms::alpakatools::uniform_elements_y(acc, nConnectedModules)) {
          uint16_t outerLowerModuleIndex = modules.moduleMap()[innerLowerModuleIndex][outerLowerModuleArrayIdx];

          unsigned int nOuterMDs = mdsOccupancy.nMDs()[outerLowerModuleIndex];

          unsigned int limit = nInnerMDs * nOuterMDs;

          if (limit == 0)
            continue;

          const unsigned int slotOffset = slotOffsets[outerLowerModuleArrayIdx];

          for (unsigned int hitIndex : cms::alpakatools::uniform_elements_x(acc, limit)) {
            unsigned int innerMDArrayIdx = hitIndex / nOuterMDs;
            unsigned int innerMDIndex = ranges.mdRanges()[innerLowerModuleIndex][0] + innerMDArrayIdx;
            unsigned int outerMDArrayIdx = hitIndex % nOuterMDs;
            const unsigned int bit = slotOffset + outerMDArrayIdx;
            if (bit < kSegPassMaskBits) {
              if (!((segPassMask[innerMDIndex * kSegPassMaskWords + bit / 64] >> (bit % 64)) & 1))
                continue;
            } else {
              if (mdsBuild.connectedMax()[innerMDIndex] == 0)
                continue;
              ModuleSegData innerMod = loadModuleSegData(modules, innerLowerModuleIndex, ptCut);
              ModuleSegData outerMod = loadModuleSegData(modules, outerLowerModuleIndex, ptCut);
              unsigned int outerMDIndex = ranges.mdRanges()[outerLowerModuleIndex][0] + outerMDArrayIdx;
              float dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax;
#ifdef CUT_VALUE_DEBUG
              float zLo, zHi, rtLo, rtHi, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;
#endif
              dPhiMin = 0;
              dPhiMax = 0;
              dPhiChangeMin = 0;
              dPhiChangeMax = 0;
              if (!runSegmentDefaultAlgo(acc,
                                         innerMod,
                                         outerMod,
                                         mds,
                                         mdsBuild,
                                         innerMDIndex,
                                         outerMDIndex,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                         dAlphaInnerMDSegment,
                                         dAlphaOuterMDSegment,
                                         dAlphaInnerMDOuterMD,
                                         zLo,
                                         zHi,
                                         rtLo,
                                         rtHi,
#endif
                                         outerMDTerms[outerMDIndex],
                                         ptCut))
                continue;
            }

            unsigned int totOccupancySegments = alpaka::atomicAdd(
                acc, &segmentsOccupancy.totOccupancySegments()[innerLowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
            if (static_cast<int>(totOccupancySegments) >= ranges.segmentModuleOccupancy()[innerLowerModuleIndex]) {
              alpaka::atomicAdd(acc, &ranges.nSegmentOverflows(), 1u, alpaka::hierarchy::Blocks{});
#ifdef WARNINGS
              printf("Segment excess alert! Module index = %d, Occupancy = %d\n",
                     innerLowerModuleIndex,
                     totOccupancySegments);
#endif
            } else {
              unsigned int segmentModuleIdx = alpaka::atomicAdd(
                  acc, &segmentsOccupancy.nSegments()[innerLowerModuleIndex], 1u, alpaka::hierarchy::Threads{});
              unsigned int segmentIdx = ranges.segmentModuleIndices()[innerLowerModuleIndex] + segmentModuleIdx;

              // The payload is recomputed by FillCompactSegments for the produced segments only.
              candidates.mdPairIndices()[segmentIdx] = hitIndex;
              candidates.connectedModuleSlots()[segmentIdx] = static_cast<uint8_t>(outerLowerModuleArrayIdx);
            }
          }
        }
      }
    }
  };

  // One MD pair of CountMiniDoubletConnections: counts a passing pair for the inner MD and sets its pass-mask bit.
  // Pairs beyond the mask get a slot without the selection, so the count stays a superset of creation.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void countSegmentPair(TAcc const& acc,
                                                       ModuleSegData const& innerMod,
                                                       ModuleSegData const& outerMod,
                                                       MiniDoubletsConst mds,
                                                       MiniDoubletsBuild mdsBuild,
                                                       unsigned int innerMDIndex,
                                                       unsigned int outerMDIndex,
                                                       unsigned int bit,
                                                       uint64_t* segPassMask,
                                                       SegOuterMDTerms const& outerTerms,
                                                       const float ptCut) {
    if (bit >= kSegPassMaskBits) {
      alpaka::atomicAdd(acc, &mdsBuild.connectedMax()[innerMDIndex], 1u, alpaka::hierarchy::Threads{});
      return;
    }
    float dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax;
#ifdef CUT_VALUE_DEBUG
    float dAlphaInner, dAlphaOuter, dAlphaIO, zLo, zHi, rtLo, rtHi;
#endif
    if (!runSegmentDefaultAlgo(acc,
                               innerMod,
                               outerMod,
                               mds,
                               mdsBuild,
                               innerMDIndex,
                               outerMDIndex,
                               dPhi,
                               dPhiMin,
                               dPhiMax,
                               dPhiChange,
                               dPhiChangeMin,
                               dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                               dAlphaInner,
                               dAlphaOuter,
                               dAlphaIO,
                               zLo,
                               zHi,
                               rtLo,
                               rtHi,
#endif
                               outerTerms,
                               ptCut))
      return;
    alpaka::atomicAdd(acc, &mdsBuild.connectedMax()[innerMDIndex], 1u, alpaka::hierarchy::Threads{});
    alpaka::atomicOr(acc,
                     &segPassMask[innerMDIndex * kSegPassMaskWords + bit / 64],
                     uint64_t(1) << (bit % 64),
                     alpaka::hierarchy::Threads{});
  }

  struct CountMiniDoubletConnections {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsBuild mdsBuild,
                                  MiniDoubletsOccupancyConst mdsOccupancy,
                                  ObjectRangesConst ranges,
                                  uint64_t* segPassMask,
                                  const SegOuterMDTerms* outerMDTerms,
                                  const float ptCut) const {
      // The atomicAdd below with hierarchy::Threads{} requires one block in x, y dimensions.
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));
      const auto& mdRanges = ranges.mdRanges();
      auto& slotOffsets = alpaka::declareSharedVar<unsigned int[max_connected_modules], __COUNTER__>(acc);

      for (uint16_t innerLowerModuleIndex : cms::alpakatools::uniform_groups_z(acc, modules.nLowerModules())) {
        const unsigned int nInnerMDs = mdsOccupancy.nMDs()[innerLowerModuleIndex];
        if (nInnerMDs == 0)
          continue;

        const uint16_t nConnectedModules = modules.nConnectedModules()[innerLowerModuleIndex];
        if (nConnectedModules == 0)
          continue;

        ModuleSegData innerMod = loadModuleSegData(modules, innerLowerModuleIndex, ptCut);
        fillSegPassMaskSlotOffsets(acc, modules, mdsOccupancy, innerLowerModuleIndex, nConnectedModules, slotOffsets);

        for (uint16_t outerLowerModuleArrayIdx : cms::alpakatools::uniform_elements_y(acc, nConnectedModules)) {
          const uint16_t outerLowerModuleIndex = modules.moduleMap()[innerLowerModuleIndex][outerLowerModuleArrayIdx];
          const unsigned int nOuterMDs = mdsOccupancy.nMDs()[outerLowerModuleIndex];
          if (nOuterMDs == 0)
            continue;

          ModuleSegData outerMod = loadModuleSegData(modules, outerLowerModuleIndex, ptCut);
          const unsigned int slotOffset = slotOffsets[outerLowerModuleArrayIdx];

          if constexpr (cms::alpakatools::requires_single_thread_per_block_v<Acc3D>) {
            // Same pair order as below, without a division per pair.
            const unsigned int innerMDBegin = mdRanges[innerLowerModuleIndex][0];
            const unsigned int outerMDBegin = mdRanges[outerLowerModuleIndex][0];
            for (unsigned int innerMDArrayIdx = 0; innerMDArrayIdx < nInnerMDs; ++innerMDArrayIdx) {
              for (unsigned int outerMDArrayIdx = 0; outerMDArrayIdx < nOuterMDs; ++outerMDArrayIdx) {
                const unsigned int outerMDIndex = outerMDBegin + outerMDArrayIdx;
                countSegmentPair(acc,
                                 innerMod,
                                 outerMod,
                                 mds,
                                 mdsBuild,
                                 innerMDBegin + innerMDArrayIdx,
                                 outerMDIndex,
                                 slotOffset + outerMDArrayIdx,
                                 segPassMask,
                                 outerMDTerms[outerMDIndex],
                                 ptCut);
              }
            }
          } else {
            // GPU: the same pair body written out in the loop (through the helper it compiles to a slower kernel).
            const unsigned int limit = nInnerMDs * nOuterMDs;
            for (unsigned int hitIndex : cms::alpakatools::uniform_elements_x(acc, limit)) {
              const unsigned int innerMDArrayIdx = hitIndex / nOuterMDs;
              const unsigned int outerMDArrayIdx = hitIndex % nOuterMDs;

              const unsigned int innerMDIndex = mdRanges[innerLowerModuleIndex][0] + innerMDArrayIdx;
              const unsigned int outerMDIndex = mdRanges[outerLowerModuleIndex][0] + outerMDArrayIdx;

              const unsigned int bit = slotOffset + outerMDArrayIdx;
              if (bit >= kSegPassMaskBits) {
                alpaka::atomicAdd(acc, &mdsBuild.connectedMax()[innerMDIndex], 1u, alpaka::hierarchy::Threads{});
                continue;
              }

              float dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax;
#ifdef CUT_VALUE_DEBUG
              float dAlphaInner, dAlphaOuter, dAlphaIO, zLo, zHi, rtLo, rtHi;
#endif
              if (!runSegmentDefaultAlgo(acc,
                                         innerMod,
                                         outerMod,
                                         mds,
                                         mdsBuild,
                                         innerMDIndex,
                                         outerMDIndex,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax,
#ifdef CUT_VALUE_DEBUG
                                         dAlphaInner,
                                         dAlphaOuter,
                                         dAlphaIO,
                                         zLo,
                                         zHi,
                                         rtLo,
                                         rtHi,
#endif
                                         outerMDTerms[outerMDIndex],
                                         ptCut))
                continue;
              alpaka::atomicAdd(acc, &mdsBuild.connectedMax()[innerMDIndex], 1u, alpaka::hierarchy::Threads{});
              alpaka::atomicOr(acc,
                               &segPassMask[innerMDIndex * kSegPassMaskWords + bit / 64],
                               uint64_t(1) << (bit % 64),
                               alpaka::hierarchy::Threads{});
            }
          }
        }
      }
    }
  };

  // Loose segment capacity of each lower module (the sum of its MDs' segment counters), one module per block; the
  // module offsets follow module order on a serial backend. nTotalSegs and nSegmentOverflows are zeroed before.
  struct CreateSegmentArrayRanges {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  ObjectRanges ranges,
                                  MiniDoubletsBuildConst mdsBuild,
                                  MiniDoubletsOccupancyConst mdsOccupancy) const {
      int& moduleCount = alpaka::declareSharedVar<int, __COUNTER__>(acc);

      for (uint16_t innerLowerModuleIndex : cms::alpakatools::uniform_groups_y(acc, modules.nLowerModules())) {
        if (cms::alpakatools::once_per_block(acc))
          moduleCount = 0;
        alpaka::syncBlockThreads(acc);

        // Sum the connected counts of all MDs in this module.
        const unsigned int nInnerMDs = mdsOccupancy.nMDs()[innerLowerModuleIndex];
        if (modules.nConnectedModules()[innerLowerModuleIndex] != 0 && nInnerMDs != 0) {
          const unsigned int firstMD = ranges.mdRanges()[innerLowerModuleIndex][0];
          for (unsigned int j : cms::alpakatools::uniform_elements_x(acc, nInnerMDs)) {
            alpaka::atomicAdd(acc,
                              &moduleCount,
                              static_cast<int>(mdsBuild.connectedMax()[firstMD + j]),
                              alpaka::hierarchy::Threads{});
          }
        }
        alpaka::syncBlockThreads(acc);

        if (cms::alpakatools::once_per_block(acc)) {
          ranges.segmentModuleOccupancy()[innerLowerModuleIndex] = moduleCount;
          ranges.segmentModuleIndices()[innerLowerModuleIndex] = alpaka::atomicAdd(
              acc, &ranges.nTotalSegs(), static_cast<unsigned int>(moduleCount), alpaka::hierarchy::Blocks{});
        }
        alpaka::syncBlockThreads(acc);  // the next module resets moduleCount
      }
    }
  };

  struct AddSegmentRangesToEventExplicit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  ObjectRanges ranges) const {
      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (segmentsOccupancy.nSegments()[i] == 0) {
          ranges.segmentRanges()[i][0] = -1;
          ranges.segmentRanges()[i][1] = -1;
        } else {
          ranges.segmentRanges()[i][0] = ranges.segmentModuleIndices()[i];
          ranges.segmentRanges()[i][1] = ranges.segmentModuleIndices()[i] + segmentsOccupancy.nSegments()[i] - 1;
        }
      }
    }
  };

  // Compact start of each lower module's segments: exclusive prefix sum of nSegments in module order.
  struct ComputeCompactSegmentOffsets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  ObjectRanges ranges,
                                  int* compactOffsets) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));
      constexpr unsigned int kMaxThreads = 1024;
      const unsigned int nThreads = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
      ALPAKA_ASSERT_ACC(nThreads <= kMaxThreads);
      const unsigned int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
      auto& partial = alpaka::declareSharedVar<int[kMaxThreads], __COUNTER__>(acc);
      auto& warpSums = alpaka::declareSharedVar<int[kMaxThreads / 16], __COUNTER__>(acc);

      // Each thread owns one contiguous chunk of modules, so the offsets follow module order.
      const unsigned int nLowerModules = modules.nLowerModules();
      const unsigned int chunk = cms::alpakatools::divide_up_by(nLowerModules, nThreads);
      const unsigned int begin = cms::alpakatools::idx_min(tid * chunk, nLowerModules);
      const unsigned int end = cms::alpakatools::idx_min(begin + chunk, nLowerModules);

      int sum = 0;
      for (unsigned int m = begin; m < end; ++m)
        sum += segmentsOccupancy.nSegments()[m];
      partial[tid] = sum;
      alpaka::syncBlockThreads(acc);
      cms::alpakatools::blockPrefixScan(acc, partial, static_cast<int32_t>(nThreads), warpSums);  // inclusive
      if (tid == nThreads - 1) {
        compactOffsets[nLowerModules] = partial[tid];
        ranges.nTotalSegs() = partial[tid];
      }
      int offset = partial[tid] - sum;
      for (unsigned int m = begin; m < end; ++m) {
        compactOffsets[m] = offset;
        offset += segmentsOccupancy.nSegments()[m];
      }
    }
  };

  // Write each module's produced segments at their compact slots. CreateSegments kept only the MD pair and the
  // outer module; the payload is recomputed here with the same algorithm (for the produced segments only).
  struct FillCompactSegments {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsBuildConst mdsBuild,
                                  MiniDoubletsOccupancyConst mdsOccupancy,
                                  ObjectRangesConst ranges,
                                  SegmentCandidatesConst candidates,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  const int* compactOffsets,
                                  Segments segments,
                                  const float ptCut) const {
      for (auto innerLowerModuleIndex : cms::alpakatools::independent_groups(acc, modules.nLowerModules())) {
        const unsigned int nSegs = segmentsOccupancy.nSegments()[innerLowerModuleIndex];
        if (nSegs == 0)
          continue;
        const unsigned int src0 = ranges.segmentModuleIndices()[innerLowerModuleIndex];
        const unsigned int dst0 = compactOffsets[innerLowerModuleIndex];
        ModuleSegData innerMod = loadModuleSegData(modules, innerLowerModuleIndex, ptCut);

        for (auto k : cms::alpakatools::independent_group_elements(acc, nSegs)) {
          const unsigned int mdPairIndex = candidates.mdPairIndices()[src0 + k];
          const uint16_t outerLowerModuleIndex =
              modules.moduleMap()[innerLowerModuleIndex][candidates.connectedModuleSlots()[src0 + k]];
          const unsigned int nOuterMDs = mdsOccupancy.nMDs()[outerLowerModuleIndex];
          const unsigned int innerMDIndex = ranges.mdRanges()[innerLowerModuleIndex][0] + mdPairIndex / nOuterMDs;
          const unsigned int outerMDIndex = ranges.mdRanges()[outerLowerModuleIndex][0] + mdPairIndex % nOuterMDs;
          ModuleSegData outerMod = loadModuleSegData(modules, outerLowerModuleIndex, ptCut);

          float dPhi = 0, dPhiMin = 0, dPhiMax = 0, dPhiChange = 0, dPhiChangeMin = 0, dPhiChangeMax = 0;
#ifdef CUT_VALUE_DEBUG
          // Debug builds rerun the full algorithm for the cut-value branches.
          float zLo, zHi, rtLo, rtHi, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;
          runSegmentDefaultAlgo(acc,
                                innerMod,
                                outerMod,
                                mds,
                                mdsBuild,
                                innerMDIndex,
                                outerMDIndex,
                                dPhi,
                                dPhiMin,
                                dPhiMax,
                                dPhiChange,
                                dPhiChangeMin,
                                dPhiChangeMax,
                                dAlphaInnerMDSegment,
                                dAlphaOuterMDSegment,
                                dAlphaInnerMDOuterMD,
                                zLo,
                                zHi,
                                rtLo,
                                rtHi,
                                segOuterMDTerms(acc, mds.anchorRt()[outerMDIndex], ptCut),
                                ptCut);
#else
          // Only the stored payload: the selection already passed in CreateSegments.
          const float xOut = mds.anchorX()[outerMDIndex];
          const float yOut = mds.anchorY()[outerMDIndex];
          if (innerMod.subdet == Barrel and outerMod.subdet == Barrel) {
            dPhiChange = segmentDPhiChangeBarrel(
                acc, mds, innerMDIndex, mds.anchorX()[innerMDIndex], mds.anchorY()[innerMDIndex], xOut, yOut);
          } else {
            dPhi = cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[outerMDIndex] - mds.anchorPhi()[innerMDIndex]);
            segmentDPhiChangesEndcap(acc,
                                     outerMod,
                                     mds,
                                     innerMDIndex,
                                     xOut,
                                     yOut,
                                     mds.anchorZ()[innerMDIndex],
                                     mds.anchorZ()[outerMDIndex],
                                     dPhi,
                                     dPhiMin,
                                     dPhiMax,
                                     dPhiChange,
                                     dPhiChangeMin,
                                     dPhiChangeMax);
          }
#endif

          addSegmentToMemory(segments,
                             innerMDIndex,
                             outerMDIndex,
                             outerLowerModuleIndex,
                             dPhiChange,
                             dPhiChangeMin,
                             dPhiChangeMax,
                             segmentDPhiChangeOut(acc, mds, innerMDIndex, outerMDIndex),
#ifdef CUT_VALUE_DEBUG
                             dPhi,
                             dPhiMin,
                             dPhiMax,
                             zHi,
                             zLo,
                             rtHi,
                             rtLo,
                             dAlphaInnerMDSegment,
                             dAlphaOuterMDSegment,
                             dAlphaInnerMDOuterMD,
#endif
                             dst0 + k);
        }
      }
    }
  };

  // Point the segment module indices at the compact layout and copy the occupancy block (incl. the pixel entry).
  struct SetCompactSegmentModuleIndices {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  SegmentsOccupancyConst looseOccupancy,
                                  const int* compactOffsets,
                                  SegmentsOccupancy segmentsOccupancy,
                                  ObjectRanges ranges) const {
      const unsigned int nLowerModules = modules.nLowerModules();
      for (unsigned int m : cms::alpakatools::uniform_elements(acc, nLowerModules + 1)) {
        segmentsOccupancy.nSegments()[m] = looseOccupancy.nSegments()[m];
        segmentsOccupancy.totOccupancySegments()[m] = looseOccupancy.totOccupancySegments()[m];
        ranges.segmentModuleIndices()[m] = compactOffsets[m];
        if (m < nLowerModules)
          ranges.segmentModuleOccupancy()[m] = looseOccupancy.nSegments()[m];
      }
    }
  };

  // Pixel-MD half of the pLS finalize: runs in the MD stage, the last reader of the Hits collection.
  struct AddPixelMiniDoubletsToEventKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  HitsBaseConst hitsBase,
                                  HitsExtendedConst hitsExtended,
                                  PixelSeedsConst pixelSeeds,
                                  MiniDoublets mds,
                                  MiniDoubletsBuild mdsBuild,
                                  uint16_t pixelModuleIndex,
                                  int size) const {
      for (int tid : cms::alpakatools::uniform_elements(acc, size)) {
        unsigned int innerMDIndex = ranges.miniDoubletModuleIndices()[pixelModuleIndex] + 2 * (tid);
        unsigned int outerMDIndex = ranges.miniDoubletModuleIndices()[pixelModuleIndex] + 2 * (tid) + 1;

        unsigned int firstHit = pixelSeeds.firstHit()[tid];
        unsigned int nHits = pixelSeeds.nHits()[tid];
        unsigned int fourthHit = nHits < 4 ? firstHit + 2 : firstHit + 3;
        addMDToMemory(acc,
                      mds,
                      mdsBuild,
                      hitsBase,
                      hitsExtended,
                      modules,
                      firstHit,
                      firstHit + 1,
                      pixelModuleIndex,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      innerMDIndex);
        addMDToMemory(acc,
                      mds,
                      mdsBuild,
                      hitsBase,
                      hitsExtended,
                      modules,
                      firstHit + 2,
                      fourthHit,
                      pixelModuleIndex,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      outerMDIndex);
      }
    }
  };

  // Segment half of the pLS finalize; the anchor rt comes from the MD copy (mds.anchorRt), not from Hits.
  struct AddPixelSegmentToEventKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ObjectRangesConst ranges,
                                  HitsBaseConst hitsBase,
                                  HitsITConst hitsIT,
                                  PixelSeedsConst pixelSeeds,
                                  MiniDoubletsConst mds,
                                  Segments segments,
                                  PixelSegments pixelSegments,
                                  uint16_t pixelModuleIndex,
                                  int size) const {
      for (int tid : cms::alpakatools::uniform_elements(acc, size)) {
        unsigned int innerMDIndex = ranges.miniDoubletModuleIndices()[pixelModuleIndex] + 2 * (tid);
        unsigned int outerMDIndex = ranges.miniDoubletModuleIndices()[pixelModuleIndex] + 2 * (tid) + 1;
        unsigned int pixelSegmentIndex = ranges.segmentModuleIndices()[pixelModuleIndex] + tid;

        //in outer hits - pt, eta, phi
        float slope = alpaka::math::sinh(acc, hitsBase.ys()[mds.outerHitIndices()[innerMDIndex]]);
        float intercept = hitsBase.zs()[mds.anchorHitIndices()[innerMDIndex]] - slope * mds.anchorRt()[innerMDIndex];
        float score_lsq =
            (mds.anchorRt()[outerMDIndex] * slope + intercept) - (hitsBase.zs()[mds.anchorHitIndices()[outerMDIndex]]);
        score_lsq = score_lsq * score_lsq;

        const Params_pLS::ArrayUxHits hits1{{packedHitIdx(mds.anchorHitIndices()[innerMDIndex], hitsBase, hitsIT),
                                             packedHitIdx(mds.anchorHitIndices()[outerMDIndex], hitsBase, hitsIT),
                                             packedHitIdx(mds.outerHitIndices()[innerMDIndex], hitsBase, hitsIT),
                                             packedHitIdx(mds.outerHitIndices()[outerMDIndex], hitsBase, hitsIT)}};

        addPixelSegmentToMemory(acc,
                                segments,
                                pixelSegments,
                                pixelSeeds,
                                mds,
                                innerMDIndex,
                                outerMDIndex,
                                pixelModuleIndex,
                                hits1,
                                pixelSeeds.deltaPhi()[tid],
                                pixelSegmentIndex,
                                tid,
                                score_lsq);
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
