#ifndef RecoTracker_LSTCore_src_alpaka_Segment_h
#define RecoTracker_LSTCore_src_alpaka_Segment_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"

#include "MiniDoublet.h"
#include "Hit.h"
#include "ObjectRanges.h"

namespace lst {
  struct Segments {
    FPX* dPhis;
    FPX* dPhiMins;
    FPX* dPhiMaxs;
    FPX* dPhiChanges;
    FPX* dPhiChangeMins;
    FPX* dPhiChangeMaxs;
    uint16_t* innerLowerModuleIndices;
    uint16_t* outerLowerModuleIndices;
    unsigned int* seedIdx;
    unsigned int* mdIndices;
    unsigned int* nMemoryLocations;
    uint8_t* logicalLayers;
    unsigned int* innerMiniDoubletAnchorHitIndices;
    unsigned int* outerMiniDoubletAnchorHitIndices;
    unsigned int* innerMiniDoubletOuterHitIndices;
    unsigned int* outerMiniDoubletOuterHitIndices;
    int* charge;
    int* superbin;
    unsigned int* nSegments;             //number of segments per inner lower module
    unsigned int* totOccupancySegments;  //number of segments per inner lower module
    uint4* pLSHitsIdxs;
    int8_t* pixelType;
    char* isQuad;
    char* isDup;
    bool* partOfPT5;
    bool* partOfPT3;
    float* ptIn;
    float* ptErr;
    float* px;
    float* py;
    float* pz;
    float* etaErr;
    float* eta;
    float* phi;
    float* score;
    float* circleCenterX;
    float* circleCenterY;
    float* circleRadius;

    template <typename TBuff>
    void setData(TBuff& buf) {
      dPhis = alpaka::getPtrNative(buf.dPhis_buf);
      dPhiMins = alpaka::getPtrNative(buf.dPhiMins_buf);
      dPhiMaxs = alpaka::getPtrNative(buf.dPhiMaxs_buf);
      dPhiChanges = alpaka::getPtrNative(buf.dPhiChanges_buf);
      dPhiChangeMins = alpaka::getPtrNative(buf.dPhiChangeMins_buf);
      dPhiChangeMaxs = alpaka::getPtrNative(buf.dPhiChangeMaxs_buf);
      innerLowerModuleIndices = alpaka::getPtrNative(buf.innerLowerModuleIndices_buf);
      outerLowerModuleIndices = alpaka::getPtrNative(buf.outerLowerModuleIndices_buf);
      seedIdx = alpaka::getPtrNative(buf.seedIdx_buf);
      mdIndices = alpaka::getPtrNative(buf.mdIndices_buf);
      nMemoryLocations = alpaka::getPtrNative(buf.nMemoryLocations_buf);
      logicalLayers = alpaka::getPtrNative(buf.logicalLayers_buf);
      innerMiniDoubletAnchorHitIndices = alpaka::getPtrNative(buf.innerMiniDoubletAnchorHitIndices_buf);
      outerMiniDoubletAnchorHitIndices = alpaka::getPtrNative(buf.outerMiniDoubletAnchorHitIndices_buf);
      innerMiniDoubletOuterHitIndices = alpaka::getPtrNative(buf.innerMiniDoubletOuterHitIndices_buf);
      outerMiniDoubletOuterHitIndices = alpaka::getPtrNative(buf.outerMiniDoubletOuterHitIndices_buf);
      charge = alpaka::getPtrNative(buf.charge_buf);
      superbin = alpaka::getPtrNative(buf.superbin_buf);
      nSegments = alpaka::getPtrNative(buf.nSegments_buf);
      totOccupancySegments = alpaka::getPtrNative(buf.totOccupancySegments_buf);
      pLSHitsIdxs = alpaka::getPtrNative(buf.pLSHitsIdxs_buf);
      pixelType = alpaka::getPtrNative(buf.pixelType_buf);
      isQuad = alpaka::getPtrNative(buf.isQuad_buf);
      isDup = alpaka::getPtrNative(buf.isDup_buf);
      partOfPT5 = alpaka::getPtrNative(buf.partOfPT5_buf);
      partOfPT3 = alpaka::getPtrNative(buf.partOfPT3_buf);
      ptIn = alpaka::getPtrNative(buf.ptIn_buf);
      ptErr = alpaka::getPtrNative(buf.ptErr_buf);
      px = alpaka::getPtrNative(buf.px_buf);
      py = alpaka::getPtrNative(buf.py_buf);
      pz = alpaka::getPtrNative(buf.pz_buf);
      etaErr = alpaka::getPtrNative(buf.etaErr_buf);
      eta = alpaka::getPtrNative(buf.eta_buf);
      phi = alpaka::getPtrNative(buf.phi_buf);
      score = alpaka::getPtrNative(buf.score_buf);
      circleCenterX = alpaka::getPtrNative(buf.circleCenterX_buf);
      circleCenterY = alpaka::getPtrNative(buf.circleCenterY_buf);
      circleRadius = alpaka::getPtrNative(buf.circleRadius_buf);
    }
  };

  template <typename TDev>
  struct SegmentsBuffer {
    Buf<TDev, FPX> dPhis_buf;
    Buf<TDev, FPX> dPhiMins_buf;
    Buf<TDev, FPX> dPhiMaxs_buf;
    Buf<TDev, FPX> dPhiChanges_buf;
    Buf<TDev, FPX> dPhiChangeMins_buf;
    Buf<TDev, FPX> dPhiChangeMaxs_buf;
    Buf<TDev, uint16_t> innerLowerModuleIndices_buf;
    Buf<TDev, uint16_t> outerLowerModuleIndices_buf;
    Buf<TDev, unsigned int> seedIdx_buf;
    Buf<TDev, unsigned int> mdIndices_buf;
    Buf<TDev, unsigned int> nMemoryLocations_buf;
    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> innerMiniDoubletAnchorHitIndices_buf;
    Buf<TDev, unsigned int> outerMiniDoubletAnchorHitIndices_buf;
    Buf<TDev, unsigned int> innerMiniDoubletOuterHitIndices_buf;
    Buf<TDev, unsigned int> outerMiniDoubletOuterHitIndices_buf;
    Buf<TDev, int> charge_buf;
    Buf<TDev, int> superbin_buf;
    Buf<TDev, unsigned int> nSegments_buf;
    Buf<TDev, unsigned int> totOccupancySegments_buf;
    Buf<TDev, uint4> pLSHitsIdxs_buf;
    Buf<TDev, int8_t> pixelType_buf;
    Buf<TDev, char> isQuad_buf;
    Buf<TDev, char> isDup_buf;
    Buf<TDev, bool> partOfPT5_buf;
    Buf<TDev, bool> partOfPT3_buf;
    Buf<TDev, float> ptIn_buf;
    Buf<TDev, float> ptErr_buf;
    Buf<TDev, float> px_buf;
    Buf<TDev, float> py_buf;
    Buf<TDev, float> pz_buf;
    Buf<TDev, float> etaErr_buf;
    Buf<TDev, float> eta_buf;
    Buf<TDev, float> phi_buf;
    Buf<TDev, float> score_buf;
    Buf<TDev, float> circleCenterX_buf;
    Buf<TDev, float> circleCenterY_buf;
    Buf<TDev, float> circleRadius_buf;

    Segments data_;

    template <typename TQueue, typename TDevAcc>
    SegmentsBuffer(unsigned int nMemoryLocationsIn,
                   uint16_t nLowerModules,
                   unsigned int maxPixelSegments,
                   TDevAcc const& devAccIn,
                   TQueue& queue)
        : dPhis_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiMins_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiMaxs_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiChanges_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiChangeMins_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          dPhiChangeMaxs_buf(allocBufWrapper<FPX>(devAccIn, nMemoryLocationsIn, queue)),
          innerLowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, nMemoryLocationsIn, queue)),
          outerLowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, nMemoryLocationsIn, queue)),
          seedIdx_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelSegments, queue)),
          mdIndices_buf(allocBufWrapper<unsigned int>(devAccIn, nMemoryLocationsIn * 2, queue)),
          nMemoryLocations_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, nMemoryLocationsIn * Params_LS::kLayers, queue)),
          innerMiniDoubletAnchorHitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, nMemoryLocationsIn, queue)),
          outerMiniDoubletAnchorHitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, nMemoryLocationsIn, queue)),
          innerMiniDoubletOuterHitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, nMemoryLocationsIn, queue)),
          outerMiniDoubletOuterHitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, nMemoryLocationsIn, queue)),
          charge_buf(allocBufWrapper<int>(devAccIn, maxPixelSegments, queue)),
          superbin_buf(allocBufWrapper<int>(devAccIn, maxPixelSegments, queue)),
          nSegments_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules + 1, queue)),
          totOccupancySegments_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules + 1, queue)),
          pLSHitsIdxs_buf(allocBufWrapper<uint4>(devAccIn, maxPixelSegments, queue)),
          pixelType_buf(allocBufWrapper<int8_t>(devAccIn, maxPixelSegments, queue)),
          isQuad_buf(allocBufWrapper<char>(devAccIn, maxPixelSegments, queue)),
          isDup_buf(allocBufWrapper<char>(devAccIn, maxPixelSegments, queue)),
          partOfPT5_buf(allocBufWrapper<bool>(devAccIn, maxPixelSegments, queue)),
          partOfPT3_buf(allocBufWrapper<bool>(devAccIn, maxPixelSegments, queue)),
          ptIn_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          ptErr_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          px_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          py_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          pz_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          etaErr_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          eta_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          phi_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          score_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          circleCenterX_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          circleCenterY_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)),
          circleRadius_buf(allocBufWrapper<float>(devAccIn, maxPixelSegments, queue)) {
      alpaka::memset(queue, nSegments_buf, 0u);
      alpaka::memset(queue, totOccupancySegments_buf, 0u);
      alpaka::memset(queue, partOfPT5_buf, false);
      alpaka::memset(queue, partOfPT3_buf, false);
      alpaka::memset(queue, pLSHitsIdxs_buf, 0u);
      alpaka::wait(queue);
    }

    inline Segments const* data() const { return &data_; }
    inline void setData(SegmentsBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(lst::Modules const& modulesInGPU,
                                                                  unsigned int moduleIndex) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modulesInGPU.subdets[moduleIndex];
    short layer = modulesInGPU.layers[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];
    short rod = modulesInGPU.rods[moduleIndex];

    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float isTighterTiltedModules_seg(short subdet, short layer, short side, short rod) {
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    return (subdet == Barrel) && (((side != Center) && (layer == 3)) ||
                                  ((side == NegZ) && (((layer == 2) && (rod > 5)) || ((layer == 1) && (rod > 9)))) ||
                                  ((side == PosZ) && (((layer == 2) && (rod < 8)) || ((layer == 1) && (rod < 4)))));
  };

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
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float moduleGapSize_seg(lst::Modules const& modulesInGPU, unsigned int moduleIndex) {
    unsigned int iL = modulesInGPU.layers[moduleIndex] - 1;
    unsigned int iR = modulesInGPU.rings[moduleIndex] - 1;
    short subdet = modulesInGPU.subdets[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center) {
      moduleSeparation = kMiniDeltaFlat[iL];
    } else if (isTighterTiltedModules_seg(modulesInGPU, moduleIndex)) {
      moduleSeparation = kMiniDeltaTilted[iL];
    } else if (subdet == Endcap) {
      moduleSeparation = kMiniDeltaEndcap[iL][iR];
    } else  //Loose tilted modules
    {
      moduleSeparation = kMiniDeltaLooseTilted[iL];
    }

    return moduleSeparation;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void dAlphaThreshold(TAcc const& acc,
                                                      float* dAlphaThresholdValues,
                                                      lst::Modules const& modulesInGPU,
                                                      lst::MiniDoublets const& mdsInGPU,
                                                      float xIn,
                                                      float yIn,
                                                      float zIn,
                                                      float rtIn,
                                                      float xOut,
                                                      float yOut,
                                                      float zOut,
                                                      float rtOut,
                                                      uint16_t innerLowerModuleIndex,
                                                      uint16_t outerLowerModuleIndex,
                                                      unsigned int innerMDIndex,
                                                      unsigned int outerMDIndex,
                                                      const float ptCut) {
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == lst::Barrel)
                       ? kMiniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut
                       : kMiniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut;

    //more accurate then outer rt - inner rt
    float segmentDr = alpaka::math::sqrt(acc, (yOut - yIn) * (yOut - yIn) + (xOut - xIn) * (xOut - xIn));

    const float dAlpha_Bfield =
        alpaka::math::asin(acc, alpaka::math::min(acc, segmentDr * k2Rinv1GeVf / ptCut, kSinAlphaMax));

    bool isInnerTilted = modulesInGPU.subdets[innerLowerModuleIndex] == lst::Barrel and
                         modulesInGPU.sides[innerLowerModuleIndex] != lst::Center;
    bool isOuterTilted = modulesInGPU.subdets[outerLowerModuleIndex] == lst::Barrel and
                         modulesInGPU.sides[outerLowerModuleIndex] != lst::Center;

    float drdzInner = modulesInGPU.drdzs[innerLowerModuleIndex];
    float drdzOuter = modulesInGPU.drdzs[outerLowerModuleIndex];
    float innerModuleGapSize = lst::moduleGapSize_seg(modulesInGPU, innerLowerModuleIndex);
    float outerModuleGapSize = lst::moduleGapSize_seg(modulesInGPU, outerLowerModuleIndex);
    const float innerminiTilt2 = isInnerTilted
                                     ? ((0.5f * 0.5f) * (kPixelPSZpitch * kPixelPSZpitch) * (drdzInner * drdzInner) /
                                        (1.f + drdzInner * drdzInner) / (innerModuleGapSize * innerModuleGapSize))
                                     : 0;

    const float outerminiTilt2 = isOuterTilted
                                     ? ((0.5f * 0.5f) * (kPixelPSZpitch * kPixelPSZpitch) * (drdzOuter * drdzOuter) /
                                        (1.f + drdzOuter * drdzOuter) / (outerModuleGapSize * outerModuleGapSize))
                                     : 0;

    float miniDelta = innerModuleGapSize;

    float sdLumForInnerMini2;
    float sdLumForOuterMini2;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == lst::Barrel) {
      sdLumForInnerMini2 = innerminiTilt2 * (dAlpha_Bfield * dAlpha_Bfield);
    } else {
      sdLumForInnerMini2 = (mdsInGPU.dphis[innerMDIndex] * mdsInGPU.dphis[innerMDIndex]) * (kDeltaZLum * kDeltaZLum) /
                           (mdsInGPU.dzs[innerMDIndex] * mdsInGPU.dzs[innerMDIndex]);
    }

    if (modulesInGPU.subdets[outerLowerModuleIndex] == lst::Barrel) {
      sdLumForOuterMini2 = outerminiTilt2 * (dAlpha_Bfield * dAlpha_Bfield);
    } else {
      sdLumForOuterMini2 = (mdsInGPU.dphis[outerMDIndex] * mdsInGPU.dphis[outerMDIndex]) * (kDeltaZLum * kDeltaZLum) /
                           (mdsInGPU.dzs[outerMDIndex] * mdsInGPU.dzs[outerMDIndex]);
    }

    // Unique stuff for the segment dudes alone
    float dAlpha_res_inner =
        0.02f / miniDelta *
        (modulesInGPU.subdets[innerLowerModuleIndex] == lst::Barrel ? 1.0f : alpaka::math::abs(acc, zIn) / rtIn);
    float dAlpha_res_outer =
        0.02f / miniDelta *
        (modulesInGPU.subdets[outerLowerModuleIndex] == lst::Barrel ? 1.0f : alpaka::math::abs(acc, zOut) / rtOut);

    float dAlpha_res = dAlpha_res_inner + dAlpha_res_outer;

    if (modulesInGPU.subdets[innerLowerModuleIndex] == lst::Barrel and
        modulesInGPU.sides[innerLowerModuleIndex] == lst::Center) {
      dAlphaThresholdValues[0] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[0] =
          dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForInnerMini2);
    }

    if (modulesInGPU.subdets[outerLowerModuleIndex] == lst::Barrel and
        modulesInGPU.sides[outerLowerModuleIndex] == lst::Center) {
      dAlphaThresholdValues[1] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
    } else {
      dAlphaThresholdValues[1] =
          dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls + sdLumForOuterMini2);
    }

    //Inner to outer
    dAlphaThresholdValues[2] = dAlpha_Bfield + alpaka::math::sqrt(acc, dAlpha_res * dAlpha_res + sdMuls * sdMuls);
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addSegmentToMemory(lst::Segments& segmentsInGPU,
                                                         lst::Modules const& modulesInGPU,
                                                         unsigned int lowerMDIndex,
                                                         unsigned int upperMDIndex,
                                                         uint16_t innerLowerModuleIndex,
                                                         uint16_t outerLowerModuleIndex,
                                                         unsigned int innerMDAnchorHitIndex,
                                                         unsigned int outerMDAnchorHitIndex,
                                                         unsigned int innerMDOuterHitIndex,
                                                         unsigned int outerMDOuterHitIndex,
                                                         float dPhi,
                                                         float dPhiMin,
                                                         float dPhiMax,
                                                         float dPhiChange,
                                                         float dPhiChangeMin,
                                                         float dPhiChangeMax,
                                                         unsigned int idx) {
    segmentsInGPU.mdIndices[idx * 2] = lowerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = upperMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = innerLowerModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = outerLowerModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerMDAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerMDAnchorHitIndex;
    segmentsInGPU.innerMiniDoubletOuterHitIndices[idx] = innerMDOuterHitIndex;
    segmentsInGPU.outerMiniDoubletOuterHitIndices[idx] = outerMDOuterHitIndex;

    segmentsInGPU.logicalLayers[idx* Params_LS::kLayers] =
        modulesInGPU.layers[innerLowerModuleIndex] + (modulesInGPU.subdets[innerLowerModuleIndex] == 4) * 6;
    segmentsInGPU.logicalLayers[idx * Params_LS::kLayers + 1] =
        modulesInGPU.layers[outerLowerModuleIndex] + (modulesInGPU.subdets[outerLowerModuleIndex] == 4) * 6;

    segmentsInGPU.dPhis[idx] = __F2H(dPhi);
    segmentsInGPU.dPhiMins[idx] = __F2H(dPhiMin);
    segmentsInGPU.dPhiMaxs[idx] = __F2H(dPhiMax);
    segmentsInGPU.dPhiChanges[idx] = __F2H(dPhiChange);
    segmentsInGPU.dPhiChangeMins[idx] = __F2H(dPhiChangeMin);
    segmentsInGPU.dPhiChangeMaxs[idx] = __F2H(dPhiChangeMax);
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelSegmentToMemory(TAcc const& acc,
                                                              lst::Segments& segmentsInGPU,
                                                              lst::MiniDoublets const& mdsInGPU,
                                                              unsigned int innerMDIndex,
                                                              unsigned int outerMDIndex,
                                                              uint16_t pixelModuleIndex,
                                                              unsigned int hitIdxs[4],
                                                              unsigned int innerAnchorHitIndex,
                                                              unsigned int outerAnchorHitIndex,
                                                              float dPhiChange,
                                                              unsigned int idx,
                                                              unsigned int pixelSegmentArrayIndex,
                                                              float score) {
    segmentsInGPU.mdIndices[idx * 2] = innerMDIndex;
    segmentsInGPU.mdIndices[idx * 2 + 1] = outerMDIndex;
    segmentsInGPU.innerLowerModuleIndices[idx] = pixelModuleIndex;
    segmentsInGPU.outerLowerModuleIndices[idx] = pixelModuleIndex;
    segmentsInGPU.innerMiniDoubletAnchorHitIndices[idx] = innerAnchorHitIndex;
    segmentsInGPU.outerMiniDoubletAnchorHitIndices[idx] = outerAnchorHitIndex;
    segmentsInGPU.dPhiChanges[idx] = __F2H(dPhiChange);
    segmentsInGPU.isDup[pixelSegmentArrayIndex] = false;
    segmentsInGPU.score[pixelSegmentArrayIndex] = score;

    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].x = hitIdxs[0];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].y = hitIdxs[1];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].z = hitIdxs[2];
    segmentsInGPU.pLSHitsIdxs[pixelSegmentArrayIndex].w = hitIdxs[3];

    //computing circle parameters
    /*
        The two anchor hits are r3PCA and r3LH. p3PCA pt, eta, phi is hitIndex1 x, y, z
        */
    float circleRadius = mdsInGPU.outerX[innerMDIndex] / (2 * k2Rinv1GeVf);
    float circlePhi = mdsInGPU.outerZ[innerMDIndex];
    float candidateCenterXs[] = {mdsInGPU.anchorX[innerMDIndex] + circleRadius * alpaka::math::sin(acc, circlePhi),
                                 mdsInGPU.anchorX[innerMDIndex] - circleRadius * alpaka::math::sin(acc, circlePhi)};
    float candidateCenterYs[] = {mdsInGPU.anchorY[innerMDIndex] - circleRadius * alpaka::math::cos(acc, circlePhi),
                                 mdsInGPU.anchorY[innerMDIndex] + circleRadius * alpaka::math::cos(acc, circlePhi)};

    //check which of the circles can accommodate r3LH better (we won't get perfect agreement)
    float bestChiSquared = lst::lst_INF;
    float chiSquared;
    size_t bestIndex;
    for (size_t i = 0; i < 2; i++) {
      chiSquared =
          alpaka::math::abs(acc,
                            alpaka::math::sqrt(acc,
                                               (mdsInGPU.anchorX[outerMDIndex] - candidateCenterXs[i]) *
                                                       (mdsInGPU.anchorX[outerMDIndex] - candidateCenterXs[i]) +
                                                   (mdsInGPU.anchorY[outerMDIndex] - candidateCenterYs[i]) *
                                                       (mdsInGPU.anchorY[outerMDIndex] - candidateCenterYs[i])) -
                                circleRadius);
      if (chiSquared < bestChiSquared) {
        bestChiSquared = chiSquared;
        bestIndex = i;
      }
    }
    segmentsInGPU.circleCenterX[pixelSegmentArrayIndex] = candidateCenterXs[bestIndex];
    segmentsInGPU.circleCenterY[pixelSegmentArrayIndex] = candidateCenterYs[bestIndex];
    segmentsInGPU.circleRadius[pixelSegmentArrayIndex] = circleRadius;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoBarrel(TAcc const& acc,
                                                                  lst::Modules const& modulesInGPU,
                                                                  lst::MiniDoublets const& mdsInGPU,
                                                                  uint16_t innerLowerModuleIndex,
                                                                  uint16_t outerLowerModuleIndex,
                                                                  unsigned int innerMDIndex,
                                                                  unsigned int outerMDIndex,
                                                                  float& dPhi,
                                                                  float& dPhiMin,
                                                                  float& dPhiMax,
                                                                  float& dPhiChange,
                                                                  float& dPhiChangeMin,
                                                                  float& dPhiChangeMax,
                                                                  const float ptCut) {
    float sdMuls = (modulesInGPU.subdets[innerLowerModuleIndex] == lst::Barrel)
                       ? kMiniMulsPtScaleBarrel[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut
                       : kMiniMulsPtScaleEndcap[modulesInGPU.layers[innerLowerModuleIndex] - 1] * 3.f / ptCut;

    float xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut;

    xIn = mdsInGPU.anchorX[innerMDIndex];
    yIn = mdsInGPU.anchorY[innerMDIndex];
    zIn = mdsInGPU.anchorZ[innerMDIndex];
    rtIn = mdsInGPU.anchorRt[innerMDIndex];

    xOut = mdsInGPU.anchorX[outerMDIndex];
    yOut = mdsInGPU.anchorY[outerMDIndex];
    zOut = mdsInGPU.anchorZ[outerMDIndex];
    rtOut = mdsInGPU.anchorRt[outerMDIndex];

    float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    float sdPVoff = 0.1f / rtOut;
    float dzDrtScale = alpaka::math::tan(acc, sdSlope) / sdSlope;  //FIXME: need appropriate value

    const float zGeom = modulesInGPU.layers[innerLowerModuleIndex] <= 2 ? 2.f * kPixelPSZpitch : 2.f * kStrip2SZpitch;

    float zLo = zIn + (zIn - kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) -
                zGeom;  //slope-correction only on outer end
    float zHi = zIn + (zIn + kDeltaZLum) * (rtOut / rtIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + zGeom;

    if ((zOut < zLo) || (zOut > zHi))
      return false;

    float sdCut = sdSlope + alpaka::math::sqrt(acc, sdMuls * sdMuls + sdPVoff * sdPVoff);

    dPhi = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

    if (alpaka::math::abs(acc, dPhi) > sdCut)
      return false;

    dPhiChange = lst::phi_mpi_pi(acc, lst::phi(acc, xOut - xIn, yOut - yIn) - mdsInGPU.anchorPhi[innerMDIndex]);

    if (alpaka::math::abs(acc, dPhiChange) > sdCut)
      return false;

    float dAlphaThresholdValues[3];
    dAlphaThreshold(acc,
                    dAlphaThresholdValues,
                    modulesInGPU,
                    mdsInGPU,
                    xIn,
                    yIn,
                    zIn,
                    rtIn,
                    xOut,
                    yOut,
                    zOut,
                    rtOut,
                    innerLowerModuleIndex,
                    outerLowerModuleIndex,
                    innerMDIndex,
                    outerMDIndex,
                    ptCut);

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    float dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    float dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    float dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    float dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    float dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    float dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    if (alpaka::math::abs(acc, dAlphaInnerMDSegment) >= dAlphaInnerMDSegmentThreshold)
      return false;
    if (alpaka::math::abs(acc, dAlphaOuterMDSegment) >= dAlphaOuterMDSegmentThreshold)
      return false;
    return alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgoEndcap(TAcc const& acc,
                                                                  lst::Modules const& modulesInGPU,
                                                                  lst::MiniDoublets const& mdsInGPU,
                                                                  uint16_t innerLowerModuleIndex,
                                                                  uint16_t outerLowerModuleIndex,
                                                                  unsigned int innerMDIndex,
                                                                  unsigned int outerMDIndex,
                                                                  float& dPhi,
                                                                  float& dPhiMin,
                                                                  float& dPhiMax,
                                                                  float& dPhiChange,
                                                                  float& dPhiChangeMin,
                                                                  float& dPhiChangeMax,
                                                                  const float ptCut) {
    float xIn, yIn, zIn, rtIn, xOut, yOut, zOut, rtOut;

    xIn = mdsInGPU.anchorX[innerMDIndex];
    yIn = mdsInGPU.anchorY[innerMDIndex];
    zIn = mdsInGPU.anchorZ[innerMDIndex];
    rtIn = mdsInGPU.anchorRt[innerMDIndex];

    xOut = mdsInGPU.anchorX[outerMDIndex];
    yOut = mdsInGPU.anchorY[outerMDIndex];
    zOut = mdsInGPU.anchorZ[outerMDIndex];
    rtOut = mdsInGPU.anchorRt[outerMDIndex];

    bool outerLayerEndcapTwoS = (modulesInGPU.subdets[outerLowerModuleIndex] == lst::Endcap) &&
                                (modulesInGPU.moduleType[outerLowerModuleIndex] == lst::TwoS);

    float sdSlope = alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * k2Rinv1GeVf / ptCut, kSinAlphaMax));
    float disks2SMinRadius = 60.f;

    float rtGeom = ((rtIn < disks2SMinRadius && rtOut < disks2SMinRadius)
                        ? (2.f * kPixelPSZpitch)
                        : ((rtIn < disks2SMinRadius || rtOut < disks2SMinRadius) ? (kPixelPSZpitch + kStrip2SZpitch)
                                                                                 : (2.f * kStrip2SZpitch)));

    //cut 0 - z compatibility
    if (zIn * zOut < 0)
      return false;

    float dz = zOut - zIn;
    // Alpaka: Needs to be moved over
    float dLum = alpaka::math::copysign(acc, kDeltaZLum, zIn);
    float drtDzScale = sdSlope / alpaka::math::tan(acc, sdSlope);

    float rtLo = alpaka::math::max(
        acc, rtIn * (1.f + dz / (zIn + dLum) * drtDzScale) - rtGeom, rtIn - 0.5f * rtGeom);  //rt should increase
    float rtHi = rtIn * (zOut - dLum) / (zIn - dLum) +
                 rtGeom;  //dLum for luminous; rGeom for measurement size; no tanTheta_loc(pt) correction

    // Completeness
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    dPhi = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

    float sdCut = sdSlope;
    if (outerLayerEndcapTwoS) {
      float dPhiPos_high =
          lst::phi_mpi_pi(acc, mdsInGPU.anchorHighEdgePhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);
      float dPhiPos_low =
          lst::phi_mpi_pi(acc, mdsInGPU.anchorLowEdgePhi[outerMDIndex] - mdsInGPU.anchorPhi[innerMDIndex]);

      dPhiMax = alpaka::math::abs(acc, dPhiPos_high) > alpaka::math::abs(acc, dPhiPos_low) ? dPhiPos_high : dPhiPos_low;
      dPhiMin = alpaka::math::abs(acc, dPhiPos_high) > alpaka::math::abs(acc, dPhiPos_low) ? dPhiPos_low : dPhiPos_high;
    } else {
      dPhiMax = dPhi;
      dPhiMin = dPhi;
    }
    if (alpaka::math::abs(acc, dPhi) > sdCut)
      return false;

    float dzFrac = dz / zIn;
    dPhiChange = dPhi / dzFrac * (1.f + dzFrac);
    dPhiChangeMin = dPhiMin / dzFrac * (1.f + dzFrac);
    dPhiChangeMax = dPhiMax / dzFrac * (1.f + dzFrac);

    if (alpaka::math::abs(acc, dPhiChange) > sdCut)
      return false;

    float dAlphaThresholdValues[3];
    dAlphaThreshold(acc,
                    dAlphaThresholdValues,
                    modulesInGPU,
                    mdsInGPU,
                    xIn,
                    yIn,
                    zIn,
                    rtIn,
                    xOut,
                    yOut,
                    zOut,
                    rtOut,
                    innerLowerModuleIndex,
                    outerLowerModuleIndex,
                    innerMDIndex,
                    outerMDIndex,
                    ptCut);

    float innerMDAlpha = mdsInGPU.dphichanges[innerMDIndex];
    float outerMDAlpha = mdsInGPU.dphichanges[outerMDIndex];
    float dAlphaInnerMDSegment = innerMDAlpha - dPhiChange;
    float dAlphaOuterMDSegment = outerMDAlpha - dPhiChange;
    float dAlphaInnerMDOuterMD = innerMDAlpha - outerMDAlpha;

    float dAlphaInnerMDSegmentThreshold = dAlphaThresholdValues[0];
    float dAlphaOuterMDSegmentThreshold = dAlphaThresholdValues[1];
    float dAlphaInnerMDOuterMDThreshold = dAlphaThresholdValues[2];

    if (alpaka::math::abs(acc, dAlphaInnerMDSegment) >= dAlphaInnerMDSegmentThreshold)
      return false;
    if (alpaka::math::abs(acc, dAlphaOuterMDSegment) >= dAlphaOuterMDSegmentThreshold)
      return false;
    return alpaka::math::abs(acc, dAlphaInnerMDOuterMD) < dAlphaInnerMDOuterMDThreshold;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runSegmentDefaultAlgo(TAcc const& acc,
                                                            lst::Modules const& modulesInGPU,
                                                            lst::MiniDoublets const& mdsInGPU,
                                                            uint16_t innerLowerModuleIndex,
                                                            uint16_t outerLowerModuleIndex,
                                                            unsigned int innerMDIndex,
                                                            unsigned int outerMDIndex,
                                                            float& dPhi,
                                                            float& dPhiMin,
                                                            float& dPhiMax,
                                                            float& dPhiChange,
                                                            float& dPhiChangeMin,
                                                            float& dPhiChangeMax,
                                                            const float ptCut) {
    if (modulesInGPU.subdets[innerLowerModuleIndex] == lst::Barrel and
        modulesInGPU.subdets[outerLowerModuleIndex] == lst::Barrel) {
      return runSegmentDefaultAlgoBarrel(acc,
                                         modulesInGPU,
                                         mdsInGPU,
                                         innerLowerModuleIndex,
                                         outerLowerModuleIndex,
                                         innerMDIndex,
                                         outerMDIndex,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax,
                                         ptCut);
    } else {
      return runSegmentDefaultAlgoEndcap(acc,
                                         modulesInGPU,
                                         mdsInGPU,
                                         innerLowerModuleIndex,
                                         outerLowerModuleIndex,
                                         innerMDIndex,
                                         outerMDIndex,
                                         dPhi,
                                         dPhiMin,
                                         dPhiMax,
                                         dPhiChange,
                                         dPhiChangeMin,
                                         dPhiChangeMax,
                                         ptCut);
    }
  };

  struct createSegmentsInGPUv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::MiniDoublets mdsInGPU,
                                  lst::Segments segmentsInGPU,
                                  lst::ObjectRanges rangesInGPU,
                                  const float ptCut) const {
      auto const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
      auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
      auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

      for (uint16_t innerLowerModuleIndex = globalBlockIdx[2]; innerLowerModuleIndex < (*modulesInGPU.nLowerModules);
           innerLowerModuleIndex += gridBlockExtent[2]) {
        unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex];
        if (nInnerMDs == 0)
          continue;

        unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];

        for (uint16_t outerLowerModuleArrayIdx = blockThreadIdx[1]; outerLowerModuleArrayIdx < nConnectedModules;
             outerLowerModuleArrayIdx += blockThreadExtent[1]) {
          uint16_t outerLowerModuleIndex =
              modulesInGPU.moduleMap[innerLowerModuleIndex * max_connected_modules + outerLowerModuleArrayIdx];

          unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex];

          unsigned int limit = nInnerMDs * nOuterMDs;

          if (limit == 0)
            continue;
          for (unsigned int hitIndex = blockThreadIdx[2]; hitIndex < limit; hitIndex += blockThreadExtent[2]) {
            unsigned int innerMDArrayIdx = hitIndex / nOuterMDs;
            unsigned int outerMDArrayIdx = hitIndex % nOuterMDs;
            if (outerMDArrayIdx >= nOuterMDs)
              continue;

            unsigned int innerMDIndex = rangesInGPU.mdRanges[innerLowerModuleIndex * 2] + innerMDArrayIdx;
            unsigned int outerMDIndex = rangesInGPU.mdRanges[outerLowerModuleIndex * 2] + outerMDArrayIdx;

            float dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax;

            unsigned int innerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[innerMDIndex];
            unsigned int outerMiniDoubletAnchorHitIndex = mdsInGPU.anchorHitIndices[outerMDIndex];
            unsigned int innerMiniDoubletOuterHitIndex = mdsInGPU.outerHitIndices[innerMDIndex];
            unsigned int outerMiniDoubletOuterHitIndex = mdsInGPU.outerHitIndices[outerMDIndex];
            dPhiMin = 0;
            dPhiMax = 0;
            dPhiChangeMin = 0;
            dPhiChangeMax = 0;
            if (runSegmentDefaultAlgo(acc,
                                      modulesInGPU,
                                      mdsInGPU,
                                      innerLowerModuleIndex,
                                      outerLowerModuleIndex,
                                      innerMDIndex,
                                      outerMDIndex,
                                      dPhi,
                                      dPhiMin,
                                      dPhiMax,
                                      dPhiChange,
                                      dPhiChangeMin,
                                      dPhiChangeMax,
                                      ptCut)) {
              unsigned int totOccupancySegments = alpaka::atomicOp<alpaka::AtomicAdd>(
                  acc, &segmentsInGPU.totOccupancySegments[innerLowerModuleIndex], 1u);
              if (static_cast<int>(totOccupancySegments) >= rangesInGPU.segmentModuleOccupancy[innerLowerModuleIndex]) {
#ifdef WARNINGS
                printf("Segment excess alert! Module index = %d, Occupancy = %d\n",
                       innerLowerModuleIndex,
                       totOccupancySegments);
#endif
              } else {
                unsigned int segmentModuleIdx =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, &segmentsInGPU.nSegments[innerLowerModuleIndex], 1u);
                unsigned int segmentIdx = rangesInGPU.segmentModuleIndices[innerLowerModuleIndex] + segmentModuleIdx;

                addSegmentToMemory(segmentsInGPU,
                                   modulesInGPU,
                                   innerMDIndex,
                                   outerMDIndex,
                                   innerLowerModuleIndex,
                                   outerLowerModuleIndex,
                                   innerMiniDoubletAnchorHitIndex,
                                   outerMiniDoubletAnchorHitIndex,
                                   innerMiniDoubletOuterHitIndex,
                                   outerMiniDoubletOuterHitIndex,
                                   dPhi,
                                   dPhiMin,
                                   dPhiMax,
                                   dPhiChange,
                                   dPhiChangeMin,
                                   dPhiChangeMax,
                                   segmentIdx);
              }
            }
          }
        }
      }
    }
  };

  struct createSegmentArrayRanges {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::ObjectRanges rangesInGPU,
                                  lst::MiniDoublets mdsInGPU,
                                  const float ptCut) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      // Initialize variables in shared memory and set to 0
      int& nTotalSegments = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalSegments = 0;
      }
      alpaka::syncBlockThreads(acc);

      // Occupancy matrix for 0.8 GeV pT Cut
      constexpr int p08_occupancy_matrix[4][4] = {
          {572, 300, 183, 62},  // category 0
          {191, 128, 0, 0},     // category 1
          {0, 107, 102, 0},     // category 2
          {0, 64, 79, 85}       // category 3
      };

      // Occupancy matrix for 0.6 GeV pT Cut, 99.9%
      constexpr int p06_occupancy_matrix[4][4] = {
          {936, 351, 256, 61},  // category 0
          {1358, 763, 0, 0},    // category 1
          {0, 210, 268, 0},     // category 2
          {0, 60, 97, 96}       // category 3
      };

      // Select the appropriate occupancy matrix based on ptCut
      const auto& occupancy_matrix = (ptCut < 0.8f) ? p06_occupancy_matrix : p08_occupancy_matrix;

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
        if (modulesInGPU.nConnectedModules[i] == 0) {
          rangesInGPU.segmentModuleIndices[i] = nTotalSegments;
          rangesInGPU.segmentModuleOccupancy[i] = 0;
          continue;
        }

        short module_rings = modulesInGPU.rings[i];
        short module_layers = modulesInGPU.layers[i];
        short module_subdets = modulesInGPU.subdets[i];
        float module_eta = alpaka::math::abs(acc, modulesInGPU.eta[i]);

        int category_number = lst::getCategoryNumber(module_layers, module_subdets, module_rings);
        int eta_number = lst::getEtaBin(module_eta);

        int occupancy = 0;
        if (category_number != -1 && eta_number != -1) {
          occupancy = occupancy_matrix[category_number][eta_number];
        }
#ifdef WARNINGS
        else {
          printf("Unhandled case in createSegmentArrayRanges! Module index = %i\n", i);
        }
#endif

        int nTotSegs = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nTotalSegments, occupancy);
        rangesInGPU.segmentModuleIndices[i] = nTotSegs;
        rangesInGPU.segmentModuleOccupancy[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (globalThreadIdx[2] == 0) {
        rangesInGPU.segmentModuleIndices[*modulesInGPU.nLowerModules] = nTotalSegments;
        *rangesInGPU.device_nTotalSegs = nTotalSegments;
      }
    }
  };

  struct addSegmentRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::Segments segmentsInGPU,
                                  lst::ObjectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
        if (segmentsInGPU.nSegments[i] == 0) {
          rangesInGPU.segmentRanges[i * 2] = -1;
          rangesInGPU.segmentRanges[i * 2 + 1] = -1;
        } else {
          rangesInGPU.segmentRanges[i * 2] = rangesInGPU.segmentModuleIndices[i];
          rangesInGPU.segmentRanges[i * 2 + 1] = rangesInGPU.segmentModuleIndices[i] + segmentsInGPU.nSegments[i] - 1;
        }
      }
    }
  };

  struct addPixelSegmentToEventKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::ObjectRanges rangesInGPU,
                                  lst::Hits hitsInGPU,
                                  lst::MiniDoublets mdsInGPU,
                                  lst::Segments segmentsInGPU,
                                  unsigned int* hitIndices0,
                                  unsigned int* hitIndices1,
                                  unsigned int* hitIndices2,
                                  unsigned int* hitIndices3,
                                  float* dPhiChange,
                                  uint16_t pixelModuleIndex,
                                  int size) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int tid = globalThreadIdx[2]; tid < size; tid += gridThreadExtent[2]) {
        unsigned int innerMDIndex = rangesInGPU.miniDoubletModuleIndices[pixelModuleIndex] + 2 * (tid);
        unsigned int outerMDIndex = rangesInGPU.miniDoubletModuleIndices[pixelModuleIndex] + 2 * (tid) + 1;
        unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + tid;

        addMDToMemory(acc,
                      mdsInGPU,
                      hitsInGPU,
                      modulesInGPU,
                      hitIndices0[tid],
                      hitIndices1[tid],
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
                      mdsInGPU,
                      hitsInGPU,
                      modulesInGPU,
                      hitIndices2[tid],
                      hitIndices3[tid],
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

        //in outer hits - pt, eta, phi
        float slope = alpaka::math::sinh(acc, hitsInGPU.ys[mdsInGPU.outerHitIndices[innerMDIndex]]);
        float intercept = hitsInGPU.zs[mdsInGPU.anchorHitIndices[innerMDIndex]] -
                          slope * hitsInGPU.rts[mdsInGPU.anchorHitIndices[innerMDIndex]];
        float score_lsq = (hitsInGPU.rts[mdsInGPU.anchorHitIndices[outerMDIndex]] * slope + intercept) -
                          (hitsInGPU.zs[mdsInGPU.anchorHitIndices[outerMDIndex]]);
        score_lsq = score_lsq * score_lsq;

        unsigned int hits1[Params_pLS::kHits];
        hits1[0] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[innerMDIndex]];
        hits1[1] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[outerMDIndex]];
        hits1[2] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[innerMDIndex]];
        hits1[3] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[outerMDIndex]];
        addPixelSegmentToMemory(acc,
                                segmentsInGPU,
                                mdsInGPU,
                                innerMDIndex,
                                outerMDIndex,
                                pixelModuleIndex,
                                hits1,
                                hitIndices0[tid],
                                hitIndices2[tid],
                                dPhiChange[tid],
                                pixelSegmentIndex,
                                tid,
                                score_lsq);
      }
    }
  };
}  // namespace lst

#endif
