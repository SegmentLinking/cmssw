#ifndef RecoTracker_LSTCore_src_alpaka_PixelQuintuplet_h
#define RecoTracker_LSTCore_src_alpaka_PixelQuintuplet_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "Triplet.h"
#include "Quintuplet.h"
#include "PixelTriplet.h"

namespace lst {
  struct PixelQuintuplets {
    unsigned int* pixelIndices;
    unsigned int* T5Indices;
    unsigned int* nPixelQuintuplets;
    unsigned int* totOccupancyPixelQuintuplets;
    bool* isDup;
    FPX* score;
    FPX* eta;
    FPX* phi;
    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    uint16_t* lowerModuleIndices;
    FPX* pixelRadius;
    FPX* quintupletRadius;
    FPX* centerX;
    FPX* centerY;
    float* rzChiSquared;
    float* rPhiChiSquared;
    float* rPhiChiSquaredInwards;

    template <typename TBuff>
    void setData(TBuff& buf) {
      pixelIndices = alpaka::getPtrNative(buf.pixelIndices_buf);
      T5Indices = alpaka::getPtrNative(buf.T5Indices_buf);
      nPixelQuintuplets = alpaka::getPtrNative(buf.nPixelQuintuplets_buf);
      totOccupancyPixelQuintuplets = alpaka::getPtrNative(buf.totOccupancyPixelQuintuplets_buf);
      isDup = alpaka::getPtrNative(buf.isDup_buf);
      score = alpaka::getPtrNative(buf.score_buf);
      eta = alpaka::getPtrNative(buf.eta_buf);
      phi = alpaka::getPtrNative(buf.phi_buf);
      logicalLayers = alpaka::getPtrNative(buf.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(buf.hitIndices_buf);
      lowerModuleIndices = alpaka::getPtrNative(buf.lowerModuleIndices_buf);
      pixelRadius = alpaka::getPtrNative(buf.pixelRadius_buf);
      quintupletRadius = alpaka::getPtrNative(buf.quintupletRadius_buf);
      centerX = alpaka::getPtrNative(buf.centerX_buf);
      centerY = alpaka::getPtrNative(buf.centerY_buf);
      rzChiSquared = alpaka::getPtrNative(buf.rzChiSquared_buf);
      rPhiChiSquared = alpaka::getPtrNative(buf.rPhiChiSquared_buf);
      rPhiChiSquaredInwards = alpaka::getPtrNative(buf.rPhiChiSquaredInwards_buf);
    }
  };

  template <typename TDev>
  struct PixelQuintupletsBuffer {
    Buf<TDev, unsigned int> pixelIndices_buf;
    Buf<TDev, unsigned int> T5Indices_buf;
    Buf<TDev, unsigned int> nPixelQuintuplets_buf;
    Buf<TDev, unsigned int> totOccupancyPixelQuintuplets_buf;
    Buf<TDev, bool> isDup_buf;
    Buf<TDev, FPX> score_buf;
    Buf<TDev, FPX> eta_buf;
    Buf<TDev, FPX> phi_buf;
    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, FPX> pixelRadius_buf;
    Buf<TDev, FPX> quintupletRadius_buf;
    Buf<TDev, FPX> centerX_buf;
    Buf<TDev, FPX> centerY_buf;
    Buf<TDev, float> rzChiSquared_buf;
    Buf<TDev, float> rPhiChiSquared_buf;
    Buf<TDev, float> rPhiChiSquaredInwards_buf;

    PixelQuintuplets data_;

    template <typename TQueue, typename TDevAcc>
    PixelQuintupletsBuffer(unsigned int maxPixelQuintuplets, TDevAcc const& devAccIn, TQueue& queue)
        : pixelIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets, queue)),
          T5Indices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets, queue)),
          nPixelQuintuplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          totOccupancyPixelQuintuplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          isDup_buf(allocBufWrapper<bool>(devAccIn, maxPixelQuintuplets, queue)),
          score_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          eta_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          phi_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxPixelQuintuplets * Params_pT5::kLayers, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuintuplets * Params_pT5::kHits, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, maxPixelQuintuplets * Params_pT5::kLayers, queue)),
          pixelRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          quintupletRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          centerX_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          centerY_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuintuplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)),
          rPhiChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)),
          rPhiChiSquaredInwards_buf(allocBufWrapper<float>(devAccIn, maxPixelQuintuplets, queue)) {
      alpaka::memset(queue, nPixelQuintuplets_buf, 0u);
      alpaka::memset(queue, totOccupancyPixelQuintuplets_buf, 0u);
      alpaka::wait(queue);
    }

    inline PixelQuintuplets const* data() const { return &data_; }
    inline void setData(PixelQuintupletsBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelQuintupletToMemory(lst::Modules const& modulesInGPU,
                                                                 lst::MiniDoublets const& mdsInGPU,
                                                                 lst::Segments const& segmentsInGPU,
                                                                 lst::Quintuplets const& quintupletsInGPU,
                                                                 lst::PixelQuintuplets& pixelQuintupletsInGPU,
                                                                 unsigned int pixelIndex,
                                                                 unsigned int T5Index,
                                                                 unsigned int pixelQuintupletIndex,
                                                                 float rzChiSquared,
                                                                 float rPhiChiSquared,
                                                                 float rPhiChiSquaredInwards,
                                                                 float score,
                                                                 float eta,
                                                                 float phi,
                                                                 float pixelRadius,
                                                                 float quintupletRadius,
                                                                 float centerX,
                                                                 float centerY) {
    pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex] = pixelIndex;
    pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex] = T5Index;
    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = false;
    pixelQuintupletsInGPU.score[pixelQuintupletIndex] = __F2H(score);
    pixelQuintupletsInGPU.eta[pixelQuintupletIndex] = __F2H(eta);
    pixelQuintupletsInGPU.phi[pixelQuintupletIndex] = __F2H(phi);

    pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex] = __F2H(pixelRadius);
    pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex] = __F2H(quintupletRadius);
    pixelQuintupletsInGPU.centerX[pixelQuintupletIndex] = __F2H(centerX);
    pixelQuintupletsInGPU.centerY[pixelQuintupletIndex] = __F2H(centerY);

    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex] = 0;
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 1] = 0;
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 2] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers];
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 3] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers + 1];
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 4] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers + 2];
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 5] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers + 3];
    pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex + 6] =
        quintupletsInGPU.logicalLayers[T5Index * Params_T5::kLayers + 4];

    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex] =
        segmentsInGPU.innerLowerModuleIndices[pixelIndex];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 1] =
        segmentsInGPU.outerLowerModuleIndices[pixelIndex];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 2] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 3] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers + 1];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 4] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers + 2];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 5] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers + 3];
    pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex + 6] =
        quintupletsInGPU.lowerModuleIndices[T5Index * Params_T5::kLayers + 4];

    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[Params_pLS::kLayers * pixelIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[Params_pLS::kLayers * pixelIndex + 1];

    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex] =
        mdsInGPU.anchorHitIndices[pixelInnerMD];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 1] =
        mdsInGPU.outerHitIndices[pixelInnerMD];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 2] =
        mdsInGPU.anchorHitIndices[pixelOuterMD];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 3] =
        mdsInGPU.outerHitIndices[pixelOuterMD];

    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 4] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 5] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 1];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 6] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 2];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 7] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 3];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 8] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 4];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 9] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 5];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 10] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 6];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 11] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 7];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 12] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 8];
    pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex + 13] =
        quintupletsInGPU.hitIndices[Params_T5::kHits * T5Index + 9];

    pixelQuintupletsInGPU.rzChiSquared[pixelQuintupletIndex] = rzChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquared[pixelQuintupletIndex] = rPhiChiSquared;
    pixelQuintupletsInGPU.rPhiChiSquaredInwards[pixelQuintupletIndex] = rPhiChiSquaredInwards;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RZChiSquaredCuts(lst::Modules const& modulesInGPU,
                                                              uint16_t lowerModuleIndex1,
                                                              uint16_t lowerModuleIndex2,
                                                              uint16_t lowerModuleIndex3,
                                                              uint16_t lowerModuleIndex4,
                                                              uint16_t lowerModuleIndex5,
                                                              float rzChiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == lst::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == lst::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == lst::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex4] == lst::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex5] == lst::TwoS);

    // This slides shows the cut threshold definition. The comments below in the code, e.g, "cat 10", is consistent with the region separation in the slides
    // https://indico.cern.ch/event/1410985/contributions/5931017/attachments/2875400/5035406/helix%20approxi%20for%20pT5%20rzchi2%20new%20results%20versions.pdf
    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12 and layer5 == 13) {  // cat 10
        return rzChiSquared < 14.031f;
      } else if (layer4 == 4 and layer5 == 12) {  // cat 12
        return rzChiSquared < 8.760f;
      } else if (layer4 == 4 and layer5 == 5) {  // cat 11
        return rzChiSquared < 3.607f;
      } else if (layer4 == 7 and layer5 == 13) {  // cat 9
        return rzChiSquared < 16.620;
      } else if (layer4 == 7 and layer5 == 8) {  // cat 8
        return rzChiSquared < 17.910f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {  // cat 7
        return rzChiSquared < 8.950f;
      } else if (layer4 == 8 and layer5 == 14) {  // cat 6
        return rzChiSquared < 14.837f;
      } else if (layer4 == 8 and layer5 == 9) {  // cat 5
        return rzChiSquared < 18.519f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {  // cat 3
        return rzChiSquared < 15.093f;
      } else if (layer4 == 9 and layer5 == 15) {  // cat 4
        return rzChiSquared < 11.200f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12 and layer5 == 13) {  // cat 20
        return rzChiSquared < 12.868f;
      } else if (layer4 == 5 and layer5 == 12) {  // cat 19
        return rzChiSquared < 6.128f;
      } else if (layer4 == 5 and layer5 == 6) {  // cat 18
        return rzChiSquared < 2.987f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 7) {
      if (layer4 == 13 and layer5 == 14) {  // cat 17
        return rzChiSquared < 19.446f;
      } else if (layer4 == 8 and layer5 == 14) {  // cat 16
        return rzChiSquared < 17.520f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14 and layer5 == 15) {  // cat 15
        return rzChiSquared < 14.71f;
      } else if (layer4 == 9 and layer5 == 15) {  // cat 14
        return rzChiSquared < 18.213f;
      }
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {  // cat 0
        return rzChiSquared < 10.016f;
      } else if (layer4 == 10 and layer5 == 16) {  // cat 1
        return rzChiSquared < 87.671f;
      } else if (layer4 == 15 and layer5 == 16) {  // cat 2
        return rzChiSquared < 5.844f;
      }
    }
    return true;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RPhiChiSquaredCuts(lst::Modules const& modulesInGPU,
                                                                uint16_t lowerModuleIndex1,
                                                                uint16_t lowerModuleIndex2,
                                                                uint16_t lowerModuleIndex3,
                                                                uint16_t lowerModuleIndex4,
                                                                uint16_t lowerModuleIndex5,
                                                                float rPhiChiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == lst::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == lst::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == lst::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex4] == lst::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex5] == lst::TwoS);

    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12 and layer5 == 13) {
        return rPhiChiSquared < 48.921f;
      } else if (layer4 == 4 and layer5 == 12) {
        return rPhiChiSquared < 97.948f;
      } else if (layer4 == 4 and layer5 == 5) {
        return rPhiChiSquared < 129.3f;
      } else if (layer4 == 7 and layer5 == 13) {
        return rPhiChiSquared < 56.21f;
      } else if (layer4 == 7 and layer5 == 8) {
        return rPhiChiSquared < 74.198f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rPhiChiSquared < 21.265f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rPhiChiSquared < 37.058f;
      } else if (layer4 == 8 and layer5 == 9) {
        return rPhiChiSquared < 42.578f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return rPhiChiSquared < 32.253f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rPhiChiSquared < 37.058f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12 and layer5 == 13) {
        return rPhiChiSquared < 97.947f;
      } else if (layer4 == 5 and layer5 == 12) {
        return rPhiChiSquared < 129.3f;
      } else if (layer4 == 5 and layer5 == 6) {
        return rPhiChiSquared < 170.68f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rPhiChiSquared < 48.92f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rPhiChiSquared < 74.2f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14 and layer5 == 15) {
        return rPhiChiSquared < 42.58f;
      } else if (layer4 == 9 and layer5 == 10) {
        return rPhiChiSquared < 37.06f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rPhiChiSquared < 48.92f;
      }
    } else if (layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15) {
      return rPhiChiSquared < 85.25f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {
        return rPhiChiSquared < 42.58f;
      } else if (layer4 == 10 and layer5 == 16) {
        return rPhiChiSquared < 37.06f;
      } else if (layer4 == 15 and layer5 == 16) {
        return rPhiChiSquared < 37.06f;
      }
    }
    return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RPhiChiSquared(TAcc const& acc,
                                                                lst::Modules const& modulesInGPU,
                                                                uint16_t* lowerModuleIndices,
                                                                float g,
                                                                float f,
                                                                float radius,
                                                                float* xs,
                                                                float* ys) {
    /*
        Compute circle parameters from 3 pixel hits, and then use them to compute the chi squared for the outer hits
        */

    float delta1[5], delta2[5], slopes[5];
    bool isFlat[5];
    float chiSquared = 0;

    computeSigmasForRegression(acc, modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    chiSquared = computeChiSquared(acc, 5, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

    return chiSquared;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RPhiChiSquaredInwards(
      float g, float f, float r, float* xPix, float* yPix) {
    /*
        Using the computed regression center and radius, compute the chi squared for the pixels
        */

    float chiSquared = 0;
    for (size_t i = 0; i < 2; i++) {
      float residual = (xPix[i] - g) * (xPix[i] - g) + (yPix[i] - f) * (yPix[i] - f) - r * r;
      chiSquared += residual * residual;
    }
    chiSquared *= 0.5f;
    return chiSquared;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT5RPhiChiSquaredInwardsCuts(lst::Modules const& modulesInGPU,
                                                                       uint16_t lowerModuleIndex1,
                                                                       uint16_t lowerModuleIndex2,
                                                                       uint16_t lowerModuleIndex3,
                                                                       uint16_t lowerModuleIndex4,
                                                                       uint16_t lowerModuleIndex5,
                                                                       float rPhiChiSquared) {
    const int layer1 = modulesInGPU.layers[lowerModuleIndex1] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex1] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex1] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex1] == lst::TwoS);
    const int layer2 = modulesInGPU.layers[lowerModuleIndex2] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex2] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex2] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex2] == lst::TwoS);
    const int layer3 = modulesInGPU.layers[lowerModuleIndex3] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex3] == lst::TwoS);
    const int layer4 = modulesInGPU.layers[lowerModuleIndex4] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex4] == lst::TwoS);
    const int layer5 = modulesInGPU.layers[lowerModuleIndex5] +
                       6 * (modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap) +
                       5 * (modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap and
                            modulesInGPU.moduleType[lowerModuleIndex5] == lst::TwoS);

    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12 and layer5 == 13) {
        return rPhiChiSquared < 451.141f;
      } else if (layer4 == 4 and layer5 == 12) {
        return rPhiChiSquared < 786.173f;
      } else if (layer4 == 4 and layer5 == 5) {
        return rPhiChiSquared < 595.545f;
      } else if (layer4 == 7 and layer5 == 13) {
        return rPhiChiSquared < 581.339f;
      } else if (layer4 == 7 and layer5 == 8) {
        return rPhiChiSquared < 112.537f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rPhiChiSquared < 225.322f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rPhiChiSquared < 1192.402f;
      } else if (layer4 == 8 and layer5 == 9) {
        return rPhiChiSquared < 786.173f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 10) {
        return rPhiChiSquared < 1037.817f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rPhiChiSquared < 1808.536f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12 and layer5 == 13) {
        return rPhiChiSquared < 684.253f;
      } else if (layer4 == 5 and layer5 == 12) {
        return rPhiChiSquared < 684.253f;
      } else if (layer4 == 5 and layer5 == 6) {
        return rPhiChiSquared < 684.253f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 13 and layer5 == 14) {
        return rPhiChiSquared < 451.141f;
      } else if (layer4 == 8 and layer5 == 14) {
        return rPhiChiSquared < 518.34f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14 and layer5 == 15) {
        return rPhiChiSquared < 2077.92f;
      } else if (layer4 == 9 and layer5 == 10) {
        return rPhiChiSquared < 74.20f;
      } else if (layer4 == 9 and layer5 == 15) {
        return rPhiChiSquared < 1808.536f;
      }
    } else if (layer1 == 3 and layer2 == 7 and layer3 == 8 and layer4 == 14 and layer5 == 15) {
      return rPhiChiSquared < 786.173f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10 and layer5 == 11) {
        return rPhiChiSquared < 1574.076f;
      } else if (layer4 == 10 and layer5 == 16) {
        return rPhiChiSquared < 5492.11f;
      } else if (layer4 == 15 and layer5 == 16) {
        return rPhiChiSquared < 2743.037f;
      }
    }
    return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelQuintupletDefaultAlgo(TAcc const& acc,
                                                                    lst::Modules const& modulesInGPU,
                                                                    lst::ObjectRanges const& rangesInGPU,
                                                                    lst::MiniDoublets const& mdsInGPU,
                                                                    lst::Segments const& segmentsInGPU,
                                                                    lst::Triplets const& tripletsInGPU,
                                                                    lst::Quintuplets const& quintupletsInGPU,
                                                                    unsigned int pixelSegmentIndex,
                                                                    unsigned int quintupletIndex,
                                                                    float& rzChiSquared,
                                                                    float& rPhiChiSquared,
                                                                    float& rPhiChiSquaredInwards,
                                                                    float& pixelRadius,
                                                                    float& quintupletRadius,
                                                                    float& centerX,
                                                                    float& centerY,
                                                                    unsigned int pixelSegmentArrayIndex,
                                                                    const float ptCut) {
    unsigned int T5InnerT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
    unsigned int T5OuterT3Index = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];

    float pixelRadiusTemp, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp, centerXTemp,
        centerYTemp;

    if (not runPixelTripletDefaultAlgo(acc,
                                       modulesInGPU,
                                       rangesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       tripletsInGPU,
                                       pixelSegmentIndex,
                                       T5InnerT3Index,
                                       pixelRadiusTemp,
                                       tripletRadius,
                                       centerXTemp,
                                       centerYTemp,
                                       rzChiSquaredTemp,
                                       rPhiChiSquaredTemp,
                                       rPhiChiSquaredInwardsTemp,
                                       ptCut,
                                       false))
      return false;

    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * T5InnerT3Index + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * T5OuterT3Index + 1];

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
    unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];

    uint16_t lowerModuleIndex1 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex];
    uint16_t lowerModuleIndex2 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 1];
    uint16_t lowerModuleIndex3 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 2];
    uint16_t lowerModuleIndex4 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 3];
    uint16_t lowerModuleIndex5 = quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex + 4];

    uint16_t lowerModuleIndices[Params_T5::kLayers] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    float rtPix[Params_pLS::kLayers] = {mdsInGPU.anchorRt[pixelInnerMDIndex], mdsInGPU.anchorRt[pixelOuterMDIndex]};
    float xPix[Params_pLS::kLayers] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
    float yPix[Params_pLS::kLayers] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
    float zPix[Params_pLS::kLayers] = {mdsInGPU.anchorZ[pixelInnerMDIndex], mdsInGPU.anchorZ[pixelOuterMDIndex]};

    float zs[Params_T5::kLayers] = {mdsInGPU.anchorZ[firstMDIndex],
                                    mdsInGPU.anchorZ[secondMDIndex],
                                    mdsInGPU.anchorZ[thirdMDIndex],
                                    mdsInGPU.anchorZ[fourthMDIndex],
                                    mdsInGPU.anchorZ[fifthMDIndex]};
    float rts[Params_T5::kLayers] = {mdsInGPU.anchorRt[firstMDIndex],
                                     mdsInGPU.anchorRt[secondMDIndex],
                                     mdsInGPU.anchorRt[thirdMDIndex],
                                     mdsInGPU.anchorRt[fourthMDIndex],
                                     mdsInGPU.anchorRt[fifthMDIndex]};

    float pixelSegmentPt = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float pixelSegmentPx = segmentsInGPU.px[pixelSegmentArrayIndex];
    float pixelSegmentPy = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pixelSegmentPz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    int pixelSegmentCharge = segmentsInGPU.charge[pixelSegmentArrayIndex];

    rzChiSquared = 0;

    //get the appropriate centers
    pixelRadius = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];

    if (pixelRadius < 5.0f * kR1GeVf) {  //only apply r-z chi2 cuts for <5GeV tracks
      rzChiSquared = computePT5RZChiSquared(acc,
                                            modulesInGPU,
                                            lowerModuleIndices,
                                            rtPix,
                                            xPix,
                                            yPix,
                                            zPix,
                                            rts,
                                            zs,
                                            pixelSegmentPt,
                                            pixelSegmentPx,
                                            pixelSegmentPy,
                                            pixelSegmentPz,
                                            pixelSegmentCharge);
      if (not passPT5RZChiSquaredCuts(modulesInGPU,
                                      lowerModuleIndex1,
                                      lowerModuleIndex2,
                                      lowerModuleIndex3,
                                      lowerModuleIndex4,
                                      lowerModuleIndex5,
                                      rzChiSquared))
        return false;
    }

    //outer T5
    float xs[Params_T5::kLayers] = {mdsInGPU.anchorX[firstMDIndex],
                                    mdsInGPU.anchorX[secondMDIndex],
                                    mdsInGPU.anchorX[thirdMDIndex],
                                    mdsInGPU.anchorX[fourthMDIndex],
                                    mdsInGPU.anchorX[fifthMDIndex]};
    float ys[Params_T5::kLayers] = {mdsInGPU.anchorY[firstMDIndex],
                                    mdsInGPU.anchorY[secondMDIndex],
                                    mdsInGPU.anchorY[thirdMDIndex],
                                    mdsInGPU.anchorY[fourthMDIndex],
                                    mdsInGPU.anchorY[fifthMDIndex]};

    //get the appropriate centers
    centerX = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    centerY = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];

    float T5CenterX = quintupletsInGPU.regressionG[quintupletIndex];
    float T5CenterY = quintupletsInGPU.regressionF[quintupletIndex];
    quintupletRadius = quintupletsInGPU.regressionRadius[quintupletIndex];

    rPhiChiSquared =
        computePT5RPhiChiSquared(acc, modulesInGPU, lowerModuleIndices, centerX, centerY, pixelRadius, xs, ys);

    if (pixelRadius < 5.0f * kR1GeVf) {
      if (not passPT5RPhiChiSquaredCuts(modulesInGPU,
                                        lowerModuleIndex1,
                                        lowerModuleIndex2,
                                        lowerModuleIndex3,
                                        lowerModuleIndex4,
                                        lowerModuleIndex5,
                                        rPhiChiSquared))
        return false;
    }

    rPhiChiSquaredInwards = computePT5RPhiChiSquaredInwards(T5CenterX, T5CenterY, quintupletRadius, xPix, yPix);

    if (quintupletsInGPU.regressionRadius[quintupletIndex] < 5.0f * kR1GeVf) {
      if (not passPT5RPhiChiSquaredInwardsCuts(modulesInGPU,
                                               lowerModuleIndex1,
                                               lowerModuleIndex2,
                                               lowerModuleIndex3,
                                               lowerModuleIndex4,
                                               lowerModuleIndex5,
                                               rPhiChiSquaredInwards))
        return false;
    }
    //trusting the T5 regression center to also be a good estimate..
    centerX = (centerX + T5CenterX) / 2;
    centerY = (centerY + T5CenterY) / 2;

    return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT5RZChiSquared(TAcc const& acc,
                                                              struct lst::Modules const& modulesInGPU,
                                                              const uint16_t* lowerModuleIndices,
                                                              const float* rtPix,
                                                              const float* xPix,
                                                              const float* yPix,
                                                              const float* zPix,
                                                              const float* rts,
                                                              const float* zs,
                                                              float pixelSegmentPt,
                                                              float pixelSegmentPx,
                                                              float pixelSegmentPy,
                                                              float pixelSegmentPz,
                                                              int pixelSegmentCharge) {
    float residual = 0;
    float error2 = 0;
    float RMSE = 0;

    // the pixel positions are in unit of cm, and need to be divided by 100 to be in consistent with unit mm.
    float Px = pixelSegmentPx, Py = pixelSegmentPy, Pz = pixelSegmentPz;
    int charge = pixelSegmentCharge;
    float x1 = xPix[1] / 100;
    float y1 = yPix[1] / 100;
    float z1 = zPix[1] / 100;
    float r1 = rtPix[1] / 100;

    float a = -100 / lst::kR1GeVf * charge;

    for (size_t i = 0; i < Params_T5::kLayers; i++) {
      float zsi = zs[i] / 100;
      float rtsi = rts[i] / 100;
      uint16_t lowerModuleIndex = lowerModuleIndices[i];
      const int moduleType = modulesInGPU.moduleType[lowerModuleIndex];
      const int moduleSide = modulesInGPU.sides[lowerModuleIndex];
      const int moduleSubdet = modulesInGPU.subdets[lowerModuleIndex];

      // calculation is detailed documented here https://indico.cern.ch/event/1185895/contributions/4982756/attachments/2526561/4345805/helix%20pT3%20summarize.pdf
      float diffr, diffz;
      float p = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);

      float rou = a / p;
      if (moduleSubdet == lst::Endcap) {
        float s = (zsi - z1) * p / Pz;
        float x = x1 + Px / a * alpaka::math::sin(acc, rou * s) - Py / a * (1 - alpaka::math::cos(acc, rou * s));
        float y = y1 + Py / a * alpaka::math::sin(acc, rou * s) + Px / a * (1 - alpaka::math::cos(acc, rou * s));
        diffr = alpaka::math::abs(acc, rtsi - alpaka::math::sqrt(acc, x * x + y * y)) * 100;
      }

      if (moduleSubdet == lst::Barrel) {
        float paraA = r1 * r1 + 2 * (Px * Px + Py * Py) / (a * a) + 2 * (y1 * Px - x1 * Py) / a - rtsi * rtsi;
        float paraB = 2 * (x1 * Px + y1 * Py) / a;
        float paraC = 2 * (y1 * Px - x1 * Py) / a + 2 * (Px * Px + Py * Py) / (a * a);
        float A = paraB * paraB + paraC * paraC;
        float B = 2 * paraA * paraB;
        float C = paraA * paraA - paraC * paraC;
        float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float solz1 = alpaka::math::asin(acc, sol1) / rou * Pz / p + z1;
        float solz2 = alpaka::math::asin(acc, sol2) / rou * Pz / p + z1;
        float diffz1 = alpaka::math::abs(acc, solz1 - zsi) * 100;
        float diffz2 = alpaka::math::abs(acc, solz2 - zsi) * 100;
        diffz = alpaka::math::min(acc, diffz1, diffz2);
      }

      residual = moduleSubdet == lst::Barrel ? diffz : diffr;

      //PS Modules
      if (moduleType == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

      //special dispensation to tilted PS modules!
      if (moduleType == 0 and moduleSubdet == lst::Barrel and moduleSide != Center) {
        float drdz = modulesInGPU.drdzs[lowerModuleIndex];
        error2 /= (1.f + drdz * drdz);
      }
      RMSE += (residual * residual) / error2;
    }

    RMSE = alpaka::math::sqrt(acc, 0.2f * RMSE);  // Divided by the degree of freedom 5.
    return RMSE;
  };

  struct createPixelQuintupletsInGPUFromMapv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::MiniDoublets mdsInGPU,
                                  lst::Segments segmentsInGPU,
                                  lst::Triplets tripletsInGPU,
                                  lst::Quintuplets quintupletsInGPU,
                                  lst::PixelQuintuplets pixelQuintupletsInGPU,
                                  unsigned int* connectedPixelSize,
                                  unsigned int* connectedPixelIndex,
                                  unsigned int nPixelSegments,
                                  lst::ObjectRanges rangesInGPU,
                                  const float ptCut) const {
      auto const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int i_pLS = globalThreadIdx[1]; i_pLS < nPixelSegments; i_pLS += gridThreadExtent[1]) {
        auto iLSModule_max = connectedPixelIndex[i_pLS] + connectedPixelSize[i_pLS];
        for (unsigned int iLSModule = connectedPixelIndex[i_pLS] + globalBlockIdx[0]; iLSModule < iLSModule_max;
             iLSModule += gridBlockExtent[0]) {
          //these are actual module indices
          uint16_t quintupletLowerModuleIndex = modulesInGPU.connectedPixels[iLSModule];
          if (quintupletLowerModuleIndex >= *modulesInGPU.nLowerModules)
            continue;
          if (modulesInGPU.moduleType[quintupletLowerModuleIndex] == lst::TwoS)
            continue;
          uint16_t pixelModuleIndex = *modulesInGPU.nLowerModules;
          if (segmentsInGPU.isDup[i_pLS])
            continue;
          unsigned int nOuterQuintuplets = quintupletsInGPU.nQuintuplets[quintupletLowerModuleIndex];

          if (nOuterQuintuplets == 0)
            continue;

          unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + i_pLS;

          //fetch the quintuplet
          for (unsigned int outerQuintupletArrayIndex = globalThreadIdx[2];
               outerQuintupletArrayIndex < nOuterQuintuplets;
               outerQuintupletArrayIndex += gridThreadExtent[2]) {
            unsigned int quintupletIndex =
                rangesInGPU.quintupletModuleIndices[quintupletLowerModuleIndex] + outerQuintupletArrayIndex;

            if (quintupletsInGPU.isDup[quintupletIndex])
              continue;

            float rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, quintupletRadius, centerX, centerY;

            bool success = runPixelQuintupletDefaultAlgo(acc,
                                                         modulesInGPU,
                                                         rangesInGPU,
                                                         mdsInGPU,
                                                         segmentsInGPU,
                                                         tripletsInGPU,
                                                         quintupletsInGPU,
                                                         pixelSegmentIndex,
                                                         quintupletIndex,
                                                         rzChiSquared,
                                                         rPhiChiSquared,
                                                         rPhiChiSquaredInwards,
                                                         pixelRadius,
                                                         quintupletRadius,
                                                         centerX,
                                                         centerY,
                                                         static_cast<unsigned int>(i_pLS),
                                                         ptCut);
            if (success) {
              unsigned int totOccupancyPixelQuintuplets =
                  alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelQuintupletsInGPU.totOccupancyPixelQuintuplets, 1u);
              if (totOccupancyPixelQuintuplets >= n_max_pixel_quintuplets) {
#ifdef WARNINGS
                printf("Pixel Quintuplet excess alert!\n");
#endif
              } else {
                unsigned int pixelQuintupletIndex =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelQuintupletsInGPU.nPixelQuintuplets, 1u);
                float eta = __H2F(quintupletsInGPU.eta[quintupletIndex]);
                float phi = __H2F(quintupletsInGPU.phi[quintupletIndex]);

                addPixelQuintupletToMemory(modulesInGPU,
                                           mdsInGPU,
                                           segmentsInGPU,
                                           quintupletsInGPU,
                                           pixelQuintupletsInGPU,
                                           pixelSegmentIndex,
                                           quintupletIndex,
                                           pixelQuintupletIndex,
                                           rzChiSquared,
                                           rPhiChiSquared,
                                           rPhiChiSquaredInwards,
                                           rPhiChiSquared,
                                           eta,
                                           phi,
                                           pixelRadius,
                                           quintupletRadius,
                                           centerX,
                                           centerY);

                tripletsInGPU.partOfPT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex]] = true;
                tripletsInGPU.partOfPT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1]] = true;
                segmentsInGPU.partOfPT5[i_pLS] = true;
                quintupletsInGPU.partOfPT5[quintupletIndex] = true;
                segmentsInGPU.partOfPT5[i_pLS] = true;
              }  // tot occupancy
            }  // end success
          }  // end T5
        }  // end iLS
      }  // end i_pLS
    }
  };
}  // namespace lst
#endif
