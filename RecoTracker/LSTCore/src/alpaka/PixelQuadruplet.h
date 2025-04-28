#ifndef RecoTracker_LSTCore_src_alpaka_PixelQuadruplet_h
#define RecoTracker_LSTCore_src_alpaka_PixelQuadruplet_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "Triplet.h"
#include "Quadruplet.h"
#include "PixelTriplet.h"

namespace lst {
  struct PixelQuadruplets {
    unsigned int* pixelIndices;
    unsigned int* T4Indices;
    unsigned int* nPixelQuadruplets;
    unsigned int* totOccupancyPixelQuadruplets;
    bool* isDup;
    FPX* score;
    FPX* eta;
    FPX* phi;
    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    uint16_t* lowerModuleIndices;
    FPX* pixelRadius;
    FPX* pixelRadiusError;
    FPX* quadrupletRadius;
    FPX* centerX;
    FPX* centerY;
    float* rzChiSquared;
    float* rPhiChiSquared;
    float* rPhiChiSquaredInwards;
    float* pt;

    template <typename TBuff>
    void setData(TBuff& buf) {
      pixelIndices = alpaka::getPtrNative(buf.pixelIndices_buf);
      T4Indices = alpaka::getPtrNative(buf.T4Indices_buf);
      nPixelQuadruplets = alpaka::getPtrNative(buf.nPixelQuadruplets_buf);
      totOccupancyPixelQuadruplets = alpaka::getPtrNative(buf.totOccupancyPixelQuadruplets_buf);
      isDup = alpaka::getPtrNative(buf.isDup_buf);
      score = alpaka::getPtrNative(buf.score_buf);
      eta = alpaka::getPtrNative(buf.eta_buf);
      phi = alpaka::getPtrNative(buf.phi_buf);
      logicalLayers = alpaka::getPtrNative(buf.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(buf.hitIndices_buf);
      lowerModuleIndices = alpaka::getPtrNative(buf.lowerModuleIndices_buf);
      pixelRadius = alpaka::getPtrNative(buf.pixelRadius_buf);
      pixelRadiusError = alpaka::getPtrNative(buf.pixelRadiusError_buf);
      quadrupletRadius = alpaka::getPtrNative(buf.quadrupletRadius_buf);
      centerX = alpaka::getPtrNative(buf.centerX_buf);
      centerY = alpaka::getPtrNative(buf.centerY_buf);
      rzChiSquared = alpaka::getPtrNative(buf.rzChiSquared_buf);
      rPhiChiSquared = alpaka::getPtrNative(buf.rPhiChiSquared_buf);
      rPhiChiSquaredInwards = alpaka::getPtrNative(buf.rPhiChiSquaredInwards_buf);
      pt = alpaka::getPtrNative(buf.pt_buf);
    }
  };

  template <typename TDev>
  struct PixelQuadrupletsBuffer {
    Buf<TDev, unsigned int> pixelIndices_buf;
    Buf<TDev, unsigned int> T4Indices_buf;
    Buf<TDev, unsigned int> nPixelQuadruplets_buf;
    Buf<TDev, unsigned int> totOccupancyPixelQuadruplets_buf;
    Buf<TDev, bool> isDup_buf;
    Buf<TDev, FPX> score_buf;
    Buf<TDev, FPX> eta_buf;
    Buf<TDev, FPX> phi_buf;
    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, FPX> pixelRadius_buf;
    Buf<TDev, FPX> pixelRadiusError_buf;
    Buf<TDev, FPX> quadrupletRadius_buf;
    Buf<TDev, FPX> centerX_buf;
    Buf<TDev, FPX> centerY_buf;
    Buf<TDev, float> rzChiSquared_buf;
    Buf<TDev, float> rPhiChiSquared_buf;
    Buf<TDev, float> rPhiChiSquaredInwards_buf;
    Buf<TDev, float> pt_buf;

    PixelQuadruplets data_;

    template <typename TQueue, typename TDevAcc>
    PixelQuadrupletsBuffer(unsigned int maxPixelQuadruplets, TDevAcc const& devAccIn, TQueue& queue)
        : pixelIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuadruplets, queue)),
          T4Indices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuadruplets, queue)),
          nPixelQuadruplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          totOccupancyPixelQuadruplets_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          isDup_buf(allocBufWrapper<bool>(devAccIn, maxPixelQuadruplets, queue)),
          score_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuadruplets, queue)),
          eta_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuadruplets, queue)),
          phi_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuadruplets, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxPixelQuadruplets * Params_pT4::kLayers, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPixelQuadruplets * Params_pT4::kHits, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, maxPixelQuadruplets * Params_pT4::kLayers, queue)),
          pixelRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuadruplets, queue)),
          pixelRadiusError_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuadruplets, queue)),
          quadrupletRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuadruplets, queue)),
          centerX_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuadruplets, queue)),
          centerY_buf(allocBufWrapper<FPX>(devAccIn, maxPixelQuadruplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelQuadruplets, queue)),
          rPhiChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPixelQuadruplets, queue)),
          rPhiChiSquaredInwards_buf(allocBufWrapper<float>(devAccIn, maxPixelQuadruplets, queue)),
          pt_buf(allocBufWrapper<float>(devAccIn, maxPixelQuadruplets, queue)) {
      alpaka::memset(queue, nPixelQuadruplets_buf, 0u);
      alpaka::memset(queue, totOccupancyPixelQuadruplets_buf, 0u);
      alpaka::wait(queue);
    }

    inline PixelQuadruplets const* data() const { return &data_; }
    inline void setData(PixelQuadrupletsBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelQuadrupletToMemory(lst::Modules const& modulesInGPU,
                                                                 lst::MiniDoublets const& mdsInGPU,
                                                                 lst::Segments const& segmentsInGPU,
                                                                 lst::Quadruplets const& quadrupletsInGPU,
                                                                 lst::PixelQuadruplets& pixelQuadrupletsInGPU,
                                                                 unsigned int pixelIndex,
                                                                 unsigned int T4Index,
                                                                 unsigned int pixelQuadrupletIndex,
                                                                 float rzChiSquared,
                                                                 float rPhiChiSquared,
                                                                 float rPhiChiSquaredInwards,
                                                                 float score,
                                                                 float eta,
                                                                 float phi,
                                                                 float pixelRadius,
                                                                 float pixelRadiusError,
                                                                 float quadrupletRadius,
                                                                 float centerX,
                                                                 float centerY,
                                                                 float pt) {
    pixelQuadrupletsInGPU.pixelIndices[pixelQuadrupletIndex] = pixelIndex;
    pixelQuadrupletsInGPU.T4Indices[pixelQuadrupletIndex] = T4Index;
    pixelQuadrupletsInGPU.isDup[pixelQuadrupletIndex] = false;
    pixelQuadrupletsInGPU.score[pixelQuadrupletIndex] = __F2H(score);
    pixelQuadrupletsInGPU.eta[pixelQuadrupletIndex] = __F2H(eta);
    pixelQuadrupletsInGPU.phi[pixelQuadrupletIndex] = __F2H(phi);

    pixelQuadrupletsInGPU.pixelRadius[pixelQuadrupletIndex] = __F2H(pixelRadius);
    pixelQuadrupletsInGPU.pixelRadiusError[pixelQuadrupletIndex] = __F2H(pixelRadiusError);
    pixelQuadrupletsInGPU.quadrupletRadius[pixelQuadrupletIndex] = __F2H(quadrupletRadius);
    pixelQuadrupletsInGPU.centerX[pixelQuadrupletIndex] = __F2H(centerX);
    pixelQuadrupletsInGPU.centerY[pixelQuadrupletIndex] = __F2H(centerY);

    pixelQuadrupletsInGPU.logicalLayers[Params_pT4::kLayers * pixelQuadrupletIndex] = 0;
    pixelQuadrupletsInGPU.logicalLayers[Params_pT4::kLayers * pixelQuadrupletIndex + 1] = 0;
    pixelQuadrupletsInGPU.logicalLayers[Params_pT4::kLayers * pixelQuadrupletIndex + 2] =
        quadrupletsInGPU.logicalLayers[T4Index * Params_T4::kLayers];
    pixelQuadrupletsInGPU.logicalLayers[Params_pT4::kLayers * pixelQuadrupletIndex + 3] =
        quadrupletsInGPU.logicalLayers[T4Index * Params_T4::kLayers + 1];
    pixelQuadrupletsInGPU.logicalLayers[Params_pT4::kLayers * pixelQuadrupletIndex + 4] =
        quadrupletsInGPU.logicalLayers[T4Index * Params_T4::kLayers + 2];
    pixelQuadrupletsInGPU.logicalLayers[Params_pT4::kLayers * pixelQuadrupletIndex + 5] =
        quadrupletsInGPU.logicalLayers[T4Index * Params_T4::kLayers + 3];

    pixelQuadrupletsInGPU.lowerModuleIndices[Params_pT4::kLayers * pixelQuadrupletIndex] =
        segmentsInGPU.innerLowerModuleIndices[pixelIndex];
    pixelQuadrupletsInGPU.lowerModuleIndices[Params_pT4::kLayers * pixelQuadrupletIndex + 1] =
        segmentsInGPU.outerLowerModuleIndices[pixelIndex];
    pixelQuadrupletsInGPU.lowerModuleIndices[Params_pT4::kLayers * pixelQuadrupletIndex + 2] =
        quadrupletsInGPU.lowerModuleIndices[T4Index * Params_T4::kLayers];
    pixelQuadrupletsInGPU.lowerModuleIndices[Params_pT4::kLayers * pixelQuadrupletIndex + 3] =
        quadrupletsInGPU.lowerModuleIndices[T4Index * Params_T4::kLayers + 1];
    pixelQuadrupletsInGPU.lowerModuleIndices[Params_pT4::kLayers * pixelQuadrupletIndex + 4] =
        quadrupletsInGPU.lowerModuleIndices[T4Index * Params_T4::kLayers + 2];
    pixelQuadrupletsInGPU.lowerModuleIndices[Params_pT4::kLayers * pixelQuadrupletIndex + 5] =
        quadrupletsInGPU.lowerModuleIndices[T4Index * Params_T4::kLayers + 3];

    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[Params_pLS::kLayers * pixelIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[Params_pLS::kLayers * pixelIndex + 1];

    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex] =
        mdsInGPU.anchorHitIndices[pixelInnerMD];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 1] =
        mdsInGPU.outerHitIndices[pixelInnerMD];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 2] =
        mdsInGPU.anchorHitIndices[pixelOuterMD];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 3] =
        mdsInGPU.outerHitIndices[pixelOuterMD];

    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 4] =
        quadrupletsInGPU.hitIndices[Params_T4::kHits * T4Index];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 5] =
        quadrupletsInGPU.hitIndices[Params_T4::kHits * T4Index + 1];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 6] =
        quadrupletsInGPU.hitIndices[Params_T4::kHits * T4Index + 2];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 7] =
        quadrupletsInGPU.hitIndices[Params_T4::kHits * T4Index + 3];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 8] =
        quadrupletsInGPU.hitIndices[Params_T4::kHits * T4Index + 4];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 9] =
        quadrupletsInGPU.hitIndices[Params_T4::kHits * T4Index + 5];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 10] =
        quadrupletsInGPU.hitIndices[Params_T4::kHits * T4Index + 6];
    pixelQuadrupletsInGPU.hitIndices[Params_pT4::kHits * pixelQuadrupletIndex + 11] =
        quadrupletsInGPU.hitIndices[Params_T4::kHits * T4Index + 7];

    pixelQuadrupletsInGPU.rzChiSquared[pixelQuadrupletIndex] = rzChiSquared;
    pixelQuadrupletsInGPU.rPhiChiSquared[pixelQuadrupletIndex] = rPhiChiSquared;
    pixelQuadrupletsInGPU.rPhiChiSquaredInwards[pixelQuadrupletIndex] = rPhiChiSquaredInwards;
    pixelQuadrupletsInGPU.pt[pixelQuadrupletIndex] = pt;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT4RZChiSquaredCuts(lst::Modules const& modulesInGPU,
                                                              uint16_t lowerModuleIndex1,
                                                              uint16_t lowerModuleIndex2,
                                                              uint16_t lowerModuleIndex3,
                                                              uint16_t lowerModuleIndex4,
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
    

    // This slides shows the cut threshold definition. The comments below in the code, e.g, "cat 10", is consistent with the region separation in the slides
    // https://indico.cern.ch/event/1410985/contributions/5931017/attachments/2875400/5035406/helix%20approxi%20for%20pT4%20rzchi2%20new%20results%20versions.pdf
    // all 99% retention cuts
    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12) {  // cat 8
        return rzChiSquared < 11.958f;
      } else if (layer4 == 4) {  // cat 6
        return rzChiSquared < 5.385f;
      } else if (layer4 == 7) {  // cat 7
        return rzChiSquared < 16.717f;
      } 
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13) {  // cat 10
        return rzChiSquared < 8.608f;
      } else if (layer4 == 8) {  // cat 9
        return rzChiSquared < 15.255f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9) {  // cat 11
        return rzChiSquared < 13.638f;
      } 
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12) {  // cat 14
        return rzChiSquared < 9.466f;
      } else if (layer4 == 5) {  // cat 13
        return rzChiSquared < 4.056f;
      } 
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 7) {
      if (layer4 == 13) {  // cat 16
        return rzChiSquared < 20.042f;
      } else if (layer4 == 8) {  // cat 15
        return rzChiSquared < 60.747f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 13) {
      if (layer4 == 14) {  // cat 17
        return rzChiSquared < 13.88f;
      } 
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14) {  // cat 19
        return rzChiSquared < 11.738f;
      } else if (layer4 == 9) {  // cat 18
        return rzChiSquared < 17.664f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 13) {
      if (layer4 == 14) {  // cat 20
        return rzChiSquared < 9.142f;
      } 
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10) {  // cat 0
        return rzChiSquared < 6.937f;;
      } else if (layer4 == 15) {  // cat 1
        return rzChiSquared < 6.066f;
      }
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 14) {
      if (layer4 == 15) {  // cat 2
        return rzChiSquared < 5.608f;
      } 
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      if (layer4 == 11) {  // cat 3
        return rzChiSquared < 11.059f;
      } else if (layer4 == 16) {  // cat 4
        return rzChiSquared < 13.641f;
      }
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 15) {
      if (layer4 == 16) {  // cat 5
        printf("region 5");
        return true;
      } 
    }
    return true;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT4RPhiChiSquaredCuts(lst::Modules const& modulesInGPU,
                                                                uint16_t lowerModuleIndex1,
                                                                uint16_t lowerModuleIndex2,
                                                                uint16_t lowerModuleIndex3,
                                                                uint16_t lowerModuleIndex4,
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
    //99% retention cuts
    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12) {  // cat 8
        return rPhiChiSquared < 41.357f;
      } else if (layer4 == 4) {  // cat 6
        return rPhiChiSquared < 72.582f;
      } else if (layer4 == 7) {  // cat 7
        return rPhiChiSquared < 43.805f;
      } 
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13) {  // cat 10
        return rPhiChiSquared < 25.676f;
      } else if (layer4 == 8) {  // cat 9
        return rPhiChiSquared < 34.761f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9) {  // cat 11
        return rPhiChiSquared < 26.225f;
      } 
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12) {  // cat 14
        return rPhiChiSquared < 66.674f;
      } else if (layer4 == 5) {  // cat 13
        return rPhiChiSquared < 94.909f;
      } 
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 7) {
      if (layer4 == 13) {  // cat 16
        return rPhiChiSquared < 41.637f;
      } else if (layer4 == 8) {  // cat 15
        return rPhiChiSquared < 34.361f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 13) {
      if (layer4 == 14) {  // cat 17
        return rPhiChiSquared < 36.066f;
      } 
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14) {  // cat 19
        return rPhiChiSquared < 37.355f;
      } else if (layer4 == 9) {  // cat 18
        return rPhiChiSquared < 46.833f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 13) {
      if (layer4 == 14) {  // cat 20
        return rPhiChiSquared < 25.718f;
      } 
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10) {  // cat 0
        return rPhiChiSquared < 36.799f;
      } else if (layer4 == 15) {  // cat 1
        return rPhiChiSquared < 43.453f;
      }
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 14) {
      if (layer4 == 15) {  // cat 2
        return rPhiChiSquared < 39.417f;
      } 
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      if (layer4 == 11) {  // cat 3
        return rPhiChiSquared < 53.360f;
      } else if (layer4 == 16) {  // cat 4
        return rPhiChiSquared < 33.764f;
      }
    } 
    return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT4RPhiChiSquared(TAcc const& acc,
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

    float delta1[4], delta2[4], slopes[4];
    bool isFlat[4];
    float chiSquared = 0;

    computeSigmasForRegression(acc, modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    chiSquared = computeChiSquared(acc, 4, xs, ys, delta1, delta2, slopes, isFlat, g, f, radius);

    return chiSquared;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT4RPhiChiSquaredInwards(
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

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT4RPhiChiSquaredInwardsCuts(lst::Modules const& modulesInGPU,
                                                                       uint16_t lowerModuleIndex1,
                                                                       uint16_t lowerModuleIndex2,
                                                                       uint16_t lowerModuleIndex3,
                                                                       uint16_t lowerModuleIndex4,
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
    //99% retention cuts
    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12) {  // cat 8
        return rPhiChiSquared < 373.435f;
      } else if (layer4 == 4) {  // cat 6
        return rPhiChiSquared < 358.781f;
      } else if (layer4 == 7) {  // cat 7
        return rPhiChiSquared < 237.174f;
      } 
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13) {  // cat 10
        return rPhiChiSquared < 250.843f;
      } else if (layer4 == 8) {  // cat 9
        return rPhiChiSquared < 309.453f;
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9) {  // cat 11
        return rPhiChiSquared < 428.893f;
      } 
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12) {  // cat 14
        return rPhiChiSquared < 789.866f;
      } else if (layer4 == 5) {  // cat 13
        return rPhiChiSquared < 460.781f;
      } 
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 7) {
      if (layer4 == 13) {  // cat 16
        return rPhiChiSquared < 798.898f;
      } else if (layer4 == 8) {  // cat 15
        return rPhiChiSquared < 1233.757f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 13) {
      if (layer4 == 14) {  // cat 17
        return rPhiChiSquared < 947.383f;
      } 
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14) {  // cat 19
        return rPhiChiSquared < 869.827f;
      } else if (layer4 == 9) {  // cat 18
        return rPhiChiSquared < 600.346f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 13) {
      if (layer4 == 14) {  // cat 20
        return rPhiChiSquared < 788.826f;
      } 
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10) {  // cat 0
        return rPhiChiSquared < 586.576f;
      } else if (layer4 == 15) {  // cat 1
        return rPhiChiSquared < 924.267f;
      }
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 14) {
      if (layer4 == 15) {  // cat 2
        return rPhiChiSquared < 2450.287f;
      } 
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      if (layer4 == 11) {  // cat 3
        return rPhiChiSquared < 586.317f;
      } else if (layer4 == 16) {  // cat 4
        // return rPhiChiSquared < 5.844f;
        return false;
      }
    }
    return true; 
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPixelQuadrupletDefaultAlgo(TAcc const& acc,
                                                                    lst::Modules const& modulesInGPU,
                                                                    lst::ObjectRanges const& rangesInGPU,
                                                                    lst::MiniDoublets const& mdsInGPU,
                                                                    lst::Segments const& segmentsInGPU,
                                                                    lst::Triplets const& tripletsInGPU,
                                                                    lst::Quadruplets const& quadrupletsInGPU,
                                                                    unsigned int pixelSegmentIndex,
                                                                    unsigned int quadrupletIndex,
                                                                    float& rzChiSquared,
                                                                    float& rPhiChiSquared,
                                                                    float& rPhiChiSquaredInwards,
                                                                    float& pixelRadius,
                                                                    float& pixelRadiusError,
                                                                    float& quadrupletRadius,
                                                                    float& centerX,
                                                                    float& centerY,
                                                                    unsigned int pixelSegmentArrayIndex,
                                                                    const float ptCut) {
    unsigned int T4InnerT3Index = quadrupletsInGPU.tripletIndices[2 * quadrupletIndex];
    unsigned int T4OuterT3Index = quadrupletsInGPU.tripletIndices[2 * quadrupletIndex + 1];

    float pixelRadiusTemp, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp, centerXTemp,
        centerYTemp;
    // return true;
    if (not runPixelTripletDefaultAlgo(acc,
                                       modulesInGPU,
                                       rangesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       tripletsInGPU,
                                       pixelSegmentIndex,
                                       T4InnerT3Index,
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
    // return true; //no other cuts for now
    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * T4InnerT3Index];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * T4InnerT3Index + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * T4OuterT3Index + 1];

    unsigned int pixelInnerMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMDIndex = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];
    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];

    uint16_t lowerModuleIndex1 = quadrupletsInGPU.lowerModuleIndices[Params_T4::kLayers * quadrupletIndex];
    uint16_t lowerModuleIndex2 = quadrupletsInGPU.lowerModuleIndices[Params_T4::kLayers * quadrupletIndex + 1];
    uint16_t lowerModuleIndex3 = quadrupletsInGPU.lowerModuleIndices[Params_T4::kLayers * quadrupletIndex + 2];
    uint16_t lowerModuleIndex4 = quadrupletsInGPU.lowerModuleIndices[Params_T4::kLayers * quadrupletIndex + 3];

    uint16_t lowerModuleIndices[Params_T4::kLayers] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4};

    float rtPix[Params_pLS::kLayers] = {mdsInGPU.anchorRt[pixelInnerMDIndex], mdsInGPU.anchorRt[pixelOuterMDIndex]};
    float xPix[Params_pLS::kLayers] = {mdsInGPU.anchorX[pixelInnerMDIndex], mdsInGPU.anchorX[pixelOuterMDIndex]};
    float yPix[Params_pLS::kLayers] = {mdsInGPU.anchorY[pixelInnerMDIndex], mdsInGPU.anchorY[pixelOuterMDIndex]};
    float zPix[Params_pLS::kLayers] = {mdsInGPU.anchorZ[pixelInnerMDIndex], mdsInGPU.anchorZ[pixelOuterMDIndex]};

    float zs[Params_T4::kLayers] = {mdsInGPU.anchorZ[firstMDIndex],
                                    mdsInGPU.anchorZ[secondMDIndex],
                                    mdsInGPU.anchorZ[thirdMDIndex],
                                    mdsInGPU.anchorZ[fourthMDIndex]};
    float rts[Params_T4::kLayers] = {mdsInGPU.anchorRt[firstMDIndex],
                                     mdsInGPU.anchorRt[secondMDIndex],
                                     mdsInGPU.anchorRt[thirdMDIndex],
                                     mdsInGPU.anchorRt[fourthMDIndex]};

    float pixelSegmentPt = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
    float pixelSegmentPx = segmentsInGPU.px[pixelSegmentArrayIndex];
    float pixelSegmentPy = segmentsInGPU.py[pixelSegmentArrayIndex];
    float pixelSegmentPz = segmentsInGPU.pz[pixelSegmentArrayIndex];
    int pixelSegmentCharge = segmentsInGPU.charge[pixelSegmentArrayIndex];

    float pixelSegmentPtError = segmentsInGPU.ptErr[pixelSegmentArrayIndex];
    pixelRadiusError = pixelSegmentPtError * kR1GeVf;

    rzChiSquared = 0;

    //get the appropriate centers
    pixelRadius = segmentsInGPU.circleRadius[pixelSegmentArrayIndex];

    rzChiSquared = computePT4RZChiSquared(acc,
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
    if (not passPT4RZChiSquaredCuts(modulesInGPU,
                                      lowerModuleIndex1,
                                      lowerModuleIndex2,
                                      lowerModuleIndex3,
                                      lowerModuleIndex4,
                                      rzChiSquared))
        return false;

    // if (pixelRadius < 5.0f * kR1GeVf) {  //only apply r-z chi2 cuts for <5GeV tracks
    //   rzChiSquared = computePT4RZChiSquared(acc,
    //                                         modulesInGPU,
    //                                         lowerModuleIndices,
    //                                         rtPix,
    //                                         xPix,
    //                                         yPix,
    //                                         zPix,
    //                                         rts,
    //                                         zs,
    //                                         pixelSegmentPt,
    //                                         pixelSegmentPx,
    //                                         pixelSegmentPy,
    //                                         pixelSegmentPz,
    //                                         pixelSegmentCharge);
    //   if (not passPT4RZChiSquaredCuts(modulesInGPU,
    //                                   lowerModuleIndex1,
    //                                   lowerModuleIndex2,
    //                                   lowerModuleIndex3,
    //                                   lowerModuleIndex4,
    //                                   rzChiSquared))
    //     return false;
    // }

    //outer T4
    float xs[Params_T4::kLayers] = {mdsInGPU.anchorX[firstMDIndex],
                                    mdsInGPU.anchorX[secondMDIndex],
                                    mdsInGPU.anchorX[thirdMDIndex],
                                    mdsInGPU.anchorX[fourthMDIndex]};
    float ys[Params_T4::kLayers] = {mdsInGPU.anchorY[firstMDIndex],
                                    mdsInGPU.anchorY[secondMDIndex],
                                    mdsInGPU.anchorY[thirdMDIndex],
                                    mdsInGPU.anchorY[fourthMDIndex]};

    // //get the appropriate centers
    centerX = segmentsInGPU.circleCenterX[pixelSegmentArrayIndex];
    centerY = segmentsInGPU.circleCenterY[pixelSegmentArrayIndex];

    float T4CenterX = quadrupletsInGPU.regressionG[quadrupletIndex];
    float T4CenterY = quadrupletsInGPU.regressionF[quadrupletIndex];
    quadrupletRadius = quadrupletsInGPU.regressionRadius[quadrupletIndex];
    float quadrupletEta = quadrupletsInGPU.eta[quadrupletIndex];

    rPhiChiSquared =
        computePT4RPhiChiSquared(acc, modulesInGPU, lowerModuleIndices, centerX, centerY, pixelRadius, xs, ys);

    // if (pixelRadius < 5.0f * kR1GeVf) {
    //   if (not passPT4RPhiChiSquaredCuts(modulesInGPU,
    //                                     lowerModuleIndex1,
    //                                     lowerModuleIndex2,
    //                                     lowerModuleIndex3,
    //                                     lowerModuleIndex4,
    //                                     rPhiChiSquared))
    //     return false;
    // }

    rPhiChiSquaredInwards = computePT4RPhiChiSquaredInwards(T4CenterX, T4CenterY, quadrupletRadius, xPix, yPix);

    // if (quadrupletsInGPU.regressionRadius[quadrupletIndex] < 5.0f * kR1GeVf) {
    //   if (not passPT4RPhiChiSquaredInwardsCuts(modulesInGPU,
    //                                            lowerModuleIndex1,
    //                                            lowerModuleIndex2,
    //                                            lowerModuleIndex3,
    //                                            lowerModuleIndex4,
    //                                            rPhiChiSquaredInwards))
    //     return false;
    // }
    float T4InnerRadius = quadrupletsInGPU.innerRadius[quadrupletIndex];
    bool inference = lst::pt4dnn::runInference(acc,
                                              T4InnerRadius,
                                              pixelSegmentPt,
                                              rPhiChiSquared,
                                              quadrupletRadius,
                                              pixelRadius,
                                              pixelRadiusError,
                                              rzChiSquared,
                                              quadrupletEta);
    if (!inference)
      return false;
    //trusting the T4 regression center to also be a good estimate..
    centerX = (centerX + T4CenterX) / 2;
    centerY = (centerY + T4CenterY) / 2;

    return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computePT4RZChiSquared(TAcc const& acc,
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

    for (size_t i = 0; i < Params_T4::kLayers; i++) {
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

  struct createPixelQuadrupletsInGPUFromMapv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::MiniDoublets mdsInGPU,
                                  lst::Segments segmentsInGPU,
                                  lst::Triplets tripletsInGPU,
                                  lst::Quadruplets quadrupletsInGPU,
                                  lst::PixelQuadruplets pixelQuadrupletsInGPU,
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
        // continue; //don't build any pT4s
        auto iLSModule_max = connectedPixelIndex[i_pLS] + connectedPixelSize[i_pLS];
        for (unsigned int iLSModule = connectedPixelIndex[i_pLS] + globalBlockIdx[0]; iLSModule < iLSModule_max;
             iLSModule += gridBlockExtent[0]) {
          //these are actual module indices
          uint16_t quadrupletLowerModuleIndex = modulesInGPU.connectedPixels[iLSModule];
          if (quadrupletLowerModuleIndex >= *modulesInGPU.nLowerModules)
            continue;
          if (modulesInGPU.moduleType[quadrupletLowerModuleIndex] == lst::TwoS)
            continue;
          uint16_t pixelModuleIndex = *modulesInGPU.nLowerModules;
          if (segmentsInGPU.isDup[i_pLS])
            continue;
          unsigned int nOuterQuadruplets = quadrupletsInGPU.nQuadruplets[quadrupletLowerModuleIndex];

          if (nOuterQuadruplets == 0)
            continue;

          unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + i_pLS;

          //fetch the quadruplet
          for (unsigned int outerQuadrupletArrayIndex = globalThreadIdx[2];
               outerQuadrupletArrayIndex < nOuterQuadruplets;
               outerQuadrupletArrayIndex += gridThreadExtent[2]) {
            unsigned int quadrupletIndex =
                rangesInGPU.quadrupletModuleIndices[quadrupletLowerModuleIndex] + outerQuadrupletArrayIndex;

            if (quadrupletsInGPU.isDup[quadrupletIndex])
              continue;
            // if (!(quadrupletsInGPU.TightDisplacedFlag[quadrupletIndex]))
            //   continue;

            float rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, pixelRadiusError, quadrupletRadius, centerX, centerY;

            bool success = runPixelQuadrupletDefaultAlgo(acc,
                                                         modulesInGPU,
                                                         rangesInGPU,
                                                         mdsInGPU,
                                                         segmentsInGPU,
                                                         tripletsInGPU,
                                                         quadrupletsInGPU,
                                                         pixelSegmentIndex,
                                                         quadrupletIndex,
                                                         rzChiSquared,
                                                         rPhiChiSquared,
                                                         rPhiChiSquaredInwards,
                                                         pixelRadius,
                                                         pixelRadiusError,
                                                         quadrupletRadius,
                                                         centerX,
                                                         centerY,
                                                         static_cast<unsigned int>(i_pLS),
                                                         ptCut);
            if (success) {
              unsigned int totOccupancyPixelQuadruplets =
                  alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelQuadrupletsInGPU.totOccupancyPixelQuadruplets, 1u);
              if (totOccupancyPixelQuadruplets >= n_max_pixel_quadruplets) {
#ifdef WARNINGS
                printf("Pixel Quadruplet excess alert!\n");
#endif
              } else {
                unsigned int pixelQuadrupletIndex =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelQuadrupletsInGPU.nPixelQuadruplets, 1u);
                float eta = __H2F(quadrupletsInGPU.eta[quadrupletIndex]);
                float phi = __H2F(quadrupletsInGPU.phi[quadrupletIndex]);
                float pt =  (__H2F(quadrupletsInGPU.innerRadius[quadrupletIndex]) * lst::k2Rinv1GeVf * 2 + segmentsInGPU.ptIn[i_pLS]) / 2; 

                addPixelQuadrupletToMemory(modulesInGPU,
                                           mdsInGPU,
                                           segmentsInGPU,
                                           quadrupletsInGPU,
                                           pixelQuadrupletsInGPU,
                                           pixelSegmentIndex,
                                           quadrupletIndex,
                                           pixelQuadrupletIndex,
                                           rzChiSquared,
                                           rPhiChiSquared,
                                           rPhiChiSquaredInwards,
                                           rPhiChiSquared,
                                           eta,
                                           phi,
                                           pixelRadius,
                                           pixelRadiusError,
                                           quadrupletRadius,
                                           centerX,
                                           centerY,
                                           pt);

                tripletsInGPU.partOfPT4[quadrupletsInGPU.tripletIndices[2 * quadrupletIndex]] = true;
                tripletsInGPU.partOfPT4[quadrupletsInGPU.tripletIndices[2 * quadrupletIndex + 1]] = true;
                segmentsInGPU.partOfPT4[i_pLS] = true;
                quadrupletsInGPU.partOfPT4[quadrupletIndex] = true;
              }  // tot occupancy
            }  // end success
          }  // end T4
        }  // end iLS
      }  // end i_pLS
    }
  };
}  // namespace lst
#endif
