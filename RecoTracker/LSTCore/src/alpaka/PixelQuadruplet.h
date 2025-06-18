#ifndef RecoTracker_LSTCore_src_alpaka_PixelQuadruplet_h
#define RecoTracker_LSTCore_src_alpaka_PixelQuadruplet_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/HitsSoA.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/QuadrupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"

#include "Quadruplet.h"
#include "PixelTriplet.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPixelQuadrupletToMemory(ModulesConst modules,
                                                                 MiniDoubletsConst mds,
                                                                 SegmentsConst segments,
                                                                 QuadrupletsConst quadruplets,
                                                                 PixelQuadruplets pixelQuadruplets,
                                                                 unsigned int pixelIndex,
                                                                 unsigned int t4Index,
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
    pixelQuadruplets.pixelIndices()[pixelQuadrupletIndex] = pixelIndex;
    pixelQuadruplets.quadrupletIndices()[pixelQuadrupletIndex] = t4Index;
    pixelQuadruplets.isDup()[pixelQuadrupletIndex] = false;
    pixelQuadruplets.score()[pixelQuadrupletIndex] = __F2H(score);
    pixelQuadruplets.eta()[pixelQuadrupletIndex] = __F2H(eta);
    pixelQuadruplets.phi()[pixelQuadrupletIndex] = __F2H(phi);

    pixelQuadruplets.pixelRadius()[pixelQuadrupletIndex] = __F2H(pixelRadius);
    pixelQuadruplets.pixelRadiusError()[pixelQuadrupletIndex] = __F2H(pixelRadiusError);
    pixelQuadruplets.quadrupletRadius()[pixelQuadrupletIndex] = __F2H(quadrupletRadius);
    pixelQuadruplets.centerX()[pixelQuadrupletIndex] = __F2H(centerX);
    pixelQuadruplets.centerY()[pixelQuadrupletIndex] = __F2H(centerY);

    pixelQuadruplets.logicalLayers()[pixelQuadrupletIndex][0] = 0;
    pixelQuadruplets.logicalLayers()[pixelQuadrupletIndex][1] = 0;
    pixelQuadruplets.logicalLayers()[pixelQuadrupletIndex][2] = quadruplets.logicalLayers()[t4Index][0];
    pixelQuadruplets.logicalLayers()[pixelQuadrupletIndex][3] = quadruplets.logicalLayers()[t4Index][1];
    pixelQuadruplets.logicalLayers()[pixelQuadrupletIndex][4] = quadruplets.logicalLayers()[t4Index][2];
    pixelQuadruplets.logicalLayers()[pixelQuadrupletIndex][5] = quadruplets.logicalLayers()[t4Index][3];

    pixelQuadruplets.lowerModuleIndices()[pixelQuadrupletIndex][0] = segments.innerLowerModuleIndices()[pixelIndex];
    pixelQuadruplets.lowerModuleIndices()[pixelQuadrupletIndex][1] = segments.outerLowerModuleIndices()[pixelIndex];
    pixelQuadruplets.lowerModuleIndices()[pixelQuadrupletIndex][2] = quadruplets.lowerModuleIndices()[t4Index][0];
    pixelQuadruplets.lowerModuleIndices()[pixelQuadrupletIndex][3] = quadruplets.lowerModuleIndices()[t4Index][1];
    pixelQuadruplets.lowerModuleIndices()[pixelQuadrupletIndex][4] = quadruplets.lowerModuleIndices()[t4Index][2];
    pixelQuadruplets.lowerModuleIndices()[pixelQuadrupletIndex][5] = quadruplets.lowerModuleIndices()[t4Index][3];

    unsigned int pixelInnerMD = segments.mdIndices()()[pixelIndex][0];
    unsigned int pixelOuterMD = segments.mdIndices()()[pixelIndex][1];

    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][0] = mds.anchorHitIndices()[pixelInnerMD];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][1] = mds.outerHitIndices()[pixelInnerMD];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][2] = mds.anchorHitIndices()[pixelOuterMD];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][3] = mds.outerHitIndices()[pixelOuterMD];

    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][4] = quadruplets.hitIndices()[t4Index][0];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][5] = quadruplets.hitIndices()[t4Index][1];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][6] = quadruplets.hitIndices()[t4Index][2];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][7] = quadruplets.hitIndices()[t4Index][3];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][8] = quadruplets.hitIndices()[t4Index][4];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][9] = quadruplets.hitIndices()[t4Index][5];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][10] = quadruplets.hitIndices()[t4Index][6];
    pixelQuadruplets.hitIndices()[pixelQuadrupletIndex][11] = quadruplets.hitIndices()[t4Index][7];

    pixelQuadruplets.rzChiSquared()[pixelQuadrupletIndex] = rzChiSquared;
    pixelQuadruplets.rPhiChiSquared()[pixelQuadrupletIndex] = rPhiChiSquared;
    pixelQuadruplets.rPhiChiSquaredInwards()[pixelQuadrupletIndex] = rPhiChiSquaredInwards;
    pixelQuadruplets.pt()[pixelQuadrupletIndex] = pt;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT4RZChiSquaredCuts(ModulesConst modules,
                                                              uint16_t lowerModuleIndex1,
                                                              uint16_t lowerModuleIndex2,
                                                              uint16_t lowerModuleIndex3,
                                                              uint16_t lowerModuleIndex4,
                                                              float rzChiSquared) {
    const int layer1 =
        modules.layers()[lowerModuleIndex1] + 6 * (modules.subdets()[lowerModuleIndex1] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modules.layers()[lowerModuleIndex2] + 6 * (modules.subdets()[lowerModuleIndex2] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modules.layers()[lowerModuleIndex3] + 6 * (modules.subdets()[lowerModuleIndex3] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS);
    const int layer4 =
        modules.layers()[lowerModuleIndex4] + 6 * (modules.subdets()[lowerModuleIndex4] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex4] == Endcap and modules.moduleType()[lowerModuleIndex4] == TwoS);
    

    // This slides shows the cut threshold definition. The comments below in the code, e.g, "cat 10", is consistent with the region separation in the slides
    // https://indico.cern.ch/event/1410985/contributions/5931017/attachments/2875400/5035406/helix%20approxi%20for%20pT4%20rzchi2%20new%20results%20versions.pdf
    // all 99% retention cuts
    if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 12) {  // cat 8
        return rzChiSquared < 10.358f; //radii and t3 scores in t4 dnn
      } else if (layer4 == 4) {  // cat 6
        return rzChiSquared < 9.514f; //radii and t3 scores in t4 dnn
      } else if (layer4 == 7) {  // cat 7
        return rzChiSquared < 9.441f; //radii and t3 scores in t4 dnn
      } 
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 13) {  // cat 10
        return rzChiSquared < 9.415f; //radii and t3 scores in t4 dnn
      } else if (layer4 == 8) {  // cat 9
        return rzChiSquared < 10.128f; //radii and t3 scores in t4 dnn
      }
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9) {  // cat 11
        return rzChiSquared < 9.943f; //radii and t3 scores in t4 dnn
      } 
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 12) {  // cat 14
        return rzChiSquared < 0.081f; //radii and t3 scores in t4 dnn
      } else if (layer4 == 5) {  // cat 13
        return true;
      } 
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 7) {
      if (layer4 == 13) {  // cat 16
        return rzChiSquared < 3.895f; //radii and t3 scores in t4 dnn
      } else if (layer4 == 8) {  // cat 15
        return true;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer4 == 13) {
      if (layer4 == 14) {  // cat 17
        return rzChiSquared < 4.158f; //radii and t3 scores in t4 dnn
      } 
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 14) {  // cat 19
        return rzChiSquared < 5.118f; //radii and t3 scores in t4 dnn
      } else if (layer4 == 9) {  // cat 18
        return rzChiSquared < 5.178f; //radii and t3 scores in t4 dnn
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 13) {
      if (layer4 == 14) {  // cat 20
        return rzChiSquared < 9.034f; //radii and t3 scores in t4 dnn
      } 
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9) {
      if (layer4 == 10) {  // cat 0
        return rzChiSquared < 10.125f; //radii and t3 scores in t4 dnn
      } else if (layer4 == 15) {  // cat 1
        return rzChiSquared < 9.769f; //radii and t3 scores in t4 dnn
      }
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 14) {
      if (layer4 == 15) {  // cat 2
        return rzChiSquared < 9.721f; //radii and t3 scores in t4 dnn
      } 
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      if (layer4 == 11) {  // cat 3
        // return rzChiSquared < 11.059f;
        // return rzChiSquared < 10.202f;
        // return rzChiSquared < 10.528f; //radii in t4 dnn
        return rzChiSquared < 10.452f; //radii and t3 scores in t4 dnn
      } else if (layer4 == 16) {  // cat 4
        return rzChiSquared < 11.063f; //radii and t3 scores in t4 dnn
      }
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 15) {
      if (layer4 == 16) {  // cat 5
        return rzChiSquared < 10.734f; //radii and t3 scores in t4 dnn
      } 
    }
    return true;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT4RPhiChiSquaredCuts(ModulesConst modules,
                                                                uint16_t lowerModuleIndex1,
                                                                uint16_t lowerModuleIndex2,
                                                                uint16_t lowerModuleIndex3,
                                                                uint16_t lowerModuleIndex4,
                                                                float rPhiChiSquared) {
    const int layer1 =
        modules.layers()[lowerModuleIndex1] + 6 * (modules.subdets()[lowerModuleIndex1] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modules.layers()[lowerModuleIndex2] + 6 * (modules.subdets()[lowerModuleIndex2] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modules.layers()[lowerModuleIndex3] + 6 * (modules.subdets()[lowerModuleIndex3] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS);
    const int layer4 =
        modules.layers()[lowerModuleIndex4] + 6 * (modules.subdets()[lowerModuleIndex4] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex4] == Endcap and modules.moduleType()[lowerModuleIndex4] == TwoS);
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
                                                                ModulesConst modules,
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

    computeSigmasForRegression(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat);
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

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPT4RPhiChiSquaredInwardsCuts(ModulesConst modules,
                                                                       uint16_t lowerModuleIndex1,
                                                                       uint16_t lowerModuleIndex2,
                                                                       uint16_t lowerModuleIndex3,
                                                                       uint16_t lowerModuleIndex4,
                                                                       float rPhiChiSquared) {
    const int layer1 =
        modules.layers()[lowerModuleIndex1] + 6 * (modules.subdets()[lowerModuleIndex1] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex1] == Endcap and modules.moduleType()[lowerModuleIndex1] == TwoS);
    const int layer2 =
        modules.layers()[lowerModuleIndex2] + 6 * (modules.subdets()[lowerModuleIndex2] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex2] == Endcap and modules.moduleType()[lowerModuleIndex2] == TwoS);
    const int layer3 =
        modules.layers()[lowerModuleIndex3] + 6 * (modules.subdets()[lowerModuleIndex3] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex3] == Endcap and modules.moduleType()[lowerModuleIndex3] == TwoS);
    const int layer4 =
        modules.layers()[lowerModuleIndex4] + 6 * (modules.subdets()[lowerModuleIndex4] == Endcap) +
        5 * (modules.subdets()[lowerModuleIndex4] == Endcap and modules.moduleType()[lowerModuleIndex4] == TwoS);
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
                                                                    ModulesConst modules,
                                                                    ObjectRangesConst ranges,
                                                                    MiniDoubletsConst mds,
                                                                    SegmentsConst segments,
                                                                    PixelSeedsConst pixelSeeds,
                                                                    PixelSegmentsConst pixelSegments,
                                                                    TripletsConst triplets,
                                                                    QuadrupletsConst quadruplets,
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
    unsigned int t4InnerT3Index = quadruplets.tripletIndices()[quadrupletIndex][0];
    unsigned int t4OuterT3Index = quadruplets.tripletIndices()[quadrupletIndex][1];

    float pixelRadiusTemp, tripletRadius, rPhiChiSquaredTemp, rzChiSquaredTemp, rPhiChiSquaredInwardsTemp, centerXTemp,
        centerYTemp, pixelRadiusErrorTemp;
    
    if (not runPixelTripletDefaultAlgo(acc,
                                       modules,
                                       ranges,
                                       mds,
                                       segments,
                                       pixelSeeds,
                                       pixelSegments,
                                       triplets,
                                       pixelSegmentIndex,
                                       t4InnerT3Index,
                                       pixelRadiusTemp,
                                       tripletRadius,
                                       centerXTemp,
                                       centerYTemp,
                                       rzChiSquaredTemp,
                                       rPhiChiSquaredTemp,
                                       rPhiChiSquaredInwardsTemp,
                                       pixelRadiusErrorTemp,
                                       ptCut,
                                       true,
                                       false))
      return false;
    
    unsigned int firstSegmentIndex = triplets.segmentIndices()[t4InnerT3Index][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[t4InnerT3Index][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[t4OuterT3Index][1];

    unsigned int pixelInnerMDIndex = segments.mdIndices()[pixelSegmentIndex][0];
    unsigned int pixelOuterMDIndex = segments.mdIndices()[pixelSegmentIndex][1];
    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];

    uint16_t lowerModuleIndex1 = quadruplets.lowerModuleIndices()[quadrupletIndex][0];
    uint16_t lowerModuleIndex2 = quadruplets.lowerModuleIndices()[quadrupletIndex][1];
    uint16_t lowerModuleIndex3 = quadruplets.lowerModuleIndices()[quadrupletIndex][2];
    uint16_t lowerModuleIndex4 = quadruplets.lowerModuleIndices()[quadrupletIndex][3];

    uint16_t lowerModuleIndices[Params_T4::kLayers] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4};

    float rtPix[Params_pLS::kLayers] = {mds.anchorRt()[pixelInnerMDIndex], mds.anchorRt()[pixelOuterMDIndex]};
    float xPix[Params_pLS::kLayers] = {mds.anchorX()[pixelInnerMDIndex], mds.anchorX()[pixelOuterMDIndex]};
    float yPix[Params_pLS::kLayers] = {mds.anchorY()[pixelInnerMDIndex], mds.anchorY()[pixelOuterMDIndex]};
    float zPix[Params_pLS::kLayers] = {mds.anchorZ()()[pixelInnerMDIndex], mds.anchorZ()[pixelOuterMDIndex]};

    float zs[Params_T4::kLayers] = {mds.anchorZ()[firstMDIndex],
                                    mds.anchorZ()[secondMDIndex],
                                    mds.anchorZ()[thirdMDIndex],
                                    mds.anchorZ()[fourthMDIndex]};
    float rts[Params_T4::kLayers] = {mds.anchorRt()[firstMDIndex],
                                     mds.anchorRt()[secondMDIndex],
                                     mds.anchorRt()[thirdMDIndex],
                                     mds.anchorRt()[fourthMDIndex]};

    float pixelSegmentPt = segments.ptIn()[pixelSegmentArrayIndex];
    float pixelSegmentPx = segments.px()[pixelSegmentArrayIndex];
    float pixelSegmentPy = segments.py()[pixelSegmentArrayIndex];
    float pixelSegmentPz = segments.pz()[pixelSegmentArrayIndex];
    int pixelSegmentCharge = segments.charge()[pixelSegmentArrayIndex];

    float pixelSegmentPtError = segments.ptErr()[pixelSegmentArrayIndex];
    pixelRadiusError = pixelSegmentPtError * kR1GeVf;

    rzChiSquared = 0;

    //get the appropriate centers
    pixelRadius = segments.circleRadius()[pixelSegmentArrayIndex];

    rzChiSquared = computePT4RZChiSquared(acc,
                                              modules,
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
    if (not passPT4RZChiSquaredCuts(modules,
                                      lowerModuleIndex1,
                                      lowerModuleIndex2,
                                      lowerModuleIndex3,
                                      lowerModuleIndex4,
                                      rzChiSquared))
        return false;

    //outer T4
    float xs[Params_T4::kLayers] = {mds.anchorX()[firstMDIndex],
                                    mds.anchorX()[secondMDIndex],
                                    mds.anchorX()[thirdMDIndex],
                                    mds.anchorX()[fourthMDIndex]};
    float ys[Params_T4::kLayers] = {mds.anchorY()[firstMDIndex],
                                    mds.anchorY()[secondMDIndex],
                                    mds.anchorY()[thirdMDIndex],
                                    mds.anchorY()[fourthMDIndex]};

    // //get the appropriate centers
    centerX = segments.circleCenterX()[pixelSegmentArrayIndex];
    centerY = segments.circleCenterY()[pixelSegmentArrayIndex];

    float T4CenterX = quadruplets.regressionG()[quadrupletIndex];
    float T4CenterY = quadruplets.regressionF()[quadrupletIndex];
    quadrupletRadius = quadruplets.regressionRadius()[quadrupletIndex];
    float quadrupletEta = quadruplets.eta()[quadrupletIndex];

    rPhiChiSquared =
        computePT4RPhiChiSquared(acc, modules, lowerModuleIndices, centerX, centerY, pixelRadius, xs, ys);

    rPhiChiSquaredInwards = computePT4RPhiChiSquaredInwards(T4CenterX, T4CenterY, quadrupletRadius, xPix, yPix);

    float T4InnerRadius = quadruplets.innerRadius()[quadrupletIndex];
    
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
                                                              ModulesConst modules,
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

    float a = -100 / kR1GeVf * charge;

    for (size_t i = 0; i < Params_T4::kLayers; i++) {
      float zsi = zs[i] / 100;
      float rtsi = rts[i] / 100;
      uint16_t lowerModuleIndex = lowerModuleIndices[i];
      const int moduleType = modules.moduleType()[lowerModuleIndex];
      const int moduleSide = modules.sides()[lowerModuleIndex];
      const int moduleSubdet = modules.subdets()[lowerModuleIndex];

      // calculation is detailed documented here https://indico.cern.ch/event/1185895/contributions/4982756/attachments/2526561/4345805/helix%20pT3%20summarize.pdf
      float diffr, diffz;
      float p = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);

      float rou = a / p;
      if (moduleSubdet == Endcap) {
        float s = (zsi - z1) * p / Pz;
        float x = x1 + Px / a * alpaka::math::sin(acc, rou * s) - Py / a * (1 - alpaka::math::cos(acc, rou * s));
        float y = y1 + Py / a * alpaka::math::sin(acc, rou * s) + Px / a * (1 - alpaka::math::cos(acc, rou * s));
        diffr = alpaka::math::abs(acc, rtsi - alpaka::math::sqrt(acc, x * x + y * y)) * 100;
      }

      if (moduleSubdet == Barrel) {
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

      residual = moduleSubdet == Barrel ? diffz : diffr;

      //PS Modules
      if (moduleType == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

      //special dispensation to tilted PS modules!
      if (moduleType == 0 and moduleSubdet == Barrel and moduleSide != Center) {
        float drdz = modules.drdzs()[lowerModuleIndex];
        error2 /= (1.f + drdz * drdz);
      }
      RMSE += (residual * residual) / error2;
    }

    RMSE = alpaka::math::sqrt(acc, 0.2f * RMSE);  // Divided by the degree of freedom 5.
    return RMSE;
  };

  struct CreatePixelQuadrupletsFromMap {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  ModulesPixelConst modulesPixel,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegments pixelSegments,
                                  TripletsConst triplets,
                                  QuadrupletsConst quadruplets,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  PixelQuadruplets pixelQuadruplets,
                                  unsigned int* connectedPixelSize,
                                  unsigned int* connectedPixelIndex,
                                  unsigned int nPixelSegments,
                                  ObjectRangesConst ranges,
                                  const float ptCut) const {
      for (unsigned int i_pLS : cms::alpakatools::uniform_elements_z(acc, nPixelSegments)) {
        auto iLSModule_max = connectedPixelIndex[i_pLS] + connectedPixelSize[i_pLS];
        for (unsigned int iLSModule :
             cms::alpakatools::uniform_elements_y(acc, connectedPixelIndex[i_pLS], iLSModule_max)) {
          //these are actual module indices
          uint16_t quadrupletLowerModuleIndex = modules.connectedPixels()[iLSModule];
          if (quadrupletLowerModuleIndex >= *modules.nLowerModules())
            continue;
          if (modules.moduleType()[quadrupletLowerModuleIndex] == TwoS)
            continue;
          uint16_t pixelModuleIndex = *modules.nLowerModules();
          if (segments.isDup()[i_pLS])
            continue;
          unsigned int nOuterQuadruplets = quadrupletsOccupancy.nQuadruplets()[quadrupletLowerModuleIndex];

          if (nOuterQuadruplets == 0)
            continue;

          unsigned int pixelSegmentIndex = ranges.segmentModuleIndices()[pixelModuleIndex] + i_pLS;

          //fetch the quadruplet
          for (unsigned int outerQuadrupletArrayIndex: cms::alpakatools::uniform_elements_x(acc, nOuterQuadruplets)) {
            unsigned int quadrupletIndex =
                ranges.quadrupletModuleIndices()[quadrupletLowerModuleIndex] + outerQuadrupletArrayIndex;

            if (quadruplets.isDup()[quadrupletIndex])
              continue;

            float rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, pixelRadiusError, quadrupletRadius, centerX, centerY;

            bool success = runPixelQuadrupletDefaultAlgo(acc,
                                                         modules,
                                                         ranges,
                                                         mds,
                                                         segments,
                                                         pixelSeeds,
                                                         pixelSegments,
                                                         triplets,
                                                         quadruplets,
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
                  alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelQuadruplets.totOccupancyPixelQuadruplets(), 1u, alpaka::hierarchy::Threads{});
              if (totOccupancyPixelQuadruplets >= n_max_pixel_quadruplets) {
#ifdef WARNINGS
                printf("Pixel Quadruplet excess alert!\n");
#endif
              } else {
                unsigned int pixelQuadrupletIndex =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, pixelQuadruplets.nPixelQuadruplets(), 1u, alpaka::hierarchy::Threads{});
                // float eta = __H2F(quadruplets.eta[quadrupletIndex]);
                // float phi = __H2F(quadruplets.phi[quadrupletIndex]);
                int layer = modules.layers()[quadrupletLowerModuleIndex];
                short layer2_adjustment;
                if (layer == 1) {
                  layer2_adjustment = 1;
                } else {
                  layer2_adjustment = 0;
                }
                unsigned int innerTripletIndex = quadruplets.tripletIndices()[quadrupletIndex][0];

                float phi =
                      mds.anchorPhi()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][0]][layer2_adjustment]]; //layer 2
                float eta =
                      mds.anchorEta()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][0]][layer2_adjustment]]; //layer 2

                float pt =  (__H2F(quadruplets.innerRadius()[quadrupletIndex]) * k2Rinv1GeVf * 2 + segments.ptIn()[i_pLS]) / 2; 

                addPixelQuadrupletToMemory(modules,
                                           mds,
                                           segments,
                                           quadruplets,
                                           pixelQuadruplets,
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

                triplets.partOfPT4()[quadruplets.tripletIndices()[quadrupletIndex][0]] = true;
                triplets.partOfPT4()[quadruplets.tripletIndices()[quadrupletIndex][1]] = true;
                segments.partOfPT4()[i_pLS] = true;
                quadruplets.partOfPT4()[quadrupletIndex] = true;
                
              }  // tot occupancy
            }  // end success
          }  // end T4
        }  // end iLS
      }  // end i_pLS
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
