#ifndef RecoTracker_LSTCore_src_alpaka_Quadruplet_h
#define RecoTracker_LSTCore_src_alpaka_Quadruplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuadrupletsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/Circle.h"

#include "Quintuplet.h"

#include "NeuralNetwork.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuadrupletToMemory(TripletsConst triplets,
                                                            Quadruplets quadruplets,
                                                            unsigned int innerTripletIndex,
                                                            unsigned int outerTripletIndex,
                                                            uint16_t lowerModule1,
                                                            uint16_t lowerModule2,
                                                            uint16_t lowerModule3,
                                                            uint16_t lowerModule4,
                                                            float innerRadius,
                                                            float outerRadius,
                                                            float pt,
                                                            float eta,
                                                            float phi,
                                                            float scores,
                                                            uint8_t layer,
                                                            unsigned int quadrupletIndex,
                                                            float rzChiSquared,
                                                            float dBeta,
                                                            float promptScore,
                                                            float displacedScore,
                                                            float fakeScore,
                                                            float regressionG,
                                                            float regressionF,
                                                            float regressionRadius,
                                                            float nonAnchorRegressionRadius) {
    quadruplets.tripletIndices()[quadrupletIndex][0] = innerTripletIndex;
    quadruplets.tripletIndices()[quadrupletIndex][1] = outerTripletIndex;

    quadruplets.lowerModuleIndices()[quadrupletIndex][0] = lowerModule1;
    quadruplets.lowerModuleIndices()[quadrupletIndex][1] = lowerModule2;
    quadruplets.lowerModuleIndices()[quadrupletIndex][2] = lowerModule3;
    quadruplets.lowerModuleIndices()[quadrupletIndex][3] = lowerModule4;
    quadruplets.innerRadius()[quadrupletIndex] = __F2H(innerRadius);
    quadruplets.outerRadius()[quadrupletIndex] = __F2H(outerRadius);
    quadruplets.pt()[quadrupletIndex] = __F2H(pt);
    quadruplets.eta()[quadrupletIndex] = __F2H(eta);
    quadruplets.phi()[quadrupletIndex] = __F2H(phi);
    quadruplets.score_rphisum()[quadrupletIndex] = __F2H(scores);
    quadruplets.layer()[quadrupletIndex] = layer;
    quadruplets.isDup()[quadrupletIndex] = 0;
    quadruplets.logicalLayers()[quadrupletIndex][0] = triplets.logicalLayers()[innerTripletIndex][0];
    quadruplets.logicalLayers()[quadrupletIndex][1] = triplets.logicalLayers()[innerTripletIndex][1];
    quadruplets.logicalLayers()[quadrupletIndex][2] = triplets.logicalLayers()[innerTripletIndex][2];
    quadruplets.logicalLayers()[quadrupletIndex][3] = triplets.logicalLayers()[outerTripletIndex][2];

    quadruplets.hitIndices()[quadrupletIndex][0] = triplets.hitIndices()[innerTripletIndex][0];
    quadruplets.hitIndices()[quadrupletIndex][1] = triplets.hitIndices()[innerTripletIndex][1];
    quadruplets.hitIndices()[quadrupletIndex][2] = triplets.hitIndices()[innerTripletIndex][2];
    quadruplets.hitIndices()[quadrupletIndex][3] = triplets.hitIndices()[innerTripletIndex][3];
    quadruplets.hitIndices()[quadrupletIndex][4] = triplets.hitIndices()[innerTripletIndex][4];
    quadruplets.hitIndices()[quadrupletIndex][5] = triplets.hitIndices()[innerTripletIndex][5];
    quadruplets.hitIndices()[quadrupletIndex][6] = triplets.hitIndices()[outerTripletIndex][4];
    quadruplets.hitIndices()[quadrupletIndex][7] = triplets.hitIndices()[outerTripletIndex][5];

    quadruplets.rzChiSquared()[quadrupletIndex] = rzChiSquared;
    quadruplets.dBeta()[quadrupletIndex] = dBeta;
    quadruplets.promptscore_t4dnn()[quadrupletIndex] = promptScore;
    quadruplets.displacedscore_t4dnn()[quadrupletIndex] = displacedScore;
    quadruplets.fakescore_t4dnn()[quadrupletIndex] = fakeScore;

    quadruplets.regressionRadius()[quadrupletIndex] = regressionRadius;
    quadruplets.nonAnchorRegressionRadius()[quadrupletIndex] = nonAnchorRegressionRadius;
    quadruplets.regressionG()[quadrupletIndex] = regressionG;
    quadruplets.regressionF()[quadrupletIndex] = regressionF;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passT4RZConstraint(TAcc const& acc,
                                                         ModulesConst modules,
                                                         MiniDoubletsConst mds,
                                                         unsigned int firstMDIndex,
                                                         unsigned int secondMDIndex,
                                                         unsigned int thirdMDIndex,
                                                         unsigned int fourthMDIndex,
                                                         uint16_t lowerModuleIndex1,
                                                         uint16_t lowerModuleIndex2,
                                                         uint16_t lowerModuleIndex3,
                                                         uint16_t lowerModuleIndex4,
                                                         float& rzChiSquared,
                                                         float inner_pt,
                                                         float innerRadius,
                                                         float g,
                                                         float f,
                                                         int charge) {
    //(g,f) is the center of the circle fitted by the innermost 3 points on x,y coordinates
    const float rt1 = mds.anchorRt()[firstMDIndex] / 100;  //in the unit of m instead of cm
    const float rt2 = mds.anchorRt()[secondMDIndex] / 100;
    const float rt3 = mds.anchorRt()[thirdMDIndex] / 100;
    const float rt4 = mds.anchorRt()[fourthMDIndex] / 100;

    const float z1 = mds.anchorZ()[firstMDIndex] / 100;
    const float z2 = mds.anchorZ()[secondMDIndex] / 100;
    const float z3 = mds.anchorZ()[thirdMDIndex] / 100;
    const float z4 = mds.anchorZ()[fourthMDIndex] / 100;

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer2 = modules.lstLayers()[lowerModuleIndex2];
    const int layer3 = modules.lstLayers()[lowerModuleIndex3];
    const int layer4 = modules.lstLayers()[lowerModuleIndex4];

    //slope computed using the internal T3s
    const int moduleType1 = modules.moduleType()[lowerModuleIndex1];  //0 is ps, 1 is 2s
    const int moduleType2 = modules.moduleType()[lowerModuleIndex2];
    const int moduleType3 = modules.moduleType()[lowerModuleIndex3];
    const int moduleType4 = modules.moduleType()[lowerModuleIndex4];

    const float x1 = mds.anchorX()[firstMDIndex] / 100;
    const float x2 = mds.anchorX()[secondMDIndex] / 100;
    const float x3 = mds.anchorX()[thirdMDIndex] / 100;
    const float x4 = mds.anchorX()[fourthMDIndex] / 100;
    const float y1 = mds.anchorY()[firstMDIndex] / 100;
    const float y2 = mds.anchorY()[secondMDIndex] / 100;
    const float y3 = mds.anchorY()[thirdMDIndex] / 100;
    const float y4 = mds.anchorY()[fourthMDIndex] / 100;

    float residual = 0;
    float error2 = 0;
    float x_center = g / 100, y_center = f / 100;
    float x_init = mds.anchorX()[thirdMDIndex] / 100;
    float y_init = mds.anchorY()[thirdMDIndex] / 100;
    float z_init = mds.anchorZ()[thirdMDIndex] / 100;
    float rt_init = mds.anchorRt()[thirdMDIndex] / 100;  //use the third MD as initial point

    if (moduleType3 == 1)  // 1: if MD3 is in 2s layer
    {
      x_init = mds.anchorX()[secondMDIndex] / 100;
      y_init = mds.anchorY()[secondMDIndex] / 100;
      z_init = mds.anchorZ()[secondMDIndex] / 100;
      rt_init = mds.anchorRt()[secondMDIndex] / 100;
    }

    //charge is determined in T3 and requiring both T3s to have the same charge

    float pseudo_phi = alpaka::math::atan(
        acc, (y_init - y_center) / (x_init - x_center));  //actually represent pi/2-phi, wrt helix axis z
    float Pt = inner_pt, Px = Pt * alpaka::math::abs(acc, alpaka::math::sin(acc, pseudo_phi)),
          Py = Pt * alpaka::math::abs(acc, cos(pseudo_phi));

    // Above line only gives you the correct value of Px and Py, but signs of Px and Py calculated below.
    // We look at if the circle is clockwise or anti-clock wise, to make it simpler, we separate the x-y plane into 4 quarters.
    if (x_init > x_center && y_init > y_center)  //1st quad
    {
      if (charge == 1)
        Py = -Py;
      if (charge == -1)
        Px = -Px;
    }
    if (x_init < x_center && y_init > y_center)  //2nd quad
    {
      if (charge == -1) {
        Px = -Px;
        Py = -Py;
      }
    }
    if (x_init < x_center && y_init < y_center)  //3rd quad
    {
      if (charge == 1)
        Px = -Px;
      if (charge == -1)
        Py = -Py;
    }
    if (x_init > x_center && y_init < y_center)  //4th quad
    {
      if (charge == 1) {
        Px = -Px;
        Py = -Py;
      }
    }

    // But if the initial T4 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of Px,Py signs on these to avoid errors
    if (moduleType3 == 0) {  // 0 is ps
      if (x4 < x3 && x3 < x2)
        Px = -alpaka::math::abs(acc, Px);
      else if (x4 > x3 && x3 > x2)
        Px = alpaka::math::abs(acc, Px);
      if (y4 < y3 && y3 < y2)
        Py = -alpaka::math::abs(acc, Py);
      else if (y4 > y3 && y3 > y2)
        Py = alpaka::math::abs(acc, Py);
    } else if (moduleType3 == 1)  // 1 is 2s
    {
      if (x3 < x2 && x2 < x1)
        Px = -alpaka::math::abs(acc, Px);
      else if (x3 > x2 && x2 > x1)
        Px = alpaka::math::abs(acc, Px);
      if (y3 < y2 && y2 < y1)
        Py = -alpaka::math::abs(acc, Py);
      else if (y3 > y2 && y2 > y1)
        Py = alpaka::math::abs(acc, Py);
    }

    //to get Pz, we use pt/pz=ds/dz, ds is the arclength between MD1 and MD3.
    float AO = alpaka::math::sqrt(acc, (x1 - x_center) * (x1 - x_center) + (y1 - y_center) * (y1 - y_center));
    float BO =
        alpaka::math::sqrt(acc, (x_init - x_center) * (x_init - x_center) + (y_init - y_center) * (y_init - y_center));
    float AB2 = (x1 - x_init) * (x1 - x_init) + (y1 - y_init) * (y1 - y_init);
    float dPhi = alpaka::math::acos(acc, (AO * AO + BO * BO - AB2) / (2 * AO * BO));
    float ds = innerRadius / 100 * dPhi;

    float Pz = (z_init - z1) / ds * Pt;
    float p = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);

    float a = -2.f * k2Rinv1GeVf * 100 * charge;  // multiply by 100 to make the correct length units

    float zsi, rtsi;
    int layeri, moduleTypei;
    rzChiSquared = 0;
    for (size_t i = 2; i < 5; i++) {
      if (i == 2) {
        zsi = z2;
        rtsi = rt2;
        layeri = layer2;
        moduleTypei = moduleType2;
      } else if (i == 3) {
        zsi = z3;
        rtsi = rt3;
        layeri = layer3;
        moduleTypei = moduleType3;
      } else if (i == 4) {
        zsi = z4;
        rtsi = rt4;
        layeri = layer4;
        moduleTypei = moduleType4;
      }

      if (moduleType3 == 0) {  //0: ps
        if (i == 3)
          continue;
      } else {
        if (i == 2)
          continue;
      }

      // calculation is copied from PixelTriplet.cc computePT3RZChiSquared
      float diffr = 0, diffz = 0;

      float rou = a / p;
      // for endcap
      float s = (zsi - z_init) * p / Pz;
      float x = x_init + Px / a * alpaka::math::sin(acc, rou * s) - Py / a * (1 - alpaka::math::cos(acc, rou * s));
      float y = y_init + Py / a * alpaka::math::sin(acc, rou * s) + Px / a * (1 - alpaka::math::cos(acc, rou * s));
      diffr = (rtsi - alpaka::math::sqrt(acc, x * x + y * y)) * 100;

      // for barrel
      if (layeri <= 6) {
        float paraA =
            rt_init * rt_init + 2 * (Px * Px + Py * Py) / (a * a) + 2 * (y_init * Px - x_init * Py) / a - rtsi * rtsi;
        float paraB = 2 * (x_init * Px + y_init * Py) / a;
        float paraC = 2 * (y_init * Px - x_init * Py) / a + 2 * (Px * Px + Py * Py) / (a * a);
        float A = paraB * paraB + paraC * paraC;
        float B = 2 * paraA * paraB;
        float C = paraA * paraA - paraC * paraC;
        float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
        float solz1 = alpaka::math::asin(acc, sol1) / rou * Pz / p + z_init;
        float solz2 = alpaka::math::asin(acc, sol2) / rou * Pz / p + z_init;
        float diffz1 = (solz1 - zsi) * 100;
        float diffz2 = (solz2 - zsi) * 100;
        // Alpaka : Needs to be moved over
        if (alpaka::math::isnan(acc, diffz1))
          diffz = diffz2;
        else if (alpaka::math::isnan(acc, diffz2))
          diffz = diffz1;
        else {
          diffz = (alpaka::math::abs(acc, diffz1) < alpaka::math::abs(acc, diffz2)) ? diffz1 : diffz2;
        }
      }
      residual = (layeri > 6) ? diffr : diffz;

      //PS Modules
      if (moduleTypei == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

      //check the tilted module, side: PosZ, NegZ, Center(for not tilted)
      float drdz;
      short side, subdets;
      if (i == 2) {
        drdz = alpaka::math::abs(acc, modules.drdzs()[lowerModuleIndex2]);
        side = modules.sides()[lowerModuleIndex2];
        subdets = modules.subdets()[lowerModuleIndex2];
      }
      if (i == 3) {
        drdz = alpaka::math::abs(acc, modules.drdzs()[lowerModuleIndex3]);
        side = modules.sides()[lowerModuleIndex3];
        subdets = modules.subdets()[lowerModuleIndex3];
      }
      if (i == 2 || i == 3) {
        residual = (layeri <= 6 && ((side == Center) or (drdz < 1))) ? diffz : diffr;
        float projection_missing2 = 1.f;
        if (drdz < 1)
          projection_missing2 =
              ((subdets == Endcap) or (side == Center)) ? 1.f : 1.f / (1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
        if (drdz > 1)
          projection_missing2 = ((subdets == Endcap) or (side == Center))
                                    ? 1.f
                                    : (drdz * drdz) / (1 + drdz * drdz);  //sin(atan(drdz)), if dr/dz>1
        error2 = error2 * projection_missing2;
      }
      rzChiSquared += 12 * (residual * residual) / error2;
    }
    // for set rzchi2 cut
    // if the 4 points are linear, helix calculation gives nan
    if (inner_pt > 100 || edm::isNotFinite(rzChiSquared)) {
      float slope;
      if (moduleType1 == 0 and moduleType2 == 0 and moduleType3 == 1)  //PSPS2S
      {
        slope = (z2 - z1) / (rt2 - rt1);
      } else {
        slope = (z3 - z1) / (rt3 - rt1);
      }
      float residual4_linear = (layer4 <= 6) ? ((z4 - z1) - slope * (rt4 - rt1)) : ((rt4 - rt1) - (z4 - z1) / slope);

      // creating a chi squared type quantity
      // 0-> PS, 1->2S
      residual4_linear = (moduleType4 == 0) ? residual4_linear / kPixelPSZpitch : residual4_linear / kStrip2SZpitch;
      residual4_linear = residual4_linear * 100;

      rzChiSquared = 12 * (residual4_linear * residual4_linear);
      return rzChiSquared < 5.839f;
    }
    float eta1 = alpaka::math::abs(acc, mds.anchorEta()[firstMDIndex]);
    uint8_t bin_index = (eta1 > 2.5f) ? (25 - 1) : static_cast<unsigned int>(eta1 / 0.1f);
    float chi2_cuts[] = {31.5082, 24.5654, 28.9223, 35.5906, 32.0746, 22.6416, 39.1476, 41.0791,
                         30.2745, 40.2882, 31.2135, 17.8911, 9.0297,  7.6862,  2.7591,  5.0587,
                         6.4014,  3.7348,  4.4768,  5.3087,  15.4535, 14.1107, 23.2778, 18.3643};

    return rzChiSquared < chi2_cuts[bin_index];
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletDefaultAlgo(TAcc const& acc,
                                                               ModulesConst modules,
                                                               MiniDoubletsConst mds,
                                                               SegmentsConst segments,
                                                               TripletsConst triplets,
                                                               uint16_t lowerModuleIndex1,
                                                               uint16_t lowerModuleIndex2,
                                                               uint16_t lowerModuleIndex3,
                                                               uint16_t lowerModuleIndex4,
                                                               unsigned int innerTripletIndex,
                                                               unsigned int outerTripletIndex,
                                                               float& regressionG,
                                                               float& regressionF,
                                                               float& regressionRadius,
                                                               float& nonAnchorRegressionRadius,
                                                               float& chiSquared,
                                                               const float ptCut,
                                                               float& rzChiSquared,
                                                               float& nonAnchorChiSquared,
                                                               float& dBeta,
                                                               float& promptScore,
                                                               float& displacedScore,
                                                               float& fakeScore) {
    unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int thirdSegmentIndex =
        triplets.segmentIndices()[outerTripletIndex][0];  //second and third segments are the same here
    unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];

    // require both T3s to have the same charge
    int innerT3charge = triplets.charge()[innerTripletIndex];
    int outerT3charge = triplets.charge()[outerTripletIndex];
    if (innerT3charge != outerT3charge)
      return false;

    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];

    float x1 = mds.anchorX()[firstMDIndex];
    float x2 = mds.anchorX()[secondMDIndex];
    float x3 = mds.anchorX()[thirdMDIndex];
    float x4 = mds.anchorX()[fourthMDIndex];

    float y1 = mds.anchorY()[firstMDIndex];
    float y2 = mds.anchorY()[secondMDIndex];
    float y3 = mds.anchorY()[thirdMDIndex];
    float y4 = mds.anchorY()[fourthMDIndex];

    float inner_circleCenterX = triplets.centerX()[innerTripletIndex];
    float inner_circleCenterY = triplets.centerY()[innerTripletIndex];
    float innerRadius = triplets.radius()[innerTripletIndex];
    float outerRadius = triplets.radius()[outerTripletIndex];
    float inner_pt = 2 * k2Rinv1GeVf * innerRadius;
    float pt = (innerRadius + outerRadius) * k2Rinv1GeVf;

    // 4 categories for sigmas
    float sigmas2[4], delta1[4], delta2[4], slopes[4];
    bool isFlat[4];

    float xVec[] = {x1, x2, x3, x4};
    float yVec[] = {y1, y2, y3, y4};

    const uint16_t lowerModuleIndices[] = {lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4};

    computeSigmasForRegression(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat, Params_T4::kLayers);
    regressionRadius = computeRadiusUsingRegression(acc,
                                                    Params_T4::kLayers,
                                                    xVec,
                                                    yVec,
                                                    delta1,
                                                    delta2,
                                                    slopes,
                                                    isFlat,
                                                    regressionG,
                                                    regressionF,
                                                    sigmas2,
                                                    chiSquared);

    //compute the other chisquared
    //non anchor is always shifted for tilted and endcap!
    float nonAnchorSigmas2[4], nonAnchorDelta1[Params_T4::kLayers], nonAnchorDelta2[Params_T4::kLayers],
        nonAnchorSlopes[Params_T4::kLayers];
    float nonAnchorxs[] = {mds.outerX()[firstMDIndex],
                           mds.outerX()[secondMDIndex],
                           mds.outerX()[thirdMDIndex],
                           mds.outerX()[fourthMDIndex]};
    float nonAnchorys[] = {mds.outerY()[firstMDIndex],
                           mds.outerY()[secondMDIndex],
                           mds.outerY()[thirdMDIndex],
                           mds.outerY()[fourthMDIndex]};

    bool nonAnchorisFlat[4];
    float nonAnchorRegressionG, nonAnchorRegressionF;

    computeSigmasForRegression(acc,
                               modules,
                               lowerModuleIndices,
                               nonAnchorDelta1,
                               nonAnchorDelta2,
                               nonAnchorSlopes,
                               nonAnchorisFlat,
                               Params_T4::kLayers,
                               false);

    nonAnchorRegressionRadius = computeRadiusUsingRegression(acc,
                                                             Params_T4::kLayers,
                                                             nonAnchorxs,
                                                             nonAnchorys,
                                                             nonAnchorDelta1,
                                                             nonAnchorDelta2,
                                                             nonAnchorSlopes,
                                                             nonAnchorisFlat,
                                                             nonAnchorRegressionG,
                                                             nonAnchorRegressionF,
                                                             nonAnchorSigmas2,
                                                             nonAnchorChiSquared);

    bool inference = lst::t4dnn::runInference(acc,
                                              mds,
                                              modules,
                                              firstMDIndex,
                                              secondMDIndex,
                                              thirdMDIndex,
                                              fourthMDIndex,
                                              lowerModuleIndex1,
                                              lowerModuleIndex2,
                                              lowerModuleIndex3,
                                              lowerModuleIndex4,
                                              innerRadius,
                                              outerRadius,
                                              promptScore,
                                              displacedScore,
                                              fakeScore,
                                              regressionRadius,
                                              nonAnchorRegressionRadius,
                                              triplets.fakeScore()[innerTripletIndex],
                                              triplets.promptScore()[innerTripletIndex],
                                              triplets.displacedScore()[innerTripletIndex],
                                              triplets.fakeScore()[outerTripletIndex],
                                              triplets.promptScore()[outerTripletIndex],
                                              triplets.displacedScore()[outerTripletIndex]);

    if (!inference) {
      return false;
    }

    //run Beta Selector for high pT T4s
    if (pt > 10) {
      if (not runQuintupletdBetaAlgoSelector(acc,
                                             modules,
                                             mds,
                                             segments,
                                             lowerModuleIndex1,
                                             lowerModuleIndex2,
                                             lowerModuleIndex3,
                                             lowerModuleIndex4,
                                             firstSegmentIndex,
                                             thirdSegmentIndex,
                                             firstMDIndex,
                                             secondMDIndex,
                                             thirdMDIndex,
                                             fourthMDIndex,
                                             dBeta,
                                             ptCut))
        return false;
    }

    if (not passT4RZConstraint(acc,
                               modules,
                               mds,
                               firstMDIndex,
                               secondMDIndex,
                               thirdMDIndex,
                               fourthMDIndex,
                               lowerModuleIndex1,
                               lowerModuleIndex2,
                               lowerModuleIndex3,
                               lowerModuleIndex4,
                               rzChiSquared,
                               inner_pt,
                               innerRadius,
                               inner_circleCenterX,
                               inner_circleCenterY,
                               innerT3charge))
      return false;

    float dxy = abs(std::hypot(regressionG, regressionF) - regressionRadius);
    float eta_layer3;
    const int layer1 = modules.layers()[lowerModuleIndex1];
    if (layer1 == 3) {
      eta_layer3 = alpaka::math::abs(acc, mds.anchorEta()[firstMDIndex]);
    } else if (layer1 == 2) {
      eta_layer3 = alpaka::math::abs(acc, mds.anchorEta()[secondMDIndex]);
    } else {
      eta_layer3 = alpaka::math::abs(acc, mds.anchorEta()[thirdMDIndex]);
    }
    if (dxy < 0.05f && eta_layer3 < 0.5f)
      return false;
    else if (dxy < 0.01f && eta_layer3 < 1.5f)
      return false;

    nonAnchorChiSquared = computeChiSquared(acc,
                                            Params_T4::kLayers,
                                            nonAnchorxs,
                                            nonAnchorys,
                                            nonAnchorDelta1,
                                            nonAnchorDelta2,
                                            nonAnchorSlopes,
                                            isFlat,
                                            regressionG,
                                            regressionF,
                                            regressionRadius);

    return true;
  };

  struct CreateQuadruplets {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  Triplets triplets,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  Quadruplets quadruplets,
                                  QuadrupletsOccupancy quadrupletsOccupancy,
                                  ObjectRangesConst ranges,
                                  uint16_t nEligibleT4Modules,
                                  const float ptCut) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));

      int& matchCount = alpaka::declareSharedVar<int, __COUNTER__>(acc);

      const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
      const auto blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

      const int threadIdX = threadIdx.x();
      const int threadIdY = threadIdx.y();
      const int blockSizeX = blockDim.x();
      const int blockSizeY = blockDim.y();
      const int blockSize = blockSizeX * blockSizeY;
      const int flatThreadIdxXY = threadIdY * blockSizeX + threadIdX;
      const int flatThreadExtent = blockSize;  // total threads per block

      for (int iter : cms::alpakatools::uniform_groups_z(acc, nEligibleT4Modules)) {
        uint16_t lowerModule1 = ranges.indicesOfEligibleT4Modules()[iter];

        if (cms::alpakatools::once_per_block(acc)) {
          matchCount = 0;
        }

        short layer2_adjustment;
        short md_adjustment;
        int layer = modules.layers()[lowerModule1];
        if (layer == 1) {
          if (modules.subdets()[lowerModule1] != Endcap)
            continue;
          layer2_adjustment = 1;
          md_adjustment = 1;
        }  // get upper segment to be in third layer
        else if (layer == 2) {
          if (modules.subdets()[lowerModule1] != Endcap)
            continue;
          layer2_adjustment = 1;
          md_adjustment = 0;
        }  // get lower segment to be in third layer
        else {
          layer2_adjustment = 0;
          md_adjustment = 0;
        }
        unsigned int nInnerTriplets = tripletsOccupancy.nTriplets()[lowerModule1];

        alpaka::syncBlockThreads(acc);

        // Step 1: Make inner and outer triplet pairs
        for (unsigned int innerTripletArrayIndex : cms::alpakatools::uniform_elements_y(acc, nInnerTriplets)) {
          unsigned int innerTripletIndex = ranges.tripletModuleIndices()[lowerModule1] + innerTripletArrayIndex;
          if (triplets.partOfPT5()[innerTripletIndex])
            continue;  //don't create T4s for T3s accounted in pT5s
          if (triplets.partOfT5()[innerTripletIndex])
            continue;  //don't create T4s for T3s accounted in T5s
          if (triplets.partOfPT3()[innerTripletIndex])
            continue;  //don't create T4s for T3s accounted in pT3s
          uint16_t lowerModule2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
          unsigned int nOuterTriplets = tripletsOccupancy.nTriplets()[lowerModule2];
          for (unsigned int outerTripletArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterTriplets)) {
            unsigned int outerTripletIndex = ranges.tripletModuleIndices()[lowerModule2] + outerTripletArrayIndex;
            if (triplets.partOfPT5()[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in pT5s
            if (triplets.partOfT5()[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in T5s
            if (triplets.partOfPT3()[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in pT3s

            unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
            unsigned int thirdSegmentIndex =
                triplets.segmentIndices()[outerTripletIndex][0];  //second and third segments are the same here
            unsigned int innerOuterInnerMiniDoubletIndex =
                segments.mdIndices()[secondSegmentIndex][0];  //inner triplet outer segment inner MD index
            unsigned int innerOuterOuterMiniDoubletIndex =
                segments.mdIndices()[secondSegmentIndex][1];  //inner triplet outer segment outer MD index
            unsigned int outerInnerInnerMiniDoubletIndex =
                segments.mdIndices()[thirdSegmentIndex][0];  //outer triplet inner segment inner MD index
            unsigned int outerOuterInnerMiniDoubletIndex =
                segments.mdIndices()[thirdSegmentIndex][1];  //outer triplet outer segment inner MD index

            //check if the 2 T3s have a common LS
            if (innerOuterInnerMiniDoubletIndex != outerInnerInnerMiniDoubletIndex)
              continue;
            if (innerOuterOuterMiniDoubletIndex != outerOuterInnerMiniDoubletIndex)
              continue;

            int mIdx = alpaka::atomicAdd(acc, &matchCount, 1, alpaka::hierarchy::Threads{});
            unsigned int quadrupletIndex = ranges.quadrupletModuleIndices()[lowerModule1] + mIdx;

#ifdef WARNINGS
            const unsigned int rightBound =
                static_cast<unsigned int>(ranges.quadrupletModuleIndices()[lowerModule1 + 1]);
            if (quadrupletIndex >= rightBound) {
              printf(
                  "Quadruplet module occupancy alert! module quadruplet starting index  = %d, Pair quadruplet index = "
                  "%d, next module quadruplet starting index = %d\n",
                  ranges.quadrupletModuleIndices()[lowerModule1],
                  mIdx,
                  ranges.quadrupletModuleIndices()[lowerModule1 + 1]);
            }
#endif

            quadruplets.preAllocatedTripletIndices()[quadrupletIndex][0] = innerTripletIndex;
            quadruplets.preAllocatedTripletIndices()[quadrupletIndex][1] = outerTripletIndex;
          }
        }

        alpaka::syncBlockThreads(acc);
        if (matchCount == 0) {
          continue;
        }

        // Step 2: Parallel processing of triplet pairs
        for (int i = flatThreadIdxXY; i < matchCount; i += flatThreadExtent) {
          unsigned int quadrupletIndex = ranges.quadrupletModuleIndices()[lowerModule1] + i;
          int innerTripletIndex = quadruplets.preAllocatedTripletIndices()[quadrupletIndex][0];
          int outerTripletIndex = quadruplets.preAllocatedTripletIndices()[quadrupletIndex][1];

          uint16_t lowerModule2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
          uint16_t lowerModule3 = triplets.lowerModuleIndices()[outerTripletIndex][1];
          uint16_t lowerModule4 = triplets.lowerModuleIndices()[outerTripletIndex][2];

          float innerRadius = triplets.radius()[innerTripletIndex];
          float outerRadius = triplets.radius()[outerTripletIndex];
          float rzChiSquared, dBeta, nonAnchorChiSquared, regressionG, regressionF, regressionRadius,
              nonAnchorRegressionRadius, chiSquared, promptScore, displacedScore, fakeScore;

          float pt = (innerRadius + outerRadius) * k2Rinv1GeVf;

          bool success = runQuadrupletDefaultAlgo(acc,
                                                  modules,
                                                  mds,
                                                  segments,
                                                  triplets,
                                                  lowerModule1,
                                                  lowerModule2,
                                                  lowerModule3,
                                                  lowerModule4,
                                                  innerTripletIndex,
                                                  outerTripletIndex,
                                                  regressionG,
                                                  regressionF,
                                                  regressionRadius,
                                                  nonAnchorRegressionRadius,
                                                  chiSquared,
                                                  ptCut,
                                                  rzChiSquared,
                                                  nonAnchorChiSquared,
                                                  dBeta,
                                                  promptScore,
                                                  displacedScore,
                                                  fakeScore);
          if (success) {
            int totOccupancyQuadruplets = alpaka::atomicAdd(
                acc, &quadrupletsOccupancy.totOccupancyQuadruplets()[lowerModule1], 1u, alpaka::hierarchy::Threads{});
            if (totOccupancyQuadruplets >= ranges.quadrupletModuleOccupancy()[lowerModule1]) {
#ifdef WARNINGS
              printf("Quadruplet excess alert! Module index = %d, Occupancy = %d\n",
                     lowerModule1,
                     totOccupancyQuadruplets);
#endif
            } else {
              int quadrupletModuleIndex = alpaka::atomicAdd(
                  acc, &quadrupletsOccupancy.nQuadruplets()[lowerModule1], 1u, alpaka::hierarchy::Threads{});
              //this if statement should never get executed!
              if (ranges.quadrupletModuleIndices()[lowerModule1] == -1) {
#ifdef WARNINGS
                printf("Quadruplets : no memory for module at module index = %d\n", lowerModule1);
#endif
              } else {
                unsigned int quadrupletIndex = ranges.quadrupletModuleIndices()[lowerModule1] + quadrupletModuleIndex;
                float phi =
                    mds.anchorPhi()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][md_adjustment]]
                                                        [layer2_adjustment]];  //layer 3
                float eta =
                    mds.anchorEta()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][md_adjustment]]
                                                        [layer2_adjustment]];  //layer 3

                float scores = chiSquared + nonAnchorChiSquared;
                addQuadrupletToMemory(triplets,
                                      quadruplets,
                                      innerTripletIndex,
                                      outerTripletIndex,
                                      lowerModule1,
                                      lowerModule2,
                                      lowerModule3,
                                      lowerModule4,
                                      innerRadius,
                                      outerRadius,
                                      pt,
                                      eta,
                                      phi,
                                      scores,
                                      layer,
                                      quadrupletIndex,
                                      rzChiSquared,
                                      dBeta,
                                      promptScore,
                                      displacedScore,
                                      fakeScore,
                                      regressionG,
                                      regressionF,
                                      regressionRadius,
                                      nonAnchorRegressionRadius);
              }
            }
          }
        }
      }
    }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isValidQuadRegion(ModulesConst modules, uint16_t lowerModule) {
    const short layer = modules.layers()[lowerModule];
    const short subdet = modules.subdets()[lowerModule];
    // Quadruplets starting outside these regions are not built.
    return (subdet == Barrel && layer > 2) || (subdet == Endcap);
  }

  struct CountTripletLSConnections {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  SegmentsConst segments,
                                  Triplets triplets,
                                  TripletsOccupancyConst tripletsOcc,
                                  ObjectRangesConst ranges) const {
      // The atomicAdd below with hierarchy::Threads{} requires one block in x, y dimensions.
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));
      const auto& mdIndices = segments.mdIndices();
      const auto& segIdx = triplets.segmentIndices();
      const auto& lmIdx = triplets.lowerModuleIndices();
      const auto& tripIdx = ranges.tripletModuleIndices();

      for (uint16_t lowerModule1 : cms::alpakatools::uniform_groups_z(acc, modules.nLowerModules())) {
        if (!isValidQuadRegion(modules, lowerModule1))
          continue;

        const unsigned int nInnerTriplets = tripletsOcc.nTriplets()[lowerModule1];
        if (nInnerTriplets == 0)
          continue;

        for (unsigned int innerTripletArrayIndex : cms::alpakatools::uniform_elements_y(acc, nInnerTriplets)) {
          const unsigned int innerTripletIndex = tripIdx[lowerModule1] + innerTripletArrayIndex;

          const uint16_t lowerModule2 = lmIdx[innerTripletIndex][1];
          const unsigned int nOuterTriplets = tripletsOcc.nTriplets()[lowerModule2];
          if (nOuterTriplets == 0)
            continue;

          const unsigned int secondSegIdx = segIdx[innerTripletIndex][1];
          const unsigned int secondMDInner = mdIndices[secondSegIdx][0];
          const unsigned int secondMDOuter = mdIndices[secondSegIdx][1];

          for (unsigned int outerTripletArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterTriplets)) {
            const unsigned int outerTripletIndex = tripIdx[lowerModule2] + outerTripletArrayIndex;
            const unsigned int thirdSegIdx = segIdx[outerTripletIndex][0];
            const unsigned int thirdMDInner = mdIndices[thirdSegIdx][0];
            const unsigned int thirdMDOuter = mdIndices[thirdSegIdx][1];

            if ((secondMDInner == thirdMDInner) && (secondMDOuter == thirdMDOuter)) {
              alpaka::atomicAdd(acc, &triplets.connectedLSMax()[innerTripletIndex], 1u, alpaka::hierarchy::Threads{});
            }
          }
        }
      }
    }
  };

  struct CreateEligibleModulesListForQuadruplets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  TripletsOccupancyConst tripletsOcc,
                                  ObjectRanges ranges,
                                  Triplets triplets) const {
      // Single-block kernel
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      int& nEligibleT4Modulesx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nTotalQuadrupletsx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalQuadrupletsx = 0;
        nEligibleT4Modulesx = 0;
      }
      alpaka::syncBlockThreads(acc);

      for (uint16_t lowerModule : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (!isValidQuadRegion(modules, lowerModule))
          continue;

        unsigned int nInnerTriplets = tripletsOcc.nTriplets()[lowerModule];
        if (nInnerTriplets == 0)
          continue;

        // Sum the real connectivity for triplets in this module
        int dynamic_count = 0;
        const unsigned int firstTripletIdx = ranges.tripletModuleIndices()[lowerModule];
        for (unsigned int t = 0; t < nInnerTriplets; ++t) {
          unsigned int tripletIndex = firstTripletIdx + t;
          dynamic_count += triplets.connectedLSMax()[tripletIndex];
        }

        if (dynamic_count == 0)
          continue;

        int nEligibleT4Modules = alpaka::atomicAdd(acc, &nEligibleT4Modulesx, 1, alpaka::hierarchy::Threads{});
        int nTotQ = alpaka::atomicAdd(acc, &nTotalQuadrupletsx, dynamic_count, alpaka::hierarchy::Threads{});

        ranges.quadrupletModuleIndices()[lowerModule] = nTotQ;
        ranges.indicesOfEligibleT4Modules()[nEligibleT4Modules] = lowerModule;
        ranges.quadrupletModuleOccupancy()[lowerModule] = dynamic_count;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.nEligibleT4Modules() = static_cast<uint16_t>(nEligibleT4Modulesx);
        ranges.nTotalQuads() = static_cast<unsigned int>(nTotalQuadrupletsx);
      }
    }
  };

  struct AddQuadrupletRangesToEventExplicit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  ObjectRanges ranges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      for (uint16_t i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (quadrupletsOccupancy.nQuadruplets()[i] == 0 or ranges.quadrupletModuleIndices()[i] == -1) {
          ranges.quadrupletRanges()[i][0] = -1;
          ranges.quadrupletRanges()[i][1] = -1;
        } else {
          ranges.quadrupletRanges()[i][0] = ranges.quadrupletModuleIndices()[i];
          ranges.quadrupletRanges()[i][1] =
              ranges.quadrupletModuleIndices()[i] + quadrupletsOccupancy.nQuadruplets()[i] - 1;
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
