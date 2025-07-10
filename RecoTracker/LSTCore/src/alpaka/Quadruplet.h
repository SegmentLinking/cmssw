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
                                                            float nonAnchorRegressionRadius,
                                                            bool tightDNNFlag,
                                                            bool tightCutFlag) {
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

    quadruplets.tightDNNFlag()[quadrupletIndex] = tightDNNFlag;
    quadruplets.tightCutFlag()[quadrupletIndex] = tightCutFlag;
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
                                                         int charge,
                                                         bool& tightCutFlag) {
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
    const int layer1 = modules.lstLayers()[lowerModuleIndex1];
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
          projection_missing2 = ((subdets == Endcap) or (side == Center))
                                    ? 1.f
                                    : 1.f / (1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
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
      if (rzChiSquared < 9.666f) //95% retention, add radii and t3 scores
        tightCutFlag = true;
      return rzChiSquared < 14.064f; //99% add reg radii and t3 scores to dnn
    }
    // The category numbers are related to module regions and layers, decoding of the region numbers can be found here in slide 2 table. https://github.com/SegmentLinking/TrackLooper/files/11420927/part.2.pdf
    // The commented numbers after each case is the region code, and can look it up from the table to see which category it belongs to. For example, //0 means T4 built with Endcap 1,2,3,4 ps modules
    // All 99% retention cuts. The tight cut flags are set at 95% displaced track retention
    if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10)  //0
    {
      if (rzChiSquared < 19.283f)
        tightCutFlag = true;
      return rzChiSquared < 28.459f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15)  //1
    {
      if (rzChiSquared < 6.298f)
        tightCutFlag = true;
      return rzChiSquared < 8.968f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 14 and layer4 == 15)  //2
    {
      if (rzChiSquared < 3.879f)
        tightCutFlag = true;
      return rzChiSquared < 5.158f;
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      if (layer4 == 11)  //3
      {
        if (rzChiSquared < 18.516f)
          tightCutFlag = true;
        return rzChiSquared < 29.270f;
      }
      if (layer4 == 16)  //4
      {
        if (rzChiSquared < 6.378f)
          tightCutFlag = true;
        return rzChiSquared < 9.310f;
      }
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 15 and layer4 == 16) //5
    { 
        if (rzChiSquared < 3.493f)
          tightCutFlag = true;
        return rzChiSquared < 4.328f;
    }
    else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 4)  //6
      {
        if (rzChiSquared < 13.252f)
          tightCutFlag = true;
        return rzChiSquared < 23.138f;
      }
      else if (layer4 == 7)  //7
      {
        if (rzChiSquared < 16.956f)
          tightCutFlag = true;
        return rzChiSquared < 29.561f;
      } else if (layer4 == 12)  //8
      {
        if (rzChiSquared < 10.924f)
          tightCutFlag = true;
        return rzChiSquared < 18.905f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 8)  //9
      {
        if (rzChiSquared < 18.263f)
          tightCutFlag = true;
        return rzChiSquared < 29.534f;
      } else if (layer4 == 13)  //10
      {
        if (rzChiSquared < 8.384f)
          tightCutFlag = true;
        return rzChiSquared < 12.608f;
      } 
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9) //11
      {
        if (rzChiSquared < 18.741f)
          tightCutFlag = true;
        return rzChiSquared < 28.270f;
      } else if (layer4 == 14)  //12
      {
        return true;
      } 
    } else if (layer1 == 2 and layer2 ==3) {
      if (layer3 == 4) {
        if (layer4 == 5)  //13
        {
          if (rzChiSquared < 4.376f)
            tightCutFlag = true;
          return rzChiSquared < 5.430f;
        }
        else if (layer4 == 12) //14
        { 
          if (rzChiSquared < 4.196f)
            tightCutFlag = true;
          return rzChiSquared < 5.176f;
        }
      }
      else if (layer3 == 7) {
        if (layer4 == 8) // 15
        { 
          if (rzChiSquared < 11.934f)
            tightCutFlag = true;
          return rzChiSquared < 20.491f;
        }
        else if (layer4 == 13) //16
        { 
          if (rzChiSquared < 9.518f)
            tightCutFlag = true;
          return rzChiSquared < 14.100f;
        }
      }
      else if (layer3 == 12 and layer4 == 13) //17
      { 
        if (rzChiSquared < 4.269f)
          tightCutFlag = true;
        return rzChiSquared < 5.499f;
      }
    } else if (layer1 == 2 and layer2 == 12 and layer3 == 13 and layer4 == 14) //18
    { 
      if (rzChiSquared < 22.481f)
        tightCutFlag = true;
      return rzChiSquared < 36.038f;
    } else if (layer1 == 2 and layer2 == 7)
    {
      if (layer3 == 8 and layer4 == 14) //19
      { 
        if (rzChiSquared < 7.06f)
          tightCutFlag = true;
        return rzChiSquared < 10.991f;
      }
      else if (layer3 == 13 and layer4 == 14) //20
      { 
        if (rzChiSquared < 3.343f)
          tightCutFlag = true;
        return rzChiSquared < 4.144f;
      }
    } else if (layer1 == 3)
    {
      if (layer2 == 4){
        if (layer3 == 5 and layer4 == 6 ) //21
        { 
          if (rzChiSquared < 56.716f)
            tightCutFlag = true;
          return rzChiSquared < 76.861f;
        }
        else if (layer3 == 12 and layer4 == 13) //24
        {
          if (rzChiSquared < 17.68f)
            tightCutFlag = true;
          return rzChiSquared < 27.034f;
        }
        else if (layer3 == 5 and layer4 == 12) //25
        {
          if (rzChiSquared < 36.004f)
            tightCutFlag = true;
          return rzChiSquared < 48.511f;
        }
      }
      else if (layer2 == 7) {
        if (layer3 == 8 and layer4 == 14) //22
        {
          if (rzChiSquared < 3.971f)
            tightCutFlag = true;
          return rzChiSquared < 6.017f;
        }
        else if (layer3 == 13 and layer4 == 14) //23
        {
          if (rzChiSquared < 3.571f)
            tightCutFlag = true;
          return rzChiSquared < 4.415f;
        }
      }
    }
    return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegressionT4(TAcc const& acc,
                                                                 ModulesConst modules,
                                                                 const uint16_t* lowerModuleIndices,
                                                                 float* delta1,
                                                                 float* delta2,
                                                                 float* slopes,
                                                                 bool* isFlat,
                                                                 unsigned int nPoints = 5,
                                                                 bool anchorHits = true) {
    /*
        Bool anchorHits required to deal with a weird edge case wherein 
        the hits ultimately used in the regression are anchor hits, but the
        lower modules need not all be Pixel Modules (in case of PS). Similarly,
        when we compute the chi squared for the non-anchor hits, the "partner module"
        need not always be a PS strip module, but all non-anchor hits sit on strip 
        modules.
        */

    ModuleType moduleType;
    short moduleSubdet, moduleSide;
    float inv1 = kWidthPS / kWidth2S;
    float inv2 = kPixelPSZpitch / kWidth2S;
    float inv3 = kStripPSZpitch / kWidth2S;
    for (size_t i = 0; i < nPoints; i++) {
      moduleType = modules.moduleType()[lowerModuleIndices[i]];
      moduleSubdet = modules.subdets()[lowerModuleIndices[i]];
      moduleSide = modules.sides()[lowerModuleIndices[i]];
      const float& drdz = modules.drdzs()[lowerModuleIndices[i]];
      slopes[i] = modules.dxdys()[lowerModuleIndices[i]];
      //category 1 - barrel PS flat
      if (moduleSubdet == Barrel and moduleType == PS and moduleSide == Center) {
        delta1[i] = inv1;
        delta2[i] = inv1;
        slopes[i] = -999.f;
        isFlat[i] = true;
      }
      //category 2 - barrel 2S
      else if (moduleSubdet == Barrel and moduleType == TwoS) {
        delta1[i] = 1.f;
        delta2[i] = 1.f;
        slopes[i] = -999.f;
        isFlat[i] = true;
      }
      //category 3 - barrel PS tilted
      else if (moduleSubdet == Barrel and moduleType == PS and moduleSide != Center) {
        delta1[i] = inv1;
        isFlat[i] = false;

        if (anchorHits) {
          delta2[i] = (inv2 * drdz / alpaka::math::sqrt(acc, 1 + drdz * drdz));
        } else {
          delta2[i] = (inv3 * drdz / alpaka::math::sqrt(acc, 1 + drdz * drdz));
        }
      }
      //category 4 - endcap PS
      else if (moduleSubdet == Endcap and moduleType == PS) {
        delta1[i] = inv1;
        isFlat[i] = false;

        /*
                despite the type of the module layer of the lower module index,
                all anchor hits are on the pixel side and all non-anchor hits are
                on the strip side!
                */
        if (anchorHits) {
          delta2[i] = inv2;
        } else {
          delta2[i] = inv3;
        }
      }
      //category 5 - endcap 2S
      else if (moduleSubdet == Endcap and moduleType == TwoS) {
        delta1[i] = 1.f;
        delta2[i] = 500.f * inv1;
        isFlat[i] = false;
      } else {
#ifdef WARNINGS
        printf("ERROR!!!!! I SHOULDN'T BE HERE!!!! subdet = %d, type = %d, side = %d\n",
               moduleSubdet,
               moduleType,
               moduleSide);
#endif
      }
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusUsingRegressionT4(TAcc const& acc,
                                                                    unsigned int nPoints,
                                                                    float* xs,
                                                                    float* ys,
                                                                    float* delta1,
                                                                    float* delta2,
                                                                    float* slopes,
                                                                    bool* isFlat,
                                                                    float& g,
                                                                    float& f,
                                                                    float* sigmas2,
                                                                    float& chiSquared) {
    float radius = 0.f;

    // Some extra variables
    // the two variables will be called x1 and x2, and y (which is x^2 + y^2)

    float sigmaX1Squared = 0.f;
    float sigmaX2Squared = 0.f;
    float sigmaX1X2 = 0.f;
    float sigmaX1y = 0.f;
    float sigmaX2y = 0.f;
    float sigmaY = 0.f;
    float sigmaX1 = 0.f;
    float sigmaX2 = 0.f;
    float sigmaOne = 0.f;

    float xPrime, yPrime, absArctanSlope, angleM;
    for (size_t i = 0; i < nPoints; i++) {
      // Computing sigmas is a very tricky affair
      // if the module is tilted or endcap, we need to use the slopes properly!

      absArctanSlope = ((slopes[i] != kVerticalModuleSlope && edm::isFinite(slopes[i]))
                            ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i]))
                            : kPi / 2.f);

      if (xs[i] > 0 and ys[i] > 0) {
        angleM = kPi / 2.f - absArctanSlope;
      } else if (xs[i] < 0 and ys[i] > 0) {
        angleM = absArctanSlope + kPi / 2.f;
      } else if (xs[i] < 0 and ys[i] < 0) {
        angleM = -(absArctanSlope + kPi / 2.f);
      } else if (xs[i] > 0 and ys[i] < 0) {
        angleM = -(kPi / 2.f - absArctanSlope);
      } else {
        angleM = 0;
      }

      if (not isFlat[i]) {
        xPrime = xs[i] * alpaka::math::cos(acc, angleM) + ys[i] * alpaka::math::sin(acc, angleM);
        yPrime = ys[i] * alpaka::math::cos(acc, angleM) - xs[i] * alpaka::math::sin(acc, angleM);
      } else {
        xPrime = xs[i];
        yPrime = ys[i];
      }
      sigmas2[i] = 4 * ((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));

      sigmaX1Squared += (xs[i] * xs[i]) / sigmas2[i];
      sigmaX2Squared += (ys[i] * ys[i]) / sigmas2[i];
      sigmaX1X2 += (xs[i] * ys[i]) / sigmas2[i];
      sigmaX1y += (xs[i] * (xs[i] * xs[i] + ys[i] * ys[i])) / sigmas2[i];
      sigmaX2y += (ys[i] * (xs[i] * xs[i] + ys[i] * ys[i])) / sigmas2[i];
      sigmaY += (xs[i] * xs[i] + ys[i] * ys[i]) / sigmas2[i];
      sigmaX1 += xs[i] / sigmas2[i];
      sigmaX2 += ys[i] / sigmas2[i];
      sigmaOne += 1.0f / sigmas2[i];
    }
    float denominator = (sigmaX1X2 - sigmaX1 * sigmaX2) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                        (sigmaX1Squared - sigmaX1 * sigmaX1) * (sigmaX2Squared - sigmaX2 * sigmaX2);

    float twoG = ((sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                  (sigmaX1y - sigmaX1 * sigmaY) * (sigmaX2Squared - sigmaX2 * sigmaX2)) /
                 denominator;
    float twoF = ((sigmaX1y - sigmaX1 * sigmaY) * (sigmaX1X2 - sigmaX1 * sigmaX2) -
                  (sigmaX2y - sigmaX2 * sigmaY) * (sigmaX1Squared - sigmaX1 * sigmaX1)) /
                 denominator;

    float c = -(sigmaY - twoG * sigmaX1 - twoF * sigmaX2) / sigmaOne;
    g = 0.5f * twoG;
    f = 0.5f * twoF;
    if (g * g + f * f - c < 0) {
#ifdef WARNINGS
      printf("FATAL! r^2 < 0!\n");
#endif
      chiSquared = -1;
      return -1;
    }

    radius = alpaka::math::sqrt(acc, g * g + f * f - c);
    // compute chi squared
    chiSquared = 0.f;
    for (size_t i = 0; i < nPoints; i++) {
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - twoG * xs[i] - twoF * ys[i] + c) / sigmas2[i];
    }
    return radius;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeT4ChiSquared(TAcc const& acc,
                                                         unsigned int nPoints,
                                                         float* xs,
                                                         float* ys,
                                                         float* delta1,
                                                         float* delta2,
                                                         float* slopes,
                                                         bool* isFlat,
                                                         float g,
                                                         float f,
                                                         float radius) {
    // given values of (g, f, radius) and a set of points (and its uncertainties)
    // compute chi squared
    float c = g * g + f * f - radius * radius;
    float chiSquared = 0.f;
    float absArctanSlope, angleM, xPrime, yPrime, sigma2;
    for (size_t i = 0; i < nPoints; i++) {
      absArctanSlope = ((slopes[i] != kVerticalModuleSlope && edm::isFinite(slopes[i]))
                            ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i]))
                            : kPi / 2.f);
      if (xs[i] > 0 and ys[i] > 0) {
        angleM = kPi / 2.f - absArctanSlope;
      } else if (xs[i] < 0 and ys[i] > 0) {
        angleM = absArctanSlope + kPi / 2.f;
      } else if (xs[i] < 0 and ys[i] < 0) {
        angleM = -(absArctanSlope + kPi / 2.f);
      } else if (xs[i] > 0 and ys[i] < 0) {
        angleM = -(kPi / 2.f - absArctanSlope);
      } else {
        angleM = 0;
      }

      if (not isFlat[i]) {
        xPrime = xs[i] * alpaka::math::cos(acc, angleM) + ys[i] * alpaka::math::sin(acc, angleM);
        yPrime = ys[i] * alpaka::math::cos(acc, angleM) - xs[i] * alpaka::math::sin(acc, angleM);
      } else {
        xPrime = xs[i];
        yPrime = ys[i];
      }
      sigma2 = 4 * ((xPrime * delta1[i]) * (xPrime * delta1[i]) + (yPrime * delta2[i]) * (yPrime * delta2[i]));
      chiSquared += (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) *
                    (xs[i] * xs[i] + ys[i] * ys[i] - 2 * g * xs[i] - 2 * f * ys[i] + c) / sigma2;
    }
    return chiSquared;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterationsT4(TAcc const& acc,
                                                               float& betaIn,
                                                               float& betaOut,
                                                               float betaAv,
                                                               float& pt_beta,
                                                               float sdIn_dr,
                                                               float sdOut_dr,
                                                               float dr,
                                                               float lIn) {
    if (lIn == 0) {
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc,
              alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaOut);
      return;
    }

    if (betaIn * betaOut > 0.f and
        (alpaka::math::abs(acc, pt_beta) < 4.f * kPt_betaMax or
         (lIn >= 11 and alpaka::math::abs(acc, pt_beta) <
                            8.f * kPt_betaMax)))  //and the pt_beta is well-defined; less strict for endcap-endcap
    {
      const float betaInUpd =
          betaIn + alpaka::math::copysign(
                       acc,
                       alpaka::math::asin(
                           acc,
                           alpaka::math::min(
                               acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
                       betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut + alpaka::math::copysign(
                        acc,
                        alpaka::math::asin(
                            acc,
                            alpaka::math::min(
                                acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
                        betaOut);  //FIXME: need a faster version
      betaAv = 0.5f * (betaInUpd + betaOutUpd);

      //1st update
      const float pt_beta_inv =
          1.f / alpaka::math::abs(acc, dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv));  //get a better pt estimate

      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf * pt_beta_inv, kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf * pt_beta_inv, kSinAlphaMax)),
          betaOut);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    } else if (lIn < 11 && alpaka::math::abs(acc, betaOut) < 0.2f * alpaka::math::abs(acc, betaIn) &&
               alpaka::math::abs(acc, pt_beta) < 12.f * kPt_betaMax)  //use betaIn sign as ref
    {
      const float pt_betaIn = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaIn);

      const float betaInUpd =
          betaIn + alpaka::math::copysign(
                       acc,
                       alpaka::math::asin(
                           acc,
                           alpaka::math::min(
                               acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), kSinAlphaMax)),
                       betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc,
                  alpaka::math::min(
                      acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn))
                   ? (0.5f * (betaInUpd + betaOutUpd))
                   : betaInUpd;

      //1st update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc,
              alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc,
              alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletdBetaCutBBBB(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex,
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float r3_InLo = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = rt_InOut - rt_InLo;

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (r3_InLo / rt_InLo);

    float midPointX = 0.5f * (mds.anchorX()[firstMDIndex] + mds.anchorX()[thirdMDIndex]);
    float midPointY = 0.5f * (mds.anchorY()[firstMDIndex] + mds.anchorY()[thirdMDIndex]);
    float diffX = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float diffY = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    float dPhi = cms::alpakatools::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions
    float alpha_InLo = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segments.dPhiChanges()[outerSegmentIndex]);

    bool isEC_lastLayer = modules.subdets()[outerOuterLowerModuleIndex] == Endcap and
                          modules.moduleType()[outerOuterLowerModuleIndex] == TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;

    alpha_OutUp = cms::alpakatools::reducePhiRange(
        acc,
        cms::alpakatools::phi(acc,
                              mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex],
                              mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) -
            mds.anchorPhi()[fourthMDIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = mds.anchorX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];
    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    float betaIn =
        alpha_InLo - cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut =
        -alpha_OutUp + cms::alpakatools::reducePhiRange(
                           acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if (isEC_lastLayer) {
      alpha_OutUp_highEdge = cms::alpakatools::reducePhiRange(
          acc,
          cms::alpakatools::phi(acc,
                                mds.anchorHighEdgeX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex],
                                mds.anchorHighEdgeY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) -
              mds.anchorHighEdgePhi()[fourthMDIndex]);
      alpha_OutUp_lowEdge = cms::alpakatools::reducePhiRange(
          acc,
          cms::alpakatools::phi(acc,
                                mds.anchorLowEdgeX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex],
                                mds.anchorLowEdgeY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) -
              mds.anchorLowEdgePhi()[fourthMDIndex]);

      tl_axis_highEdge_x = mds.anchorHighEdgeX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
      tl_axis_highEdge_y = mds.anchorHighEdgeY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];
      tl_axis_lowEdge_x = mds.anchorLowEdgeX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
      tl_axis_lowEdge_y = mds.anchorLowEdgeY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];

      betaOutRHmin = -alpha_OutUp_highEdge + cms::alpakatools::reducePhiRange(
                                                 acc,
                                                 cms::alpakatools::phi(acc, tl_axis_highEdge_x, tl_axis_highEdge_y) -
                                                     mds.anchorHighEdgePhi()[fourthMDIndex]);
      betaOutRHmax = -alpha_OutUp_lowEdge +
                     cms::alpakatools::reducePhiRange(acc,
                                                      cms::alpakatools::phi(acc, tl_axis_lowEdge_x, tl_axis_lowEdge_y) -
                                                          mds.anchorLowEdgePhi()[fourthMDIndex]);
    }

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg = alpaka::math::sqrt(acc,
                                              (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                                      (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                                  (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                                      (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = drt_tl_axis * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);
    int lIn = 5;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) *
                                                (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) +
                                            (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) *
                                                (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]));
    float sdOut_d = mds.anchorRt()[fourthMDIndex] - mds.anchorRt()[thirdMDIndex];

    runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);
    
    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.f;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.f;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confimm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg = alpaka::math::max(
        acc,
        alpaka::math::abs(acc, alpha_InLo),
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg = alpaka::math::max(
        acc,
        alpaka::math::abs(acc, alpha_OutLo),
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = alpaka::math::sin(acc, dPhi);

    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut = (alpaka::math::sqrt(acc,
                                      mds.anchorHighEdgeX()[fourthMDIndex] * mds.anchorHighEdgeX()[fourthMDIndex] +
                                          mds.anchorHighEdgeY()[fourthMDIndex] * mds.anchorHighEdgeY()[fourthMDIndex]) -
                   alpaka::math::sqrt(acc,
                                      mds.anchorLowEdgeX()[fourthMDIndex] * mds.anchorLowEdgeX()[fourthMDIndex] +
                                          mds.anchorLowEdgeY()[fourthMDIndex] * mds.anchorLowEdgeY()[fourthMDIndex])) *
                  sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaROut2 + 
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));

    dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletdBetaCutBBEE(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex,
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float rIn = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (rIn / rt_InLo);

    float midPointX = 0.5f * (mds.anchorX()[firstMDIndex] + mds.anchorX()[thirdMDIndex]);
    float midPointY = 0.5f * (mds.anchorY()[firstMDIndex] + mds.anchorY()[thirdMDIndex]);
    float diffX = mds.anchorX()[thirdMDIndex] - mds.anchorX()[firstMDIndex];
    float diffY = mds.anchorY()[thirdMDIndex] - mds.anchorY()[firstMDIndex];

    float dPhi = cms::alpakatools::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float sdIn_alpha_min = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
    float sdIn_alpha_max = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;

    float sdOut_dPhiPos =
        cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[thirdMDIndex]);

    float sdOut_dPhiChange = __H2F(segments.dPhiChanges()[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segments.dPhiChangeMins()[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segments.dPhiChangeMaxs()[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mds.anchorX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];

    float betaIn =
        sdIn_alpha - cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut =
        -sdOut_alphaOut + cms::alpakatools::reducePhiRange(
                              acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    bool isEC_secondLayer = (modules.subdets()[innerOuterLowerModuleIndex] == Endcap) and
                            (modules.moduleType()[innerOuterLowerModuleIndex] == TwoS);

    if (isEC_secondLayer) {
      betaInRHmin = betaIn - sdIn_alpha_min + sdIn_alpha;
      betaInRHmax = betaIn - sdIn_alpha_max + sdIn_alpha;
    }

    betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
    betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

    float swapTemp;
    if (alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax)) {
      swapTemp = betaOutRHmin;
      betaOutRHmin = betaOutRHmax;
      betaOutRHmax = swapTemp;
    }

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }

    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                               (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                           (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                               (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

    float lIn = 5;
    float lOut = 11;

    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) *
                                                (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) +
                                            (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) *
                                                (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]));
    float sdOut_d = mds.anchorRt()[fourthMDIndex] - mds.anchorRt()[thirdMDIndex];

    runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdIn_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdOut_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = alpaka::math::sin(acc, dPhi);

    const float dBetaRIn2 = 0;  // TODO-RH
    float dBetaROut = 0;
    if (modules.moduleType()[outerOuterLowerModuleIndex] == TwoS) {
      dBetaROut = (alpaka::math::sqrt(acc,
                                      mds.anchorHighEdgeX()[fourthMDIndex] * mds.anchorHighEdgeX()[fourthMDIndex] +
                                          mds.anchorHighEdgeY()[fourthMDIndex] * mds.anchorHighEdgeY()[fourthMDIndex]) -
                   alpaka::math::sqrt(acc,
                                      mds.anchorLowEdgeX()[fourthMDIndex] * mds.anchorLowEdgeX()[fourthMDIndex] +
                                          mds.anchorLowEdgeY()[fourthMDIndex] * mds.anchorLowEdgeY()[fourthMDIndex])) *
                  sinDPhi / dr;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletdBetaCutEEEE(TAcc const& acc,
                                                                ModulesConst modules,
                                                                MiniDoubletsConst mds,
                                                                SegmentsConst segments,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                uint16_t outerInnerLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                unsigned int fourthMDIndex,
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mds.anchorRt()[firstMDIndex];
    float rt_InOut = mds.anchorRt()[secondMDIndex];
    float rt_OutLo = mds.anchorRt()[thirdMDIndex];

    float z_InLo = mds.anchorZ()[firstMDIndex];
    float z_OutLo = mds.anchorZ()[thirdMDIndex];

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);
    float sdIn_alpha = __H2F(segments.dPhiChanges()[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;  //weird
    float sdOut_dPhiPos =
        cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[fourthMDIndex] - mds.anchorPhi()[thirdMDIndex]);

    float sdOut_dPhiChange = __H2F(segments.dPhiChanges()[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segments.dPhiChangeMins()[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segments.dPhiChangeMaxs()[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = cms::alpakatools::reducePhiRange(acc, sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mds.anchorX()[fourthMDIndex] - mds.anchorX()[firstMDIndex];
    float tl_axis_y = mds.anchorY()[fourthMDIndex] - mds.anchorY()[firstMDIndex];

    float betaIn =
        sdIn_alpha - cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[firstMDIndex]);

    float sdIn_alphaRHmin = __H2F(segments.dPhiChangeMins()[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segments.dPhiChangeMaxs()[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    float betaOut =
        -sdOut_alphaOut + cms::alpakatools::reducePhiRange(
                              acc, cms::alpakatools::phi(acc, tl_axis_x, tl_axis_y) - mds.anchorPhi()[fourthMDIndex]);

    float betaOutRHmin = betaOut - sdOut_alphaOutRHmin + sdOut_alphaOut;
    float betaOutRHmax = betaOut - sdOut_alphaOutRHmax + sdOut_alphaOut;

    float swapTemp;
    if (alpaka::math::abs(acc, betaOutRHmin) > alpaka::math::abs(acc, betaOutRHmax)) {
      swapTemp = betaOutRHmin;
      betaOutRHmin = betaOutRHmax;
      betaOutRHmax = swapTemp;
    }

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }
    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) *
                                               (mds.anchorX()[secondMDIndex] - mds.anchorX()[firstMDIndex]) +
                                           (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]) *
                                               (mds.anchorY()[secondMDIndex] - mds.anchorY()[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

    int lIn = 11;   //endcap
    int lOut = 13;  //endcap

    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) *
                                                (mds.anchorX()[fourthMDIndex] - mds.anchorX()[thirdMDIndex]) +
                                            (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]) *
                                                (mds.anchorY()[fourthMDIndex] - mds.anchorY()[thirdMDIndex]));
    float sdOut_d = mds.anchorRt()[fourthMDIndex] - mds.anchorRt()[thirdMDIndex];

    runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

    const float betaInMMSF = (alpaka::math::abs(acc, betaInRHmin + betaInRHmax) > 0)
                                 ? (2.f * betaIn / alpaka::math::abs(acc, betaInRHmin + betaInRHmax))
                                 : 0.;  //mean value of min,max is the old betaIn
    const float betaOutMMSF = (alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax) > 0)
                                  ? (2.f * betaOut / alpaka::math::abs(acc, betaOutRHmin + betaOutRHmax))
                                  : 0.;
    betaInRHmin *= betaInMMSF;
    betaInRHmax *= betaInMMSF;
    betaOutRHmin *= betaOutMMSF;
    betaOutRHmax *= betaOutMMSF;

    float min_ptBeta_maxPtBeta = alpaka::math::min(
        acc, alpaka::math::abs(acc, pt_beta), kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdIn_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, sdOut_alpha),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletdBetaAlgoSelector(TAcc const& acc,
                                                                     ModulesConst modules,
                                                                     MiniDoubletsConst mds,
                                                                     SegmentsConst segments,
                                                                     uint16_t innerInnerLowerModuleIndex,
                                                                     uint16_t innerOuterLowerModuleIndex,
                                                                     uint16_t outerInnerLowerModuleIndex,
                                                                     uint16_t outerOuterLowerModuleIndex,
                                                                     unsigned int innerSegmentIndex,
                                                                     unsigned int outerSegmentIndex,
                                                                     unsigned int firstMDIndex,
                                                                     unsigned int secondMDIndex,
                                                                     unsigned int thirdMDIndex,
                                                                     unsigned int fourthMDIndex,
                                                                     float& dBeta,
                                                                     const float ptCut) {
    short innerInnerLowerModuleSubdet = modules.subdets()[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modules.subdets()[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modules.subdets()[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modules.subdets()[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
        outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Barrel) {
      return runQuadrupletdBetaCutBBBB(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuadrupletdBetaCutBBEE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Endcap) {
      return runQuadrupletdBetaCutBBBB(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuadrupletdBetaCutBBEE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == Endcap and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuadrupletdBetaCutEEEE(acc,
                                       modules,
                                       mds,
                                       segments,
                                       innerInnerLowerModuleIndex,
                                       innerOuterLowerModuleIndex,
                                       outerInnerLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       fourthMDIndex,
                                       dBeta,
                                       ptCut);
    }

    return false;
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
                                                               float& fakeScore,
                                                               float& x_5,
                                                               bool& tightDNNFlag,
                                                               bool& tightCutFlag) {
    unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0]; //second and third segments are the same here
    unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];

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
      return false;
    if (innerOuterOuterMiniDoubletIndex != outerOuterInnerMiniDoubletIndex)
      return false; 
    
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
    float pt = (innerRadius+outerRadius) * k2Rinv1GeVf;
    
    // 4 categories for sigmas
    float sigmas2[4], delta1[4], delta2[4], slopes[4];
    bool isFlat[4];

    float xVec[] = {x1, x2, x3, x4};
    float yVec[] = {y1, y2, y3, y4};

    const uint16_t lowerModuleIndices[] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4};

    computeSigmasForRegressionT4(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    regressionRadius = computeRadiusUsingRegressionT4(acc,
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
    float nonAnchorDelta1[Params_T4::kLayers], nonAnchorDelta2[Params_T4::kLayers], nonAnchorSlopes[Params_T4::kLayers];
    float nonAnchorxs[] = {mds.outerX()[firstMDIndex],
                           mds.outerX()[secondMDIndex],
                           mds.outerX()[thirdMDIndex],
                           mds.outerX()[fourthMDIndex]};
    float nonAnchorys[] = {mds.outerY()[firstMDIndex],
                           mds.outerY()[secondMDIndex],
                           mds.outerY()[thirdMDIndex],
                           mds.outerY()[fourthMDIndex]};

    computeSigmasForRegressionT4(acc,
                               modules,
                               lowerModuleIndices,
                               nonAnchorDelta1,
                               nonAnchorDelta2,
                               nonAnchorSlopes,
                               isFlat,
                               Params_T4::kLayers,
                               false);
    
    nonAnchorRegressionRadius = computeRadiusUsingRegressionT4(acc,
                                                    Params_T4::kLayers,
                                                    nonAnchorxs,
                                                    nonAnchorys,
                                                    nonAnchorDelta1,
                                                    nonAnchorDelta2,
                                                    nonAnchorSlopes,
                                                    isFlat,
                                                    regressionG,
                                                    regressionF,
                                                    sigmas2,
                                                    chiSquared);                           

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
                                              tightDNNFlag,
                                              regressionRadius,
                                              nonAnchorRegressionRadius,
                                              triplets.fakeScore()[innerTripletIndex],
                                              triplets.promptScore()[innerTripletIndex],
                                              triplets.displacedScore()[innerTripletIndex],
                                              triplets.fakeScore()[outerTripletIndex],
                                              triplets.promptScore()[outerTripletIndex],
                                              triplets.displacedScore()[outerTripletIndex]);

    if (!inference)
      return false;

    //run Beta Selector for high pT T4s
    if (pt >10) {
      if (not runQuadrupletdBetaAlgoSelector(acc,
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
                              innerT3charge,
                              tightCutFlag))
    return false; 

    
    nonAnchorChiSquared = computeT4ChiSquared(acc,
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
      for (int iter : cms::alpakatools::uniform_elements_z(acc, nEligibleT4Modules)) {
        uint16_t lowerModule1 = ranges.indicesOfEligibleT4Modules()[iter];
        short layer2_adjustment;
        short md_adjustment;
        int layer = modules.layers()[lowerModule1];
        if (layer == 1) {
          layer2_adjustment = 1;
          md_adjustment = 1;
        }  // get upper segment to be in third layer
        else if (layer ==2) {
          layer2_adjustment = 1; 
          md_adjustment = 0;
        }  // get lower segment to be in third layer
        else {
          layer2_adjustment = 0; 
          md_adjustment = 0;
        }
        unsigned int nInnerTriplets = tripletsOccupancy.nTriplets()[lowerModule1];
        for (unsigned int innerTripletArrayIndex : cms::alpakatools::uniform_elements_y(acc, nInnerTriplets)) {
          unsigned int innerTripletIndex = ranges.tripletModuleIndices()[lowerModule1] + innerTripletArrayIndex;
          // if (triplets.partOfPT5()[innerTripletIndex])
          //     continue;  //don't create T4s for T3s accounted in pT5s
          // if (triplets.partOfPT3()[innerTripletIndex])
          //     continue;  //don't create T4s for T3s accounted in pT3s
          // if (triplets.partOfT5()[innerTripletIndex])
          //     continue;  //don't create T4s for T3s accounted in T5s
          uint16_t lowerModule2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
          unsigned int nOuterTriplets = tripletsOccupancy.nTriplets()[lowerModule2];
          for (unsigned int outerTripletArrayIndex : cms::alpakatools::uniform_elements_x(acc, nOuterTriplets)) {
            unsigned int outerTripletIndex = ranges.tripletModuleIndices()[lowerModule2] + outerTripletArrayIndex;
            // if (triplets.partOfPT5()[outerTripletIndex])
            //   continue;  //don't create T4s for T3s accounted in pT5s
            // if (triplets.partOfPT3[outerTripletIndex])
            //   continue;  //don't create T4s for T3s accounted in pT3s
            // if (triplets.partOfT5()[outerTripletIndex])
            //   continue;  //don't create T4s for T3s accounted in T5s
            uint16_t lowerModule3 = triplets.lowerModuleIndices()[outerTripletIndex][1];
            uint16_t lowerModule4 = triplets.lowerModuleIndices()[outerTripletIndex][2];
            float innerRadius = triplets.radius()[innerTripletIndex];
            float outerRadius = triplets.radius()[outerTripletIndex];  
            float rzChiSquared, dBeta, nonAnchorChiSquared, regressionG, regressionF, regressionRadius, nonAnchorRegressionRadius, chiSquared, promptScore, displacedScore, x_5, fakeScore; 
            bool tightDNNFlag = false;
            bool tightCutFlag = false;
      
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
                                                    fakeScore,
                                                    x_5,
                                                    tightDNNFlag,
                                                    tightCutFlag); 
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
                  unsigned int quadrupletIndex =
                      ranges.quadrupletModuleIndices()[lowerModule1] + quadrupletModuleIndex;
                  float phi =
                      mds.anchorPhi()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][md_adjustment]][layer2_adjustment]]; //layer 3
                  float eta =
                      mds.anchorEta()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][md_adjustment]][layer2_adjustment]]; //layer 3
                  
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
                                        nonAnchorRegressionRadius,
                                        tightDNNFlag,
                                        tightCutFlag);

                  // triplets.partOfT4[quadrupletsInGPU.tripletIndices[2 * quadrupletIndex]] = true;
                  // triplets.partOfT4[quadrupletsInGPU.tripletIndices[2 * quadrupletIndex + 1]] = true;
                }
              }
            }
          }
        }
      }
    }
  };

  struct CreateEligibleModulesListForQuadruplets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  TripletsOccupancyConst tripletsOccupancy,
                                  ObjectRanges ranges,
                                  Triplets triplets,
                                  const float ptCut) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      // Initialize variables in shared memory and set to 0
      int& nEligibleT4Modulesx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nTotalQuadrupletsx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalQuadrupletsx = 0;
        nEligibleT4Modulesx = 0;
      }
      alpaka::syncBlockThreads(acc);

      // Occupancy matrix for 0.8 GeV pT Cut
      constexpr int p08_occupancy_matrix[4][4] = {
          {500000, 500000, 500000, 500000},  // category 0
          {500000, 500000, 500000, 500000},          // category 1
          {500000, 500000, 500000, 500000},          // category 2
          {500000, 500000, 500000, 500000}       // category 3
      };

      // Occupancy matrix for 0.6 GeV pT Cut, 99.99% //FIXME -recalculate 
      constexpr int p06_occupancy_matrix[4][4] = {
          {1500000, 1500000, 1500000, 1500000},  // category 0
          {1500000, 1500000, 1500000, 1500000},          // category 1
          {1500000, 1500000, 1500000, 1500000},          // category 2
          {1500000, 1500000, 1500000, 1500000}       // category 3
      };

      // Select the appropriate occupancy matrix based on ptCut
      const auto& occupancy_matrix = (ptCut < 0.8f) ? p06_occupancy_matrix : p08_occupancy_matrix;

      for (int i : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        // Condition for a quadruple to exist for a module
        // T4s don't exist for layers 4, 5, 6 barrel, and layers 3,4,5 endcap
        short module_rings = modules.rings()[i];
        short module_layers = modules.layers()[i];
        short module_subdets = modules.subdets()[i];
        float module_eta = alpaka::math::abs(acc, modules.eta()[i]);

        if (tripletsOccupancy.nTriplets()[i] == 0)
          continue;
        if (module_subdets == Barrel and module_layers > 3)
          continue;
        if (module_subdets == Endcap and module_layers > 2)
          continue;

        int dynamic_count = 0;

        // How many triplets are in module i?
        int nTriplets_i = tripletsOccupancy.nTriplets()[i];
        int firstTripletIdx = ranges.tripletModuleIndices()[i];

        // Loop over all triplets that live in module i
        for (int t = 0; t < nTriplets_i; t++) {
          int tripletIndex = firstTripletIdx + t;
          uint16_t outerModule = triplets.lowerModuleIndices()[tripletIndex][1];
          dynamic_count += tripletsOccupancy.nTriplets()[outerModule];
        }

        int category_number = getCategoryNumber(module_layers, module_subdets, module_rings);
        int eta_number = getEtaBin(module_eta);

#ifdef WARNINGS
        if (category_number == -1 || eta_number == -1) {
          printf("Unhandled case in createEligibleModulesListForQuintupletsGPU! Module index = %i\n", i);
        }
#endif
        // Get matrix-based cap (use dynamic_count as fallback)
        int matrix_cap =
            (category_number != -1 && eta_number != -1) ? occupancy_matrix[category_number][eta_number] : 0;

        // Cap occupancy at minimum of dynamic count and matrix value
        int occupancy = alpaka::math::min(acc, dynamic_count, matrix_cap);
        // if (dynamic_count > matrix_cap){
        //   printf("dynamic count: %d, matrix_cap: %d\n", dynamic_count, matrix_cap);
        // }

        int nEligibleT4Modules = alpaka::atomicAdd(acc, &nEligibleT4Modulesx, 1, alpaka::hierarchy::Threads{});
        int nTotQ = alpaka::atomicAdd(acc, &nTotalQuadrupletsx, occupancy, alpaka::hierarchy::Threads{});

        ranges.quadrupletModuleIndices()[i] = nTotQ;
        // printf("in create elig mod list: ranges.quadrupletModuleIndices()[%d] = %d\n", i, ranges.quadrupletModuleIndices()[i]);
        ranges.indicesOfEligibleT4Modules()[nEligibleT4Modules] = i;
        ranges.quadrupletModuleOccupancy()[i] = occupancy;
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
