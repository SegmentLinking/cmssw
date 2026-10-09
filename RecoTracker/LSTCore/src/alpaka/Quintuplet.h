#ifndef RecoTracker_LSTCore_src_alpaka_Quintuplet_h
#define RecoTracker_LSTCore_src_alpaka_Quintuplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/Circle.h"

#include "NeuralNetwork.h"
#include "TripletAccessors.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuintupletToMemory(ModulesConst modules,
                                                            MiniDoubletsConst mds,
                                                            SegmentsConst segments,
                                                            TripletsConst triplets,
                                                            Quintuplets quintuplets,
                                                            QuintupletsByMD quintupletsByMD0,
                                                            QuintupletsByMD quintupletsByMD1,
                                                            unsigned int innerTripletIndex,
                                                            unsigned int outerTripletIndex,
                                                            uint16_t lowerModule1,
                                                            uint16_t lowerModule2,
                                                            uint16_t lowerModule3,
                                                            uint16_t lowerModule4,
                                                            uint16_t lowerModule5,
                                                            float innerRadius,
                                                            float bridgeRadius,
                                                            float regressionCenterX,
                                                            float regressionCenterY,
                                                            float regressionRadius,
                                                            float rzChiSquared,
                                                            float rPhiChiSquared,
                                                            float dBeta1,
                                                            float dBeta2,
                                                            float eta,
                                                            float phi,
                                                            uint8_t layer,
                                                            unsigned int quintupletIndex,
                                                            unsigned int quintupletByMD0Index,
                                                            unsigned int quintupletByMD1Index,
                                                            const float (&t5Embed)[Params_T5::kEmbed],
                                                            float dnnScore) {
    quintuplets.tripletIndices()[quintupletIndex][0] = innerTripletIndex;
    quintuplets.tripletIndices()[quintupletIndex][1] = outerTripletIndex;

    quintuplets.lowerModuleIndices()[quintupletIndex][0] = lowerModule1;
    quintuplets.lowerModuleIndices()[quintupletIndex][1] = lowerModule2;
    quintuplets.lowerModuleIndices()[quintupletIndex][2] = lowerModule3;
    quintuplets.lowerModuleIndices()[quintupletIndex][3] = lowerModule4;
    quintuplets.lowerModuleIndices()[quintupletIndex][4] = lowerModule5;
    quintuplets.innerRadius()[quintupletIndex] = __F2H(innerRadius);
    quintuplets.eta()[quintupletIndex] = __F2H(eta);
    quintuplets.phi()[quintupletIndex] = __F2H(phi);
    quintuplets.isDup()[quintupletIndex] = 0;
    quintuplets.nLayers()[quintupletIndex] = Params_T5::kBaseLayers;
    quintuplets.regressionRadius()[quintupletIndex] = regressionRadius;
    quintuplets.regressionCenterX()[quintupletIndex] = regressionCenterX;
    quintuplets.regressionCenterY()[quintupletIndex] = regressionCenterY;
    quintuplets.logicalLayers()[quintupletIndex][0] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[innerTripletIndex][0]);
    quintuplets.logicalLayers()[quintupletIndex][1] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[innerTripletIndex][1]);
    quintuplets.logicalLayers()[quintupletIndex][2] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[innerTripletIndex][2]);
    quintuplets.logicalLayers()[quintupletIndex][3] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[outerTripletIndex][1]);
    quintuplets.logicalLayers()[quintupletIndex][4] =
        getLogicalLayer(modules, triplets.lowerModuleIndices()[outerTripletIndex][2]);

    unsigned int innerT3Hits[Params_T3::kHits], outerT3Hits[Params_T3::kHits];
    getTripletHitIndices(mds, segments, triplets, innerTripletIndex, innerT3Hits);
    getTripletHitIndices(mds, segments, triplets, outerTripletIndex, outerT3Hits);

    quintuplets.hitIndices()[quintupletIndex][0] = innerT3Hits[0];
    quintuplets.hitIndices()[quintupletIndex][1] = innerT3Hits[1];
    quintuplets.hitIndices()[quintupletIndex][2] = innerT3Hits[2];
    quintuplets.hitIndices()[quintupletIndex][3] = innerT3Hits[3];
    quintuplets.hitIndices()[quintupletIndex][4] = innerT3Hits[4];
    quintuplets.hitIndices()[quintupletIndex][5] = innerT3Hits[5];
    quintuplets.hitIndices()[quintupletIndex][6] = outerT3Hits[2];
    quintuplets.hitIndices()[quintupletIndex][7] = outerT3Hits[3];
    quintuplets.hitIndices()[quintupletIndex][8] = outerT3Hits[4];
    quintuplets.hitIndices()[quintupletIndex][9] = outerT3Hits[5];
#ifdef CUT_VALUE_DEBUG
    quintuplets.bridgeRadius()[quintupletIndex] = bridgeRadius;
    quintuplets.rzChiSquared()[quintupletIndex] = rzChiSquared;
    quintuplets.chiSquared()[quintupletIndex] = rPhiChiSquared;
    quintuplets.dBeta1()[quintupletIndex] = dBeta1;
    quintuplets.dBeta2()[quintupletIndex] = dBeta2;
#endif
    quintuplets.dnnScore()[quintupletIndex] = dnnScore;

    CMS_UNROLL_LOOP
    for (unsigned int i = 0; i < Params_T5::kEmbed; ++i) {
      quintuplets.t5Embed()[quintupletIndex][i] = t5Embed[i];
    }

    // Initialize extended layer slots with sentinel values
    for (int i = Params_T5::kBaseLayers; i < Params_T5::kLayers; ++i) {
      quintuplets.logicalLayers()[quintupletIndex][i] = 0;
      quintuplets.lowerModuleIndices()[quintupletIndex][i] = lst::kTCEmptyLowerModule;
      quintuplets.hitIndices()[quintupletIndex][2 * i] = lst::kTCEmptyHitIdx;
      quintuplets.hitIndices()[quintupletIndex][2 * i + 1] = lst::kTCEmptyHitIdx;
    }

    auto const& lsIdx = triplets.segmentIndices();
    auto const& mdIdx = segments.mdIndices();
    const uint32_t md1Bar = mdIdx[lsIdx[innerTripletIndex][1]][0] & kT5ByMDBarCodeMask;
    const uint32_t md2Bar = mdIdx[lsIdx[innerTripletIndex][1]][1] & kT5ByMDBarCodeMask;
    const uint32_t md3Bar = mdIdx[lsIdx[outerTripletIndex][1]][0] & kT5ByMDBarCodeMask;
    const uint32_t md4Bar = mdIdx[lsIdx[outerTripletIndex][1]][1] & kT5ByMDBarCodeMask;
    const uint32_t md1234Bar =
        md1Bar | (md2Bar << kT5ByMDBarOffset) | (md3Bar << (kT5ByMDBarOffset * 2)) | (md4Bar << (kT5ByMDBarOffset * 3));
    if (quintupletByMD0Index != kInvalidU32Idx) {
      quintupletsByMD0.quintupletIndex()[quintupletByMD0Index] = quintupletIndex;
      quintupletsByMD0.mdBarCode()[quintupletByMD0Index] = md1234Bar;
    }
    if (quintupletByMD1Index != kInvalidU32Idx) {
      quintupletsByMD1.quintupletIndex()[quintupletByMD1Index] = quintupletIndex;
      //save the starting logical layer index instead of md1Bar
      quintupletsByMD1.mdBarCode()[quintupletByMD1Index] =
          (quintuplets.logicalLayers()[quintupletIndex][0] & kT5ByMDBarCodeMask) | (md1234Bar & ~kT5ByMDBarCodeMask);
    }
  }

  // Helix start of passT5RZConstraint from the inner triplet alone (its circle centre and MDs 1-3).
  struct T5RZInnerTerms {
    float x_init, y_init, z_init, rt_init, Px, Py, Pz, momentum, chargeTimesField;
  };

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE T5RZInnerTerms computeT5RZInnerTerms(TAcc const& acc,
                                                                      ModulesConst modules,
                                                                      MiniDoubletsConst mds,
                                                                      SegmentsConst segments,
                                                                      TripletsConst triplets,
                                                                      unsigned int innerTripletIndex) {
    const uint16_t lowerModuleIndex3 = triplets.lowerModuleIndices()[innerTripletIndex][2];
    const unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    const unsigned int firstMDIndex = segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][0]][0];
    const unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    const unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    const float innerRadius = triplets.radius()[innerTripletIndex];
    const float inner_pt = 2 * k2Rinv1GeVf * innerRadius;
    const float z1 = mds.anchorZ()[firstMDIndex] / 100;               //in the unit of m instead of cm
    const int moduleType3 = modules.moduleType()[lowerModuleIndex3];  //0 is ps, 1 is 2s
    const float x1 = mds.anchorX()[firstMDIndex] / 100;
    const float x3 = mds.anchorX()[thirdMDIndex] / 100;
    const float y1 = mds.anchorY()[firstMDIndex] / 100;
    const float y3 = mds.anchorY()[thirdMDIndex] / 100;

    // centre of the circle fitted by the innermost 3 points on x,y coordinates
    float x_center = triplets.centerX()[innerTripletIndex] / 100,
          y_center = triplets.centerY()[innerTripletIndex] / 100;
    float x_init = mds.anchorX()[thirdMDIndex] / 100;
    float y_init = mds.anchorY()[thirdMDIndex] / 100;
    float z_init = mds.anchorZ()[thirdMDIndex] / 100;
    float rt_init = mds.anchorRt()[thirdMDIndex] / 100;  //use the second MD as initial point

    if (moduleType3 == 1)  // 1: if MD3 is in 2s layer
    {
      x_init = mds.anchorX()[secondMDIndex] / 100;
      y_init = mds.anchorY()[secondMDIndex] / 100;
      z_init = mds.anchorZ()[secondMDIndex] / 100;
      rt_init = mds.anchorRt()[secondMDIndex] / 100;
    }

    // start from a circle of inner T3.
    // to determine the charge
    int charge = 0;
    float slope3c = (y3 - y_center) / (x3 - x_center);
    float slope1c = (y1 - y_center) / (x1 - x_center);
    // these 4 "if"s basically separate the x-y plane into 4 quarters. It determines geometrically how a circle and line slope goes and their positions, and we can get the charges correspondingly.
    if ((y3 - y_center) > 0 && (y1 - y_center) > 0) {
      if (slope1c > 0 && slope3c < 0)
        charge = -1;  // on x axis of a quarter, 3 hits go anti-clockwise
      else if (slope1c < 0 && slope3c > 0)
        charge = 1;  // on x axis of a quarter, 3 hits go clockwise
      else if (slope3c > slope1c)
        charge = -1;
      else if (slope3c < slope1c)
        charge = 1;
    } else if ((y3 - y_center) < 0 && (y1 - y_center) < 0) {
      if (slope1c < 0 && slope3c > 0)
        charge = 1;
      else if (slope1c > 0 && slope3c < 0)
        charge = -1;
      else if (slope3c > slope1c)
        charge = -1;
      else if (slope3c < slope1c)
        charge = 1;
    } else if ((y3 - y_center) < 0 && (y1 - y_center) > 0) {
      if ((x3 - x_center) > 0 && (x1 - x_center) > 0)
        charge = 1;
      else if ((x3 - x_center) < 0 && (x1 - x_center) < 0)
        charge = -1;
    } else if ((y3 - y_center) > 0 && (y1 - y_center) < 0) {
      if ((x3 - x_center) > 0 && (x1 - x_center) > 0)
        charge = -1;
      else if ((x3 - x_center) < 0 && (x1 - x_center) < 0)
        charge = 1;
    }

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

    //to get Pz, we use pt/pz=ds/dz, ds is the arclength between MD1 and MD3.
    float AO = alpaka::math::sqrt(acc, (x1 - x_center) * (x1 - x_center) + (y1 - y_center) * (y1 - y_center));
    float BO =
        alpaka::math::sqrt(acc, (x_init - x_center) * (x_init - x_center) + (y_init - y_center) * (y_init - y_center));
    float AB2 = (x1 - x_init) * (x1 - x_init) + (y1 - y_init) * (y1 - y_init);
    float dPhi = alpaka::math::acos(acc, (AO * AO + BO * BO - AB2) / (2 * AO * BO));
    float ds = innerRadius / 100 * dPhi;

    float Pz = (z_init - z1) / ds * Pt;
    float momentum = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);

    float chargeTimesField = -2.f * k2Rinv1GeVf * 100 * charge;  // multiply by 100 to make the correct length units

    return {x_init, y_init, z_init, rt_init, Px, Py, Pz, momentum, chargeTimesField};
  }

  //bounds can be found at http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_RZFix/t5_rz_thresholds.txt
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passT5RZConstraint(TAcc const& acc,
                                                         ModulesConst modules,
                                                         MiniDoubletsConst mds,
                                                         T5RZInnerTerms const& rzInner,
                                                         unsigned int firstMDIndex,
                                                         unsigned int secondMDIndex,
                                                         unsigned int thirdMDIndex,
                                                         unsigned int fourthMDIndex,
                                                         unsigned int fifthMDIndex,
                                                         uint16_t lowerModuleIndex1,
                                                         uint16_t lowerModuleIndex2,
                                                         uint16_t lowerModuleIndex3,
                                                         uint16_t lowerModuleIndex4,
                                                         uint16_t lowerModuleIndex5,
                                                         float& rzChiSquared,
                                                         float inner_pt) {
    const float rt1 = mds.anchorRt()[firstMDIndex] / 100;  //in the unit of m instead of cm
    const float rt2 = mds.anchorRt()[secondMDIndex] / 100;
    const float rt3 = mds.anchorRt()[thirdMDIndex] / 100;
    const float rt4 = mds.anchorRt()[fourthMDIndex] / 100;
    const float rt5 = mds.anchorRt()[fifthMDIndex] / 100;

    const float z1 = mds.anchorZ()[firstMDIndex] / 100;
    const float z2 = mds.anchorZ()[secondMDIndex] / 100;
    const float z3 = mds.anchorZ()[thirdMDIndex] / 100;
    const float z4 = mds.anchorZ()[fourthMDIndex] / 100;
    const float z5 = mds.anchorZ()[fifthMDIndex] / 100;

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer1 = modules.lstLayers()[lowerModuleIndex1];
    const int layer2 = modules.lstLayers()[lowerModuleIndex2];
    const int layer3 = modules.lstLayers()[lowerModuleIndex3];
    const int layer4 = modules.lstLayers()[lowerModuleIndex4];
    const int layer5 = modules.lstLayers()[lowerModuleIndex5];

    //slope computed using the internal T3s
    const int moduleType1 = modules.moduleType()[lowerModuleIndex1];  //0 is ps, 1 is 2s
    const int moduleType2 = modules.moduleType()[lowerModuleIndex2];
    const int moduleType3 = modules.moduleType()[lowerModuleIndex3];
    const int moduleType4 = modules.moduleType()[lowerModuleIndex4];
    const int moduleType5 = modules.moduleType()[lowerModuleIndex5];

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
    const float x_init = rzInner.x_init, y_init = rzInner.y_init, z_init = rzInner.z_init, rt_init = rzInner.rt_init;
    // p and a keep the names of the helix code below (as in computePT3RZChiSquared)
    const float Pz = rzInner.Pz, p = rzInner.momentum, a = rzInner.chargeTimesField;
    float Px = rzInner.Px, Py = rzInner.Py;

    // But if the initial T5 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of Px,Py signs on these to avoid errors
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

    float zsi, rtsi;
    int layeri, moduleTypei;
    rzChiSquared = 0;
    for (size_t i = 2; i < 6; i++) {
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
      } else if (i == 5) {
        zsi = z5;
        rtsi = rt5;
        layeri = layer5;
        moduleTypei = moduleType5;
      }

      if (moduleType3 == 0) {  //0: ps
        if (i == 3)
          continue;
      } else {
        if (i == 2)
          continue;
      }

      // calculation is copied from PixelTriplet.h computePT3RZChiSquared
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
        if (edm::isNotFinite(diffz1))
          diffz = diffz2;
        else if (edm::isNotFinite(diffz2))
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
    // if the 5 points are linear, helix calculation gives nan
    if (inner_pt > 100 || edm::isNotFinite(rzChiSquared)) {
      float slope;
      if (moduleType1 == 0 and moduleType2 == 0 and moduleType3 == 1)  //PSPS2S
      {
        slope = (z2 - z1) / (rt2 - rt1);
      } else {
        slope = (z3 - z1) / (rt3 - rt1);
      }
      float residual4_linear = (layer4 <= 6) ? ((z4 - z1) - slope * (rt4 - rt1)) : ((rt4 - rt1) - (z4 - z1) / slope);
      float residual5_linear = (layer4 <= 6) ? ((z5 - z1) - slope * (rt5 - rt1)) : ((rt5 - rt1) - (z5 - z1) / slope);

      // creating a chi squared type quantity
      // 0-> PS, 1->2S
      residual4_linear = (moduleType4 == 0) ? residual4_linear / kPixelPSZpitch : residual4_linear / kStrip2SZpitch;
      residual5_linear = (moduleType5 == 0) ? residual5_linear / kPixelPSZpitch : residual5_linear / kStrip2SZpitch;
      residual4_linear = residual4_linear * 100;
      residual5_linear = residual5_linear * 100;

      rzChiSquared = 12 * (residual4_linear * residual4_linear + residual5_linear * residual5_linear);
      return rzChiSquared < 4.677f;
    }

    // The category numbers are related to module regions and layers, decoding of the region numbers can be found here in slide 2 table. https://github.com/SegmentLinking/TrackLooper/files/11420927/part.2.pdf
    // The commented numbers after each case is the region code, and can look it up from the table to see which category it belongs to. For example, //0 means T5 built with Endcap 1,2,3,4,5 ps modules
    if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 11)  //0
    {
      return true;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10 and layer5 == 16)  //1
    {
      return rzChiSquared < 37.956f;
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15 and layer5 == 16)  //2
    {
      return rzChiSquared < 11.622f;
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8 and layer4 == 9) {
      if (layer5 == 10)  //3
      {
        return true;
      }
      if (layer5 == 15)  //4
      {
        return rzChiSquared < 37.941f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 9)  //5
      {
        return true;
      }
      if (layer4 == 8 and layer5 == 14)  //6
      {
        return rzChiSquared < 52.561f;
      } else if (layer4 == 13 and layer5 == 14)  //7
      {
        return rzChiSquared < 13.76f;
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 7 and layer5 == 8)  //8
      {
        return rzChiSquared < 44.247f;
      } else if (layer4 == 7 and layer5 == 13)  //9
      {
        return rzChiSquared < 33.752f;
      } else if (layer4 == 12 and layer5 == 13)  //10
      {
        return rzChiSquared < 21.213f;
      } else if (layer4 == 4 and layer5 == 5)  //11
      {
        return rzChiSquared < 29.035f;
      } else if (layer4 == 4 and layer5 == 12)  //12
      {
        return rzChiSquared < 23.037f;
      }
    } else if (layer1 == 2 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9 and layer5 == 15)  //14
      {
        return rzChiSquared < 41.036f;
      } else if (layer4 == 14 and layer5 == 15)  //15
      {
        return rzChiSquared < 14.092f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 7) {
      if (layer4 == 8 and layer5 == 14)  //16
      {
        return rzChiSquared < 23.748f;
      }
      if (layer4 == 13 and layer5 == 14)  //17
      {
        return rzChiSquared < 17.945f;
      }
    } else if (layer1 == 2 and layer2 == 3 and layer3 == 4) {
      if (layer4 == 5 and layer5 == 6)  //18
      {
        return rzChiSquared < 8.803f;
      } else if (layer4 == 5 and layer5 == 12)  //19
      {
        return rzChiSquared < 7.930f;
      }

      else if (layer4 == 12 and layer5 == 13)  //20
      {
        return rzChiSquared < 7.626f;
      }
    }
    return true;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T5HasCommonMiniDoublet(TripletsConst triplets,
                                                             SegmentsConst segments,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int outerTripletIndex) {
    unsigned int innerOuterSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int outerInnerSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    unsigned int innerOuterOuterMiniDoubletIndex =
        segments.mdIndices()[innerOuterSegmentIndex][1];  //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex =
        segments.mdIndices()[outerInnerSegmentIndex][0];  //outer triplet inner segment inner MD index

    return (innerOuterOuterMiniDoubletIndex == outerInnerInnerMiniDoubletIndex);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegression(TAcc const& acc,
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
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusUsingRegression(TAcc const& acc,
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

      absArctanSlope =
          (edm::isFinite(slopes[i]) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i])) : kPi / 2.f);

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
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeChiSquared(TAcc const& acc,
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
      absArctanSlope =
          (edm::isFinite(slopes[i]) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i])) : kPi / 2.f);
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
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void runDeltaBetaIterations(TAcc const& acc,
                                                             float& betaIn,
                                                             float& betaOut,
                                                             float& pt_beta,
                                                             float sdIn_dr,
                                                             float sdOut_dr,
                                                             float dr,
                                                             float lIn,
                                                             bool useBetaInSign = false) {
    if (lIn == 0) {
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          useBetaInSign ? betaIn : betaOut);
      return;
    }

    if (betaIn * betaOut > 0.f and
        (alpaka::math::abs(acc, pt_beta) < 4.f * kPt_betaMax or
         (lIn >= 11 and alpaka::math::abs(acc, pt_beta) <
                            8.f * kPt_betaMax)))  //and the pt_beta is well-defined; less strict for endcap-endcap
    {
      const float betaInUpd =
          betaIn +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
              betaOut);  //FIXME: need a faster version
      float betaAv = 0.5f * (betaInUpd + betaOutUpd);

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
          betaIn +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc,
                  alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      float betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn))
                         ? (0.5f * (betaInUpd + betaOutUpd))
                         : betaInUpd;

      //1st update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdIn_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc, alpaka::math::min(acc, sdOut_dr * k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    }
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletdBetaCutBBBB(TAcc const& acc,
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
                                                                const float ptCut,
                                                                const float dBetaCut2Scale = 1.f) {
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

    alpha_OutUp = segments.dPhiChangeOuts()[outerSegmentIndex];

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

    // outer MD strip edges (2S endcap only): anchor xy +- the module's strip half-vector
    float highEdgeX_OutUp = 0.f, highEdgeY_OutUp = 0.f, lowEdgeX_OutUp = 0.f, lowEdgeY_OutUp = 0.f;
    if (isEC_lastLayer) {
      highEdgeX_OutUp = mds.anchorX()[fourthMDIndex] + modules.edgeDx()[outerOuterLowerModuleIndex];
      highEdgeY_OutUp = mds.anchorY()[fourthMDIndex] + modules.edgeDy()[outerOuterLowerModuleIndex];
      lowEdgeX_OutUp = mds.anchorX()[fourthMDIndex] - modules.edgeDx()[outerOuterLowerModuleIndex];
      lowEdgeY_OutUp = mds.anchorY()[fourthMDIndex] - modules.edgeDy()[outerOuterLowerModuleIndex];
      const float highEdgePhi_OutUp = alpaka::math::atan2(acc, highEdgeY_OutUp, highEdgeX_OutUp);
      const float lowEdgePhi_OutUp = alpaka::math::atan2(acc, lowEdgeY_OutUp, lowEdgeX_OutUp);
      alpha_OutUp_highEdge = cms::alpakatools::reducePhiRange(
          acc,
          cms::alpakatools::phi(
              acc, highEdgeX_OutUp - mds.anchorX()[thirdMDIndex], highEdgeY_OutUp - mds.anchorY()[thirdMDIndex]) -
              highEdgePhi_OutUp);
      alpha_OutUp_lowEdge = cms::alpakatools::reducePhiRange(
          acc,
          cms::alpakatools::phi(
              acc, lowEdgeX_OutUp - mds.anchorX()[thirdMDIndex], lowEdgeY_OutUp - mds.anchorY()[thirdMDIndex]) -
              lowEdgePhi_OutUp);

      tl_axis_highEdge_x = highEdgeX_OutUp - mds.anchorX()[firstMDIndex];
      tl_axis_highEdge_y = highEdgeY_OutUp - mds.anchorY()[firstMDIndex];
      tl_axis_lowEdge_x = lowEdgeX_OutUp - mds.anchorX()[firstMDIndex];
      tl_axis_lowEdge_y = lowEdgeY_OutUp - mds.anchorY()[firstMDIndex];

      betaOutRHmin = -alpha_OutUp_highEdge +
                     cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_highEdge_x, tl_axis_highEdge_y) - highEdgePhi_OutUp);
      betaOutRHmax = -alpha_OutUp_lowEdge +
                     cms::alpakatools::reducePhiRange(
                         acc, cms::alpakatools::phi(acc, tl_axis_lowEdge_x, tl_axis_lowEdge_y) - lowEdgePhi_OutUp);
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

    runDeltaBetaIterations(acc, betaIn, betaOut, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

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

    const float alphaInAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_InLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float alphaOutAbsReg =
        alpaka::math::max(acc,
                          alpaka::math::abs(acc, alpha_OutLo),
                          alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * k2Rinv1GeVf / 3.0f, kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = alpaka::math::sin(acc, dPhi);

    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut = (alpaka::math::sqrt(acc, highEdgeX_OutUp * highEdgeX_OutUp + highEdgeY_OutUp * highEdgeY_OutUp) -
                   alpaka::math::sqrt(acc, lowEdgeX_OutUp * lowEdgeX_OutUp + lowEdgeY_OutUp * lowEdgeY_OutUp)) *
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
    return dBeta * dBeta <= dBetaCut2 * dBetaCut2Scale;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletdBetaCutBBEE(TAcc const& acc,
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
                                                                const float ptCut,
                                                                const float dBetaCut2Scale = 1.f) {
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

    runDeltaBetaIterations(acc, betaIn, betaOut, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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
      // outer MD strip edges: anchor xy +- the module's strip half-vector
      const float highEdgeX_OutUp = mds.anchorX()[fourthMDIndex] + modules.edgeDx()[outerOuterLowerModuleIndex];
      const float highEdgeY_OutUp = mds.anchorY()[fourthMDIndex] + modules.edgeDy()[outerOuterLowerModuleIndex];
      const float lowEdgeX_OutUp = mds.anchorX()[fourthMDIndex] - modules.edgeDx()[outerOuterLowerModuleIndex];
      const float lowEdgeY_OutUp = mds.anchorY()[fourthMDIndex] - modules.edgeDy()[outerOuterLowerModuleIndex];
      dBetaROut = (alpaka::math::sqrt(acc, highEdgeX_OutUp * highEdgeX_OutUp + highEdgeY_OutUp * highEdgeY_OutUp) -
                   alpaka::math::sqrt(acc, lowEdgeX_OutUp * lowEdgeX_OutUp + lowEdgeY_OutUp * lowEdgeY_OutUp)) *
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
    return dBeta * dBeta <= dBetaCut2 * dBetaCut2Scale;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletdBetaCutEEEE(TAcc const& acc,
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
                                                                const float ptCut,
                                                                const float dBetaCut2Scale = 1.f) {
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

    runDeltaBetaIterations(acc, betaIn, betaOut, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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
    return dBeta * dBeta <= dBetaCut2 * dBetaCut2Scale;
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletdBetaAlgoSelector(TAcc const& acc,
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
                                                                     const float ptCut,
                                                                     const float dBetaCut2Scale = 1.f) {
    short innerInnerLowerModuleSubdet = modules.subdets()[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modules.subdets()[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modules.subdets()[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modules.subdets()[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
        outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Barrel) {
      return runQuintupletdBetaCutBBBB(acc,
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
                                       ptCut,
                                       dBetaCut2Scale);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletdBetaCutBBEE(acc,
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
                                       ptCut,
                                       dBetaCut2Scale);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Barrel and
               outerInnerLowerModuleSubdet == Barrel and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletdBetaCutBBBB(acc,
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
                                       ptCut,
                                       dBetaCut2Scale);
    } else if (innerInnerLowerModuleSubdet == Barrel and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletdBetaCutBBEE(acc,
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
                                       ptCut,
                                       dBetaCut2Scale);
    } else if (innerInnerLowerModuleSubdet == Endcap and innerOuterLowerModuleSubdet == Endcap and
               outerInnerLowerModuleSubdet == Endcap and outerOuterLowerModuleSubdet == Endcap) {
      return runQuintupletdBetaCutEEEE(acc,
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
                                       ptCut,
                                       dBetaCut2Scale);
    }

    return false;
  }

  // Terms of the trig-free dBeta bound (rejectDBetaByBound) that depend only on the inner segment MD1 -> MD2.
  struct DBetaBoundInner {
    float x1 = 0.f, y1 = 0.f, rt1 = 0.f;
    float alphaIn = 0.f;    // dPhiChange of the inner segment
    float sdIn = 0.f;       // xy length of the inner segment
    float drtIn = 0.f;      // rt(MD2) - rt(MD1)
    float mulsScale = 0.f;  // kMulsInGeV^2 * r3(MD1) / rt(MD1)
    bool barrel = false;    // MD1 and MD2 on barrel modules
  };

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE DBetaBoundInner makeDBetaBoundInner(TAcc const& acc,
                                                                     ModulesConst modules,
                                                                     MiniDoubletsConst mds,
                                                                     SegmentsConst segments,
                                                                     TripletsConst triplets,
                                                                     unsigned int innerTripletIndex) {
    DBetaBoundInner inner;
    const unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    const unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    const unsigned int secondMDIndex = segments.mdIndices()[firstSegmentIndex][1];
    inner.barrel = modules.subdets()[triplets.lowerModuleIndices()[innerTripletIndex][0]] == Barrel and
                   modules.subdets()[triplets.lowerModuleIndices()[innerTripletIndex][1]] == Barrel;
    inner.x1 = mds.anchorX()[firstMDIndex];
    inner.y1 = mds.anchorY()[firstMDIndex];
    inner.rt1 = mds.anchorRt()[firstMDIndex];
    inner.alphaIn = __H2F(segments.dPhiChanges()[firstSegmentIndex]);
    const float segmentX = mds.anchorX()[secondMDIndex] - inner.x1;
    const float segmentY = mds.anchorY()[secondMDIndex] - inner.y1;
    inner.sdIn = alpaka::math::sqrt(acc, segmentX * segmentX + segmentY * segmentY);
    inner.drtIn = mds.anchorRt()[secondMDIndex] - inner.rt1;
    const float z1 = mds.anchorZ()[firstMDIndex];
    inner.mulsScale = (kMulsInGeV * kMulsInGeV) * alpaka::math::sqrt(acc, z1 * z1 + inner.rt1 * inner.rt1) / inner.rt1;
    return inner;
  }

  // Trig-free sufficient reject of runQuintupletdBetaCutBBBB if its last module is not a 2S endcap module (no RH, lum,
  // ROut terms): bounds the raw angles, the runDeltaBetaIterations correction and the cut; false = run the exact cut.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool rejectDBetaByBound(TAcc const& acc,
                                                         DBetaBoundInner const& inner,
                                                         ModulesConst modules,
                                                         MiniDoubletsConst mds,
                                                         uint16_t outerInnerLowerModuleIndex,
                                                         uint16_t outerOuterLowerModuleIndex,
                                                         unsigned int thirdMDIndex,
                                                         unsigned int fourthMDIndex,
                                                         float alphaOut) {
    const short lastSubdet = modules.subdets()[outerOuterLowerModuleIndex];
    if (!inner.barrel or modules.subdets()[outerInnerLowerModuleIndex] != Barrel or
        !(lastSubdet == Barrel or (lastSubdet == Endcap and modules.moduleType()[outerOuterLowerModuleIndex] != TwoS)))
      return false;
    const float x4 = mds.anchorX()[fourthMDIndex], y4 = mds.anchorY()[fourthMDIndex];
    const float axisX = x4 - inner.x1, axisY = y4 - inner.y1;
    // tan u and tan v share one cross product; |tan| < 2 keeps their rounding far below the angle margin
    const float cross = inner.x1 * y4 - inner.y1 * x4;
    const float dotIn = inner.x1 * axisX + inner.y1 * axisY, dotOut = x4 * axisX + y4 * axisY;
    const float halfAbsCross = 0.5f * alpaka::math::abs(acc, cross);
    if (!(dotIn > halfAbsCross and dotOut > halfAbsCross))
      return false;
    const float tanIn = cross / dotIn, tanOut = cross / dotOut;
    // atan(t) lies between t - t^3/3 and t; 1e-5 rad per angle covers the stored anchor phi and atan2 rounding
    constexpr float kAngleMargin = 1e-5f;
    const float cubeIn = tanIn * tanIn * tanIn * (1.f / 3.f), cubeOut = tanOut * tanOut * tanOut * (1.f / 3.f);
    const float center = inner.alphaIn + alphaOut - tanIn - tanOut;
    const float cubes = cubeIn + cubeOut;  // tanIn and tanOut have the sign of cross
    const float low = center + (cubes < 0.f ? cubes : 0.f) - 2.f * kAngleMargin;
    const float high = center + (cubes > 0.f ? cubes : 0.f) + 2.f * kAngleMargin;
    const float distance = low > 0.f ? low : -high;  // distance of 0 from [low, high] when positive
    if (!(distance > 0.f))
      return false;
    const float betaInMax =
        alpaka::math::abs(acc, inner.alphaIn - tanIn) + alpaka::math::abs(acc, cubeIn) + kAngleMargin;
    const float betaOutMax = alpaka::math::abs(acc, tanOut - alphaOut) + alpaka::math::abs(acc, cubeOut) + kAngleMargin;
    const float betaMax = betaInMax > betaOutMax ? betaInMax : betaOutMax;
    // The iterations move dBeta by +-(asin(aIn) - asin(aOut)), a = min(sd |sin t| / drt, kSinAlphaMax) for one angle
    // |t| <= betaMax + asin(a0); asin(a) <= a + a^3 on [0, 1] and asin'(a) <= 1 / (1 - a^2).
    const float x3 = mds.anchorX()[thirdMDIndex], y3 = mds.anchorY()[thirdMDIndex];
    const float sdOut = alpaka::math::sqrt(acc, (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3));
    const float sdMax = inner.sdIn > sdOut ? inner.sdIn : sdOut;
    const float invDrt = 1.f / alpaka::math::sqrt(acc, axisX * axisX + axisY * axisY);
    const float sinT0 = betaMax < 1.f ? betaMax : 1.f;
    const float scaled0 = sdMax * sinT0 * invDrt;
    const float a0 = scaled0 < kSinAlphaMax ? scaled0 : kSinAlphaMax;
    const float angleT = betaMax + a0 + a0 * a0 * a0;
    const float sinT = angleT < 1.f ? angleT : 1.f;
    const float scaled1 = sdMax * sinT * invDrt;
    const float a1 = scaled1 < kSinAlphaMax ? scaled1 : kSinAlphaMax;
    const float deltaMax =
        alpaka::math::abs(acc, inner.sdIn - sdOut) * sinT * invDrt / (1.f - a1 * a1) * 1.0001f + 1e-6f;
    const float lower = distance - deltaMax;
    if (!(lower > 0.f))
      return false;
    // The final |betaAv| <= betaMax + asin(a1) bounds 1 / min(|pt_beta|, kPt_betaMax) from above
    const float angleAv = betaMax + a1 + a1 * a1 * a1;
    const float sinAv = angleAv < 1.f ? angleAv : 1.f;
    const float invPtBound = sinAv * invDrt * (1.f / k2Rinv1GeVf);
    const float invPt = invPtBound > 1.f / kPt_betaMax ? invPtBound : 1.f / kPt_betaMax;
    const float rt3 = mds.anchorRt()[thirdMDIndex];
    const float thetaMuls2 = inner.mulsScale * (0.1f + 0.2f * (rt3 - inner.rt1) / 50.f);
    const float mulsTerm = thetaMuls2 > 0.f ? thetaMuls2 * 16.f * invPt * invPt : 0.f;
    // The cut with dBetaRes = 0.02 / resDen, both sides multiplied by resDen^2
    const float sdOutDr = mds.anchorRt()[fourthMDIndex] - rt3;
    const float resDen = sdOutDr < inner.drtIn ? sdOutDr : inner.drtIn;
    const float resDen2 = resDen * resDen;
    return lower * lower * resDen2 > (0.0008f + mulsTerm * resDen2) * 1.0001f;
  }

  // The two dBeta cuts of runQuintupletSelection. The first depends only on the inner triplet and the first segment of
  // the outer triplet (thirdSegmentIndex), so the builders evaluate it once per (inner triplet, segment) key.
  // Key k < kT5DBetaMaskBits of an inner triplet records its first-cut decision in a per-triplet bit mask.
  using T5KeyMask = unsigned long long;
  constexpr unsigned int kT5DBetaMaskBits = 64;
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passQuintupletDBeta1(TAcc const& acc,
                                                           ModulesConst modules,
                                                           MiniDoubletsConst mds,
                                                           SegmentsConst segments,
                                                           TripletsConst triplets,
                                                           uint16_t lowerModuleIndex1,
                                                           unsigned int innerTripletIndex,
                                                           unsigned int thirdSegmentIndex,
                                                           float& dBeta1,
                                                           const float ptCut,
                                                           const float dBetaCut2Scale = 1.f) {
    const uint16_t lowerModuleIndex2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
    const uint16_t lowerModuleIndex3 = triplets.lowerModuleIndices()[innerTripletIndex][2];
    const uint16_t lowerModuleIndex4 = segments.outerLowerModuleIndices()[thirdSegmentIndex];
    const unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    const unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    const unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    const unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    const unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    const unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    return runQuintupletdBetaAlgoSelector(acc,
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
                                          dBeta1,
                                          ptCut,
                                          dBetaCut2Scale);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passQuintupletDBeta2(TAcc const& acc,
                                                           ModulesConst modules,
                                                           MiniDoubletsConst mds,
                                                           SegmentsConst segments,
                                                           TripletsConst triplets,
                                                           uint16_t lowerModuleIndex1,
                                                           unsigned int innerTripletIndex,
                                                           unsigned int outerTripletIndex,
                                                           float& dBeta2,
                                                           const float ptCut,
                                                           const float dBetaCut2Scale = 1.f) {
    const uint16_t lowerModuleIndex2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
    const uint16_t lowerModuleIndex4 = triplets.lowerModuleIndices()[outerTripletIndex][1];
    const uint16_t lowerModuleIndex5 = triplets.lowerModuleIndices()[outerTripletIndex][2];
    const unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    const unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    const unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    const unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];
    const unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    const unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    const unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    const unsigned int fifthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];
    return runQuintupletdBetaAlgoSelector(acc,
                                          modules,
                                          mds,
                                          segments,
                                          lowerModuleIndex1,
                                          lowerModuleIndex2,
                                          lowerModuleIndex4,
                                          lowerModuleIndex5,
                                          firstSegmentIndex,
                                          fourthSegmentIndex,
                                          firstMDIndex,
                                          secondMDIndex,
                                          fourthMDIndex,
                                          fifthMDIndex,
                                          dBeta2,
                                          ptCut,
                                          dBetaCut2Scale);
  }

  // passQuintupletDBeta2 depends only on (first inner, last outer segment): memo[last outer segment] keeps the decision
  // for the last first segment it was evaluated with, as firstSegment << 1 | pass (0xFFFFFFFF = empty).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passQuintupletDBeta2Memo(TAcc const& acc,
                                                               ModulesConst modules,
                                                               MiniDoubletsConst mds,
                                                               SegmentsConst segments,
                                                               TripletsConst triplets,
                                                               uint16_t lowerModuleIndex1,
                                                               unsigned int innerTripletIndex,
                                                               unsigned int outerTripletIndex,
                                                               const float ptCut,
                                                               unsigned int* __restrict__ dBeta2Memo,
                                                               DBetaBoundInner const& boundInner) {
    const unsigned int tag = triplets.segmentIndices()[innerTripletIndex][0] << 1;
    const unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];
    const unsigned int memo = dBeta2Memo[fourthSegmentIndex];
    if ((memo & ~1u) == tag)
      return memo & 1u;
    if constexpr (cms::alpakatools::requires_single_thread_per_block_v<TAcc>) {
      if (rejectDBetaByBound(acc,
                             boundInner,
                             modules,
                             mds,
                             triplets.lowerModuleIndices()[outerTripletIndex][1],
                             triplets.lowerModuleIndices()[outerTripletIndex][2],
                             segments.mdIndices()[fourthSegmentIndex][0],
                             segments.mdIndices()[fourthSegmentIndex][1],
                             segments.dPhiChangeOuts()[fourthSegmentIndex])) {
        dBeta2Memo[fourthSegmentIndex] = tag;
        return false;
      }
    }
    float dBeta2;
    const bool pass = passQuintupletDBeta2(
        acc, modules, mds, segments, triplets, lowerModuleIndex1, innerTripletIndex, outerTripletIndex, dBeta2, ptCut);
    dBeta2Memo[fourthSegmentIndex] = tag | (pass ? 1u : 0u);
    return pass;
  }

  // Radius of the circle through the anchor hits of MDs 2, 3, 4 (inner triplet and the outer triplet's first segment).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeT5BridgeRadius(TAcc const& acc,
                                                             MiniDoubletsConst mds,
                                                             SegmentsConst segments,
                                                             TripletsConst triplets,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int thirdSegmentIndex) {
    const unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    const unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    const unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    const unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    return std::get<0>(computeRadiusFromThreeAnchorHits(acc,
                                                        mds.anchorX()[secondMDIndex],
                                                        mds.anchorY()[secondMDIndex],
                                                        mds.anchorX()[thirdMDIndex],
                                                        mds.anchorY()[thirdMDIndex],
                                                        mds.anchorX()[fourthMDIndex],
                                                        mds.anchorY()[fourthMDIndex]));
  }

  // T5/T4 DNN inputs over the N MDs: MD-direction log-likelihood (mean, max; module frame, circle through anchors
  // iA,iB,iC), local density (T3s leaving MD iMid and MD 0, MDs in the first module) and dcaXY of that circle.
  // Not inlined on ROCm: inlining it into the counting kernels crashes the gfx90a register allocator (ROCm 7.2).
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define LST_DNN_FEATURES_INLINE [[gnu::noinline]]
#else
#define LST_DNN_FEATURES_INLINE ALPAKA_FN_INLINE
#endif
  template <int N, int iA, int iB, int iC, int iMid, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC LST_DNN_FEATURES_INLINE
      dnn::t5dnn::ExtraFeatures computeDnnFeatures(TAcc const& acc,
                                                   ModulesConst modules,
                                                   MiniDoubletsConst mds,
                                                   MiniDoubletsOccupancyConst mdOccupancy,
                                                   TripletsRangesConst tripletsRangesByMD,
                                                   const uint16_t (&lm)[N],
                                                   const unsigned int (&md)[N]) {
    float ax[N], ay[N], az[N];
    for (int i = 0; i < N; ++i) {
      ax[i] = mds.anchorX()[md[i]];
      ay[i] = mds.anchorY()[md[i]];
      az[i] = mds.anchorZ()[md[i]];
    }
    const auto circle = computeRadiusFromThreeAnchorHits(acc, ax[iA], ay[iA], ax[iB], ay[iB], ax[iC], ay[iC]);
    const float cr = std::get<0>(circle), cx = std::get<1>(circle), cy = std::get<2>(circle);
    const float chx = ax[iC] - ax[iA], chy = ay[iC] - ay[iA];
    const float chord = alpaka::math::sqrt(acc, chx * chx + chy * chy);
    // Same radius as cr, measured from the first anchor: numerically safer than cr for nearly straight tracks.
    const float radiusAtFirstAnchor =
        alpaka::math::sqrt(acc, (ax[iA] - cx) * (ax[iA] - cx) + (ay[iA] - cy) * (ay[iA] - cy));
    float arc = chord;
    if (edm::isFinite(radiusAtFirstAnchor) && radiusAtFirstAnchor > 0.f)
      arc = 2.f * radiusAtFirstAnchor *
            alpaka::math::asin(acc, alpaka::math::min(acc, chord / (2.f * radiusAtFirstAnchor), 1.f));
    const float cotTheta = (az[iC] - az[iA]) / arc;
    float sumW = 0.f, maxW = 0.f, maxPull = 0.f;
    for (int i = 0; i < N; ++i) {
      const uint16_t lowerModuleIndex = lm[i];
      float tx = cy - ay[i], ty = ax[i] - cx;
      const float tn = alpaka::math::sqrt(acc, tx * tx + ty * ty);
      tx /= tn;
      ty /= tn;
      if (!edm::isFinite(tx) || !edm::isFinite(ty)) {
        tx = chx / chord;
        ty = chy / chord;
      }
      if (tx * chx + ty * chy < 0.f) {
        tx = -tx;
        ty = -ty;
      }
      // t = unit tangent of the circle at the MD anchor (plus cotTheta along z); in the module frame,
      // u = direction across the strips in the sensor plane, n = sensor normal (tilted barrel via drdz, endcap = z).
      float ux, uy, nx, ny, nz;
      if (modules.subdets()[lowerModuleIndex] == Barrel) {
        const float cphi = alpaka::math::cos(acc, modules.phi()[lowerModuleIndex]);
        const float sphi = alpaka::math::sin(acc, modules.phi()[lowerModuleIndex]);
        ux = -sphi;
        uy = cphi;
        nx = cphi;
        ny = sphi;
        nz = ((az[i] > 0.f) - (az[i] < 0.f)) * modules.drdzs()[lowerModuleIndex];
      } else {
        const float slope = modules.dxdys()[lowerModuleIndex];
        ux = 0.f;
        uy = 1.f;
        if (edm::isFinite(slope)) {
          ux = 1.f / alpaka::math::sqrt(acc, 1.f + slope * slope);
          uy = slope * ux;
        }
        nx = 0.f;
        ny = 0.f;
        nz = 1.f;
      }
      const float dx = mds.outerX()[md[i]] - ax[i];
      const float dy = mds.outerY()[md[i]] - ay[i];
      const float dz = mds.outerZ()[md[i]] - az[i];
      const float vu = tx * ux + ty * uy;
      const float vn = tx * nx + ty * ny + cotTheta * nz;
      const float residual = (dx * ux + dy * uy) - vu * (dx * nx + dy * ny + dz * nz) / vn;
      const float width = (modules.moduleType()[lowerModuleIndex] == PS) ? kWidthPS : kWidth2S;
      const float pull = alpaka::math::abs(acc, residual) / (width * 0.40824829f);
      const float mdDirW = -alpaka::math::log(acc, alpaka::math::max(acc, 1.f - pull * 0.40824829f, 0.05f));
      sumW += mdDirW;
      maxW = alpaka::math::max(acc, maxW, mdDirW);
      maxPull = alpaka::math::max(acc, maxPull, pull);
    }
    dnn::t5dnn::ExtraFeatures feat;
    feat.mdDirMeanW = sumW / N;
    feat.mdDirMaxW = maxW;
    feat.nT3OutMid = tripletsRangesByMD.n()[md[iMid]];
    feat.nT3OutFirst = tripletsRangesByMD.n()[md[0]];
    feat.nMDFirstMod = mdOccupancy.nMDs()[lm[0]];
    feat.dcaXY = alpaka::math::abs(acc, alpaka::math::sqrt(acc, cx * cx + cy * cy) - cr);
    feat.mdDirMaxPull = maxPull;
    return feat;
  }

  // The cuts of the T5 algorithm after the two dBeta cuts (bridgeRadius and the first cut come from the
  // (inner triplet, first outer segment) key); computeQuintupletFits adds the embedding and fits of a selected T5.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletSelectionOuter(TAcc const& acc,
                                                                  ModulesConst modules,
                                                                  MiniDoubletsConst mds,
                                                                  SegmentsConst segments,
                                                                  TripletsConst triplets,
                                                                  MiniDoubletsOccupancyConst mdOccupancy,
                                                                  TripletsRangesConst tripletsRangesByMD,
                                                                  uint16_t lowerModuleIndex1,
                                                                  unsigned int innerTripletIndex,
                                                                  unsigned int outerTripletIndex,
                                                                  float bridgeRadius,
                                                                  T5RZInnerTerms const& rzInner,
                                                                  float& innerRadius,
                                                                  float& outerRadius,
                                                                  float& rzChiSquared,
                                                                  float& dnnScore) {
    const uint16_t lowerModuleIndex2 = triplets.lowerModuleIndices()[innerTripletIndex][1];
    const uint16_t lowerModuleIndex3 = triplets.lowerModuleIndices()[innerTripletIndex][2];
    const uint16_t lowerModuleIndex4 = triplets.lowerModuleIndices()[outerTripletIndex][1];
    const uint16_t lowerModuleIndex5 = triplets.lowerModuleIndices()[outerTripletIndex][2];

    unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];

    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    unsigned int fifthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];

    outerRadius = triplets.radius()[outerTripletIndex];
    innerRadius = triplets.radius()[innerTripletIndex];

    float inner_pt = 2 * k2Rinv1GeVf * innerRadius;

    if (not passT5RZConstraint(acc,
                               modules,
                               mds,
                               rzInner,
                               firstMDIndex,
                               secondMDIndex,
                               thirdMDIndex,
                               fourthMDIndex,
                               fifthMDIndex,
                               lowerModuleIndex1,
                               lowerModuleIndex2,
                               lowerModuleIndex3,
                               lowerModuleIndex4,
                               lowerModuleIndex5,
                               rzChiSquared,
                               inner_pt))
      return false;

    const uint16_t t5Lm[Params_T5::kBaseLayers] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};
    const unsigned int t5Md[Params_T5::kBaseLayers] = {
        firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex};
    const auto t5Feat = computeDnnFeatures<Params_T5::kBaseLayers, 0, 2, 4, 2>(
        acc, modules, mds, mdOccupancy, tripletsRangesByMD, t5Lm, t5Md);
    float dnnOutput[dnn::t5dnn::kOutputFeatures];
    const bool inference = lst::t5dnn::runInference(acc,
                                                    mds,
                                                    firstMDIndex,
                                                    secondMDIndex,
                                                    thirdMDIndex,
                                                    fourthMDIndex,
                                                    fifthMDIndex,
                                                    innerRadius,
                                                    outerRadius,
                                                    bridgeRadius,
                                                    triplets.fakeScore()[innerTripletIndex],
                                                    triplets.promptScore()[innerTripletIndex],
                                                    triplets.displacedScore()[innerTripletIndex],
                                                    triplets.fakeScore()[outerTripletIndex],
                                                    triplets.promptScore()[outerTripletIndex],
                                                    triplets.displacedScore()[outerTripletIndex],
                                                    t5Feat,
                                                    dnnOutput);
    dnnScore = 1.f - dnnOutput[0];
    if (!inference)  // T5-building cut
      return false;
    return true;
  }

  // Every cut of the T5 algorithm for one (inner, outer) triplet pair.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuintupletSelection(TAcc const& acc,
                                                             ModulesConst modules,
                                                             MiniDoubletsConst mds,
                                                             SegmentsConst segments,
                                                             TripletsConst triplets,
                                                             MiniDoubletsOccupancyConst mdOccupancy,
                                                             TripletsRangesConst tripletsRangesByMD,
                                                             uint16_t lowerModuleIndex1,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int outerTripletIndex,
                                                             float& innerRadius,
                                                             float& outerRadius,
                                                             float& bridgeRadius,
                                                             float& rzChiSquared,
                                                             float& dBeta1,
                                                             float& dBeta2,
                                                             float& dnnScore,
                                                             const float ptCut) {
    const unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    bridgeRadius = computeT5BridgeRadius(acc, mds, segments, triplets, innerTripletIndex, thirdSegmentIndex);
    if (not passQuintupletDBeta1(acc,
                                 modules,
                                 mds,
                                 segments,
                                 triplets,
                                 lowerModuleIndex1,
                                 innerTripletIndex,
                                 thirdSegmentIndex,
                                 dBeta1,
                                 ptCut))
      return false;
    if (not passQuintupletDBeta2(acc,
                                 modules,
                                 mds,
                                 segments,
                                 triplets,
                                 lowerModuleIndex1,
                                 innerTripletIndex,
                                 outerTripletIndex,
                                 dBeta2,
                                 ptCut))
      return false;
    return runQuintupletSelectionOuter(acc,
                                       modules,
                                       mds,
                                       segments,
                                       triplets,
                                       mdOccupancy,
                                       tripletsRangesByMD,
                                       lowerModuleIndex1,
                                       innerTripletIndex,
                                       outerTripletIndex,
                                       bridgeRadius,
                                       computeT5RZInnerTerms(acc, modules, mds, segments, triplets, innerTripletIndex),
                                       innerRadius,
                                       outerRadius,
                                       rzChiSquared,
                                       dnnScore);
  }

  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeQuintupletFits(TAcc const& acc,
                                                            ModulesConst modules,
                                                            MiniDoubletsConst mds,
                                                            SegmentsConst segments,
                                                            TripletsConst triplets,
                                                            uint16_t lowerModuleIndex1,
                                                            uint16_t lowerModuleIndex2,
                                                            uint16_t lowerModuleIndex3,
                                                            uint16_t lowerModuleIndex4,
                                                            uint16_t lowerModuleIndex5,
                                                            unsigned int innerTripletIndex,
                                                            unsigned int outerTripletIndex,
                                                            float innerRadius,
                                                            float outerRadius,
                                                            float bridgeRadius,
                                                            float& regressionCenterX,
                                                            float& regressionCenterY,
                                                            float& regressionRadius,
                                                            float& chiSquared,
                                                            float (&t5Embed)[Params_T5::kEmbed]) {
    unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    unsigned int secondSegmentIndex = triplets.segmentIndices()[innerTripletIndex][1];
    unsigned int thirdSegmentIndex = triplets.segmentIndices()[outerTripletIndex][0];
    unsigned int fourthSegmentIndex = triplets.segmentIndices()[outerTripletIndex][1];

    unsigned int firstMDIndex = segments.mdIndices()[firstSegmentIndex][0];
    unsigned int secondMDIndex = segments.mdIndices()[secondSegmentIndex][0];
    unsigned int thirdMDIndex = segments.mdIndices()[secondSegmentIndex][1];
    unsigned int fourthMDIndex = segments.mdIndices()[thirdSegmentIndex][1];
    unsigned int fifthMDIndex = segments.mdIndices()[fourthSegmentIndex][1];

    float x1 = mds.anchorX()[firstMDIndex];
    float x2 = mds.anchorX()[secondMDIndex];
    float x3 = mds.anchorX()[thirdMDIndex];
    float x4 = mds.anchorX()[fourthMDIndex];
    float x5 = mds.anchorX()[fifthMDIndex];

    float y1 = mds.anchorY()[firstMDIndex];
    float y2 = mds.anchorY()[secondMDIndex];
    float y3 = mds.anchorY()[thirdMDIndex];
    float y4 = mds.anchorY()[fourthMDIndex];
    float y5 = mds.anchorY()[fifthMDIndex];

    lst::t5embdnn::runEmbed(acc,
                            mds,
                            firstMDIndex,
                            secondMDIndex,
                            thirdMDIndex,
                            fourthMDIndex,
                            fifthMDIndex,
                            innerRadius,
                            outerRadius,
                            bridgeRadius,
                            triplets.fakeScore()[innerTripletIndex],
                            triplets.promptScore()[innerTripletIndex],
                            triplets.displacedScore()[innerTripletIndex],
                            triplets.fakeScore()[outerTripletIndex],
                            triplets.promptScore()[outerTripletIndex],
                            triplets.displacedScore()[outerTripletIndex],
                            t5Embed);

    // 5 categories for sigmas
    float sigmas2[5], delta1[5], delta2[5], slopes[5];
    bool isFlat[5];

    float xVec[] = {x1, x2, x3, x4, x5};
    float yVec[] = {y1, y2, y3, y4, y5};
    const uint16_t lowerModuleIndices[] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    computeSigmasForRegression(acc, modules, lowerModuleIndices, delta1, delta2, slopes, isFlat);
    regressionRadius = computeRadiusUsingRegression(acc,
                                                    Params_T5::kBaseLayers,
                                                    xVec,
                                                    yVec,
                                                    delta1,
                                                    delta2,
                                                    slopes,
                                                    isFlat,
                                                    regressionCenterX,
                                                    regressionCenterY,
                                                    sigmas2,
                                                    chiSquared);
  }

  // Stores a selected (inner, outer) triplet pair; FinalizeQuintuplets writes the full quintuplet.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuintupletCandidate(TAcc const& acc,
                                                             MiniDoubletsT5CountsConst mdT5Counts,
                                                             SegmentsConst segments,
                                                             Triplets triplets,
                                                             QuintupletsLoose quintupletsLoose,
                                                             QuintupletsOccupancy quintupletsOccupancy,
                                                             QuintupletsRanges quintupletsRangesByMD0,
                                                             QuintupletsRanges quintupletsRangesByMD1,
                                                             ObjectRanges ranges,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int outerTripletIndex,
                                                             uint16_t lowerModule1,
                                                             float bridgeRadius,
                                                             float dnnScore,
                                                             [[maybe_unused]] float rzChiSquared,
                                                             [[maybe_unused]] float dBeta1,
                                                             [[maybe_unused]] float dBeta2) {
    const auto& mdIndices = segments.mdIndices();
    const auto& segIdx = triplets.segmentIndices();
    int totOccupancyQuintuplets = alpaka::atomicAdd(
        acc, &quintupletsOccupancy.totOccupancyQuintuplets()[lowerModule1], 1u, alpaka::hierarchy::Blocks{});
    if (totOccupancyQuintuplets >= ranges.quintupletModuleOccupancy()[lowerModule1]) {
      // A module at the fixed cap kNQuintupletThreshold drops by design; anything else is a counting shortfall.
      alpaka::atomicAdd(acc,
                        ranges.quintupletModuleOccupancy()[lowerModule1] == kNQuintupletThreshold
                            ? &ranges.nQuintupletCapDrops()
                            : &ranges.nQuintupletOverflows(),
                        1u,
                        alpaka::hierarchy::Blocks{});
#ifdef WARNINGS
      printf("Quintuplet excess alert! Module index = %d, Occupancy = %d\n", lowerModule1, totOccupancyQuintuplets);
#endif
    } else {
      int quintupletModuleIndex =
          alpaka::atomicAdd(acc, &quintupletsOccupancy.nQuintuplets()[lowerModule1], 1u, alpaka::hierarchy::Blocks{});
      unsigned int quintupletIndex = ranges.quintupletModuleIndices()[lowerModule1] + quintupletModuleIndex;
      auto const ls0Index = segIdx[innerTripletIndex][0];
      auto const md0Index = mdIndices[ls0Index][0];
      auto const quintupletByMD0Local =
          alpaka::atomicAdd(acc, &quintupletsRangesByMD0.n()[md0Index], 1u, alpaka::hierarchy::Blocks{});
      auto quintupletByMD0Index = quintupletByMD0Local + quintupletsRangesByMD0.offset()[md0Index];
      if (quintupletByMD0Local >= mdT5Counts.connectedT5s0Max()[md0Index]) {
        alpaka::atomicSub(acc, &quintupletsRangesByMD0.n()[md0Index], 1u, alpaka::hierarchy::Blocks{});
        quintupletByMD0Index = kInvalidU32Idx;
        alpaka::atomicAdd(acc, &ranges.nT5byMDOverflows(), 1u, alpaka::hierarchy::Blocks{});
      }
      auto const md1Index = mdIndices[ls0Index][1];
      auto const quintupletByMD1Local =
          alpaka::atomicAdd(acc, &quintupletsRangesByMD1.n()[md1Index], 1u, alpaka::hierarchy::Blocks{});
      auto quintupletByMD1Index = quintupletByMD1Local + quintupletsRangesByMD1.offset()[md1Index];
      if (quintupletByMD1Local >= mdT5Counts.connectedT5s1Max()[md1Index]) {
        alpaka::atomicSub(acc, &quintupletsRangesByMD1.n()[md1Index], 1u, alpaka::hierarchy::Blocks{});
        quintupletByMD1Index = kInvalidU32Idx;
        alpaka::atomicAdd(acc, &ranges.nT5byMDOverflows(), 1u, alpaka::hierarchy::Blocks{});
      }

      // The fits and the full quintuplet are written by FinalizeQuintuplets into the compact collection.
      quintupletsLoose.tripletIndices()[quintupletIndex][0] = innerTripletIndex;
      quintupletsLoose.tripletIndices()[quintupletIndex][1] = outerTripletIndex;
      quintupletsLoose.byMDIndices()[quintupletIndex][0] = quintupletByMD0Index;
      quintupletsLoose.byMDIndices()[quintupletIndex][1] = quintupletByMD1Index;
      quintupletsLoose.bridgeRadius()[quintupletIndex] = bridgeRadius;
      quintupletsLoose.dnnScore()[quintupletIndex] = dnnScore;
#ifdef CUT_VALUE_DEBUG
      quintupletsLoose.rzChiSquared()[quintupletIndex] = rzChiSquared;
      quintupletsLoose.dBeta1()[quintupletIndex] = dBeta1;
      quintupletsLoose.dBeta2()[quintupletIndex] = dBeta2;
#endif

      triplets.partOfT5()[innerTripletIndex] = true;
      triplets.partOfT5()[outerTripletIndex] = true;
    }
  }

  // Writes the full quintuplet of every selected pair (in module order, keeping the order within each module) to
  // the exactly sized collection, and the per-module occupancy; fills the by-MD lists with the compact indices.
  struct FinalizeQuintuplets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  TripletsConst triplets,
                                  MiniDoubletsOccupancyConst mdOccupancy,
                                  TripletsRangesConst tripletsRangesByMD,
                                  QuintupletsLooseConst quintupletsLoose,
                                  QuintupletsOccupancyConst looseOccupancy,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancy quintupletsOccupancy,
                                  QuintupletsByMD quintupletsByMD0,
                                  QuintupletsByMD quintupletsByMD1,
                                  ObjectRangesConst ranges,
                                  int const* __restrict__ compactIndices,
                                  unsigned int nEligibleModules) const {
      for (unsigned int m : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        quintupletsOccupancy[m] = looseOccupancy[m];
      }
      const auto& mdIndices = segments.mdIndices();
      const auto& segIdx = triplets.segmentIndices();
      const auto& lmIdx = triplets.lowerModuleIndices();
      for (unsigned int iter : cms::alpakatools::independent_groups(acc, nEligibleModules)) {
        const uint16_t lowerModule1 = ranges.indicesOfEligibleT5Modules()[iter];
        const int looseOffset = ranges.quintupletModuleIndices()[lowerModule1];
        if (looseOffset == -1)
          continue;
        const int compactOffset = compactIndices[lowerModule1];
        const auto layer = modules.layers()[lowerModule1];
        //get upper segment to be in second layer
        //assumes only 1 and 2 are possible here (isValidQuintRegion)
        const short layer2_adjustment = layer == 1 ? 1 : 0;
        for (unsigned int k :
             cms::alpakatools::independent_group_elements(acc, looseOccupancy.nQuintuplets()[lowerModule1])) {
          const unsigned int looseIndex = looseOffset + k;
          const unsigned int innerTripletIndex = quintupletsLoose.tripletIndices()[looseIndex][0];
          const unsigned int outerTripletIndex = quintupletsLoose.tripletIndices()[looseIndex][1];
          const uint16_t lowerModule2 = lmIdx[innerTripletIndex][1];
          const uint16_t lowerModule3 = lmIdx[innerTripletIndex][2];
          const uint16_t lowerModule4 = lmIdx[outerTripletIndex][1];
          const uint16_t lowerModule5 = lmIdx[outerTripletIndex][2];
          const float innerRadius = triplets.radius()[innerTripletIndex];
          const float outerRadius = triplets.radius()[outerTripletIndex];
          const float bridgeRadius = quintupletsLoose.bridgeRadius()[looseIndex];

          float regressionCenterX, regressionCenterY, regressionRadius, chiSquared;
          float t5Embed[Params_T5::kEmbed] = {0.f};
          computeQuintupletFits(acc,
                                modules,
                                mds,
                                segments,
                                triplets,
                                lowerModule1,
                                lowerModule2,
                                lowerModule3,
                                lowerModule4,
                                lowerModule5,
                                innerTripletIndex,
                                outerTripletIndex,
                                innerRadius,
                                outerRadius,
                                bridgeRadius,
                                regressionCenterX,
                                regressionCenterY,
                                regressionRadius,
                                chiSquared,
                                t5Embed);

          float rzChiSquared = 0.f, dBeta1 = 0.f, dBeta2 = 0.f;
#ifdef CUT_VALUE_DEBUG
          rzChiSquared = quintupletsLoose.rzChiSquared()[looseIndex];
          dBeta1 = quintupletsLoose.dBeta1()[looseIndex];
          dBeta2 = quintupletsLoose.dBeta2()[looseIndex];
#endif
          auto const ls0Index = segIdx[innerTripletIndex][0];
          float phi = mds.anchorPhi()[mdIndices[ls0Index][layer2_adjustment]];
          float eta = mds.anchorEta()[mdIndices[ls0Index][layer2_adjustment]];
          addQuintupletToMemory(modules,
                                mds,
                                segments,
                                triplets,
                                quintuplets,
                                quintupletsByMD0,
                                quintupletsByMD1,
                                innerTripletIndex,
                                outerTripletIndex,
                                lowerModule1,
                                lowerModule2,
                                lowerModule3,
                                lowerModule4,
                                lowerModule5,
                                innerRadius,
                                bridgeRadius,
                                regressionCenterX,
                                regressionCenterY,
                                regressionRadius,
                                rzChiSquared,
                                chiSquared,
                                dBeta1,
                                dBeta2,
                                eta,
                                phi,
                                layer,
                                compactOffset + k,
                                quintupletsLoose.byMDIndices()[looseIndex][0],
                                quintupletsLoose.byMDIndices()[looseIndex][1],
                                t5Embed,
                                quintupletsLoose.dnnScore()[looseIndex]);
          quintuplets.partOfPT5()[compactOffset + k] = false;
#ifdef CUT_VALUE_DEBUG
          {
            const uint16_t t5Lm[Params_T5::kBaseLayers] = {
                lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5};
            const unsigned int t5Md[Params_T5::kBaseLayers] = {mdIndices[segIdx[innerTripletIndex][0]][0],
                                                               mdIndices[segIdx[innerTripletIndex][0]][1],
                                                               mdIndices[segIdx[innerTripletIndex][1]][1],
                                                               mdIndices[segIdx[outerTripletIndex][0]][1],
                                                               mdIndices[segIdx[outerTripletIndex][1]][1]};
            const auto t5Feat = computeDnnFeatures<Params_T5::kBaseLayers, 0, 2, 4, 2>(
                acc, modules, mds, mdOccupancy, tripletsRangesByMD, t5Lm, t5Md);
            quintuplets.mdDirMeanW()[compactOffset + k] = t5Feat.mdDirMeanW;
            quintuplets.mdDirMaxW()[compactOffset + k] = t5Feat.mdDirMaxW;
            quintuplets.nT3OutMid()[compactOffset + k] = t5Feat.nT3OutMid;
            quintuplets.nT3OutFirst()[compactOffset + k] = t5Feat.nT3OutFirst;
            quintuplets.nMDFirstMod()[compactOffset + k] = t5Feat.nMDFirstMod;
            quintuplets.dcaXY()[compactOffset + k] = t5Feat.dcaXY;
          }
#endif
        }
      }
    }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isValidQuintRegion(ModulesConst modules, uint16_t lowerModule) {
    const short layer = modules.layers()[lowerModule];
    const short subdet = modules.subdets()[lowerModule];
    // Quintuplets starting outside these regions are not built.
    return (subdet == Barrel && layer < 3) || (subdet == Endcap && layer <= 1);
  }

  // Keys of the T5 builders: the segments that carry triplets, listed by their inner MD (in ascending segment index on
  // the serial backend, so walking them and tripletsBySegment visits the triplets of an MD in tripletsByMD order).
  struct FillT5KeysByMD {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsOccupancyConst mdOccupancy,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  TripletsOccupancyConst tripletsOcc,
                                  TripletsRangesConst tripletsRangesBySegment,
                                  ObjectRangesConst ranges,
                                  unsigned int* __restrict__ nKeysByMD,
                                  unsigned int* __restrict__ keyOffsetByMD,
                                  unsigned int* __restrict__ keys) const {
      for (uint16_t lowerModule : cms::alpakatools::independent_groups(acc, modules.nLowerModules())) {
        if (tripletsOcc.nTriplets()[lowerModule] == 0)
          continue;
        const unsigned int firstSegment = ranges.segmentRanges()[lowerModule][0];
        const unsigned int nSegments = segmentsOccupancy.nSegments()[lowerModule];
        for (unsigned int i : cms::alpakatools::independent_group_elements(acc, nSegments)) {
          const unsigned int seg = firstSegment + i;
          if (tripletsRangesBySegment.n()[seg] > 0)
            alpaka::atomicAdd(acc, &nKeysByMD[segments.mdIndices()[seg][0]], 1u, alpaka::hierarchy::Threads{});
        }
        alpaka::syncBlockThreads(acc);
        if (cms::alpakatools::once_per_block(acc)) {
          unsigned int offset = firstSegment;
          const unsigned int firstMD = ranges.mdRanges()[lowerModule][0];
          const unsigned int nMDs = mdOccupancy.nMDs()[lowerModule];
          for (unsigned int md = firstMD; md < firstMD + nMDs; ++md) {
            keyOffsetByMD[md] = offset;
            offset += nKeysByMD[md];
            nKeysByMD[md] = 0;
          }
        }
        alpaka::syncBlockThreads(acc);
        for (unsigned int i : cms::alpakatools::independent_group_elements(acc, nSegments)) {
          const unsigned int seg = firstSegment + i;
          if (tripletsRangesBySegment.n()[seg] > 0) {
            const unsigned int md = segments.mdIndices()[seg][0];
            keys[keyOffsetByMD[md] + alpaka::atomicAdd(acc, &nKeysByMD[md], 1u, alpaka::hierarchy::Threads{})] = seg;
          }
        }
      }
    }
  };

  // The counting kernel runs over all triplets (a row of threads per inner triplet), not one block per module, so
  // that the few dense modules of a jet core spread over the device.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isDenseT5Pair(TripletsOccupancyConst tripletsOcc,
                                                    uint16_t lowerModule1,
                                                    uint16_t lowerModule3) {
    return tripletsOcc.nTriplets()[lowerModule1] >= kNTripletThreshold ||
           tripletsOcc.nTriplets()[lowerModule3] >= kNTripletThreshold;
  }

  // First dBeta cut of the first kT5DBetaMaskBits keys of every inner triplet, one thread per key.
  struct CountT5KeyCuts {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  TripletsConst triplets,
                                  unsigned int const* __restrict__ nKeysByMD,
                                  unsigned int const* __restrict__ keyOffsetByMD,
                                  unsigned int const* __restrict__ keys,
                                  const unsigned int nTriplets,
                                  const float ptCut,
                                  T5KeyMask* __restrict__ dBetaPassMask) const {
      // The atomicOr below with hierarchy::Threads{} requires one block in the x and z dimensions.
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1) &&
                        (alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[2] == 1));
      const auto& mdIndices = segments.mdIndices();
      const auto& segIdx = triplets.segmentIndices();
      const auto& lmIdx = triplets.lowerModuleIndices();

      for (unsigned int innerTripletIndex : cms::alpakatools::uniform_elements_y(acc, nTriplets)) {
        const uint16_t lowerModule1 = lmIdx[innerTripletIndex][0];
        if (!isValidQuintRegion(modules, lowerModule1))
          continue;
        const unsigned int secondMDOuter = mdIndices[segIdx[innerTripletIndex][1]][1];
        const unsigned int nKeysByMD3 = nKeysByMD[secondMDOuter];
        const unsigned int nKeys = nKeysByMD3 < kT5DBetaMaskBits ? nKeysByMD3 : kT5DBetaMaskBits;
        const unsigned int keyOffset = keyOffsetByMD[secondMDOuter];
        // Serial backend: a trig-free bound rejects most keys first (on a GPU it only adds divergence).
        DBetaBoundInner boundInner;
        if constexpr (cms::alpakatools::requires_single_thread_per_block_v<Acc3D>)
          boundInner = makeDBetaBoundInner(acc, modules, mds, segments, triplets, innerTripletIndex);
        for (unsigned int k : cms::alpakatools::uniform_elements_x(acc, nKeys)) {
          if constexpr (cms::alpakatools::requires_single_thread_per_block_v<Acc3D>) {
            const unsigned int keySegment = keys[keyOffset + k];
            if (rejectDBetaByBound(acc,
                                   boundInner,
                                   modules,
                                   mds,
                                   lmIdx[innerTripletIndex][2],
                                   segments.outerLowerModuleIndices()[keySegment],
                                   secondMDOuter,
                                   mdIndices[keySegment][1],
                                   segments.dPhiChangeOuts()[keySegment]))
              continue;
          }
          float dBeta1;
          if (passQuintupletDBeta1(acc,
                                   modules,
                                   mds,
                                   segments,
                                   triplets,
                                   lowerModule1,
                                   innerTripletIndex,
                                   keys[keyOffset + k],
                                   dBeta1,
                                   ptCut))
            alpaka::atomicOr(acc, &dBetaPassMask[innerTripletIndex], T5KeyMask(1) << k, alpaka::hierarchy::Threads{});
        }
      }
    }
  };

  // First dBeta cut of key `key` of an inner triplet: from the key-cut mask for the first kT5DBetaMaskBits keys.
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passQuintupletKey(TAcc const& acc,
                                                        ModulesConst modules,
                                                        MiniDoubletsConst mds,
                                                        SegmentsConst segments,
                                                        TripletsConst triplets,
                                                        uint16_t lowerModule1,
                                                        unsigned int innerTripletIndex,
                                                        unsigned int thirdSegmentIndex,
                                                        unsigned int key,
                                                        T5KeyMask passMask,
                                                        const float ptCut) {
    if (key < kT5DBetaMaskBits)
      return (passMask >> key) & 1u;
    float dBeta1;
    return passQuintupletDBeta1(
        acc, modules, mds, segments, triplets, lowerModule1, innerTripletIndex, thirdSegmentIndex, dBeta1, ptCut);
  }

  // Full T5 selection of one (inner, outer) triplet pair; a selected pair is counted per module and per MD and
  // recorded (the first `capacity`).
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void selectQuintupletPair(TAcc const& acc,
                                                           ModulesConst modules,
                                                           MiniDoubletsConst mds,
                                                           MiniDoubletsT5Counts mdT5Counts,
                                                           SegmentsConst segments,
                                                           TripletsConst triplets,
                                                           MiniDoubletsOccupancyConst mdOccupancy,
                                                           TripletsRangesConst tripletsRangesByMD,
                                                           uint16_t lowerModule1,
                                                           unsigned int innerTripletIndex,
                                                           unsigned int outerTripletIndex,
                                                           float bridgeRadius,
                                                           T5RZInnerTerms const& rzInner,
                                                           const float ptCut,
                                                           unsigned int* __restrict__ dBeta2Memo,
                                                           DBetaBoundInner const& boundInner,
                                                           unsigned int* __restrict__ moduleT5Count,
                                                           unsigned int* __restrict__ nSelected,
                                                           const unsigned int capacity,
                                                           unsigned int* __restrict__ selectedT3s,
                                                           float* __restrict__ selectedBridgeRadius,
                                                           float* __restrict__ selectedDnnScore) {
    if (!passQuintupletDBeta2Memo(acc,
                                  modules,
                                  mds,
                                  segments,
                                  triplets,
                                  lowerModule1,
                                  innerTripletIndex,
                                  outerTripletIndex,
                                  ptCut,
                                  dBeta2Memo,
                                  boundInner))
      return;
    float innerRadius, outerRadius, rzChi2, dnnScore;
    if (!runQuintupletSelectionOuter(acc,
                                     modules,
                                     mds,
                                     segments,
                                     triplets,
                                     mdOccupancy,
                                     tripletsRangesByMD,
                                     lowerModule1,
                                     innerTripletIndex,
                                     outerTripletIndex,
                                     bridgeRadius,
                                     rzInner,
                                     innerRadius,
                                     outerRadius,
                                     rzChi2,
                                     dnnScore))
      return;
    const unsigned int firstSegmentIndex = triplets.segmentIndices()[innerTripletIndex][0];
    alpaka::atomicAdd(acc, &moduleT5Count[lowerModule1], 1u, alpaka::hierarchy::Blocks{});
    alpaka::atomicAdd(acc,
                      &mdT5Counts.connectedT5s0Max()[segments.mdIndices()[firstSegmentIndex][0]],
                      1u,
                      alpaka::hierarchy::Blocks{});
    alpaka::atomicAdd(acc,
                      &mdT5Counts.connectedT5s1Max()[segments.mdIndices()[firstSegmentIndex][1]],
                      1u,
                      alpaka::hierarchy::Blocks{});
    const unsigned int slot = alpaka::atomicAdd(acc, nSelected, 1u, alpaka::hierarchy::Blocks{});
    if (slot < capacity) {
      selectedT3s[2 * slot] = innerTripletIndex;
      selectedT3s[2 * slot + 1] = outerTripletIndex;
      selectedBridgeRadius[slot] = bridgeRadius;
      selectedDnnScore[slot] = dnnScore;
    }
  }

  // Runs the full T5 selection once per candidate pair: counts the selected pairs per module and per MD and records
  // them (the first `capacity`; createQuintuplets re-runs this kernel with a larger list if more are selected).
  // On the GPU a block is one row of threads per inner triplet, which takes the keys in chunks: first one thread per
  // key (first dBeta cut, bridge radius), then one thread per (key, outer triplet) pair, so all threads select pairs.
  constexpr unsigned int kT5KeysPerChunk = 32;
  struct CountTripletConnections {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  MiniDoubletsConst mds,
                                  MiniDoubletsT5Counts mdT5Counts,
                                  SegmentsConst segments,
                                  TripletsConst triplets,
                                  TripletsBySegmentConst tripletsBySegment,
                                  TripletsRangesConst tripletsRangesBySegment,
                                  TripletsRangesConst tripletsRangesByMD,
                                  MiniDoubletsOccupancyConst mdOccupancy,
                                  unsigned int const* __restrict__ nKeysByMD,
                                  unsigned int const* __restrict__ keyOffsetByMD,
                                  unsigned int const* __restrict__ keys,
                                  const unsigned int nTriplets,
                                  const float ptCut,
                                  T5KeyMask const* __restrict__ dBetaPassMask,
                                  unsigned int* __restrict__ moduleT5Count,
                                  unsigned int* __restrict__ nSelected,
                                  const unsigned int capacity,
                                  unsigned int* __restrict__ selectedT3s,
                                  float* __restrict__ selectedBridgeRadius,
                                  float* __restrict__ selectedDnnScore,
                                  unsigned int* __restrict__ dBeta2Memo) const {
      // The block synchronizations below require the threads of a block to share one inner triplet.
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[1] == 1));
      const auto& mdIndices = segments.mdIndices();
      const auto& segIdx = triplets.segmentIndices();
      const auto& lmIdx = triplets.lowerModuleIndices();

      for (unsigned int innerTripletIndex : cms::alpakatools::uniform_elements_y(acc, nTriplets)) {
        const uint16_t lowerModule1 = lmIdx[innerTripletIndex][0];
        if (!isValidQuintRegion(modules, lowerModule1))
          continue;
        const unsigned int secondMDOuter = mdIndices[segIdx[innerTripletIndex][1]][1];
        const unsigned int nKeys = nKeysByMD[secondMDOuter];
        const unsigned int keyOffset = keyOffsetByMD[secondMDOuter];
        const T5KeyMask passMask = dBetaPassMask[innerTripletIndex];
        if (passMask == 0 && nKeys <= kT5DBetaMaskBits)
          continue;

        if constexpr (cms::alpakatools::requires_single_thread_per_block_v<Acc3D>) {
          // Serial backend: the keys in order, then the outer triplets of each key in order.
          const DBetaBoundInner boundInner =
              makeDBetaBoundInner(acc, modules, mds, segments, triplets, innerTripletIndex);
          const T5RZInnerTerms rzInner =
              computeT5RZInnerTerms(acc, modules, mds, segments, triplets, innerTripletIndex);
          for (unsigned int key = 0; key < nKeys; ++key) {
            //asynchronous stop; exact truncation here is not important
            if (moduleT5Count[lowerModule1] > kNQuintupletThreshold)
              break;
            const unsigned int thirdSegIdx = keys[keyOffset + key];
            if (!passQuintupletKey(acc,
                                   modules,
                                   mds,
                                   segments,
                                   triplets,
                                   lowerModule1,
                                   innerTripletIndex,
                                   thirdSegIdx,
                                   key,
                                   passMask,
                                   ptCut))
              continue;
            const float bridgeRadius =
                computeT5BridgeRadius(acc, mds, segments, triplets, innerTripletIndex, thirdSegIdx);
            const unsigned int nOuter = tripletsRangesBySegment.n()[thirdSegIdx];
            const unsigned int outerOffset = tripletsRangesBySegment.offset()[thirdSegIdx];
            for (unsigned int outerIndex = 0; outerIndex < nOuter; ++outerIndex) {
              if (moduleT5Count[lowerModule1] > kNQuintupletThreshold)
                break;
              selectQuintupletPair(acc,
                                   modules,
                                   mds,
                                   mdT5Counts,
                                   segments,
                                   triplets,
                                   mdOccupancy,
                                   tripletsRangesByMD,
                                   lowerModule1,
                                   innerTripletIndex,
                                   tripletsBySegment.tripletIndex()[outerOffset + outerIndex],
                                   bridgeRadius,
                                   rzInner,
                                   ptCut,
                                   dBeta2Memo,
                                   boundInner,
                                   moduleT5Count,
                                   nSelected,
                                   capacity,
                                   selectedT3s,
                                   selectedBridgeRadius,
                                   selectedDnnScore);
            }
          }
        } else {
          auto& chunkOuterOffset = alpaka::declareSharedVar<unsigned int[kT5KeysPerChunk], __COUNTER__>(acc);
          auto& chunkNOuter = alpaka::declareSharedVar<unsigned int[kT5KeysPerChunk], __COUNTER__>(acc);
          auto& chunkBridgeRadius = alpaka::declareSharedVar<float[kT5KeysPerChunk], __COUNTER__>(acc);
          for (unsigned int firstKey = 0; firstKey < nKeys; firstKey += kT5KeysPerChunk) {
            const unsigned int nChunkKeys = nKeys - firstKey < kT5KeysPerChunk ? nKeys - firstKey : kT5KeysPerChunk;
            alpaka::syncBlockThreads(acc);  // the previous chunk is done with the shared arrays
            for (unsigned int keyInChunk : cms::alpakatools::uniform_elements_x(acc, nChunkKeys)) {
              const unsigned int key = firstKey + keyInChunk;
              const unsigned int thirdSegIdx = keys[keyOffset + key];
              const bool pass = passQuintupletKey(acc,
                                                  modules,
                                                  mds,
                                                  segments,
                                                  triplets,
                                                  lowerModule1,
                                                  innerTripletIndex,
                                                  thirdSegIdx,
                                                  key,
                                                  passMask,
                                                  ptCut);
              chunkNOuter[keyInChunk] = pass ? tripletsRangesBySegment.n()[thirdSegIdx] : 0;
              if (pass) {
                chunkOuterOffset[keyInChunk] = tripletsRangesBySegment.offset()[thirdSegIdx];
                chunkBridgeRadius[keyInChunk] =
                    computeT5BridgeRadius(acc, mds, segments, triplets, innerTripletIndex, thirdSegIdx);
              }
            }
            alpaka::syncBlockThreads(acc);
            unsigned int nPairs = 0;
            for (unsigned int keyInChunk = 0; keyInChunk < nChunkKeys; ++keyInChunk)
              nPairs += chunkNOuter[keyInChunk];
            // Pairs in key order; each thread walks forward to the key of its next pair.
            unsigned int pairKey = 0, firstPairOfKey = 0;
            for (unsigned int pair : cms::alpakatools::uniform_elements_x(acc, nPairs)) {
              if (moduleT5Count[lowerModule1] > kNQuintupletThreshold)
                break;
              while (pair >= firstPairOfKey + chunkNOuter[pairKey]) {
                firstPairOfKey += chunkNOuter[pairKey];
                ++pairKey;
              }
              selectQuintupletPair(acc,
                                   modules,
                                   mds,
                                   mdT5Counts,
                                   segments,
                                   triplets,
                                   mdOccupancy,
                                   tripletsRangesByMD,
                                   lowerModule1,
                                   innerTripletIndex,
                                   tripletsBySegment.tripletIndex()[chunkOuterOffset[pairKey] + pair - firstPairOfKey],
                                   chunkBridgeRadius[pairKey],
                                   computeT5RZInnerTerms(acc, modules, mds, segments, triplets, innerTripletIndex),
                                   ptCut,
                                   dBeta2Memo,
                                   DBetaBoundInner{},
                                   moduleT5Count,
                                   nSelected,
                                   capacity,
                                   selectedT3s,
                                   selectedBridgeRadius,
                                   selectedDnnScore);
            }
          }
        }
      }
    }
  };

  // Stores the pairs selected by CountTripletConnections: the dense inner triplets in the first launch, the sparse ones
  // in the second, so each module keeps the T5 order of the former dense and per-module creation kernels.
  struct AddSelectedQuintuplets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  [[maybe_unused]] ModulesConst modules,
                                  [[maybe_unused]] MiniDoubletsConst mds,
                                  MiniDoubletsT5CountsConst mdT5Counts,
                                  [[maybe_unused]] MiniDoubletsOccupancyConst mdOccupancy,
                                  SegmentsConst segments,
                                  Triplets triplets,
                                  TripletsOccupancyConst tripletsOcc,
                                  [[maybe_unused]] TripletsRangesConst tripletsRangesByMD,
                                  QuintupletsLoose quintupletsLoose,
                                  QuintupletsOccupancy quintupletsOccupancy,
                                  QuintupletsRanges quintupletsRangesByMD0,
                                  QuintupletsRanges quintupletsRangesByMD1,
                                  ObjectRanges ranges,
                                  unsigned int const* __restrict__ selectedT3s,
                                  float const* __restrict__ selectedBridgeRadius,
                                  float const* __restrict__ selectedDnnScore,
                                  const unsigned int nSelected,
                                  const bool densePass,
                                  [[maybe_unused]] const float ptCut) const {
      const auto& lmIdx = triplets.lowerModuleIndices();
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, nSelected)) {
        const unsigned int innerTripletIndex = selectedT3s[2 * i];
        const uint16_t lowerModule1 = lmIdx[innerTripletIndex][0];
        if (isDenseT5Pair(tripletsOcc, lowerModule1, lmIdx[innerTripletIndex][2]) != densePass)
          continue;
        const unsigned int outerTripletIndex = selectedT3s[2 * i + 1];
        float rzChiSquared = 0.f, dBeta1 = 0.f, dBeta2 = 0.f;
#ifdef CUT_VALUE_DEBUG
        float innerRadius, outerRadius, bridgeRadius, dnnScore;
        runQuintupletSelection(acc,
                               modules,
                               mds,
                               segments,
                               triplets,
                               mdOccupancy,
                               tripletsRangesByMD,
                               lowerModule1,
                               innerTripletIndex,
                               outerTripletIndex,
                               innerRadius,
                               outerRadius,
                               bridgeRadius,
                               rzChiSquared,
                               dBeta1,
                               dBeta2,
                               dnnScore,
                               ptCut);
#endif
        addQuintupletCandidate(acc,
                               mdT5Counts,
                               segments,
                               triplets,
                               quintupletsLoose,
                               quintupletsOccupancy,
                               quintupletsRangesByMD0,
                               quintupletsRangesByMD1,
                               ranges,
                               innerTripletIndex,
                               outerTripletIndex,
                               lowerModule1,
                               selectedBridgeRadius[i],
                               selectedDnnScore[i],
                               rzChiSquared,
                               dBeta1,
                               dBeta2);
      }
    }
  };

  struct CreateEligibleModulesListForQuintuplets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  TripletsOccupancyConst tripletsOcc,
                                  ObjectRanges ranges,
                                  unsigned int const* __restrict__ moduleT5Count) const {
      // Single-block kernel
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      int& nEligibleT5Modulesx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      int& nTotalQuintupletsx = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nEligibleT5Modulesx = 0;
        nTotalQuintupletsx = 0;
      }
      alpaka::syncBlockThreads(acc);

      for (uint16_t lowerModule : cms::alpakatools::uniform_elements(acc, modules.nLowerModules())) {
        if (!isValidQuintRegion(modules, lowerModule))
          continue;

        unsigned int nInnerTriplets = tripletsOcc.nTriplets()[lowerModule];
        if (nInnerTriplets == 0)
          continue;

        int dynamic_count = static_cast<int>(moduleT5Count[lowerModule]);

        if (dynamic_count == 0)
          continue;
        if (dynamic_count > kNQuintupletThreshold)
          dynamic_count = kNQuintupletThreshold;

        int nEligibleT5Modules = alpaka::atomicAdd(acc, &nEligibleT5Modulesx, 1, alpaka::hierarchy::Threads{});
        int nTotQ = alpaka::atomicAdd(acc, &nTotalQuintupletsx, dynamic_count, alpaka::hierarchy::Threads{});

        ranges.quintupletModuleIndices()[lowerModule] = nTotQ;
        ranges.indicesOfEligibleT5Modules()[nEligibleT5Modules] = lowerModule;
        ranges.quintupletModuleOccupancy()[lowerModule] = dynamic_count;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        ranges.nEligibleT5Modules() = static_cast<uint16_t>(nEligibleT5Modulesx);
        ranges.nTotalQuints() = static_cast<unsigned int>(nTotalQuintupletsx);
        ranges.nTotalQuintsByMD0() = 0;  // summed by CreateQuintupletRangesByMD, which runs next
        ranges.nTotalQuintsByMD1() = 0;
        ranges.nQuintupletOverflows() = 0;
        ranges.nT5byMDOverflows() = 0;
        ranges.nQuintupletCapDrops() = 0;
      }
    }
  };

  // T5-by-MD list ranges, one lower module per block: the module takes its slice of each list with one atomic and
  // splits it over its MDs (in module order and then MD order on a serial backend).
  struct CreateQuintupletRangesByMD {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  ObjectRanges ranges,
                                  MiniDoubletsT5CountsConst mdT5Counts,
                                  MiniDoubletsOccupancyConst mdsOcc,
                                  QuintupletsRanges quintupletsRangesByMD0,
                                  QuintupletsRanges quintupletsRangesByMD1) const {
      auto& moduleCount = alpaka::declareSharedVar<unsigned int[2], __COUNTER__>(acc);
      auto& moduleStart = alpaka::declareSharedVar<unsigned int[2], __COUNTER__>(acc);
      unsigned int const* __restrict__ connectedMax0 = mdT5Counts.connectedT5s0Max().data();
      unsigned int const* __restrict__ connectedMax1 = mdT5Counts.connectedT5s1Max().data();

      for (uint16_t lowerModule : cms::alpakatools::uniform_groups_y(acc, modules.nLowerModules())) {
        const unsigned int nMDs = mdsOcc.nMDs()[lowerModule];
        if (nMDs == 0)
          continue;
        const unsigned int firstIdx = ranges.miniDoubletModuleIndices()[lowerModule];

        if (cms::alpakatools::once_per_block(acc)) {
          moduleCount[0] = 0;
          moduleCount[1] = 0;
        }
        alpaka::syncBlockThreads(acc);
        for (unsigned int idx : cms::alpakatools::uniform_elements_x(acc, nMDs)) {
          alpaka::atomicAdd(acc, &moduleCount[0], connectedMax0[firstIdx + idx], alpaka::hierarchy::Threads{});
          alpaka::atomicAdd(acc, &moduleCount[1], connectedMax1[firstIdx + idx], alpaka::hierarchy::Threads{});
        }
        alpaka::syncBlockThreads(acc);
        if (cms::alpakatools::once_per_block(acc)) {
          moduleStart[0] =
              alpaka::atomicAdd(acc, &ranges.nTotalQuintsByMD0(), moduleCount[0], alpaka::hierarchy::Blocks{});
          moduleStart[1] =
              alpaka::atomicAdd(acc, &ranges.nTotalQuintsByMD1(), moduleCount[1], alpaka::hierarchy::Blocks{});
          moduleCount[0] = 0;  // now the cursor within the module's slice
          moduleCount[1] = 0;
        }
        alpaka::syncBlockThreads(acc);
        for (unsigned int idx : cms::alpakatools::uniform_elements_x(acc, nMDs)) {
          const unsigned int mdIndex = firstIdx + idx;
          quintupletsRangesByMD0.offset()[mdIndex] =
              moduleStart[0] +
              alpaka::atomicAdd(acc, &moduleCount[0], connectedMax0[mdIndex], alpaka::hierarchy::Threads{});
          quintupletsRangesByMD0.n()[mdIndex] = 0;
          quintupletsRangesByMD1.offset()[mdIndex] =
              moduleStart[1] +
              alpaka::atomicAdd(acc, &moduleCount[1], connectedMax1[mdIndex], alpaka::hierarchy::Threads{});
          quintupletsRangesByMD1.n()[mdIndex] = 0;
        }
        alpaka::syncBlockThreads(acc);  // the next module resets moduleCount
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
