#ifndef RecoTracker_LSTCore_src_alpaka_Triplet_h
#define RecoTracker_LSTCore_src_alpaka_Triplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "ObjectRanges.h"

namespace lst {
  struct Triplets {
    unsigned int* segmentIndices;
    uint16_t* lowerModuleIndices;  //3 of them
    unsigned int* nTriplets;
    unsigned int* totOccupancyTriplets;
    unsigned int* nMemoryLocations;
    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    FPX* betaIn;
    float* circleRadius;
    float* circleCenterX;
    float* circleCenterY;
    bool* partOfPT5;
    bool* partOfT5;
    bool* partOfPT3;

#ifdef CUT_VALUE_DEBUG
    //debug variables
    float* zOut;
    float* rtOut;
    float* betaInCut;
#endif
    template <typename TBuff>
    void setData(TBuff& buf) {
      segmentIndices = alpaka::getPtrNative(buf.segmentIndices_buf);
      lowerModuleIndices = alpaka::getPtrNative(buf.lowerModuleIndices_buf);
      nTriplets = alpaka::getPtrNative(buf.nTriplets_buf);
      totOccupancyTriplets = alpaka::getPtrNative(buf.totOccupancyTriplets_buf);
      nMemoryLocations = alpaka::getPtrNative(buf.nMemoryLocations_buf);
      logicalLayers = alpaka::getPtrNative(buf.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(buf.hitIndices_buf);
      betaIn = alpaka::getPtrNative(buf.betaIn_buf);
      circleRadius = alpaka::getPtrNative(buf.circleRadius_buf);
      circleCenterX = alpaka::getPtrNative(buf.circleCenterX_buf);
      circleCenterY = alpaka::getPtrNative(buf.circleCenterY_buf);
      partOfPT5 = alpaka::getPtrNative(buf.partOfPT5_buf);
      partOfT5 = alpaka::getPtrNative(buf.partOfT5_buf);
      partOfPT3 = alpaka::getPtrNative(buf.partOfPT3_buf);
#ifdef CUT_VALUE_DEBUG
      zOut = alpaka::getPtrNative(buf.zOut_buf);
      rtOut = alpaka::getPtrNative(buf.rtOut_buf);
      betaInCut = alpaka::getPtrNative(buf.betaInCut_buf);
#endif
    }
  };

  template <typename TDev>
  struct TripletsBuffer {
    Buf<TDev, unsigned int> segmentIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, unsigned int> nTriplets_buf;
    Buf<TDev, unsigned int> totOccupancyTriplets_buf;
    Buf<TDev, unsigned int> nMemoryLocations_buf;
    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, FPX> betaIn_buf;
    Buf<TDev, float> circleRadius_buf;
    Buf<TDev, float> circleCenterX_buf;
    Buf<TDev, float> circleCenterY_buf;
    Buf<TDev, bool> partOfPT5_buf;
    Buf<TDev, bool> partOfT5_buf;
    Buf<TDev, bool> partOfPT3_buf;

#ifdef CUT_VALUE_DEBUG
    Buf<TDev, float> zOut_buf;
    Buf<TDev, float> rtOut_buf;
    Buf<TDev, float> deltaPhiPos_buf;
    Buf<TDev, float> deltaPhi_buf;
    Buf<TDev, float> zLo_buf;
    Buf<TDev, float> zHi_buf;
    Buf<TDev, float> zLoPointed_buf;
    Buf<TDev, float> zHiPointed_buf;
    Buf<TDev, float> dPhiCut_buf;
    Buf<TDev, float> betaInCut_buf;
    Buf<TDev, float> rtLo_buf;
    Buf<TDev, float> rtHi_buf;
#endif

    Triplets data_;

    template <typename TQueue, typename TDevAcc>
    TripletsBuffer(unsigned int maxTriplets, unsigned int nLowerModules, TDevAcc const& devAccIn, TQueue& queue)
        : segmentIndices_buf(allocBufWrapper<unsigned int>(devAccIn, 2 * maxTriplets, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, Params_T3::kLayers * maxTriplets, queue)),
          nTriplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          totOccupancyTriplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          nMemoryLocations_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxTriplets * Params_T3::kLayers, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxTriplets * Params_T3::kHits, queue)),
          betaIn_buf(allocBufWrapper<FPX>(devAccIn, maxTriplets, queue)),
          circleRadius_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          circleCenterX_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          circleCenterY_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          partOfPT5_buf(allocBufWrapper<bool>(devAccIn, maxTriplets, queue)),
          partOfT5_buf(allocBufWrapper<bool>(devAccIn, maxTriplets, queue)),
          partOfPT3_buf(allocBufWrapper<bool>(devAccIn, maxTriplets, queue))
#ifdef CUT_VALUE_DEBUG
          ,
          zOut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          rtOut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          deltaPhiPos_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          deltaPhi_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          zLo_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          zHi_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          zLoPointed_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          zHiPointed_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          dPhiCut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          betaInCut_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          rtLo_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue)),
          rtHi_buf(allocBufWrapper<float>(devAccIn, maxTriplets, queue))
#endif
    {
      alpaka::memset(queue, nTriplets_buf, 0u);
      alpaka::memset(queue, totOccupancyTriplets_buf, 0u);
      alpaka::memset(queue, partOfPT5_buf, false);
      alpaka::memset(queue, partOfT5_buf, false);
      alpaka::memset(queue, partOfPT3_buf, false);
    }

    inline Triplets const* data() const { return &data_; }
    inline void setData(TripletsBuffer& buf) { data_.setData(buf); }
  };

#ifdef CUT_VALUE_DEBUG
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(lst::Modules const& modulesInGPU,
                                                         lst::MiniDoublets const& mdsInGPU,
                                                         lst::Segments const& segmentsInGPU,
                                                         lst::Triplets& tripletsInGPU,
                                                         unsigned int innerSegmentIndex,
                                                         unsigned int outerSegmentIndex,
                                                         uint16_t innerInnerLowerModuleIndex,
                                                         uint16_t middleLowerModuleIndex,
                                                         uint16_t outerOuterLowerModuleIndex,
                                                         float zOut,
                                                         float rtOut,
                                                         float betaIn,
                                                         float betaInCut,
                                                         float circleRadius,
                                                         float circleCenterX,
                                                         float circleCenterY,
                                                         unsigned int tripletIndex)
#else
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTripletToMemory(lst::Modules const& modulesInGPU,
                                                         lst::MiniDoublets const& mdsInGPU,
                                                         lst::Segments const& segmentsInGPU,
                                                         lst::Triplets& tripletsInGPU,
                                                         unsigned int innerSegmentIndex,
                                                         unsigned int outerSegmentIndex,
                                                         uint16_t innerInnerLowerModuleIndex,
                                                         uint16_t middleLowerModuleIndex,
                                                         uint16_t outerOuterLowerModuleIndex,
                                                         float betaIn,
                                                         float circleRadius,
                                                         float circleCenterX,
                                                         float circleCenterY,
                                                         unsigned int tripletIndex)
#endif
  {
    tripletsInGPU.segmentIndices[tripletIndex * 2] = innerSegmentIndex;
    tripletsInGPU.segmentIndices[tripletIndex * 2 + 1] = outerSegmentIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * Params_T3::kLayers] = innerInnerLowerModuleIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * Params_T3::kLayers + 1] = middleLowerModuleIndex;
    tripletsInGPU.lowerModuleIndices[tripletIndex * Params_T3::kLayers + 2] = outerOuterLowerModuleIndex;

    tripletsInGPU.betaIn[tripletIndex] = __F2H(betaIn);
    tripletsInGPU.circleRadius[tripletIndex] = circleRadius;
    tripletsInGPU.circleCenterX[tripletIndex] = circleCenterX;
    tripletsInGPU.circleCenterY[tripletIndex] = circleCenterY;
    tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers] =
        modulesInGPU.layers[innerInnerLowerModuleIndex] + (modulesInGPU.subdets[innerInnerLowerModuleIndex] == 4) * 6;
    tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers + 1] =
        modulesInGPU.layers[middleLowerModuleIndex] + (modulesInGPU.subdets[middleLowerModuleIndex] == 4) * 6;
    tripletsInGPU.logicalLayers[tripletIndex * Params_T3::kLayers + 2] =
        modulesInGPU.layers[outerOuterLowerModuleIndex] + (modulesInGPU.subdets[outerOuterLowerModuleIndex] == 4) * 6;
    //get the hits
    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits] = mdsInGPU.anchorHitIndices[firstMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 1] = mdsInGPU.outerHitIndices[firstMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 2] = mdsInGPU.anchorHitIndices[secondMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 3] = mdsInGPU.outerHitIndices[secondMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 4] = mdsInGPU.anchorHitIndices[thirdMDIndex];
    tripletsInGPU.hitIndices[tripletIndex * Params_T3::kHits + 5] = mdsInGPU.outerHitIndices[thirdMDIndex];
#ifdef CUT_VALUE_DEBUG
    tripletsInGPU.zOut[tripletIndex] = zOut;
    tripletsInGPU.rtOut[tripletIndex] = rtOut;
    tripletsInGPU.betaInCut[tripletIndex] = betaInCut;
#endif
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passRZConstraint(TAcc const& acc,
                                                       struct lst::Modules const& modulesInGPU,
                                                       struct lst::MiniDoublets const& mdsInGPU,
                                                       struct lst::Segments const& segmentsInGPU,
                                                       uint16_t& innerInnerLowerModuleIndex,
                                                       uint16_t& middleLowerModuleIndex,
                                                       uint16_t& outerOuterLowerModuleIndex,
                                                       unsigned int& firstMDIndex,
                                                       unsigned int& secondMDIndex,
                                                       unsigned int& thirdMDIndex,
                                                       float& circleRadius,
                                                       float& circleCenterX, 
                                                       float& circleCenterY) {

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer1 = modulesInGPU.lstLayers[innerInnerLowerModuleIndex];
    const int layer2 = modulesInGPU.lstLayers[middleLowerModuleIndex];
    const int layer3 = modulesInGPU.lstLayers[outerOuterLowerModuleIndex];

    //get r and z
    const float r1 = mdsInGPU.anchorRt[firstMDIndex] / 100; // all the values are stored in the unit of cm, in the calculation below we need to be cautious if we want to use the meter unit
    const float r2 = mdsInGPU.anchorRt[secondMDIndex] / 100;
    const float r3 = mdsInGPU.anchorRt[thirdMDIndex] / 100;

    const float z1 = mdsInGPU.anchorZ[firstMDIndex] / 100;
    const float z2 = mdsInGPU.anchorZ[secondMDIndex] / 100;
    const float z3 = mdsInGPU.anchorZ[thirdMDIndex] / 100;

    float residual = 100 * (z2 - ((z3 - z1) / (r3 - r1) * (r2 - r1) + z1));
    if (layer1 == 1 && layer2 == 7) {
      return alpaka::math::abs(acc, residual) < 1.0f;          // Region 9
    } else if (layer1 == 3 && layer2==4) {
      if (layer3 == 5) {
        return alpaka::math::abs(acc, residual) < 3.7127972f;  // Region 20
      } else if (layer3 == 12) {
        return alpaka::math::abs(acc, residual) < 5.0f;        // Region 21
      }
    } else if (layer1 == 4) {
      if (layer2 == 12) {
        return alpaka::math::abs(acc, residual) < 6.3831687f;  // Region 22
      } else if (layer2 == 5) {
        if (layer3 == 6) {
          return alpaka::math::abs(acc, residual) < 4.362525f; // Region 23
        } else if (layer3 == 12) {
          return alpaka::math::abs(acc, residual) < 5.0f;      // Region 24
        }
      }
    } 

    //get the type of module: 0 is ps, 1 is 2s
    const int moduleType3 = modulesInGPU.moduleType[outerOuterLowerModuleIndex];

    //get the x,y position of each MD
    const float x1 = mdsInGPU.anchorX[firstMDIndex] / 100;
    const float x2 = mdsInGPU.anchorX[secondMDIndex] / 100;
    const float x3 = mdsInGPU.anchorX[thirdMDIndex] / 100;

    const float y1 = mdsInGPU.anchorY[firstMDIndex] / 100;
    const float y2 = mdsInGPU.anchorY[secondMDIndex] / 100;
    const float y3 = mdsInGPU.anchorY[thirdMDIndex] / 100;

    //set initial and target points
    float x_init = x2;
    float y_init = y2;
    float z_init = z2;
    float r_init = r2;

    float z_target = z3;
    float r_target = r3;

    float x_other = x1;
    float y_other = y1;
    float z_other = z1;
    float r_other = r1;

    //use MD2 for regions 5 and 19
    if ((layer1 == 8 && layer2 == 14 && layer3 == 15) || (layer1 == 3 && layer2 == 12 && layer3 == 13)){
      x_init = x1;
      y_init = y1;
      z_init = z1;
      r_init = r1;

      z_target = z2;
      r_target = r2;

      x_other = x3;
      y_other = y3;
      z_other = z3;
      r_other = r3;
    }

    //use the 3 MDs to fit a circle. This is the circle parameters, for circle centers and circle radius
    float x_center = circleCenterX / 100;
    float y_center = circleCenterY / 100;
    float Pt = 2 * k2Rinv1GeVf * circleRadius; //k2Rinv1GeVf is already in cm^(-1)

    //determine the charge
    int charge = 0;
    float slope12 = (y2 - y1) / (x2 - x1);
    float slope23 = (y3 - y2) / (x3 - x2);
    if (slope12 > 0 and slope23 < 0) {
      if (x1 < x2 and x2 < x3)
        charge = 1;
      else if (x1 > x2 and x2 > x3)
        charge = 1;
      else if (y1 < y2 and y2 < y3)
        charge = -1;
      else if (y1 > y2 and y2 > y3)
        charge = -1;
    } else if (slope12 < 0 and slope23 > 0) {
      if (x1 < x2 and x2 < x3)
        charge = -1;
      else if (x1 > x2 and x2 > x3)
        charge = -1;
      else if (y1 < y2 and y2 < y3)
        charge = 1;
      else if (y1 > y2 and y2 > y3)
        charge = 1;
    } else if (slope12 > 0 and slope23 > 0) {
      if (slope12 > slope23)
        charge = 1;
      else
        charge = -1;
    } else if (slope12 < 0 and slope23 < 0) {
      if (slope12 > slope23)
        charge = 1;
      else
        charge = -1;
    }

    //get the absolute value of px and py at the initial point
    float pseudo_phi = alpaka::math::atan(
        acc, (y_init - y_center) / (x_init - x_center));  //actually represent pi/2-phi, wrt helix axis z
    float Px = Pt * alpaka::math::abs(acc, alpaka::math::sin(acc, pseudo_phi)),
          Py = Pt * alpaka::math::abs(acc, cos(pseudo_phi));

    //Above line only gives you the correct value of Px and Py, but signs of Px and Py calculated below.
    //We look at if the circle is clockwise or anti-clock wise, to make it simpler, we separate the x-y plane into 4 quarters.
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

    //But if the initial T5 curve goes across quarters(i.e. cross axis to separate the quarters), need special redeclaration of Px,Py signs on these to avoid errors
    if (x3 < x2 && x2 < x1)
      Px = -alpaka::math::abs(acc, Px);
    if (x3 > x2 && x2 > x1)
      Px = alpaka::math::abs(acc, Px);
    if (y3 < y2 && y2 < y1)
      Py = -alpaka::math::abs(acc, Py);
    if (y3 > y2 && y2 > y1)
      Py = alpaka::math::abs(acc, Py);

    float AO = alpaka::math::sqrt(acc, (x_other - x_center) * (x_other - x_center) + (y_other - y_center) * (y_other - y_center)); 
    float BO = alpaka::math::sqrt(acc, (x_init - x_center) * (x_init - x_center) + (y_init - y_center) * (y_init - y_center));
    float AB2 = (x_other - x_init) * (x_other - x_init) + (y_other - y_init) * (y_other - y_init); 
    float dPhi = alpaka::math::acos(acc, (AO * AO + BO * BO - AB2) / (2 * AO * BO)); //Law of Cosines
    float ds = circleRadius / 100 * dPhi;
    float Pz = (z2 - z1) / ds * Pt; 

    //for regions 5 and 19
    if ((layer1 == 8 && layer2 == 14 && layer3 == 15) || (layer1 == 3 && layer2 == 12 && layer3 == 13)) {
      Pz = (z3 - z1) / ds * Pt;
    }

    float p = alpaka::math::sqrt(acc, Px * Px + Py * Py + Pz * Pz);
    float a = -2.f * k2Rinv1GeVf * 100 * charge;
    float rou = a / p;

    float rzChiSquared = 0;
    float error = 0;

    //check the tilted module, side: PosZ, NegZ, Center(for not tilted)
    float drdz = alpaka::math::abs(acc, modulesInGPU.drdzs[outerOuterLowerModuleIndex]);
    short side = modulesInGPU.sides[outerOuterLowerModuleIndex];
    short subdets = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    //calculate residual
    if (layer3 <= 6 && ((side == lst::Center) or (drdz < 1))) { // for barrel
      float paraA = r_init * r_init + 2 * (Px * Px + Py * Py) / (a * a) + 2 * (y_init * Px - x_init * Py) / a - r3 * r3;
      float paraB = 2 * (x_init * Px + y_init * Py) / a;
      float paraC = 2 * (y_init * Px - x_init * Py) / a + 2 * (Px * Px + Py * Py) / (a * a);
      float A = paraB * paraB + paraC * paraC;
      float B = 2 * paraA * paraB;
      float C = paraA * paraA - paraC * paraC;
      float sol1 = (-B + alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
      float sol2 = (-B - alpaka::math::sqrt(acc, B * B - 4 * A * C)) / (2 * A);
      float solz1 = alpaka::math::asin(acc, sol1) / rou * Pz / p + z_init;
      float solz2 = alpaka::math::asin(acc, sol2) / rou * Pz / p + z_init;
      float diffz1 = (solz1 - z3) * 100;
      float diffz2 = (solz2 - z3) * 100;
      if (alpaka::math::isnan(acc, diffz1))
        residual = diffz2;
      else if (alpaka::math::isnan(acc, diffz2))
        residual = diffz1;
      else {
        residual = (alpaka::math::abs(acc, diffz1) < alpaka::math::abs(acc, diffz2)) ? diffz1 : diffz2;
      }
    } else { // for endcap
      float s = (z_target - z_init) * p / Pz;
      float x = x_init + Px / a * alpaka::math::sin(acc, rou * s) - Py / a * (1 - alpaka::math::cos(acc, rou * s));
      float y = y_init + Py / a * alpaka::math::sin(acc, rou * s) + Px / a * (1 - alpaka::math::cos(acc, rou * s));
      residual = (r_target - alpaka::math::sqrt(acc, x * x + y * y)) * 100; 
    }
    
    // error
    if (moduleType3 == 0) { 
      error = 0.15f;        //PS
    } else  {              
      error = 5.0f;         //2S
    }

    float projection_missing = 1;
    if (drdz < 1)
      projection_missing = ((subdets == lst::Endcap) or (side == lst::Center))
                                ? 1.f
                                : 1 / alpaka::math::sqrt(acc, 1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
    if (drdz > 1)
      projection_missing = ((subdets == lst::Endcap) or (side == lst::Center))
                                ? 1.f
                                : drdz / alpaka::math::sqrt(acc, 1 + drdz * drdz);  //sin(atan(drdz)), if dr/dz>1
    error = error * projection_missing;

    rzChiSquared = 12 * (residual * residual) / (error * error);

    //helix calculation failed, use linear approximation
    if (alpaka::math::isnan(acc, rzChiSquared) || circleRadius < 0) {
      float slope = (z_other - z1) / (r_other - r1);

      residual = (layer3 <= 6) ? ((z_target - z1) - slope * (r_target - r1)) : ((r_target - r1) - (z_target - z1) / slope);
      residual = (moduleType3 == 0) ? residual / 0.15f : residual / 5.0f;
      residual = residual * 100;

      rzChiSquared = 12 * residual * residual;
      return rzChiSquared < 2.7711823f;
    }

    // cuts
    if (layer1 == 7) {
      if (layer2 == 8) {
        if (layer3 == 9) {
          return rzChiSquared < 65.47191f;   // Region 0
        } else if (layer3 == 14) {
          return rzChiSquared < 3.3200853f;   // Region 1
        }
      } else if (layer2 == 13) {
        return rzChiSquared < 17.194584f;      // Region 2
      }
    } else if (layer1 == 8) {
      if (layer2 == 9) {
        if (layer3 == 10) {
          return rzChiSquared < 114.91959f;    // Region 3
        } else if (layer3 == 15) {
          return rzChiSquared < 3.4359624f;   // Region 4
        } 
      } else if (layer2 == 14) {
        return rzChiSquared < 4.6487956f;     // Region 5
      }
    } else if (layer1 == 9) {
      if (layer2 == 10) {
        if (layer3 == 11) {
          return rzChiSquared < 97.34339f;    // Region 6
        } else if (layer3 == 16) {
          return rzChiSquared < 3.095819f;    // Region 7
        }
      } else if (layer2 == 15) {
        return rzChiSquared < 11.477617f;     // Region 8
      }
    } else if (layer1 == 1) {
      if (layer3 == 7) {
        return rzChiSquared < 96.949936f;   // Region 10
      } else if (layer3 == 3) {
        return rzChiSquared < 458.43982f;    // Region 11
      }
    } else if (layer1 == 2) {
      if (layer2 == 7) {
        if (layer3 == 8) {
          return rzChiSquared < 218.82303f;   // Region 12
        } else if (layer3 == 13) {
          return rzChiSquared < 3.155554f;    // Region 13
        }
      } else if (layer2 == 3) {
        if (layer3 == 7) {
          return rzChiSquared < 235.5005f;    // Region 14
        } else if (layer3 == 12) {
          return rzChiSquared < 3.8522234f;    // Region 15
        } else if (layer3 == 4) {
          return rzChiSquared < 3.5852437f;   // Region 16
        }
      }
    } else if (layer1 == 3) {
      if (layer2 == 7) {
        if (layer3 == 8) {
          return rzChiSquared < 42.68f;   // Region 17
        } else if (layer3 == 13) {
          return rzChiSquared < 3.853796f;   // Region 18
        }
      } else if (layer2 == 12) {
        return rzChiSquared < 6.2774787f;     // Region 19
      }
    }
    return false;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBB(TAcc const& acc,
                                                                lst::Modules const& modulesInGPU,
                                                                lst::MiniDoublets const& mdsInGPU,
                                                                lst::Segments const& segmentsInGPU,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t middleLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                unsigned int innerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut,
                                                                const float ptCut) {
    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == lst::PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeVOut =
        alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    float rtRatio_OutIn = rtOut / rtIn;  // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale = alpaka::math::tan(acc, alpha1GeVOut) / alpha1GeVOut;  // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
    float zpitchOut = (isPSOut ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);

    const float zHi =
        zIn + (zIn + lst::kDeltaZLum) * (rtRatio_OutIn - 1.f) * (zIn < 0.f ? 1.f : dzDrtScale) + (zpitchIn + zpitchOut);
    const float zLo = zIn + (zIn - lst::kDeltaZLum) * (rtRatio_OutIn - 1.f) * (zIn > 0.f ? 1.f : dzDrtScale) -
                      (zpitchIn + zpitchOut);  //slope-correction only on outer end

    //Cut 1 - z compatibility
    if ((zOut < zLo) || (zOut > zHi))
      return false;

    float drt_OutIn = (rtOut - rtIn);

    float r3In = alpaka::math::sqrt(acc, zIn * zIn + rtIn * rtIn);
    float drt_InSeg = rtMid - rtIn;
    float dz_InSeg = zMid - zIn;
    float dr3_InSeg =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

    float coshEta = dr3_InSeg / drt_InSeg;
    float dzErr = (zpitchIn + zpitchOut) * (zpitchIn + zpitchOut) * 2.f;

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rtOut - rtIn) / 50.f) * (r3In / rtIn);
    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    dzErr += muls2 * drt_OutIn * drt_OutIn / 3.f * coshEta * coshEta;
    dzErr = alpaka::math::sqrt(acc, dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutIn;
    const float zWindow =
        dzErr / drt_InSeg * drt_OutIn +
        (zpitchIn + zpitchOut);  //FIXME for lst::ptCut lower than ~0.8 need to add curv path correction
    const float zLoPointed = zIn + dzMean * (zIn > 0.f ? 1.f : dzDrtScale) - zWindow;
    const float zHiPointed = zIn + dzMean * (zIn < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Constructing upper and lower bound

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    if ((zOut < zLoPointed) || (zOut > zHiPointed))
      return false;

    // raw betaIn value without any correction, based on the mini-doublet hit positions
    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    betaIn = alpha_InLo - lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg =
        alpaka::math::sqrt(acc,
                           (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                   (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                   (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    betaInCut =
        alpaka::math::asin(
            acc, alpaka::math::min(acc, (-rt_InSeg + drt_tl_axis) * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    //Cut #3: first beta cut
    return alpaka::math::abs(acc, betaIn) < betaInCut;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintBBE(TAcc const& acc,
                                                                lst::Modules const& modulesInGPU,
                                                                lst::MiniDoublets const& mdsInGPU,
                                                                lst::Segments const& segmentsInGPU,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t middleLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                uint16_t innerOuterLowerModuleIndex,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut,
                                                                const float ptCut) {
    bool isPSIn = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == lst::PS);
    bool isPSOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::PS);

    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitchIn = (isPSIn ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
    float zpitchOut = (isPSOut ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
    float zGeom = zpitchIn + zpitchOut;

    // Cut #0: Preliminary (Only here in endcap case)
    if (zIn * zOut <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, lst::kDeltaZLum, zIn);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::PS;
    float rtGeom1 = isOutSgInnerMDPS ? lst::kPixelPSZpitch : lst::kStrip2SZpitch;
    float zGeom1 = alpaka::math::copysign(acc, zGeom, zIn);
    float rtLo = rtIn * (1.f + (zOut - zIn - zGeom1) / (zIn + zGeom1 + dLum) / dzDrtScale) -
                 rtGeom1;  //slope correction only on the lower end

    //Cut #1: rt condition
    float zInForHi = zIn - zGeom1 - dLum;
    if (zInForHi * zIn < 0) {
      zInForHi = alpaka::math::copysign(acc, 0.1f, zIn);
    }
    float rtHi = rtIn * (1.f + (zOut - zIn + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    float rIn = alpaka::math::sqrt(acc, zIn * zIn + rtIn * rtIn);

    const float drtSDIn = rtMid - rtIn;
    const float dzSDIn = zMid - zIn;
    const float dr3SDIn =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

    const float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    const float dzOutInAbs = alpaka::math::abs(acc, zOut - zIn);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = lst::kPixelPSZpitch;
    const float kZ = (zOut - zIn) / dzSDIn;
    float drtErr =
        zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
    const float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2 * (rtOut - rtIn) / 50.f) * (rIn / rtIn);
    const float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    drtErr += muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta;
    drtErr = alpaka::math::sqrt(acc, drtErr);

    //Cut #3: rt-z pointed

    if ((kZ < 0) || (rtOut < rtLo) || (rtOut > rtHi))
      return false;

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);

    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    betaIn = sdIn_alpha - lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;

    float swapTemp;

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }

    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                               (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                                           (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr + dr) * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / sdIn_d);

    //Cut #4: first beta cut
    return alpaka::math::abs(acc, betaInRHmin) < betaInCut;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraintEEE(TAcc const& acc,
                                                                lst::Modules const& modulesInGPU,
                                                                lst::MiniDoublets const& mdsInGPU,
                                                                lst::Segments const& segmentsInGPU,
                                                                uint16_t innerInnerLowerModuleIndex,
                                                                uint16_t middleLowerModuleIndex,
                                                                uint16_t outerOuterLowerModuleIndex,
                                                                unsigned int firstMDIndex,
                                                                unsigned int secondMDIndex,
                                                                unsigned int thirdMDIndex,
                                                                float& zOut,
                                                                float& rtOut,
                                                                unsigned int innerSegmentIndex,
                                                                unsigned int outerSegmentIndex,
                                                                float& betaIn,
                                                                float& betaInCut,
                                                                const float ptCut) {
    float rtIn = mdsInGPU.anchorRt[firstMDIndex];
    float rtMid = mdsInGPU.anchorRt[secondMDIndex];
    rtOut = mdsInGPU.anchorRt[thirdMDIndex];

    float zIn = mdsInGPU.anchorZ[firstMDIndex];
    float zMid = mdsInGPU.anchorZ[secondMDIndex];
    zOut = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_Out =
        alpaka::math::asin(acc, alpaka::math::min(acc, rtOut * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_Out) / alpha1GeV_Out;  // The track can bend in r-z plane slightly

    // Cut #0: Preliminary (Only here in endcap case)
    if (zIn * zOut <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, lst::kDeltaZLum, zIn);
    bool isOutSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == lst::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgOuterMDPS)  ? 2.f * lst::kPixelPSZpitch
                   : (isInSgInnerMDPS or isOutSgOuterMDPS) ? lst::kPixelPSZpitch + lst::kStrip2SZpitch
                                                           : 2.f * lst::kStrip2SZpitch;

    float dz = zOut - zIn;
    const float rtLo = rtIn * (1.f + dz / (zIn + dLum) / dzDrtScale) - rtGeom;  //slope correction only on the lower end
    const float rtHi = rtIn * (1.f + dz / (zIn - dLum)) + rtGeom;

    //Cut #1: rt condition
    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    bool isInSgOuterMDPS = modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::PS;

    float drtSDIn = rtMid - rtIn;
    float dzSDIn = zMid - zIn;
    float dr3SDIn =
        alpaka::math::sqrt(acc, rtMid * rtMid + zMid * zMid) - alpaka::math::sqrt(acc, rtIn * rtIn + zIn * zIn);

    float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    float dzOutInAbs = alpaka::math::abs(acc, zOut - zIn);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (zOut - zIn) / dzSDIn;
    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rtOut - rtIn) / 50.f);

    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;

    float drtErr = alpaka::math::sqrt(
        acc,
        lst::kPixelPSZpitch * lst::kPixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) +
            muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rtIn + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rtIn + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS)  // If both PS then we can point
    {
      if ((kZ < 0) || (rtOut < rtLo_point) || (rtOut > rtHi_point))
        return false;
    }

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);

    float tl_axis_x = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    betaIn = sdIn_alpha - lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float sdIn_alphaRHmin = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    float swapTemp;

    if (alpaka::math::abs(acc, betaInRHmin) > alpaka::math::abs(acc, betaInRHmax)) {
      swapTemp = betaInRHmin;
      betaInRHmin = betaInRHmax;
      betaInRHmax = swapTemp;
    }
    float sdIn_dr = alpaka::math::sqrt(acc,
                                       (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                               (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                                           (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    betaInCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, (-sdIn_dr + dr) * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / sdIn_d);

    //Cut #4: first beta cut
    return alpaka::math::abs(acc, betaInRHmin) < betaInCut;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passPointingConstraint(TAcc const& acc,
                                                             lst::Modules const& modulesInGPU,
                                                             lst::MiniDoublets const& mdsInGPU,
                                                             lst::Segments const& segmentsInGPU,
                                                             uint16_t innerInnerLowerModuleIndex,
                                                             uint16_t middleLowerModuleIndex,
                                                             uint16_t outerOuterLowerModuleIndex,
                                                             unsigned int firstMDIndex,
                                                             unsigned int secondMDIndex,
                                                             unsigned int thirdMDIndex,
                                                             float& zOut,
                                                             float& rtOut,
                                                             uint16_t innerOuterLowerModuleIndex,
                                                             unsigned int innerSegmentIndex,
                                                             unsigned int outerSegmentIndex,
                                                             float& betaIn,
                                                             float& betaInCut,
                                                             const float ptCut) {
    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short middleLowerModuleSubdet = modulesInGPU.subdets[middleLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == lst::Barrel and middleLowerModuleSubdet == lst::Barrel and
        outerOuterLowerModuleSubdet == lst::Barrel) {
      return passPointingConstraintBBB(acc,
                                       modulesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       zOut,
                                       rtOut,
                                       innerSegmentIndex,
                                       betaIn,
                                       betaInCut,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Barrel and middleLowerModuleSubdet == lst::Barrel and
               outerOuterLowerModuleSubdet == lst::Endcap) {
      return passPointingConstraintBBE(acc,
                                       modulesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       zOut,
                                       rtOut,
                                       innerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       betaIn,
                                       betaInCut,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Barrel and middleLowerModuleSubdet == lst::Endcap and
               outerOuterLowerModuleSubdet == lst::Endcap) {
      return passPointingConstraintBBE(acc,
                                       modulesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       zOut,
                                       rtOut,
                                       innerOuterLowerModuleIndex,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       betaIn,
                                       betaInCut,
                                       ptCut);

    }

    else if (innerInnerLowerModuleSubdet == lst::Endcap and middleLowerModuleSubdet == lst::Endcap and
             outerOuterLowerModuleSubdet == lst::Endcap) {
      return passPointingConstraintEEE(acc,
                                       modulesInGPU,
                                       mdsInGPU,
                                       segmentsInGPU,
                                       innerInnerLowerModuleIndex,
                                       middleLowerModuleIndex,
                                       outerOuterLowerModuleIndex,
                                       firstMDIndex,
                                       secondMDIndex,
                                       thirdMDIndex,
                                       zOut,
                                       rtOut,
                                       innerSegmentIndex,
                                       outerSegmentIndex,
                                       betaIn,
                                       betaInCut,
                                       ptCut);
    }
    return false;  // failsafe
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeRadiusFromThreeAnchorHits(
      TAcc const& acc, float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f) {
    float radius = 0.f;

    //(g,f) -> center
    //first anchor hit - (x1,y1), second anchor hit - (x2,y2), third anchor hit - (x3, y3)

    float denomInv = 1.0f / ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    float xy1sqr = x1 * x1 + y1 * y1;

    float xy2sqr = x2 * x2 + y2 * y2;

    float xy3sqr = x3 * x3 + y3 * y3;

    g = 0.5f * ((y3 - y2) * xy1sqr + (y1 - y3) * xy2sqr + (y2 - y1) * xy3sqr) * denomInv;

    f = 0.5f * ((x2 - x3) * xy1sqr + (x3 - x1) * xy2sqr + (x1 - x2) * xy3sqr) * denomInv;

    float c = ((x2 * y3 - x3 * y2) * xy1sqr + (x3 * y1 - x1 * y3) * xy2sqr + (x1 * y2 - x2 * y1) * xy3sqr) * denomInv;

    if (((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0) || (g * g + f * f - c < 0)) {
#ifdef WARNINGS
      printf("three collinear points or FATAL! r^2 < 0!\n");
#endif
      radius = -1.f;
    } else
      radius = alpaka::math::sqrt(acc, g * g + f * f - c);

    return radius;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runTripletConstraintsAndAlgo(TAcc const& acc,
                                                                   lst::Modules const& modulesInGPU,
                                                                   lst::MiniDoublets const& mdsInGPU,
                                                                   lst::Segments const& segmentsInGPU,
                                                                   uint16_t innerInnerLowerModuleIndex,
                                                                   uint16_t middleLowerModuleIndex,
                                                                   uint16_t outerOuterLowerModuleIndex,
                                                                   unsigned int innerSegmentIndex,
                                                                   unsigned int outerSegmentIndex,
                                                                   float& zOut,
                                                                   float& rtOut,
                                                                   float& betaIn,
                                                                   float& betaInCut,
                                                                   float& circleRadius,
                                                                   float& circleCenterX,
                                                                   float& circleCenterY,
                                                                   const float ptCut) {
    //this cut reduces the number of candidates by a factor of 4, i.e., 3 out of 4 warps can end right here!
    if (segmentsInGPU.mdIndices[2 * innerSegmentIndex + 1] != segmentsInGPU.mdIndices[2 * outerSegmentIndex])
      return false;

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * innerSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * outerSegmentIndex + 1];

    if (not(passPointingConstraint(acc,
                                   modulesInGPU,
                                   mdsInGPU,
                                   segmentsInGPU,
                                   innerInnerLowerModuleIndex,
                                   middleLowerModuleIndex,
                                   outerOuterLowerModuleIndex,
                                   firstMDIndex,
                                   secondMDIndex,
                                   thirdMDIndex,
                                   zOut,
                                   rtOut,
                                   middleLowerModuleIndex,
                                   innerSegmentIndex,
                                   outerSegmentIndex,
                                   betaIn,
                                   betaInCut,
                                   ptCut)))
      return false;

    float x1 = mdsInGPU.anchorX[firstMDIndex];
    float x2 = mdsInGPU.anchorX[secondMDIndex];
    float x3 = mdsInGPU.anchorX[thirdMDIndex];
    float y1 = mdsInGPU.anchorY[firstMDIndex];
    float y2 = mdsInGPU.anchorY[secondMDIndex];
    float y3 = mdsInGPU.anchorY[thirdMDIndex];

    circleRadius = computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3, circleCenterX, circleCenterY);

    if (not(passRZConstraint(acc,
                             modulesInGPU,
                             mdsInGPU,
                             segmentsInGPU,
                             innerInnerLowerModuleIndex,
                             middleLowerModuleIndex,
                             outerOuterLowerModuleIndex,
                             firstMDIndex,
                             secondMDIndex,
                             thirdMDIndex,
                             circleRadius,
                             circleCenterX, 
                             circleCenterY)))
      return false;

    return true;
  };

  struct createTripletsInGPUv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::MiniDoublets mdsInGPU,
                                  lst::Segments segmentsInGPU,
                                  lst::Triplets tripletsInGPU,
                                  lst::ObjectRanges rangesInGPU,
                                  uint16_t* index_gpu,
                                  uint16_t nonZeroModules,
                                  const float ptCut) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t innerLowerModuleArrayIdx = globalThreadIdx[0]; innerLowerModuleArrayIdx < nonZeroModules;
           innerLowerModuleArrayIdx += gridThreadExtent[0]) {
        uint16_t innerInnerLowerModuleIndex = index_gpu[innerLowerModuleArrayIdx];
        if (innerInnerLowerModuleIndex >= *modulesInGPU.nLowerModules)
          continue;

        uint16_t nConnectedModules = modulesInGPU.nConnectedModules[innerInnerLowerModuleIndex];
        if (nConnectedModules == 0)
          continue;

        unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
        for (unsigned int innerSegmentArrayIndex = globalThreadIdx[1]; innerSegmentArrayIndex < nInnerSegments;
             innerSegmentArrayIndex += gridThreadExtent[1]) {
          unsigned int innerSegmentIndex =
              rangesInGPU.segmentRanges[innerInnerLowerModuleIndex * 2] + innerSegmentArrayIndex;

          // middle lower module - outer lower module of inner segment
          uint16_t middleLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

          unsigned int nOuterSegments = segmentsInGPU.nSegments[middleLowerModuleIndex];
          for (unsigned int outerSegmentArrayIndex = globalThreadIdx[2]; outerSegmentArrayIndex < nOuterSegments;
               outerSegmentArrayIndex += gridThreadExtent[2]) {
            unsigned int outerSegmentIndex =
                rangesInGPU.segmentRanges[2 * middleLowerModuleIndex] + outerSegmentArrayIndex;

            uint16_t outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

            float zOut, rtOut, betaIn, betaInCut, circleRadius, circleCenterX, circleCenterY;

            bool success = runTripletConstraintsAndAlgo(acc,
                                                        modulesInGPU,
                                                        mdsInGPU,
                                                        segmentsInGPU,
                                                        innerInnerLowerModuleIndex,
                                                        middleLowerModuleIndex,
                                                        outerOuterLowerModuleIndex,
                                                        innerSegmentIndex,
                                                        outerSegmentIndex,
                                                        zOut,
                                                        rtOut,
                                                        betaIn,
                                                        betaInCut,
                                                        circleRadius,
                                                        circleCenterX,
                                                        circleCenterY,
                                                        ptCut);

            if (success) {
              unsigned int totOccupancyTriplets = alpaka::atomicOp<alpaka::AtomicAdd>(
                  acc, &tripletsInGPU.totOccupancyTriplets[innerInnerLowerModuleIndex], 1u);
              if (static_cast<int>(totOccupancyTriplets) >=
                  rangesInGPU.tripletModuleOccupancy[innerInnerLowerModuleIndex]) {
#ifdef WARNINGS
                printf("Triplet excess alert! Module index = %d, Occupancy = %d\n",
                       innerInnerLowerModuleIndex,
                       totOccupancyTriplets);
#endif
              } else {
                unsigned int tripletModuleIndex =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, &tripletsInGPU.nTriplets[innerInnerLowerModuleIndex], 1u);
                unsigned int tripletIndex =
                    rangesInGPU.tripletModuleIndices[innerInnerLowerModuleIndex] + tripletModuleIndex;
#ifdef CUT_VALUE_DEBUG
                addTripletToMemory(modulesInGPU,
                                   mdsInGPU,
                                   segmentsInGPU,
                                   tripletsInGPU,
                                   innerSegmentIndex,
                                   outerSegmentIndex,
                                   innerInnerLowerModuleIndex,
                                   middleLowerModuleIndex,
                                   outerOuterLowerModuleIndex,
                                   zOut,
                                   rtOut,
                                   betaIn,
                                   betaInCut,
                                   circleRadius,
                                   circleCenterX,
                                   circleCenterY,
                                   tripletIndex);
#else
                addTripletToMemory(modulesInGPU,
                                   mdsInGPU,
                                   segmentsInGPU,
                                   tripletsInGPU,
                                   innerSegmentIndex,
                                   outerSegmentIndex,
                                   innerInnerLowerModuleIndex,
                                   middleLowerModuleIndex,
                                   outerOuterLowerModuleIndex,
                                   betaIn,
                                   circleRadius,
                                   circleCenterX,
                                   circleCenterY,
                                   tripletIndex);
#endif
              }
            }
          }
        }
      }
    }
  };

  struct createTripletArrayRanges {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::ObjectRanges rangesInGPU,
                                  lst::Segments segmentsInGPU,
                                  const float ptCut) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      // Initialize variables in shared memory and set to 0
      int& nTotalTriplets = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc)) {
        nTotalTriplets = 0;
      }
      alpaka::syncBlockThreads(acc);

      // Occupancy matrix for 0.8 GeV pT Cut
      constexpr int p08_occupancy_matrix[4][4] = {
          {543, 235, 88, 46},  // category 0
          {755, 347, 0, 0},    // category 1
          {0, 0, 0, 0},        // category 2
          {0, 38, 46, 39}      // category 3
      };

      // Occupancy matrix for 0.6 GeV pT Cut, 99.9%
      constexpr int p06_occupancy_matrix[4][4] = {
          {1146, 544, 216, 83},  // category 0
          {1032, 275, 0, 0},     // category 1
          {0, 0, 0, 0},          // category 2
          {0, 115, 110, 76}      // category 3
      };

      // Select the appropriate occupancy matrix based on ptCut
      const auto& occupancy_matrix = (ptCut < 0.8f) ? p06_occupancy_matrix : p08_occupancy_matrix;

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
        if (segmentsInGPU.nSegments[i] == 0) {
          rangesInGPU.tripletModuleIndices[i] = nTotalTriplets;
          rangesInGPU.tripletModuleOccupancy[i] = 0;
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
          printf("Unhandled case in createTripletArrayRanges! Module index = %i\n", i);
        }
#endif

        rangesInGPU.tripletModuleOccupancy[i] = occupancy;
        unsigned int nTotT = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nTotalTriplets, occupancy);
        rangesInGPU.tripletModuleIndices[i] = nTotT;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (globalThreadIdx[2] == 0) {
        *rangesInGPU.device_nTotalTrips = nTotalTriplets;
      }
    }
  };

  struct addTripletRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::Triplets tripletsInGPU,
                                  lst::ObjectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
        if (tripletsInGPU.nTriplets[i] == 0) {
          rangesInGPU.tripletRanges[i * 2] = -1;
          rangesInGPU.tripletRanges[i * 2 + 1] = -1;
        } else {
          rangesInGPU.tripletRanges[i * 2] = rangesInGPU.tripletModuleIndices[i];
          rangesInGPU.tripletRanges[i * 2 + 1] = rangesInGPU.tripletModuleIndices[i] + tripletsInGPU.nTriplets[i] - 1;
        }
      }
    }
  };
}  // namespace lst
#endif
