#ifndef RecoTracker_LSTCore_src_alpaka_Quadruplet_h
#define RecoTracker_LSTCore_src_alpaka_Quadruplet_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"

#include "NeuralNetwork.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "ObjectRanges.h"
#include "Triplet.h"

namespace lst {
  struct Quadruplets {
    unsigned int* tripletIndices;
    uint16_t* lowerModuleIndices;
    unsigned int* nQuadruplets;
    unsigned int* totOccupancyQuadruplets;
    unsigned int* nMemoryLocations;

    FPX* innerRadius;
    FPX* outerRadius;
    FPX* pt;
    FPX* eta;
    FPX* phi;
    uint8_t* layer;
    char* isDup;

    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    float* rzChiSquared;

    template <typename TBuff>
    void setData(TBuff& buf) {
      tripletIndices = alpaka::getPtrNative(buf.tripletIndices_buf);
      lowerModuleIndices = alpaka::getPtrNative(buf.lowerModuleIndices_buf);
      nQuadruplets = alpaka::getPtrNative(buf.nQuadruplets_buf);
      totOccupancyQuadruplets = alpaka::getPtrNative(buf.totOccupancyQuadruplets_buf);
      nMemoryLocations = alpaka::getPtrNative(buf.nMemoryLocations_buf);
      innerRadius = alpaka::getPtrNative(buf.innerRadius_buf);
      outerRadius = alpaka::getPtrNative(buf.outerRadius_buf);
      pt = alpaka::getPtrNative(buf.pt_buf);
      eta = alpaka::getPtrNative(buf.eta_buf);
      phi = alpaka::getPtrNative(buf.phi_buf);
      layer = alpaka::getPtrNative(buf.layer_buf);
      isDup = alpaka::getPtrNative(buf.isDup_buf);
      logicalLayers = alpaka::getPtrNative(buf.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(buf.hitIndices_buf);
      rzChiSquared = alpaka::getPtrNative(buf.rzChiSquared_buf);
    }
  };

  template <typename TDev>
  struct QuadrupletsBuffer {
    Buf<TDev, unsigned int> tripletIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, unsigned int> nQuadruplets_buf;
    Buf<TDev, unsigned int> totOccupancyQuadruplets_buf;
    Buf<TDev, unsigned int> nMemoryLocations_buf;

    Buf<TDev, FPX> innerRadius_buf;
    Buf<TDev, FPX> outerRadius_buf;
    Buf<TDev, FPX> pt_buf;
    Buf<TDev, FPX> eta_buf;
    Buf<TDev, FPX> phi_buf;
    Buf<TDev, uint8_t> layer_buf;
    Buf<TDev, char> isDup_buf;

    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, float> rzChiSquared_buf;

    Quadruplets data_;

    template <typename TQueue, typename TDevAcc>
    QuadrupletsBuffer(unsigned int nTotalQuadruplets, unsigned int nLowerModules, TDevAcc const& devAccIn, TQueue& queue)
        : tripletIndices_buf(allocBufWrapper<unsigned int>(devAccIn, 2 * nTotalQuadruplets, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, Params_T4::kLayers * nTotalQuadruplets, queue)),
          nQuadruplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          totOccupancyQuadruplets_buf(allocBufWrapper<unsigned int>(devAccIn, nLowerModules, queue)),
          nMemoryLocations_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          innerRadius_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuadruplets, queue)),
          outerRadius_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuadruplets, queue)),
          pt_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuadruplets, queue)),
          eta_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuadruplets, queue)),
          phi_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuadruplets, queue)),
          layer_buf(allocBufWrapper<uint8_t>(devAccIn, nTotalQuadruplets, queue)),
          isDup_buf(allocBufWrapper<char>(devAccIn, nTotalQuadruplets, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, Params_T4::kLayers * nTotalQuadruplets, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, Params_T4::kHits * nTotalQuadruplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)) {
      alpaka::memset(queue, nQuadruplets_buf, 0u);
      alpaka::memset(queue, totOccupancyQuadruplets_buf, 0u);
      alpaka::memset(queue, isDup_buf, 0u);
      alpaka::wait(queue);
    }

    inline Quadruplets const* data() const { return &data_; }
    inline void setData(QuadrupletsBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkIntervalOverlapT4(float firstMin,
                                                           float firstMax,
                                                           float secondMin,
                                                           float secondMax) {
    return ((firstMin <= secondMin) && (secondMin < firstMax)) || ((secondMin < firstMin) && (firstMin < secondMax));
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addQuadrupletToMemory(lst::Triplets const& tripletsInGPU,
                                                            lst::Quadruplets& quadrupletsInGPU,
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
                                                            // float scores,
                                                            uint8_t layer,
                                                            unsigned int quadrupletIndex) {
    quadrupletsInGPU.tripletIndices[2 * quadrupletIndex] = innerTripletIndex;
    quadrupletsInGPU.tripletIndices[2 * quadrupletIndex + 1] = outerTripletIndex;

    quadrupletsInGPU.lowerModuleIndices[Params_T4::kLayers * quadrupletIndex] = lowerModule1;
    quadrupletsInGPU.lowerModuleIndices[Params_T4::kLayers * quadrupletIndex + 1] = lowerModule2;
    quadrupletsInGPU.lowerModuleIndices[Params_T4::kLayers * quadrupletIndex + 2] = lowerModule3;
    quadrupletsInGPU.lowerModuleIndices[Params_T4::kLayers * quadrupletIndex + 3] = lowerModule4;
    quadrupletsInGPU.innerRadius[quadrupletIndex] = __F2H(innerRadius);
    quadrupletsInGPU.outerRadius[quadrupletIndex] = __F2H(outerRadius);
    quadrupletsInGPU.pt[quadrupletIndex] = __F2H(pt);
    quadrupletsInGPU.eta[quadrupletIndex] = __F2H(eta);
    quadrupletsInGPU.phi[quadrupletIndex] = __F2H(phi);
    quadrupletsInGPU.layer[quadrupletIndex] = layer;
    quadrupletsInGPU.isDup[quadrupletIndex] = 0;
    quadrupletsInGPU.logicalLayers[Params_T4::kLayers * quadrupletIndex] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex];
    quadrupletsInGPU.logicalLayers[Params_T4::kLayers * quadrupletIndex + 1] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex + 1];
    quadrupletsInGPU.logicalLayers[Params_T4::kLayers * quadrupletIndex + 2] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex + 2];
    quadrupletsInGPU.logicalLayers[Params_T4::kLayers * quadrupletIndex + 3] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * outerTripletIndex + 1];

    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex];
    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex + 1] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 1];
    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex + 2] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 2];
    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex + 3] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 3];
    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex + 4] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 4];
    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex + 5] =
        tripletsInGPU.hitIndices[Params_T3::kHits * innerTripletIndex + 5];
    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex + 6] =
        tripletsInGPU.hitIndices[Params_T3::kHits * outerTripletIndex + 2];
    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex + 7] =
        tripletsInGPU.hitIndices[Params_T3::kHits * outerTripletIndex + 3];
  };

  //bounds can be found at http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_RZFix/t5_rz_thresholds.txt
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passT4RZConstraint(TAcc const& acc,
                                                         lst::Modules const& modulesInGPU,
                                                         lst::MiniDoublets const& mdsInGPU,
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
                                                         float inner_pt,
                                                         float innerRadius,
                                                         float g,
                                                         float f) {
    //(g,f) is the center of the circle fitted by the innermost 3 points on x,y coordinates
    const float& rt1 = mdsInGPU.anchorRt[firstMDIndex] / 100;  //in the unit of m instead of cm
    const float& rt2 = mdsInGPU.anchorRt[secondMDIndex] / 100;
    const float& rt3 = mdsInGPU.anchorRt[thirdMDIndex] / 100;
    const float& rt4 = mdsInGPU.anchorRt[fourthMDIndex] / 100;
    const float& rt5 = mdsInGPU.anchorRt[fifthMDIndex] / 100;

    const float& z1 = mdsInGPU.anchorZ[firstMDIndex] / 100;
    const float& z2 = mdsInGPU.anchorZ[secondMDIndex] / 100;
    const float& z3 = mdsInGPU.anchorZ[thirdMDIndex] / 100;
    const float& z4 = mdsInGPU.anchorZ[fourthMDIndex] / 100;
    const float& z5 = mdsInGPU.anchorZ[fifthMDIndex] / 100;

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer1 = modulesInGPU.lstLayers[lowerModuleIndex1];
    const int layer2 = modulesInGPU.lstLayers[lowerModuleIndex2];
    const int layer3 = modulesInGPU.lstLayers[lowerModuleIndex3];
    const int layer4 = modulesInGPU.lstLayers[lowerModuleIndex4];
    const int layer5 = modulesInGPU.lstLayers[lowerModuleIndex5];

    //slope computed using the internal T3s
    const int moduleType1 = modulesInGPU.moduleType[lowerModuleIndex1];  //0 is ps, 1 is 2s
    const int moduleType2 = modulesInGPU.moduleType[lowerModuleIndex2];
    const int moduleType3 = modulesInGPU.moduleType[lowerModuleIndex3];
    const int moduleType4 = modulesInGPU.moduleType[lowerModuleIndex4];
    const int moduleType5 = modulesInGPU.moduleType[lowerModuleIndex5];

    const float& x1 = mdsInGPU.anchorX[firstMDIndex] / 100;
    const float& x2 = mdsInGPU.anchorX[secondMDIndex] / 100;
    const float& x3 = mdsInGPU.anchorX[thirdMDIndex] / 100;
    const float& x4 = mdsInGPU.anchorX[fourthMDIndex] / 100;
    const float& y1 = mdsInGPU.anchorY[firstMDIndex] / 100;
    const float& y2 = mdsInGPU.anchorY[secondMDIndex] / 100;
    const float& y3 = mdsInGPU.anchorY[thirdMDIndex] / 100;
    const float& y4 = mdsInGPU.anchorY[fourthMDIndex] / 100;

    float residual = 0;
    float error2 = 0;
    float x_center = g / 100, y_center = f / 100;
    float x_init = mdsInGPU.anchorX[thirdMDIndex] / 100;
    float y_init = mdsInGPU.anchorY[thirdMDIndex] / 100;
    float z_init = mdsInGPU.anchorZ[thirdMDIndex] / 100;
    float rt_init = mdsInGPU.anchorRt[thirdMDIndex] / 100;  //use the second MD as initial point

    if (moduleType3 == 1)  // 1: if MD3 is in 2s layer
    {
      x_init = mdsInGPU.anchorX[secondMDIndex] / 100;
      y_init = mdsInGPU.anchorY[secondMDIndex] / 100;
      z_init = mdsInGPU.anchorZ[secondMDIndex] / 100;
      rt_init = mdsInGPU.anchorRt[secondMDIndex] / 100;
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

      // calculation is copied from PixelTriplet.cc lst::computePT3RZChiSquared
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
        drdz = alpaka::math::abs(acc, modulesInGPU.drdzs[lowerModuleIndex2]);
        side = modulesInGPU.sides[lowerModuleIndex2];
        subdets = modulesInGPU.subdets[lowerModuleIndex2];
      }
      if (i == 3) {
        drdz = alpaka::math::abs(acc, modulesInGPU.drdzs[lowerModuleIndex3]);
        side = modulesInGPU.sides[lowerModuleIndex3];
        subdets = modulesInGPU.subdets[lowerModuleIndex3];
      }
      if (i == 2 || i == 3) {
        residual = (layeri <= 6 && ((side == lst::Center) or (drdz < 1))) ? diffz : diffr;
        float projection_missing2 = 1.f;
        if (drdz < 1)
          projection_missing2 = ((subdets == lst::Endcap) or (side == lst::Center))
                                    ? 1.f
                                    : 1.f / (1 + drdz * drdz);  // cos(atan(drdz)), if dr/dz<1
        if (drdz > 1)
          projection_missing2 = ((subdets == lst::Endcap) or (side == lst::Center))
                                    ? 1.f
                                    : (drdz * drdz) / (1 + drdz * drdz);  //sin(atan(drdz)), if dr/dz>1
        error2 = error2 * projection_missing2;
      }
      rzChiSquared += 12 * (residual * residual) / error2;
    }
    // for set rzchi2 cut
    // if the 5 points are linear, helix calculation gives nan
    // Alpaka : Needs to be moved over
    if (inner_pt > 100 || alpaka::math::isnan(acc, rzChiSquared)) {
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
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4HasCommonMiniDoublet(lst::Triplets const& tripletsInGPU,
                                                             lst::Segments const& segmentsInGPU,
                                                             unsigned int innerTripletIndex,
                                                             unsigned int outerTripletIndex) {
    unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int innerOuterOuterMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * innerOuterSegmentIndex + 1];  //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * outerInnerSegmentIndex];  //outer triplet inner segment inner MD index

    return (innerOuterOuterMiniDoubletIndex == outerInnerInnerMiniDoubletIndex);
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeErrorInRadiusT4(TAcc const& acc,
                                                           float* x1Vec,
                                                           float* y1Vec,
                                                           float* x2Vec,
                                                           float* y2Vec,
                                                           float* x3Vec,
                                                           float* y3Vec,
                                                           float& minimumRadius,
                                                           float& maximumRadius) {
    //brute force
    float candidateRadius;
    float g, f;
    minimumRadius = lst::lst_INF;
    maximumRadius = 0.f;
    for (size_t i = 0; i < 3; i++) {
      float x1 = x1Vec[i];
      float y1 = y1Vec[i];
      for (size_t j = 0; j < 3; j++) {
        float x2 = x2Vec[j];
        float y2 = y2Vec[j];
        for (size_t k = 0; k < 3; k++) {
          float x3 = x3Vec[k];
          float y3 = y3Vec[k];
          candidateRadius = computeRadiusFromThreeAnchorHits(acc, x1, y1, x2, y2, x3, y3, g, f);
          maximumRadius = alpaka::math::max(acc, candidateRadius, maximumRadius);
          minimumRadius = alpaka::math::min(acc, candidateRadius, minimumRadius);
        }
      }
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4matchRadiiBBBEE12378(TAcc const& acc,
                                                           float innerRadius,
                                                           float outerRadius,
                                                           float outerRadiusMin2S,
                                                           float outerRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    float innerInvRadiusErrorBound = 0.178f;
    float outerInvRadiusErrorBound = 0.507f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlapT4(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, outerInvRadiusMin, 1.0f / outerRadiusMax2S),
                                alpaka::math::max(acc, outerInvRadiusMax, 1.0f / outerRadiusMin2S));
  };

  /*bounds for high Pt taken from : http://uaf-10.t2.ucsd.edu/~bsathian/SDL/T5_efficiency/efficiencies/new_efficiencies/efficiencies_20210513_T5_recovering_high_Pt_efficiencies/highE_radius_matching/highE_bounds.txt */
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4matchRadiiBBBBB(TAcc const& acc,
                                                      float innerRadius,
                                                      float outerRadius) {
    float innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    float innerInvRadiusErrorBound = 0.1512f;
    float outerInvRadiusErrorBound = 0.1781f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf)) {
      innerInvRadiusErrorBound = 0.4449f;
      outerInvRadiusErrorBound = 0.4033f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlapT4(innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4matchRadiiBBBBE(TAcc const& acc,
                                                      float innerRadius,
                                                      float outerRadius) {
    float innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    float innerInvRadiusErrorBound = 0.1781f;
    float outerInvRadiusErrorBound = 0.2167f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf)) {
      innerInvRadiusErrorBound = 0.4750f;
      outerInvRadiusErrorBound = 0.3903f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlapT4(innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax);
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4matchRadiiBBBEE23478(TAcc const& acc,
                                                           float innerRadius,
                                                           float outerRadius,
                                                           float outerRadiusMin2S,
                                                           float outerRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    float innerInvRadiusErrorBound = 0.2097f;
    float outerInvRadiusErrorBound = 0.8557f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlapT4(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, outerInvRadiusMin, 1.0f / outerRadiusMax2S),
                                alpaka::math::max(acc, outerInvRadiusMax, 1.0f / outerRadiusMin2S));
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4matchRadiiBBBEE34578(TAcc const& acc,
                                                           float innerRadius,
                                                           float outerRadius,
                                                           float outerRadiusMin2S,
                                                           float outerRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    float innerInvRadiusErrorBound = 0.066f;
    float outerInvRadiusErrorBound = 0.617f;

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlapT4(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, outerInvRadiusMin, 1.0f / outerRadiusMax2S),
                                alpaka::math::max(acc, outerInvRadiusMax, 1.0f / outerRadiusMin2S));
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4matchRadiiBBEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float outerRadius,
                                                      float outerRadiusMin2S,
                                                      float outerRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    float innerInvRadiusErrorBound = 0.6376f;
    float outerInvRadiusErrorBound = 2.1381f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf))  //as good as no selections!
    {
      innerInvRadiusErrorBound = 12.9173f;
      outerInvRadiusErrorBound = 5.1700f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlapT4(innerInvRadiusMin,
                                innerInvRadiusMax,
                                alpaka::math::min(acc, outerInvRadiusMin, 1.0f / outerRadiusMax2S),
                                alpaka::math::max(acc, outerInvRadiusMax, 1.0f / outerRadiusMin2S));
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4matchRadiiBEEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float outerRadius,
                                                      float innerRadiusMin2S,
                                                      float innerRadiusMax2S,
                                                      float outerRadiusMin2S,
                                                      float outerRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    float innerInvRadiusErrorBound = 1.9382f;
    float outerInvRadiusErrorBound = 3.7280f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf)) {
      innerInvRadiusErrorBound = 23.2713f;
      outerInvRadiusErrorBound = 21.7980f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlapT4(alpaka::math::min(acc, innerInvRadiusMin, 1.0 / innerRadiusMax2S),
                                alpaka::math::max(acc, innerInvRadiusMax, 1.0 / innerRadiusMin2S),
                                alpaka::math::min(acc, outerInvRadiusMin, 1.0 / outerRadiusMax2S),
                                alpaka::math::max(acc, outerInvRadiusMax, 1.0 / outerRadiusMin2S));
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool T4matchRadiiEEEEE(TAcc const& acc,
                                                      float innerRadius,
                                                      float outerRadius,
                                                      float innerRadiusMin2S,
                                                      float innerRadiusMax2S,
                                                      float outerRadiusMin2S,
                                                      float outerRadiusMax2S) {
    float innerInvRadiusMin, innerInvRadiusMax, outerInvRadiusMin, outerInvRadiusMax;

    float innerInvRadiusErrorBound = 1.9382f;
    float outerInvRadiusErrorBound = 2.2091f;

    if (innerRadius > 2.0f / (2.f * k2Rinv1GeVf)) {
      innerInvRadiusErrorBound = 22.5226f;
      outerInvRadiusErrorBound = 21.0966f;
    }

    innerInvRadiusMax = (1.f + innerInvRadiusErrorBound) / innerRadius;
    innerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - innerInvRadiusErrorBound) / innerRadius);

    outerInvRadiusMax = (1.f + outerInvRadiusErrorBound) / outerRadius;
    outerInvRadiusMin = alpaka::math::max(acc, 0.f, (1.f - outerInvRadiusErrorBound) / outerRadius);

    return checkIntervalOverlapT4(alpaka::math::min(acc, innerInvRadiusMin, 1.0 / innerRadiusMax2S),
                                alpaka::math::max(acc, innerInvRadiusMax, 1.0 / innerRadiusMin2S),
                                alpaka::math::min(acc, outerInvRadiusMin, 1.0 / outerRadiusMax2S),
                                alpaka::math::max(acc, outerInvRadiusMax, 1.0 / outerRadiusMin2S));
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void computeSigmasForRegressionT4(TAcc const& acc,
                                                                 lst::Modules const& modulesInGPU,
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
      moduleType = modulesInGPU.moduleType[lowerModuleIndices[i]];
      moduleSubdet = modulesInGPU.subdets[lowerModuleIndices[i]];
      moduleSide = modulesInGPU.sides[lowerModuleIndices[i]];
      const float& drdz = modulesInGPU.drdzs[lowerModuleIndices[i]];
      slopes[i] = modulesInGPU.dxdys[lowerModuleIndices[i]];
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
      absArctanSlope = ((slopes[i] != lst::lst_INF) ? alpaka::math::abs(acc, alpaka::math::atan(acc, slopes[i]))
                                                    : 0.5f * float(M_PI));
      if (xs[i] > 0 and ys[i] > 0) {
        angleM = 0.5f * float(M_PI) - absArctanSlope;
      } else if (xs[i] < 0 and ys[i] > 0) {
        angleM = absArctanSlope + 0.5f * float(M_PI);
      } else if (xs[i] < 0 and ys[i] < 0) {
        angleM = -(absArctanSlope + 0.5f * float(M_PI));
      } else if (xs[i] > 0 and ys[i] < 0) {
        angleM = -(0.5f * float(M_PI) - absArctanSlope);
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
              alpaka::math::min(acc, sdOut_dr * lst::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), lst::kSinAlphaMax)),
          betaOut);
      return;
    }

    if (betaIn * betaOut > 0.f and
        (alpaka::math::abs(acc, pt_beta) < 4.f * lst::kPt_betaMax or
         (lIn >= 11 and alpaka::math::abs(acc, pt_beta) <
                            8.f * lst::kPt_betaMax)))  //and the pt_beta is well-defined; less strict for endcap-endcap
    {
      const float betaInUpd =
          betaIn + alpaka::math::copysign(
                       acc,
                       alpaka::math::asin(
                           acc,
                           alpaka::math::min(
                               acc, sdIn_dr * lst::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), lst::kSinAlphaMax)),
                       betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut + alpaka::math::copysign(
                        acc,
                        alpaka::math::asin(
                            acc,
                            alpaka::math::min(
                                acc, sdOut_dr * lst::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), lst::kSinAlphaMax)),
                        betaOut);  //FIXME: need a faster version
      betaAv = 0.5f * (betaInUpd + betaOutUpd);

      //1st update
      const float pt_beta_inv =
          1.f / alpaka::math::abs(acc, dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaAv));  //get a better pt estimate

      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(acc, alpaka::math::min(acc, sdIn_dr * lst::k2Rinv1GeVf * pt_beta_inv, lst::kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(acc, alpaka::math::min(acc, sdOut_dr * lst::k2Rinv1GeVf * pt_beta_inv, lst::kSinAlphaMax)),
          betaOut);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * lst::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    } else if (lIn < 11 && alpaka::math::abs(acc, betaOut) < 0.2f * alpaka::math::abs(acc, betaIn) &&
               alpaka::math::abs(acc, pt_beta) < 12.f * lst::kPt_betaMax)  //use betaIn sign as ref
    {
      const float pt_betaIn = dr * k2Rinv1GeVf / alpaka::math::sin(acc, betaIn);

      const float betaInUpd =
          betaIn + alpaka::math::copysign(
                       acc,
                       alpaka::math::asin(
                           acc,
                           alpaka::math::min(
                               acc, sdIn_dr * lst::k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), lst::kSinAlphaMax)),
                       betaIn);  //FIXME: need a faster version
      const float betaOutUpd =
          betaOut +
          alpaka::math::copysign(
              acc,
              alpaka::math::asin(
                  acc,
                  alpaka::math::min(
                      acc, sdOut_dr * lst::k2Rinv1GeVf / alpaka::math::abs(acc, pt_betaIn), lst::kSinAlphaMax)),
              betaIn);  //FIXME: need a faster version
      betaAv = (alpaka::math::abs(acc, betaOut) > 0.2f * alpaka::math::abs(acc, betaIn))
                   ? (0.5f * (betaInUpd + betaOutUpd))
                   : betaInUpd;

      //1st update
      pt_beta = dr * lst::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
      betaIn += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc,
              alpaka::math::min(acc, sdIn_dr * lst::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), lst::kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      betaOut += alpaka::math::copysign(
          acc,
          alpaka::math::asin(
              acc,
              alpaka::math::min(acc, sdOut_dr * lst::k2Rinv1GeVf / alpaka::math::abs(acc, pt_beta), lst::kSinAlphaMax)),
          betaIn);  //FIXME: need a faster version
      //update the av and pt
      betaAv = 0.5f * (betaIn + betaOut);
      //2nd update
      pt_beta = dr * lst::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);  //get a better pt estimate
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletDefaultAlgoBBBB(TAcc const& acc,
                                                                   lst::Modules const& modulesInGPU,
                                                                   lst::MiniDoublets const& mdsInGPU,
                                                                   lst::Segments const& segmentsInGPU,
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
                                                                   const float ptCut) {
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == lst::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == lst::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    float rtRatio_OutLoInLo = rt_OutLo / rt_InLo;  // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);

    float zHi = z_InLo + (z_InLo + lst::kDeltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) +
                (zpitch_InLo + zpitch_OutLo);
    float zLo = z_InLo + (z_InLo - lst::kDeltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) -
                (zpitch_InLo + zpitch_OutLo);

    //Cut 1 - z compatibility
    if ((z_OutLo < zLo) || (z_OutLo > zHi))
      return false;

    float drt_OutLo_InLo = (rt_OutLo - rt_InLo);
    float r3_InLo = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = rt_InOut - rt_InLo;
    float dz_InSeg = z_InOut - z_InLo;
    float dr3_InSeg = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                      alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

    float coshEta = dr3_InSeg / drt_InSeg;
    float dzErr = (zpitch_InLo + zpitch_OutLo) * (zpitch_InLo + zpitch_OutLo) * 2.f;

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (r3_InLo / rt_InLo);
    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    dzErr += muls2 * drt_OutLo_InLo * drt_OutLo_InLo / 3.f * coshEta * coshEta;
    dzErr = alpaka::math::sqrt(acc, dzErr);

    // Constructing upper and lower bound
    const float dzMean = dz_InSeg / drt_InSeg * drt_OutLo_InLo;
    const float zWindow =
        dzErr / drt_InSeg * drt_OutLo_InLo +
        (zpitch_InLo + zpitch_OutLo);  //FIXME for lst::ptCut lower than ~0.8 need to add curv path correction
    float zLoPointed = z_InLo + dzMean * (z_InLo > 0.f ? 1.f : dzDrtScale) - zWindow;
    float zHiPointed = z_InLo + dzMean * (z_InLo < 0.f ? 1.f : dzDrtScale) + zWindow;

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment inner MD)
    if ((z_OutLo < zLoPointed) || (z_OutLo > zHiPointed))
      return false;

    float pvOffset = 0.1f / rt_OutLo;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[secondMDIndex]);
    // Cut #3: FIXME:deltaPhiPos can be tighter
    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float dPhi = lst::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // Cut #4: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    // First obtaining the raw betaIn and betaOut values without any correction and just purely based on the mini-doublet hit positions

    float alpha_InLo = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float alpha_OutLo = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);

    bool isEC_lastLayer = modulesInGPU.subdets[outerOuterLowerModuleIndex] == lst::Endcap and
                          modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::TwoS;

    float alpha_OutUp, alpha_OutUp_highEdge, alpha_OutUp_lowEdge;

    alpha_OutUp = lst::phi_mpi_pi(acc,
                                  lst::phi(acc,
                                           mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex],
                                           mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) -
                                      mdsInGPU.anchorPhi[fourthMDIndex]);

    alpha_OutUp_highEdge = alpha_OutUp;
    alpha_OutUp_lowEdge = alpha_OutUp;

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
    float tl_axis_highEdge_x = tl_axis_x;
    float tl_axis_highEdge_y = tl_axis_y;
    float tl_axis_lowEdge_x = tl_axis_x;
    float tl_axis_lowEdge_y = tl_axis_y;

    float betaIn =
        alpha_InLo - lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut =
        -alpha_OutUp + lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    if (isEC_lastLayer) {
      alpha_OutUp_highEdge =
          lst::phi_mpi_pi(acc,
                          lst::phi(acc,
                                   mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex],
                                   mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) -
                              mdsInGPU.anchorHighEdgePhi[fourthMDIndex]);
      alpha_OutUp_lowEdge =
          lst::phi_mpi_pi(acc,
                          lst::phi(acc,
                                   mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex],
                                   mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) -
                              mdsInGPU.anchorLowEdgePhi[fourthMDIndex]);

      tl_axis_highEdge_x = mdsInGPU.anchorHighEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
      tl_axis_highEdge_y = mdsInGPU.anchorHighEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];
      tl_axis_lowEdge_x = mdsInGPU.anchorLowEdgeX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
      tl_axis_lowEdge_y = mdsInGPU.anchorLowEdgeY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];

      betaOutRHmin = -alpha_OutUp_highEdge + lst::phi_mpi_pi(acc,
                                                             lst::phi(acc, tl_axis_highEdge_x, tl_axis_highEdge_y) -
                                                                 mdsInGPU.anchorHighEdgePhi[fourthMDIndex]);
      betaOutRHmax = -alpha_OutUp_lowEdge + lst::phi_mpi_pi(acc,
                                                            lst::phi(acc, tl_axis_lowEdge_x, tl_axis_lowEdge_y) -
                                                                mdsInGPU.anchorLowEdgePhi[fourthMDIndex]);
    }

    //beta computation
    float drt_tl_axis = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

    float corrF = 1.f;
    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg =
        alpaka::math::sqrt(acc,
                           (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                   (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                   (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float betaInCut =
        alpaka::math::asin(
            acc,
            alpaka::math::min(acc, (-rt_InSeg * corrF + drt_tl_axis) * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / drt_InSeg);

    //Cut #5: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = drt_tl_axis * lst::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);
    int lIn = 5;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) *
                                                (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) +
                                            (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) *
                                                (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    lst::runDeltaBetaIterationsT4(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);

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
        acc, alpaka::math::abs(acc, pt_beta), lst::kPt_betaMax);  //need to confimm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg = alpaka::math::max(
        acc,
        alpaka::math::abs(acc, alpha_InLo),
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * lst::k2Rinv1GeVf / 3.0f, lst::kSinAlphaMax)));
    const float alphaOutAbsReg = alpaka::math::max(
        acc,
        alpaka::math::abs(acc, alpha_OutLo),
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / 3.0f, lst::kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * lst::kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * lst::kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = alpaka::math::sin(acc, dPhi);

    const float dBetaRIn2 = 0;  // TODO-RH
    float dBetaROut = 0;
    if (isEC_lastLayer) {
      dBetaROut =
          (alpaka::math::sqrt(acc,
                              mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) -
           alpaka::math::sqrt(acc,
                              mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) *
          sinDPhi / drt_tl_axis;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;

    float betaOutCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, drt_tl_axis * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));

    float dBeta = betaIn - betaOut;
    return dBeta * dBeta <= dBetaCut2;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletDefaultAlgoBBEE(TAcc const& acc,
                                                                   lst::Modules const& modulesInGPU,
                                                                   lst::MiniDoublets const& mdsInGPU,
                                                                   lst::Segments const& segmentsInGPU,
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
                                                                   const float ptCut) {
    bool isPS_InLo = (modulesInGPU.moduleType[innerInnerLowerModuleIndex] == lst::PS);
    bool isPS_OutLo = (modulesInGPU.moduleType[outerInnerLowerModuleIndex] == lst::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
    float zpitch_OutLo = (isPS_OutLo ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
    float zGeom = zpitch_InLo + zpitch_OutLo;

    // Cut #0: Preliminary (Only here in endcap case)
    if (z_InLo * z_OutLo <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, lst::kDeltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == lst::PS;
    float rtGeom1 = isOutSgInnerMDPS ? lst::kPixelPSZpitch : lst::kStrip2SZpitch;
    float zGeom1 = alpaka::math::copysign(acc, zGeom, z_InLo);
    float rtLo = rt_InLo * (1.f + (z_OutLo - z_InLo - zGeom1) / (z_InLo + zGeom1 + dLum) / dzDrtScale) -
                 rtGeom1;  //slope correction only on the lower end
    float rtOut = rt_OutLo;

    //Cut #1: rt condition
    if (rtOut < rtLo)
      return false;

    float zInForHi = z_InLo - zGeom1 - dLum;
    if (zInForHi * z_InLo < 0) {
      zInForHi = alpaka::math::copysign(acc, 0.1f, z_InLo);
    }
    float rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

    //Cut #2: rt condition
    if ((rt_OutLo < rtLo) || (rt_OutLo > rtHi))
      return false;

    float rIn = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                          alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);

    const float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    const float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InLo);
    const float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);
    const float zGeom1_another = lst::kPixelPSZpitch;
    float kZ = (z_OutLo - z_InLo) / dzSDIn;
    float drtErr =
        zGeom1_another * zGeom1_another * drtSDIn * drtSDIn / dzSDIn / dzSDIn * (1.f - 2.f * kZ + 2.f * kZ * kZ);
    const float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (rIn / rt_InLo);
    const float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;
    drtErr += muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta;
    drtErr = alpaka::math::sqrt(acc, drtErr);

    //Cut #3: rt-z pointed
    if ((kZ < 0) || (rtOut < rtLo) || (rtOut > rtHi))
      return false;

    const float pvOffset = 0.1f / rt_OutLo;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[secondMDIndex]);

    //Cut #4: deltaPhiPos can be tighter
    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float dPhi = lst::deltaPhi(acc, midPointX, midPointY, diffX, diffY);
    // Cut #5: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdIn_alpha_min = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alpha_max = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;

    float sdOut_alphaOut = lst::phi_mpi_pi(acc,
                                           lst::phi(acc,
                                                    mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex],
                                                    mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) -
                                               mdsInGPU.anchorPhi[fourthMDIndex]);

    float sdOut_alphaOut_min = lst::phi_mpi_pi(
        acc, __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMins[outerSegmentIndex]));
    float sdOut_alphaOut_max = lst::phi_mpi_pi(
        acc, __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]) - __H2F(segmentsInGPU.dPhiMaxs[outerSegmentIndex]));

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float betaIn =
        sdIn_alpha - lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float betaInRHmin = betaIn;
    float betaInRHmax = betaIn;
    float betaOut =
        -sdOut_alphaOut + lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[fourthMDIndex]);

    float betaOutRHmin = betaOut;
    float betaOutRHmax = betaOut;

    bool isEC_secondLayer = (modulesInGPU.subdets[innerOuterLowerModuleIndex] == lst::Endcap) and
                            (modulesInGPU.moduleType[innerOuterLowerModuleIndex] == lst::TwoS);

    if (isEC_secondLayer) {
      betaInRHmin = betaIn - sdIn_alpha_min + sdIn_alpha;
      betaInRHmax = betaIn - sdIn_alpha_max + sdIn_alpha;
    }

    betaOutRHmin = betaOut - sdOut_alphaOut_min + sdOut_alphaOut;
    betaOutRHmax = betaOut - sdOut_alphaOut_max + sdOut_alphaOut;

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
                                       (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                               (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                                           (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    const float corrF = 1.f;
    float betaInCut =
        alpaka::math::asin(
            acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / sdIn_d);

    //Cut #6: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * lst::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

    float lIn = 5;
    float lOut = 11;

    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) *
                                                (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) +
                                            (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) *
                                                (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    lst::runDeltaBetaIterationsT4(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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
        acc, alpaka::math::abs(acc, pt_beta), lst::kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg = alpaka::math::max(
        acc,
        alpaka::math::abs(acc, sdIn_alpha),
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * lst::k2Rinv1GeVf / 3.0f, lst::kSinAlphaMax)));
    const float alphaOutAbsReg = alpaka::math::max(
        acc,
        alpaka::math::abs(acc, sdOut_alpha),
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / 3.0f, lst::kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * lst::kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * lst::kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);
    const float sinDPhi = alpaka::math::sin(acc, dPhi);

    const float dBetaRIn2 = 0;  // TODO-RH
    float dBetaROut = 0;
    if (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::TwoS) {
      dBetaROut =
          (alpaka::math::sqrt(acc,
                              mdsInGPU.anchorHighEdgeX[fourthMDIndex] * mdsInGPU.anchorHighEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorHighEdgeY[fourthMDIndex] * mdsInGPU.anchorHighEdgeY[fourthMDIndex]) -
           alpaka::math::sqrt(acc,
                              mdsInGPU.anchorLowEdgeX[fourthMDIndex] * mdsInGPU.anchorLowEdgeX[fourthMDIndex] +
                                  mdsInGPU.anchorLowEdgeY[fourthMDIndex] * mdsInGPU.anchorLowEdgeY[fourthMDIndex])) *
          sinDPhi / dr;
    }

    const float dBetaROut2 = dBetaROut * dBetaROut;
    float betaOutCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, dr * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBet
    return dBeta * dBeta <= dBetaCut2;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletDefaultAlgoEEEE(TAcc const& acc,
                                                                   lst::Modules const& modulesInGPU,
                                                                   lst::MiniDoublets const& mdsInGPU,
                                                                   lst::Segments const& segmentsInGPU,
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
                                                                   const float ptCut) {
    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly

    // Cut #0: Preliminary (Only here in endcap case)
    if ((z_InLo * z_OutLo) <= 0)
      return false;

    float dLum = alpaka::math::copysign(acc, lst::kDeltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == lst::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == lst::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS)  ? 2.f * lst::kPixelPSZpitch
                   : (isInSgInnerMDPS or isOutSgInnerMDPS) ? lst::kPixelPSZpitch + lst::kStrip2SZpitch
                                                           : 2.f * lst::kStrip2SZpitch;

    float dz = z_OutLo - z_InLo;
    float rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom;  //slope correction only on the lower end

    float rtOut = rt_OutLo;

    //Cut #1: rt condition

    float rtHi = rt_InLo * (1.f + dz / (z_InLo - dLum)) + rtGeom;

    if ((rtOut < rtLo) || (rtOut > rtHi))
      return false;

    bool isInSgOuterMDPS = modulesInGPU.moduleType[innerOuterLowerModuleIndex] == lst::PS;

    const float drtSDIn = rt_InOut - rt_InLo;
    const float dzSDIn = z_InOut - z_InLo;
    const float dr3SDIn = alpaka::math::sqrt(acc, rt_InOut * rt_InOut + z_InOut * z_InOut) -
                          alpaka::math::sqrt(acc, rt_InLo * rt_InLo + z_InLo * z_InLo);
    float coshEta = dr3SDIn / drtSDIn;  //direction estimate
    float dzOutInAbs = alpaka::math::abs(acc, z_OutLo - z_InLo);
    float multDzDr = dzOutInAbs * coshEta / (coshEta * coshEta - 1.f);

    float kZ = (z_OutLo - z_InLo) / dzSDIn;
    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);

    float muls2 = thetaMuls2 * 9.f / (ptCut * ptCut) * 16.f;

    float drtErr = alpaka::math::sqrt(
        acc,
        lst::kPixelPSZpitch * lst::kPixelPSZpitch * 2.f / (dzSDIn * dzSDIn) * (dzOutInAbs * dzOutInAbs) +
            muls2 * multDzDr * multDzDr / 3.f * coshEta * coshEta);

    float drtMean = drtSDIn * dzOutInAbs / alpaka::math::abs(acc, dzSDIn);
    float rtWindow = drtErr + rtGeom;
    float rtLo_point = rt_InLo + drtMean / dzDrtScale - rtWindow;
    float rtHi_point = rt_InLo + drtMean + rtWindow;

    // Cut #3: rt-z pointed
    // https://github.com/slava77/cms-tkph2-ntuple/blob/superDoubletLinked-91X-noMock/doubletAnalysis.C#L3765

    if (isInSgInnerMDPS and isInSgOuterMDPS)  // If both PS then we can point
    {
      if (kZ < 0 || rtOut < rtLo_point || rtOut > rtHi_point)
        return false;
    }

    float pvOffset = 0.1f / rtOut;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[secondMDIndex]);

    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float dPhi = lst::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // Cut #5: deltaPhiChange
    if (alpaka::math::abs(acc, dPhi) > dPhiCut)
      return false;

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;  //weird
    float sdOut_dPhiPos = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[thirdMDIndex]);

    float sdOut_dPhiChange = __H2F(segmentsInGPU.dPhiChanges[outerSegmentIndex]);
    float sdOut_dPhiChange_min = __H2F(segmentsInGPU.dPhiChangeMins[outerSegmentIndex]);
    float sdOut_dPhiChange_max = __H2F(segmentsInGPU.dPhiChangeMaxs[outerSegmentIndex]);

    float sdOut_alphaOutRHmin = lst::phi_mpi_pi(acc, sdOut_dPhiChange_min - sdOut_dPhiPos);
    float sdOut_alphaOutRHmax = lst::phi_mpi_pi(acc, sdOut_dPhiChange_max - sdOut_dPhiPos);
    float sdOut_alphaOut = lst::phi_mpi_pi(acc, sdOut_dPhiChange - sdOut_dPhiPos);

    float tl_axis_x = mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float tl_axis_y = mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float betaIn =
        sdIn_alpha - lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[firstMDIndex]);

    float sdIn_alphaRHmin = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alphaRHmax = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float betaInRHmin = betaIn + sdIn_alphaRHmin - sdIn_alpha;
    float betaInRHmax = betaIn + sdIn_alphaRHmax - sdIn_alpha;

    float betaOut =
        -sdOut_alphaOut + lst::phi_mpi_pi(acc, lst::phi(acc, tl_axis_x, tl_axis_y) - mdsInGPU.anchorPhi[fourthMDIndex]);

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
                                       (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                               (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                                           (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);
    const float corrF = 1.f;
    float betaInCut =
        alpaka::math::asin(
            acc, alpaka::math::min(acc, (-sdIn_dr * corrF + dr) * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / sdIn_d);

    //Cut #6: first beta cut
    if (alpaka::math::abs(acc, betaInRHmin) >= betaInCut)
      return false;

    float betaAv = 0.5f * (betaIn + betaOut);
    float pt_beta = dr * lst::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);

    int lIn = 11;   //endcap
    int lOut = 13;  //endcap

    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) *
                                                (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) +
                                            (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) *
                                                (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    lst::runDeltaBetaIterationsT4(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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
        acc, alpaka::math::abs(acc, pt_beta), lst::kPt_betaMax);  //need to confirm the range-out value of 7 GeV
    const float dBetaMuls2 = thetaMuls2 * 16.f / (min_ptBeta_maxPtBeta * min_ptBeta_maxPtBeta);

    const float alphaInAbsReg = alpaka::math::max(
        acc,
        alpaka::math::abs(acc, sdIn_alpha),
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_InLo * lst::k2Rinv1GeVf / 3.0f, lst::kSinAlphaMax)));
    const float alphaOutAbsReg = alpaka::math::max(
        acc,
        alpaka::math::abs(acc, sdOut_alpha),
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / 3.0f, lst::kSinAlphaMax)));
    const float dBetaInLum = lIn < 11 ? 0.0f : alpaka::math::abs(acc, alphaInAbsReg * lst::kDeltaZLum / z_InLo);
    const float dBetaOutLum = lOut < 11 ? 0.0f : alpaka::math::abs(acc, alphaOutAbsReg * lst::kDeltaZLum / z_OutLo);
    const float dBetaLum2 = (dBetaInLum + dBetaOutLum) * (dBetaInLum + dBetaOutLum);

    const float dBetaRIn2 = 0;  // TODO-RH

    float dBetaROut2 = 0;  //TODO-RH
    float betaOutCut =
        alpaka::math::asin(acc, alpaka::math::min(acc, dr * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax)) +
        (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2);

    //Cut #6: The real beta cut
    if (alpaka::math::abs(acc, betaOut) >= betaOutCut)
      return false;

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    float dBeta = betaIn - betaOut;
    //Cut #7: Cut on dBeta
    return dBeta * dBeta <= dBetaCut2;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletAlgoSelector(TAcc const& acc,
                                                                lst::Modules const& modulesInGPU,
                                                                lst::MiniDoublets const& mdsInGPU,
                                                                lst::Segments const& segmentsInGPU,
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
                                                                const float ptCut) {
    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modulesInGPU.subdets[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == lst::Barrel and innerOuterLowerModuleSubdet == lst::Barrel and
        outerInnerLowerModuleSubdet == lst::Barrel and outerOuterLowerModuleSubdet == lst::Barrel) {
      return runQuadrupletDefaultAlgoBBBB(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
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
                                          ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Barrel and innerOuterLowerModuleSubdet == lst::Barrel and
               outerInnerLowerModuleSubdet == lst::Endcap and outerOuterLowerModuleSubdet == lst::Endcap) {
      return runQuadrupletDefaultAlgoBBEE(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
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
                                          ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Barrel and innerOuterLowerModuleSubdet == lst::Barrel and
               outerInnerLowerModuleSubdet == lst::Barrel and outerOuterLowerModuleSubdet == lst::Endcap) {
      return runQuadrupletDefaultAlgoBBBB(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
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
                                          ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Barrel and innerOuterLowerModuleSubdet == lst::Endcap and
               outerInnerLowerModuleSubdet == lst::Endcap and outerOuterLowerModuleSubdet == lst::Endcap) {
      return runQuadrupletDefaultAlgoBBEE(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
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
                                          ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Endcap and innerOuterLowerModuleSubdet == lst::Endcap and
               outerInnerLowerModuleSubdet == lst::Endcap and outerOuterLowerModuleSubdet == lst::Endcap) {
      return runQuadrupletDefaultAlgoEEEE(acc,
                                          modulesInGPU,
                                          mdsInGPU,
                                          segmentsInGPU,
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
                                          ptCut);
    }

    return false;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletDefaultAlgo(TAcc const& acc,
                                                               struct lst::Modules& modulesInGPU,
                                                               struct lst::MiniDoublets& mdsInGPU,
                                                               struct lst::Segments& segmentsInGPU,
                                                               struct lst::Triplets& tripletsInGPU,
                                                               uint16_t lowerModuleIndex1,
                                                               uint16_t lowerModuleIndex2,
                                                               uint16_t lowerModuleIndex3,
                                                               uint16_t lowerModuleIndex4,
                                                               uint16_t lowerModuleIndex5,
                                                               unsigned int innerTripletIndex,
                                                               unsigned int outerTripletIndex,
                                                               float& innerRadius,
                                                               float& outerRadius,
                                                               float& rzChiSquared,
                                                               float& chiSquared,
                                                               const float ptCut) {
    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];

    unsigned int innerOuterOuterMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];  //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * thirdSegmentIndex];  //outer triplet inner segment inner MD index

    //this cut reduces the number of candidates by a factor of 3, i.e., 2 out of 3 warps can end right here!
    if (innerOuterOuterMiniDoubletIndex != outerInnerInnerMiniDoubletIndex)
      return false;

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];
    unsigned int fifthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1];

    if (not runQuadrupletAlgoSelector(acc,
                                      modulesInGPU,
                                      mdsInGPU,
                                      segmentsInGPU,
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
                                      ptCut))
      return false;

    if (not runQuadrupletAlgoSelector(acc,
                                      modulesInGPU,
                                      mdsInGPU,
                                      segmentsInGPU,
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
                                      ptCut))
      return false;

    float x1 = mdsInGPU.anchorX[firstMDIndex];
    float x2 = mdsInGPU.anchorX[secondMDIndex];
    float x3 = mdsInGPU.anchorX[thirdMDIndex];
    float x4 = mdsInGPU.anchorX[fourthMDIndex];
    float x5 = mdsInGPU.anchorX[fifthMDIndex];

    float y1 = mdsInGPU.anchorY[firstMDIndex];
    float y2 = mdsInGPU.anchorY[secondMDIndex];
    float y3 = mdsInGPU.anchorY[thirdMDIndex];
    float y4 = mdsInGPU.anchorY[fourthMDIndex];
    float y5 = mdsInGPU.anchorY[fifthMDIndex];

    //construct the arrays
    float x1Vec[] = {x1, x1, x1};
    float y1Vec[] = {y1, y1, y1};
    float x2Vec[] = {x2, x2, x2};
    float y2Vec[] = {y2, y2, y2};
    float x3Vec[] = {x3, x3, x3};
    float y3Vec[] = {y3, y3, y3};

    if (modulesInGPU.subdets[lowerModuleIndex1] == lst::Endcap and
        modulesInGPU.moduleType[lowerModuleIndex1] == lst::TwoS) {
      x1Vec[1] = mdsInGPU.anchorLowEdgeX[firstMDIndex];
      x1Vec[2] = mdsInGPU.anchorHighEdgeX[firstMDIndex];

      y1Vec[1] = mdsInGPU.anchorLowEdgeY[firstMDIndex];
      y1Vec[2] = mdsInGPU.anchorHighEdgeY[firstMDIndex];
    }
    if (modulesInGPU.subdets[lowerModuleIndex2] == lst::Endcap and
        modulesInGPU.moduleType[lowerModuleIndex2] == lst::TwoS) {
      x2Vec[1] = mdsInGPU.anchorLowEdgeX[secondMDIndex];
      x2Vec[2] = mdsInGPU.anchorHighEdgeX[secondMDIndex];

      y2Vec[1] = mdsInGPU.anchorLowEdgeY[secondMDIndex];
      y2Vec[2] = mdsInGPU.anchorHighEdgeY[secondMDIndex];
    }
    if (modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap and
        modulesInGPU.moduleType[lowerModuleIndex3] == lst::TwoS) {
      x3Vec[1] = mdsInGPU.anchorLowEdgeX[thirdMDIndex];
      x3Vec[2] = mdsInGPU.anchorHighEdgeX[thirdMDIndex];

      y3Vec[1] = mdsInGPU.anchorLowEdgeY[thirdMDIndex];
      y3Vec[2] = mdsInGPU.anchorHighEdgeY[thirdMDIndex];
    }

    float innerRadiusMin2S, innerRadiusMax2S;
    computeErrorInRadiusT4(acc, x1Vec, y1Vec, x2Vec, y2Vec, x3Vec, y3Vec, innerRadiusMin2S, innerRadiusMax2S);

    for (int i = 0; i < 3; i++) {
      x1Vec[i] = x4;
      y1Vec[i] = y4;
    }
    if (modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap and
        modulesInGPU.moduleType[lowerModuleIndex4] == lst::TwoS) {
      x1Vec[1] = mdsInGPU.anchorLowEdgeX[fourthMDIndex];
      x1Vec[2] = mdsInGPU.anchorHighEdgeX[fourthMDIndex];

      y1Vec[1] = mdsInGPU.anchorLowEdgeY[fourthMDIndex];
      y1Vec[2] = mdsInGPU.anchorHighEdgeY[fourthMDIndex];
    }

    for (int i = 0; i < 3; i++) {
      x2Vec[i] = x5;
      y2Vec[i] = y5;
    }
    if (modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap and
        modulesInGPU.moduleType[lowerModuleIndex5] == lst::TwoS) {
      x2Vec[1] = mdsInGPU.anchorLowEdgeX[fifthMDIndex];
      x2Vec[2] = mdsInGPU.anchorHighEdgeX[fifthMDIndex];

      y2Vec[1] = mdsInGPU.anchorLowEdgeY[fifthMDIndex];
      y2Vec[2] = mdsInGPU.anchorHighEdgeY[fifthMDIndex];
    }

    float outerRadiusMin2S, outerRadiusMax2S;
    computeErrorInRadiusT4(acc, x3Vec, y3Vec, x1Vec, y1Vec, x2Vec, y2Vec, outerRadiusMin2S, outerRadiusMax2S);

    float g, f;
    outerRadius = tripletsInGPU.circleRadius[outerTripletIndex];
    innerRadius = tripletsInGPU.circleRadius[innerTripletIndex];
    g = tripletsInGPU.circleCenterX[innerTripletIndex];
    f = tripletsInGPU.circleCenterY[innerTripletIndex];

#ifdef USE_RZCHI2
    float inner_pt = 2 * k2Rinv1GeVf * innerRadius;

    if (not passT4RZConstraint(acc,
                               modulesInGPU,
                               mdsInGPU,
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
                               inner_pt,
                               innerRadius,
                               g,
                               f))
      return false;
#else
    rzChiSquared = -1;
#endif
    if (innerRadius < 0.95f * ptCut / (2.f * k2Rinv1GeVf))
      return false;

    //split by category
    bool matchedRadii;
    if (modulesInGPU.subdets[lowerModuleIndex1] == lst::Barrel and
        modulesInGPU.subdets[lowerModuleIndex2] == lst::Barrel and
        modulesInGPU.subdets[lowerModuleIndex3] == lst::Barrel and
        modulesInGPU.subdets[lowerModuleIndex4] == lst::Barrel and
        modulesInGPU.subdets[lowerModuleIndex5] == lst::Barrel) {
      matchedRadii = T4matchRadiiBBBBB(acc, innerRadius, outerRadius);
    } else if (modulesInGPU.subdets[lowerModuleIndex1] == lst::Barrel and
               modulesInGPU.subdets[lowerModuleIndex2] == lst::Barrel and
               modulesInGPU.subdets[lowerModuleIndex3] == lst::Barrel and
               modulesInGPU.subdets[lowerModuleIndex4] == lst::Barrel and
               modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap) {
      matchedRadii = T4matchRadiiBBBBE(acc, innerRadius, outerRadius);
    } else if (modulesInGPU.subdets[lowerModuleIndex1] == lst::Barrel and
               modulesInGPU.subdets[lowerModuleIndex2] == lst::Barrel and
               modulesInGPU.subdets[lowerModuleIndex3] == lst::Barrel and
               modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap and
               modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap) {
      if (modulesInGPU.layers[lowerModuleIndex1] == 1) {
        matchedRadii =
            T4matchRadiiBBBEE12378(acc, innerRadius, outerRadius, outerRadiusMin2S, outerRadiusMax2S);
      } else if (modulesInGPU.layers[lowerModuleIndex1] == 2) {
        matchedRadii =
            T4matchRadiiBBBEE23478(acc, innerRadius, outerRadius, outerRadiusMin2S, outerRadiusMax2S);
      } else {
        matchedRadii =
            T4matchRadiiBBBEE34578(acc, innerRadius, outerRadius, outerRadiusMin2S, outerRadiusMax2S);
      }
    }

    else if (modulesInGPU.subdets[lowerModuleIndex1] == lst::Barrel and
             modulesInGPU.subdets[lowerModuleIndex2] == lst::Barrel and
             modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap and
             modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap and
             modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap) {
      matchedRadii = T4matchRadiiBBEEE(acc, innerRadius,  outerRadius, outerRadiusMin2S, outerRadiusMax2S);
    } else if (modulesInGPU.subdets[lowerModuleIndex1] == lst::Barrel and
               modulesInGPU.subdets[lowerModuleIndex2] == lst::Endcap and
               modulesInGPU.subdets[lowerModuleIndex3] == lst::Endcap and
               modulesInGPU.subdets[lowerModuleIndex4] == lst::Endcap and
               modulesInGPU.subdets[lowerModuleIndex5] == lst::Endcap) {
      matchedRadii = T4matchRadiiBEEEE(acc,
                                     innerRadius,
                                     outerRadius,
                                     innerRadiusMin2S,
                                     innerRadiusMax2S,
                                     outerRadiusMin2S,
                                     outerRadiusMax2S);
    } else {
      matchedRadii = T4matchRadiiEEEEE(acc,
                                     innerRadius,
                                     outerRadius,
                                     innerRadiusMin2S,
                                     innerRadiusMax2S,
                                     outerRadiusMin2S,
                                     outerRadiusMax2S);
    }

    //compute regression radius right here - this computation is expensive!!!
    if (not matchedRadii)
      return false;

    float xVec[] = {x1, x2, x3, x4, x5};
    float yVec[] = {y1, y2, y3, y4, y5};
    const uint16_t lowerModuleIndices[] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4, lowerModuleIndex5};

    // 5 categories for sigmas
    float sigmas2[5], delta1[5], delta2[5], slopes[5];
    bool isFlat[5];

    computeSigmasForRegressionT4(acc, modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);


#ifdef USE_T5_DNN
    unsigned int mdIndices[] = {firstMDIndex, secondMDIndex, thirdMDIndex, fourthMDIndex, fifthMDIndex};
    float inference = lst::t5dnn::runInference(acc,
                                               modulesInGPU,
                                               mdsInGPU,
                                               segmentsInGPU,
                                               tripletsInGPU,
                                               xVec,
                                               yVec,
                                               mdIndices,
                                               lowerModuleIndices,
                                               innerTripletIndex,
                                               outerTripletIndex,
                                               innerRadius,
                                               outerRadius);
    if (inference <= lst::t5dnn::kLSTWp2)                               // T5-building cut
      return false;
#endif

    //compute the other chisquared
    //non anchor is always shifted for tilted and endcap!
    float nonAnchorDelta1[Params_T4::kLayers], nonAnchorDelta2[Params_T4::kLayers], nonAnchorSlopes[Params_T4::kLayers];
    float nonAnchorxs[] = {mdsInGPU.outerX[firstMDIndex],
                           mdsInGPU.outerX[secondMDIndex],
                           mdsInGPU.outerX[thirdMDIndex],
                           mdsInGPU.outerX[fourthMDIndex],
                           mdsInGPU.outerX[fifthMDIndex]};
    float nonAnchorys[] = {mdsInGPU.outerY[firstMDIndex],
                           mdsInGPU.outerY[secondMDIndex],
                           mdsInGPU.outerY[thirdMDIndex],
                           mdsInGPU.outerY[fourthMDIndex],
                           mdsInGPU.outerY[fifthMDIndex]};

    computeSigmasForRegressionT4(acc,
                               modulesInGPU,
                               lowerModuleIndices,
                               nonAnchorDelta1,
                               nonAnchorDelta2,
                               nonAnchorSlopes,
                               isFlat,
                               Params_T4::kLayers,
                               false);
    return true;
  };

  struct createQuadrupletsInGPUv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::MiniDoublets mdsInGPU,
                                  lst::Segments segmentsInGPU,
                                  lst::Triplets tripletsInGPU,
                                  lst::Quadruplets quadrupletsInGPU,
                                  lst::ObjectRanges rangesInGPU,
                                  uint16_t nEligibleT4Modules, 
                                  const float ptCut) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

    //   for (int iter = globalThreadIdx[0]; iter < nEligibleT5Modules; iter += gridThreadExtent[0]) {
    // all modules (non-zero) are eligible now since not doing duplicate removal yet
    for (int iter = globalThreadIdx[0]; iter < nEligibleT4Modules; iter += gridThreadExtent[0]) {
        uint16_t lowerModule1 = rangesInGPU.indicesOfEligibleT4Modules[iter];
        // short layer2_adjustment;
        int layer = modulesInGPU.layers[lowerModule1];
        // if (layer == 1) {
        //   layer2_adjustment = 1;
        // }  // get upper segment to be in second layer
        // else if (layer == 2) {
        //   layer2_adjustment = 0;
        // }  // get lower segment to be in second layer
        // else {
        //   continue;
        // }
        unsigned int nInnerTriplets = tripletsInGPU.nTriplets[lowerModule1];
        for (unsigned int innerTripletArrayIndex = globalThreadIdx[1]; innerTripletArrayIndex < nInnerTriplets;
             innerTripletArrayIndex += gridThreadExtent[1]) {
          unsigned int innerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule1] + innerTripletArrayIndex;
          if (tripletsInGPU.partOfPT5[innerTripletIndex])
              continue;  //don't create T4s for T3s accounted in pT5s
          if (tripletsInGPU.partOfPT3[innerTripletIndex])
              continue;  //don't create T4s for T3s accounted in pT3s
          if (tripletsInGPU.partOfT5[innerTripletIndex])
              continue;  //don't create T4s for T3s accounted in T5s
          uint16_t lowerModule2 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * innerTripletIndex + 1];
          unsigned int nOuterTriplets = tripletsInGPU.nTriplets[lowerModule2];
          for (unsigned int outerTripletArrayIndex = globalThreadIdx[2]; outerTripletArrayIndex < nOuterTriplets;
               outerTripletArrayIndex += gridThreadExtent[2]) {
            unsigned int outerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule2] + outerTripletArrayIndex;
            if (tripletsInGPU.partOfPT5[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in pT5s
            if (tripletsInGPU.partOfPT3[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in pT3s
            if (tripletsInGPU.partOfT5[outerTripletIndex])
              continue;  //don't create T4s for T3s accounted in T5s
            uint16_t lowerModule3 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * outerTripletIndex + 1];
            uint16_t lowerModule4 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * outerTripletIndex + 2];
            float innerRadius = tripletsInGPU.circleRadius[innerTripletIndex];
            float outerRadius = tripletsInGPU.circleRadius[outerTripletIndex];
            // float innerRadius, outerRadius, rzChiSquared;  //required for making distributions

            // bool success = runQuadrupletDefaultAlgo(acc,
            //                                         modulesInGPU,
            //                                         mdsInGPU,
            //                                         segmentsInGPU,
            //                                         tripletsInGPU,
            //                                         lowerModule1,
            //                                         lowerModule2,
            //                                         lowerModule3,
            //                                         lowerModule4,
            //                                         lowerModule5,
            //                                         innerTripletIndex,
            //                                         outerTripletIndex,
            //                                         innerRadius,
            //                                         outerRadius,
            //                                         rzChiSquared,
            //                                         ptCut);
            bool success = true;
            // int counter = 0;

            if (success) {
              int totOccupancyQuadruplets =
                  alpaka::atomicOp<alpaka::AtomicAdd>(acc, &quadrupletsInGPU.totOccupancyQuadruplets[lowerModule1], 1u);
              // alpaka::atomicOp<alpaka::AtomicAdd>(acc, counter, 1u);
              if (totOccupancyQuadruplets >= rangesInGPU.quadrupletModuleOccupancy[lowerModule1]) {
#ifdef WARNINGS
                printf("Quadruplet excess alert! Module index = %d, Occupancy = %d\n",
                       lowerModule1,
                       totOccupancyQuadruplets);
#endif
              } else {
                int quadrupletModuleIndex =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, &quadrupletsInGPU.nQuadruplets[lowerModule1], 1u);
                //this if statement should never get executed!
                if (rangesInGPU.quadrupletModuleIndices[lowerModule1] == -1) {
#ifdef WARNINGS
                  printf("Quadruplets : no memory for module at module index = %d\n", lowerModule1);
#endif
                } else {
                  unsigned int quadrupletIndex =
                      rangesInGPU.quadrupletModuleIndices[lowerModule1] + quadrupletModuleIndex;
                  // float phi =
                  //     mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex +
                  //                                                                                 layer2_adjustment]]];
                  // float eta =
                  //     mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex +
                  //                                                                                 layer2_adjustment]]];
                  //test phi and eta without layer adjustment
                  float phi =
                      mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex]]];
                  float eta =
                      mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex]]];
                  float pt = (innerRadius + outerRadius) * lst::k2Rinv1GeVf;
                  addQuadrupletToMemory(tripletsInGPU,
                                        quadrupletsInGPU,
                                        innerTripletIndex,
                                        outerTripletIndex,
                                        lowerModule1,
                                        lowerModule2,
                                        lowerModule3,
                                        lowerModule4,
                                        innerRadius,
                                        outerRadius,
                                        // rzChiSquared,
                                        pt,
                                        eta,
                                        phi,
                                        // scores,
                                        layer,
                                        quadrupletIndex);

                  // tripletsInGPU.partOfT4[quadrupletsInGPU.tripletIndices[2 * quadrupletIndex]] = true;
                  // tripletsInGPU.partOfT4[quadrupletsInGPU.tripletIndices[2 * quadrupletIndex + 1]] = true;
                }
              }
            }
          }
        }
      }
    }
  };

  struct createEligibleModulesListForQuadrupletsGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::Triplets tripletsInGPU,
                                  lst::ObjectRanges rangesInGPU,
                                  const float ptCut) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

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
          {336, 414, 231, 146},  // category 0
          {0, 0, 0, 0},          // category 1
          {0, 0, 0, 0},          // category 2
          {0, 0, 191, 106}       // category 3
      };

      // Occupancy matrix for 0.6 GeV pT Cut, 99.99%
      constexpr int p06_occupancy_matrix[4][4] = {
          {325, 237, 217, 176},  // category 0
          {0, 0, 0, 0},          // category 1
          {0, 0, 0, 0},          // category 2
          {0, 0, 129, 180}       // category 3
      };

      // Select the appropriate occupancy matrix based on ptCut
      const auto& occupancy_matrix = (ptCut < 0.8f) ? p06_occupancy_matrix : p08_occupancy_matrix;

      for (int i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
        // Condition for a quadruple to exist for a module
        // T4s don't exist for layers 4, 5, 6 barrel, and layers 3,4,5 endcap
        short module_rings = modulesInGPU.rings[i];
        short module_layers = modulesInGPU.layers[i];
        short module_subdets = modulesInGPU.subdets[i];
        float module_eta = alpaka::math::abs(acc, modulesInGPU.eta[i]);

        if (tripletsInGPU.nTriplets[i] == 0)
          continue;
        if (module_subdets == lst::Barrel and module_layers > 3)
          continue;
        if (module_subdets == lst::Endcap and module_layers > 2)
          continue;

        int nEligibleT4Modules = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nEligibleT4Modulesx, 1);

        int category_number = lst::getCategoryNumber(module_layers, module_subdets, module_rings);
        int eta_number = lst::getEtaBin(module_eta);

        int occupancy = 0;
        if (category_number != -1 && eta_number != -1) {
          occupancy = occupancy_matrix[category_number][eta_number];
        }
#ifdef WARNINGS
        else {
          printf("Unhandled case in createEligibleModulesListForQuadrupletsGPU! Module index = %i\n", i);
        }
#endif

        int nTotQ = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nTotalQuadrupletsx, occupancy);
        rangesInGPU.quadrupletModuleIndices[i] = nTotQ;
        rangesInGPU.indicesOfEligibleT4Modules[nEligibleT4Modules] = i;
        rangesInGPU.quadrupletModuleOccupancy[i] = occupancy;
      }

      // Wait for all threads to finish before reporting final values
      alpaka::syncBlockThreads(acc);
      if (globalThreadIdx[2] == 0) {
        *rangesInGPU.nEligibleT4Modules = static_cast<uint16_t>(nEligibleT4Modulesx);
        *rangesInGPU.device_nTotalQuads = static_cast<unsigned int>(nTotalQuadrupletsx);
      }
    }
  };

  struct addQuadrupletRangesToEventExplicit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::Quadruplets quadrupletsInGPU,
                                  lst::ObjectRanges rangesInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (uint16_t i = globalThreadIdx[2]; i < *modulesInGPU.nLowerModules; i += gridThreadExtent[2]) {
        if (quadrupletsInGPU.nQuadruplets[i] == 0 or rangesInGPU.quadrupletModuleIndices[i] == -1) {
          rangesInGPU.quadrupletRanges[i * 2] = -1;
          rangesInGPU.quadrupletRanges[i * 2 + 1] = -1;
        } else {
          rangesInGPU.quadrupletRanges[i * 2] = rangesInGPU.quadrupletModuleIndices[i];
          rangesInGPU.quadrupletRanges[i * 2 + 1] =
              rangesInGPU.quadrupletModuleIndices[i] + quadrupletsInGPU.nQuadruplets[i] - 1;
        }
      }
    }
  };
}  // namespace lst
#endif
