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
    FPX* score_rphisum;
    uint8_t* layer;
    char* isDup;

    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    float* rzChiSquared;

    float* dBeta;
    float* promptscore_t4dnn;
    float* displacedscore_t4dnn;
    float* fakescore_t4dnn;

    float* regressionRadius;
    float* nonAnchorRegressionRadius;
    float* regressionG;
    float* regressionF;

    bool* partOfPT4;
    bool* TightPromptFlag;
    bool* TightDisplacedFlag;
    bool* TightCutFlag;
    bool* partOfTC;

    float* uncertainty;

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
      score_rphisum = alpaka::getPtrNative(buf.score_rphisum_buf);
      layer = alpaka::getPtrNative(buf.layer_buf);
      isDup = alpaka::getPtrNative(buf.isDup_buf);
      logicalLayers = alpaka::getPtrNative(buf.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(buf.hitIndices_buf);
      rzChiSquared = alpaka::getPtrNative(buf.rzChiSquared_buf);
      dBeta = alpaka::getPtrNative(buf.dBeta_buf);
      promptscore_t4dnn = alpaka::getPtrNative(buf.promptscore_t4dnn_buf);
      displacedscore_t4dnn = alpaka::getPtrNative(buf.displacedscore_t4dnn_buf); 
      fakescore_t4dnn = alpaka::getPtrNative(buf.fakescore_t4dnn_buf);   
      regressionRadius = alpaka::getPtrNative(buf.regressionRadius_buf);
      nonAnchorRegressionRadius = alpaka::getPtrNative(buf.nonAnchorRegressionRadius_buf);
      regressionG = alpaka::getPtrNative(buf.regressionG_buf);
      regressionF = alpaka::getPtrNative(buf.regressionF_buf);
      partOfPT4 = alpaka::getPtrNative(buf.partOfPT4_buf);
      TightPromptFlag = alpaka::getPtrNative(buf.TightPromptFlag_buf);
      TightDisplacedFlag = alpaka::getPtrNative(buf.TightDisplacedFlag_buf);
      TightCutFlag = alpaka::getPtrNative(buf.TightCutFlag_buf);
      partOfTC = alpaka::getPtrNative(buf.partOfTC_buf);
      uncertainty = alpaka::getPtrNative(buf.uncertainty_buf);
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
    Buf<TDev, FPX> score_rphisum_buf;
    Buf<TDev, uint8_t> layer_buf;
    Buf<TDev, char> isDup_buf;

    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, float> rzChiSquared_buf;
    Buf<TDev, float> dBeta_buf;
    Buf<TDev, float> promptscore_t4dnn_buf;
    Buf<TDev, float> displacedscore_t4dnn_buf;
    Buf<TDev, float> fakescore_t4dnn_buf;

    Buf<TDev, float> regressionRadius_buf;
    Buf<TDev, float> nonAnchorRegressionRadius_buf;
    Buf<TDev, float> regressionG_buf;
    Buf<TDev, float> regressionF_buf;

    Buf<TDev, bool> partOfPT4_buf;
    Buf<TDev, bool> TightPromptFlag_buf;
    Buf<TDev, bool> TightDisplacedFlag_buf;
    Buf<TDev, bool> TightCutFlag_buf;
    Buf<TDev, bool> partOfTC_buf;
    Buf<TDev, float> uncertainty_buf;

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
          score_rphisum_buf(allocBufWrapper<FPX>(devAccIn, nTotalQuadruplets, queue)),
          layer_buf(allocBufWrapper<uint8_t>(devAccIn, nTotalQuadruplets, queue)),
          isDup_buf(allocBufWrapper<char>(devAccIn, nTotalQuadruplets, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, Params_T4::kLayers * nTotalQuadruplets, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, Params_T4::kHits * nTotalQuadruplets, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          dBeta_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          promptscore_t4dnn_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          displacedscore_t4dnn_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          fakescore_t4dnn_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          regressionRadius_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          nonAnchorRegressionRadius_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          regressionG_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          regressionF_buf(allocBufWrapper<float>(devAccIn, nTotalQuadruplets, queue)),
          partOfPT4_buf(allocBufWrapper<bool>(devAccIn, nTotalQuadruplets, queue)),
          TightPromptFlag_buf(allocBufWrapper<bool>(devAccIn, nTotalQuadruplets, queue)),
          TightDisplacedFlag_buf(allocBufWrapper<bool>(devAccIn, nTotalQuadruplets, queue)),
          TightCutFlag_buf(allocBufWrapper<bool>(devAccIn, nTotalQuadruplets, queue)),
          partOfTC_buf(allocBufWrapper<bool>(devAccIn, nTotalQuadruplets, queue)),
          uncertainty_buf(allocBufWrapper<float>(devAccIn, Params_T4::kLayers * nTotalQuadruplets, queue)) {
      alpaka::memset(queue, nQuadruplets_buf, 0u);
      alpaka::memset(queue, totOccupancyQuadruplets_buf, 0u);
      alpaka::memset(queue, isDup_buf, 0u);
      alpaka::memset(queue, partOfPT4_buf, false);
      alpaka::memset(queue, TightPromptFlag_buf, false);
      alpaka::memset(queue, TightDisplacedFlag_buf, false);
      alpaka::memset(queue, partOfTC_buf, false);
      alpaka::memset(queue, TightCutFlag_buf, false);
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
                                                            bool TightPromptFlag,
                                                            bool TightDisplacedFlag,
                                                            bool TightCutFlag,
                                                            float* error2s) {
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
    quadrupletsInGPU.score_rphisum[quadrupletIndex] = __F2H(scores);
    quadrupletsInGPU.layer[quadrupletIndex] = layer;
    quadrupletsInGPU.isDup[quadrupletIndex] = 0;
    quadrupletsInGPU.logicalLayers[Params_T4::kLayers * quadrupletIndex] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex];
    quadrupletsInGPU.logicalLayers[Params_T4::kLayers * quadrupletIndex + 1] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex + 1];
    quadrupletsInGPU.logicalLayers[Params_T4::kLayers * quadrupletIndex + 2] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * innerTripletIndex + 2];
    quadrupletsInGPU.logicalLayers[Params_T4::kLayers * quadrupletIndex + 3] =
        tripletsInGPU.logicalLayers[Params_T3::kLayers * outerTripletIndex + 2];

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
        tripletsInGPU.hitIndices[Params_T3::kHits * outerTripletIndex + 4];
    quadrupletsInGPU.hitIndices[Params_T4::kHits * quadrupletIndex + 7] =
        tripletsInGPU.hitIndices[Params_T3::kHits * outerTripletIndex + 5];
    
    quadrupletsInGPU.rzChiSquared[quadrupletIndex] = rzChiSquared;
    quadrupletsInGPU.dBeta[quadrupletIndex] = dBeta;
    quadrupletsInGPU.promptscore_t4dnn[quadrupletIndex] = promptScore;
    quadrupletsInGPU.displacedscore_t4dnn[quadrupletIndex] = displacedScore;
    quadrupletsInGPU.fakescore_t4dnn[quadrupletIndex] = fakeScore;

    quadrupletsInGPU.regressionRadius[quadrupletIndex] = regressionRadius;
    quadrupletsInGPU.nonAnchorRegressionRadius[quadrupletIndex] = nonAnchorRegressionRadius;
    quadrupletsInGPU.regressionG[quadrupletIndex] = regressionG;
    quadrupletsInGPU.regressionF[quadrupletIndex] = regressionF;

    quadrupletsInGPU.TightPromptFlag[quadrupletIndex] = TightPromptFlag;
    quadrupletsInGPU.TightDisplacedFlag[quadrupletIndex] = TightDisplacedFlag;
    quadrupletsInGPU.TightCutFlag[quadrupletIndex] = TightCutFlag;

    quadrupletsInGPU.uncertainty[Params_T4::kLayers * quadrupletIndex] = error2s[0];
    quadrupletsInGPU.uncertainty[Params_T4::kLayers * quadrupletIndex + 1] = error2s[1];
    quadrupletsInGPU.uncertainty[Params_T4::kLayers * quadrupletIndex + 2] = error2s[2];
    quadrupletsInGPU.uncertainty[Params_T4::kLayers * quadrupletIndex + 3] = error2s[3];

  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passT4RZConstraint(TAcc const& acc,
                                                         lst::Modules const& modulesInGPU,
                                                         lst::MiniDoublets const& mdsInGPU,
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
                                                         bool& TightCutFlag) {
    //(g,f) is the center of the circle fitted by the innermost 3 points on x,y coordinates
    const float& rt1 = mdsInGPU.anchorRt[firstMDIndex] / 100;  //in the unit of m instead of cm
    const float& rt2 = mdsInGPU.anchorRt[secondMDIndex] / 100;
    const float& rt3 = mdsInGPU.anchorRt[thirdMDIndex] / 100;
    const float& rt4 = mdsInGPU.anchorRt[fourthMDIndex] / 100;

    const float& z1 = mdsInGPU.anchorZ[firstMDIndex] / 100;
    const float& z2 = mdsInGPU.anchorZ[secondMDIndex] / 100;
    const float& z3 = mdsInGPU.anchorZ[thirdMDIndex] / 100;
    const float& z4 = mdsInGPU.anchorZ[fourthMDIndex] / 100;

    // Using lst_layer numbering convention defined in ModuleMethods.h
    const int layer1 = modulesInGPU.lstLayers[lowerModuleIndex1];
    const int layer2 = modulesInGPU.lstLayers[lowerModuleIndex2];
    const int layer3 = modulesInGPU.lstLayers[lowerModuleIndex3];
    const int layer4 = modulesInGPU.lstLayers[lowerModuleIndex4];

    //slope computed using the internal T3s
    const int moduleType1 = modulesInGPU.moduleType[lowerModuleIndex1];  //0 is ps, 1 is 2s
    const int moduleType2 = modulesInGPU.moduleType[lowerModuleIndex2];
    const int moduleType3 = modulesInGPU.moduleType[lowerModuleIndex3];
    const int moduleType4 = modulesInGPU.moduleType[lowerModuleIndex4];

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
    float rt_init = mdsInGPU.anchorRt[thirdMDIndex] / 100;  //use the third MD as initial point

    if (moduleType3 == 1)  // 1: if MD3 is in 2s layer
    {
      x_init = mdsInGPU.anchorX[secondMDIndex] / 100;
      y_init = mdsInGPU.anchorY[secondMDIndex] / 100;
      z_init = mdsInGPU.anchorZ[secondMDIndex] / 100;
      rt_init = mdsInGPU.anchorRt[secondMDIndex] / 100;
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
    // if the 4 points are linear, helix calculation gives nan
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

      // creating a chi squared type quantity
      // 0-> PS, 1->2S
      residual4_linear = (moduleType4 == 0) ? residual4_linear / kPixelPSZpitch : residual4_linear / kStrip2SZpitch;
      residual4_linear = residual4_linear * 100;

      rzChiSquared = 12 * (residual4_linear * residual4_linear);
      // return rzChiSquared < 10.683f; //full T3s 99%
      // return rzChiSquared < 12.293f; //full T3s 99%, pT>=10GeV
      // if (rzChiSquared < 2.595f) //95% displaced retention, add uncert
      // if (rzChiSquared < 4.24f) //95% displaced retention, add radii
      if (rzChiSquared < 9.666f) //95% retention, add radii and t3 scores
        TightCutFlag = true;
      // return rzChiSquared < 3.693f; //full T3s, 99%, add uncert to DNN
      // return rzChiSquared < 9.666f; //99% add reg radii to dnn
      return rzChiSquared < 14.064f; //99% add reg radii and t3 scores to dnn
      // return true;
    }
    // return true;
    // The category numbers are related to module regions and layers, decoding of the region numbers can be found here in slide 2 table. https://github.com/SegmentLinking/TrackLooper/files/11420927/part.2.pdf
    // The commented numbers after each case is the region code, and can look it up from the table to see which category it belongs to. For example, //0 means T5 built with Endcap 1,2,3,4,5 ps modules
    if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 10)  //0
    {
      // if (rzChiSquared < 19.246f) //95% displaced retention
      // if (rzChiSquared < 19.360f) //95% displaced retention, add uncert
      // if (rzChiSquared < 19.239f) //95% displaced retention, add radii
      if (rzChiSquared < 19.283f) //95% displaced retention, add radii and t3 scores
        TightCutFlag = true;
      // return rzChiSquared < 28.056f; //full t3s, multi dnn 99%
      // return rzChiSquared < 28.772f; //full t3s, multi dnn add uncert to dnn 99%
      // return rzChiSquared < 28.574f; //99% add reg radii to dnn
      return rzChiSquared < 28.459f; //99% add reg radii and t3 scores to dnn
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 9 and layer4 == 15)  //1
    {
      // if (rzChiSquared < 6.274f) //95% displaced retention
      // if (rzChiSquared < 6.295f) //95% displaced retention, add uncert
      // if (rzChiSquared < 6.295f) //95% displaced retention, add radii
      if (rzChiSquared < 6.298f) //95% displaced retention, add radii and t3 scores
        TightCutFlag = true;
      // return rzChiSquared < 8.968f; //full t3s, multi dnn 99%
      // return rzChiSquared < 8.971f; //full t3s, multi dnn add uncert to dnn 99%
      // return rzChiSquared < 8.999f; //99% add reg radii to dnn
      return rzChiSquared < 8.968f; //99% add reg radii and t3 scores to dnn
    } else if (layer1 == 7 and layer2 == 8 and layer3 == 14 and layer4 == 15)  //2
    {
      // if (rzChiSquared < 3.824f) //95% displaced retention
      // if (rzChiSquared < 3.886f) //95% displaced retention, add uncert
      // if (rzChiSquared < 3.871f) //95% displaced retention, add radii
      if (rzChiSquared < 3.879f) //95% displaced retention, add radii and t3 scores
        TightCutFlag = true;
      // return rzChiSquared < 5.032f; //full t3s, multi dnn 99%
      // return rzChiSquared < 5.160f; //full t3s, multi dnn add uncert to dnn 99%
      // return rzChiSquared < 5.091f; //99% add reg radii to dnn
      return rzChiSquared < 5.158f; //99% add reg radii and t3 scores to dnn
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 10) {
      if (layer4 == 11)  //3
      {
        // if (rzChiSquared < 18.572f) //95% displaced retention
        // if (rzChiSquared < 18.714f) //95% displaced retention, add uncert
        // if (rzChiSquared < 18.414f) //95% displaced retention, add radii
        if (rzChiSquared < 18.516f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 29.292f; //full t3s, multi dnn 99%
        // return rzChiSquared < 29.924f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 29.755f; //99% add reg radii to dnn
        return rzChiSquared < 29.270f; //99% add reg radii and t3 scores to dnn
      }
      if (layer4 == 16)  //4
      {
        // if (rzChiSquared < 6.362f) //95% displaced retention
        // if (rzChiSquared < 6.467f) //95% displaced retention, add uncert
        // if (rzChiSquared < 6.418f) //95% displaced retention, add radii
        if (rzChiSquared < 6.378f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 9.359f; //full t3s, multi dnn 99%
        // return rzChiSquared < 9.357f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 9.359f; //99% add reg radii to dnn
        return rzChiSquared < 9.310f; //99% add reg radii and t3 scores to dnn
      }
    } else if (layer1 == 8 and layer2 == 9 and layer3 == 15 and layer4 == 16) //5
    { 
        // if (rzChiSquared < 3.379f) //95% displaced retention
        // if (rzChiSquared < 3.412f) //95% displaced retention, add uncert
        // if (rzChiSquared < 3.474f) //95% displaced retention, add radii
        if (rzChiSquared < 3.493f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 4.190f; //full t3s, multi dnn 99%
        // return rzChiSquared < 4.347f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 4.382f; //99% add reg radii to dnn
        return rzChiSquared < 4.328f; //99% add reg radii and t3 scores to dnn
    }
    else if (layer1 == 1 and layer2 == 2 and layer3 == 3) {
      if (layer4 == 4)  //6
      {
        // if (rzChiSquared < 13.331f) //95% displaced retention
        // if (rzChiSquared < 13.366f) //95% displaced retention, add uncert
        // if (rzChiSquared < 13.499f) //95% displaced retention, add radii
        if (rzChiSquared < 13.252f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 23.235f; //full t3s, multi dnn 99%
        // return rzChiSquared < 23.585f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 23.296f; //99% add reg radii to dnn
        return rzChiSquared < 23.138f; //99% add reg radii and t3 scores to dnn
      }
      else if (layer4 == 7)  //7
      {
        // if (rzChiSquared < 17.074f) //95% displaced retention
        // if (rzChiSquared < 17.074f) //95% displaced retention, add uncert
        // if (rzChiSquared < 17.276f) //95% displaced retention, add radii
        if (rzChiSquared < 16.956f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 29.561f; //full t3s, multi dnn 99%
        // return rzChiSquared < 28.862f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 29.243f; //99% add reg radii to dnn
        return rzChiSquared < 29.561f; //99% add reg radii and t3 scores to dnn
      } else if (layer4 == 12)  //8
      {
        // if (rzChiSquared < 10.879f) //95% displaced retention
        // if (rzChiSquared < 10.887f) //95% displaced retention, add uncert
        // if (rzChiSquared <10.887f) //95% displaced retention, add radii
        if (rzChiSquared < 10.924f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 18.687f; //full t3s, multi dnn 99%
        // return rzChiSquared < 19.136f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 18.74f; //99% add reg radii to dnn
        return rzChiSquared < 18.905f; //99% add reg radii and t3 scores to dnn
      }
    } else if (layer1 == 1 and layer2 == 2 and layer3 == 7) {
      if (layer4 == 8)  //9
      {
        // if (rzChiSquared < 18.237f) //95% displaced retention
        // if (rzChiSquared < 18.039f) //95% displaced retention, add uncert
        // if (rzChiSquared < 18.263f) //95% displaced retention, add radii
        if (rzChiSquared < 18.263f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 30.072f; //full t3s, multi dnn 99%
        // return rzChiSquared < 28.911f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 29.803f; //99% add reg radii to dnn
        return rzChiSquared < 29.534f; //99% add reg radii and t3 scores to dnn
      } else if (layer4 == 13)  //10
      {
        // if (rzChiSquared < 8.467f) //95% displaced retention
        // if (rzChiSquared < 8.418f) //95% displaced retention, add uncert
        // if (rzChiSquared < 8.384f) //95% displaced retention, add radii
        if (rzChiSquared < 8.384f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 12.88f; //full t3s, multi dnn 99%
        // return rzChiSquared < 12.797f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 12.664f; //99% add reg radii to dnn
        return rzChiSquared < 12.608f; //99% add reg radii and t3 scores to dnn
      } 
    } else if (layer1 == 1 and layer2 == 7 and layer3 == 8) {
      if (layer4 == 9) //11
      {
        // if (rzChiSquared < 18.778f) //95% displaced retention
        // if (rzChiSquared < 18.850f) //95% displaced retention, add uncert
        // if (rzChiSquared < 18.856f) //95% displaced retention, add radii
        if (rzChiSquared < 18.741f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 27.908f; //full t3s, multi dnn 99%GeV
        // return rzChiSquared < 28.349f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 28.224f; //99% add reg radii to dnn
        return rzChiSquared < 28.270f; //99% add reg radii and t3 scores to dnn
      } else if (layer4 == 14)  //12
      {
        return true; // leftover T3s, //full T3s 99%
      } 
    } else if (layer1 == 2 and layer2 ==3) {
      if (layer3 == 4) {
        if (layer4 == 5)  //13
        {
          // if (rzChiSquared < 4.426f) //95% displaced retention
          // if (rzChiSquared < 4.438f) //95% displaced retention, add uncert
          // if (rzChiSquared < 4.396f) //95% displaced retention, add radii
          if (rzChiSquared < 4.376f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return rzChiSquared < 5.419f; //full t3s, multi dnn 99%
          // return rzChiSquared < 5.423f; //full t3s, multi dnn add uncert to dnn 99%
          // return rzChiSquared < 5.292f; //99% add reg radii to dnn
          return rzChiSquared < 5.430f; //99% add reg radii and t3 scores to dnn
        }
        else if (layer4 == 12) //14
        { 
          // if (rzChiSquared <4.07f) //95% displaced retention
          // if (rzChiSquared < 4.119f) //95% displaced retention, add uncert
          // if (rzChiSquared < 4.072f) //95% displaced retention, add radii
          if (rzChiSquared < 4.196f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return rzChiSquared < 4.949f; //full t3s, multi dnn 99%
          // return rzChiSquared < 5.285f; //full t3s, multi dnn add uncert to dnn 99%
          // return rzChiSquared < 5.233f; //99% add reg radii to dnn
          return rzChiSquared < 5.176f; //99% add reg radii and t3 scores to dnn
        }
      }
      else if (layer3 == 7) {
        if (layer4 == 8) // 15
        { 
          // return true; // leftover T3s, //full T3s 99%
          // if (rzChiSquared < 11.934f) //95% displaced retention
          // if (rzChiSquared < 11.935f) //95% displaced retention, add uncert
          // if (rzChiSquared < 11.934f) //95% displaced retention, add radii
          if (rzChiSquared < 11.934f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return rzChiSquared < 20.491f; //full t3s, multi dnn 99%
          // return rzChiSquared < 20.491f; //full t3s, multi dnn add uncert to dnn 99%
          // return rzChiSquared < 20.491f; //99% add reg radii to dnn
          return rzChiSquared < 20.491f; //99% add reg radii and t3 scores to dnn
        }
        else if (layer4 == 13) //16
        { 
          // if (rzChiSquared < 9.671f) //95% displaced retention
          // if (rzChiSquared < 9.672f) //95% displaced retention, add uncert
          // if (rzChiSquared < 9.545f) //95% displaced retention, add radii
          if (rzChiSquared < 9.518f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return rzChiSquared < 14.355f; //full t3s, multi dnn 99%
          // return rzChiSquared < 14.527f; //full t3s, multi dnn add uncert to dnn 99%
          // return rzChiSquared < 14.477f; //99% add reg radii to dnn
          return rzChiSquared < 14.100f; //99% add reg radii and t3 scores to dnn
        }
      }
      else if (layer3 == 12 and layer4 == 13) //17
      { 
        // if (rzChiSquared < 4.107f) //95% displaced retention
        // if (rzChiSquared < 4.230f) //95% displaced retention, add uncert
        // if (rzChiSquared < 4.201f) //95% displaced retention, add radii
        if (rzChiSquared < 4.269f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 5.160f; //full t3s, multi dnn 99%
        // return rzChiSquared < 5.306f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 5.157f; //99% add reg radii to dnn
        return rzChiSquared < 5.499f; //99% add reg radii and t3 scores to dnn
      }
    } else if (layer1 == 2 and layer2 == 12 and layer3 == 13 and layer4 == 14) //18
    { 
      // if (rzChiSquared < 22.416f) //95% displaced retention
      // if (rzChiSquared < 22.481f) //95% displaced retention, add uncert
      // if (rzChiSquared < 22.5f) //95% displaced retention, add radii
      if (rzChiSquared < 22.481f) //95% displaced retention, add radii and t3 scores
        TightCutFlag = true;
      // return rzChiSquared < 35.778f; //full t3s, multi dnn 99%
      // return rzChiSquared < 35.844f; //full t3s, multi dnn add uncert to dnn 99%
      // return rzChiSquared < 35.778f; //99% add reg radii to dnn
      return rzChiSquared < 36.038f; //99% add reg radii and t3 scores to dnn
    } else if (layer1 == 2 and layer2 == 7)
    {
      if (layer3 == 8 and layer4 == 14) //19
      { 
        // if (rzChiSquared < 7.129f) //95% displaced retention
        // if (rzChiSquared < 7.129f) //95% displaced retention, add uncert
        // if (rzChiSquared < 7.086f) //95% displaced retention, add radii
        if (rzChiSquared < 7.06f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 10.899f; //full t3s, multi dnn 99%
        // return rzChiSquared < 11.166f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 11.011f; //99% add reg radii to dnn
        return rzChiSquared < 10.991f; //99% add reg radii and t3 scores to dnn
      }
      else if (layer3 == 13 and layer4 == 14) //20
      { 
        // if (rzChiSquared < 3.217f) //95% displaced retention
        // if (rzChiSquared < 3.311f) //95% displaced retention, add uncert
        // if (rzChiSquared < 3.37f) //95% displaced retention, add radii
        if (rzChiSquared < 3.343f) //95% displaced retention, add radii and t3 scores
          TightCutFlag = true;
        // return rzChiSquared < 3.876f; //full t3s, multi dnn 99%
        // return rzChiSquared < 3.868f; //full t3s, multi dnn add uncert to dnn 99%
        // return rzChiSquared < 4.159f; //99% add reg radii to dnn
        return rzChiSquared < 4.144f; //99% add reg radii and t3 scores to dnn
      }
    } else if (layer1 == 3)
    {
      if (layer2 == 4){
        if (layer3 == 5 and layer4 == 6 ) //21
        { 
          // if (rzChiSquared < 56.658f) //95% displaced retention
          // if (rzChiSquared < 57.321f) //95% displaced retention, add uncert
          // if (rzChiSquared < 57.047f) //95% displaced retention, add radii
          if (rzChiSquared < 56.716f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return rzChiSquared < 75.304f; //full t3s, multi dnn 99%
          // return rzChiSquared < 77.495f; //full t3s, multi dnn add uncert to dnn 99%
          // return rzChiSquared < 78.026f; //99% add reg radii to dnn
          return rzChiSquared < 76.861f; //99% add reg radii and t3 scores to dnn
        }
        else if (layer3 == 12 and layer4 == 13) //24
        {
          // if (rzChiSquared < 16.991f) //95% displaced retention
          // if (rzChiSquared < 17.851f) //95% displaced retention
          // if (rzChiSquared < 18.597f) //95% displaced retention, add radii
          if (rzChiSquared < 17.68f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return rzChiSquared < 24.917f; //full t3s, multi dnn 99%
          // return rzChiSquared < 26.821f; //full t3s, multi dnn add uncert to dnn 99%
          // return rzChiSquared < 27.695f; //99% add reg radii to dnn
          return rzChiSquared < 27.034f; //99% add reg radii and t3 scores to dnn
        }
        else if (layer3 == 5 and layer4 == 12) //25
        {
          // if (rzChiSquared < 35.097f) //95% displaced retention
          // if (rzChiSquared < 36.715f) //95% displaced retention, add uncert to dnn
          // if (rzChiSquared < 35.939f) //95% displaced retention, add radii
          if (rzChiSquared < 36.004f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return rzChiSquared < 44.968f; //full t3s, multi dnn 99%
          // return rzChiSquared < 46.854f; //full t3s, multi dnn add uncert to dnn 99%
          // return rzChiSquared < 46.246f; //99% add reg radii to dnn
          return rzChiSquared < 48.511f; //99% add reg radii and t3 scores to dnn
        }
      }
      else if (layer2 == 7) {
        if (layer3 == 8 and layer4 == 14) //22
        {
          // if (rzChiSquared < 3.971f) //95% displaced retention, add uncert (with and without)
          // if (rzChiSquared < 3.971f) //95% displaced retention, add radii
          if (rzChiSquared < 3.971f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return true; // leftover T3s, full T3s
          // return rzChiSquared < 6.822f; //full t3s, multi dnn 99%
          return rzChiSquared < 6.017f; //99% add reg radii and t3 scores to dnn
        }
        else if (layer3 == 13 and layer4 == 14) //23
        {
          // if (rzChiSquared < 3.53f) //95% displaced retention
          // if (rzChiSquared < 3.494f) //95% displaced retention, add uncert
          // if (rzChiSquared < 3.569f) //95% displaced retention, add radii
          if (rzChiSquared < 3.571f) //95% displaced retention, add radii and t3 scores
            TightCutFlag = true;
          // return rzChiSquared < 4.342f; //full t3s, multi dnn 99%
            // return rzChiSquared < 4.363f; //full t3s, multi dnn add uncert to dnn 99%
          // return rzChiSquared < 4.418f; //99% add reg radii to dnn
          return rzChiSquared < 4.415f; //99% add reg radii and t3 scores to dnn
        }
      }
    }
    return true;
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
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletdBetaCutBBBB(TAcc const& acc,
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
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float r3_InLo = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    float drt_InSeg = rt_InOut - rt_InLo;

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (r3_InLo / rt_InLo);

    float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float dPhi = lst::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

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

    //innerOuterAnchor - innerInnerAnchor
    const float rt_InSeg =
        alpaka::math::sqrt(acc,
                           (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                   (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                   (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));

    float betaAv = 0.5f * (betaIn + betaOut);
    // printf("betaAv first round:%f\n", betaAv);
    float pt_beta = drt_tl_axis * lst::k2Rinv1GeVf / alpaka::math::sin(acc, betaAv);
    int lIn = 5;
    int lOut = isEC_lastLayer ? 11 : 5;
    float sdOut_dr = alpaka::math::sqrt(acc,
                                        (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) *
                                                (mdsInGPU.anchorX[fourthMDIndex] - mdsInGPU.anchorX[thirdMDIndex]) +
                                            (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]) *
                                                (mdsInGPU.anchorY[fourthMDIndex] - mdsInGPU.anchorY[thirdMDIndex]));
    float sdOut_d = mdsInGPU.anchorRt[fourthMDIndex] - mdsInGPU.anchorRt[thirdMDIndex];

    lst::runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, rt_InSeg, sdOut_dr, drt_tl_axis, lIn);
    // printf("betaIn: %f, betaOut: %f\n", betaIn, betaOut);
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

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, drt_InSeg);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaROut2 + 
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));

    dBeta = betaIn - betaOut;
    // if (dBeta > 1){
    //   printf("dBetaCut2 in BBBB: %f, dBeta2:%f \n", dBetaCut2, dBeta*dBeta);
    // }
    // printf("dBetaCut2 in BBBB: %f, dBeta2:%f \n", dBetaCut2, dBeta*dBeta);
    return dBeta * dBeta <= dBetaCut2;
    // return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletdBetaCutBBEE(TAcc const& acc,
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
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float rIn = alpaka::math::sqrt(acc, z_InLo * z_InLo + rt_InLo * rt_InLo);
    const float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f) * (rIn / rt_InLo);

    float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    float dPhi = lst::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    float sdIn_alpha = __H2F(segmentsInGPU.dPhiChanges[innerSegmentIndex]);
    float sdIn_alpha_min = __H2F(segmentsInGPU.dPhiChangeMins[innerSegmentIndex]);
    float sdIn_alpha_max = __H2F(segmentsInGPU.dPhiChangeMaxs[innerSegmentIndex]);
    float sdOut_alpha = sdIn_alpha;

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
                                       (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) *
                                               (mdsInGPU.anchorX[secondMDIndex] - mdsInGPU.anchorX[firstMDIndex]) +
                                           (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]) *
                                               (mdsInGPU.anchorY[secondMDIndex] - mdsInGPU.anchorY[firstMDIndex]));
    float sdIn_d = rt_InOut - rt_InLo;

    float dr = alpaka::math::sqrt(acc, tl_axis_x * tl_axis_x + tl_axis_y * tl_axis_y);

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

    lst::runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 + dBetaRIn2 + dBetaROut2 + 
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    dBeta = betaIn - betaOut;

    return dBeta * dBeta <= dBetaCut2;
    // return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletdBetaCutEEEE(TAcc const& acc,
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
                                                                float& dBeta,
                                                                const float ptCut) {
    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[thirdMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[thirdMDIndex];

    float thetaMuls2 = (kMulsInGeV * kMulsInGeV) * (0.1f + 0.2f * (rt_OutLo - rt_InLo) / 50.f);
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

    lst::runDeltaBetaIterations(acc, betaIn, betaOut, betaAv, pt_beta, sdIn_dr, sdOut_dr, dr, lIn);

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

    float dBetaRes = 0.02f / alpaka::math::min(acc, sdOut_d, sdIn_d);
    float dBetaCut2 =
        (dBetaRes * dBetaRes * 2.0f + dBetaMuls2 + dBetaLum2 +
         0.25f *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)) *
             (alpaka::math::abs(acc, betaInRHmin - betaInRHmax) + alpaka::math::abs(acc, betaOutRHmin - betaOutRHmax)));
    dBeta = betaIn - betaOut;

    return dBeta * dBeta <= dBetaCut2;
    // return true;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runQuadrupletdBetaAlgoSelector(TAcc const& acc,
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
                                                                     float& dBeta,
                                                                     const float ptCut) {
    short innerInnerLowerModuleSubdet = modulesInGPU.subdets[innerInnerLowerModuleIndex];
    short innerOuterLowerModuleSubdet = modulesInGPU.subdets[innerOuterLowerModuleIndex];
    short outerInnerLowerModuleSubdet = modulesInGPU.subdets[outerInnerLowerModuleIndex];
    short outerOuterLowerModuleSubdet = modulesInGPU.subdets[outerOuterLowerModuleIndex];

    if (innerInnerLowerModuleSubdet == lst::Barrel and innerOuterLowerModuleSubdet == lst::Barrel and
        outerInnerLowerModuleSubdet == lst::Barrel and outerOuterLowerModuleSubdet == lst::Barrel) {
      return runQuadrupletdBetaCutBBBB(acc,
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
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Barrel and innerOuterLowerModuleSubdet == lst::Barrel and
               outerInnerLowerModuleSubdet == lst::Endcap and outerOuterLowerModuleSubdet == lst::Endcap) {
      return runQuadrupletdBetaCutBBEE(acc,
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
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Barrel and innerOuterLowerModuleSubdet == lst::Barrel and
               outerInnerLowerModuleSubdet == lst::Barrel and outerOuterLowerModuleSubdet == lst::Endcap) {
      return runQuadrupletdBetaCutBBBB(acc,
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
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Barrel and innerOuterLowerModuleSubdet == lst::Endcap and
               outerInnerLowerModuleSubdet == lst::Endcap and outerOuterLowerModuleSubdet == lst::Endcap) {
      return runQuadrupletdBetaCutBBEE(acc,
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
                                       dBeta,
                                       ptCut);
    } else if (innerInnerLowerModuleSubdet == lst::Endcap and innerOuterLowerModuleSubdet == lst::Endcap and
               outerInnerLowerModuleSubdet == lst::Endcap and outerOuterLowerModuleSubdet == lst::Endcap) {
      return runQuadrupletdBetaCutEEEE(acc,
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
                                       dBeta,
                                       ptCut);
    }

    return false;
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
    bool isPS_OutOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[fourthMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[fourthMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    // float rtRatio_OutLoInLo = rt_OutLo / rt_InLo;  // Outer segment beginning rt divided by inner segment beginning rt;
    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
    float zpitch_OutLo = (isPS_OutOut ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);

    // float zHi = z_InLo + (z_InLo + lst::kDeltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo < 0.f ? 1.f : dzDrtScale) +
    //             (zpitch_InLo + zpitch_OutLo);
    // float zLo = z_InLo + (z_InLo - lst::kDeltaZLum) * (rtRatio_OutLoInLo - 1.f) * (z_InLo > 0.f ? 1.f : dzDrtScale) -
    //             (zpitch_InLo + zpitch_OutLo);

    // //Cut #1: z compatibility
    // if ((z_OutLo < zLo) || (z_OutLo > zHi))
    //   return false; 

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

    // Cut #2: Pointed Z (Inner segment two MD points to outer segment outer MD)
    // printf("z_outLo %f zLoPointed %f zHiPointed %f\n", z_OutLo, zLoPointed, zHiPointed);
    if ((z_OutLo < zLoPointed) || (z_OutLo > zHiPointed))
      return false;

    float pvOffset = 0.1f / rt_OutLo;
    float dPhiCut = alpha1GeV_OutLo + alpaka::math::sqrt(acc, muls2 + pvOffset * pvOffset);

    float deltaPhiPos = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[firstMDIndex]);
    // float dphi = mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[firstMDIndex];
    // printf("phi4-phi1 %f deltaPhiPos %f dPhiCut %f\n", dphi, deltaPhiPos, dPhiCut);
    // Cut #3: FIXME:deltaPhiPos can be tighter
    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    return true;

    // float midPointX = 0.5f * (mdsInGPU.anchorX[firstMDIndex] + mdsInGPU.anchorX[thirdMDIndex]);
    // float midPointY = 0.5f * (mdsInGPU.anchorY[firstMDIndex] + mdsInGPU.anchorY[thirdMDIndex]);
    // float diffX = mdsInGPU.anchorX[thirdMDIndex] - mdsInGPU.anchorX[firstMDIndex];
    // float diffY = mdsInGPU.anchorY[thirdMDIndex] - mdsInGPU.anchorY[firstMDIndex];

    // float dPhi = lst::deltaPhi(acc, midPointX, midPointY, diffX, diffY);

    // // Cut #4: deltaPhiChange
    // if (alpaka::math::abs(acc, dPhi) > dPhiCut)
    //   return false;

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
    bool isPS_OutOut = (modulesInGPU.moduleType[outerOuterLowerModuleIndex] == lst::PS);

    float rt_InLo = mdsInGPU.anchorRt[firstMDIndex];
    float rt_InOut = mdsInGPU.anchorRt[secondMDIndex];
    float rt_OutLo = mdsInGPU.anchorRt[fourthMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[fourthMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly
    float zpitch_InLo = (isPS_InLo ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
    float zpitch_OutLo = (isPS_OutOut ? lst::kPixelPSZpitch : lst::kStrip2SZpitch);
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

    // //Cut #1: rt condition
    // if (rtOut < rtLo)
    //   return false;

    float zInForHi = z_InLo - zGeom1 - dLum;
    if (zInForHi * z_InLo < 0) {
      zInForHi = alpaka::math::copysign(acc, 0.1f, z_InLo);
    }
    float rtHi = rt_InLo * (1.f + (z_OutLo - z_InLo + zGeom1) / zInForHi) + rtGeom1;

    // //Cut #2: rt condition
    // if ((rt_OutLo < rtLo) || (rt_OutLo > rtHi))
    //   return false;

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

    float deltaPhiPos = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[firstMDIndex]);

    //Cut #4: deltaPhiPos can be tighter
    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    return true;

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
    float rt_OutLo = mdsInGPU.anchorRt[fourthMDIndex];

    float z_InLo = mdsInGPU.anchorZ[firstMDIndex];
    float z_InOut = mdsInGPU.anchorZ[secondMDIndex];
    float z_OutLo = mdsInGPU.anchorZ[fourthMDIndex];

    float alpha1GeV_OutLo =
        alpaka::math::asin(acc, alpaka::math::min(acc, rt_OutLo * lst::k2Rinv1GeVf / ptCut, lst::kSinAlphaMax));

    float dzDrtScale =
        alpaka::math::tan(acc, alpha1GeV_OutLo) / alpha1GeV_OutLo;  // The track can bend in r-z plane slightly

    // Cut #0: Preliminary (Only here in endcap case)
    if ((z_InLo * z_OutLo) <= 0)
      return false;

    // float dLum = alpaka::math::copysign(acc, lst::kDeltaZLum, z_InLo);
    bool isOutSgInnerMDPS = modulesInGPU.moduleType[outerInnerLowerModuleIndex] == lst::PS;
    bool isInSgInnerMDPS = modulesInGPU.moduleType[innerInnerLowerModuleIndex] == lst::PS;

    float rtGeom = (isInSgInnerMDPS and isOutSgInnerMDPS)  ? 2.f * lst::kPixelPSZpitch
                   : (isInSgInnerMDPS or isOutSgInnerMDPS) ? lst::kPixelPSZpitch + lst::kStrip2SZpitch
                                                           : 2.f * lst::kStrip2SZpitch;

    // float dz = z_OutLo - z_InLo;
    // float rtLo = rt_InLo * (1.f + dz / (z_InLo + dLum) / dzDrtScale) - rtGeom;  //slope correction only on the lower end

    float rtOut = rt_OutLo;

    //Cut #1: rt condition

    // float rtHi = rt_InLo * (1.f + dz / (z_InLo - dLum)) + rtGeom;

    // if ((rtOut < rtLo) || (rtOut > rtHi))
    //   return false;

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

    float deltaPhiPos = lst::phi_mpi_pi(acc, mdsInGPU.anchorPhi[fourthMDIndex] - mdsInGPU.anchorPhi[firstMDIndex]);

    if (alpaka::math::abs(acc, deltaPhiPos) > dPhiCut)
      return false;

    return true;

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
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool basicCutsT4(TAcc const& acc,
                                                  struct lst::MiniDoublets& mdsInGPU,
                                                  float innerRadius,
                                                  float outerRadius,
                                                  unsigned int firstMDIndex,
                                                  unsigned int secondMDIndex,
                                                  unsigned int thirdMDIndex,
                                                  unsigned int fourthMDIndex) {
    float absEta1 = alpaka::math::abs(acc, mdsInGPU.anchorEta[firstMDIndex]);
    float absEta2 = alpaka::math::abs(acc, mdsInGPU.anchorEta[secondMDIndex]);
    float absEta3 = alpaka::math::abs(acc, mdsInGPU.anchorEta[thirdMDIndex]);
    float absEta4 = alpaka::math::abs(acc, mdsInGPU.anchorEta[fourthMDIndex]);

    float dEta12 = alpaka::math::abs(acc, absEta2-absEta1); 
    float dEta23 = alpaka::math::abs(acc, absEta3-absEta2);
    float dEta34 = alpaka::math::abs(acc, absEta4-absEta3);  

    float radRatio = innerRadius/outerRadius;
     //90% cut
    // if (radRatio > 1.65834f) //no T3 DNN
    if (radRatio > 1.60852f) //add T3 DNN
      return false;
    
    // if (dEta12 > 0.06549f) //no T3 DNN
    if (dEta12 > 0.06488f) // add T3 DNN
      return false;
    // if (dEta23 > 0.04392f) // no T3 DNN
    if (dEta23 > 0.04334f) //add T3 DNN
      return false;
    // if (dEta34 > 0.04020f) // no T3 DNN
    if (dEta34 > 0.03997f) //add T3 DNN
      return false;
    
    //95% cut
    // if (radRatio > 2.43435f)
    //   return false;
    
    // if (dEta12 > 0.08445f)
    //   return false;
    // if (dEta23 > 0.05428f)
    //   return false;
    // if (dEta34 > 0.04872f)
    //   return false; 
    
    return true;
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
                                                               bool& TightPromptFlag,
                                                               bool& TightDisplacedFlag,
                                                               bool& TightCutFlag,
                                                               float* error2s) {
    unsigned int firstSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
    unsigned int secondSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
    unsigned int thirdSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex]; //second and third segments are the same here
    unsigned int fourthSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];

    unsigned int innerOuterInnerMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * secondSegmentIndex];  //inner triplet outer segment inner MD index
    unsigned int innerOuterOuterMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];  //inner triplet outer segment outer MD index
    unsigned int outerInnerInnerMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * thirdSegmentIndex];  //outer triplet inner segment inner MD index
    // unsigned int outerOuterInnerMiniDoubletIndex =
    //     segmentsInGPU.mdIndices[2 * fourthSegmentIndex];  //outer triplet outer segment inner MD index
    unsigned int outerOuterInnerMiniDoubletIndex =
        segmentsInGPU.mdIndices[2 * thirdSegmentIndex + 1];  //outer triplet outer segment inner MD index

    //check if the 2 T3s have a common LS
    if (innerOuterInnerMiniDoubletIndex != outerInnerInnerMiniDoubletIndex)
      return false;
    if (innerOuterOuterMiniDoubletIndex != outerOuterInnerMiniDoubletIndex)
      return false; 
    
    // require both T3s to have the same charge
    int innerT3charge = tripletsInGPU.charge[innerTripletIndex];
    int outerT3charge = tripletsInGPU.charge[outerTripletIndex];
    if (innerT3charge != outerT3charge)
      return false;

    unsigned int firstMDIndex = segmentsInGPU.mdIndices[2 * firstSegmentIndex];
    unsigned int secondMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex];
    unsigned int thirdMDIndex = segmentsInGPU.mdIndices[2 * secondSegmentIndex + 1];
    unsigned int fourthMDIndex = segmentsInGPU.mdIndices[2 * fourthSegmentIndex + 1]; 

    float x1 = mdsInGPU.anchorX[firstMDIndex];
    float x2 = mdsInGPU.anchorX[secondMDIndex];
    float x3 = mdsInGPU.anchorX[thirdMDIndex];
    float x4 = mdsInGPU.anchorX[fourthMDIndex];

    float y1 = mdsInGPU.anchorY[firstMDIndex];
    float y2 = mdsInGPU.anchorY[secondMDIndex];
    float y3 = mdsInGPU.anchorY[thirdMDIndex];
    float y4 = mdsInGPU.anchorY[fourthMDIndex];

    float inner_circleCenterX = tripletsInGPU.circleCenterX[innerTripletIndex];
    float inner_circleCenterY = tripletsInGPU.circleCenterY[innerTripletIndex];
    float innerRadius = tripletsInGPU.circleRadius[innerTripletIndex];
    float outerRadius = tripletsInGPU.circleRadius[outerTripletIndex];
    float inner_pt = 2 * k2Rinv1GeVf * innerRadius;
    float pt = (innerRadius+outerRadius) * k2Rinv1GeVf;

    // if (not basicCutsT4(acc,
    //                     mdsInGPU,
    //                     innerRadius,
    //                     outerRadius,
    //                     firstMDIndex,
    //                     secondMDIndex,
    //                     thirdMDIndex,
    //                     fourthMDIndex))
    //   return false;

    const int moduleType1 = modulesInGPU.moduleType[lowerModuleIndex1];  //0 is ps, 1 is 2s
    const int moduleType2 = modulesInGPU.moduleType[lowerModuleIndex2];
    const int moduleType3 = modulesInGPU.moduleType[lowerModuleIndex3];
    const int moduleType4 = modulesInGPU.moduleType[lowerModuleIndex4];
    
    for (size_t i = 0; i < 4; i++) {
      float error2;
      int moduleTypei;
      if (i == 0) {
        moduleTypei = moduleType1;
      } else if (i == 1) {
        moduleTypei = moduleType2;
      } else if (i == 2) {
        moduleTypei = moduleType3;
      } else if (i == 3) {
        moduleTypei = moduleType4;
      } 
      if (moduleTypei == 0) {
        error2 = kPixelPSZpitch * kPixelPSZpitch;
      } else  //2S modules
      {
        error2 = kStrip2SZpitch * kStrip2SZpitch;
      }

      //check the tilted module, side: PosZ, NegZ, Center(for not tilted)
      float drdz;
      short side, subdets;
      if (i == 0) {
        drdz = alpaka::math::abs(acc, modulesInGPU.drdzs[lowerModuleIndex1]);
        side = modulesInGPU.sides[lowerModuleIndex1];
        subdets = modulesInGPU.subdets[lowerModuleIndex1];
      }
      if (i == 1) {
        drdz = alpaka::math::abs(acc, modulesInGPU.drdzs[lowerModuleIndex2]);
        side = modulesInGPU.sides[lowerModuleIndex2];
        subdets = modulesInGPU.subdets[lowerModuleIndex2];
      }
      if (i == 2) {
        drdz = alpaka::math::abs(acc, modulesInGPU.drdzs[lowerModuleIndex3]);
        side = modulesInGPU.sides[lowerModuleIndex3];
        subdets = modulesInGPU.subdets[lowerModuleIndex3];
      }
      if (i==0 || i == 1 || i == 2) {
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
      error2s[i] = error2;
    }
    
    // 4 categories for sigmas
    float sigmas2[4], delta1[4], delta2[4], slopes[4];
    bool isFlat[4];

    float xVec[] = {x1, x2, x3, x4};
    float yVec[] = {y1, y2, y3, y4};

    const uint16_t lowerModuleIndices[] = {
        lowerModuleIndex1, lowerModuleIndex2, lowerModuleIndex3, lowerModuleIndex4};

    computeSigmasForRegressionT4(acc, modulesInGPU, lowerModuleIndices, delta1, delta2, slopes, isFlat);
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
    float nonAnchorxs[] = {mdsInGPU.outerX[firstMDIndex],
                           mdsInGPU.outerX[secondMDIndex],
                           mdsInGPU.outerX[thirdMDIndex],
                           mdsInGPU.outerX[fourthMDIndex]};
    float nonAnchorys[] = {mdsInGPU.outerY[firstMDIndex],
                           mdsInGPU.outerY[secondMDIndex],
                           mdsInGPU.outerY[thirdMDIndex],
                           mdsInGPU.outerY[fourthMDIndex]};

    computeSigmasForRegressionT4(acc,
                               modulesInGPU,
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
                                              mdsInGPU,
                                              modulesInGPU,
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
                                              TightPromptFlag,
                                              TightDisplacedFlag,
                                              error2s,
                                              regressionRadius,
                                              nonAnchorRegressionRadius,
                                              tripletsInGPU.fakeScore[innerTripletIndex],
                                              tripletsInGPU.promptScore[innerTripletIndex],
                                              tripletsInGPU.displacedScore[innerTripletIndex],
                                              tripletsInGPU.fakeScore[outerTripletIndex],
                                              tripletsInGPU.promptScore[outerTripletIndex],
                                              tripletsInGPU.displacedScore[outerTripletIndex]);

    if (!inference)
      return false;
    // if (not runQuadrupletdBetaAlgoSelector(acc,
    //                                       modulesInGPU,
    //                                       mdsInGPU,
    //                                       segmentsInGPU,
    //                                       lowerModuleIndex1,
    //                                       lowerModuleIndex2,
    //                                       lowerModuleIndex3,
    //                                       lowerModuleIndex4,
    //                                       firstSegmentIndex,
    //                                       thirdSegmentIndex,
    //                                       firstMDIndex,
    //                                       secondMDIndex,
    //                                       thirdMDIndex,
    //                                       fourthMDIndex,
    //                                       dBeta,
    //                                       ptCut))
    //   return false;
    //run Beta Selector for high pT T4s
    if (pt >10) {
      if (not runQuadrupletdBetaAlgoSelector(acc,
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
                                           dBeta,
                                           ptCut))
      return false;
    }
    

    // if (not runQuadrupletAlgoSelector(acc,
    //                                   modulesInGPU,
    //                                   mdsInGPU,
    //                                   segmentsInGPU,
    //                                   lowerModuleIndex1,
    //                                   lowerModuleIndex2,
    //                                   lowerModuleIndex3,
    //                                   lowerModuleIndex4,
    //                                   firstSegmentIndex,
    //                                   thirdSegmentIndex,
    //                                   firstMDIndex,
    //                                   secondMDIndex,
    //                                   thirdMDIndex,
    //                                   fourthMDIndex,
    //                                   ptCut))
    //   return false;
    // if (pt>10){
    //   if (not passT4RZConstraint(acc,
    //                             modulesInGPU,
    //                             mdsInGPU,
    //                             firstMDIndex, 
    //                             secondMDIndex, 
    //                             thirdMDIndex, 
    //                             fourthMDIndex, 
    //                             lowerModuleIndex1, 
    //                             lowerModuleIndex2, 
    //                             lowerModuleIndex3, 
    //                             lowerModuleIndex4, 
    //                             rzChiSquared, 
    //                             inner_pt, 
    //                             innerRadius, 
    //                             inner_circleCenterX, 
    //                             inner_circleCenterY, 
    //                             innerT3charge))
    //     return false;
    // }
    if (not passT4RZConstraint(acc,
                              modulesInGPU,
                              mdsInGPU,
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
                              TightCutFlag))
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
    // float rphisum = chiSquared + nonAnchorChiSquared; 

    // if (rphisum < 5.76f)
    //   TightRPhiFlag = true; //~95% retention for displaced tracks
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
        // continue; //dont make any t4s
        uint16_t lowerModule1 = rangesInGPU.indicesOfEligibleT4Modules[iter];
        short layer2_adjustment;
        short md_adjustment;
        int layer = modulesInGPU.layers[lowerModule1];
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
        unsigned int nInnerTriplets = tripletsInGPU.nTriplets[lowerModule1];
        for (unsigned int innerTripletArrayIndex = globalThreadIdx[1]; innerTripletArrayIndex < nInnerTriplets;
             innerTripletArrayIndex += gridThreadExtent[1]) {
          unsigned int innerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule1] + innerTripletArrayIndex;
          // if (tripletsInGPU.partOfPT5[innerTripletIndex])
          //     continue;  //don't create T4s for T3s accounted in pT5s
          // if (tripletsInGPU.partOfPT3[innerTripletIndex])
          //     continue;  //don't create T4s for T3s accounted in pT3s
          // if (tripletsInGPU.partOfT5[innerTripletIndex])
          //     continue;  //don't create T4s for T3s accounted in T5s
          uint16_t lowerModule2 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * innerTripletIndex + 1];
          unsigned int nOuterTriplets = tripletsInGPU.nTriplets[lowerModule2];
          for (unsigned int outerTripletArrayIndex = globalThreadIdx[2]; outerTripletArrayIndex < nOuterTriplets;
               outerTripletArrayIndex += gridThreadExtent[2]) {
            unsigned int outerTripletIndex = rangesInGPU.tripletModuleIndices[lowerModule2] + outerTripletArrayIndex;
            // if (tripletsInGPU.partOfPT5[outerTripletIndex])
            //   continue;  //don't create T4s for T3s accounted in pT5s
            // if (tripletsInGPU.partOfPT3[outerTripletIndex])
            //   continue;  //don't create T4s for T3s accounted in pT3s
            // if (tripletsInGPU.partOfT5[outerTripletIndex])
            //   continue;  //don't create T4s for T3s accounted in T5s
            uint16_t lowerModule3 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * outerTripletIndex + 1];
            uint16_t lowerModule4 = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers * outerTripletIndex + 2];
            float innerRadius = tripletsInGPU.circleRadius[innerTripletIndex];
            float outerRadius = tripletsInGPU.circleRadius[outerTripletIndex];  
            float rzChiSquared, dBeta, nonAnchorChiSquared, regressionG, regressionF, regressionRadius, nonAnchorRegressionRadius, chiSquared, promptScore, displacedScore, x_5, fakeScore; 
            float error2s[4]; 
            bool TightPromptFlag = false;
            bool TightDisplacedFlag = false;
            bool TightCutFlag = false;
      
            float pt = (innerRadius + outerRadius) * lst::k2Rinv1GeVf;
            // if (pt> 10){
            //   continue; //just build low pt T4s to check
            // }
            //selections: shared LS, same charge, rzChiSquared
            bool success = runQuadrupletDefaultAlgo(acc,
                                                    modulesInGPU,
                                                    mdsInGPU,
                                                    segmentsInGPU,
                                                    tripletsInGPU,
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
                                                    TightPromptFlag,
                                                    TightDisplacedFlag,
                                                    TightCutFlag,
                                                    error2s);
            // bool success = true;

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
                  float phi =
                      mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex +
                                                                                                  layer2_adjustment] + md_adjustment]]; //layer 3
                  float eta =
                      mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex +
                                                                                                  layer2_adjustment] + md_adjustment]]; //layer 3
                  //test phi and eta without layer adjustment
                  // float phi =
                  //     mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex]]];
                  // float eta =
                  //     mdsInGPU.anchorEta[segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex]]];
                  
                  float scores = chiSquared + nonAnchorChiSquared;
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
                                        TightPromptFlag,
                                        TightDisplacedFlag,
                                        TightCutFlag,
                                        error2s);

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

        int dynamic_count = 0;

        // How many triplets are in module i?
        int nTriplets_i = tripletsInGPU.nTriplets[i];
        int firstTripletIdx = rangesInGPU.tripletModuleIndices[i];

        // Loop over all triplets that live in module i
        for (int t = 0; t < nTriplets_i; t++) {
          int tripletIndex = firstTripletIdx + t;
          uint16_t outerModule = tripletsInGPU.lowerModuleIndices[Params_T3::kLayers *tripletIndex+1];
          dynamic_count += tripletsInGPU.nTriplets[outerModule];
        }

        // int nEligibleT4Modules = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &nEligibleT4Modulesx, 1);

        int category_number = lst::getCategoryNumber(module_layers, module_subdets, module_rings);
        int eta_number = lst::getEtaBin(module_eta);

//         int occupancy = 0;
//         if (category_number != -1 && eta_number != -1) {
//           occupancy = occupancy_matrix[category_number][eta_number];
//         }
// #ifdef WARNINGS
//         else {
//           printf("Unhandled case in createEligibleModulesListForQuadrupletsGPU! Module index = %i\n", i);
//         }
// #endif
        // Get matrix-based cap (use dynamic_count as fallback)
        int matrix_cap =
            (category_number != -1 && eta_number != -1) ? occupancy_matrix[category_number][eta_number] : 0;
        // Cap occupancy at minimum of dynamic count and matrix value
        int occupancy = alpaka::math::min(acc, dynamic_count, matrix_cap);
        if (dynamic_count > matrix_cap){
          printf("dynamic count: %d, matrix_cap: %d\n", dynamic_count, matrix_cap);
        }

        int nEligibleT4Modules = alpaka::atomicAdd(acc, &nEligibleT4Modulesx, 1);
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
