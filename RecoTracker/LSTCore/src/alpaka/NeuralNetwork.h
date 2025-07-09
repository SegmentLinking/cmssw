#ifndef RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h
#define RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"

#include "T5NeuralNetworkWeights.h"
#include "T3NeuralNetworkWeights.h"
#include "pT3NeuralNetworkWeights.h"
#include "T5EmbedNetworkWeights.h"
#include "pLSEmbedNetworkWeights.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  template <int FEATURES, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void softmax_activation(TAcc const& acc, float (&input)[FEATURES]) {
    float sum = 0.f;
    // Compute exp and sum
    CMS_UNROLL_LOOP
    for (unsigned int i = 0; i < FEATURES; ++i) {
      input[i] = alpaka::math::exp(acc, input[i]);
      sum += input[i];
    }

    // Normalize
    CMS_UNROLL_LOOP
    for (unsigned int i = 0; i < FEATURES; ++i) {
      input[i] /= sum;
    }
  }

  template <int FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void relu_activation(float (&input)[FEATURES]) {
    CMS_UNROLL_LOOP
    for (unsigned int col = 0; col < FEATURES; ++col) {
      input[col] = (input[col] > 0.f) ? input[col] : 0.f;
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float sigmoid_activation(TAcc const& acc, const float x) {
    return alpaka::math::exp(acc, x) / (alpaka::math::exp(acc, x) + 1.f);
  }

  template <int IN_FEATURES, int OUT_FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void linear_layer(const float (&input)[IN_FEATURES],
                                                   float (&output)[OUT_FEATURES],
                                                   const float (&weights)[IN_FEATURES][OUT_FEATURES],
                                                   const float (&biases)[OUT_FEATURES]) {
    CMS_UNROLL_LOOP
    for (unsigned int i = 0; i < OUT_FEATURES; ++i) {
      output[i] = biases[i];
      CMS_UNROLL_LOOP
      for (int j = 0; j < IN_FEATURES; ++j) {
        output[i] += input[j] * weights[j][i];
      }
    }
  }

  namespace t3dnn {
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                     MiniDoubletsConst mds,
                                                     const unsigned int mdIndex1,
                                                     const unsigned int mdIndex2,
                                                     const unsigned int mdIndex3,
                                                     const float radius,
                                                     const float betaIn,
                                                     float (&output)[dnn::t3dnn::kOutputFeatures]) {
      // Constants for T3 DNN
      constexpr unsigned int kInputFeatures = 14;
      constexpr unsigned int kHiddenFeatures = 32;

      // Extract hit information
      float eta1 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex1]);  // inner T3 anchor hit 1 eta
      float eta2 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex2]);  // inner T3 anchor hit 2 eta
      float eta3 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex3]);  // inner T3 anchor hit 3 eta

      float phi1 = mds.anchorPhi()[mdIndex1];  // inner T3 anchor hit 1 phi
      float phi2 = mds.anchorPhi()[mdIndex2];  // inner T3 anchor hit 2 phi
      float phi3 = mds.anchorPhi()[mdIndex3];  // inner T3 anchor hit 3 phi

      float z1 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex1]);  // inner T3 anchor hit 1 z
      float z2 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex2]);  // inner T3 anchor hit 2 z
      float z3 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex3]);  // inner T3 anchor hit 3 z

      float r1 = mds.anchorRt()[mdIndex1];  // inner T3 anchor hit 1 r
      float r2 = mds.anchorRt()[mdIndex2];  // inner T3 anchor hit 2 r
      float r3 = mds.anchorRt()[mdIndex3];  // inner T3 anchor hit 3 r

      // Build input feature vector matching training order
      float x[kInputFeatures] = {
          eta1 / dnn::t3dnn::kEta_norm,                   // First hit eta normalized
          alpaka::math::abs(acc, phi1) / dnn::kPhi_norm,  // First hit phi normalized
          z1 / dnn::t3dnn::kZ_max,                        // First hit z normalized
          r1 / dnn::t3dnn::kR_max,                        // First hit r normalized

          eta2 - eta1,                                                   // Difference in eta between hit 2 and 1
          cms::alpakatools::deltaPhi(acc, phi2, phi1) / dnn::kPhi_norm,  // Difference in phi between hit 2 and 1
          (z2 - z1) / dnn::t3dnn::kZ_max,  // Difference in z between hit 2 and 1 normalized
          (r2 - r1) / dnn::t3dnn::kR_max,  // Difference in r between hit 2 and 1 normalized

          eta3 - eta2,                                                   // Difference in eta between hit 3 and 2
          cms::alpakatools::deltaPhi(acc, phi3, phi2) / dnn::kPhi_norm,  // Difference in phi between hit 3 and 2
          (z3 - z2) / dnn::t3dnn::kZ_max,  // Difference in z between hit 3 and 2 normalized
          (r3 - r2) / dnn::t3dnn::kR_max,  // Difference in r between hit 3 and 2 normalized

          alpaka::math::log10(acc, radius),  // T3's circle radius
          betaIn                             // Beta angle of inner segment
      };

      float x_1[kHiddenFeatures];  // Layer 1 output
      float x_2[kHiddenFeatures];  // Layer 2 output

      // Layer 1: Linear + Relu
      linear_layer<kInputFeatures, kHiddenFeatures>(x, x_1, dnn::t3dnn::wgtT_layer1, dnn::t3dnn::bias_layer1);
      relu_activation<kHiddenFeatures>(x_1);

      // Layer 2: Linear + Relu
      linear_layer<kHiddenFeatures, kHiddenFeatures>(x_1, x_2, dnn::t3dnn::wgtT_layer2, dnn::t3dnn::bias_layer2);
      relu_activation<kHiddenFeatures>(x_2);

      // Layer 3: Linear + Softmax
      linear_layer<kHiddenFeatures, dnn::t3dnn::kOutputFeatures>(
          x_2, output, dnn::t3dnn::wgtT_output_layer, dnn::t3dnn::bias_output_layer);
      softmax_activation<dnn::t3dnn::kOutputFeatures>(acc, output);

      // Get pt and eta bin indices
      float t3_pt = radius * lst::k2Rinv1GeVf * 2;
      uint8_t pt_index = (t3_pt > 5);
      uint8_t bin_index = (eta1 > 2.5f) ? (dnn::kEtaBins - 1) : static_cast<unsigned int>(eta1 / dnn::kEtaSize);

      return output[1] > dnn::t3dnn::kWp_prompt[pt_index][bin_index] ||
             output[2] > dnn::t3dnn::kWp_displaced[pt_index][bin_index];
    }
  }  // namespace t3dnn

  namespace pt3dnn {

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                     const float rPhiChiSquared,
                                                     const float tripletRadius,
                                                     const float pixelRadius,
                                                     const float pixRadiusError,
                                                     const float rzChiSquared,
                                                     const float pixelEta,
                                                     const float pixelPt) {
      constexpr unsigned int kInputFeatures = 6;
      constexpr unsigned int kHiddenFeatures = 32;
      constexpr unsigned int kOutputFeatures = 1;

      float x[kInputFeatures] = {alpaka::math::log10(acc, rPhiChiSquared),
                                 alpaka::math::log10(acc, tripletRadius),
                                 alpaka::math::log10(acc, pixelRadius),
                                 alpaka::math::log10(acc, pixRadiusError),
                                 alpaka::math::log10(acc, rzChiSquared),
                                 alpaka::math::abs(acc, pixelEta) / dnn::pt3dnn::kEta_norm};

      float x1[kHiddenFeatures];
      float x2[kHiddenFeatures];
      float x3[kOutputFeatures];

      linear_layer<kInputFeatures, kHiddenFeatures>(x, x1, dnn::pt3dnn::wgtT_layer1, dnn::pt3dnn::bias_layer1);
      relu_activation<kHiddenFeatures>(x1);

      linear_layer<kHiddenFeatures, kHiddenFeatures>(x1, x2, dnn::pt3dnn::wgtT_layer2, dnn::pt3dnn::bias_layer2);
      relu_activation<kHiddenFeatures>(x2);

      linear_layer<kHiddenFeatures, kOutputFeatures>(
          x2, x3, dnn::pt3dnn::wgtT_output_layer, dnn::pt3dnn::bias_output_layer);
      float output = sigmoid_activation(acc, x3[0]);

      uint8_t bin_index = (alpaka::math::abs(acc, pixelEta) > 2.5f)
                              ? (dnn::kEtaBins - 1)
                              : static_cast<unsigned int>(alpaka::math::abs(acc, pixelEta) / dnn::kEtaSize);

      if (pixelPt > 5.0f)
        return output > dnn::pt3dnn::kWpHigh;

      return output > dnn::pt3dnn::kWp[bin_index];
    }

  }  // namespace pt3dnn

  namespace t5dnn {

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float runInference(TAcc const& acc,
                                                      ModulesConst modules,
                                                      MiniDoubletsConst mds,
                                                      SegmentsConst segments,
                                                      TripletsConst triplets,
                                                      const float* xVec,
                                                      const float* yVec,
                                                      const unsigned int* mdIndices,
                                                      const uint16_t* lowerModuleIndices,
                                                      unsigned int innerTripletIndex,
                                                      unsigned int outerTripletIndex,
                                                      float innerRadius,
                                                      float outerRadius,
                                                      float bridgeRadius) {
      // Unpack x-coordinates of hits
      float x1 = xVec[0];
      float x2 = xVec[1];
      float x3 = xVec[2];
      float x4 = xVec[3];
      float x5 = xVec[4];
      // Unpack y-coordinates of hits
      float y1 = yVec[0];
      float y2 = yVec[1];
      float y3 = yVec[2];
      float y4 = yVec[3];
      float y5 = yVec[4];
      // Unpack module indices
      unsigned int mdIndex1 = mdIndices[0];
      unsigned int mdIndex2 = mdIndices[1];
      unsigned int mdIndex3 = mdIndices[2];
      unsigned int mdIndex4 = mdIndices[3];
      unsigned int mdIndex5 = mdIndices[4];
      // Unpack module indices
      uint16_t lowerModuleIndex1 = lowerModuleIndices[0];
      uint16_t lowerModuleIndex2 = lowerModuleIndices[1];
      uint16_t lowerModuleIndex3 = lowerModuleIndices[2];
      uint16_t lowerModuleIndex4 = lowerModuleIndices[3];
      uint16_t lowerModuleIndex5 = lowerModuleIndices[4];
      // Compute some convenience variables
      short layer2_adjustment = 0;
      if (modules.layers()[lowerModuleIndex1] == 1) {
        layer2_adjustment = 1;  // get upper segment to be in second layer
      }
      unsigned int md_idx_for_t5_eta_phi =
          segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][0]][layer2_adjustment];
      bool is_endcap1 = (modules.subdets()[lowerModuleIndex1] == 4);  // true if anchor hit 1 is in the endcap
      bool is_endcap2 = (modules.subdets()[lowerModuleIndex2] == 4);  // true if anchor hit 2 is in the endcap
      bool is_endcap3 = (modules.subdets()[lowerModuleIndex3] == 4);  // true if anchor hit 3 is in the endcap
      bool is_endcap4 = (modules.subdets()[lowerModuleIndex4] == 4);  // true if anchor hit 4 is in the endcap
      bool is_endcap5 = (modules.subdets()[lowerModuleIndex5] == 4);  // true if anchor hit 5 is in the endcap

      // Build DNN input vector (corresponding output N-tuple branch noted in parenthetical in comment)
      float x[38] = {
          alpaka::math::log10(acc, 2 * k2Rinv1GeVf * innerRadius),      // inner T3 pT (t3_pt)
          mds.anchorEta()[mdIndex1],                                    // inner T3 anchor hit 1 eta (t3_0_eta)
          mds.anchorPhi()[mdIndex1],                                    // inner T3 anchor hit 1 phi (t3_0_phi)
          mds.anchorZ()[mdIndex1],                                      // inner T3 anchor hit 1 z (t3_0_z)
          alpaka::math::sqrt(acc, x1 * x1 + y1 * y1),                   // inner T3 anchor hit 1 r (t3_0_r)
          float(modules.layers()[lowerModuleIndex1] + 6 * is_endcap1),  // inner T3 anchor hit 1 layer (t3_0_layer)
          mds.anchorEta()[mdIndex2],                                    // inner T3 anchor hit 2 eta (t3_2_eta)
          mds.anchorPhi()[mdIndex2],                                    // inner T3 anchor hit 2 phi (t3_2_phi)
          mds.anchorZ()[mdIndex2],                                      // inner T3 anchor hit 2 z (t3_2_z)
          alpaka::math::sqrt(acc, x2 * x2 + y2 * y2),                   // inner T3 anchor hit 2 r (t3_2_r)
          float(modules.layers()[lowerModuleIndex2] + 6 * is_endcap2),  // inner T3 anchor hit 2 layer (t3_2_layer)
          mds.anchorEta()[mdIndex3],                                    // inner T3 anchor hit 3 eta (t3_4_eta)
          mds.anchorPhi()[mdIndex3],                                    // inner T3 anchor hit 3 phi (t3_4_phi)
          mds.anchorZ()[mdIndex3],                                      // inner T3 anchor hit 3 z (t3_4_z)
          alpaka::math::sqrt(acc, x3 * x3 + y3 * y3),                   // inner T3 anchor hit 3 r (t3_4_r)
          float(modules.layers()[lowerModuleIndex3] + 6 * is_endcap3),  // inner T3 anchor hit 3 layer (t3_4_layer)
          alpaka::math::log10(acc, 2 * k2Rinv1GeVf * outerRadius),      // outer T3 pT (t3_pt)
          mds.anchorEta()[mdIndex3],                                    // outer T3 anchor hit 4 eta (t3_0_eta)
          mds.anchorPhi()[mdIndex3],                                    // outer T3 anchor hit 4 phi (t3_0_phi)
          mds.anchorZ()[mdIndex3],                                      // outer T3 anchor hit 3 eta (t3_0_z)
          alpaka::math::sqrt(acc, x3 * x3 + y3 * y3),                   // outer T3 anchor hit 3 r (t3_0_r)
          float(modules.layers()[lowerModuleIndex3] + 6 * is_endcap3),  // outer T3 anchor hit 3 layer (t3_0_layer)
          mds.anchorEta()[mdIndex4],                                    // outer T3 anchor hit 4 eta (t3_2_eta)
          mds.anchorPhi()[mdIndex4],                                    // outer T3 anchor hit 4 phi (t3_2_phi)
          mds.anchorZ()[mdIndex4],                                      // outer T3 anchor hit 4 z (t3_2_z)
          alpaka::math::sqrt(acc, x4 * x4 + y4 * y4),                   // outer T3 anchor hit 4 r (t3_2_r)
          float(modules.layers()[lowerModuleIndex4] + 6 * is_endcap4),  // outer T3 anchor hit 4 layer (t3_2_layer)
          mds.anchorEta()[mdIndex5],                                    // outer T3 anchor hit 5 eta (t3_4_eta)
          mds.anchorPhi()[mdIndex5],                                    // outer T3 anchor hit 5 phi (t3_4_phi)
          mds.anchorZ()[mdIndex5],                                      // outer T3 anchor hit 5 z (t3_4_z)
          alpaka::math::sqrt(acc, x5 * x5 + y5 * y5),                   // outer T3 anchor hit 5 r (t3_4_r)
          float(modules.layers()[lowerModuleIndex5] + 6 * is_endcap5),  // outer T3 anchor hit 5 layer (t3_4_layer)
          alpaka::math::log10(acc, (innerRadius + outerRadius) * k2Rinv1GeVf),  // T5 pT (t5_pt)
          mds.anchorEta()[md_idx_for_t5_eta_phi],                               // T5 eta (t5_eta)
          mds.anchorPhi()[md_idx_for_t5_eta_phi],                               // T5 phi (t5_phi)
          alpaka::math::log10(acc, innerRadius),                                // T5 inner radius (t5_innerRadius)
          alpaka::math::log10(acc, bridgeRadius),                               // T5 bridge radius (t5_bridgeRadius)
          alpaka::math::log10(acc, outerRadius)                                 // T5 outer radius (t5_outerRadius)
      };

      // (0): Linear(in_features=38, out_features=32, bias=True) => x = x*W_T + b
      float x_0[32];
      for (unsigned int col = 0; col < 32; ++col) {
        x_0[col] = 0;
        for (unsigned int inner = 0; inner < 38; ++inner) {
          x_0[col] += x[inner] * dnn::t5dnn::wgtT_0[inner][col];
        }
        x_0[col] += dnn::t5dnn::bias_0[col];
      }

      // (1): ReLU()
      float x_1[32];
      for (unsigned int col = 0; col < 32; ++col) {
        x_1[col] = (x_0[col] > 0.f) ? x_0[col] : 0.f;
      }

      // (2): Linear(in_features=32, out_features=32, bias=True) => x = x*W_T + b
      float x_2[32];
      for (unsigned int col = 0; col < 32; ++col) {
        x_2[col] = 0;
        for (unsigned int inner = 0; inner < 32; ++inner) {
          x_2[col] += x_1[inner] * dnn::t5dnn::wgtT_2[inner][col];
        }
        x_2[col] += dnn::t5dnn::bias_2[col];
      }

      // (3): ReLU()
      float x_3[32];
      for (unsigned int col = 0; col < 32; ++col) {
        x_3[col] = (x_2[col] > 0.f) ? x_2[col] : 0.f;
      }

      // (4): Linear(in_features=32, out_features=1, bias=True) => x = x*W_T + b
      float x_4[1];
      for (unsigned int col = 0; col < 1; ++col) {
        x_4[col] = 0;
        for (unsigned int inner = 0; inner < 32; ++inner) {
          x_4[col] += x_3[inner] * dnn::t5dnn::wgtT_4[inner][col];
        }
        x_4[col] += dnn::t5dnn::bias_4[col];
      }

      // (5): Sigmoid()
      float x_5[1];
      for (unsigned int col = 0; col < 1; ++col) {
        x_5[col] = alpaka::math::exp(acc, x_4[col]) / (alpaka::math::exp(acc, x_4[col]) + 1);
      }

      return x_5[0];
    }

  }  // namespace t5dnn

  namespace t5embdnn {
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void runEmbed(TAcc const& acc,
                                                 MiniDoubletsConst mds,
                                                 unsigned int mdIndex1,
                                                 unsigned int mdIndex2,
                                                 unsigned int mdIndex3,
                                                 unsigned int mdIndex4,
                                                 unsigned int mdIndex5,
                                                 float innerRadius,
                                                 float outerRadius,
                                                 float bridgeRadius,
                                                 float fakeScore1,
                                                 float promptScore1,
                                                 float dispScore1,
                                                 float fakeScore2,
                                                 float promptScore2,
                                                 float dispScore2,
                                                 float (&embedding)[Params_T5::kEmbed]) {
      constexpr unsigned int kInputFeatures = 30;
      constexpr unsigned int kHiddenFeatures = 32;

      float eta1 = mds.anchorEta()[mdIndex1];
      float eta2 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex2]);
      float eta3 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex3]);
      float eta4 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex4]);
      float eta5 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex5]);

      float phi1 = mds.anchorPhi()[mdIndex1];
      float phi2 = mds.anchorPhi()[mdIndex2];
      float phi3 = mds.anchorPhi()[mdIndex3];
      float phi4 = mds.anchorPhi()[mdIndex4];
      float phi5 = mds.anchorPhi()[mdIndex5];

      float z1 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex1]);
      float z2 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex2]);
      float z3 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex3]);
      float z4 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex4]);
      float z5 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex5]);

      float r1 = mds.anchorRt()[mdIndex1];
      float r2 = mds.anchorRt()[mdIndex2];
      float r3 = mds.anchorRt()[mdIndex3];
      float r4 = mds.anchorRt()[mdIndex4];
      float r5 = mds.anchorRt()[mdIndex5];

      float x[kInputFeatures] = {eta1 / dnn::t5dnn::kEta_norm,
                                 alpaka::math::cos(acc, phi1),
                                 alpaka::math::sin(acc, phi1),
                                 z1 / dnn::t5dnn::kZ_max,
                                 r1 / dnn::t5dnn::kR_max,

                                 eta2 - alpaka::math::abs(acc, eta1),
                                 cms::alpakatools::deltaPhi(acc, phi2, phi1),
                                 (z2 - z1) / dnn::t5dnn::kZ_max,
                                 (r2 - r1) / dnn::t5dnn::kR_max,

                                 eta3 - eta2,
                                 cms::alpakatools::deltaPhi(acc, phi3, phi2),
                                 (z3 - z2) / dnn::t5dnn::kZ_max,
                                 (r3 - r2) / dnn::t5dnn::kR_max,

                                 eta4 - eta3,
                                 cms::alpakatools::deltaPhi(acc, phi4, phi3),
                                 (z4 - z3) / dnn::t5dnn::kZ_max,
                                 (r4 - r3) / dnn::t5dnn::kR_max,

                                 eta5 - eta4,
                                 cms::alpakatools::deltaPhi(acc, phi5, phi4),
                                 (z5 - z4) / dnn::t5dnn::kZ_max,
                                 (r5 - r4) / dnn::t5dnn::kR_max,

                                 1.0f / innerRadius,
                                 1.0f / bridgeRadius,
                                 1.0f / outerRadius,

                                 fakeScore1,
                                 promptScore1,
                                 dispScore1,
                                 (fakeScore2 - fakeScore1),
                                 (promptScore2 - promptScore1),
                                 (dispScore2 - dispScore1)};

      float h1[kHiddenFeatures];
      float h2[kHiddenFeatures];

      linear_layer<kInputFeatures, kHiddenFeatures>(x, h1, dnn::t5embdnn::wgtT_fc1, dnn::t5embdnn::bias_fc1);
      relu_activation<kHiddenFeatures>(h1);

      linear_layer<kHiddenFeatures, kHiddenFeatures>(h1, h2, dnn::t5embdnn::wgtT_fc2, dnn::t5embdnn::bias_fc2);
      relu_activation<kHiddenFeatures>(h2);

      linear_layer<kHiddenFeatures, Params_T5::kEmbed>(h2, embedding, dnn::t5embdnn::wgtT_fc3, dnn::t5embdnn::bias_fc3);
    }

  }  // namespace t5embdnn

  namespace plsembdnn {
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void runEmbed(TAcc const& acc,
                                                 const float eta,
                                                 const float etaErr,
                                                 const float phi,
                                                 const float circleCenterX,
                                                 const float circleCenterY,
                                                 const float circleRadius,
                                                 const float ptIn,
                                                 const float ptErr,
                                                 const bool isQuad,
                                                 float (&embedding)[Params_pLS::kEmbed]) {
      constexpr unsigned int kInputFeatures = 10;
      constexpr unsigned int kHiddenFeatures = 32;

      float x[kInputFeatures] = {eta / dnn::plsembdnn::kEta_norm,
                                 etaErr / dnn::plsembdnn::kEtaErr_norm,
                                 alpaka::math::cos(acc, phi),
                                 alpaka::math::sin(acc, phi),
                                 1.0f / ptIn,
                                 alpaka::math::log10(acc, ptErr),
                                 isQuad ? 1.0f : 0.0f,
                                 alpaka::math::log10(acc, alpaka::math::abs(acc, circleCenterX)),
                                 alpaka::math::log10(acc, alpaka::math::abs(acc, circleCenterY)),
                                 alpaka::math::log10(acc, circleRadius)};

      float h1[kHiddenFeatures];
      float h2[kHiddenFeatures];

      linear_layer<kInputFeatures, kHiddenFeatures>(x, h1, dnn::plsembdnn::wgtT_fc1, dnn::plsembdnn::bias_fc1);
      relu_activation<kHiddenFeatures>(h1);

      linear_layer<kHiddenFeatures, kHiddenFeatures>(h1, h2, dnn::plsembdnn::wgtT_fc2, dnn::plsembdnn::bias_fc2);
      relu_activation<kHiddenFeatures>(h2);

      linear_layer<kHiddenFeatures, Params_pLS::kEmbed>(
          h2, embedding, dnn::plsembdnn::wgtT_fc3, dnn::plsembdnn::bias_fc3);
    }

  }  // namespace plsembdnn

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
