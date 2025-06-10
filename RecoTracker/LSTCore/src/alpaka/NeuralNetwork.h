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
#include "NeuralNetworkWeights.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "pT4NeuralNetworkWeights.h"

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
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                     MiniDoubletsConst mds,
                                                     const unsigned int mdIndex1,
                                                     const unsigned int mdIndex2,
                                                     const unsigned int mdIndex3,
                                                     const unsigned int mdIndex4,
                                                     const unsigned int mdIndex5,
                                                     const float innerRadius,
                                                     const float outerRadius,
                                                     const float bridgeRadius) {
      // Constants
      constexpr unsigned int kInputFeatures = 23;
      constexpr unsigned int kHiddenFeatures = 32;

      float eta1 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex1]);  // inner T3 anchor hit 1 eta
      float eta2 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex2]);  // inner T3 anchor hit 2 eta
      float eta3 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex3]);  // inner T3 anchor hit 3 eta
      float eta4 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex4]);  // outer T3 anchor hit 4 eta
      float eta5 = alpaka::math::abs(acc, mds.anchorEta()[mdIndex5]);  // outer T3 anchor hit 5 eta

      float phi1 = mds.anchorPhi()[mdIndex1];  // inner T3 anchor hit 1 phi
      float phi2 = mds.anchorPhi()[mdIndex2];  // inner T3 anchor hit 2 phi
      float phi3 = mds.anchorPhi()[mdIndex3];  // inner T3 anchor hit 3 phi
      float phi4 = mds.anchorPhi()[mdIndex4];  // outer T3 anchor hit 4 phi
      float phi5 = mds.anchorPhi()[mdIndex5];  // outer T3 anchor hit 5 phi

      float z1 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex1]);  // inner T3 anchor hit 1 z
      float z2 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex2]);  // inner T3 anchor hit 2 z
      float z3 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex3]);  // inner T3 anchor hit 3 z
      float z4 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex4]);  // outer T3 anchor hit 4 z
      float z5 = alpaka::math::abs(acc, mds.anchorZ()[mdIndex5]);  // outer T3 anchor hit 5 z

      float r1 = mds.anchorRt()[mdIndex1];  // inner T3 anchor hit 1 r
      float r2 = mds.anchorRt()[mdIndex2];  // inner T3 anchor hit 2 r
      float r3 = mds.anchorRt()[mdIndex3];  // inner T3 anchor hit 3 r
      float r4 = mds.anchorRt()[mdIndex4];  // outer T3 anchor hit 4 r
      float r5 = mds.anchorRt()[mdIndex5];  // outer T3 anchor hit 5 r

      // Build the input feature vector using pairwise differences after the first hit
      float x[kInputFeatures] = {
          eta1 / dnn::t5dnn::kEta_norm,                   // inner T3: First hit eta normalized
          alpaka::math::abs(acc, phi1) / dnn::kPhi_norm,  // inner T3: First hit phi normalized
          z1 / dnn::t5dnn::kZ_max,                        // inner T3: First hit z normalized
          r1 / dnn::t5dnn::kR_max,                        // inner T3: First hit r normalized

          eta2 - eta1,  // inner T3: Difference in eta between hit 2 and 1
          cms::alpakatools::deltaPhi(acc, phi2, phi1) /
              dnn::kPhi_norm,              // inner T3: Difference in phi between hit 2 and 1
          (z2 - z1) / dnn::t5dnn::kZ_max,  // inner T3: Difference in z between hit 2 and 1 normalized
          (r2 - r1) / dnn::t5dnn::kR_max,  // inner T3: Difference in r between hit 2 and 1 normalized

          eta3 - eta2,  // inner T3: Difference in eta between hit 3 and 2
          cms::alpakatools::deltaPhi(acc, phi3, phi2) /
              dnn::kPhi_norm,              // inner T3: Difference in phi between hit 3 and 2
          (z3 - z2) / dnn::t5dnn::kZ_max,  // inner T3: Difference in z between hit 3 and 2 normalized
          (r3 - r2) / dnn::t5dnn::kR_max,  // inner T3: Difference in r between hit 3 and 2 normalized

          eta4 - eta3,  // outer T3: Difference in eta between hit 4 and 3
          cms::alpakatools::deltaPhi(acc, phi4, phi3) /
              dnn::kPhi_norm,              // outer T3: Difference in phi between hit 4 and 3
          (z4 - z3) / dnn::t5dnn::kZ_max,  // outer T3: Difference in z between hit 4 and 3 normalized
          (r4 - r3) / dnn::t5dnn::kR_max,  // outer T3: Difference in r between hit 4 and 3 normalized

          eta5 - eta4,  // outer T3: Difference in eta between hit 5 and 4
          cms::alpakatools::deltaPhi(acc, phi5, phi4) /
              dnn::kPhi_norm,              // outer T3: Difference in phi between hit 5 and 4
          (z5 - z4) / dnn::t5dnn::kZ_max,  // outer T3: Difference in z between hit 5 and 4 normalized
          (r5 - r4) / dnn::t5dnn::kR_max,  // outer T3: Difference in r between hit 5 and 4 normalized

          alpaka::math::log10(acc, innerRadius),   // T5 inner radius
          alpaka::math::log10(acc, bridgeRadius),  // T5 bridge radius
          alpaka::math::log10(acc, outerRadius)    // T5 outer radius
      };

      float x_1[kHiddenFeatures];  // Layer 1 output
      float x_2[kHiddenFeatures];  // Layer 2 output
      float x_3[1];                // Layer 3 linear output

      // Layer 1: Linear + Relu
      linear_layer<kInputFeatures, kHiddenFeatures>(x, x_1, dnn::t5dnn::wgtT_layer1, dnn::t5dnn::bias_layer1);
      relu_activation<kHiddenFeatures>(x_1);

      // Layer 2: Linear + Relu
      linear_layer<kHiddenFeatures, kHiddenFeatures>(x_1, x_2, dnn::t5dnn::wgtT_layer2, dnn::t5dnn::bias_layer2);
      relu_activation<kHiddenFeatures>(x_2);

      // Layer 3: Linear + Sigmoid
      linear_layer<kHiddenFeatures, 1>(x_2, x_3, dnn::t5dnn::wgtT_output_layer, dnn::t5dnn::bias_output_layer);
      float x_5 = sigmoid_activation(acc, x_3[0]);

      // Get the bin index based on abs(eta) of first hit and t5_pt
      float t5_pt = innerRadius * lst::k2Rinv1GeVf * 2;

      uint8_t pt_index = (t5_pt > 5.0f);
      uint8_t bin_index = (eta1 > 2.5f) ? (dnn::kEtaBins - 1) : static_cast<unsigned int>(eta1 / dnn::kEtaSize);

    // Compare x_5 to the cut value for the relevant bin
    return x_5 > dnn::t5dnn::kWp[pt_index][bin_index];
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

namespace lst::t4dnn {
  template <int FEATURES, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void softmax_activation(TAcc const& acc, float (&input)[FEATURES]) {
    float sum = 0.f;
    // Compute exp and sum
#pragma unroll FEATURES
    for (unsigned int i = 0; i < FEATURES; ++i) {
      input[i] = alpaka::math::exp(acc, input[i]);
      sum += input[i];
    }

    // Normalize
#pragma unroll FEATURES
    for (unsigned int i = 0; i < FEATURES; ++i) {
      input[i] /= sum;
    }
  }

  template <int FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void relu_activation(float (&input)[FEATURES]) {
#pragma unroll FEATURES
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
#pragma unroll OUT_FEATURES
    for (unsigned int i = 0; i < OUT_FEATURES; ++i) {
      output[i] = biases[i];
#pragma unroll IN_FEATURES
      for (int j = 0; j < IN_FEATURES; ++j) {
        output[i] += input[j] * weights[j][i];
      }
    }
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float delta_phi(const float phi1, const float phi2) {
    float delta = phi1 - phi2;
    // Adjust delta to be within the range [-M_PI, M_PI]
    if (delta > kPi) {
      delta -= 2 * kPi;
    } else if (delta < -kPi) {
      delta += 2 * kPi;
    }
    return delta;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                   lst::MiniDoublets const& mdsInGPU,
                                                   lst::Modules const& modulesInGPU,
                                                   const unsigned int mdIndex1,
                                                   const unsigned int mdIndex2,
                                                   const unsigned int mdIndex3,
                                                   const unsigned int mdIndex4,
                                                   uint16_t lowerModuleIndex1,
                                                   uint16_t lowerModuleIndex2,
                                                   uint16_t lowerModuleIndex3,
                                                   uint16_t lowerModuleIndex4,
                                                   const float innerRadius,
                                                   const float outerRadius,
                                                   float& promptScore,
                                                   float& displacedScore,
                                                   float& fakeScore,
                                                   bool& TightPromptFlag,
                                                   bool& TightDisplacedFlag,
                                                   float* errors,
                                                   const float regressionRadius,
                                                   const float nonAnchorRegressionRadius,
                                                   float fakeScore1,
                                                   float promptScore1,
                                                   float displacedScore1,
                                                   float fakeScore2,
                                                   float promptScore2,
                                                   float displacedScore2) {
    // Constants
    // constexpr unsigned int kinputFeatures = 19; //no additional
    constexpr unsigned int kinputFeatures = 27;  //add radii =21, add uncert =23, add radii and t3 scores=27, all = 31
    // constexpr unsigned int kinputFeatures = 31;  //add radii =21, add uncert =23, add radii and t3 scores=27, all = 31
    constexpr unsigned int khiddenFeatures = 32;
    // constexpr unsigned int khiddenFeatures = 64;
    constexpr unsigned int koutputFeatures = 3;

    const int layer1 = modulesInGPU.lstLayers[lowerModuleIndex1];
    const int layer2 = modulesInGPU.lstLayers[lowerModuleIndex2];
    const int layer3 = modulesInGPU.lstLayers[lowerModuleIndex3];
    const int layer4 = modulesInGPU.lstLayers[lowerModuleIndex4];

    float eta1 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex1]);  // inner T3 anchor hit 1 eta (t3_0_eta)
    float eta2 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex2]);  // inner T3 anchor hit 2 eta (t3_2_eta)
    float eta3 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex3]);  // inner T3 anchor hit 3 eta (t3_4_eta)
    float eta4 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex4]);  // outer T3 anchor hit 4 eta (t3_2_eta)

    float phi1 = mdsInGPU.anchorPhi[mdIndex1];  // inner T3 anchor hit 1 phi
    float phi2 = mdsInGPU.anchorPhi[mdIndex2];  // inner T3 anchor hit 2 phi
    float phi3 = mdsInGPU.anchorPhi[mdIndex3];  // inner T3 anchor hit 3 phi
    float phi4 = mdsInGPU.anchorPhi[mdIndex4];  // outer T3 anchor hit 4 phi

    float z1 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex1]);  // inner T3 anchor hit 1 z (t3_0_z)
    float z2 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex2]);  // inner T3 anchor hit 2 z (t3_2_z)
    float z3 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex3]);  // inner T3 anchor hit 3 z (t3_4_z)
    float z4 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex4]);  // outer T3 anchor hit 4 z (t3_2_z)

    float r1 = mdsInGPU.anchorRt[mdIndex1];  // inner T3 anchor hit 1 r (t3_0_r)
    float r2 = mdsInGPU.anchorRt[mdIndex2];  // inner T3 anchor hit 2 r (t3_2_r)
    float r3 = mdsInGPU.anchorRt[mdIndex3];  // inner T3 anchor hit 3 r (t3_4_r)
    float r4 = mdsInGPU.anchorRt[mdIndex4];  // outer T3 anchor hit 4 r (t3_2_r)

    // Build the input feature vector using pairwise differences after the first hit
    float x[kinputFeatures] = {
        eta1 / kEta_norm,  // inner T3: First hit eta normalized
        alpaka::math::abs(acc, phi1) / kPhi_norm,  // inner T3: First hit phi normalized
        z1 / kZ_max,       // inner T3: First hit z normalized
        r1 / kR_max,       // inner T3: First hit r normalized

        eta2 - eta1,         // inner T3: Difference in eta between hit 2 and 1
        delta_phi(phi2, phi1) / kPhi_norm,         // inner T3: Difference in phi between hit 2 and 1
        (z2 - z1) / kZ_max,  // inner T3: Difference in z between hit 2 and 1 normalized
        (r2 - r1) / kR_max,  // inner T3: Difference in r between hit 2 and 1 normalized

        eta3 - eta2,         // inner T3: Difference in eta between hit 3 and 2
        delta_phi(phi3, phi2) / kPhi_norm,         // inner T3: Difference in phi between hit 3 and 2
        (z3 - z2) / kZ_max,  // inner T3: Difference in z between hit 3 and 2 normalized
        (r3 - r2) / kR_max,  // inner T3: Difference in r between hit 3 and 2 normalized

        eta4 - eta3,         // outer T3: Difference in eta between hit 4 and 3
        delta_phi(phi4, phi3) / kPhi_norm,         // inner T3: Difference in phi between hit 4 and 3
        (z4 - z3) / kZ_max,  // outer T3: Difference in z between hit 4 and 3 normalized
        (r4 - r3) / kR_max,  // outer T3: Difference in r between hit 4 and 3 normalized

        alpaka::math::log10(acc, innerRadius),   // T5 inner radius (t5_innerRadius)
        alpaka::math::log10(acc, outerRadius),    // T5 outer radius (t5_outerRadius)
        alpaka::math::log10(acc, innerRadius/outerRadius),    // radius ratio
        alpaka::math::log10(acc, regressionRadius), 
        alpaka::math::log10(acc, nonAnchorRegressionRadius),
        fakeScore1,
        promptScore1,
        displacedScore1,
        (fakeScore2- fakeScore1),
        (promptScore2 - promptScore1),
        (displacedScore2 - displacedScore1),
        // errors[0],
        // errors[1],
        // errors[2],
        // errors[3]
    };

    float x_1[khiddenFeatures];  // Layer 1 output
    float x_2[khiddenFeatures];  // Layer 2 output
    // float x_4[khiddenFeatures];
    // float x_5[khiddenFeatures];
    float x_3[koutputFeatures];  // Layer 3 output (3 classes) multi-class version
    // float x_3[1];                // Layer 3 linear output

    // Layer 1: Linear + Relu
    linear_layer<kinputFeatures, khiddenFeatures>(x, x_1, wgtT_layer1, bias_layer1);
    relu_activation<khiddenFeatures>(x_1);

    // Layer 2: Linear + Relu
    linear_layer<khiddenFeatures, khiddenFeatures>(x_1, x_2, wgtT_layer2, bias_layer2);
    relu_activation<khiddenFeatures>(x_2);

    // linear_layer<khiddenFeatures, khiddenFeatures>(x_2, x_4, wgtT_layer2, bias_layer2);
    // relu_activation<khiddenFeatures>(x_4);

    // linear_layer<khiddenFeatures, khiddenFeatures>(x_4, x_5, wgtT_layer2, bias_layer2);
    // relu_activation<khiddenFeatures>(x_5);

    // // Layer 3: Linear + Sigmoid
    // linear_layer<khiddenFeatures, 1>(x_2, x_3, wgtT_output_layer, bias_output_layer);
    // x_5 = sigmoid_activation(acc, x_3[0]);

    // Layer 3: Linear + Softmax multi-class version
    linear_layer<khiddenFeatures, koutputFeatures>(x_2, x_3, wgtT_output_layer, bias_output_layer);
    // linear_layer<khiddenFeatures, koutputFeatures>(x_5, x_3, wgtT_output_layer, bias_output_layer);
    softmax_activation<koutputFeatures>(acc, x_3);

    // Get the bin index based on abs(eta) of first hit and t4_pt
    // float t4_pt = innerRadius * lst::k2Rinv1GeVf * 2;
    float t4_pt = (innerRadius + outerRadius) * lst::k2Rinv1GeVf; //t4 pt is average

    uint8_t pt_index = (t4_pt > 5);
    uint8_t bin_index = (eta1 > 2.5f) ? (kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.25f);
    // uint8_t bin_index = (eta1 > 1.0f) ? (kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.1f);

    // Compare x_5 to the cut value for the relevant bin
    // return x_5 > kWp[pt_index][bin_index];
    promptScore = x_3[1];
    displacedScore = x_3[2];
    fakeScore = x_3[0];

    // if (x_3[1] > kWp_prompt_tight[pt_index][bin_index] || x_3[2] > kWp_displaced[pt_index][bin_index])
    //   TightPromptFlag = true;
    // if (x_3[1] > kWp_prompt[pt_index][bin_index] || x_3[2] > kWp_displaced_tight[pt_index][bin_index]) 
      // TightDisplacedFlag = true;
    // if (x_3[1] < 0.06 and x_3[2] > 0.34 and x_3[2] < 0.78) {
    //   TightPromptFlag = false;
    // } else {
    //   TightPromptFlag = true;
    // }
    TightDisplacedFlag = false;
    if (layer1 == 1){ //barrel 1
      // if ((x_3[2] < 0.344f or x_3[2] > 0.627f) and x_3[0]<0.111f){ //95% true disp retention
      // if ((x_3[2] < 0.377f or x_3[2] > 0.568f) and x_3[0]<0.127f){ //95% true disp retention, add uncert to dnn
      // if ((x_3[2] < 0.364f or x_3[2] > 0.697f) and x_3[0]<0.077f){ //90% true disp retention, add uncert to dnn
      //   TightDisplacedFlag = true;
      if (layer2==2) { 
        if (layer3 == 3) {
          if (layer4 == 4) {//reg 6 80% cut
            // if ((x_3[2] < 0.358f or x_3[2] > 0.839f) and x_3[0]<0.046f)
            // if ((x_3[2] < 0.401f or x_3[2] > 0.728f) and x_3[0]<0.108f) //85/90% add radii
            // if ((x_3[2] < 0.436f or x_3[2] > 0.730f) and x_3[0]<0.111f) //90% add radii and t3 score
            if ((x_3[2] < 0.436f or x_3[2] > 0.730f) and x_3[0]<0.077f) //90/85% add radii and t3 score
              TightDisplacedFlag = true;
          }
          else if (layer4 == 7) { //reg 7 95%
            // if ((x_3[2] < 0.983f or x_3[2] > 0.99f) and x_3[0]<0.069f)
            // if ((x_3[2] < 0.257f or x_3[2] > 0.303f) and x_3[0]<0.085f) //95% add radii
            if ((x_3[2] < 0.374f or x_3[2] > 0.741) and x_3[0]<0.076f) //90/85% add radii and t3 score
              TightDisplacedFlag = true;
          }
          else if (layer4 == 13) { //reg 8 80%
            // if ((x_3[2] < 0.365f or x_3[2] > 0.841f) and x_3[0]<0.085f)
            // if ((x_3[2] < 0.482f or x_3[2] > 0.662f) and x_3[0]<0.180f) //90% add radii
            // if ((x_3[2] < 0.289f or x_3[2] > 0.710f) and x_3[0]<0.150f) //90% add radii and t3 score
            if ((x_3[2] < 0.289f or x_3[2] > 0.710f) and x_3[0]<0.103f) //90/85% add radii and t3 score
              TightDisplacedFlag = true;
          }
        }
        else if (layer3==7) {
          if (layer4 == 8) { //reg 9 95%
            // if ((x_3[2] < 0.429f or x_3[2] > 0.701f) and x_3[0]<0.050f)
            // if ((x_3[2] < 0.469f or x_3[2] > 0.721f) and x_3[0]<0.065f) //95% add radii
            if ((x_3[2] < 0.883f or x_3[2] > 0.924f) and x_3[0]<0.049f) //90% add radii and t3 score
                TightDisplacedFlag = true;
          }
          // else if (layer4 == 13) { //reg 10 90%
          //   if ((x_3[2] < 0.358f or x_3[2] > 0.685f) and x_3[0]<0.070f)
          //       TightDisplacedFlag = true;
          // }
        }
      } else if (layer2 == 7) {
        if (layer3 == 8) {
          if (layer4 == 9) { //reg 11 95%
            // if ((x_3[2] < 0.220f or x_3[2] > 0.553f) and x_3[0]<0.087f)
            // if ((x_3[2] < 0.362f or x_3[2] > 0.708f) and x_3[0]<0.047f) //95% add radii
            if ((x_3[2] < 0.308f or x_3[2] > 0.780f) and x_3[0]<0.025f) //90% add radii and t3 score
                TightDisplacedFlag = true;
          }
        }
      }
    } else if (layer1 == 2) { //barrel 2
      // if ((x_3[2] < 0.355f or x_3[2] > 0.673f) and x_3[0]<0.125f) { //90% true retention (tighter since most number of fakes)
      // if ((x_3[2] < 0.352f or x_3[2] > 0.577f) and x_3[0]<0.247f){ //90% true disp retention, add uncert to dnn
      // if ((x_3[2] < 0.371f or x_3[2] > 0.644f) and x_3[0]<0.198f){ //87% true disp retention, (85% fake score cut) add uncert to dnn
      //   TightDisplacedFlag = true;
      // }
      if (layer2 == 3) {
        if (layer3 == 4) {
          if (layer4 == 5) { //reg 13 75/80%
            // if ((x_3[2] < 0.358f or x_3[2] > 0.819f) and x_3[0]<0.090f)
            // if ((x_3[2] < 0.447f or x_3[2] > 0.779f) and x_3[0]<0.193f) //85/90% add radii
            // if ((x_3[2] < 0.306f or x_3[2] > 0.697f) and x_3[0]<0.221f) //90% add radii and t3 score
            if ((x_3[2] < 0.306f or x_3[2] > 0.697f) and x_3[0]<0.166f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
          } else if (layer4 == 12) { //reg 14 75/80%
            // if ((x_3[2] < 0.527f or x_3[2] > 0.742f) and x_3[0]<0.320f)
            // if ((x_3[2] < 0.446f or x_3[2] > 0.663f) and x_3[0]<0.346f) //90% add radii
            // if ((x_3[2] < 0.211f or x_3[2] > 0.532f) and x_3[0]<0.302f) //90% add radii and t3 score
            if ((x_3[2] < 0.211f or x_3[2] > 0.532f) and x_3[0]<0.233f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
          }
        } else if (layer3 == 7) {
          if (layer4 == 13) { //reg16 95%
            // if ((x_3[2] < 0.815f or x_3[2] > 0.895f) and x_3[0]<0.174f)
            // if ((x_3[2] < 0.686f or x_3[2] > 0.733f) and x_3[0]<0.265f) //95% add radii
            if ((x_3[2] < 0.842f or x_3[2] > 0.909f) and x_3[0]<0.057f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
          }
        } else if (layer3 == 12) { //reg 17 75/80%
          // if ((x_3[2] < 0.329f or x_3[2] > 0.668f) and x_3[0]<0.227f)
          // if ((x_3[2] < 0.375f or x_3[2] > 0.54f) and x_3[0]<0.432f) //90% add radii
          // if ((x_3[2] < 0.316f or x_3[2] > 0.476f) and x_3[0]<0.359f) //90% add radii and t3 score
          if ((x_3[2] < 0.316f or x_3[2] > 0.476f) and x_3[0]<0.280f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
        }
      } else if (layer2 == 7) {
        if (layer3 == 8) { //reg 19 95
          // if ((x_3[2] < 0.301f or x_3[2] > 0.619f) and x_3[0]<0.159f)
          // if ((x_3[2] < 0.648f or x_3[2] > 0.748f) and x_3[0]<0.141f) //95% add radii
          if ((x_3[2] < 0.582f or x_3[2] > 0.852f) and x_3[0]<0.044f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
        } else if (layer3 == 13) { //reg 20 75/80%
          // if ((x_3[2] < 0.255f or x_3[2] > 0.627f) and x_3[0]<0.246f)
          // if ((x_3[2] < 0.344f or x_3[2] > 0.579f) and x_3[0]<0.303f) //90% add radii
          // if ((x_3[2] < 0.229f or x_3[2] > 0.496f) and x_3[0]<0.438f) //90% add radii and t3 score
          if ((x_3[2] < 0.229f or x_3[2] > 0.496f) and x_3[0]<0.261f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
        }
      }
    } else if (layer1 == 3) { //barrel 3
      // if ((x_3[2] < 0.434f or x_3[2] > 0.550f) and x_3[0]<0.306f) { //95% true retention
      // if ((x_3[2] < 0.359f or x_3[2] > 0.439f) and x_3[0]<0.498f){ //95% true disp retention, add uncert to dnn
      // if ((x_3[2] < 0.360f or x_3[2] > 0.501f) and x_3[0]<0.428f){ //90% true disp retention, add uncert to dnn
      //   TightDisplacedFlag = true;
      // }
      if (layer2 == 4) {
        if (layer3 == 5) {
          if (layer4 == 6) { //reg 21 75/80%
            // if ((x_3[2] < 0.250f or x_3[2] > 0.696f) and x_3[0]<0.237f)
            // if ((x_3[2] < 0.324f or x_3[2] > 0.642f) and x_3[0]<0.371f) //85/90% add radii
            // if ((x_3[2] < 0.418f or x_3[2] > 0.613f) and x_3[0]<0.344f) //90% add radii and t3 score
            if ((x_3[2] < 0.418f or x_3[2] > 0.613f) and x_3[0]<0.276f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
          } else if (layer4 == 12) { //reg 25 75/80%
            // if ((x_3[2] < 0.578f or x_3[2] > 0.773f) and x_3[0]<0.077f)
            // if ((x_3[2] < 0.691f or x_3[2] > 0.793f) and x_3[0]<0.510f) //90% add radii
            // if ((x_3[2] < 0.229f or x_3[2] > 0.482f) and x_3[0]<0.463f) //90% add radii and t3 score
            if ((x_3[2] < 0.229f or x_3[2] > 0.482f) and x_3[0]<0.376f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
          }
        } else if (layer3 == 12) { //reg 24 75/80%
          // if ((x_3[2] < 0.280f or x_3[2] > 0.502f) and x_3[0]<0.475f)
          // if ((x_3[2] < 0.569f or x_3[2] > 0.692f) and x_3[0]<0.529f) //85/90% add radii
          // if ((x_3[2] < 0.293f or x_3[2] > 0.492f) and x_3[0]<0.44f) //90% add radii and t3 score
          if ((x_3[2] < 0.293f or x_3[2] > 0.492f) and x_3[0]<0.376f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
        }
      } else if (layer2 == 7) { //reg 23 95%
        // if ((x_3[2] < 0.624f or x_3[2] > 0.671f) and x_3[0]<0.459f)
        // if ((x_3[2] < 0.487f or x_3[2] > 0.585f) and x_3[0]<0.467f) //95% add radii
        // if ((x_3[2] < 0.390f or x_3[2] > 0.648f) and x_3[0]<0.230f) //90% add radii and t3 score
        if ((x_3[2] < 0.390f or x_3[2] > 0.648f) and x_3[0]<0.159f) //90/85% add radii and t3 score
                TightDisplacedFlag = true;
      }
    } else if (layer1 == 7) { //endcap 1
      // if ((x_3[2] < 0.352f or x_3[2] > 0.609f) and x_3[0]<0.125f) { //95% true retention
      // if ((x_3[2] < 0.626f or x_3[2] > 0.718f) and x_3[0]<0.224f){ //95% true disp retention, add uncert to dnn
      // if ((x_3[2] < 0.626f or x_3[2] > 0.718f) and x_3[0]<0.132f){ //95% true disp retention, (90% fake cut) add uncert to dnn
        // TightDisplacedFlag = true;
      // }
      if (layer2 == 8) {
        if (layer3 == 9) {
          if (layer4 ==10) { //reg 0
            // if ((x_3[2] < 0.299f or x_3[2] > 0.783f) and x_3[0]<0.012f) //90% add radii
            if ((x_3[2] < 0.405f or x_3[2] > 0.871f) and x_3[0]<0.012f) //90% add radii and t3 score
              TightDisplacedFlag = true;
          } else if (layer4 == 15) { //reg 1
            // if ((x_3[2] < 0.463f or x_3[2] > 0.710f) and x_3[0]<0.049f) //90% add radii
            if ((x_3[2] < 0.438f or x_3[2] > 0.825f) and x_3[0]<0.045f) //90% add radii and t3 score
              TightDisplacedFlag = true;
          }
        } else if (layer3 == 14) { //reg 2
          // if ((x_3[2] < 0.319f or x_3[2] > 0.635f) and x_3[0]<0.186f) //90% add radii
          if ((x_3[2] < 0.319f or x_3[2] > 0.635f) and x_3[0]<0.134f) //90% add radii and t3 score
            TightDisplacedFlag = true;
        }
      }
    } else { //endcap 2
      // if ((x_3[2] < 0.358f or x_3[2] > 0.576f) and x_3[0]<0.144f) { //95% true retention
      // if ((x_3[2] < 0.443f or x_3[2] > 0.583f) and x_3[0]<0.220f){ //95% true disp retention, add uncert to dnn
      // if ((x_3[2] < 0.443f or x_3[2] > 0.583f) and x_3[0]<0.124f){ //95% true disp retention, (90% fake cut) add uncert to dnn
      //   TightDisplacedFlag = true;
      // }
      if (layer2 == 9) {
        if (layer3 == 10) {
          if (layer4 == 11) { //reg 3
            // if ((x_3[2] < 0.846f or x_3[2] > 0.934f) and x_3[0]<0.018f) //90% add radii
            if ((x_3[2] < 0.793f or x_3[2] > 0.866f) and x_3[0]<0.021f) //90% add radii and t3 score
              TightDisplacedFlag = true;
          } else if (layer4 == 16) { //reg 4
            // if ((x_3[2] < 0.517f or x_3[2] > 0.801f) and x_3[0]<0.069f) //90% add radii
            if ((x_3[2] < 0.281f or x_3[2] > 0.805f) and x_3[0]<0.060f) //90/95% add radii and t3 score
              TightDisplacedFlag = true;
          }
        } else { //reg 5
          // if ((x_3[2] < 0.350f or x_3[2] > 0.651f) and x_3[0]<0.247f) //90% add radii
          // if ((x_3[2] < 0.431f or x_3[2] > 0.75f) and x_3[0]<0.205f) //90% add radii and t3 score
          if ((x_3[2] < 0.431f or x_3[2] > 0.75f) and x_3[0]<0.168f) //90/85% add radii and t3 score
            TightDisplacedFlag = true;
        }
      }
    }

    return x_3[1] > kWp_prompt[pt_index][bin_index] || x_3[2] > kWp_displaced[pt_index][bin_index];
  } 

}  //namespace lst::t4dnn
//#endif
namespace lst::t3dnn {
  template <int FEATURES, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void softmax_activation(TAcc const& acc, float (&input)[FEATURES]) {
    float sum = 0.f;
    // Compute exp and sum
#pragma unroll FEATURES
    for (unsigned int i = 0; i < FEATURES; ++i) {
      input[i] = alpaka::math::exp(acc, input[i]);
      sum += input[i];
    }

    // Normalize
#pragma unroll FEATURES
    for (unsigned int i = 0; i < FEATURES; ++i) {
      input[i] /= sum;
    }
  }

  template <int FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void relu_activation(float (&input)[FEATURES]) {
#pragma unroll FEATURES
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
#pragma unroll OUT_FEATURES
    for (unsigned int i = 0; i < OUT_FEATURES; ++i) {
      output[i] = biases[i];
#pragma unroll IN_FEATURES
      for (int j = 0; j < IN_FEATURES; ++j) {
        output[i] += input[j] * weights[j][i];
      }
    }
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float delta_phi(const float phi1, const float phi2) {
    float delta = phi1 - phi2;
    // Adjust delta to be within the range [-M_PI, M_PI]
    if (delta > kPi) {
      delta -= 2 * kPi;
    } else if (delta < -kPi) {
      delta += 2 * kPi;
    }
    return delta;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                   lst::MiniDoublets const& mdsInGPU,
                                                   const unsigned int mdIndex1,
                                                   const unsigned int mdIndex2,
                                                   const unsigned int mdIndex3,
                                                   const float radius,
                                                   const float betaIn,
                                                   float (&output) [3]) {
    // Constants for T3 DNN
    constexpr unsigned int kinputFeatures = 14;
    constexpr unsigned int khiddenFeatures = 32;
    constexpr unsigned int koutputFeatures = 3;
    // Extract hit information
    float eta1 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex1]);  // inner T3 anchor hit 1 eta
    float eta2 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex2]);  // inner T3 anchor hit 2 eta
    float eta3 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex3]);  // inner T3 anchor hit 3 eta
    float phi1 = mdsInGPU.anchorPhi[mdIndex1];  // inner T3 anchor hit 1 phi
    float phi2 = mdsInGPU.anchorPhi[mdIndex2];  // inner T3 anchor hit 2 phi
    float phi3 = mdsInGPU.anchorPhi[mdIndex3];  // inner T3 anchor hit 3 phi
    float z1 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex1]);  // inner T3 anchor hit 1 z
    float z2 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex2]);  // inner T3 anchor hit 2 z
    float z3 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex3]);  // inner T3 anchor hit 3 z
    float r1 = mdsInGPU.anchorRt[mdIndex1];  // inner T3 anchor hit 1 r
    float r2 = mdsInGPU.anchorRt[mdIndex2];  // inner T3 anchor hit 2 r
    float r3 = mdsInGPU.anchorRt[mdIndex3];  // inner T3 anchor hit 3 r
    // Build input feature vector matching training order
    float x[kinputFeatures] = {
        eta1 / kEta_norm,                          // First hit eta normalized
        alpaka::math::abs(acc, phi1) / kPhi_norm,  // First hit phi normalized
        z1 / kZ_max,                               // First hit z normalized
        r1 / kR_max,                               // First hit r normalized
        eta2 - eta1,                             // Difference in eta between hit 2 and 1
        delta_phi(phi2, phi1) / kPhi_norm,  // Difference in phi between hit 2 and 1
        (z2 - z1) / kZ_max,                      // Difference in z between hit 2 and 1 normalized
        (r2 - r1) / kR_max,                      // Difference in r between hit 2 and 1 normalized
        eta3 - eta2,                             // Difference in eta between hit 3 and 2
        delta_phi(phi3, phi2) / kPhi_norm,  // Difference in phi between hit 3 and 2
        (z3 - z2) / kZ_max,                      // Difference in z between hit 3 and 2 normalized
        (r3 - r2) / kR_max,                      // Difference in r between hit 3 and 2 normalized
        alpaka::math::log10(acc, radius),  // T3's circle radius
        betaIn                             // Beta angle of inner segment
    };
    float x_1[khiddenFeatures];  // Layer 1 output
    float x_2[khiddenFeatures];  // Layer 2 output
    // Layer 1: Linear + Relu
    linear_layer<kinputFeatures, khiddenFeatures>(x, x_1, t3dnn::wgtT_layer1, t3dnn::bias_layer1);
    relu_activation<khiddenFeatures>(x_1);
    // Layer 2: Linear + Relu
    linear_layer<khiddenFeatures, khiddenFeatures>(x_1, x_2, t3dnn::wgtT_layer2, t3dnn::bias_layer2);
    relu_activation<khiddenFeatures>(x_2);
    // Layer 3: Linear + Softmax
    linear_layer<khiddenFeatures, koutputFeatures>(x_2, output, t3dnn::wgtT_output_layer, t3dnn::bias_output_layer);
    softmax_activation<koutputFeatures>(acc, output);
    // Get pt and eta bin indices
    float t3_pt = radius * lst::k2Rinv1GeVf * 2;
    uint8_t pt_index = (t3_pt > 5);
    uint8_t bin_index = (eta1 > 2.5f) ? (kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.25f);

    return output[1] > kWp_prompt[pt_index][bin_index] || 
           output[2] > kWp_displaced[pt_index][bin_index];
  }
}  // namespace lst::t3dnn
//#ifdef USE_T4_PT4
namespace lst::pt4dnn {
  template <int FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void relu_activation(float (&input)[FEATURES]) {
#pragma unroll FEATURES
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
#pragma unroll OUT_FEATURES
    for (unsigned int i = 0; i < OUT_FEATURES; ++i) {
      output[i] = biases[i];
#pragma unroll IN_FEATURES
      for (int j = 0; j < IN_FEATURES; ++j) {
        output[i] += input[j] * weights[j][i];
      }
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                   const float t4InnerRadius,
                                                   const float pLS_pt,
                                                   const float pt4_rPhiChiSquared,
                                                   const float pt4_quad_rad,
                                                   const float pt4_pix_rad,
                                                   const float pt4_pixRadError,
                                                   const float pt4_rzChiSquared,
                                                   const float pt4_eta) {
    constexpr unsigned int kinputFeatures = 6;
    float x[kinputFeatures] = {alpaka::math::log10(acc, pt4_rPhiChiSquared),
                               alpaka::math::log10(acc, pt4_quad_rad),
                               alpaka::math::log10(acc, pt4_pix_rad),
                               alpaka::math::log10(acc, pt4_pixRadError),
                               alpaka::math::log10(acc, (pt4_rzChiSquared < 0.f) ? 1e-3f : pt4_rzChiSquared),
                               alpaka::math::abs(acc, pt4_eta) / kEta_norm};

    constexpr unsigned int khiddenFeatures = 32;
    constexpr unsigned int koutputFeatures = 1;
    float x1[khiddenFeatures];
    float x2[khiddenFeatures];
    float x3[koutputFeatures];

    linear_layer<kinputFeatures, khiddenFeatures>(x, x1, wgtT_layer1, bias_layer1);
    relu_activation<khiddenFeatures>(x1);

    linear_layer<khiddenFeatures, khiddenFeatures>(x1, x2, wgtT_layer2, bias_layer2);
    relu_activation<khiddenFeatures>(x2);

    linear_layer<khiddenFeatures, koutputFeatures>(
        x2, x3, wgtT_output_layer, bias_output_layer);
    float output = sigmoid_activation(acc, x3[0]);

    uint8_t bin_index = (alpaka::math::abs(acc, pt4_eta) > kEta_norm)
                            ? (kEtaBins - 1)
                            : static_cast<unsigned int>(alpaka::math::abs(acc, pt4_eta) / 0.25f);

    return output > kWp[bin_index];
    // float pt4_pt =(t4InnerRadius*lst::k2Rinv1GeVf * 2 + pLS_pt)/2;
    // uint8_t pt_index = (pt4_pt > 5);
    // return output > kWp[pt_index][bin_index];
  }

}  // namespace pt4dnn
//#endif

namespace lst::pt3dnn {
  template <int FEATURES>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void relu_activation(float (&input)[FEATURES]) {
#pragma unroll FEATURES
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
#pragma unroll OUT_FEATURES
    for (unsigned int i = 0; i < OUT_FEATURES; ++i) {
      output[i] = biases[i];
#pragma unroll IN_FEATURES
      for (int j = 0; j < IN_FEATURES; ++j) {
        output[i] += input[j] * weights[j][i];
      }
    }
  }

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
                                 alpaka::math::abs(acc, pixelEta) / lst::t3dnn::kEta_norm};

      float x1[kHiddenFeatures];
      float x2[kHiddenFeatures];
      float x3[kOutputFeatures];

      linear_layer<kInputFeatures, kHiddenFeatures>(x, x1, lst::pt3dnn::wgtT_layer1, lst::pt3dnn::bias_layer1);
      relu_activation<kHiddenFeatures>(x1);

      linear_layer<kHiddenFeatures, kHiddenFeatures>(x1, x2, lst::pt3dnn::wgtT_layer2, lst::pt3dnn::bias_layer2);
      relu_activation<kHiddenFeatures>(x2);

      linear_layer<kHiddenFeatures, kOutputFeatures>(
          x2, x3, lst::pt3dnn::wgtT_output_layer, lst::pt3dnn::bias_output_layer);
      float output = sigmoid_activation(acc, x3[0]);

      uint8_t bin_index = (alpaka::math::abs(acc, pixelEta) > 2.5f)
                              ? (lst::pt3dnn::kEtaBins - 1)
                              : static_cast<unsigned int>(alpaka::math::abs(acc, pixelEta) / lst::pt3dnn::kEtaSize);

      if (pixelPt > 5.0f)
        return output > lst::pt3dnn::kWpHigh;

      return output > lst::pt3dnn::kWp[bin_index];
    }

  }  // namespace pt3dnn

#endif
