#ifndef RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h
#define RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"

#include "NeuralNetworkWeights.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "T3NeuralNetworkWeights.h"

namespace lst::t5dnn {

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
    if (delta > M_PI) {
      delta -= 2 * M_PI;
    } else if (delta < -M_PI) {
      delta += 2 * M_PI;
    }

    return delta;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runInference(TAcc const& acc,
                                                   lst::MiniDoublets const& mdsInGPU,
                                                   const unsigned int mdIndex1,
                                                   const unsigned int mdIndex2,
                                                   const unsigned int mdIndex3,
                                                   const unsigned int mdIndex4,
                                                   const unsigned int mdIndex5,
                                                   const float innerRadius,
                                                   const float outerRadius,
                                                   const float bridgeRadius) {
    // Constants
    constexpr unsigned int kinputFeatures = 23;
    constexpr unsigned int khiddenFeatures = 32;

    float eta1 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex1]);  // inner T3 anchor hit 1 eta (t3_0_eta)
    float eta2 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex2]);  // inner T3 anchor hit 2 eta (t3_2_eta)
    float eta3 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex3]);  // inner T3 anchor hit 3 eta (t3_4_eta)
    float eta4 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex4]);  // outer T3 anchor hit 4 eta (t3_2_eta)
    float eta5 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex5]);  // outer T3 anchor hit 5 eta (t3_4_eta)

    float phi1 = mdsInGPU.anchorPhi[mdIndex1];  // inner T3 anchor hit 1 phi (t3_0_phi)
    float phi2 = mdsInGPU.anchorPhi[mdIndex2];  // inner T3 anchor hit 2 phi (t3_2_phi)
    float phi3 = mdsInGPU.anchorPhi[mdIndex3];  // inner T3 anchor hit 3 phi (t3_4_phi)
    float phi4 = mdsInGPU.anchorPhi[mdIndex4];  // outer T3 anchor hit 4 phi (t3_2_phi)
    float phi5 = mdsInGPU.anchorPhi[mdIndex5];  // outer T3 anchor hit 5 phi (t3_4_phi)

    float z1 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex1]);  // inner T3 anchor hit 1 z (t3_0_z)
    float z2 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex2]);  // inner T3 anchor hit 2 z (t3_2_z)
    float z3 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex3]);  // inner T3 anchor hit 3 z (t3_4_z)
    float z4 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex4]);  // outer T3 anchor hit 4 z (t3_2_z)
    float z5 = alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex5]);  // outer T3 anchor hit 5 z (t3_4_z)

    float r1 = mdsInGPU.anchorRt[mdIndex1];  // inner T3 anchor hit 1 r (t3_0_r)
    float r2 = mdsInGPU.anchorRt[mdIndex2];  // inner T3 anchor hit 2 r (t3_2_r)
    float r3 = mdsInGPU.anchorRt[mdIndex3];  // inner T3 anchor hit 3 r (t3_4_r)
    float r4 = mdsInGPU.anchorRt[mdIndex4];  // outer T3 anchor hit 4 r (t3_2_r)
    float r5 = mdsInGPU.anchorRt[mdIndex5];  // outer T3 anchor hit 5 r (t3_4_r)

    // Build the input feature vector using pairwise differences after the first hit
    float x[kinputFeatures] = {
        eta1 / kEta_norm,                          // inner T3: First hit eta normalized
        alpaka::math::abs(acc, phi1) / kPhi_norm,  // inner T3: First hit phi normalized
        z1 / kZ_max,                               // inner T3: First hit z normalized
        r1 / kR_max,                               // inner T3: First hit r normalized

        eta2 - eta1,                        // inner T3: Difference in eta between hit 2 and 1
        delta_phi(phi2, phi1) / kPhi_norm,  // inner T3: Difference in phi between hit 2 and 1
        (z2 - z1) / kZ_max,                 // inner T3: Difference in z between hit 2 and 1 normalized
        (r2 - r1) / kR_max,                 // inner T3: Difference in r between hit 2 and 1 normalized

        eta3 - eta2,                        // inner T3: Difference in eta between hit 3 and 2
        delta_phi(phi3, phi2) / kPhi_norm,  // inner T3: Difference in phi between hit 3 and 2
        (z3 - z2) / kZ_max,                 // inner T3: Difference in z between hit 3 and 2 normalized
        (r3 - r2) / kR_max,                 // inner T3: Difference in r between hit 3 and 2 normalized

        eta4 - eta3,                        // outer T3: Difference in eta between hit 4 and 3
        delta_phi(phi4, phi3) / kPhi_norm,  // inner T3: Difference in phi between hit 4 and 3
        (z4 - z3) / kZ_max,                 // outer T3: Difference in z between hit 4 and 3 normalized
        (r4 - r3) / kR_max,                 // outer T3: Difference in r between hit 4 and 3 normalized

        eta5 - eta4,                        // outer T3: Difference in eta between hit 5 and 4
        delta_phi(phi5, phi4) / kPhi_norm,  // inner T3: Difference in phi between hit 5 and 4
        (z5 - z4) / kZ_max,                 // outer T3: Difference in z between hit 5 and 4 normalized
        (r5 - r4) / kR_max,                 // outer T3: Difference in r between hit 5 and 4 normalized

        alpaka::math::log10(acc, innerRadius),   // T5 inner radius (t5_innerRadius)
        alpaka::math::log10(acc, bridgeRadius),  // T5 bridge radius (t5_bridgeRadius)
        alpaka::math::log10(acc, outerRadius)    // T5 outer radius (t5_outerRadius)
    };

    float x_1[khiddenFeatures];  // Layer 1 output
    float x_2[khiddenFeatures];  // Layer 2 output
    float x_3[1];                // Layer 3 linear output

    // Layer 1: Linear + Relu
    linear_layer<kinputFeatures, khiddenFeatures>(x, x_1, wgtT_layer1, bias_layer1);
    relu_activation<khiddenFeatures>(x_1);

    // Layer 2: Linear + Relu
    linear_layer<khiddenFeatures, khiddenFeatures>(x_1, x_2, wgtT_layer2, bias_layer2);
    relu_activation<khiddenFeatures>(x_2);

    // Layer 3: Linear + Sigmoid
    linear_layer<khiddenFeatures, 1>(x_2, x_3, wgtT_output_layer, bias_output_layer);
    float x_5 = sigmoid_activation(acc, x_3[0]);

    // Get the bin index based on abs(eta) of first hit and t5_pt
    float t5_pt = innerRadius * lst::k2Rinv1GeVf * 2;

    uint8_t pt_index = (t5_pt > 5);
    uint8_t bin_index = (eta1 > 2.5f) ? (kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.25f);

    // Compare x_5 to the cut value for the relevant bin
    return x_5 > kWp[pt_index][bin_index];
  }

}  //namespace lst::t5dnn

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
                                                   const unsigned int mdIndex1,
                                                   const unsigned int mdIndex2,
                                                   const unsigned int mdIndex3,
                                                   const unsigned int mdIndex4,
                                                   const float innerRadius,
                                                   const float outerRadius) {
    // Constants
    constexpr unsigned int kinputFeatures = 19; 
    constexpr unsigned int khiddenFeatures = 32;
    constexpr unsigned int koutputFeatures = 3;

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
        alpaka::math::log10(acc, innerRadius/outerRadius)    // radius ratio
    };

    float x_1[khiddenFeatures];  // Layer 1 output
    float x_2[khiddenFeatures];  // Layer 2 output
    float x_3[koutputFeatures];  // Layer 3 output (3 classes) multi-class version
    // float x_3[1];                // Layer 3 linear output

    // Layer 1: Linear + Relu
    linear_layer<kinputFeatures, khiddenFeatures>(x, x_1, wgtT_layer1, bias_layer1);
    relu_activation<khiddenFeatures>(x_1);

    // Layer 2: Linear + Relu
    linear_layer<khiddenFeatures, khiddenFeatures>(x_1, x_2, wgtT_layer2, bias_layer2);
    relu_activation<khiddenFeatures>(x_2);

    // // Layer 3: Linear + Sigmoid
    // linear_layer<khiddenFeatures, 1>(x_2, x_3, wgtT_output_layer, bias_output_layer);
    // x_5 = sigmoid_activation(acc, x_3[0]);

    // Layer 3: Linear + Softmax multi-class version
    linear_layer<khiddenFeatures, koutputFeatures>(x_2, x_3, wgtT_output_layer, bias_output_layer);
    softmax_activation<koutputFeatures>(acc, x_3);

    // Get the bin index based on abs(eta) of first hit and t4_pt
    // float t4_pt = innerRadius * lst::k2Rinv1GeVf * 2;
    float t4_pt = (innerRadius + outerRadius) * lst::k2Rinv1GeVf; //t4 pt is average

    uint8_t pt_index = (t4_pt > 5);
    uint8_t bin_index = (eta1 > 2.5f) ? (kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.25f);
    // uint8_t bin_index = (eta1 > 1.0f) ? (kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.1f);

    // Compare x_5 to the cut value for the relevant bin
    // return x_5 > kWp[pt_index][bin_index];
    return x_3[1] > kWp_prompt[pt_index][bin_index] || x_3[2] > kWp_displaced[pt_index][bin_index];
  }

}  //namespace lst::t4dnn

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
                                                   const float betaIn) {
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
    float x_3[koutputFeatures];  // Layer 3 output (3 classes)
    // Layer 1: Linear + Relu
    linear_layer<kinputFeatures, khiddenFeatures>(x, x_1, t3dnn::wgtT_layer1, t3dnn::bias_layer1);
    relu_activation<khiddenFeatures>(x_1);
    // Layer 2: Linear + Relu
    linear_layer<khiddenFeatures, khiddenFeatures>(x_1, x_2, t3dnn::wgtT_layer2, t3dnn::bias_layer2);
    relu_activation<khiddenFeatures>(x_2);
    // Layer 3: Linear + Softmax
    linear_layer<khiddenFeatures, koutputFeatures>(x_2, x_3, t3dnn::wgtT_output_layer, t3dnn::bias_output_layer);
    softmax_activation<koutputFeatures>(acc, x_3);
    // Get pt and eta bin indices
    float t3_pt = radius * lst::k2Rinv1GeVf * 2;
    uint8_t pt_index = (t3_pt > 5);
    uint8_t bin_index = (eta1 > 2.5f) ? (kEtaBins - 1) : static_cast<unsigned int>(eta1 / 0.25f);
    return x_3[1] > kWp_prompt[pt_index][bin_index] || x_3[2] > kWp_displaced[pt_index][bin_index];
  }
}  // namespace lst::t3dnn

#endif
