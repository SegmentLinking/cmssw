#ifndef RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h
#define RecoTracker_LSTCore_src_alpaka_NeuralNetwork_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "NeuralNetworkWeights.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "Triplet.h"

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

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float runInference(TAcc const& acc,
                                                    lst::Modules const& modulesInGPU,
                                                    lst::MiniDoublets const& mdsInGPU,
                                                    lst::Segments const& segmentsInGPU,
                                                    lst::Triplets const& tripletsInGPU,
                                                    const unsigned int* mdIndices,
                                                    const uint16_t* lowerModuleIndices,
                                                    unsigned int innerTripletIndex,
                                                    unsigned int outerTripletIndex,
                                                    float innerRadius,
                                                    float outerRadius,
                                                    float bridgeRadius) {
    // Unpack module indices
    unsigned int mdIndex1 = mdIndices[0];
    unsigned int mdIndex2 = mdIndices[1];
    unsigned int mdIndex3 = mdIndices[2];
    unsigned int mdIndex4 = mdIndices[3];
    unsigned int mdIndex5 = mdIndices[4];

    // Constants
    constexpr unsigned int kinputFeatures = 18;
    constexpr unsigned int khiddenFeatures = 32;

    float eta1 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex1]);  // inner T3 anchor hit 1 eta (t3_0_eta)
    float eta2 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex2]);  // inner T3 anchor hit 2 eta (t3_2_eta)
    float eta3 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex3]);  // inner T3 anchor hit 3 eta (t3_4_eta)
    float eta4 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex4]);  // outer T3 anchor hit 4 eta (t3_2_eta)
    float eta5 = alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex5]);  // outer T3 anchor hit 5 eta (t3_4_eta)

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
        eta1 / kEta_norm,  // inner T3: First hit eta normalized
        z1 / kZ_max,       // inner T3: First hit z normalized
        r1 / kR_max,       // inner T3: First hit r normalized

        eta2 - eta1,         // inner T3: Difference in eta between hit 2 and 1
        (z2 - z1) / kZ_max,  // inner T3: Difference in z between hit 2 and 1 normalized
        (r2 - r1) / kR_max,  // inner T3: Difference in r between hit 2 and 1 normalized

        eta3 - eta2,         // inner T3: Difference in eta between hit 3 and 2
        (z3 - z2) / kZ_max,  // inner T3: Difference in z between hit 3 and 2 normalized
        (r3 - r2) / kR_max,  // inner T3: Difference in r between hit 3 and 2 normalized

        eta4 - eta3,         // outer T3: Difference in eta between hit 4 and 3
        (z4 - z3) / kZ_max,  // outer T3: Difference in z between hit 4 and 3 normalized
        (r4 - r3) / kR_max,  // outer T3: Difference in r between hit 4 and 3 normalized

        eta5 - eta4,         // outer T3: Difference in eta between hit 5 and 4
        (z5 - z4) / kZ_max,  // outer T3: Difference in z between hit 5 and 4 normalized
        (r5 - r4) / kR_max,  // outer T3: Difference in r between hit 5 and 4 normalized

        alpaka::math::log10(acc, innerRadius),   // T5 inner radius (t5_innerRadius)
        alpaka::math::log10(acc, bridgeRadius),  // T5 bridge radius (t5_bridgeRadius)
        alpaka::math::log10(acc, outerRadius)    // T5 outer radius (t5_outerRadius)
    };

    // Layer 1: Linear
    float x_1[khiddenFeatures];
    linear_layer<kinputFeatures, khiddenFeatures>(x, x_1, wgtT_0, bias_0);

    // Layer 1: ReLU
    relu_activation<khiddenFeatures>(x_1);

    // Layer 2: Linear
    float x_2[khiddenFeatures];
    linear_layer<khiddenFeatures, khiddenFeatures>(x_1, x_2, wgtT_2, bias_2);

    // Layer 2: ReLU
    relu_activation<khiddenFeatures>(x_2);

    // Layer 3: Linear
    float x_3[1];
    linear_layer<khiddenFeatures, 1>(x_2, x_3, wgtT_4, bias_4);

    // Layer 3: Sigmoid
    float x_5 = sigmoid_activation(acc, x_3[0]);

    // Compute some convenience variables for t5_eta
    short layer2_adjustment = 0;
    if (modulesInGPU.layers[lowerModuleIndices[0]] == 1) {
      layer2_adjustment = 1;  // get upper segment to be in second layer
    }
    unsigned int md_idx_for_t5_eta_phi =
        segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex + layer2_adjustment]];

    // Get the bin index based on abs(t5_eta) and t5_pt
    float t5_eta = alpaka::math::abs(acc, mdsInGPU.anchorEta[md_idx_for_t5_eta_phi]);
    float t5_pt = innerRadius * lst::k2Rinv1GeVf * 2;

    uint8_t pt_index = (t5_pt > 5) ? 1 : 0;
    uint8_t bin_index = (t5_eta > 2.5f) ? (kEtaBins - 1) : static_cast<unsigned int>(t5_eta / 0.25f);

    // Compare x_5 to the cut value for the relevant bin
    return x_5 > kWp[pt_index][bin_index];
  }

}  //namespace lst::t5dnn

#endif
