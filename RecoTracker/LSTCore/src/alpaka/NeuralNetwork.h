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
    for (unsigned int i = 0; i < OUT_FEATURES; ++i) {
      output[i] = biases[i];
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
    if (modulesInGPU.layers[lowerModuleIndex1] == 1) {
      layer2_adjustment = 1;  // get upper segment to be in second layer
    }
    unsigned int md_idx_for_t5_eta_phi =
        segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex + layer2_adjustment]];
    bool is_endcap1 = (modulesInGPU.subdets[lowerModuleIndex1] == 4);  // true if anchor hit 1 is in the endcap
    bool is_endcap2 = (modulesInGPU.subdets[lowerModuleIndex2] == 4);  // true if anchor hit 2 is in the endcap
    bool is_endcap3 = (modulesInGPU.subdets[lowerModuleIndex3] == 4);  // true if anchor hit 3 is in the endcap
    bool is_endcap4 = (modulesInGPU.subdets[lowerModuleIndex4] == 4);  // true if anchor hit 4 is in the endcap
    bool is_endcap5 = (modulesInGPU.subdets[lowerModuleIndex5] == 4);  // true if anchor hit 5 is in the endcap

    float t5_eta = mdsInGPU.anchorEta[md_idx_for_t5_eta_phi];

    // Constants
    constexpr unsigned int kinputFeatures = 24;
    constexpr unsigned int khiddenFeatures = 32;

    // Build DNN input vector (corresponding output N-tuple branch noted in parenthetical in comment)
    float x[kinputFeatures] = {
        mdsInGPU.anchorEta[mdIndex1],                                    // inner T3 anchor hit 1 eta (t3_0_eta)
        mdsInGPU.anchorZ[mdIndex1] / t5dnn::Z_max,                       // inner T3 anchor hit 1 z (t3_0_z)
        alpaka::math::sqrt(acc, x1 * x1 + y1 * y1) / t5dnn::R_max,       // inner T3 anchor hit 1 r (t3_0_r)
        float(modulesInGPU.layers[lowerModuleIndex1] + 6 * is_endcap1),  // inner T3 anchor hit 1 layer (t3_0_layer)
        mdsInGPU.anchorEta[mdIndex2],                                    // inner T3 anchor hit 2 eta (t3_2_eta)
        mdsInGPU.anchorZ[mdIndex2] / t5dnn::Z_max,                       // inner T3 anchor hit 2 z (t3_2_z)
        alpaka::math::sqrt(acc, x2 * x2 + y2 * y2) / t5dnn::R_max,       // inner T3 anchor hit 2 r (t3_2_r)
        float(modulesInGPU.layers[lowerModuleIndex2] + 6 * is_endcap2),  // inner T3 anchor hit 2 layer (t3_2_layer)
        mdsInGPU.anchorEta[mdIndex3],                                    // inner T3 anchor hit 3 eta (t3_4_eta)
        mdsInGPU.anchorZ[mdIndex3] / t5dnn::Z_max,                       // inner T3 anchor hit 3 z (t3_4_z)
        alpaka::math::sqrt(acc, x3 * x3 + y3 * y3) / t5dnn::R_max,       // inner T3 anchor hit 3 r (t3_4_r)
        float(modulesInGPU.layers[lowerModuleIndex3] + 6 * is_endcap3),  // inner T3 anchor hit 3 layer (t3_4_layer)
        mdsInGPU.anchorEta[mdIndex4],                                    // outer T3 anchor hit 4 eta (t3_2_eta)
        mdsInGPU.anchorZ[mdIndex4] / t5dnn::Z_max,                       // outer T3 anchor hit 4 z (t3_2_z)
        alpaka::math::sqrt(acc, x4 * x4 + y4 * y4) / t5dnn::R_max,       // outer T3 anchor hit 4 r (t3_2_r)
        float(modulesInGPU.layers[lowerModuleIndex4] + 6 * is_endcap4),  // outer T3 anchor hit 4 layer (t3_2_layer)
        mdsInGPU.anchorEta[mdIndex5],                                    // outer T3 anchor hit 5 eta (t3_4_eta)
        mdsInGPU.anchorZ[mdIndex5] / t5dnn::Z_max,                       // outer T3 anchor hit 5 z (t3_4_z)
        alpaka::math::sqrt(acc, x5 * x5 + y5 * y5) / t5dnn::R_max,       // outer T3 anchor hit 5 r (t3_4_r)
        float(modulesInGPU.layers[lowerModuleIndex5] + 6 * is_endcap5),  // outer T3 anchor hit 5 layer (t3_4_layer)
        t5_eta,                                                          // T5 eta (t5_eta)
        alpaka::math::log10(acc, innerRadius),                           // T5 inner radius (t5_innerRadius)
        alpaka::math::log10(acc, bridgeRadius),                          // T5 bridge radius (t5_bridgeRadius)
        alpaka::math::log10(acc, outerRadius)                            // T5 outer radius (t5_outerRadius)
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

    // Get the bin index based on abs(t5_eta) and t5_pt
    float abs_t5_eta = alpaka::math::abs(acc, t5_eta);
    float t5_pt = innerRadius * lst::k2Rinv1GeVf * 2;

    uint8_t pt_index = (t5_pt > 5) ? 1 : 0;
    uint8_t bin_index = (abs_t5_eta > 2.5f) ? 9 : static_cast<unsigned int>(abs_t5_eta / 0.25f);

    // Compare x_5 to the cut value for the relevant bin
    return x_5 > kLSTWp[pt_index][bin_index];
  }

}  //namespace lst::t5dnn

#endif
