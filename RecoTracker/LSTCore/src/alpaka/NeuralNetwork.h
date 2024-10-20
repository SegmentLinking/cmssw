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

    // Compute some convenience variables
    short layer2_adjustment = 0;
    if (modulesInGPU.layers[lowerModuleIndices[0]] == 1) {
      layer2_adjustment = 1;  // get upper segment to be in second layer
    }
    unsigned int md_idx_for_t5_eta_phi =
        segmentsInGPU.mdIndices[2 * tripletsInGPU.segmentIndices[2 * innerTripletIndex + layer2_adjustment]];

    float t5_eta = mdsInGPU.anchorEta[md_idx_for_t5_eta_phi];

    // Constants
    constexpr unsigned int kinputFeatures = 18;
    constexpr unsigned int khiddenFeatures = 32;

    // Build DNN input vector (corresponding output N-tuple branch noted in parenthetical in comment)
    float x[kinputFeatures] = {
        alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex1]) / kEta_norm,  // inner T3 anchor hit 1 eta (t3_0_eta)
        alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex1]) / kZ_max,       // inner T3 anchor hit 1 z (t3_0_z)
        alpaka::math::sqrt(acc, x1 * x1 + y1 * y1) / kR_max,               // inner T3 anchor hit 1 r (t3_0_r)
        alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex2]) / kEta_norm,  // inner T3 anchor hit 2 eta (t3_2_eta)
        alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex2]) / kZ_max,       // inner T3 anchor hit 2 z (t3_2_z)
        alpaka::math::sqrt(acc, x2 * x2 + y2 * y2) / kR_max,               // inner T3 anchor hit 2 r (t3_2_r)
        alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex3]) / kEta_norm,  // inner T3 anchor hit 3 eta (t3_4_eta)
        alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex3]) / kZ_max,       // inner T3 anchor hit 3 z (t3_4_z)
        alpaka::math::sqrt(acc, x3 * x3 + y3 * y3) / kR_max,               // inner T3 anchor hit 3 r (t3_4_r)
        alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex4]) / kEta_norm,  // outer T3 anchor hit 4 eta (t3_2_eta)
        alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex4]) / kZ_max,       // outer T3 anchor hit 4 z (t3_2_z)
        alpaka::math::sqrt(acc, x4 * x4 + y4 * y4) / kR_max,               // outer T3 anchor hit 4 r (t3_2_r)
        alpaka::math::abs(acc, mdsInGPU.anchorEta[mdIndex5]) / kEta_norm,  // outer T3 anchor hit 5 eta (t3_4_eta)
        alpaka::math::abs(acc, mdsInGPU.anchorZ[mdIndex5]) / kZ_max,       // outer T3 anchor hit 5 z (t3_4_z)
        alpaka::math::sqrt(acc, x5 * x5 + y5 * y5) / kR_max,               // outer T3 anchor hit 5 r (t3_4_r)

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

    // Get the bin index based on abs(t5_eta) and t5_pt
    float abs_t5_eta = alpaka::math::abs(acc, t5_eta);
    float t5_pt = innerRadius * lst::k2Rinv1GeVf * 2;

    uint8_t pt_index = (t5_pt > 5) ? 1 : 0;
    uint8_t bin_index = (abs_t5_eta > 2.5f) ? (kEtaBins - 1) : static_cast<unsigned int>(abs_t5_eta / 0.25f);

    // Compare x_5 to the cut value for the relevant bin
    return x_5 > kWp[pt_index][bin_index];
  }

}  //namespace lst::t5dnn

#endif
