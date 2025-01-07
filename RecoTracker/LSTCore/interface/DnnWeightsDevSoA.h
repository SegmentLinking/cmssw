#ifndef RecoTracker_LSTCore_interface_DnnWeightsDevSoA_h
#define RecoTracker_LSTCore_interface_DnnWeightsDevSoA_h

#include "RecoTracker/LSTCore/interface/DenseLayer.h"

namespace lst {

  /**
   * Data structure holding multiple dense layers for the DNN weights.
   */
  struct DnnWeightsDevData {
    DenseLayer<23, 32> layer1;
    DenseLayer<32, 32> layer2;
    DenseLayer<32, 1> layer3;
  };

}  // namespace lst

#endif  // RecoTracker_LSTCore_interface_DnnWeightsDevSoA_h