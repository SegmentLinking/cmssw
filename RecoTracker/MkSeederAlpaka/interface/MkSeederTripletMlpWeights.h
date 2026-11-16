#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederTripletMlpWeights_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederTripletMlpWeights_h

namespace mkseeder {

  // The weights of the triplet MLP (19 features, two ReLU hidden layers of 32 and 16 nodes, one output): the
  // features are standardised with featureMean and featureScale; the weights of a layer are row-major by input
  // (hidden1Weights[i * kNHidden1 + j] from input i to node j)
  struct MkSeederTripletMlpWeights {
    static constexpr int kNFeatures = 19, kNHidden1 = 32, kNHidden2 = 16;
    float featureMean[kNFeatures];
    float featureScale[kNFeatures];
    float hidden1Weights[kNFeatures * kNHidden1];
    float hidden1Bias[kNHidden1];
    float hidden2Weights[kNHidden1 * kNHidden2];
    float hidden2Bias[kNHidden2];
    float outputWeights[kNHidden2];
    float outputBias;
    // the z mirror fold (JSON zFold, optional): +1 / -1 evaluates every triplet as its mirror image on the cot theta >= 0
    // / < 0 side (the signed features: the hits' z, z0, the r-z residual and cot theta times +-sign(cot theta)); 0 none
    float zFold = 0;
  };

}  // namespace mkseeder

#endif
