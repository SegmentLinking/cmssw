#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederForest_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederForest_h

// The seed classifier's trees (MkSeederForestSoA.h) on the host, the ES product of MkSeederForestESProducer, and their
// reader.

#include <string>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederForestSoA.h"

namespace mkseeder {

  using MkSeederForestHostCollection = PortableHostCollection<MkSeederForestSoA>;

  // The trees of an xgboost model saved by Booster.save_model("x.json") (xgboost 1.7's JSON format; the objectives
  // binary:logistic and reg:logistic, the gbtree booster, numerical splits), padded to full depth. Read:
  // learner.learner_model_param.{base_score, num_feature, num_class}, learner.objective.name and per tree of
  // learner.gradient_booster.model.trees its split_indices, split_conditions, left_children, right_children and
  // default_left: a node goes left when x[feature] < split_condition (float), a NaN follows default_left, a leaf's
  // value is its split_condition; the base margin is logit(base_score). Throws cms::Exception for a file it cannot
  // read, a model it does not support or one deeper than kMaxForestDepth.
  constexpr int kMaxForestDepth = 10;
  MkSeederForestHostCollection readMkSeederForest(const std::string &path);

}  // namespace mkseeder

#endif
