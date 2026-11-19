#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <vector>

#include <nlohmann/json.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederForest.h"

namespace {
  using json = nlohmann::json;

  // xgboost writes the learner parameters as strings ("5E-1", "41"); numbers are accepted too
  double asDouble(const json &j) { return j.is_string() ? std::stod(j.get<std::string>()) : j.get<double>(); }
  long asLong(const json &j) {
    if (j.is_string())
      return std::stol(j.get<std::string>());
    if (j.is_boolean())
      return j.get<bool>();
    return j.get<long>();
  }
  const json &at(const json &j, const char *key, const char *where) {
    const auto it = j.find(key);
    if (it == j.end())
      throw cms::Exception("MkSeederForest") << "no '" << key << "' in " << where;
    return *it;
  }

  // a node of the model: the split (left < 0: a leaf, whose value is the split condition)
  struct Node {
    float value;
    int left, right, feature;
    bool defaultLeft;
  };
}  // namespace

namespace mkseeder {

  MkSeederForestHostCollection readMkSeederForest(const std::string &path) {
    std::ifstream in(path);
    if (!in)
      throw cms::Exception("MkSeederForest") << "cannot open " << path;
    json doc;
    try {
      doc = json::parse(in);
    } catch (const json::exception &e) {
      throw cms::Exception("MkSeederForest") << "cannot parse " << path << ": " << e.what();
    }
    std::vector<Node> nodes;
    std::vector<int> roots;
    int nFeatures = 0;
    double baseMargin = 0;
    try {
      const json &learner = at(doc, "learner", path.c_str());
      const json &modelParameters = at(learner, "learner_model_param", "learner");
      nFeatures = asLong(at(modelParameters, "num_feature", "learner_model_param"));
      if (modelParameters.contains("num_class") && asLong(modelParameters["num_class"]) > 1)
        throw cms::Exception("MkSeederForest") << path << ": multi-class models are not supported";
      const double baseScore = asDouble(at(modelParameters, "base_score", "learner_model_param"));
      const std::string objective = at(at(learner, "objective", "learner"), "name", "objective").get<std::string>();
      if (objective != "binary:logistic" && objective != "reg:logistic")
        throw cms::Exception("MkSeederForest")
            << path << ": objective " << objective << ", only binary:logistic and reg:logistic are supported";
      if (!(baseScore > 0 && baseScore < 1))
        throw cms::Exception("MkSeederForest") << path << ": base_score " << baseScore << " is not in (0, 1)";
      baseMargin = -std::log(1. / baseScore - 1.);
      const json &booster = at(learner, "gradient_booster", "learner");
      const std::string boosterName = at(booster, "name", "gradient_booster").get<std::string>();
      if (boosterName != "gbtree")
        throw cms::Exception("MkSeederForest") << path << ": booster " << boosterName << ", only gbtree is supported";
      for (const json &tree : at(at(booster, "model", "gradient_booster"), "trees", "model")) {
        const json &splitIndices = at(tree, "split_indices", "tree");
        const json &splitConditions = at(tree, "split_conditions", "tree");
        const json &leftChildren = at(tree, "left_children", "tree");
        const json &rightChildren = at(tree, "right_children", "tree");
        const json &defaultLeft = at(tree, "default_left", "tree");
        const size_t n = leftChildren.size();
        if (splitIndices.size() != n || splitConditions.size() != n || rightChildren.size() != n ||
            defaultLeft.size() != n || n == 0)
          throw cms::Exception("MkSeederForest") << path << ": inconsistent node arrays in tree " << roots.size();
        if (tree.contains("split_type"))
          for (const json &splitType : tree["split_type"])
            if (asLong(splitType) != 0)
              throw cms::Exception("MkSeederForest") << path << ": categorical splits are not supported";
        const int first = nodes.size();
        roots.push_back(first);
        for (size_t k = 0; k < n; ++k) {
          Node node;
          node.value = static_cast<float>(splitConditions[k].get<double>());
          const long left = asLong(leftChildren[k]), right = asLong(rightChildren[k]);
          const bool leaf = left < 0;
          node.left = leaf ? -1 : first + left;
          node.right = leaf ? -1 : first + right;
          node.feature = leaf ? 0 : asLong(splitIndices[k]);
          node.defaultLeft = asLong(defaultLeft[k]) != 0;
          // children come after their parent in xgboost's trees (which also rules out a cycle)
          if (!leaf && (left <= long(k) || left >= long(n) || right <= long(k) || right >= long(n) ||
                        node.feature < 0 || node.feature >= nFeatures))
            throw cms::Exception("MkSeederForest") << path << ": bad node " << k << " in tree " << roots.size() - 1;
          nodes.push_back(node);
        }
      }
    } catch (const json::exception &e) {
      throw cms::Exception("MkSeederForest") << "unexpected content in " << path << ": " << e.what();
    }
    if (roots.empty())
      throw cms::Exception("MkSeederForest") << path << ": no trees";

    // the trees padded to the depth of the deepest one
    auto depthOf = [&](auto &self, int k) -> int {
      return nodes[k].left < 0 ? 0 : 1 + std::max(self(self, nodes[k].left), self(self, nodes[k].right));
    };
    int depth = 1;
    for (const int root : roots)
      depth = std::max(depth, depthOf(depthOf, root));
    if (depth > kMaxForestDepth)
      throw cms::Exception("MkSeederForest")
          << path << ": trees of depth " << depth << ", at most " << kMaxForestDepth << " are supported";
    const int nSplits = (1 << depth) - 1, nLeaves = 1 << depth, nTrees = roots.size();
    MkSeederForestHostCollection forest(cms::alpakatools::host(), nTrees * nSplits, nTrees * nLeaves);
    auto splits = forest.view().nodes();
    auto leaves = forest.view().leaves();
    splits.nTrees() = nTrees, splits.depth() = depth, splits.nFeatures() = nFeatures;
    splits.baseMargin() = baseMargin;
    for (int t = 0; t < nTrees; ++t) {
      const int splitOffset = t * nSplits, leafOffset = t * nLeaves;
      // node k of the model at heap position h and level
      auto fill = [&](auto &self, int k, int h, int level) -> void {
        const Node &node = nodes[k];
        if (level == depth) {
          leaves.value(leafOffset + h - nSplits) = node.value;
          return;
        }
        auto split = splits[splitOffset + h];
        if (node.left < 0) {
          // a padding split below a leaf
          split.threshold() = std::numeric_limits<float>::infinity(), split.feature() = 0, split.nanGoesRight() = 0;
          self(self, k, 2 * h + 1, level + 1);
          self(self, k, 2 * h + 2, level + 1);
          return;
        }
        split.threshold() = node.value, split.feature() = node.feature, split.nanGoesRight() = !node.defaultLeft;
        self(self, node.left, 2 * h + 1, level + 1);
        self(self, node.right, 2 * h + 2, level + 1);
      };
      fill(fill, roots[t], 0, 0);
    }
    return forest;
  }  // readMkSeederForest()

}  // namespace mkseeder
