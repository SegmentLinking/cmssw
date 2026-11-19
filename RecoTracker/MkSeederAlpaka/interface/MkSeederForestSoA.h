#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederForestSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederForestSoA_h

// The seed classifier's gradient-boosted trees (an xgboost model, readMkSeederForest()), padded to the same full depth:
// per tree its 2^depth - 1 split nodes in heap order (node h's children are 2h + 1 and 2h + 2) and its 2^depth leaves,
// tree after tree. A split sends a feature value x right when !(x < threshold), a NaN by nanGoesRight; a padding
// split (below a leaf of the model) always goes left (threshold +inf, NaN left), and the leaves below it all carry the
// leaf's value. The margin of a seed is baseMargin plus its leaves of all trees, summed in tree order in double; the
// score is 1 / (1 + exp(-margin)).

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  GENERATE_SOA_LAYOUT(MkSeederForestNodesLayout,
                      SOA_COLUMN(float, threshold),
                      SOA_COLUMN(uint32_t, feature),
                      SOA_COLUMN(uint8_t, nanGoesRight),
                      SOA_SCALAR(uint32_t, nTrees),
                      SOA_SCALAR(int32_t, depth),
                      SOA_SCALAR(int32_t, nFeatures),
                      SOA_SCALAR(double, baseMargin))

  GENERATE_SOA_LAYOUT(MkSeederForestLeavesLayout, SOA_COLUMN(float, value))

  GENERATE_SOA_BLOCKS(MkSeederForestLayout,
                      SOA_BLOCK(nodes, MkSeederForestNodesLayout),
                      SOA_BLOCK(leaves, MkSeederForestLeavesLayout))

  using MkSeederForestNodesSoA = MkSeederForestNodesLayout<>;
  using MkSeederForestLeavesSoA = MkSeederForestLeavesLayout<>;
  using MkSeederForestSoA = MkSeederForestLayout<>;

}  // namespace mkseeder

#endif
