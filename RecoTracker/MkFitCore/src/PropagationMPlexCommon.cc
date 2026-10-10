#include "PropagationMPlex.h"

//#define DEBUG
#include "Debug.h"

namespace mkfit {

  template void portable::applyMaterialEffects<NN>(const MPlexQF<NN>&,
                                                   const MPlexQF<NN>&,
                                                   const MPlexQF<NN>&,
                                                   const MPlexHV<NN>&,
                                                   MPlexLS<NN>&,
                                                   MPlexLV<NN>&,
                                                   const int,
                                                   const PropagationEnv&);
  template void portable::MultHelixPropFull<NN>(const MPlexLL<NN>&, const MPlexLS<NN>&, MPlexLL<NN>&);
  template void portable::MultHelixPropTranspFull<NN>(const MPlexLL<NN>&, const MPlexLL<NN>&, MPlexLS<NN>&);

}  // namespace mkfit
