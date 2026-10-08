#include "PropagationMPlex.h"

//#define DEBUG
#include "Debug.h"

namespace mkfit {

  template void portable::propagateHelixToZMPlex<NN>(const MPlexLS<NN>&,
                                                     const MPlexLV<NN>&,
                                                     const MPlexQI<NN>&,
                                                     const MPlexQF<NN>&,
                                                     MPlexLS<NN>&,
                                                     MPlexLV<NN>&,
                                                     MPlexQI<NN>&,
                                                     const int,
                                                     const PropagationFlags&,
                                                     const MPlexQI<NN>*);
  template void portable::helixAtZ<NN>(const MPlexLV<NN>&,
                                       const MPlexQI<NN>&,
                                       const MPlexQF<NN>&,
                                       MPlexLV<NN>&,
                                       MPlexLL<NN>&,
                                       MPlexQI<NN>&,
                                       const int,
                                       const PropagationFlags&);

}  // namespace mkfit
