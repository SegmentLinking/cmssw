#include "PropagationMPlex.h"

//#define DEBUG
#include "Debug.h"

namespace mkfit {

  template void portable::propagateLineToRMPlex<NN>(const MPlexLS<NN>&,
                                                    const MPlexLV<NN>&,
                                                    const MPlexHS<NN>&,
                                                    const MPlexHV<NN>&,
                                                    MPlexLS<NN>&,
                                                    MPlexLV<NN>&,
                                                    const int);
  template void portable::propagateHelixToRMPlex<NN>(const MPlexLS<NN>&,
                                                     const MPlexLV<NN>&,
                                                     const MPlexQI<NN>&,
                                                     const MPlexQF<NN>&,
                                                     MPlexLS<NN>&,
                                                     MPlexLV<NN>&,
                                                     MPlexQI<NN>&,
                                                     const int,
                                                     const PropagationFlags&,
                                                     const MPlexQI<NN>*);
  template void portable::helixAtRFromIterativeCCSFullJac<NN>(
      const MPlexLV<NN>&, const MPlexQI<NN>&, const MPlexQF<NN>&, MPlexLV<NN>&, MPlexLL<NN>&, const int);
  template void portable::helixAtRFromIterativeCCS<NN>(const MPlexLV<NN>&,
                                                       const MPlexQI<NN>&,
                                                       const MPlexQF<NN>&,
                                                       MPlexLV<NN>&,
                                                       MPlexLL<NN>&,
                                                       MPlexQI<NN>&,
                                                       const int,
                                                       const PropagationFlags&);

}  // namespace mkfit
