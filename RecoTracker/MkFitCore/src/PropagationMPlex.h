#ifndef RecoTracker_MkFitCore_src_PropagationMPlex_h
#define RecoTracker_MkFitCore_src_PropagationMPlex_h

#include "Matrix.h"

#include "RecoTracker/MkFitCore/interface/portable/PropagationMPlex.h"

namespace mkfit {

  using portable::applyMaterialEffects;
  using portable::helixAtPlane;
  using portable::helixAtRFromIterativeCCS;
  using portable::helixAtRFromIterativeCCSFullJac;
  using portable::helixAtZ;
  using portable::MultHelixPropFull;
  using portable::MultHelixPropTranspFull;
  using portable::propagateHelixToPlaneMPlex;
  using portable::propagateHelixToRMPlex;
  using portable::propagateHelixToZMPlex;
  using portable::propagateLineToRMPlex;
  using portable::squashPhiMPlex;
  using portable::squashPhiMPlexGeneral;

  // MkFitCore's instantiations, in src/PropagationMPlex*.cc: no other translation unit compiles these functions.
  extern template void portable::applyMaterialEffects<NN>(const MPlexQF<NN>&,
                                                          const MPlexQF<NN>&,
                                                          const MPlexQF<NN>&,
                                                          const MPlexHV<NN>&,
                                                          MPlexLS<NN>&,
                                                          MPlexLV<NN>&,
                                                          const int,
                                                          const PropagationEnv&);
  extern template void portable::MultHelixPropFull<NN>(const MPlexLL<NN>&, const MPlexLS<NN>&, MPlexLL<NN>&);
  extern template void portable::MultHelixPropTranspFull<NN>(const MPlexLL<NN>&, const MPlexLL<NN>&, MPlexLS<NN>&);
  extern template void portable::propagateLineToRMPlex<NN>(const MPlexLS<NN>&,
                                                           const MPlexLV<NN>&,
                                                           const MPlexHS<NN>&,
                                                           const MPlexHV<NN>&,
                                                           MPlexLS<NN>&,
                                                           MPlexLV<NN>&,
                                                           const int);
  extern template void portable::propagateHelixToRMPlex<NN>(const MPlexLS<NN>&,
                                                            const MPlexLV<NN>&,
                                                            const MPlexQI<NN>&,
                                                            const MPlexQF<NN>&,
                                                            MPlexLS<NN>&,
                                                            MPlexLV<NN>&,
                                                            MPlexQI<NN>&,
                                                            const int,
                                                            const PropagationFlags&,
                                                            const MPlexQI<NN>*);
  extern template void portable::helixAtRFromIterativeCCSFullJac<NN>(
      const MPlexLV<NN>&, const MPlexQI<NN>&, const MPlexQF<NN>&, MPlexLV<NN>&, MPlexLL<NN>&, const int);
  extern template void portable::helixAtRFromIterativeCCS<NN>(const MPlexLV<NN>&,
                                                              const MPlexQI<NN>&,
                                                              const MPlexQF<NN>&,
                                                              MPlexLV<NN>&,
                                                              MPlexLL<NN>&,
                                                              MPlexQI<NN>&,
                                                              const int,
                                                              const PropagationFlags&);
  extern template void portable::propagateHelixToZMPlex<NN>(const MPlexLS<NN>&,
                                                            const MPlexLV<NN>&,
                                                            const MPlexQI<NN>&,
                                                            const MPlexQF<NN>&,
                                                            MPlexLS<NN>&,
                                                            MPlexLV<NN>&,
                                                            MPlexQI<NN>&,
                                                            const int,
                                                            const PropagationFlags&,
                                                            const MPlexQI<NN>*);
  extern template void portable::helixAtZ<NN>(const MPlexLV<NN>&,
                                              const MPlexQI<NN>&,
                                              const MPlexQF<NN>&,
                                              MPlexLV<NN>&,
                                              MPlexLL<NN>&,
                                              MPlexQI<NN>&,
                                              const int,
                                              const PropagationFlags&);
  extern template void portable::helixAtPlane<NN>(const MPlexLV<NN>&,
                                                  const MPlexQI<NN>&,
                                                  const MPlexHV<NN>&,
                                                  const MPlexHV<NN>&,
                                                  MPlexQF<NN>&,
                                                  MPlexLV<NN>&,
                                                  MPlexLL<NN>&,
                                                  MPlexQI<NN>&,
                                                  const int,
                                                  const PropagationFlags&);
  extern template void portable::propagateHelixToPlaneMPlex<NN>(const MPlexLS<NN>&,
                                                                const MPlexLV<NN>&,
                                                                const MPlexQI<NN>&,
                                                                const MPlexHV<NN>&,
                                                                const MPlexHV<NN>&,
                                                                MPlexLS<NN>&,
                                                                MPlexLV<NN>&,
                                                                MPlexQI<NN>&,
                                                                const int,
                                                                const PropagationFlags&,
                                                                const MPlexQI<NN>*);

}  // end namespace mkfit
#endif
