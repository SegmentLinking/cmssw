#ifndef RecoTracker_MkFitCore_src_FinalFit_h
#define RecoTracker_MkFitCore_src_FinalFit_h

#include "Matrix.h"
#include "RecoTracker/MkFitCore/interface/FunctionTypes.h"

namespace mkfit {

  class PropagationFlags;

  // The final fit's propagation to the module planes and its Kalman update on them (MkFitter, called from
  // MkBuilder::fit_tracks()), composed from the steps in PlaneSteps.h.  The final fit has its own sequences so that
  // what only it does stays out of the propagation and Kalman functions that track finding uses.

  namespace final_fit {

    // Propagation of (inErr, inPar) to the planes (plPnt, plNrm), with material at the destination.
    void propagate(const MPlexLS& inErr,
                   const MPlexLV& inPar,
                   const MPlexQI& inChg,
                   const MPlexHV& plPnt,
                   const MPlexHV& plNrm,
                   MPlexLS& outErr,
                   MPlexLV& outPar,
                   MPlexQI& outFailFlag,
                   const int N_proc,
                   const PropagationFlags& pflags,
                   const MPlexQI* noMatEffPtr);

    // Propagation to the hits' planes (unless propToHit is false: the state is already there), then the Kalman
    // update with the hits and its chi2.  A negative q/p after the update flips the charge.
    void propagate_update(const MPlexLS& psErr,
                          const MPlexLV& psPar,
                          MPlexQI& Chg,
                          const MPlexHS& msErr,
                          const MPlexHV& msPar,
                          const MPlexHV& plNrm,
                          const MPlexHV& plDir,
                          const MPlexHV& plPnt,
                          MPlexLS& outErr,
                          MPlexLV& outPar,
                          MPlexQI& outFailFlag,
                          MPlexQF& outChi2,
                          const int N_proc,
                          const PropagationFlags& pflags,
                          const bool propToHit,
                          const MPlexQI* noMatEffPtr,
                          const MPlexQI* doCPE,
                          cpe_func cpe_corr_func);

  }  // namespace final_fit

}  // namespace mkfit

#endif
