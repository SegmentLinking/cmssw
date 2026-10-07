#ifndef RecoTracker_MkFitCore_src_FinalFit_h
#define RecoTracker_MkFitCore_src_FinalFit_h

#include "Matrix.h"
#include "RecoTracker/MkFitCore/interface/FunctionTypes.h"

namespace mkfit {

  class PropagationFlags;
  struct FinalFitFlags;

  // The final fit's propagation to the module planes and its Kalman update on them (MkFitter, called from
  // MkBuilder::fit_tracks()), composed from the steps in PlaneSteps.h.  The final fit has its own sequences so that
  // what only it does stays out of the propagation and Kalman functions that track finding uses.

  namespace final_fit {

    // What stays fixed over one pass of the final fit: the propagation flags, the final fit's own choices, the
    // direction of the pass (the forward pass runs outward), the per-lane |p| at which the scattering noise is
    // evaluated (nullptr: the running estimate), and the number of sub-steps of each propagation.
    struct Pass {
      const PropagationFlags& pflags;
      const FinalFitFlags& ffflags;
      const bool outward;
      const float* ms_ref_p = nullptr;
      const int n_sub = 1;
    };

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
                   const Pass& pass,
                   const MPlexQI* noMatEffPtr);

    // Propagation of (inErr, inPar) to the planes in nSub sub-steps: the parameters through nSub - 1 drifts of
    // fixed path length s0/nSub (s0 = the first path-length estimate to the plane), each with the field model of a
    // full step, then onto the plane with the full solve; the covariance with the Jacobian of the whole step;
    // material at the destination.  split[n] = false keeps lane n as one step.
    void propagate_sub_steps(const MPlexLS& inErr,
                             const MPlexLV& inPar,
                             const MPlexQI& inChg,
                             const MPlexHV& plPnt,
                             const MPlexHV& plNrm,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             MPlexQI& outFailFlag,
                             const int N_proc,
                             const Pass& pass,
                             const int nSub,
                             const bool* split,
                             const MPlexQI* noMatEffPtr);

    // Propagation to the hits' planes (unless propToHit is false: the state is already there), then the Kalman
    // update with the hits and its chi2.  A negative q/p after the update flips the charge.  With pass.n_sub > 1
    // the lanes not yet on their plane are propagated in sub-steps.
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
                          const Pass& pass,
                          const bool propToHit,
                          const MPlexQI* noMatEffPtr,
                          const MPlexQI* doCPE,
                          cpe_func cpe_corr_func);

  }  // namespace final_fit

}  // namespace mkfit

#endif
