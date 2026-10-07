#include "FinalFit.h"
#include "PlaneSteps.h"

#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"

namespace mkfit::final_fit {

  using namespace plane;

  namespace {

    // The Kalman update of the predicted state (psErr, psPar) with the hits, and its chi2.
    void update_on_plane(const MPlexLS& psErr,
                         const MPlexLV& psPar,
                         const MPlexQI& inChg,
                         const MPlexHS& msErr,
                         const MPlexHV& msPar,
                         const MPlexHV& plNrm,
                         const MPlexHV& plDir,
                         const MPlexHV& plPnt,
                         MPlexLS& outErr,
                         MPlexLV& outPar,
                         MPlexQF& outChi2,
                         const int N_proc,
                         const PropagationFlags& pflags,
                         const MPlexQI* doCPE,
                         cpe_func cpe_corr_func) {
      const TrackRef pred{psPar, inChg, N_proc};
      const PlaneRef pl{plPnt, plNrm};

      MPlexQF bFld;
      local_field(pred, pflags.use_param_b_field, bFld);

      LocalPred L;
      to_local(pred, psErr, pl, plDir, bFld, L);

      LocalMeas M;
      measurement(L, msPar, msErr, plPnt, M);
      if (doCPE && cpe_corr_func)
        measurement_cpe(*doCPE, cpe_corr_func, L, M);

      Residual R;
      residual(L, M, R);
      chi2(R, outChi2);

      LocalUpd U;
      update(L, R, U);
      to_global(U, L, pl, inChg, bFld, outPar, outErr);
    }

  }  // namespace

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
                 const MPlexQI* noMatEffPtr) {
    const TrackRef in{inPar, inChg, N_proc};
    const PlaneRef pl{plPnt, plNrm};

    MPlexLL errorProp{0.0f};
    outFailFlag.setVal(0.f);

    FieldAt f;
    field_at_start(in, pflags, f);
    StartTrig t;
    start_trig(in, t);
    PathSolve p;
    path_solve(in, t, pl, f, p);
    drift(in, t, f, p.s, outPar);
    jacobian(in, t, outPar, f, p.s, errorProp);

    transport_cov(errorProp, inErr, outErr);

    if (pflags.apply_material) {
      MaterialAt m;
      material_grid(*pflags.tracker_info, outPar, noMatEffPtr, N_proc, m);
      eloss_sign_from_path(p.s, noMatEffPtr, N_proc, m);
      apply_material(m, plNrm, outErr, outPar, N_proc);
    }

    finish(in, inErr, outFailFlag, outPar, outErr);
  }

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
                        cpe_func cpe_corr_func) {
    if (propToHit) {
      MPlexLS propErr;
      MPlexLV propPar;
      propagate(psErr, psPar, Chg, plPnt, plNrm, propErr, propPar, outFailFlag, N_proc, pflags, noMatEffPtr);
      update_on_plane(propErr,
                      propPar,
                      Chg,
                      msErr,
                      msPar,
                      plNrm,
                      plDir,
                      plPnt,
                      outErr,
                      outPar,
                      outChi2,
                      N_proc,
                      pflags,
                      doCPE,
                      cpe_corr_func);
    } else {
      update_on_plane(psErr,
                      psPar,
                      Chg,
                      msErr,
                      msPar,
                      plNrm,
                      plDir,
                      plPnt,
                      outErr,
                      outPar,
                      outChi2,
                      N_proc,
                      pflags,
                      doCPE,
                      cpe_corr_func);
    }
    for (int n = 0; n < NN; ++n) {
      if (outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

}  // namespace mkfit::final_fit
