#ifndef RecoTracker_MkFitAlpaka_src_alpaka_prop_PlaneSteps_h
#define RecoTracker_MkFitAlpaka_src_alpaka_prop_PlaneSteps_h

// Portable copy of RecoTracker/MkFitCore/src/PlaneSteps.h: propagation to a plane and the Kalman operation on it as
// a sequence of steps (namespace plane), templated on N like the rest of src/alpaka/prop. The step bodies are those
// of PropagationMPlexPlane.cc (field_at_start .. finish) and KalmanUtilsMPlex.cc (namespace plane), same operations
// in the same order; they share the per-slot helpers of PropagationMPlex.h and KalmanUtilsMPlex.h.
// Device deviations: the material comes from PropagationFlags::material (MaterialView) instead of TrackerInfo, and
// the field constants from PropagationFlags::startField().

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexBackend.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationFlags.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationMPlex.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::plane {

  // Track parameters the caller holds, with the charge and the number of used lanes.
  template <idx_t N>
  struct TrackRef {
    const MPlexLV<N>& par;
    const MPlexQI<N>& chg;
    const int n_proc;
  };

  // The plane: a point on it and its normal.
  template <idx_t N>
  struct PlaneRef {
    const MPlexHV<N>& pnt;
    const MPlexHV<N>& nrm;
  };

  // Bz [T] per lane and the curvature factor kinv = -q * sol/100 * Bz.
  template <idx_t N>
  struct FieldAt {
    MPlexQF<N> b;
    MPlexQF<N> kinv;
  };

  // sin and cos of the start state's momentum phi and theta.
  template <idx_t N>
  struct StartTrig {
    MPlexQF<N> sinP, cosP;
    MPlexQF<N> sinT, cosT;
  };

  // The path length to the plane while it is being solved, and the start state's straight-line solution.
  template <idx_t N>
  struct PathSolve {
    MPlexQF<N> s{0.0f};
    MPlexQF<N> s_line;
  };

  // Material at the destination: radiation length, Bethe-Bloch xi, sign of the energy loss.
  template <idx_t N>
  struct MaterialAt {
    MPlexQF<N> radl;
    MPlexQF<N> xi;
    MPlexQF<N> sign;
  };

  // The predicted state in the local frame of the plane.
  template <idx_t N>
  struct LocalPred {
    MPlexHH<N> rot;
    MPlex2V<N> xlo;
    MPlex5V<N> lp;
    MPlexQI<N> pz_sign;
    MPlex5S<N> err;
  };

  // The measurement in the local frame of the plane.
  template <idx_t N>
  struct LocalMeas {
    MPlex2V<N> par;
    MPlex2S<N> err;
  };

  // Residual, the INVERSE of its covariance, and the determinant of the covariance before the inversion.
  template <idx_t N>
  struct Residual {
    MPlex2V<N> r;
    MPlex2S<N> s_inv;
    MPlexQF<N> det;
  };

  // The updated state in the local frame.
  template <idx_t N>
  struct LocalUpd {
    MPlex5V<N> lp;
    MPlex5S<N> err;
  };

  // ---- Propagation to a plane (PropagationMPlexPlane.cc)

  // field_at_start
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void field_at_start(const TrackRef<N>& in,
                                                          const PropagationFlags& pf,
                                                          FieldAt<N>& f);

  // start_trig
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void start_trig(const TrackRef<N>& in, StartTrig<N>& t);

  // path_init (pathInit_impl, with the ambiguous-root rule)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_init(
      const TrackRef<N>& in, const StartTrig<N>& t, const PlaneRef<N>& pl, const FieldAt<N>& f, PathSolve<N>& p);

  // path_refine_from (pathRefine_impl)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_refine_from(const TrackRef<N>& in,
                                                            const StartTrig<N>& t,
                                                            const PlaneRef<N>& pl,
                                                            const FieldAt<N>& f,
                                                            const MPlexLV<N>& at,
                                                            PathSolve<N>& p);

  // path_refine
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_refine(
      const TrackRef<N>& in, const StartTrig<N>& t, const PlaneRef<N>& pl, const FieldAt<N>& f, PathSolve<N>& p);

  // path_close (pathClose_impl)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_close(const TrackRef<N>& in, PathSolve<N>& p);

  // path_solve: path_init, Config::nSStepsInProp2Plane - 1 refinements, path_close
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_solve(
      const TrackRef<N>& in, const StartTrig<N>& t, const PlaneRef<N>& pl, const FieldAt<N>& f, PathSolve<N>& p);

  // path_from_perp: s = sPerp / sinT
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_from_perp(const TrackRef<N>& in,
                                                          const StartTrig<N>& t,
                                                          const MPlexQF<N>& sPerp,
                                                          MPlexQF<N>& s);

  // drift (parsFromPathL_impl)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void drift(
      const TrackRef<N>& in, const StartTrig<N>& t, const FieldAt<N>& f, const MPlexQF<N>& s, MPlexLV<N>& outPar);

  // jacobian (errPropFromPathL_impl)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void jacobian(const TrackRef<N>& in,
                                                    const StartTrig<N>& t,
                                                    const MPlexLV<N>& outPar,
                                                    const FieldAt<N>& f,
                                                    const MPlexQF<N>& s,
                                                    MPlexLL<N>& J);

  // transport_cov: J C J^T
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void transport_cov(const MPlexLL<N>& J,
                                                         const MPlexLS<N>& inErr,
                                                         MPlexLS<N>& outErr);

  // material_grid, from PropagationFlags::material in place of TrackerInfo::material_checked
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void material_grid(const PropagationFlags& pf,
                                                         const MPlexLV<N>& par,
                                                         const MPlexQI<N>* noMatEffPtr,
                                                         const int N_proc,
                                                         MaterialAt<N>& m);

  // eloss_sign_from_path
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void eloss_sign_from_path(const MPlexQF<N>& s,
                                                                const MPlexQI<N>* noMatEffPtr,
                                                                const int N_proc,
                                                                MaterialAt<N>& m);

  // apply_material (applyMaterialEffects)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void apply_material(const MaterialAt<N>& m,
                                                          const MPlexHV<N>& plNrm,
                                                          MPlexLS<N>& err,
                                                          MPlexLV<N>& par,
                                                          const int N_proc,
                                                          const MPlexQF<N>* ms_ref_p = nullptr);

  // finish: phi into [-pi, pi), the start state restored on failed lanes
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void finish(
      const TrackRef<N>& in, const MPlexLS<N>& inErr, const MPlexQI<N>& failFlag, MPlexLV<N>& par, MPlexLS<N>& err);

  // ---- Kalman operation on a plane (KalmanUtilsMPlex.cc, namespace plane)

  // local_field
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void local_field(const TrackRef<N>& pred,
                                                       const PropagationFlags& pf,
                                                       const bool use_param_b_field,
                                                       MPlexQF<N>& bFld,
                                                       const bool fit_b_field = false);

  // to_local
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void to_local(const TrackRef<N>& pred,
                                                    const MPlexLS<N>& psErr,
                                                    const PlaneRef<N>& pl,
                                                    const MPlexHV<N>& plDir,
                                                    const MPlexQF<N>& bFld,
                                                    LocalPred<N>& L);

  // measurement
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void measurement(const LocalPred<N>& L,
                                                       const MPlexHV<N>& msPar,
                                                       const MPlexHS<N>& msErr,
                                                       const MPlexHV<N>& plPnt,
                                                       LocalMeas<N>& M);

  // residual (with det of the residual covariance before the inversion)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void residual(const LocalPred<N>& L, const LocalMeas<N>& M, Residual<N>& R);

  // chi2
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void chi2(const Residual<N>& R, MPlexQF<N>& outChi2);

  // chi2_scaled: the track part of the residual covariance scaled by trk_scale2
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void chi2_scaled(
      const LocalPred<N>& L, const LocalMeas<N>& M, const Residual<N>& R, const float trk_scale2, MPlexQF<N>& out);

  // update
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void update(const LocalPred<N>& L, const Residual<N>& R, LocalUpd<N>& U);

  // to_global
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void to_global(const LocalUpd<N>& U,
                                                     const LocalPred<N>& L,
                                                     const PlaneRef<N>& pl,
                                                     const MPlexQI<N>& inChg,
                                                     const MPlexQF<N>& bFld,
                                                     MPlexLV<N>& outPar,
                                                     MPlexLS<N>& outErr);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::plane

#endif
