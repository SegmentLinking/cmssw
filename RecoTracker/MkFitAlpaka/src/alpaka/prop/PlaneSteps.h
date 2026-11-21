#ifndef RecoTracker_MkFitAlpaka_src_alpaka_prop_PlaneSteps_h
#define RecoTracker_MkFitAlpaka_src_alpaka_prop_PlaneSteps_h

// Portable copy of RecoTracker/MkFitCore/src/PlaneSteps.h: propagation to a plane and the Kalman operation on it as
// a sequence of steps (namespace plane), templated on N like the rest of src/alpaka/prop. The step bodies are those
// of PropagationMPlexPlane.cc (field_at_start .. finish) and KalmanUtilsMPlex.cc (namespace plane), same operations
// in the same order; they share the per-slot helpers of PropagationMPlex.h and KalmanUtilsMPlex.h.
// Device deviations: the material comes from PropagationFlags::material (MaterialView) instead of TrackerInfo, and
// the field constants from it too (local_field takes the flags for them). Not ported: material_given and
// measurement_cpe, which the v2p2 finder does not call.

#include <cmath>
#include <cstddef>

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
                                                          FieldAt<N>& f) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      f.kinv[n] = in.chg(n, 0, 0) < 0 ? Const::sol_over_100 : -Const::sol_over_100;
      f.b[n] = pf.use_param_b_field ? propdetail::bFieldFromZXY(pf, in.par(n, 2, 0), in.par(n, 0, 0), in.par(n, 1, 0))
                                    : Config::Bfield;
      f.kinv[n] *= f.b[n];
    }
  }

  // start_trig
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void start_trig(const TrackRef<N>& in, StartTrig<N>& t) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      vdt::fast_sincosf(in.par(n, 4, 0), t.sinP[n], t.cosP[n]);
      vdt::fast_sincosf(in.par(n, 5, 0), t.sinT[n], t.cosT[n]);
    }
  }

  // path_init (pathInit_impl, with the ambiguous-root rule)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_init(
      const TrackRef<N>& in, const StartTrig<N>& t, const PlaneRef<N>& pl, const FieldAt<N>& f, PathSolve<N>& p) {
    const MPlexLV<N>& inPar = in.par;
    const MPlexHV<N>& plPnt = pl.pnt;
    const MPlexHV<N>& plNrm = pl.nrm;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float delta0 = inPar(n, 0, 0) - plPnt(n, 0, 0);
      const float delta1 = inPar(n, 1, 0) - plPnt(n, 1, 0);
      const float delta2 = inPar(n, 2, 0) - plPnt(n, 2, 0);

      // determine solution for straight line
      p.s_line[n] = -(plNrm(n, 0, 0) * delta0 + plNrm(n, 1, 0) * delta1 + plNrm(n, 2, 0) * delta2) /
                    (plNrm(n, 0, 0) * t.cosP[n] * t.sinT[n] + plNrm(n, 1, 0) * t.sinP[n] * t.sinT[n] +
                     plNrm(n, 2, 0) * t.cosT[n]);

      // The helix and the disk solutions are computed for every lane, and the plane type picks one after.
      float amb;
      float sH = propdetail::getS(delta0,
                                  delta1,
                                  delta2,
                                  plNrm(n, 0, 0),
                                  plNrm(n, 1, 0),
                                  plNrm(n, 2, 0),
                                  t.sinP[n],
                                  t.cosP[n],
                                  t.sinT[n],
                                  t.cosT[n],
                                  inPar(n, 3, 0),
                                  in.chg(n, 0, 0),
                                  f.kinv[n],
                                  &amb);
      const float sZ = (plPnt(n, 2, 0) - inPar(n, 2, 0)) / t.cosT[n];
      if (n < in.n_proc) {
        // The rare lanes with two comparable roots, see getSNearPlanePoint().
        if (amb < 0.f && std::abs(plNrm(n, 2, 0)) < 1.f)
          sH = propdetail::getSNearPlanePoint(delta0,
                                              delta1,
                                              delta2,
                                              plNrm(n, 0, 0),
                                              plNrm(n, 1, 0),
                                              plNrm(n, 2, 0),
                                              t.sinP[n],
                                              t.cosP[n],
                                              t.sinT[n],
                                              t.cosT[n],
                                              inPar(n, 3, 0),
                                              f.kinv[n],
                                              sH);
        p.s[n] = std::abs(plNrm(n, 2, 0)) < 1.f ? sH : sZ;
      }
    }
  }

  // path_refine_from (pathRefine_impl)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_refine_from(const TrackRef<N>& in,
                                                            const StartTrig<N>& t,
                                                            const PlaneRef<N>& pl,
                                                            const FieldAt<N>& f,
                                                            const MPlexLV<N>& at,
                                                            PathSolve<N>& p) {
    const MPlexHV<N>& plPnt = pl.pnt;
    const MPlexHV<N>& plNrm = pl.nrm;
    // As in path_init: the helix solution for every lane, then the disk one, with its scalar cos, on the lanes that
    // need it.
    float sH[N];
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float delta0 = at(n, 0, 0) - plPnt(n, 0, 0);
      const float delta1 = at(n, 1, 0) - plPnt(n, 1, 0);
      const float delta2 = at(n, 2, 0) - plPnt(n, 2, 0);

      float sinP, cosP;
      vdt::fast_sincosf(at(n, 4, 0), sinP, cosP);
      // Note, sinT/cosT not updated

      sH[n] = propdetail::getS(delta0,
                               delta1,
                               delta2,
                               plNrm(n, 0, 0),
                               plNrm(n, 1, 0),
                               plNrm(n, 2, 0),
                               sinP,
                               cosP,
                               t.sinT[n],
                               t.cosT[n],
                               in.par(n, 3, 0),
                               in.chg(n, 0, 0),
                               f.kinv[n]);
    }
    for (int n = 0; n < N; ++n) {
      if (n < in.n_proc) {
        if (!(std::abs(plNrm(n, 2, 0)) < 1.f))
          sH[n] = (plPnt(n, 2, 0) - at(n, 2, 0)) / std::cos(at(n, 5, 0));
      }
    }
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      if (n < in.n_proc)
        p.s[n] += sH[n];
    }
  }

  // path_refine
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_refine(
      const TrackRef<N>& in, const StartTrig<N>& t, const PlaneRef<N>& pl, const FieldAt<N>& f, PathSolve<N>& p) {
    MPlexLV<N> at(0.0f);
    drift(in, t, f, p.s, at);
    path_refine_from(in, t, pl, f, at, p);
  }

  // path_close (pathClose_impl)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_close(const TrackRef<N>& in, PathSolve<N>& p) {
    // use linear approximation if s did not converge (for very high pT tracks)
    for (int n = 0; n < in.n_proc; ++n) {
      if (isFinite(p.s[n]) == false && isFinite(p.s_line[n]))  // replace with sl even if not fully correct
        p.s[n] = p.s_line[n];
    }
  }

  // path_solve: path_init, Config::nSStepsInProp2Plane - 1 refinements, path_close
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_solve(
      const TrackRef<N>& in, const StartTrig<N>& t, const PlaneRef<N>& pl, const FieldAt<N>& f, PathSolve<N>& p) {
    path_init(in, t, pl, f, p);
    for (int i = 0; i < Config::nSStepsInProp2Plane - 1; ++i)
      path_refine(in, t, pl, f, p);
    path_close(in, p);
  }

  // path_from_perp: s = sPerp / sinT
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void path_from_perp(const TrackRef<N>& in,
                                                          const StartTrig<N>& t,
                                                          const MPlexQF<N>& sPerp,
                                                          MPlexQF<N>& s) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n)
      s[n] = sPerp[n] / t.sinT[n];
  }

  // drift (parsFromPathL_impl)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void drift(
      const TrackRef<N>& in, const StartTrig<N>& t, const FieldAt<N>& f, const MPlexQF<N>& s, MPlexLV<N>& outPar) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n)
      propdetail::parsFromPathL_slot(in.par, outPar, f.kinv[n], s[n], t, n);
  }

  // jacobian (errPropFromPathL_impl)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void jacobian(const TrackRef<N>& in,
                                                    const StartTrig<N>& t,
                                                    const MPlexLV<N>& outPar,
                                                    const FieldAt<N>& f,
                                                    const MPlexQF<N>& s,
                                                    MPlexLL<N>& J) {
    propdetail::errPropFromPathL_impl(in.par, in.chg, t.sinP, t.cosP, t.sinT, t.cosT, outPar, f.b, s, J, in.n_proc);
  }

  // transport_cov: J C J^T
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void transport_cov(const MPlexLL<N>& J,
                                                         const MPlexLS<N>& inErr,
                                                         MPlexLS<N>& outErr) {
    // Matriplex version of:
    // result.errors = ROOT::Math::Similarity(errorProp, outErr);
    MPlexLL<N> temp(0.0f);
    propdetail::MultHelixPlaneProp(J, inErr, temp);
    propdetail::MultHelixPlanePropTransp(J, temp, outErr);
  }

  // material_grid, from PropagationFlags::material in place of TrackerInfo::material_checked
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void material_grid(const PropagationFlags& pf,
                                                         const MPlexLV<N>& par,
                                                         const MPlexQI<N>* noMatEffPtr,
                                                         const int N_proc,
                                                         MaterialAt<N>& m) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      if (n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0))) {
        m.radl(n, 0, 0) = 0.f;
        m.xi(n, 0, 0) = 0.f;
      } else {
        const float hypo = hipo(par(n, 0, 0), par(n, 1, 0));
        const auto mat = material_checked(pf.material, std::abs(par(n, 2, 0)), hypo);
        m.radl(n, 0, 0) = mat.radl;
        m.xi(n, 0, 0) = mat.bbxi;
      }
    }
  }

  // eloss_sign_from_path
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void eloss_sign_from_path(const MPlexQF<N>& s,
                                                                const MPlexQI<N>* noMatEffPtr,
                                                                const int N_proc,
                                                                MaterialAt<N>& m) {
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const bool none = n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0));
      m.sign(n, 0, 0) = none ? -1.f : (s(n, 0, 0) > 0.f ? 1.f : -1.f);
    }
  }

  // material_grid and eloss_sign_from_path called with a literal nullptr, as upstream calls them (N is not deducible
  // from it)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void material_grid(
      const PropagationFlags& pf, const MPlexLV<N>& par, std::nullptr_t, const int N_proc, MaterialAt<N>& m) {
    material_grid(pf, par, static_cast<const MPlexQI<N>*>(nullptr), N_proc, m);
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void eloss_sign_from_path(const MPlexQF<N>& s,
                                                                std::nullptr_t,
                                                                const int N_proc,
                                                                MaterialAt<N>& m) {
    eloss_sign_from_path(s, static_cast<const MPlexQI<N>*>(nullptr), N_proc, m);
  }

  // apply_material (applyMaterialEffects)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void apply_material(const MaterialAt<N>& m,
                                                          const MPlexHV<N>& plNrm,
                                                          MPlexLS<N>& err,
                                                          MPlexLV<N>& par,
                                                          const int N_proc,
                                                          const MPlexQF<N>* ms_ref_p = nullptr) {
    applyMaterialEffects(m.radl, m.xi, m.sign, plNrm, err, par, N_proc, ms_ref_p);
  }

  // finish: phi into [-pi, pi), the start state restored on failed lanes
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void finish(
      const TrackRef<N>& in, const MPlexLS<N>& inErr, const MPlexQI<N>& failFlag, MPlexLV<N>& par, MPlexLS<N>& err) {
    squashPhiMPlex(par, in.n_proc);  // ensure phi is between |pi|

    // PROP-FAIL-ENABLE To keep physics changes minimal, we always restore the
    // state to input when propagation fails -- as was the default before.
    for (int i = 0; i < in.n_proc; ++i) {
      if (failFlag(i, 0, 0)) {
        par.copySlot(i, in.par);
        err.copySlot(i, inErr);
      }
    }
  }

  // ---- Kalman operation on a plane (KalmanUtilsMPlex.cc, namespace plane)

  // local_field
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void local_field(const TrackRef<N>& pred,
                                                       const PropagationFlags& pf,
                                                       const bool use_param_b_field,
                                                       MPlexQF<N>& bFld,
                                                       const bool fit_b_field = false) {
    const MPlexLV<N>& psPar = pred.par;
    const Config::BFieldParams& bfp = fit_b_field ? pf.material.bFieldFit : pf.material.bField;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      bFld(n, 0, 0) =
          use_param_b_field
              ? 0.01f * Const::sol * Config::bFieldFromZR(bfp, psPar(n, 2, 0), hipo(psPar(n, 0, 0), psPar(n, 1, 0)))
              : 0.01f * Const::sol * Config::Bfield;
    }
  }

  // to_local
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void to_local(const TrackRef<N>& pred,
                                                    const MPlexLS<N>& psErr,
                                                    const PlaneRef<N>& pl,
                                                    const MPlexHV<N>& plDir,
                                                    const MPlexQF<N>& bFld,
                                                    LocalPred<N>& L) {
    using namespace kalmandetail;
    const MPlexLV<N>& psPar = pred.par;
    const MPlexQI<N>& inChg = pred.chg;
    const MPlexHV<N>& plPnt = pl.pnt;
    const MPlexHV<N>& plNrm = pl.nrm;
    MPlexHH<N>& rot = L.rot;
    MPlex2V<N>& xlo = L.xlo;
    MPlex5V<N>& lp = L.lp;
    MPlexQI<N>& pzSign = L.pz_sign;
    MPlex5S<N>& psErrLoc = L.err;

    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      rot(n, 0, 0) = plDir(n, 0, 0);
      rot(n, 0, 1) = plDir(n, 1, 0);
      rot(n, 0, 2) = plDir(n, 2, 0);
      rot(n, 1, 0) = plNrm(n, 1, 0) * plDir(n, 2, 0) - plNrm(n, 2, 0) * plDir(n, 1, 0);
      rot(n, 1, 1) = plNrm(n, 2, 0) * plDir(n, 0, 0) - plNrm(n, 0, 0) * plDir(n, 2, 0);
      rot(n, 1, 2) = plNrm(n, 0, 0) * plDir(n, 1, 0) - plNrm(n, 1, 0) * plDir(n, 0, 0);
      rot(n, 2, 0) = plNrm(n, 0, 0);
      rot(n, 2, 1) = plNrm(n, 1, 0);
      rot(n, 2, 2) = plNrm(n, 2, 0);
    }

    // get local parameters
    MPlexHV<N> xd;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      xd(n, 0, 0) = psPar(n, 0, 0) - plPnt(n, 0, 0);
      xd(n, 1, 0) = psPar(n, 1, 0) - plPnt(n, 1, 0);
      xd(n, 2, 0) = psPar(n, 2, 0) - plPnt(n, 2, 0);
    }
    RotateResidualsOnPlane<N>(rot, xd, xlo);

    MPlexQF<N> sinP, sinT, cosP, cosT, pt;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      pt(n, 0, 0) = 1.f / psPar(n, 3, 0);
      sinP(n, 0, 0) = std::sin(psPar(n, 4, 0));
      cosP(n, 0, 0) = std::cos(psPar(n, 4, 0));
      sinT(n, 0, 0) = std::sin(psPar(n, 5, 0));
      cosT(n, 0, 0) = std::cos(psPar(n, 5, 0));
    }

    MPlexHV<N> pgl;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      pgl(n, 0, 0) = cosP(n, 0, 0) * pt(n, 0, 0);
      pgl(n, 1, 0) = sinP(n, 0, 0) * pt(n, 0, 0);
      pgl(n, 2, 0) = cosT(n, 0, 0) * pt(n, 0, 0) / sinT(n, 0, 0);
    }

    MPlexHV<N> plo;
    RotateVectorOnPlane(rot, pgl, plo);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      lp(n, 0, 0) = inChg(n, 0, 0) * psPar(n, 3, 0) * sinT(n, 0, 0);
      lp(n, 1, 0) = plo(n, 0, 0) / plo(n, 2, 0);
      lp(n, 2, 0) = plo(n, 1, 0) / plo(n, 2, 0);
      lp(n, 3, 0) = xlo(n, 0, 0);
      lp(n, 4, 0) = xlo(n, 1, 0);
    }
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      pzSign(n, 0, 0) = plo(n, 2, 0) > 0.f ? 1 : -1;
    }

    //now we need the jacobian to convert from CCS to curvilinear
    // code from TrackState::jacobianCCSToCurvilinear
    MPlex56<N> jacCCS2Curv(0.f);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      jacCCS2Curv(n, 0, 3) = inChg(n, 0, 0) * sinT(n, 0, 0);
      jacCCS2Curv(n, 0, 5) = inChg(n, 0, 0) * cosT(n, 0, 0) * psPar(n, 3, 0);
      jacCCS2Curv(n, 1, 5) = -1.f;
      jacCCS2Curv(n, 2, 4) = 1.f;
      jacCCS2Curv(n, 3, 0) = -sinP(n, 0, 0);
      jacCCS2Curv(n, 3, 1) = cosP(n, 0, 0);
      jacCCS2Curv(n, 4, 0) = -cosP(n, 0, 0) * cosT(n, 0, 0);
      jacCCS2Curv(n, 4, 1) = -sinP(n, 0, 0) * cosT(n, 0, 0);
      jacCCS2Curv(n, 4, 2) = sinT(n, 0, 0);
    }

    //now we need the jacobian from curv to local
    // code from TrackingTools/AnalyticalJacobians/src/JacobianCurvilinearToLocal.cc
    MPlexHV<N> un;
    MPlexHV<N> vn;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float abslp00 = std::abs(lp(n, 0, 0));
      vn(n, 2, 0) = std::max(1.e-30f, abslp00 * pt(n, 0, 0));
      un(n, 0, 0) = -pgl(n, 1, 0) * abslp00 / vn(n, 2, 0);
      un(n, 1, 0) = pgl(n, 0, 0) * abslp00 / vn(n, 2, 0);
      un(n, 2, 0) = 0.f;
      vn(n, 0, 0) = -pgl(n, 2, 0) * abslp00 * un(n, 1, 0);
      vn(n, 1, 0) = pgl(n, 2, 0) * abslp00 * un(n, 0, 0);
    }
    MPlexHV<N> u;
    RotateVectorOnPlane(rot, un, u);
    MPlexHV<N> v;
    RotateVectorOnPlane(rot, vn, v);
    MPlex55<N> jacCurv2Loc(0.f);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float bF = bFld(n, 0, 0);
      const float qh2 = bF * lp(n, 0, 0);
      const float t1r = std::sqrt(1.f + lp(n, 1, 0) * lp(n, 1, 0) + lp(n, 2, 0) * lp(n, 2, 0)) * pzSign(n, 0, 0);
      const float t2r = t1r * t1r;
      const float t3r = t1r * t2r;
      jacCurv2Loc(n, 0, 0) = 1.f;
      jacCurv2Loc(n, 1, 1) = -u(n, 1, 0) * t2r;
      jacCurv2Loc(n, 1, 2) = v(n, 1, 0) * vn(n, 2, 0) * t2r;
      jacCurv2Loc(n, 2, 1) = u(n, 0, 0) * t2r;
      jacCurv2Loc(n, 2, 2) = -v(n, 0, 0) * vn(n, 2, 0) * t2r;
      jacCurv2Loc(n, 3, 3) = v(n, 1, 0) * t1r;
      jacCurv2Loc(n, 3, 4) = -u(n, 1, 0) * t1r;
      jacCurv2Loc(n, 4, 3) = -v(n, 0, 0) * t1r;
      jacCurv2Loc(n, 4, 4) = u(n, 0, 0) * t1r;
      const float cosz = -vn(n, 2, 0) * qh2;
      const float ui = u(n, 2, 0) * t3r;
      const float vi = v(n, 2, 0) * t3r;
      jacCurv2Loc(n, 1, 3) = -ui * v(n, 1, 0) * cosz;
      jacCurv2Loc(n, 1, 4) = -vi * v(n, 1, 0) * cosz;
      jacCurv2Loc(n, 2, 3) = ui * v(n, 0, 0) * cosz;
      jacCurv2Loc(n, 2, 4) = vi * v(n, 0, 0) * cosz;
    }

    // jacobian for converting from CCS to Loc (via Curv)
    MPlex56<N> jacCCS2Loc;
    JacCCS2Loc(jacCurv2Loc, jacCCS2Curv, jacCCS2Loc);

    // local error!
    MPlex56<N> temp56;
    PsErrLoc(jacCCS2Loc, psErr, temp56);
    PsErrLocTransp(temp56, jacCCS2Loc, psErrLoc);
  }

  // measurement
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void measurement(const LocalPred<N>& L,
                                                       const MPlexHV<N>& msPar,
                                                       const MPlexHS<N>& msErr,
                                                       const MPlexHV<N>& plPnt,
                                                       LocalMeas<N>& M) {
    using namespace kalmandetail;
    const MPlexHH<N>& rot = L.rot;
    MPlex2V<N>& mslo = M.par;
    MPlex2S<N>& msErr_loc = M.err;

    MPlexHV<N> md;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      md(n, 0, 0) = msPar(n, 0, 0) - plPnt(n, 0, 0);
      md(n, 1, 0) = msPar(n, 1, 0) - plPnt(n, 1, 0);
      md(n, 2, 0) = msPar(n, 2, 0) - plPnt(n, 2, 0);
    }
    RotateResidualsOnPlane<N>(rot, md, mslo);
    MPlex2H<N> temp2Hmsl;
    ProjectResErr<N>(rot, msErr, temp2Hmsl);
    ProjectResErrTransp<N>(rot, temp2Hmsl, msErr_loc);
  }

  // residual (with det of the residual covariance before the inversion)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void residual(const LocalPred<N>& L, const LocalMeas<N>& M, Residual<N>& R) {
    const MPlex2V<N>& xlo = L.xlo;
    const MPlex5S<N>& psErrLoc = L.err;
    const MPlex2V<N>& mslo = M.par;
    const MPlex2S<N>& msErr_loc = M.err;
    MPlex2V<N>& res_loc = R.r;
    MPlex2S<N>& resErr_loc = R.s_inv;

    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      res_loc(n, 0, 0) = mslo(n, 0, 0) - xlo(n, 0, 0);
      res_loc(n, 1, 0) = mslo(n, 1, 0) - xlo(n, 1, 0);
    }

    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      resErr_loc(n, 0, 0) = psErrLoc(n, 3, 3) + msErr_loc(n, 0, 0);
      resErr_loc(n, 0, 1) = psErrLoc(n, 3, 4) + msErr_loc(n, 0, 1);
      resErr_loc(n, 1, 1) = psErrLoc(n, 4, 4) + msErr_loc(n, 1, 1);
    }
    //invert the 2x2 matrix, keeping the determinant (Cramer computes it in double either way)
    double determ[N];
    kalmandetail::invertCramerSym(resErr_loc, determ);
    for (int n = 0; n < N; ++n)
      R.det[n] = (float)determ[n];
  }

  // chi2
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void chi2(const Residual<N>& R, MPlexQF<N>& outChi2) {
    kalmandetail::Chi2Similarity(R.r, R.s_inv, outChi2);
  }

  // chi2_scaled: the track part of the residual covariance scaled by trk_scale2
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void chi2_scaled(
      const LocalPred<N>& L, const LocalMeas<N>& M, const Residual<N>& R, const float trk_scale2, MPlexQF<N>& out) {
    const MPlex5S<N>& psErrLoc = L.err;
    const MPlex2S<N>& msErr_loc = M.err;
    MPlex2S<N> accErr_loc;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      accErr_loc(n, 0, 0) = trk_scale2 * psErrLoc(n, 3, 3) + msErr_loc(n, 0, 0);
      accErr_loc(n, 0, 1) = trk_scale2 * psErrLoc(n, 3, 4) + msErr_loc(n, 0, 1);
      accErr_loc(n, 1, 1) = trk_scale2 * psErrLoc(n, 4, 4) + msErr_loc(n, 1, 1);
    }
    kalmandetail::invertCramerSym(accErr_loc);
    kalmandetail::Chi2Similarity(R.r, accErr_loc, out);
  }

  // update
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void update(const LocalPred<N>& L, const Residual<N>& R, LocalUpd<N>& U) {
    const MPlex5V<N>& lp = L.lp;
    const MPlex5S<N>& psErrLoc = L.err;
    const MPlex2V<N>& res_loc = R.r;
    const MPlex2S<N>& resErr_loc = R.s_inv;
    MPlex5V<N>& lp_upd = U.lp;
    MPlex5S<N>& psErrLoc_upd = U.err;

    MPlex52<N> K;  // kalman gain
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      for (int j = 0; j < 5; ++j) {
        K(n, j, 0) = resErr_loc(n, 0, 0) * psErrLoc(n, j, 3) + resErr_loc(n, 0, 1) * psErrLoc(n, j, 4);
        K(n, j, 1) = resErr_loc(n, 0, 1) * psErrLoc(n, j, 3) + resErr_loc(n, 1, 1) * psErrLoc(n, j, 4);
      }
    }

    kalmandetail::MultResidualsAdd(K, lp, res_loc, lp_upd);

    MPlex55<N> ImKH(0.f);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      for (int j = 0; j < 5; ++j) {
        ImKH(n, j, j) = 1.f;
        ImKH(n, j, 3) -= K(n, j, 0);
        ImKH(n, j, 4) -= K(n, j, 1);
      }
    }
    kalmandetail::PsErrLocUpd(ImKH, psErrLoc, psErrLoc_upd);
  }

  // to_global
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void to_global(const LocalUpd<N>& U,
                                                     const LocalPred<N>& L,
                                                     const PlaneRef<N>& pl,
                                                     const MPlexQI<N>& inChg,
                                                     const MPlexQF<N>& bFld,
                                                     MPlexLV<N>& outPar,
                                                     MPlexLS<N>& outErr) {
    using namespace kalmandetail;
    const MPlexHV<N>& plPnt = pl.pnt;
    const MPlexHH<N>& rot = L.rot;
    const MPlex5V<N>& lp = L.lp;
    const MPlexQI<N>& pzSign = L.pz_sign;
    const MPlex5V<N>& lp_upd = U.lp;
    const MPlex5S<N>& psErrLoc_upd = U.err;
    MPlexQF<N> sinP, sinT, cosP, cosT, pt;
    MPlexHV<N> un, vn;

    //convert local updated parameters into CCS
    MPlexHV<N> lxu;
    MPlexHV<N> lpu;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      lxu(n, 0, 0) = lp_upd(n, 3, 0);
      lxu(n, 1, 0) = lp_upd(n, 4, 0);
      lxu(n, 2, 0) = 0.f;
      lpu(n, 2, 0) =
          pzSign(n, 0, 0) / (std::max(std::abs(lp_upd(n, 0, 0)), 1.e-9f) *
                             std::sqrt(1.f + lp_upd(n, 1, 0) * lp_upd(n, 1, 0) + lp_upd(n, 2, 0) * lp_upd(n, 2, 0)));
      lpu(n, 0, 0) = lpu(n, 2, 0) * lp_upd(n, 1, 0);
      lpu(n, 1, 0) = lpu(n, 2, 0) * lp_upd(n, 2, 0);
    }
    MPlexHV<N> gxu;
    RotateVectorOnPlaneTransp(rot, lxu, gxu);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      gxu(n, 0, 0) += plPnt(n, 0, 0);
      gxu(n, 1, 0) += plPnt(n, 1, 0);
      gxu(n, 2, 0) += plPnt(n, 2, 0);
    }
    MPlexHV<N> gpu;
    RotateVectorOnPlaneTransp(rot, lpu, gpu);

    MPlexQF<N> p;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      pt(n, 0, 0) = std::sqrt(gpu.At(n, 0, 0) * gpu.At(n, 0, 0) + gpu.At(n, 1, 0) * gpu.At(n, 1, 0));
      p(n, 0, 0) = std::sqrt(pt.At(n, 0, 0) * pt.At(n, 0, 0) + gpu.At(n, 2, 0) * gpu.At(n, 2, 0));
      sinP(n, 0, 0) = gpu.At(n, 1, 0) / pt(n, 0, 0);
      cosP(n, 0, 0) = gpu.At(n, 0, 0) / pt(n, 0, 0);
      sinT(n, 0, 0) = pt(n, 0, 0) / p(n, 0, 0);
      cosT(n, 0, 0) = gpu.At(n, 2, 0) / p(n, 0, 0);
    }

    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      outPar(n, 0, 0) = gxu.At(n, 0, 0);
      outPar(n, 1, 0) = gxu.At(n, 1, 0);
      outPar(n, 2, 0) = gxu.At(n, 2, 0);
      outPar(n, 3, 0) = 1.f / pt(n, 0, 0);
      outPar(n, 4, 0) = getPhi(gpu.At(n, 0, 0), gpu.At(n, 1, 0));  //fixme VDT or something?
      outPar(n, 5, 0) = getTheta(pt(n, 0, 0), gpu.At(n, 2, 0));
    }

    //now we need the jacobian to convert from curvilinear to CCS
    // code from TrackState::jacobianCurvilinearToCCS
    MPlex65<N> jacCurv2CCS(0.f);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      jacCurv2CCS(n, 0, 3) = -sinP(n, 0, 0);
      jacCurv2CCS(n, 0, 4) = -cosT(n, 0, 0) * cosP(n, 0, 0);
      jacCurv2CCS(n, 1, 3) = cosP(n, 0, 0);
      jacCurv2CCS(n, 1, 4) = -cosT(n, 0, 0) * sinP(n, 0, 0);
      jacCurv2CCS(n, 2, 4) = sinT(n, 0, 0);
      jacCurv2CCS(n, 3, 0) = inChg(n, 0, 0) / sinT(n, 0, 0);
      jacCurv2CCS(n, 3, 1) = outPar(n, 3, 0) * cosT(n, 0, 0) / sinT(n, 0, 0);
      jacCurv2CCS(n, 4, 2) = 1.f;
      jacCurv2CCS(n, 5, 1) = -1.f;
      if (std::signbit(lp_upd(n, 0, 0)) != std::signbit(lp(n, 0, 0))) {
        outPar(n, 3, 0) = -outPar(n, 3, 0);
        jacCurv2CCS(n, 3, 0) = -jacCurv2CCS(n, 3, 0);
      }
    }

    //now we need the jacobian from local to curv
    // code from TrackingTools/AnalyticalJacobians/src/JacobianLocalToCurvilinear.cc
    MPlexHV<N> tnl;
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float abslpupd00 = std::max(std::abs(lp_upd(n, 0, 0)), 1.e-9f);
      tnl(n, 0, 0) = lpu(n, 0, 0) * abslpupd00;
      tnl(n, 1, 0) = lpu(n, 1, 0) * abslpupd00;
      tnl(n, 2, 0) = lpu(n, 2, 0) * abslpupd00;
    }
    MPlexHV<N> tn;
    RotateVectorOnPlaneTransp(rot, tnl, tn);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      vn(n, 2, 0) = std::max(1.e-30f, std::sqrt(tn(n, 0, 0) * tn(n, 0, 0) + tn(n, 1, 0) * tn(n, 1, 0)));
      un(n, 0, 0) = -tn(n, 1, 0) / vn(n, 2, 0);
      un(n, 1, 0) = tn(n, 0, 0) / vn(n, 2, 0);
      un(n, 2, 0) = 0.f;
      vn(n, 0, 0) = -tn(n, 2, 0) * un(n, 1, 0);
      vn(n, 1, 0) = tn(n, 2, 0) * un(n, 0, 0);
    }
    MPlex55<N> jacLoc2Curv(0.f);
    MPLEX_SIMD
    for (int n = 0; n < N; ++n) {
      const float bF = bFld(n, 0, 0);
      const float qh2 = bF * lp_upd(n, 0, 0);
      const float cosl1 = 1.f / vn(n, 2, 0);
      const float uj = un(n, 0, 0) * rot(n, 0, 0) + un(n, 1, 0) * rot(n, 0, 1);
      const float uk = un(n, 0, 0) * rot(n, 1, 0) + un(n, 1, 0) * rot(n, 1, 1);
      const float vj = vn(n, 0, 0) * rot(n, 0, 0) + vn(n, 1, 0) * rot(n, 0, 1) + vn(n, 2, 0) * rot(n, 0, 2);
      const float vk = vn(n, 0, 0) * rot(n, 1, 0) + vn(n, 1, 0) * rot(n, 1, 1) + vn(n, 2, 0) * rot(n, 1, 2);
      const float cosz = vn(n, 2, 0) * qh2;
      jacLoc2Curv(n, 0, 0) = 1.f;
      jacLoc2Curv(n, 1, 1) = tnl(n, 2, 0) * vj;
      jacLoc2Curv(n, 1, 2) = tnl(n, 2, 0) * vk;
      jacLoc2Curv(n, 2, 1) = tnl(n, 2, 0) * uj * cosl1;
      jacLoc2Curv(n, 2, 2) = tnl(n, 2, 0) * uk * cosl1;
      jacLoc2Curv(n, 3, 3) = uj;
      jacLoc2Curv(n, 3, 4) = uk;
      jacLoc2Curv(n, 4, 3) = vj;
      jacLoc2Curv(n, 4, 4) = vk;
      jacLoc2Curv(n, 2, 3) = tnl(n, 0, 0) * (cosz * cosl1);
      jacLoc2Curv(n, 2, 4) = tnl(n, 1, 0) * (cosz * cosl1);
    }

    // jacobian for converting from Loc to CCS (via Curv)
    MPlex65<N> jacLoc2CCS;
    JacLoc2CCS(jacCurv2CCS, jacLoc2Curv, jacLoc2CCS);

    // CCS error!
    MPlex65<N> temp65;
    OutErrCCS(jacLoc2CCS, psErrLoc_upd, temp65);
    OutErrCCSTransp(temp65, jacLoc2CCS, outErr);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::plane

#endif
