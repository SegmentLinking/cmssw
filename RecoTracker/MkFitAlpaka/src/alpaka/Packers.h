#ifndef RecoTracker_MkFitAlpaka_src_alpaka_Packers_h
#define RecoTracker_MkFitAlpaka_src_alpaka_Packers_h

// MPlex <-> SoA packers shared by every kernel that runs the MkFitCore Matriplex code (select, engine K3/K5, bkfit,
// fit). Each one is the per-slot body of a MkFitCore copy-in/copy-out, same element order, no arithmetic:
//   loadCandState / storeCandState   MkFinder::copy_in / copy_out(TrackCand): m_Err, m_Par, m_Chg (MkFinder.h:213-251)
//   loadTrack / storeTrack           MkFinder/MkFitter copy_in / copy_out(Track): m_Err, m_Par, m_Chg, m_Chi2
//   loadHit                          MkFinder/MkFitter: m_msErr.copyIn(n, hit.errArray()); m_msPar.copyIn(n, hit.posArray())
//   loadModulePlane / zeroModulePlane MkFinder::packModuleNormDirPnt (MkFinder.cc:242-263): norm = zdir, dir = xdir,
//                                    pnt = pos of module_info(hit.detIDinLayer()), zeros for an unused slot
// Two input forms of the hit / module loaders:
//   HitSoAConstView + ESView                    (fit, select)
//   EngineHitInputs (column pointers + tables)  (engine K3a/K5, bkfit); row = hitRow(in, layer, index)
// n = slot in the Matriplex (0 on GPU backends where N = 1). Error matrices are the packed lower triangle in both
// worlds (CandState.err[21], TrackSoA errors[21], HitSoA e00 e10 e11 e20 e21 e22, MatriplexSym fArray order).

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/Matrix.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace mkfitdev::pack {

  // ---- candidates (cands CandState)
  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadCandState(
      CandState const& c, int n, MPlexLS<N>& err, MPlexLV<N>& par, MPlexQI<N>& chg) {
    // direct element stores in the engine's former order (par, err, charge): the same values as copyIn, but the
    // store form changed GPU/CPU codegen of the inlined Kalman code at rounding level
    for (int i = 0; i < 6; ++i)
      par.At(n, i, 0) = c.par[i];
    for (int i = 0; i < 21; ++i)
      err.fArray[i * N + n] = c.err[i];  // MatriplexSym packed lower triangle == CandState.err order
    chg.At(n, 0, 0) = c.charge;
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void storeCandState(
      MPlexLS<N> const& err, MPlexLV<N> const& par, MPlexQI<N> const& chg, int n, CandState& c) {
    err.copyOut(n, c.err);
    par.copyOut(n, c.par);
    c.charge = chg.constAt(n, 0, 0);
  }

  // ---- tracks (clean TrackSoA row)
  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadTrack(
      TrackSoAConstView t, int row, int n, MPlexLS<N>& err, MPlexLV<N>& par, MPlexQI<N>& chg, MPlexQF<N>& chi2) {
    err.copyIn(n, t[row].errors().v);
    par.copyIn(n, t[row].params().v);
    chg(n, 0, 0) = t[row].charge();
    chi2(n, 0, 0) = t[row].chi2();
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void storeTrack(MPlexLS<N> const& err,
                                                      MPlexLV<N> const& par,
                                                      MPlexQI<N> const& chg,
                                                      MPlexQF<N> const& chi2,
                                                      int n,
                                                      TrackSoAView t,
                                                      int row) {
    err.copyOut(n, t[row].errors().v);
    par.copyOut(n, t[row].params().v);
    t[row].charge() = chg.constAt(n, 0, 0);
    t[row].chi2() = chi2.constAt(n, 0, 0);
  }

  // ---- hits (hits HitSoA row = hitBase of the wrapper + MkFitCore hit index)
  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadHit(
      HitSoAConstView h, uint32_t row, int n, MPlexHS<N>& msErr, MPlexHV<N>& msPar) {
    // direct element stores (the final fit's former form; see loadCandState)
    msPar.At(n, 0, 0) = h[row].x();
    msPar.At(n, 1, 0) = h[row].y();
    msPar.At(n, 2, 0) = h[row].z();
    msErr.At(n, 0, 0) = h[row].e00();
    msErr.At(n, 1, 0) = h[row].e10();
    msErr.At(n, 1, 1) = h[row].e11();
    msErr.At(n, 2, 0) = h[row].e20();
    msErr.At(n, 2, 1) = h[row].e21();
    msErr.At(n, 2, 2) = h[row].e22();
  }

  // ---- module plane of a hit (es ModuleInfo row of (layer, hit.detIDinLayer()))
  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadModulePlane(
      ESView const& es, int layer, uint32_t packed, int n, MPlexHV<N>& norm, MPlexHV<N>& dir, MPlexHV<N>& pnt) {
    const auto m = es.modules[es.moduleRow(layer, static_cast<int>(hitpack::detIDinLayer(packed)))];
    norm.At(n, 0, 0) = m.zdir_x();
    norm.At(n, 1, 0) = m.zdir_y();
    norm.At(n, 2, 0) = m.zdir_z();
    dir.At(n, 0, 0) = m.xdir_x();
    dir.At(n, 1, 0) = m.xdir_y();
    dir.At(n, 2, 0) = m.xdir_z();
    pnt.At(n, 0, 0) = m.pos_x();
    pnt.At(n, 1, 0) = m.pos_y();
    pnt.At(n, 2, 0) = m.pos_z();
  }

  // ---- hits and module planes from the engine's input form (EngineHitInputs, interface/cands/CandEngineTypes.h)
  // Hit row in the HitSoA columns: pixel layers wrapper index, strip layers nPixel + wrapper index.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int hitRow(EngineHitInputs const& in, int layer, int index) {
    return (in.layers[layer].isPixel ? 0 : int(in.nPixel)) + index;
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadHit(
      EngineHitInputs const& in, int row, int n, MPlexHS<N>& msErr, MPlexHV<N>& msPar) {
    // direct element stores (the engine's and bkfit's former form; see loadCandState)
    msPar.At(n, 0, 0) = in.x[row];
    msPar.At(n, 1, 0) = in.y[row];
    msPar.At(n, 2, 0) = in.z[row];
    // MkFitCore Hit error order (e00, e10, e11, e20, e21, e22) == MatriplexSym packed order
    msErr.fArray[0 * N + n] = in.e00[row];
    msErr.fArray[1 * N + n] = in.e10[row];
    msErr.fArray[2 * N + n] = in.e11[row];
    msErr.fArray[3 * N + n] = in.e20[row];
    msErr.fArray[4 * N + n] = in.e21[row];
    msErr.fArray[5 * N + n] = in.e22[row];
  }

  // module of the hit in `row` on `layer`: EngineModule row = layers[layer].moduleBegin + detIDinLayer(packed)
  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadModulePlane(
      EngineHitInputs const& in, int layer, int row, int n, MPlexHV<N>& norm, MPlexHV<N>& dir, MPlexHV<N>& pnt) {
    const EngineModule& m = in.modules[in.layers[layer].moduleBegin + int(hitpack::detIDinLayer(in.packed[row]))];
    for (int i = 0; i < 3; ++i) {
      norm.At(n, i, 0) = m.nrm[i];
      dir.At(n, i, 0) = m.dir[i];
      pnt.At(n, i, 0) = m.pnt[i];
    }
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void zeroModulePlane(int n, MPlexHV<N>& norm, MPlexHV<N>& dir, MPlexHV<N>& pnt) {
    for (int i = 0; i < 3; ++i) {
      norm(n, i, 0) = 0.0f;
      dir(n, i, 0) = 0.0f;
      pnt(n, i, 0) = 0.0f;
    }
  }

}  // namespace mkfitdev::pack

#endif
