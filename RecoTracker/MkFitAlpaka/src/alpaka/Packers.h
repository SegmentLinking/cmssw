#ifndef RecoTracker_MkFitAlpaka_src_alpaka_Packers_h
#define RecoTracker_MkFitAlpaka_src_alpaka_Packers_h

// MPlex <-> SoA packers of the engine and the backward fit: the per-slot bodies of MkFitCore's copy-in, same element
// order, no arithmetic (MkFinder::copy_in(TrackCand), m_msErr / m_msPar copyIn, MkFinder::packModuleNormDirPnt).
// slot = slot in the Matriplex (0 on GPU backends where N = 1).

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/Matrix.h"

namespace mkfitdev::pack {

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadCandState(
      CandState const& cand, int slot, MPlexLS<N>& err, MPlexLV<N>& par, MPlexQI<N>& chg) {
    // direct element stores: the same values as copyIn, but the store form changes the codegen at rounding level
    for (int i = 0; i < 6; ++i)
      par.At(slot, i, 0) = cand.par[i];
    for (int i = 0; i < 21; ++i)
      err.fArray[i * N + slot] = cand.err[i];  // MatriplexSym packed lower triangle == CandState.err order
    chg.At(slot, 0, 0) = cand.charge;
  }

  // hit row in the HitSoA columns: pixel layers wrapper index, strip layers nPixel + wrapper index
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int hitRow(EngineHitInputs const& in, int layer, int index) {
    return (in.layers[layer].isPixel ? 0 : int(in.nPixel)) + index;
  }

  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadHit(
      EngineHitInputs const& in, int row, int slot, MPlexHS<N>& msErr, MPlexHV<N>& msPar) {
    msPar.At(slot, 0, 0) = in.x[row];
    msPar.At(slot, 1, 0) = in.y[row];
    msPar.At(slot, 2, 0) = in.z[row];
    // MkFitCore Hit error order (e00, e10, e11, e20, e21, e22) == MatriplexSym packed order
    msErr.fArray[0 * N + slot] = in.e00[row];
    msErr.fArray[1 * N + slot] = in.e10[row];
    msErr.fArray[2 * N + slot] = in.e11[row];
    msErr.fArray[3 * N + slot] = in.e20[row];
    msErr.fArray[4 * N + slot] = in.e21[row];
    msErr.fArray[5 * N + slot] = in.e22[row];
  }

  // module of the hit in `row` on `layer`: EngineModule row = layers[layer].moduleBegin + detIDinLayer(packed)
  template <int N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadModulePlane(
      EngineHitInputs const& in, int layer, int row, int slot, MPlexHV<N>& norm, MPlexHV<N>& dir, MPlexHV<N>& pnt) {
    const EngineModule& plane = in.modules[in.layers[layer].moduleBegin + int(hitpack::detIDinLayer(in.packed[row]))];
    for (int i = 0; i < 3; ++i) {
      norm.At(slot, i, 0) = plane.nrm[i];
      dir.At(slot, i, 0) = plane.dir[i];
      pnt.At(slot, i, 0) = plane.pnt[i];
    }
  }

}  // namespace mkfitdev::pack

#endif
