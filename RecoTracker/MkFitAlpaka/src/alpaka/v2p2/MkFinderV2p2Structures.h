#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkFinderV2p2Structures_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkFinderV2p2Structures_h

// Portable copy of RecoTracker/MkFitCore/src/MkFinderV2p2Structures.h / .cc. The per-candidate and per-node storage
// (CandIncoming, PassedHitMatch, CandNode, CCandRep) is the SoA of interface/v2p2/V2p2SoA.h; this file has what
// operates on it: the bounded pqueue of CandIncoming, the TrackCand accessors the finder reads, and the Kalman batch
// of the in-layer search (BaseArgs + KalmanOpArgs, do_kalman_stuff). Templated on N items like src/alpaka/prop
// (N = 1 per thread on GPU backends, kNN rows per thread on CPU backends).

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexBackend.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2SoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/Packers.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PlaneSteps.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationFlags.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/MiniPropagators.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using ::mkfitdev::CandBook;
  using ::mkfitdev::CandState;
  using ::mkfitdev::EngineHitInputs;
  using ::mkfitdev::SeedCandsRef;
  using ::mkfitdev::V2p2InitialState;
  using ::mkfitdev::V2p2PassedHitMatch;
  using ::mkfitdev::V2p2TrackState;

  //----------------------------------------------------------------------------
  // CandIncoming::pq_push / pq_pop: a max-heap on ddphi (PassedHitMatch::operator<, worst on top) in pq[0, size),
  // with exactly the steps of libstdc++'s std::push_heap / std::pop_heap (__push_heap, __adjust_heap), so that the
  // kept set equals MkFitCore's for the same arrival order, ties included. A heap operation, not a sort.

  // libstdc++ std::__push_heap with PassedHitMatch::operator< (ddphi)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void pq_push_hole(V2p2PassedHitMatch* pq,
                                                        int holeIndex,
                                                        int topIndex,
                                                        const V2p2PassedHitMatch& value) {
    int parent = (holeIndex - 1) / 2;
    while (holeIndex > topIndex && pq[parent].ddphi < value.ddphi) {
      pq[holeIndex] = pq[parent];
      holeIndex = parent;
      parent = (holeIndex - 1) / 2;
    }
    pq[holeIndex] = value;
  }

  // pq_push: pq[size] = e, then std::push_heap(pq, pq + size + 1); returns size + 1
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int pq_push(V2p2PassedHitMatch* pq, int size, const V2p2PassedHitMatch& e) {
    pq_push_hole(pq, size, 0, e);
    return size + 1;
  }

  // pq_top
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const V2p2PassedHitMatch& pq_top(const V2p2PassedHitMatch* pq) { return pq[0]; }

  // pq_pop: std::pop_heap(pq, pq + size), then pop_back; returns size - 1
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int pq_pop(V2p2PassedHitMatch* pq, int size) {
    if (size > 1) {
      // std::__pop_heap: the last element is re-inserted from the root by std::__adjust_heap
      const int len = size - 1;
      const V2p2PassedHitMatch value = pq[len];
      pq[len] = pq[0];
      int holeIndex = 0;
      int secondChild = 0;
      while (secondChild < (len - 1) / 2) {
        secondChild = 2 * (secondChild + 1);
        if (pq[secondChild].ddphi < pq[secondChild - 1].ddphi)
          --secondChild;
        pq[holeIndex] = pq[secondChild];
        holeIndex = secondChild;
      }
      if ((len & 1) == 0 && secondChild == (len - 2) / 2) {
        secondChild = 2 * (secondChild + 1);
        pq[holeIndex] = pq[secondChild - 1];
        holeIndex = secondChild - 1;
      }
      pq_push_hole(pq, holeIndex, 0, value);
    }
    return size - 1;
  }

  //----------------------------------------------------------------------------
  // The TrackCand / TrackState accessors MkFinderV2p2 reads, on one candidate slot of a seed. posRsq() as MkFitCore's
  // x86-64-v3 build contracts it, fma(x, x, y * y) (the clone engine's K1 does the same).
  struct TrackCandView {
    const CandBook* book;
    const CandState* state;
    const SeedCandsRef* seed;  // for getLastHitIdx(): the HoT node book->lastHitIdx

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float x() const { return state->par[0]; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float y() const { return state->par[1]; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float z() const { return state->par[2]; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float posR() const { return std::sqrt(posRsq()); }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float posRsq() const { return std::fma(x(), x(), y() * y()); }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float posPhi() const { return ::mkfitdev::getPhi(x(), y()); }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float momPhi() const { return state->par[4]; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float pT() const { return std::abs(1.f / state->par[3]); }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float pz() const { return pT() / std::tan(state->par[5]); }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float p() const { return pT() / std::sin(state->par[5]); }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int charge() const { return state->charge; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int nAllMinusOneHits() const {
      return book->nInsideMinusOne + book->nTailMinusOne;
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int nTailMinusOneHits() const { return book->nTailMinusOne; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int getLastHitIdx() const { return seed->hot(book->lastHitIdx).index; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float score() const { return book->score; }
  };

  //----------------------------------------------------------------------------
  // BaseArgs + KalmanOpArgs: one Kalman batch of the in-layer search, N (candidate, hit) items. The constructor
  // zeroes every Matriplex as MkFitCore's member initializers do. The item outcome is read from tsChi2, tsDetV,
  // tsChi2Acc and the updated tsPar / tsErr / tsChg of its slot (KalmanOpArgs::ItemOut).
  template <idx_t N>
  struct KalmanOpArgs {
    // ---- BaseArgs
    mini_propagators::InitialStatePlex<N> tsXyz;
    MPlexLS<N> tsErr;
    MPlexLV<N> tsPar;
    MPlexQI<N> tsChg;
    MPlexQF<N> sPerp;
    MPlexLS<N> propErr;
    MPlexLV<N> propPar;
    MPlexQI<N> outFailFlag;
    int N_filled;

    // ---- KalmanOpArgs: where each lane came from (candp = seed row + TrackCand index; hot = hit index + layer)
    int cand_seed[N];
    int cand_ic[N];
    int hot_index[N];
    int hot_layer[N];
    int parent_idx[N];
    int hit_pos[N];

    MPlexHS<N> msErr;
    MPlexHV<N> msPar;
    MPlexHV<N> plNrm;
    MPlexHV<N> plDir;
    MPlexHV<N> plPnt;

    MPlexQF<N> tsChi2;
    MPlexQF<N> tsDetV;
    MPlexQF<N> tsChi2Acc;

    bool m_solve_plane;
    float m_chi2_trk_fac;

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE KalmanOpArgs(bool solve_plane, float chi2_trk_fac)
        : tsErr(0.0f),
          tsPar(0.0f),
          tsChg(0),
          sPerp(0.0f),
          propErr(0.0f),
          propPar(0.0f),
          outFailFlag(0),
          N_filled(0),
          msErr(0.0f),
          msPar(0.0f),
          plNrm(0.0f),
          plDir(0.0f),
          plPnt(0.0f),
          tsChi2(0.0f),
          tsDetV(0.0f),
          tsChi2Acc(0.0f),
          m_solve_plane(solve_plane),
          m_chi2_trk_fac(chi2_trk_fac) {
      for (int i = 0; i < N; ++i) {
        tsXyz.x[i] = tsXyz.y[i] = tsXyz.z[i] = 0.0f;
        tsXyz.px[i] = tsXyz.py[i] = tsXyz.pz[i] = 0.0f;
        tsXyz.dalpha[i] = 0.0f;
        tsXyz.fail_flag[i] = 0;
        tsXyz.inv_pt[i] = tsXyz.inv_k[i] = tsXyz.theta[i] = 0.0f;
        cand_seed[i] = cand_ic[i] = hot_index[i] = hot_layer[i] = parent_idx[i] = hit_pos[i] = -1;
      }
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void reset() { N_filled = 0; }

    // item_begin(cand, hot, par_idx, hpos)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void item_begin(
        int seed, int ic, int hit_index, int hit_layer, int par_idx, int hpos) {
      cand_seed[N_filled] = seed;
      cand_ic[N_filled] = ic;
      hot_index[N_filled] = hit_index;
      hot_layer[N_filled] = hit_layer;
      parent_idx[N_filled] = par_idx;
      hit_pos[N_filled] = hpos;
    }

    // item_finished(): ++N_filled == N
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool item_finished() { return ++N_filled == N; }

    // Start from a TrackState directly, for the solve-plane path: there is no
    // pre-solved crossing, so tsXyz is not filled and compute_pars() must be
    // skipped -- propPar is a pure output there, where on the sPerp path it is
    // an input carrying the already-known intersection.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void load_state_err_chg(const V2p2TrackState& st) {
      tsPar.copyIn(N_filled, st.par);
      tsErr.copyIn(N_filled, st.err);
      tsChg[N_filled] = st.charge;
    }

    // load_state_err_chg(params_on_hit, prev_state): the depth-0 path, the search's crossing and the TrackCand state
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void load_state_err_chg(const V2p2InitialState& params_on_hit,
                                                                const CandState& prev_state) {
      tsXyz.x[N_filled] = params_on_hit.x;
      tsXyz.y[N_filled] = params_on_hit.y;
      tsXyz.z[N_filled] = params_on_hit.z;
      tsXyz.px[N_filled] = params_on_hit.px;
      tsXyz.py[N_filled] = params_on_hit.py;
      tsXyz.pz[N_filled] = params_on_hit.pz;
      tsXyz.dalpha[N_filled] = params_on_hit.dalpha;
      tsXyz.fail_flag[N_filled] = params_on_hit.fail_flag;
      tsXyz.inv_pt[N_filled] = params_on_hit.inv_pt;
      tsXyz.inv_k[N_filled] = params_on_hit.inv_k;
      tsXyz.theta[N_filled] = params_on_hit.theta;
      tsPar.copyIn(N_filled, prev_state.par);  // the propagation steps need the initial parameters, too
      tsErr.copyIn(N_filled, prev_state.err);
      tsChg[N_filled] = prev_state.charge;
    }

    // load_hit_module(hit, module_info): hit position / errors and the module plane of hit (layer, hit_orig_index);
    // returns the hit's detIDinLayer
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int load_hit_module(const EngineHitInputs& hits,
                                                            int layer,
                                                            uint32_t hit_orig_index) {
      const int row = ::mkfitdev::pack::hitRow(hits, layer, int(hit_orig_index));
      ::mkfitdev::pack::loadHit<N>(hits, row, N_filled, msErr, msPar);
      ::mkfitdev::pack::loadModulePlane<N>(hits, layer, row, N_filled, plNrm, plDir, plPnt);
      return int(::mkfitdev::hitpack::detIDinLayer(hits.packed[row]));
    }

    // compute_pars()
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void compute_pars() {
      // The search's crossing on the module plane (tsXyz) becomes propPar and the
      // transverse path length sPerp; tsPar keeps the parameters at the previous
      // hit, which the jacobian needs. The crossing is the Hermite cubic solved
      // onto the plane, not a helix solution.
      for (int i = 0; i < N; ++i) {
        propPar.At(i, 0, 0) = tsXyz.x[i];
        propPar.At(i, 1, 0) = tsXyz.y[i];
        propPar.At(i, 2, 0) = tsXyz.z[i];
        propPar.At(i, 3, 0) = tsXyz.inv_pt[i];
        propPar.At(i, 4, 0) = vdt::fast_atan2f(tsXyz.py[i], tsXyz.px[i]);
        propPar.At(i, 5, 0) = tsXyz.theta[i];

        sPerp[i] = tsXyz.dalpha[i] / (tsXyz.inv_pt[i] * tsXyz.inv_k[i]);
      }
    }

    // do_kalman_stuff() (MkFinderV2p2Structures.cc): inter = finding_inter_layer_pflags, intra =
    // finding_intra_layer_pflags (taken when m_solve_plane)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void do_kalman_stuff(const PropagationFlags& inter,
                                                             const PropagationFlags& intra);

    // the updated state of slot i (ItemOut::state)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void item_state(int i, V2p2TrackState& state) const {
      tsPar.copyOut(i, state.par);
      tsErr.copyOut(i, state.err);
      state.charge = tsChg[i];
    }
  };

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void KalmanOpArgs<N>::do_kalman_stuff(const PropagationFlags& inter,
                                                                            const PropagationFlags& intra) {
    // Propagation to the module plane and the Kalman operation on it, composed from the steps of
    // PlaneSteps.h. m_solve_plane marks steps within one layer (second and later hits of an in-layer
    // path): they take the intra-layer propagation flags and solve for the path to the plane.
    // Otherwise the search has already put the parameters on the plane (propPar, compute_pars())
    // and supplies the transverse path length sPerp, so only the covariance is transported.
    using namespace plane;

    const PropagationFlags& pf = m_solve_plane ? intra : inter;
    const TrackRef<N> in{tsPar, tsChg, N_filled};
    const PlaneRef<N> pl{plPnt, plNrm};

    FieldAt<N> f;
    field_at_start(in, pf, f);
    StartTrig<N> t;
    start_trig(in, t);
    MPlexQF<N> s;
    if (m_solve_plane) {
      PathSolve<N> p;
      path_solve(in, t, pl, f, p);
      s = p.s;
      drift(in, t, f, s, propPar);
    } else {
      path_from_perp(in, t, sPerp, s);
    }
    MPlexLL<N> J(0.0f);
    jacobian(in, t, propPar, f, s, J);
    transport_cov(J, tsErr, propErr);
    if (pf.apply_material) {
      MaterialAt<N> m;
      material_grid<N>(pf, propPar, nullptr, N_filled, m);
      eloss_sign_from_path<N>(s, nullptr, N_filled, m);
      apply_material(m, plNrm, propErr, propPar, N_filled);
    }
    outFailFlag.setVal(0);
    finish(in, tsErr, outFailFlag, propPar, propErr);

    const float acc_scale2 = m_chi2_trk_fac * m_chi2_trk_fac;
    const bool acc_separate = acc_scale2 != 1.0f;

    const TrackRef<N> pred{propPar, tsChg, N_filled};
    MPlexQF<N> bFld;
    local_field(pred, pf, false, bFld);
    LocalPred<N> L;
    to_local(pred, propErr, pl, plDir, bFld, L);
    LocalMeas<N> M;
    measurement(L, msPar, msErr, plPnt, M);
    Residual<N> R;
    residual(L, M, R);
    chi2(R, tsChi2);
    // det V is read only by the likelihood score.
    tsDetV = R.det;
    if (acc_separate)
      chi2_scaled(L, M, R, acc_scale2, tsChi2Acc);
    else
      tsChi2Acc = tsChi2;
    LocalUpd<N> U;
    update(L, R, U);
    to_global(U, L, pl, tsChg, bFld, tsPar, tsErr);
    kalmanCheckChargeFlip(tsPar, tsChg, N_filled);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
