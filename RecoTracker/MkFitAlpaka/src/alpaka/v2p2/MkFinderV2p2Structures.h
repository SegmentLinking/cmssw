#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkFinderV2p2Structures_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkFinderV2p2Structures_h

// Portable copy of RecoTracker/MkFitCore/src/MkFinderV2p2Structures.h / .cc. The per-candidate and per-node storage
// (CandIncoming, PassedHitMatch, CandNode, CCandRep) is the SoA of interface/v2p2/V2p2SoA.h; this file has what
// operates on it: the bounded pqueue of CandIncoming, the TrackCand accessors the finder reads, and the Kalman batch
// of the in-layer search (BaseArgs + KalmanOpArgs, do_kalman_stuff). Templated on N items like src/alpaka/prop
// (N = 1 per thread on GPU backends, kNN rows per thread on CPU backends).

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexBackend.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2SoA.h"
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

  // pq_push: pq[size] = e, then std::push_heap(pq, pq + size + 1); returns size + 1
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int pq_push(V2p2PassedHitMatch* pq, int size, const V2p2PassedHitMatch& e);

  // pq_top
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const V2p2PassedHitMatch& pq_top(const V2p2PassedHitMatch* pq) { return pq[0]; }

  // pq_pop: std::pop_heap(pq, pq + size), then pop_back; returns size - 1
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int pq_pop(V2p2PassedHitMatch* pq, int size);

  //----------------------------------------------------------------------------
  // The TrackCand / TrackState accessors MkFinderV2p2 reads, on one candidate slot of a seed. posRsq() as MkFitCore's
  // x86-64-v3 build contracts it, fma(x, x, y * y) (the clone engine's K1 does the same).
  struct TrackCandView {
    const CandBook* book;
    const CandState* state;
    const SeedCandsRef* seed;  // for getLastHitIdx(): the HoT node book->lastHitIdx

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float x() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float y() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float z() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float posR() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float posRsq() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float posPhi() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float invpT() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float momPhi() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float theta() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float pT() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float pz() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float p() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int charge() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int nFoundHits() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int nAllMinusOneHits() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int nTailMinusOneHits() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int getLastHitIdx() const;
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float score() const;
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

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE KalmanOpArgs(bool solve_plane, float chi2_trk_fac);

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void reset() { N_filled = 0; }

    // item_begin(cand, hot, par_idx, hpos)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void item_begin(
        int seed, int ic, int hit_index, int hit_layer, int par_idx, int hpos);

    // item_finished(): ++N_filled == N
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool item_finished() { return ++N_filled == N; }

    // load_state_err_chg(const TrackState&): the solve-plane path, from the parent node's state
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void load_state_err_chg(const V2p2TrackState& st);

    // load_state_err_chg(params_on_hit, prev_state): the depth-0 path, the search's crossing and the TrackCand state
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void load_state_err_chg(const V2p2InitialState& params_on_hit,
                                                                const CandState& prev_state);

    // load_hit_module(hit, module_info): hit position / errors and the module plane of hit (layer, hit_orig_index);
    // returns the hit's detIDinLayer
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int load_hit_module(const EngineHitInputs& hits,
                                                            int layer,
                                                            uint32_t hit_orig_index);

    // compute_pars()
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void compute_pars();

    // do_kalman_stuff() (MkFinderV2p2Structures.cc): inter = finding_inter_layer_pflags, intra =
    // finding_intra_layer_pflags (taken when m_solve_plane)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void do_kalman_stuff(const PropagationFlags& inter,
                                                             const PropagationFlags& intra);

    // the updated state of slot i (ItemOut::state)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void item_state(int i, V2p2TrackState& state) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
