#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkFinderV2p2_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkFinderV2p2_h

// Portable copy of RecoTracker/MkFitCore/src/MkFinderV2p2.h / .cc: the stage functions of the v2p2 finder, in the
// order of MkFinderV2p2.cc. MkFitCore runs one finder per block of seeds with NN-wide batches and queues; the device
// runs every seed of the event through one plan entry at a time (src/alpaka/v2p2/V2p2Kernels.h):
//   V0 thread per seed       awaken_candidates, begin_layer, begin_next_Ccrep_in_layer
//   V2 block per CandIncoming prop_to_layer_edges, determine_search_windows, select_hits / preselect_hit_batch,
//                            order_hits_along_path
//   V3 thread per item        expand_in_layer by depth: KalmanOpArgs::do_kalman_stuff, harvest_nodes
//   V4 thread per seed       select_and_materialise, end_layer
// The batching and completion machinery of MkFitCore (LayerBatch / HitBatch refills, rk_ batches, the node free list,
// early selection) only fills SIMD lanes and changes no result; it has no device counterpart. Results differ from
// MkFitCore only where MkFitCore's own order is unspecified (equal scores in the end-of-layer selection, see
// v2p2SelBefore in interface/v2p2/V2p2SoA.h). Not ported, constexpr-off in CMSSW: InLayer::sister_hole_drop,
// paired plan entries (m_layer_sec), MKFIT_TRACE / standalone diagnostics, TrackCand::m_stop_reason (trace only).

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/LayerOfHitsAccess.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Params.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Score.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2SoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/MiniPropagators.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkBins.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkFinderV2p2Structures.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkRZLimits.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using ::mkfitdev::LayerStepFeatures;

  static_assert(!::mkfitdev::Config::V2p2::InLayer::sister_hole_drop,
                "drop_sister_dominated_holes is not ported (needs module detids and stereo flags on the device)");

  // What MkFinderV2p2 reads from mp_job and mp_steeringparams_iter for one plan entry. The kernels fill everything
  // but rz, which begin_layer() sets per seed (the layer of the seed's region).
  struct V2p2LayerCtx {
    int layer;                                // (*mp_steeringparams_iter)->m_layer
    bool is_outward;                          // mp_steeringparams_iter->is_outward()
    MkRZLimits rz;                            // m_rz_limits
    ::mkfitdev::EngineIterParams ip;          // mp_job->params_cur()
    int capacity;                             // CombCandidate::capacity() = MkJob::max_max_cands()
    ::mkfitdev::V2p2Params vp;                // mp_job->v2p2()
    bool param_b;                             // prop_to_layer_edges' param_b (Window::mini_prop_param_b resolved)
    ::mkfitdev::Config::BFieldParams bfield;  // the constants of Config::bFieldFromZR (MaterialView::bField)
  };

  // The LayerBatch of process_layer_batch() for ONE candidate, plus the per-candidate locals of select_hits().
  // No member initializers: it lives in block shared memory.
  struct V2p2CandWindow {
    MkBins<1> B;
    MkBinTrackCovExtract<1> TCE;
    MkBinLimits<1> BL_p;
    mini_propagators::Hermite3D<1> H;
    ::mkfitdev::WSRResult wsr;  // CandIncoming::m_wsr

    // select_hits(): line pre-cut
    bool pc_on;
    float pc_qb1, pc_q1, pc_gq, pc_phi1, pc_gphi, pc_tol_q, pc_tol_phi;
    // select_hits(): the density region (LayerBatch::rho_dphi / rho_dq and its centre and half widths)
    float rho_dphi, rho_dq;
    float rho_phi_c, rho_phi_half, rho_q_lo, rho_q_hi;
  };

  // A contiguous range [begin, end) of LayerOfHits internal indices: the hits of consecutive phi bins of one q row.
  struct V2p2HitSegment {
    uint32_t begin, end;
  };

  //----------------------------------------------------------------------------
  // Layer control

  // awaken_candidates() for one seed: a Dormant seed picked up at prevLayer becomes Finding, its TrackCands get
  // score 0. Returns true if it woke up.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool awaken_candidates(SeedCandsRef& seed, int prevLayer);

  // begin_layer(): m_rz_limits of the entry's layer
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void begin_layer(V2p2LayerCtx& ctx,
                                                       ::mkfitdev::LayerInfoSoA::ConstView geo,
                                                       int layer);

  // end_layer(), per seed: true if every TrackCand has stopped (the caller sets Finished)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool end_layer_finished(const SeedCandsRef& seed);

  // begin_next_Ccrep_in_layer() for one seed: beam width (late_max_cands), stopping cuts (stop HoT appended to the
  // seed's pool, TrackCand order), quadrant pre-check. active[ic] = 1 for the TrackCands queued as CandIncoming;
  // max_cands = CCandRep::m_max_cands.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void begin_next_Ccrep_in_layer(SeedCandsRef& seed,
                                                                     const V2p2LayerCtx& ctx,
                                                                     int8_t (&active)[::mkfitdev::kMaxCandsPerSeed],
                                                                     int& max_cands);

  // TrackCand::StopReason_e, the value of stop_cuts_at_pickup()
  enum V2p2StopReason : int { SR_NotStopped = 0, SR_MinPt, SR_Looper, SR_TooManyHoles };

  // stop_cuts_at_pickup()
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE V2p2StopReason stop_cuts_at_pickup(const TrackCandView& tc,
                                                                         const V2p2LayerCtx& ctx);

  // long_step_to_layer()
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool long_step_to_layer(const TrackCandView& tc, const V2p2LayerCtx& ctx);

  // fake_hit_index()
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int fake_hit_index(const TrackCandView& tc,
                                                         const ::mkfitdev::WSRResult& wsr,
                                                         const V2p2LayerCtx& ctx);

  //----------------------------------------------------------------------------
  // Phase 1 to 4 for one CandIncoming (process_layer_batch, LayerBatch one candidate wide)

  // prop_to_layer_edges(): m_isp from the TrackCand state, the two layer crossings
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void prop_to_layer_edges(V2p2CandWindow& w,
                                                               const TrackCandView& tc,
                                                               const V2p2LayerCtx& ctx);

  // determine_search_windows() (calls determine_wsr; the Hermite cubic through the crossings)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_search_windows(V2p2CandWindow& w,
                                                                    const TrackCandView& tc,
                                                                    const ::mkfitdev::LayerOfHitsAccess& L,
                                                                    const V2p2LayerCtx& ctx);

  // determine_wsr()
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_wsr(V2p2CandWindow& w, const V2p2LayerCtx& ctx);

  // select_hits(), the per-candidate part before the bin walk: line pre-cut and density region of the layer
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void select_hits_setup(V2p2CandWindow& w, const ::mkfitdev::LayerOfHitsAccess& L);

  // select_hits(), the bin walk of q row qi (q outer loop, BL_p.q1 <= qi < BL_p.q2): its phi bins p1 .. p2 (masked)
  // as at most two contiguous internal-index segments in walk order; returns their number (0 when p1 == p2)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int select_hits_row_segments(const V2p2CandWindow& w,
                                                                   const ::mkfitdev::LayerOfHitsAccess& L,
                                                                   uint16_t qi,
                                                                   V2p2HitSegment (&seg)[2]);

  // select_hits(), per walked hit (internal index hi): the density count (++n_scanned in the region) and the line
  // pre-cut; true = the hit goes to preselect_hit_batch(). No hit mask on the device (clustersToSkip is empty).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool select_hits_scan_hit(const V2p2CandWindow& w,
                                                                const ::mkfitdev::LayerOfHitsAccess& L,
                                                                uint32_t hi,
                                                                int& n_scanned);

  // select_hits(), the end: m_log_rho from the density count
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float select_hits_log_rho(const V2p2CandWindow& w, int n_scanned);

  // surface_referenced_dq() for one hit: h3 = the Hermite state on the hit's module plane, n = the module normal
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float surface_referenced_dq(float dq_track_fallback,
                                                                  const MkBinTrackCovExtract<1>& TCE,
                                                                  int pi,
                                                                  float h3_x,
                                                                  float h3_y,
                                                                  float h3_px,
                                                                  float h3_py,
                                                                  float h3_pz,
                                                                  float nx,
                                                                  float ny,
                                                                  float nz,
                                                                  bool is_barrel);

  // preselect_hit_batch() for n <= N hits of one candidate (internal indices hit_idcs[0, n)): the Hermite cubic
  // solved onto each hit's module plane, the dq / dphi cut; pass[k] and phm[k] for k < n (phm as MkFitCore builds it)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void preselect_hit_batch(const V2p2CandWindow& w,
                                                               const ::mkfitdev::LayerOfHitsAccess& L,
                                                               const EngineHitInputs& hits,
                                                               const uint32_t (&hit_idcs)[N],
                                                               int n,
                                                               bool (&pass)[N],
                                                               V2p2PassedHitMatch (&phm)[N]);

  // preselect_hit_batch(), the reduction of one passed hit: a full pqueue (size >= max_presel_hits) takes it only in
  // place of its worst; returns the new size
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int pq_offer(V2p2PassedHitMatch* pq,
                                                   int size,
                                                   int max_presel_hits,
                                                   const V2p2PassedHitMatch& e);

  // order_hits_along_path(): the pqueue survivors pq[0, n) into layer_hits[0, n) in path order, by bounded insertion
  // with the comparator of MkFitCore's std::sort (a total order: identical result)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void order_hits_along_path(const V2p2CandWindow& w,
                                                                 const V2p2PassedHitMatch* pq,
                                                                 int n,
                                                                 V2p2PassedHitMatch* layer_hits);

  //----------------------------------------------------------------------------
  // In-layer combinatorial search (expand_in_layer, rk_*, harvest_nodes)

  // rk_expand_node()'s same-module veto: true if a node on the path from ni to its root took a hit of `module` on
  // `layer`
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool same_module_on_path(::mkfitdev::V2p2NodeSoA::ConstView nodes,
                                                               int ni,
                                                               int layer,
                                                               int module);

  // harvest_nodes(): the acceptance, chi2_acc < hit_chi2_cut (false for NaN)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool harvest_accept(float chi2_acc, const ::mkfitdev::V2p2Params& vp);

  // harvest_nodes(): the path features, accumulated parent -> child (parent_n_hits = 0 and sums 0 for a root)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void harvest_features(float parent_chi2_sum,
                                                            float parent_log_det_v_sum,
                                                            int parent_n_hits,
                                                            float chi2,
                                                            float det_v,
                                                            float& chi2_sum,
                                                            float& log_det_v_sum,
                                                            int& n_hits);

  // fill_step_geometry()
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void fill_step_geometry(LayerStepFeatures& f,
                                                              float log_rho,
                                                              const V2p2LayerCtx& ctx);

  //----------------------------------------------------------------------------
  // End-of-layer selection

  // select_and_materialise() for one seed. seed = the current buffer (its TrackCands stay readable), outBooks /
  // outStates = the other buffer (kMaxCandsPerSeed slots); HoT nodes are appended to the seed's pool. cands = the
  // seed's CandIncoming rows (v2p2CandRow(s, ic)), nodes + head = its in-layer paths (V2p2SeedSoA::nodeHead and
  // V2p2NodeSoA::nextInSeed). The top max_cands competitors by v2p2SelBefore, in that order. Returns the number of
  // TrackCands written (ccand.size() afterwards); nan_score = true if a NaN score entered the ranking.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int select_and_materialise(SeedCandsRef& seed,
                                                                 CandBook* outBooks,
                                                                 CandState* outStates,
                                                                 ::mkfitdev::V2p2CandSoA::ConstView cands,
                                                                 int s,
                                                                 ::mkfitdev::V2p2NodeSoA::ConstView nodes,
                                                                 int head,
                                                                 int max_cands,
                                                                 const V2p2LayerCtx& ctx,
                                                                 bool& nan_score);

  // offer_best_short()
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void offer_best_short(SeedCandsRef& seed,
                                                            const CandBook& book,
                                                            const CandState& state);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
