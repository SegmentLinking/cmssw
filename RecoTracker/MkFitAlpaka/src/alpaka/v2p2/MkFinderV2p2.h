#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkFinderV2p2_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkFinderV2p2_h

// Portable copy of RecoTracker/MkFitCore/src/MkFinderV2p2.h / .cc: the stage functions of the v2p2 finder, in the
// order of MkFinderV2p2.cc. MkFitCore runs one finder per block of seeds with NN-wide batches and queues; the device
// runs every seed of the event through one plan entry at a time (src/alpaka/v2p2/V2p2Kernels.h):
//   V0 thread per seed       awaken_candidates, begin_layer, begin_next_Ccrep_in_layer
//   V2 block per LayerBatch  prop_to_layer_edges, determine_search_windows, select_hits / preselect_hit_batch,
//                            order_hits_along_path
//   V3 the entry's search    expand_in_layer, rk_expand_node: KalmanOpArgs::do_kalman_stuff, harvest_nodes
//   V4 lanes per seed        select_and_materialise, end_layer
// The completion machinery of MkFitCore (LayerBatch / HitBatch refills across seeds, rk_ batch queues, the node free
// list, early selection) only fills SIMD lanes and changes no result; it has no device counterpart. Results differ from
// MkFitCore only where MkFitCore's own order is unspecified (equal scores in the end-of-layer selection, see
// v2p2SelBefore in interface/v2p2/V2p2SoA.h). Not ported, constexpr-off in CMSSW: InLayer::sister_hole_drop,
// paired plan entries (m_layer_sec), MKFIT_TRACE / standalone diagnostics, TrackCand::m_stop_reason (trace only).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>

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
#include "RecoTracker/MkFitAlpaka/src/alpaka/Packers.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/MiniPropagators.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkBins.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkFinderV2p2Structures.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkRZLimits.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using ::mkfitdev::LayerStepFeatures;

  static_assert(!::mkfitdev::Config::V2p2::InLayer::sister_hole_drop,
                "drop_sister_dominated_holes is not ported (needs the module detids of the hits on the device)");

  // What MkFinderV2p2 reads from mp_job and mp_steeringparams_iter for one plan entry. The kernels fill everything
  // but rz, which begin_layer() sets per seed (the layer of the seed's region).
  struct V2p2LayerCtx {
    bool is_outward;                          // mp_steeringparams_iter->is_outward()
    MkRZLimits rz;                            // m_rz_limits
    ::mkfitdev::EngineIterParams ip;          // mp_job->params_cur()
    int capacity;                             // CombCandidate::capacity() = MkJob::max_max_cands()
    ::mkfitdev::V2p2Params vp;                // mp_job->v2p2()
    bool param_b;                             // prop_to_layer_edges' param_b (Window::mini_prop_param_b resolved)
    ::mkfitdev::Config::BFieldParams bfield;  // the constants of Config::bFieldFromZR (MaterialView::bField)
  };

  // MkFinderV2p2::win_dphi_trk_fac(), win_dq_trk_fac(): the track terms of the search window, the iteration's
  // V2p2Params, else the built-in ones
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float win_dphi_trk_fac(const ::mkfitdev::V2p2Params& vp) {
    return vp.window_dphi_trk_fac > 0.0f ? vp.window_dphi_trk_fac : ::mkfitdev::Config::V2p2::Window::dphi_trk_fac;
  }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float win_dq_trk_fac(const ::mkfitdev::V2p2Params& vp) {
    return vp.window_dq_trk_fac > 0.0f ? vp.window_dq_trk_fac : ::mkfitdev::Config::V2p2::Window::dq_trk_fac;
  }

  // The LayerBatch of process_layer_batch(), N candidates wide and indexed by i, plus the per-candidate locals of
  // select_hits(). No member initializers: it lives in block shared memory.
  template <idx_t N>
  struct V2p2CandWindow {
    int N_proc;
    MkBins<N> B;
    MkBinTrackCovExtract<N> TCE;
    MkBinLimits<N> BL_p;
    mini_propagators::Hermite3D<N> H;
    ::mkfitdev::WSRResult wsr[N];  // CandIncoming::m_wsr

    // select_hits(): line pre-cut
    bool pc_on[N];
    float pc_qb1[N], pc_q1[N], pc_gq[N], pc_phi1[N], pc_gphi[N], pc_tol_q[N], pc_tol_phi[N];
    // select_hits(): the density region (LayerBatch::rho_dphi / rho_dq and its centre and half widths)
    float rho_dphi[N], rho_dq[N];
    float rho_phi_c[N], rho_phi_half[N], rho_q_lo[N], rho_q_hi[N];
  };

  // A contiguous range [begin, end) of LayerOfHits internal indices: the hits of consecutive phi bins of one q row.
  struct V2p2HitSegment {
    uint32_t begin, end;
  };

  // TrackCand::StopReason_e, the value of stop_cuts_at_pickup()
  enum V2p2StopReason : int { SR_NotStopped = 0, SR_MinPt, SR_Looper, SR_TooManyHoles };

  // Declared ahead of their callers, defined below in the order of MkFinderV2p2.cc.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE V2p2StopReason stop_cuts_at_pickup(const TrackCandView& tc,
                                                                         const V2p2LayerCtx& ctx);
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool long_step_to_layer(const TrackCandView& tc, const V2p2LayerCtx& ctx);
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_wsr(V2p2CandWindow<N>& w, const V2p2LayerCtx& ctx);
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void offer_best_short(SeedCandsRef& seed,
                                                            const CandBook& book,
                                                            const CandState& state);

  //----------------------------------------------------------------------------
  // Layer control

  // awaken_candidates() for one seed: a Dormant seed picked up at prevLayer becomes Finding, its TrackCands get
  // score 0. Returns true if it woke up.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool awaken_candidates(SeedCandsRef& seed, int prevLayer) {
    // Awaken dormant ccands with hits on current layer for processing on the next layer.
    // prevLayer -1: the region's plan has ended, nothing to pick up.
    if (prevLayer >= 0 && *seed.state == ::mkfitdev::kDormant && *seed.pickupLayer == prevLayer) {
      *seed.state = ::mkfitdev::kFinding;
      // Start the in-flight score accumulator. Any constant seed term is
      // common to every candidate of this seed and so cancels in the per-seed
      // selection; the cross-seed comparison is done at the end, on the
      // official score.
      for (int ic = 0; ic < *seed.nCands; ++ic)
        seed.cands[ic].score = 0.0f;
      return true;
    }
    return false;
  }

  // begin_layer(): m_rz_limits of the entry's layer
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void begin_layer(V2p2LayerCtx& ctx,
                                                       ::mkfitdev::LayerInfoSoA::ConstView geo,
                                                       int layer) {
    // Setup m_rz_limits.
    ctx.rz.setup(geo, layer, ctx.is_outward);
  }

  // end_layer(), per seed: true if every TrackCand has stopped (the caller sets Finished)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool end_layer_finished(const SeedCandsRef& seed) {
    namespace Policy = ::mkfitdev::Config::V2p2::Policy;
    // A CombCandidate is finished once every TrackCand under it has stopped.
    bool is_finished = false;
    if (Policy::use_stop_cuts || Policy::use_hole_limits) {
      is_finished = true;
      for (int ic = 0; ic < *seed.nCands; ++ic) {
        if (seed.hot(seed.cands[ic].lastHitIdx).index != ::mkfitdev::kHitStopIdx) {
          is_finished = false;
          break;
        }
      }
    }
    return is_finished;
  }

  // begin_next_Ccrep_in_layer() for one seed: beam width (late_max_cands), stopping cuts (stop HoT appended to the
  // seed's pool, TrackCand order), quadrant pre-check. active[ic] = 1 for the TrackCands queued as CandIncoming;
  // max_cands = CCandRep::m_max_cands.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void begin_next_Ccrep_in_layer(SeedCandsRef& seed,
                                                                     const V2p2LayerCtx& ctx,
                                                                     int8_t (&active)[::mkfitdev::kMaxCandsPerSeed],
                                                                     int& max_cands) {
    namespace InLayer = ::mkfitdev::Config::V2p2::InLayer;
    namespace Policy = ::mkfitdev::Config::V2p2::Policy;
    for (int ic = 0; ic < ::mkfitdev::kMaxCandsPerSeed; ++ic)
      active[ic] = 0;

    // Beam width for this layer, per seed: see InLayer::late_max_cands. The
    // TrackCands are in score order from the previous selection, so dropping
    // the tail here is the selection having kept fewer.
    // The capacity is reserved for the larger of the forward and backward widths
    // (MkJob::max_max_cands()); the search direction's own maxCandsPerSeed applies.
    max_cands = std::min(ctx.capacity, ctx.ip.maxCandsPerSeed);
    if (InLayer::late_max_cands > 0 && *seed.nCands > 0 && seed.cands[0].nFound > InLayer::late_max_cands_hits &&
        !long_step_to_layer(TrackCandView{&seed.cands[0], &seed.states[0], &seed}, ctx)) {
      max_cands = std::min(max_cands, (int)InLayer::late_max_cands);
      if (*seed.nCands > max_cands)
        *seed.nCands = max_cands;  // ccand.resize()
    }

    for (int ic = 0; ic < *seed.nCands; ++ic) {
      const TrackCandView tcand{&seed.cands[ic], &seed.states[ic], &seed};

      if (tcand.getLastHitIdx() == ::mkfitdev::kHitStopIdx)
        continue;

      // Stopping cuts, at pull-in: see stop_cuts_at_pickup().
      if (Policy::use_stop_cuts) {
        const V2p2StopReason sr = stop_cuts_at_pickup(tcand, ctx);
        if (sr != SR_NotStopped) {
          seedAddHitIdx(seed, seed.cands[ic], ::mkfitdev::kHitStopIdx, ctx.rz.m_layer_id, 0.0f);
          continue;
        }
      }

      // "Layer already passed in z" pre-check, before any propagation. A failed
      // check skips the layer without recording anything, as WSR_Outside does.

      if (!ctx.rz.rz_quadrant_check(tcand.z(), tcand.pz())) {
        continue;
      }

      // Create and register a CandIncoming for processing.
      active[ic] = 1;
    }
  }

  // stop_cuts_at_pickup() -- minPtCut and the looper stop, which need only the
  // candidate's own state. The looper test covers both wrap images of
  // |posPhi - momPhi|. See doc/MkFinderV2p2-DesignNotes.md,
  // "Candidate pickup and stopping cuts".
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE V2p2StopReason stop_cuts_at_pickup(const TrackCandView& tc,
                                                                         const V2p2LayerCtx& ctx) {
    namespace Policy = ::mkfitdev::Config::V2p2::Policy;
    if (tc.pT() < ctx.ip.minPtCut)
      return SR_MinPt;

    // Forward search only: going inward the track is leaving the turning region,
    // so a large angle there is not evidence that it is about to curl up.
    if (ctx.is_outward && tc.pT() < Policy::looper_max_pt &&
        tc.posRsq() > Policy::looper_min_r * Policy::looper_min_r) {
      const float max_ang = Policy::looper_max_angle;
      const float dphi = std::abs(tc.posPhi() - tc.momPhi());
      if (dphi > max_ang && dphi < Const::TwoPI - max_ang)
        return SR_Looper;
    }

    return SR_NotStopped;
  }

  // Straight-line path length from the candidate's state (its last hit) to the
  // layer's near surface, against InLayer::late_max_cands_step_cm. Barrel: radial gap
  // times p / pT; endcap: z gap times p / |pz|. Curvature is ignored, so the
  // barrel path is underestimated for strongly bending tracks.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool long_step_to_layer(const TrackCandView& tc, const V2p2LayerCtx& ctx) {
    namespace InLayer = ::mkfitdev::Config::V2p2::InLayer;
    if (InLayer::late_max_cands_step_cm <= 0.0f)
      return false;
    const MkRZLimits& li = ctx.rz;
    float s;
    // std::max({a, b, c}) as two std::max: the same first-largest choice
    if (li.m_is_barrel) {
      const float r = tc.posR();
      const float dr = std::max(std::max(li.m_rin - r, r - li.m_rout), 0.0f);
      s = dr * tc.p() / std::max(1e-3f, tc.pT());
    } else {
      const float z = tc.z();
      const float dz = std::max(std::max(li.m_zmin - z, z - li.m_zmax), 0.0f);
      s = dz * tc.p() / std::max(1e-3f, std::abs(tc.pz()));
    }
    if (s > InLayer::late_max_cands_step_cm) {
      return true;
    }
    return false;
  }

  // fake_hit_index() -- the fake HoT for a candidate that took no hit: the hole
  // limits choose miss or stop, then the WSR overrides with edge or gap, in
  // V1's order.
  // select_and_materialise(): a layer the candidate does not reach (WSR_Outside) is a hole for a seed with at most
  // V2p2Params::outside_hole_max_seed_hits hits, until the candidate has a hit beyond its seed
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool outside_as_hole(const TrackCandView& tc, const ::mkfitdev::V2p2Params& vp) {
    const int outside_max_seed = vp.outside_hole_max_seed_hits;
    return outside_max_seed > 0 && tc.getNSeedHits() <= outside_max_seed && tc.nFoundHits() <= tc.getNSeedHits();
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int fake_hit_index(const TrackCandView& tc,
                                                         const ::mkfitdev::WSRResult& wsr,
                                                         const V2p2LayerCtx& ctx) {
    namespace Policy = ::mkfitdev::Config::V2p2::Policy;
    const auto& iter_params = ctx.ip;

    int fake_hit_idx = ::mkfitdev::kHitMissIdx;

    if (Policy::use_hole_limits &&
        (tc.nAllMinusOneHits() >= iter_params.maxHolesPerCand || tc.nTailMinusOneHits() >= iter_params.maxConsecHoles))
      fake_hit_idx = ::mkfitdev::kHitStopIdx;

    if (Policy::use_wsr) {
      if (wsr.wsr == ::mkfitdev::WSR_Edge)
        fake_hit_idx = ::mkfitdev::kHitEdgeIdx;
      else if (wsr.in_gap)
        fake_hit_idx = ::mkfitdev::kHitInGapIdx;
    }

    return fake_hit_idx;
  }

  //----------------------------------------------------------------------------
  // Phase 1 to 4 for one batch of up to N CandIncomings of one layer (process_layer_batch, LayerBatch N wide)

  // Phase 1 -- which candidates, and where do they meet the layer.
  // prop_to_layer_edges(): m_isp from the TrackCand states tc[0, n_proc), the two layer crossings
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void prop_to_layer_edges(V2p2CandWindow<N>& w,
                                                               const TrackCandView* tc,
                                                               int n_proc,
                                                               const V2p2LayerCtx& ctx) {
    MkBins<N>& B = w.B;
    const int N_proc = w.N_proc = n_proc;
    // MkBins already computes every slot (free ones repeat candidate 0): the full-width change needs nothing here
    B.m_n_proc = N_proc;

    float phi[N];
    int chg[N];
    float bf[N];

    // Slots from N_proc up repeat candidate 0: valid inputs, never read.
    for (int i = 0; i < N; ++i) {
      const TrackCandView& c = tc[i < N_proc ? i : 0];
      // Copy in x, y,z, invpT, theta.
      B.m_isp.copyIn_partial_track_state(i, c.state->par);
      phi[i] = c.momPhi();
      chg[i] = c.charge();
      bf[i] = Config::Bfield;
      if (ctx.param_b)
        bf[i] = Config::bFieldFromZR(ctx.bfield, c.z(), std::hypot(c.x(), c.y()));
    }
    B.m_isp.init_momentum_vec_and_k(phi, chg, bf);

    // m_sp1 at the layer's near bounding surface, m_sp2 at the far one.
    B.prop_to_limits_in_order(ctx.rz);
  }

  // Phase 2 -- the search windows. The position block of the track covariance is
  // transported to m_sp2 (MkBins::transport_position_cov()); from it come
  // dphi_track and dq_track, and from those the WSR verdict and the binnor
  // ranges. The function also builds the Hermite cubic through the two layer
  // crossings. See doc/MkFinderV2p2-DesignNotes.md, "Search window".
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_search_windows(V2p2CandWindow<N>& w,
                                                                    const TrackCandView* tc,
                                                                    const ::mkfitdev::LayerOfHitsAccess& L,
                                                                    const V2p2LayerCtx& ctx) {
    const int N_proc = w.N_proc;
    MkBins<N>& B = w.B;

    // Covariance position block at m_sp2, transported from the previous hit.
    MPlexLV<N> par0;
    MPlexLS<N> err0;
    // Slots from N_proc up repeat candidate 0: valid inputs, never read.
    for (int i = 0; i < N; ++i) {
      par0.copyIn(i, tc[i < N_proc ? i : 0].state->par);
      err0.copyIn(i, tc[i < N_proc ? i : 0].state->err);
    }
    B.transport_position_cov(par0, err0, w.TCE);

    B.determine_bin_windows(w.TCE);

    // The WSR edge fuzz uses dq_track, so the verdict follows determine_bin_windows().
    determine_wsr(w, ctx);

    B.find_bin_ranges(L, w.BL_p, win_dphi_trk_fac(ctx.vp), win_dq_trk_fac(ctx.vp));

    // Hermite cubic through the two layer crossings, the trajectory model for the
    // per-hit plane solve. See the design notes, "Layer crossings and the Hermite
    // cubic".
    w.H.calculate_coeffs(B.m_sp1, B.m_sp2, B.m_isp.inv_k);
  }

  // Phase 2b -- within-sensitive-region verdict per candidate: does the segment
  // between the two layer crossings lie in sensitive material. The fail flags
  // of the two crossings answer the radial half; the segment's q extent,
  // widened by Policy::wsr_n_sigma, answers the rest. See
  // doc/MkFinderV2p2-DesignNotes.md, "Within-sensitive-region verdict".
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_wsr(V2p2CandWindow<N>& w, const V2p2LayerCtx& ctx) {
    namespace Policy = ::mkfitdev::Config::V2p2::Policy;
    using ::mkfitdev::WSR_Edge;
    using ::mkfitdev::WSR_Inside;
    using ::mkfitdev::WSR_Outside;
    // m_dq_track is 3 sigma.
    const float dq_to_n_sigma = Policy::wsr_n_sigma / 3.0f;

    const int N_proc = w.N_proc;
    const MkBins<N>& B = w.B;

    const MkRZLimits& LI = ctx.rz;
    const bool is_barrel = LI.m_is_barrel;
    const float q_lo = is_barrel ? LI.m_zmin : LI.m_rin;
    const float q_hi = is_barrel ? LI.m_zmax : LI.m_rout;

    for (int i = 0; i < N_proc; ++i) {
      const bool f_entry = B.m_sp1.fail_flag[i] != 0;
      const bool f_exit = B.m_sp2.fail_flag[i] != 0;

      ::mkfitdev::WSRResult wr;

      if (f_entry && f_exit) {
        // Clear miss: the track does not reach the layer at all.
        wr = ::mkfitdev::WSRResult{WSR_Outside, false};
      } else {
        const float dq = dq_to_n_sigma * B.m_dq_track[i];
        const float qa = B.m_q_min[i], qb = B.m_q_max[i];

        if (qb < q_lo - dq || qa > q_hi + dq)
          wr = ::mkfitdev::WSRResult{WSR_Outside, false};
        else if (qa > q_lo + dq && qb < q_hi - dq)
          wr = ::mkfitdev::WSRResult{WSR_Inside, false};
        else
          wr = ::mkfitdev::WSRResult{WSR_Edge, false};

        // Near miss -- in the layer, but turning round inside it.
        if ((f_entry || f_exit) && wr.wsr == WSR_Inside)
          wr.wsr = WSR_Edge;

        // Endcap discs have a hole at small r. Sitting wholly inside it is a
        // miss with a reason; touching it is an edge, and either way m_in_gap
        // records that the absence of a hit is explained.
        if (!is_barrel && LI.m_layer_has_r_range_hole && wr.wsr != WSR_Outside) {
          const float h_lo = LI.m_layer_hole_r_min, h_hi = LI.m_layer_hole_r_max;
          if (qa > h_lo + dq && qb < h_hi - dq) {
            wr = ::mkfitdev::WSRResult{WSR_Outside, true};
          } else if (qb > h_lo - dq && qa < h_hi + dq) {
            wr.wsr = WSR_Edge;
            wr.in_gap = true;
          }
        }
      }

      w.wsr[i] = wr;
    }
  }

  // Phase 3 -- walk the bin ranges and pre-select hits.
  // select_hits(), the part of candidate i before its bin walk: line pre-cut and density region of the layer
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void select_hits_setup(V2p2CandWindow<N>& w,
                                                             int i,
                                                             const ::mkfitdev::LayerOfHitsAccess& L,
                                                             const ::mkfitdev::V2p2Params& vp) {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    namespace PreCut = ::mkfitdev::Config::V2p2::PreCut;
    const MkBins<N>& B = w.B;

    // The iteration's window factors, read once per call.
    const float dphi_trk_fac = win_dphi_trk_fac(vp), dq_trk_fac = win_dq_trk_fac(vp);

    // Line pre-cut, per candidate: the straight line m_sp1 -> m_sp2 in
    // (qbar, q) and (qbar, phi), and the hit-independent tolerance terms.
    // See doc/MkFinderV2p2-DesignNotes.md, "Line pre-cut".
    const bool barrel = B.m_is_barrel;
    bool pc_on = PreCut::q || PreCut::phi;
    float pc_qb1 = 0, pc_q1 = 0, pc_gq = 0, pc_phi1 = 0, pc_gphi = 0, pc_tol_q = 0, pc_tol_phi = 0;
    if (pc_on) {
      const float x1 = B.m_sp1.x[i], y1 = B.m_sp1.y[i], z1 = B.m_sp1.z[i];
      const float x2 = B.m_sp2.x[i], y2 = B.m_sp2.y[i], z2 = B.m_sp2.z[i];
      const float r1 = std::hypot(x1, y1), r2 = std::hypot(x2, y2);
      pc_qb1 = barrel ? r1 : z1;
      pc_q1 = barrel ? z1 : r1;
      const float dqb = (barrel ? r2 : z2) - pc_qb1;
      if (std::abs(dqb) < 1e-4f) {
        pc_on = false;  // no span to interpolate along
      } else {
        pc_gq = std::clamp(((barrel ? z2 : r2) - pc_q1) / dqb, -20.0f, 20.0f);
        pc_phi1 = B.m_phi1[i];
        pc_gphi = std::clamp(squashPhiGeneral(B.m_phi2[i] - pc_phi1) / dqb, -20.0f, 20.0f);
        pc_tol_q = PreCut::dq_slack * dq_trk_fac * B.m_dq_track[i] * (1.0f + pc_gq * pc_gq);
        pc_tol_phi = PreCut::dphi_slack * Window::capped_dphi(dphi_trk_fac * B.m_dphi_track[i]);
      }
    }
    w.pc_on[i] = pc_on;
    w.pc_qb1[i] = pc_qb1;
    w.pc_q1[i] = pc_q1;
    w.pc_gq[i] = pc_gq;
    w.pc_phi1[i] = pc_phi1;
    w.pc_gphi[i] = pc_gphi;
    w.pc_tol_q[i] = pc_tol_q;
    w.pc_tol_phi[i] = pc_tol_phi;

    // The region rho counts over and divides by, see the end of this function: the
    // window with the cut's padding of this sub-layer, as the fetch.
    const float phi_hit_term =
        Window::phi_per_hit ? Window::dphi_hit_fac * L.max_hit_phi_half_extent() : Window::dphi_flat_rad;
    const float rho_dphi = Window::capped_dphi(dphi_trk_fac * B.m_dphi_track[i] + phi_hit_term);
    const float rho_dq = dq_trk_fac * B.m_dq_track[i] + Window::dq_hit_fac * L.max_hit_q_half_length();
    // LayerBatch::rho_dphi / rho_dq start at zero
    w.rho_dphi[i] = std::max(0.0f, rho_dphi);
    w.rho_dq[i] = std::max(0.0f, rho_dq);
    // Centre and half-width from MkBins: across phi = +-pi, m_phi_min and m_phi_max are swapped for the bin walk.
    w.rho_phi_c[i] = B.m_phi_center[i];
    w.rho_phi_half[i] = B.m_phi_delta[i] + rho_dphi;
    w.rho_q_lo[i] = B.m_q_min[i] - rho_dq;
    w.rho_q_hi[i] = B.m_q_max[i] + rho_dq;
  }

  // select_hits(), the bin walk of candidate i in q row qi (q outer loop, BL_p.q1 <= qi < BL_p.q2): its phi bins
  // p1 .. p2 (masked) as at most two contiguous internal-index segments in walk order; returns their number (0 when
  // p1 == p2)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int select_hits_row_segments(const V2p2CandWindow<N>& w,
                                                                   int i,
                                                                   const ::mkfitdev::LayerOfHitsAccess& L,
                                                                   uint16_t qi,
                                                                   V2p2HitSegment (&seg)[2]) {
    // The hits of one q row are in phi-bin order (binnor sort key: q bin, then phi), so the walk over consecutive
    // phi bins visits consecutive internal indices; an empty bin stores first = 0 and is skipped.
    const uint16_t p1 = w.BL_p.p1[i], p2 = w.BL_p.p2[i];
    if (p1 == p2)
      return 0;
    // [p1, p2), or across the phi wrap [p1, end of ring) then [0, p2)
    const int nRange = (p1 < p2 || p2 == 0) ? 1 : 2;
    const int lo[2] = {p1, 0};
    const int hi[2] = {p1 < p2 ? int(p2) : int(::mkfitdev::kNPhiBins), int(p2)};
    int nSeg = 0;
    for (int range = 0; range < nRange; ++range) {
      int first = lo[range];
      while (first < hi[range] && ::mkfitdev::binCount(L.binContent(first, qi)) == 0)
        ++first;
      if (first == hi[range])
        continue;
      int last = hi[range] - 1;
      while (::mkfitdev::binCount(L.binContent(last, qi)) == 0)
        --last;
      const uint32_t contentFirst = L.binContent(first, qi), contentLast = L.binContent(last, qi);
      seg[nSeg].begin = ::mkfitdev::binFirst(contentFirst);
      seg[nSeg].end = ::mkfitdev::binFirst(contentLast) + ::mkfitdev::binCount(contentLast);
      ++nSeg;
    }
    return nSeg;
  }

  // select_hits(), per walked hit of candidate i (internal index hi): the density count (++n_scanned in the region)
  // and the line pre-cut; true = the hit goes to preselect_hit_batch(). No hit mask on the device (clustersToSkip is
  // empty).
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool select_hits_scan_hit(
      const V2p2CandWindow<N>& w, int i, const ::mkfitdev::LayerOfHitsAccess& L, uint32_t hi, int& n_scanned) {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    namespace PreCut = ::mkfitdev::Config::V2p2::PreCut;
    if (std::abs(squashPhiGeneral(L.hit_phi(hi) - w.rho_phi_c[i])) < w.rho_phi_half[i] && L.hit_q(hi) > w.rho_q_lo[i] &&
        L.hit_q(hi) < w.rho_q_hi[i])
      ++n_scanned;

    if (w.pc_on[i]) {
      const bool barrel = w.B.m_is_barrel;
      const float dqb_h = L.hit_qbar(hi) - w.pc_qb1[i];
      const float qbar_term = barrel ? PreCut::qbar_fac * L.hit_qbar_half_extent(hi) : 0.0f;
      bool reject = false;
      if (PreCut::q) {
        const float dev = std::abs(L.hit_q(hi) - (w.pc_q1[i] + dqb_h * w.pc_gq[i]));
        reject = dev > w.pc_tol_q[i] + Window::dq_hit_fac * L.hit_q_half_length(hi) + std::abs(w.pc_gq[i]) * qbar_term;
      }
      if (!reject && PreCut::phi) {
        const float dev = std::abs(squashPhiGeneral(L.hit_phi(hi) - (w.pc_phi1[i] + dqb_h * w.pc_gphi[i])));
        const float hit_term =
            Window::phi_per_hit ? Window::dphi_hit_fac * L.hit_phi_half_extent(hi) : Window::dphi_flat_rad;
        reject = dev > w.pc_tol_phi[i] + hit_term + std::abs(w.pc_gphi[i]) * qbar_term;
      }
      if (reject) {
        return false;
      }
    }
    return true;
  }

  // select_hits(), the end: m_log_rho from the density count
  // Local hit density, hits per cm^2, over the cut region: the window widened by
  // the cut's track and largest-hit-extent terms as the fetch computes them, but
  // without bin rounding or spare bins. Hits are counted over the same region, not
  // over the pre-selected ones, so the estimate does not depend on the cut it is
  // about to feed. The phi extent becomes a length at the crossing radius. The count
  // can be 0 with a hit about to be scored, so the scored hit is counted too.
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float select_hits_log_rho(const V2p2CandWindow<N>& w, int i, int n_scanned) {
    const MkBins<N>& B = w.B;
    const float r = std::max(0.1f, std::hypot(B.m_sp2.x[i], B.m_sp2.y[i]));
    // m_phi_delta is the half-width of the crossing span, also across phi = +-pi; the region is at most the full turn.
    const float w_phi = std::min(2.0f * (B.m_phi_delta[i] + w.rho_dphi[i]), float(::mkfitdev::Const::TwoPI));
    const float w_q = (B.m_q_max[i] - B.m_q_min[i]) + 2.0f * w.rho_dq[i];
    const float area = std::max(1e-4f, w_phi * r * w_q);
    return std::log(std::max(1e-6f, (n_scanned + 1.0f) / area));
  }

  // surface_referenced_dq() -- the track's q error referenced to the hit's
  // module plane rather than to the fixed path length it was transported to:
  //     v = e_q - ((e_q.p^)/(n^.p^)) n^ ,   sigma_q^2 = v^T C v
  // with n^ the module normal. Returns 3 sigma_q, the convention of
  // MkBins::m_dq_track, which is also the fallback. See
  // doc/MkFinderV2p2-DesignNotes.md, "Search window".
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float surface_referenced_dq(float dq_track_fallback,
                                                                  const MkBinTrackCovExtract<N>& TCE,
                                                                  int pi,
                                                                  float h3_x,
                                                                  float h3_y,
                                                                  float h3_px,
                                                                  float h3_py,
                                                                  float h3_pz,
                                                                  float nx,
                                                                  float ny,
                                                                  float nz,
                                                                  bool is_barrel) {
    const float px = h3_px, py = h3_py, pz = h3_pz;
    const float np = nx * px + ny * py + nz * pz;  // (n^.p) -- |p| cancels below
    // q direction: z in the barrel -- so e_q.p is just pz, with no dot product
    // and no hipo -- and r^ in the endcap.
    float ex = 0.0f, ey = 0.0f, ez = 1.0f, eqp = pz;
    bool ok = (np != 0.0f);
    if (ok && !is_barrel) {
      const float rr = hipo(h3_x, h3_y);
      ok = (rr > 0.0f);
      if (ok) {
        ex = h3_x / rr;
        ey = h3_y / rr;
        ez = 0.0f;
        eqp = ex * px + ey * py;
      }
    }
    if (!ok)
      return dq_track_fallback;

    // f = (e_q.p^)/(n^.p^); |p| cancels, so no normalisation is needed. Clamped:
    // it diverges only at grazing incidence on the module, which is not an
    // operating point (the cluster is then too wide in phi to be a hit).
    const float f = std::clamp(eqp / np, -20.0f, 20.0f);
    const float v0 = ex - f * nx, v1 = ey - f * ny, v2 = ez - f * nz;
    const float var = v0 * v0 * TCE.m_cov_0_0[pi] + v1 * v1 * TCE.m_cov_1_1[pi] + v2 * v2 * TCE.m_cov_2_2[pi] +
                      2.0f * (v0 * v1 * TCE.m_cov_0_1[pi] + v0 * v2 * TCE.m_cov_0_2[pi] + v1 * v2 * TCE.m_cov_1_2[pi]);
    if (var <= 0.0f)
      return dq_track_fallback;
    return 3.0f * std::sqrt(var);
  }

  // Phase 3b -- pre-select one N-wide batch of (candidate, hit) pairs: hit hit_idcs[h] (original index
  // hit_orig_idcs[h]) of candidate cand_idcs[h] (HitBatch).
  //
  // Solves the Hermite cubic onto each hit's own module plane, applies the
  // dq / dphi pre-selection cut; the caller pushes survivors into the
  // candidate's bounded priority queue (pq_offer()).
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void preselect_hit_batch(const V2p2CandWindow<N>& w,
                                                               const ::mkfitdev::LayerOfHitsAccess& L,
                                                               const EngineHitInputs& hits,
                                                               const int (&cand_idcs)[N],
                                                               const uint32_t (&hit_idcs)[N],
                                                               const uint32_t (&hit_orig_idcs)[N],
                                                               int n,
                                                               bool (&pass)[N],
                                                               V2p2PassedHitMatch (&phm)[N],
                                                               const ::mkfitdev::V2p2Params& vp) {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    namespace mp = mini_propagators;
    const MkBins<N>& B = w.B;
    const MkBinTrackCovExtract<N>& TCE = w.TCE;

    // Zeroed: the lanes from n up are not filled, and init_coeffs() reads
    // all of them.
    MPlexHV<N> module_pos(0.0f);
    MPlexHV<N> module_norm(0.0f);
    mp::Hermite3D<N> h3d(0.0f);

    // Extract hit / target module data
    for (int h = 0; h < n; ++h) {
      const int row = ::mkfitdev::pack::hitRow(hits, L.layer, int(hit_orig_idcs[h]));
      const uint32_t mid = ::mkfitdev::hitpack::detIDinLayer(hits.packed[row]);
      const ::mkfitdev::EngineModule& mi = hits.modules[hits.layers[L.layer].moduleBegin + int(mid)];
      for (int k = 0; k < 3; ++k) {
        module_pos.At(h, k, 0) = mi.pnt[k];
        module_norm.At(h, k, 0) = mi.nrm[k];
      }

      h3d.copyIn(h, w.H, cand_idcs[h]);
    }

    mp::Hermite3DOnPlane<N> h3dop;
    h3dop.init_coeffs(h3d, module_pos, module_norm);
    // NOTE: solve() is a SINGLE Newton step, not an iteration to convergence.
    h3dop.solve();

    mp::StatePlex<N> h3_state;
    for (int h = 0; h < N; ++h) {
      h3_state.dalpha[h] = 0.0f;
      h3_state.fail_flag[h] = 0;
    }
    h3d.evaluate(h3dop.m_T, h3_state);
    // h3_state.dalpha is interpolated per hit below.

    // phi of each crossing, for the whole batch ahead of the per-hit loop below, which does not vectorise.
    float h3_phi[N];
    MPLEX_SIMD
    for (int h = 0; h < N; ++h)
      h3_phi[h] = ::mkfitdev::Matriplex::internal::atan2_poly(h3_state.y[h], h3_state.x[h]);

    // The iteration's window factors, read once per batch.
    const float dphi_trk_fac = win_dphi_trk_fac(vp), dq_trk_fac = win_dq_trk_fac(vp);

    // The cut is taken on h3_state, the cubic solved onto the hit's module plane.
    for (int h = 0; h < N; ++h) {
      pass[h] = false;
      if (h >= n)
        continue;
      const int i = cand_idcs[h];
      float q;
      if (B.m_is_barrel) {
        q = h3_state.z[h];
      } else {
        q = hipo(h3_state.x[h], h3_state.y[h]);
      }
      const float ddq = std::abs(q - L.hit_q(hit_idcs[h]));
      const float phi = h3_phi[h];
      const float ddphi = cdist(std::abs(phi - L.hit_phi(hit_idcs[h])));

      // q error referenced to this module's plane, see surface_referenced_dq().
      const float dq_trk = Window::surface_q ? surface_referenced_dq(B.m_dq_track[i],
                                                                     TCE,
                                                                     i,
                                                                     h3_state.x[h],
                                                                     h3_state.y[h],
                                                                     h3_state.px[h],
                                                                     h3_state.py[h],
                                                                     h3_state.pz[h],
                                                                     module_norm.At(h, 0, 0),
                                                                     module_norm.At(h, 1, 0),
                                                                     module_norm.At(h, 2, 0),
                                                                     B.m_is_barrel)
                                             : B.m_dq_track[i];

      // The pre-selection cut. See doc/MkFinderV2p2-DesignNotes.md, "Search window".
      const float dq_cut = dq_trk_fac * dq_trk + Window::dq_hit_fac * L.hit_q_half_length(hit_idcs[h]);
      const float dphi_cut = Window::capped_dphi(
          dphi_trk_fac * B.m_dphi_track[i] +
          (Window::phi_per_hit ? Window::dphi_hit_fac * L.hit_phi_half_extent(hit_idcs[h]) : Window::dphi_flat_rad));
      pass[h] = ddq < dq_cut && ddphi < dphi_cut;

      // The grazing-incidence cut on cos^2 of the incidence on the hit's module, off at the default. See
      // doc/MkFinderV2p2-DesignNotes.md, "Grazing incidence".
      if (Window::min_cos_incidence > 0.0f) {  // a plain if: the branch is compiled in every build
        const float pn = h3_state.px[h] * module_norm.At(h, 0, 0) + h3_state.py[h] * module_norm.At(h, 1, 0) +
                         h3_state.pz[h] * module_norm.At(h, 2, 0);
        const float p2 =
            h3_state.px[h] * h3_state.px[h] + h3_state.py[h] * h3_state.py[h] + h3_state.pz[h] * h3_state.pz[h];
        const float cos2_inc = pn * pn / p2;
        if (cos2_inc < Window::min_cos_incidence * Window::min_cos_incidence)
          pass[h] = false;
      }

      // For every hit, not only those that pass.
      h3_state.dalpha[h] = B.m_sp1.dalpha[i] + h3dop.m_T[h] * (B.m_sp2.dalpha[i] - B.m_sp1.dalpha[i]);

      // InitialState(h3_state, h, is_plex, h): the crossing, with inv_pt, inv_k and theta of the candidate's m_isp
      phm[h].ddphi = ddphi;
      phm[h].hit_orig_index = hit_orig_idcs[h];
      phm[h].hit_index = hit_idcs[h];
      phm[h].layer = L.layer;
      phm[h].mixed_state = V2p2InitialState{h3_state.x[h],
                                            h3_state.y[h],
                                            h3_state.z[h],
                                            h3_state.px[h],
                                            h3_state.py[h],
                                            h3_state.pz[h],
                                            h3_state.dalpha[h],
                                            h3_state.fail_flag[h],
                                            B.m_isp.inv_pt[i],
                                            B.m_isp.inv_k[i],
                                            B.m_isp.theta[i]};
    }
  }

  // preselect_hit_batch(), the reduction of one passed hit: a full pqueue (size >= max_presel_hits) takes it only in
  // place of its worst; returns the new size
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int pq_offer(V2p2PassedHitMatch* pq,
                                                   int size,
                                                   int max_presel_hits,
                                                   const V2p2PassedHitMatch& e) {
    // A full queue takes the hit only in place of its worst one.
    const bool full = size >= max_presel_hits;
    if (full && !(e.ddphi < pq_top(pq).ddphi))
      return size;
    if (full)
      size = pq_pop(pq, size);
    return pq_push(pq, size, e);
  }

  // Phase 4 -- move the pqueue survivors into the per-candidate hit list.
  // order_hits_along_path(): the pqueue survivors pq[0, n) of candidate i into layer_hits[0, n) in path order, by
  // bounded insertion with the comparator of MkFitCore's std::sort (a total order: identical result)
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void order_hits_along_path(
      const V2p2CandWindow<N>& w, int i, const V2p2PassedHitMatch* pq, int n, V2p2PassedHitMatch* layer_hits) {
    // Path order for the in-layer search: dalpha is monotone in path length
    // for one candidate, but it is a signed turning angle, so its sign goes
    // with the charge. m_sp1 is the near bounding surface and m_sp2 the far
    // one in the search direction, so dalpha times the sign of its change
    // from m_sp1 to m_sp2 grows along the path, in either search direction.
    // See doc/MkFinderV2p2-DesignNotes.md, "Reduction and hit ordering".
    const float dir = w.B.m_sp2.dalpha[i] >= w.B.m_sp1.dalpha[i] ? 1.0f : -1.0f;
    auto before = [dir](const V2p2PassedHitMatch& a, const V2p2PassedHitMatch& b) {
      const float ka = dir * a.mixed_state.dalpha, kb = dir * b.mixed_state.dalpha;
      if (ka != kb)
        return ka < kb;
      // Equal keys: hits on one module, or on coplanar modules of one layer,
      // share the plane crossing. Larger ddphi first, the order the
      // pqueue drain gave and the results were validated with, then by
      // hit: an order that does not depend on the pqueues'.
      if (a.ddphi != b.ddphi)
        return a.ddphi > b.ddphi;
      return a.hit_orig_index < b.hit_orig_index;
    };
    for (int k = 0; k < n; ++k) {
      const V2p2PassedHitMatch e = pq[k];
      int j = k;
      while (j > 0 && before(e, layer_hits[j - 1])) {
        layer_hits[j] = layer_hits[j - 1];
        --j;
      }
      layer_hits[j] = e;
    }
  }

  //----------------------------------------------------------------------------
  // In-layer combinatorial search (expand_in_layer, rk_*, harvest_nodes)

  // rk_expand_node()'s same-module veto: true if a node on the path from ni to its root took a hit of `module` on
  // `layer`
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool same_module_on_path(::mkfitdev::V2p2NodeSoA::ConstView nodes,
                                                               int ni,
                                                               int layer,
                                                               int module) {
    // A second hit from a module already on this path is a split cluster or
    // another track's hit, not an overlap: skip it. The walk back is at most
    // V2p2Params::max_hits_per_layer long.
    bool same_module = false;
    for (int ci = ni, d = 0; ci >= 0 && !same_module && d < ::mkfitdev::kV2p2MaxHitsPerLayer;
         ci = nodes.parentNode(ci), ++d) {
      if (nodes.hitLayer(ci) != layer)
        continue;
      same_module = nodes.module(ci) == module;
    }
    return same_module;
  }

  // harvest_nodes(): the acceptance, chi2_acc < hit_chi2_cut (false for NaN) with chi2 and chi2_acc not negative and
  // det V positive
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool harvest_accept(float chi2,
                                                          float chi2_acc,
                                                          float det_v,
                                                          const ::mkfitdev::V2p2Params& vp) {
    // Also rejects NaN. The score would reward a negative chi2 (broken predicted covariance, design notes "Grazing
    // incidence") and det V <= 0 (indefinite residual covariance, ln(det V) floored).
    return chi2_acc < vp.hit_chi2_cut && chi2_acc >= 0.0f && chi2 >= 0.0f && det_v > 0.0f;
  }

  // harvest_nodes(): the path features, accumulated parent -> child (parent_n_hits = 0 and sums 0 for a root)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void harvest_features(float parent_chi2_sum,
                                                            float parent_log_det_v_sum,
                                                            int parent_n_hits,
                                                            float chi2,
                                                            float det_v,
                                                            float& chi2_sum,
                                                            float& log_det_v_sum,
                                                            int& n_hits) {
    n_hits = parent_n_hits + 1;
    chi2_sum = parent_chi2_sum + chi2;
    log_det_v_sum = parent_log_det_v_sum + std::log(std::max(1e-30f, det_v));
  }

  // fill_step_geometry(): the parts of a layer step that do not depend on which hits were taken: where
  // the step goes and which way. Filled once per path root and carried down the
  // tree, and once per hole entry of the end-of-layer selection.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void fill_step_geometry(LayerStepFeatures& f,
                                                              float log_rho,
                                                              const V2p2LayerCtx& ctx) {
    const MkRZLimits& li = ctx.rz;
    f.layer_to = (short)li.m_layer_id;
    f.is_outward = li.is_outward();
    f.is_pixel = li.m_layer_is_pixel;
    f.is_barrel = li.m_is_barrel;
    f.log_rho = log_rho;
  }

  //----------------------------------------------------------------------------
  // End-of-layer selection: every in-layer path, every candidate as the hole it
  // would record, and every candidate that did not enter the layer compete on
  // the additive layer-step score; the top CCandRep::m_max_cands survive.
  // See doc/MkFinderV2p2-DesignNotes.md, "End-of-layer selection".

  // One competitor in the end-of-layer selection (MkFinderV2p2::SelEntry), with its rank key.
  struct V2p2SelEntry {
    int tcand_idx;  // the TrackCand of the CombCandidate it continues
    int node_idx;   // pool row of its in-layer path; -1 = it took no hit here
    int fake_hit;   // with add_fake, the fake HoT to record: miss, edge, gap or stop
    bool add_fake;  // false: nothing recorded (layer out of reach, or already stopped)
    float score;    // its score after this layer
    uint64_t ord;   // v2p2SelOrdinal
  };

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool v2p2SelEntryBefore(const V2p2SelEntry& a, const V2p2SelEntry& b) {
    return ::mkfitdev::v2p2SelBefore(a.score, a.ord, b.score, b.ord);
  }

  // m_sel reduced to its top n_max on the fly: a strict total order, so the kept set and its order do not depend on
  // the order the competitors arrive in, nor on how they are split and merged (std::partial_sort's order differs
  // only among equal scores)
  struct V2p2SelTop {
    V2p2SelEntry sel[::mkfitdev::kMaxCandsPerSeed];  // the best n_keep, best first
    int n_keep;
    int n_sel;       // competitors offered
    bool nan_score;  // a NaN score was offered
    // InLayer::reserve_hole_slot: the best competitor that declined the layer
    bool has_decliner;
    V2p2SelEntry best_decliner;
    bool hot_overflow;  // a survivor built by this lane did not fit in the HoT pool

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void offer_decliner(const V2p2SelEntry& e) {
      if (!has_decliner || v2p2SelEntryBefore(e, best_decliner)) {
        best_decliner = e;
        has_decliner = true;
      }
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void offer(const V2p2SelEntry& e, int n_max) {
      ++n_sel;
      if (std::isnan(e.score))
        nan_score = true;
      if (::mkfitdev::Config::V2p2::InLayer::reserve_hole_slot && e.node_idx < 0)
        offer_decliner(e);
      if (n_max <= 0)
        return;
      int j;
      if (n_keep < n_max) {
        j = n_keep++;
      } else {
        if (!v2p2SelEntryBefore(e, sel[n_max - 1]))
          return;
        j = n_max - 1;
      }
      while (j > 0 && v2p2SelEntryBefore(e, sel[j - 1])) {
        sel[j] = sel[j - 1];
        --j;
      }
      sel[j] = e;
    }

    // the top n_max of the competitors offered here and to o
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void merge(const V2p2SelTop& o, int n_max) {
      for (int k = 0; k < o.n_keep; ++k)
        offer(o.sel[k], n_max);
      n_sel += o.n_sel - o.n_keep;
      nan_score = nan_score || o.nan_score;
      if (o.has_decliner)
        offer_decliner(o.best_decliner);
    }
  };

  // The lanes that run select_and_materialise() for one seed: lane of n_lanes (a power of two), tops = their shared
  // V2p2SelTop[n_lanes]. Every thread of the block calls it together (block barriers inside).
  struct V2p2SelGroup {
    int lane;
    int n_lanes;
    V2p2SelTop* tops;
  };

  // n_max of select_and_materialise(): ccrep.m_max_cands > 0 ? ccrep.m_max_cands : ccand.capacity(); the device has
  // kMaxCandsPerSeed slots
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int sel_n_max(int max_cands, const V2p2LayerCtx& ctx) {
    return std::min(max_cands > 0 ? max_cands : ctx.capacity, int(::mkfitdev::kMaxCandsPerSeed));
  }

  // The m_sel entry of TrackCand ic: with a CandIncoming the hole it would record (unchanged if WSR_Outside, see below),
  // else the TrackCand unchanged; with best-short a stopped one leaves the beam instead (offered to best-short, false).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool sel_entry_of_tcand(SeedCandsRef& seed,
                                                              ::mkfitdev::V2p2CandSoA::ConstView cands,
                                                              int s,
                                                              int ic,
                                                              const V2p2LayerCtx& ctx,
                                                              V2p2SelEntry& e) {
    namespace InLayer = ::mkfitdev::Config::V2p2::InLayer;
    namespace Policy = ::mkfitdev::Config::V2p2::Policy;
    using ::mkfitdev::v2p2SelOrdinal;
    const int row = ::mkfitdev::v2p2CandRow(s, ic);
    if (cands.active(row)) {
      const TrackCandView tc{&seed.cands[ic], &seed.states[ic], &seed};

      // Every CandIncoming also competes as the hole it would record, including
      // those that found paths. In the log-likelihood score (Score::mode 1) a hole
      // adds exactly 0, whatever its kind; hole penalties apply in the final pick.
      // WSR_Outside is not a hole: the candidate competes unchanged, unless
      // V2p2Params::outside_hole_max_seed_hits makes it one for a short seed.
      const uint64_t ord = v2p2SelOrdinal(::mkfitdev::kV2p2SelHole, ic, 0, 0);
      if (Policy::use_wsr && cands.wsr(row) == ::mkfitdev::WSR_Outside && !outside_as_hole(tc, ctx.vp)) {
        e = V2p2SelEntry{ic, -1, 0, false, tc.score(), ord};
      } else {
        const int fake = fake_hit_index(tc, ::mkfitdev::WSRResult{cands.wsr(row), cands.inGap(row) != 0}, ctx);
        LayerStepFeatures f{};
        fill_step_geometry(f, 0.0f, ctx);
        f.hole_kind = ::mkfitdev::v2p2_hole_kind(fake);
        e = V2p2SelEntry{ic, -1, fake, true, tc.score() + ::mkfitdev::v2p2_layer_step_score(f, ctx.vp), ord};
      }
      return true;
    }

    // Stopped or pull-in-skipped TrackCands, unchanged, at their own score. With
    // best-short on (outward only), stopped ones leave the beam instead and the
    // best is kept on the CombCandidate.
    const bool best_short = InLayer::best_short && ctx.rz.is_outward();
    if (best_short && seed.hot(seed.cands[ic].lastHitIdx).index == ::mkfitdev::kHitStopIdx) {
      offer_best_short(seed, seed.cands[ic], seed.states[ic]);
      return false;
    }
    e = V2p2SelEntry{ic, -1, 0, false, seed.cands[ic].score, v2p2SelOrdinal(::mkfitdev::kV2p2SelUntouched, ic, 0, 0)};
    return true;
  }

  // The m_sel entry of the in-layer path ending at pool row ni of seed s (cands = its CandIncoming rows)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE V2p2SelEntry sel_entry_of_node(const SeedCandsRef& seed,
                                                                     ::mkfitdev::V2p2CandSoA::ConstView cands,
                                                                     int s,
                                                                     ::mkfitdev::V2p2NodeSoA::ConstView nodes,
                                                                     int ni,
                                                                     const V2p2LayerCtx& ctx) {
    const int ic = nodes.ic(ni);
    // m_feat of the node: the root's step geometry and the sums accumulated parent -> child
    LayerStepFeatures f{};
    fill_step_geometry(f, cands.logRho(::mkfitdev::v2p2CandRow(s, ic)), ctx);
    f.n_hits = nodes.nHits(ni);
    f.n_overlap = f.n_hits - 1;
    f.hole_kind = ::mkfitdev::V2P2_NoHole;
    f.chi2_sum = nodes.chi2Sum(ni);
    f.log_det_v_sum = nodes.logDetVSum(ni);
    return V2p2SelEntry{ic,
                        ni,
                        0,
                        false,
                        seed.cands[ic].score + ::mkfitdev::v2p2_layer_step_score(f, ctx.vp),
                        ::mkfitdev::v2p2SelOrdinal(::mkfitdev::kV2p2SelNode, ic, f.n_hits, nodes.pathKey(ni))};
  }

  // HoT nodes a survivor appends: its path, or its fake hit
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int sel_entry_hots(const V2p2SelEntry& e,
                                                         ::mkfitdev::V2p2NodeSoA::ConstView nodes) {
    return e.node_idx >= 0 ? int(nodes.nHits(e.node_idx)) : (e.add_fake ? 1 : 0);
  }

  // One survivor as a copy of its TrackCand plus its path (or fake hit), with its HoTs at pool nodes hot_node,
  // hot_node + 1, ... (the nodes seedAddHitIdx() gives when the survivors are built in rank order). Returns false if
  // a HoT is past the pool.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool materialise_survivor(SeedCandsRef& seed,
                                                                const V2p2SelEntry& e,
                                                                ::mkfitdev::V2p2NodeSoA::ConstView nodes,
                                                                const V2p2LayerCtx& ctx,
                                                                int hot_node,
                                                                CandBook& nc,
                                                                CandState& ns) {
    bool fits = true;
    nc = seed.cands[e.tcand_idx];
    ns = seed.states[e.tcand_idx];

    if (e.node_idx >= 0) {
      int chain[::mkfitdev::kV2p2MaxHitsPerLayer];
      int n_chain = 0;
      for (int ci = e.node_idx; ci >= 0 && n_chain < ::mkfitdev::kV2p2MaxHitsPerLayer; ci = nodes.parentNode(ci))
        chain[n_chain++] = ci;
      for (int c = n_chain - 1; c >= 0; --c) {
        const int n = chain[c];
        fits = seedSetHitIdx(seed, nc, hot_node++, nodes.hitIndex(n), nodes.hitLayer(n), nodes.chi2(n)) && fits;
        if (c != n_chain - 1) {
          ++nc.nOverlap;  // a second hit in one layer: an overlap, or the sister sensor of a paired entry
        }
      }
      // setState(): parameters, errors and charge of the path's last node
      const V2p2TrackState st = nodes.state(e.node_idx);
      for (std::size_t i = 0; i < std::size(st.par); ++i)
        ns.par[i] = st.par[i];
      for (std::size_t i = 0; i < std::size(st.err); ++i)
        ns.err[i] = st.err[i];
      ns.charge = st.charge;
    } else if (e.add_fake) {
      fits = seedSetHitIdx(seed, nc, hot_node, e.fake_hit, ctx.rz.m_layer_id, 0.0f);
    }

    // score_ holds the summed layer-step score during finding; the final-pick
    // scorer overwrites it in mergeCandsAndBestShortOne() at the end of the search.
    nc.score = e.score;
    return fits;
  }

  // select_and_materialise() for one seed by the lanes of g (all threads of the block call it; in_layer false: the
  // barriers only): the top max_cands by v2p2SelBefore into outBooks / outStates; returns their number (lane 0).
  // hot_pool: reserve(acc, need) before the survivors' HoT nodes are appended (lane 0), attach(seed) after (every
  // lane): the seed's HoT pool can grow (cands/CandSeedOpsKernels.h SeedHotPool).
  template <typename TAcc, typename THotPool>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int select_and_materialise(TAcc const& acc,
                                                            const V2p2SelGroup& g,
                                                            bool in_layer,
                                                            SeedCandsRef& seed,
                                                            const THotPool& hot_pool,
                                                            CandBook* outBooks,
                                                            CandState* outStates,
                                                            ::mkfitdev::V2p2CandSoA::ConstView cands,
                                                            int s,
                                                            ::mkfitdev::V2p2NodeSoA::ConstView nodes,
                                                            const int32_t* node_list,
                                                            int n_list,
                                                            int node_chain,
                                                            int max_cands,
                                                            const V2p2LayerCtx& ctx,
                                                            bool& nan_score) {
    namespace InLayer = ::mkfitdev::Config::V2p2::InLayer;
    const int n_max = in_layer ? sel_n_max(max_cands, ctx) : 0;
    const int n_tc = in_layer ? *seed.nCands : 0;
    // best-short writes the CombCandidate: one lane in TrackCand order then
    const int ic_first = InLayer::best_short ? (g.lane == 0 ? 0 : n_tc) : g.lane;
    const int ic_step = InLayer::best_short ? 1 : g.n_lanes;

    if (in_layer) {
      V2p2SelTop top{};
      // Every TrackCand (as a hole or unchanged), split over the lanes.
      for (int ic = ic_first; ic < n_tc; ic += ic_step) {
        V2p2SelEntry e;
        if (sel_entry_of_tcand(seed, cands, s, ic, ctx, e))
          top.offer(e, n_max);
      }
      // Every in-layer path: the list split over the lanes, then the chain.
      for (int k = g.lane; k < n_list; k += g.n_lanes)
        top.offer(sel_entry_of_node(seed, cands, s, nodes, node_list[k], ctx), n_max);
      if (g.lane == 0) {
        for (int ni = node_chain; ni >= 0; ni = nodes.nextInSeed(ni))
          top.offer(sel_entry_of_node(seed, cands, s, nodes, ni, ctx), n_max);
      }
      g.tops[g.lane] = top;
    }
    alpaka::syncBlockThreads(acc);
    for (int half = g.n_lanes / 2; half > 0; half /= 2) {
      if (in_layer && g.lane < half)
        g.tops[g.lane].merge(g.tops[g.lane + half], n_max);
      alpaka::syncBlockThreads(acc);
    }

    V2p2SelTop& top = g.tops[0];
    if (in_layer && g.lane == 0 && top.n_sel > 0) {
      // Reserve a hole slot: keep the best candidate that declined the layer, even
      // if outranked, so one branch stays open to an earlier hit being wrong.
      if (InLayer::reserve_hole_slot && top.n_keep > 1) {
        bool kept_a_decliner = false;
        for (int k = 0; k < top.n_keep && !kept_a_decliner; ++k)
          kept_a_decliner = (top.sel[k].node_idx < 0);
        if (!kept_a_decliner && top.has_decliner) {
          top.sel[top.n_keep - 1] = top.best_decliner;
        }
      }
      int hots = 0;
      for (int k = 0; k < top.n_keep; ++k)
        hots += sel_entry_hots(top.sel[k], nodes);
      hot_pool.reserve(acc, hots);
    }
    alpaka::syncBlockThreads(acc);
    if (in_layer)
      hot_pool.attach(seed);

    // Build the survivors as copies in the other buffer: several winners can descend from the same TrackCand --
    // that is what branching is -- so the sources have to stay readable while the copies are made. One survivor
    // per lane, its HoTs after those of the better survivors.
    const int n_keep = (in_layer && top.n_sel > 0) ? top.n_keep : 0;
    const int hot0 = in_layer ? *seed.nHots : 0;
    bool hot_overflow = false;
    for (int k = g.lane; k < n_keep; k += g.n_lanes) {
      int hot_node = hot0;
      for (int j = 0; j < k; ++j)
        hot_node += sel_entry_hots(top.sel[j], nodes);
      CandBook nc;
      CandState ns;
      hot_overflow = !materialise_survivor(seed, top.sel[k], nodes, ctx, hot_node, nc, ns) || hot_overflow;
      outBooks[k] = nc;
      outStates[k] = ns;
    }
    if (in_layer)
      g.tops[g.lane].hot_overflow = hot_overflow;
    alpaka::syncBlockThreads(acc);
    if (!in_layer || g.lane != 0)
      return 0;
    nan_score = top.nan_score;
    if (top.n_sel == 0)
      return 0;  // every TrackCand was offered to best-short and left the beam
    int hots = 0;
    for (int k = 0; k < n_keep; ++k) {
      hots += sel_entry_hots(top.sel[k], nodes);
      if (g.tops[k % g.n_lanes].hot_overflow)
        *seed.overflowBits |= ::mkfitdev::kOverflowHotsBit;
    }
    *seed.nHots = std::min(hot0 + hots, seed.hotCap);

    // Candidates that stopped IN this layer leave the beam the same way.
    const bool best_short = InLayer::best_short && ctx.rz.is_outward();
    int n_out = 0;
    for (int k = 0; k < n_keep; ++k) {
      if (best_short && seed.hot(outBooks[k].lastHitIdx).index == ::mkfitdev::kHitStopIdx) {
        offer_best_short(seed, outBooks[k], outStates[k]);
        continue;
      }
      if (n_out != k) {
        outBooks[n_out] = outBooks[k];
        outStates[n_out] = outStates[k];
      }
      ++n_out;
    }
    return n_out;
  }

  // offer_best_short(): keep the best-scoring stopped candidate of this CombCandidate. The score is
  // the in-flight accumulator, which is on one scale within a seed -- the only
  // comparison made here.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void offer_best_short(SeedCandsRef& seed,
                                                            const CandBook& book,
                                                            const CandState& state) {
    if (!*seed.bestShortValid || book.score > seed.bestShort->score) {
      *seed.bestShort = book;
      *seed.bestShortState = state;
      *seed.bestShortValid = 1;
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
