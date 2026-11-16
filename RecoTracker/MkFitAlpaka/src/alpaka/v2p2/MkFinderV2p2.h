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

#include <algorithm>
#include <cmath>
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
#include "RecoTracker/MkFitAlpaka/src/alpaka/Packers.h"
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

  // TrackCand::StopReason_e, the value of stop_cuts_at_pickup()
  enum V2p2StopReason : int { SR_NotStopped = 0, SR_MinPt, SR_Looper, SR_TooManyHoles };

  // Declared ahead of their callers, defined below in the order of MkFinderV2p2.cc.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE V2p2StopReason stop_cuts_at_pickup(const TrackCandView& tc,
                                                                         const V2p2LayerCtx& ctx);
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool long_step_to_layer(const TrackCandView& tc, const V2p2LayerCtx& ctx);
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_wsr(V2p2CandWindow& w, const V2p2LayerCtx& ctx);
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
  // Phase 1 to 4 for one CandIncoming (process_layer_batch, LayerBatch one candidate wide)

  // Phase 1 -- which candidates, and where do they meet the layer.
  // prop_to_layer_edges(): m_isp from the TrackCand state, the two layer crossings
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void prop_to_layer_edges(V2p2CandWindow& w,
                                                               const TrackCandView& tc,
                                                               const V2p2LayerCtx& ctx) {
    MkBins<1>& B = w.B;
    B.m_n_proc = 1;

    float phi[1] = {0.0f};
    int chg[1] = {0};
    float bf[1] = {Config::Bfield};

    // Copy in x, y,z, invpT, theta.
    B.m_isp.copyIn_partial_track_state(0, tc.state->par);
    phi[0] = tc.momPhi();
    chg[0] = tc.charge();
    if (ctx.param_b)
      bf[0] = Config::bFieldFromZR(ctx.bfield, tc.z(), std::hypot(tc.x(), tc.y()));
    B.m_isp.init_momentum_vec_and_k(phi, chg, bf);

    // m_sp1 at the layer's near bounding surface, m_sp2 at the far one.
    B.prop_to_limits_in_order(ctx.rz);
  }

  // Phase 2 -- the search windows. The position block of the track covariance is
  // transported to m_sp2 (MkBins::transport_position_cov()); from it come
  // dphi_track and dq_track, and from those the WSR verdict and the binnor
  // ranges. The function also builds the Hermite cubic through the two layer
  // crossings. See doc/MkFinderV2p2-DesignNotes.md, "Search window".
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_search_windows(V2p2CandWindow& w,
                                                                    const TrackCandView& tc,
                                                                    const ::mkfitdev::LayerOfHitsAccess& L,
                                                                    const V2p2LayerCtx& ctx) {
    MkBins<1>& B = w.B;

    // Covariance position block at m_sp2, transported from the previous hit.
    MPlexLV<1> par0;
    MPlexLS<1> err0;
    par0.copyIn(0, tc.state->par);
    err0.copyIn(0, tc.state->err);
    B.transport_position_cov(par0, err0, w.TCE);

    B.determine_bin_windows(w.TCE);

    // The WSR edge fuzz uses dq_track, so the verdict follows determine_bin_windows().
    determine_wsr(w, ctx);

    B.find_bin_ranges(L, w.BL_p);

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
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void determine_wsr(V2p2CandWindow& w, const V2p2LayerCtx& ctx) {
    namespace Policy = ::mkfitdev::Config::V2p2::Policy;
    using ::mkfitdev::WSR_Edge;
    using ::mkfitdev::WSR_Inside;
    using ::mkfitdev::WSR_Outside;
    // m_dq_track is 3 sigma.
    const float dq_to_n_sigma = Policy::wsr_n_sigma / 3.0f;

    const MkBins<1>& B = w.B;
    const int i = 0;

    const MkRZLimits& LI = ctx.rz;
    const bool is_barrel = LI.m_is_barrel;
    const float q_lo = is_barrel ? LI.m_zmin : LI.m_rin;
    const float q_hi = is_barrel ? LI.m_zmax : LI.m_rout;

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

    w.wsr = wr;
  }

  // Phase 3 -- walk the bin ranges and pre-select hits.
  // select_hits(), the per-candidate part before the bin walk: line pre-cut and density region of the layer
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void select_hits_setup(V2p2CandWindow& w,
                                                             const ::mkfitdev::LayerOfHitsAccess& L) {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    namespace PreCut = ::mkfitdev::Config::V2p2::PreCut;
    const MkBins<1>& B = w.B;
    const int i = 0;

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
        pc_phi1 = vdt::fast_atan2f(y1, x1);
        pc_gphi = std::clamp(squashPhiGeneral(vdt::fast_atan2f(y2, x2) - pc_phi1) / dqb, -20.0f, 20.0f);
        pc_tol_q = PreCut::dq_slack * Window::dq_trk_fac * B.m_dq_track[i] * (1.0f + pc_gq * pc_gq);
        pc_tol_phi = PreCut::dphi_slack * Window::dphi_trk_fac * B.m_dphi_track[i];
      }
    }
    w.pc_on = pc_on;
    w.pc_qb1 = pc_qb1;
    w.pc_q1 = pc_q1;
    w.pc_gq = pc_gq;
    w.pc_phi1 = pc_phi1;
    w.pc_gphi = pc_gphi;
    w.pc_tol_q = pc_tol_q;
    w.pc_tol_phi = pc_tol_phi;

    // The region rho counts over and divides by, see the end of this function: the
    // window with the cut's padding of this sub-layer, as the fetch.
    const float phi_hit_term =
        Window::phi_per_hit ? Window::dphi_hit_fac * L.max_hit_phi_half_extent() : Window::dphi_flat_rad;
    const float rho_dphi = std::min(Window::dphi_trk_fac * B.m_dphi_track[i] + phi_hit_term,
                                    float(kMaxHalfPhiWindow));  // by value: no ODR-use in device code
    const float rho_dq = Window::dq_trk_fac * B.m_dq_track[i] + Window::dq_hit_fac * L.max_hit_q_half_length();
    // LayerBatch::rho_dphi / rho_dq start at zero
    w.rho_dphi = std::max(0.0f, rho_dphi);
    w.rho_dq = std::max(0.0f, rho_dq);
    w.rho_phi_c = 0.5f * (B.m_phi_min[i] + B.m_phi_max[i]);
    w.rho_phi_half = 0.5f * (B.m_phi_max[i] - B.m_phi_min[i]) + rho_dphi;
    w.rho_q_lo = B.m_q_min[i] - rho_dq;
    w.rho_q_hi = B.m_q_max[i] + rho_dq;
  }

  // select_hits(), the bin walk of q row qi (q outer loop, BL_p.q1 <= qi < BL_p.q2): its phi bins p1 .. p2 (masked)
  // as at most two contiguous internal-index segments in walk order; returns their number (0 when p1 == p2)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int select_hits_row_segments(const V2p2CandWindow& w,
                                                                   const ::mkfitdev::LayerOfHitsAccess& L,
                                                                   uint16_t qi,
                                                                   V2p2HitSegment (&seg)[2]) {
    // The hits of one q row are in phi-bin order (binnor sort key: q bin, then phi), so the walk over consecutive
    // phi bins visits consecutive internal indices; an empty bin stores first = 0 and is skipped.
    const uint16_t p1 = w.BL_p.p1[0], p2 = w.BL_p.p2[0];
    if (p1 == p2)
      return 0;
    // [p1, p2), or across the phi wrap [p1, end of ring) then [0, p2)
    const int nRange = (p1 < p2 || p2 == 0) ? 1 : 2;
    const int lo[2] = {p1, 0};
    const int hi[2] = {p1 < p2 ? int(p2) : int(::mkfitdev::kNPhiBins), int(p2)};
    int n = 0;
    for (int r = 0; r < nRange; ++r) {
      int a = lo[r];
      while (a < hi[r] && ::mkfitdev::binCount(L.binContent(a, qi)) == 0)
        ++a;
      if (a == hi[r])
        continue;
      int b = hi[r] - 1;
      while (::mkfitdev::binCount(L.binContent(b, qi)) == 0)
        --b;
      const uint32_t ca = L.binContent(a, qi), cb = L.binContent(b, qi);
      seg[n].begin = ::mkfitdev::binFirst(ca);
      seg[n].end = ::mkfitdev::binFirst(cb) + ::mkfitdev::binCount(cb);
      ++n;
    }
    return n;
  }

  // select_hits(), per walked hit (internal index hi): the density count (++n_scanned in the region) and the line
  // pre-cut; true = the hit goes to preselect_hit_batch(). No hit mask on the device (clustersToSkip is empty).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool select_hits_scan_hit(const V2p2CandWindow& w,
                                                                const ::mkfitdev::LayerOfHitsAccess& L,
                                                                uint32_t hi,
                                                                int& n_scanned) {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    namespace PreCut = ::mkfitdev::Config::V2p2::PreCut;
    if (std::abs(squashPhiGeneral(L.hit_phi(hi) - w.rho_phi_c)) < w.rho_phi_half && L.hit_q(hi) > w.rho_q_lo &&
        L.hit_q(hi) < w.rho_q_hi)
      ++n_scanned;

    if (w.pc_on) {
      const bool barrel = w.B.m_is_barrel;
      const float dqb_h = L.hit_qbar(hi) - w.pc_qb1;
      const float qbar_term = barrel ? PreCut::qbar_fac * L.hit_qbar_half_extent(hi) : 0.0f;
      bool reject = false;
      if (PreCut::q) {
        const float dev = std::abs(L.hit_q(hi) - (w.pc_q1 + dqb_h * w.pc_gq));
        reject = dev > w.pc_tol_q + Window::dq_hit_fac * L.hit_q_half_length(hi) + std::abs(w.pc_gq) * qbar_term;
      }
      if (!reject && PreCut::phi) {
        const float dev = std::abs(squashPhiGeneral(L.hit_phi(hi) - (w.pc_phi1 + dqb_h * w.pc_gphi)));
        const float hit_term =
            Window::phi_per_hit ? Window::dphi_hit_fac * L.hit_phi_half_extent(hi) : Window::dphi_flat_rad;
        reject = dev > w.pc_tol_phi + hit_term + std::abs(w.pc_gphi) * qbar_term;
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
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float select_hits_log_rho(const V2p2CandWindow& w, int n_scanned) {
    const MkBins<1>& B = w.B;
    const int i = 0;
    const float r = std::max(0.1f, std::hypot(B.m_sp2.x[i], B.m_sp2.y[i]));
    const float w_phi = (B.m_phi_max[i] - B.m_phi_min[i]) + 2.0f * w.rho_dphi;
    const float w_q = (B.m_q_max[i] - B.m_q_min[i]) + 2.0f * w.rho_dq;
    const float area = std::max(1e-4f, w_phi * r * w_q);
    return std::log(std::max(1e-6f, (n_scanned + 1.0f) / area));
  }

  // surface_referenced_dq() -- the track's q error referenced to the hit's
  // module plane rather than to the fixed path length it was transported to:
  //     v = e_q - ((e_q.p^)/(n^.p^)) n^ ,   sigma_q^2 = v^T C v
  // with n^ the module normal. Returns 3 sigma_q, the convention of
  // MkBins::m_dq_track, which is also the fallback. See
  // doc/MkFinderV2p2-DesignNotes.md, "Search window".
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

  // Phase 3b -- pre-select one batch of (candidate, hit) pairs.
  //
  // Solves the Hermite cubic onto each hit's own module plane, applies the
  // dq / dphi pre-selection cut; the caller pushes survivors into the
  // candidate's bounded priority queue (pq_offer()).
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void preselect_hit_batch(const V2p2CandWindow& w,
                                                               const ::mkfitdev::LayerOfHitsAccess& L,
                                                               const EngineHitInputs& hits,
                                                               const uint32_t (&hit_idcs)[N],
                                                               int n,
                                                               bool (&pass)[N],
                                                               V2p2PassedHitMatch (&phm)[N]) {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    namespace mp = mini_propagators;
    const MkBins<1>& B = w.B;
    const MkBinTrackCovExtract<1>& TCE = w.TCE;

    // Zeroed: the lanes from n up are not filled, and init_coeffs() reads
    // all of them.
    MPlexHV<N> module_pos(0.0f);
    MPlexHV<N> module_norm(0.0f);
    mp::Hermite3D<N> h3d;
    for (int h = 0; h < N; ++h) {
      for (int k = 0; k < 4; ++k)
        h3d.m_Hx[k][h] = h3d.m_Hy[k][h] = h3d.m_Hz[k][h] = 0.0f;
      h3d.m_Hderfac[h] = 0.0f;
    }

    // Extract hit / target module data
    uint32_t hit_orig_idcs[N];
    for (int h = 0; h < n; ++h) {
      hit_orig_idcs[h] = L.getOriginalHitIndex(hit_idcs[h]);
      const int row = ::mkfitdev::pack::hitRow(hits, L.layer, int(hit_orig_idcs[h]));
      const uint32_t mid = ::mkfitdev::hitpack::detIDinLayer(hits.packed[row]);
      const ::mkfitdev::EngineModule& mi = hits.modules[hits.layers[L.layer].moduleBegin + int(mid)];
      for (int k = 0; k < 3; ++k) {
        module_pos.At(h, k, 0) = mi.pnt[k];
        module_norm.At(h, k, 0) = mi.nrm[k];
      }

      h3d.copyIn(h, w.H, 0);
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

    // The cut is taken on h3_state, the cubic solved onto the hit's module plane.
    for (int h = 0; h < N; ++h) {
      pass[h] = false;
      if (h >= n)
        continue;
      float q;
      if (B.m_is_barrel) {
        q = h3_state.z[h];
      } else {
        q = hipo(h3_state.x[h], h3_state.y[h]);
      }
      const float ddq = std::abs(q - L.hit_q(hit_idcs[h]));
      const float phi = vdt::fast_atan2f(h3_state.y[h], h3_state.x[h]);
      const float ddphi = cdist(std::abs(phi - L.hit_phi(hit_idcs[h])));

      // q error referenced to this module's plane, see surface_referenced_dq().
      const float dq_trk = Window::surface_q ? surface_referenced_dq(B.m_dq_track[0],
                                                                     TCE,
                                                                     0,
                                                                     h3_state.x[h],
                                                                     h3_state.y[h],
                                                                     h3_state.px[h],
                                                                     h3_state.py[h],
                                                                     h3_state.pz[h],
                                                                     module_norm.At(h, 0, 0),
                                                                     module_norm.At(h, 1, 0),
                                                                     module_norm.At(h, 2, 0),
                                                                     B.m_is_barrel)
                                             : B.m_dq_track[0];

      // The pre-selection cut. See doc/MkFinderV2p2-DesignNotes.md, "Search window".
      const float dq_cut = Window::dq_trk_fac * dq_trk + Window::dq_hit_fac * L.hit_q_half_length(hit_idcs[h]);
      const float dphi_cut =
          Window::dphi_trk_fac * B.m_dphi_track[0] +
          (Window::phi_per_hit ? Window::dphi_hit_fac * L.hit_phi_half_extent(hit_idcs[h]) : Window::dphi_flat_rad);
      pass[h] = ddq < dq_cut && ddphi < dphi_cut;

      // For every hit, not only those that pass.
      h3_state.dalpha[h] = B.m_sp1.dalpha[0] + h3dop.m_T[h] * (B.m_sp2.dalpha[0] - B.m_sp1.dalpha[0]);

      // InitialState(h3_state, h, is_plex, h): the crossing, with inv_pt, inv_k and theta of m_isp
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
                                            B.m_isp.inv_pt[0],
                                            B.m_isp.inv_k[0],
                                            B.m_isp.theta[0]};
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
  // order_hits_along_path(): the pqueue survivors pq[0, n) into layer_hits[0, n) in path order, by bounded insertion
  // with the comparator of MkFitCore's std::sort (a total order: identical result)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void order_hits_along_path(const V2p2CandWindow& w,
                                                                 const V2p2PassedHitMatch* pq,
                                                                 int n,
                                                                 V2p2PassedHitMatch* layer_hits) {
    // Path order for the in-layer search: dalpha is monotone in path length
    // for one candidate, but it is a signed turning angle, so its sign goes
    // with the charge. m_sp1 is the near bounding surface and m_sp2 the far
    // one in the search direction, so dalpha times the sign of its change
    // from m_sp1 to m_sp2 grows along the path, in either search direction.
    // See doc/MkFinderV2p2-DesignNotes.md, "Reduction and hit ordering".
    const float dir = w.B.m_sp2.dalpha[0] >= w.B.m_sp1.dalpha[0] ? 1.0f : -1.0f;
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

  // harvest_nodes(): the acceptance, chi2_acc < hit_chi2_cut (false for NaN)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool harvest_accept(float chi2_acc, const ::mkfitdev::V2p2Params& vp) {
    return chi2_acc < vp.hit_chi2_cut;  // also rejects NaN
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
                                                                 bool& nan_score) {
    namespace InLayer = ::mkfitdev::Config::V2p2::InLayer;
    namespace Policy = ::mkfitdev::Config::V2p2::Policy;
    using ::mkfitdev::kMaxCandsPerSeed;
    using ::mkfitdev::v2p2CandRow;
    using ::mkfitdev::v2p2SelOrdinal;
    // max_cands = ccrep.m_max_cands > 0 ? ccrep.m_max_cands : ccand.capacity(); the device has kMaxCandsPerSeed slots
    const int n_max = std::min(max_cands > 0 ? max_cands : ctx.capacity, int(kMaxCandsPerSeed));
    nan_score = false;

    // m_sel reduced to its top n_max on the fly: a strict total order, so the kept set and its order do not
    // depend on the order the competitors arrive in (std::partial_sort's order differs only among equal scores)
    V2p2SelEntry sel[kMaxCandsPerSeed];
    int n_sel = 0;
    int n_keep = 0;
    // InLayer::reserve_hole_slot: the best competitor that declined the layer
    V2p2SelEntry best_decliner{-1, -1, 0, false, 0.0f, 0};
    bool has_decliner = false;
    auto offer = [&](const V2p2SelEntry& e) {
      ++n_sel;
      if (!(e.score == e.score))
        nan_score = true;
      if (InLayer::reserve_hole_slot && e.node_idx < 0 && (!has_decliner || v2p2SelEntryBefore(e, best_decliner))) {
        best_decliner = e;
        has_decliner = true;
      }
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
    };

    // Which TrackCands got a CandIncoming this layer. The rest are entered
    // unchanged below.
    const int n_tc = *seed.nCands;

    for (int ic = 0; ic < n_tc; ++ic) {
      const int row = v2p2CandRow(s, ic);
      if (!cands.active(row))
        continue;

      const TrackCandView tc{&seed.cands[ic], &seed.states[ic], &seed};

      // Every CandIncoming also competes as the hole it would record, including
      // those that found paths. In the log-likelihood score (Score::mode 1) a hole
      // adds exactly 0, whatever its kind; hole penalties apply in the final pick.
      // WSR_Outside is not a hole: the candidate competes unchanged.
      const uint64_t ord = v2p2SelOrdinal(::mkfitdev::kV2p2SelHole, ic, 0, 0);
      if (Policy::use_wsr && cands.wsr(row) == ::mkfitdev::WSR_Outside) {
        offer(V2p2SelEntry{ic, -1, 0, false, tc.score(), ord});
      } else {
        const int fake = fake_hit_index(tc, ::mkfitdev::WSRResult{cands.wsr(row), cands.inGap(row) != 0}, ctx);
        LayerStepFeatures f{};
        fill_step_geometry(f, 0.0f, ctx);
        f.hole_kind = ::mkfitdev::v2p2_hole_kind(fake);
        offer(V2p2SelEntry{ic, -1, fake, true, tc.score() + ::mkfitdev::v2p2_layer_step_score(f, ctx.vp), ord});
      }
    }

    // nodes: the seed's list is bounded by the pool, every row appears once
    for (int ni = head; ni >= 0; ni = nodes.nextInSeed(ni)) {
      const int ic = nodes.ic(ni);
      // m_feat of the node: the root's step geometry and the sums accumulated parent -> child
      LayerStepFeatures f{};
      fill_step_geometry(f, cands.logRho(v2p2CandRow(s, ic)), ctx);
      f.n_hits = nodes.nHits(ni);
      f.n_overlap = f.n_hits - 1;
      f.hole_kind = ::mkfitdev::V2P2_NoHole;
      f.chi2_sum = nodes.chi2Sum(ni);
      f.log_det_v_sum = nodes.logDetVSum(ni);
      offer(V2p2SelEntry{ic,
                         ni,
                         0,
                         false,
                         seed.cands[ic].score + ::mkfitdev::v2p2_layer_step_score(f, ctx.vp),
                         v2p2SelOrdinal(::mkfitdev::kV2p2SelNode, ic, f.n_hits, nodes.pathKey(ni))});
    }

    // Stopped or pull-in-skipped TrackCands, unchanged, at their own score. With
    // best-short on (outward only), stopped ones leave the beam instead and the
    // best is kept on the CombCandidate.
    const bool best_short = InLayer::best_short && ctx.rz.is_outward();
    for (int ic = 0; ic < n_tc; ++ic) {
      if (cands.active(v2p2CandRow(s, ic)))
        continue;
      if (best_short && seed.hot(seed.cands[ic].lastHitIdx).index == ::mkfitdev::kHitStopIdx) {
        offer_best_short(seed, seed.cands[ic], seed.states[ic]);
        continue;
      }
      offer(V2p2SelEntry{
          ic, -1, 0, false, seed.cands[ic].score, v2p2SelOrdinal(::mkfitdev::kV2p2SelUntouched, ic, 0, 0)});
    }

    if (n_sel == 0) {
      return 0;  // every TrackCand was offered to best-short and left the beam
    }

    // Reserve a hole slot: keep the best candidate that declined the layer, even
    // if outranked, so one branch stays open to an earlier hit being wrong.
    if (InLayer::reserve_hole_slot && n_keep > 1) {
      bool kept_a_decliner = false;
      for (int k = 0; k < n_keep && !kept_a_decliner; ++k)
        kept_a_decliner = (sel[k].node_idx < 0);
      if (!kept_a_decliner && has_decliner) {
        sel[n_keep - 1] = best_decliner;
      }
    }

    // Build the survivors as copies in the other buffer: several winners can descend from the same TrackCand --
    // that is what branching is -- so the sources have to stay readable while the copies are made.
    // Candidates that stopped IN this layer leave the beam the same way.
    int n_out = 0;
    for (int k = 0; k < n_keep; ++k) {
      const V2p2SelEntry& e = sel[k];
      CandBook nc = seed.cands[e.tcand_idx];
      CandState ns = seed.states[e.tcand_idx];

      if (e.node_idx >= 0) {
        int chain[::mkfitdev::kV2p2MaxHitsPerLayer];
        int n_chain = 0;
        for (int ci = e.node_idx; ci >= 0 && n_chain < ::mkfitdev::kV2p2MaxHitsPerLayer; ci = nodes.parentNode(ci))
          chain[n_chain++] = ci;
        for (int c = n_chain - 1; c >= 0; --c) {
          const int n = chain[c];
          seedAddHitIdx(seed, nc, nodes.hitIndex(n), nodes.hitLayer(n), nodes.chi2(n));
          if (c != n_chain - 1) {
            ++nc.nOverlap;  // a second hit in one layer: an overlap, or the sister sensor of a paired entry
          }
        }
        // setState(): parameters, errors and charge of the path's last node
        const V2p2TrackState st = nodes.state(e.node_idx);
        for (int i = 0; i < 6; ++i)
          ns.par[i] = st.par[i];
        for (int i = 0; i < 21; ++i)
          ns.err[i] = st.err[i];
        ns.charge = st.charge;
      } else if (e.add_fake) {
        seedAddHitIdx(seed, nc, e.fake_hit, ctx.rz.m_layer_id, 0.0f);
      }

      // score_ holds the summed layer-step score during finding; the final-pick
      // scorer overwrites it in mergeCandsAndBestShortOne() at the end of the search.
      nc.score = e.score;

      if (best_short && seed.hot(nc.lastHitIdx).index == ::mkfitdev::kHitStopIdx) {
        offer_best_short(seed, nc, ns);
        continue;
      }
      outBooks[n_out] = nc;
      outStates[n_out] = ns;
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
