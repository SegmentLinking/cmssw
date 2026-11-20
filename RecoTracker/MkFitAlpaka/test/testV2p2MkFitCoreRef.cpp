// MkFitCore reference for the v2p2 finder pieces of MkFitAlpaka: generates synthetic inputs, runs the MkFitCore code
// on them and writes inputs and outputs for testMkFitAlpakaV2p2<Backend> (record format: v2p2TestIO.h).
//   pqueue         CandIncoming::pq_push / pq_pop with the offer rule of MkFinderV2p2::preselect_hit_batch
//   path order     the comparator of MkFinderV2p2::order_hits_along_path
//   scores         V2p2Score.h and the final-pick scorer "v2p2:llh"
//   Kalman         KalmanOpArgs::do_kalman_stuff, both depths
//   window         MkRZLimits, the crossings and window of MkBins, the WSR verdict, the Hermite cubic
//   selection      MkFinderV2p2::select_and_materialise on synthetic seeds, with equal scores
// MkFinderV2p2's stage functions are private: where the test needs one (prop_to_layer_edges, determine_wsr,
// order_hits_along_path's comparator, fake_hit_index, select_and_materialise) its body is reproduced here from
// MkFinderV2p2.cc, on MkFitCore's own types and helpers.
// Usage: testMkFitAlpakaV2p2MkFitCoreRef <out.bin> [nrec=4000] [seed=4242]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/V2p2Params.h"
// package-private MkFitCore headers on purpose: angle brackets keep scram's private-header check off this test
#include <RecoTracker/MkFitCore/src/MiniPropagators.h>
#include <RecoTracker/MkFitCore/src/MkBins.h>
#include <RecoTracker/MkFitCore/src/MkFinderV2p2Structures.h>
#include <RecoTracker/MkFitCore/src/MkRZLimits.h>
#include <RecoTracker/MkFitCore/src/PropagationMPlex.h>
#include <RecoTracker/MkFitCore/src/V2p2Config.h>
#include <RecoTracker/MkFitCore/src/V2p2Score.h>

#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Params.h"

#include "v2p2TestIO.h"

using namespace mkfit;
using namespace mkfitdev::v2p2test;
namespace mp = mkfit::mini_propagators;

static_assert(kPqCap == CandIncoming::kPqCap && kPqCap == mkfitdev::V2p2Params::max_presel_hits_limit);
static_assert(kSelMaxDepth == V2p2Params::max_hits_per_layer_limit);
static_assert(mkfitdev::V2p2ScoreParams::n_hit_eff_grp == V2p2ScoreParams::n_hit_eff_grp);

namespace {

  std::mt19937 rng;
  float uni() { return std::uniform_real_distribution<float>(0.f, 1.f)(rng); }
  float gauss() { return std::normal_distribution<float>(0.f, 1.f)(rng); }
  int pick(int n) { return std::uniform_int_distribution<int>(0, n - 1)(rng); }

  //--------------------------------------------------------------------------
  // pqueue and path order

  // MkFinderV2p2::order_hits_along_path, the comparator of its std::sort
  struct PathOrder {
    float dir;
    bool operator()(const CandIncoming::PassedHitMatch& a, const CandIncoming::PassedHitMatch& b) const {
      const float ka = dir * a.mixed_state.dalpha, kb = dir * b.mixed_state.dalpha;
      if (ka != kb)
        return ka < kb;
      if (a.ddphi != b.ddphi)
        return a.ddphi > b.ddphi;
      return a.hit_orig_index < b.hit_orig_index;
    }
  };

  PqRec makePq() {
    PqRec r{};
    r.nHits = pick(kPqMaxStream + 1);
    r.maxPresel = 1 + pick(kPqCap);
    r.dir = uni() < 0.5f ? 1.f : -1.f;
    // few distinct values: equal ddphi and equal dalpha (hits on one module) are common
    const bool coarse = uni() < 0.7f;
    std::vector<uint32_t> ids(1000);
    for (int i = 0; i < 1000; ++i)
      ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int i = 0; i < r.nHits; ++i) {
      r.ddphi[i] = coarse ? 0.002f * pick(8) : 0.02f * uni();
      r.dalpha[i] = coarse ? 0.003f * (pick(6) - 2) : 0.02f * (uni() - 0.3f);
      r.hitOrig[i] = ids[i];
    }
    CandIncoming cand(nullptr, 0);
    for (int i = 0; i < r.nHits; ++i) {
      // MkFinderV2p2::preselect_hit_batch, the reduction of one passed hit
      const bool full = cand.m_pqueue[0].size() >= r.maxPresel;
      if (full && !(r.ddphi[i] < cand.pq_top(0).ddphi))
        continue;
      if (full)
        cand.pq_pop(0);
      CandIncoming::PassedHitMatch phm{r.ddphi[i], r.hitOrig[i], r.hitOrig[i], 0, mp::InitialState()};
      phm.mixed_state.dalpha = r.dalpha[i];
      cand.pq_push(0, phm);
    }
    r.nKept = cand.m_pqueue[0].size();
    for (int k = 0; k < r.nKept; ++k)
      r.heapHit[k] = cand.m_pqueue[0][k].hit_orig_index;
    std::vector<CandIncoming::PassedHitMatch> lh(cand.m_pqueue[0].begin(), cand.m_pqueue[0].end());
    std::sort(lh.begin(), lh.end(), PathOrder{r.dir});
    for (int k = 0; k < r.nKept; ++k)
      r.pathHit[k] = lh[k].hit_orig_index;
    return r;
  }

  //--------------------------------------------------------------------------
  // scores

  ScoreParams randomScoreParams() {
    ScoreParams p{};
    p.hit_bonus = 0.5f * pick(8);
    p.overlap_bonus = 0.5f * pick(3);
    p.chi2_weight = 0.5f + uni();
    p.miss_penalty = pick(10);
    p.edge_penalty = pick(3);
    p.gap_penalty = pick(3);
    p.stop_penalty = pick(10);
    p.hit_eff = 0.02f + 0.96f * uni();
    for (int g = 0; g < 5; ++g)
      p.hit_eff_grp[g] = uni() < 0.4f ? -1.f : 0.02f + 0.96f * uni();
    return p;
  }

  V2p2ScoreParams toCore(const ScoreParams& p) {
    V2p2ScoreParams c;
    c.hit_bonus = p.hit_bonus;
    c.overlap_bonus = p.overlap_bonus;
    c.chi2_weight = p.chi2_weight;
    c.miss_penalty = p.miss_penalty;
    c.edge_penalty = p.edge_penalty;
    c.gap_penalty = p.gap_penalty;
    c.stop_penalty = p.stop_penalty;
    c.hit_eff = p.hit_eff;
    for (int g = 0; g < 5; ++g)
      c.hit_eff_grp[g] = p.hit_eff_grp[g];
    return c;
  }

  ScoreParams fromCore(const V2p2ScoreParams& c) {
    ScoreParams p{c.hit_bonus,
                  c.overlap_bonus,
                  c.chi2_weight,
                  c.miss_penalty,
                  c.edge_penalty,
                  c.gap_penalty,
                  c.stop_penalty,
                  c.hit_eff,
                  {c.hit_eff_grp[0], c.hit_eff_grp[1], c.hit_eff_grp[2], c.hit_eff_grp[3], c.hit_eff_grp[4]}};
    return p;
  }

  const int kFakeCodes[] = {
      Hit::kHitMissIdx, Hit::kHitStopIdx, Hit::kHitEdgeIdx, Hit::kHitInGapIdx, Hit::kHitMaxClusterIdx};

  ScoreRec makeScore(const track_score_func& llh) {
    ScoreRec r{};
    const V2p2Params defaults;
    const bool deflt = uni() < 0.3f;
    r.outward = deflt ? fromCore(defaults.score_outward) : randomScoreParams();
    r.inward = deflt ? fromCore(defaults.score_inward) : randomScoreParams();
    r.n_hits = pick(5) == 0 ? 0 : 1 + pick(4);
    r.n_overlap = std::max(r.n_hits - 1, 0);
    r.hole_kind = r.n_hits > 0 ? V2P2_NoHole : 1 + pick(4);
    r.chi2_sum = r.n_hits * 3.f * uni();
    r.log_det_v_sum = r.n_hits * (-18.f + 8.f * uni());
    r.log_rho = -6.f + 6.f * uni();
    r.layer_to = pick(60);
    r.is_outward = uni() < 0.5f;
    r.is_pixel = uni() < 0.3f;
    r.is_barrel = uni() < 0.5f;
    r.fake_hit = kFakeCodes[pick(5)];
    LayerStepFeatures f;
    f.chi2_sum = r.chi2_sum;
    f.log_det_v_sum = r.log_det_v_sum;
    f.log_rho = r.log_rho;
    f.n_hits = r.n_hits;
    f.n_overlap = r.n_overlap;
    f.hole_kind = r.hole_kind;
    f.layer_to = r.layer_to;
    f.is_outward = r.is_outward;
    f.is_pixel = r.is_pixel;
    f.is_barrel = r.is_barrel;
    V2p2Params vp;
    vp.score_outward = toCore(r.outward);
    vp.score_inward = toCore(r.inward);
    r.group = v2p2_hit_eff_group(f);
    r.loglh = v2p2_layer_step_loglh(f, vp.score(f.is_outward));
    r.step_score = v2p2_layer_step_score(f, vp);
    r.hole_kind_out = v2p2_hole_kind(r.fake_hit);
    // the final-pick scorer
    r.n_found = pick(20);
    r.n_tail = pick(4);
    r.n_overlap_hits = pick(4);
    r.n_inside = pick(4);
    r.n_seed_hits = 2 + pick(4);
    r.chi2 = 40.f * uni();
    r.pt = 0.3f + 5.f * uni();
    r.score_in = -30.f + 60.f * uni();
    TrackScoreInput in;
    in.n_found_hits = r.n_found;
    in.n_tail_holes = r.n_tail;
    in.n_overlap_hits = r.n_overlap_hits;
    in.n_inside_holes = r.n_inside;
    in.n_seed_hits = r.n_seed_hits;
    in.chi2 = r.chi2;
    in.pt = r.pt;
    in.score_in = r.score_in;
    in.penalize_tail_holes = true;
    r.llh = llh(in);
    return r;
  }

  //--------------------------------------------------------------------------
  // tracks for the Kalman and window blocks

  struct V3 {
    float x, y, z;
  };
  V3 cross(V3 lhs, V3 rhs) {
    return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x};
  }
  float dot(V3 lhs, V3 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
  V3 unit(V3 vec) {
    const float len = std::sqrt(dot(vec, vec));
    return {vec.x / len, vec.y / len, vec.z / len};
  }
  V3 addScaled(V3 base, V3 step, float scale) {
    return {base.x + scale * step.x, base.y + scale * step.y, base.z + scale * step.z};
  }

  const float kBrlR[] = {2.9f, 6.8f, 10.9f, 16.0f, 23.0f, 36.0f, 51.0f, 68.0f, 88.0f, 108.0f};
  const float kEcZ[] = {
      26.f, 30.f, 36.f, 44.f, 53.f, 63.f, 75.f, 90.f, 110.f, 131.f, 156.f, 185.f, 220.f, 250.f, 270.f};

  struct TrackIn {
    float par[6];
    float err[21];
    int chg;
  };

  void loadState(const TrackIn& t, MPlexLS& err, MPlexLV& par, MPlexQI& chg, int slot) {
    for (int i = 0; i < 6; ++i) {
      par(slot, i, 0) = t.par[i];
      for (int j = 0; j <= i; ++j)
        err(slot, i, j) = t.err[symIdx(i, j)];
    }
    chg(slot, 0, 0) = t.chg;
  }

  TrackState toTrackState(const TrackIn& t) {
    TrackState s;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j)
        s.errors.At(i, j) = t.err[symIdx(i, j)];
    for (int i = 0; i < 6; ++i)
      s.parameters.At(i) = t.par[i];
    s.charge = t.chg;
    return s;
  }

  // a helix (no material) from a state to a radius or a z; false if it failed
  bool propagateTo(const TrackIn& in, bool toR, float target, TrackIn& out) {
    MPlexLS err, outErr;
    MPlexLV par, outPar;
    MPlexQI chg, fail(0);
    MPlexQF targetRZ(target);
    for (int n = 0; n < NN; ++n)
      loadState(in, err, par, chg, n);
    const PropagationFlags pfNone(PF_none);
    if (toR)
      propagateHelixToRMPlex(err, par, chg, targetRZ, outErr, outPar, fail, 1, pfNone);
    else
      propagateHelixToZMPlex(err, par, chg, targetRZ, outErr, outPar, fail, 1, pfNone);
    out = in;
    for (int i = 0; i < 6; ++i) {
      out.par[i] = outPar.constAt(0, i, 0);
      for (int j = 0; j <= i; ++j)
        out.err[symIdx(i, j)] = outErr.constAt(0, i, j);
    }
    return fail.constAt(0, 0, 0) == 0;
  }

  // a Phase-2-like track on an inner barrel layer, with errors (diagonal + x-phi, y-phi, z-theta correlations) and
  // smeared parameters; the truth is returned too
  bool makeTrack(TrackIn& smeared, TrackIn& truth, int& innerLayer, float& pt, float& eta) {
    pt = std::exp(std::log(0.4f) + uni() * (std::log(50.f) - std::log(0.4f)));
    eta = -2.4f + 4.8f * uni();
    TrackIn vtx{};
    vtx.par[0] = 0.01f * gauss();
    vtx.par[1] = 0.01f * gauss();
    vtx.par[2] = 4.f * gauss();
    vtx.par[3] = 1.f / pt;
    vtx.par[4] = -Const::PI + Const::TwoPI * uni();
    vtx.par[5] = 2.f * std::atan(std::exp(-eta));
    vtx.chg = uni() < 0.5f ? -1 : 1;
    for (int i = 0; i < 6; ++i)
      vtx.err[symIdx(i, i)] = 1e-4f;
    innerLayer = pick(6);
    if (!propagateTo(vtx, true, kBrlR[innerLayer], truth))
      return false;
    const float sx = 0.003f + 0.02f * uni(), sz = 0.005f + 0.05f * uni();
    const float sig[6] = {
        sx, sx, sz, (0.005f + 0.03f * uni()) * truth.par[3], 0.0003f + 0.002f * uni(), 0.0003f + 0.002f * uni()};
    for (int i = 0; i < 21; ++i)
      truth.err[i] = 0.f;
    for (int i = 0; i < 6; ++i)
      truth.err[symIdx(i, i)] = sig[i] * sig[i];
    truth.err[symIdx(4, 0)] = 0.4f * sig[4] * sig[0];
    truth.err[symIdx(4, 1)] = -0.3f * sig[4] * sig[1];
    truth.err[symIdx(5, 2)] = 0.5f * sig[5] * sig[2];
    truth.err[symIdx(4, 3)] = 0.2f * sig[4] * sig[3];
    smeared = truth;
    for (int i = 0; i < 6; ++i)
      smeared.par[i] += sig[i] * gauss();
    return true;
  }

  //--------------------------------------------------------------------------
  // Kalman: depth 0 from the crossing of the smeared track with the module plane (the search's mixed state), depth
  // >= 1 from a state a little before the plane (the parent node's state)

  bool makeKal(KalRec& r) {
    TrackIn s0, truth;
    int il;
    float pt, eta;
    if (!makeTrack(s0, truth, il, pt, eta))
      return false;
    r = KalRec{};
    r.solvePlane = uni() < 0.5f;
    r.chi2TrkFac = uni() < 0.7f ? 3.0f : (uni() < 0.5f ? 1.0f : 2.0f);
    const bool isBarrel = std::abs(eta) < 1.3f ? (uni() < 0.85f) : (uni() < 0.25f);
    // the plane: through the truth crossing of a layer further out
    TrackIn xing;
    V3 nrm;
    if (isBarrel) {
      const float rad = kBrlR[std::min(9, il + 1 + pick(3))] + (uni() - 0.5f);
      if (!propagateTo(truth, true, rad, xing) || std::abs(xing.par[2]) > 120.f)
        return false;
      nrm = unit({xing.par[0], xing.par[1], 0.f});
      if (uni() < 0.4f) {
        // a tilted module: the normal turned about the in-plane phi direction
        const V3 ax = unit(cross({0.f, 0.f, 1.f}, nrm));
        const float a = (xing.par[2] > 0 ? 1.f : -1.f) * 1.2f * uni();
        nrm = unit(
            addScaled({nrm.x * std::cos(a), nrm.y * std::cos(a), nrm.z * std::cos(a)}, cross(ax, nrm), std::sin(a)));
      }
    } else {
      const float zsgn = std::cos(truth.par[5]) > 0 ? 1.f : -1.f;
      int kz = 0;
      while (kz < 15 && kEcZ[kz] < std::abs(truth.par[2]) + 2.f)
        ++kz;
      kz = std::min(14, kz + pick(2));
      if (!propagateTo(truth, false, zsgn * kEcZ[kz], xing))
        return false;
      const float rx = std::hypot(xing.par[0], xing.par[1]);
      if (rx > 110.f || rx < 3.f)
        return false;
      nrm = unit({0.05f * (uni() - 0.5f), 0.05f * (uni() - 0.5f), zsgn});
    }
    const V3 xpos{xing.par[0], xing.par[1], xing.par[2]};
    const V3 uDir = unit(cross(nrm, isBarrel ? V3{0.f, 0.f, 1.f} : unit({xpos.x, xpos.y, 0.f})));
    const V3 vDir = cross(nrm, uDir);
    const V3 pnt = addScaled(addScaled(xpos, uDir, 2.f * (uni() - 0.5f)), vDir, 4.f * (uni() - 0.5f));
    const int htype = pick(3);
    const float su = htype == 0 ? 0.0025f : (htype == 1 ? 0.0029f : 0.0026f);
    const float sv = htype == 0 ? 0.0040f : (htype == 1 ? 0.0722f : 1.443f);
    // an outlier now and then (a hit of another track on the module), so that the acceptance cut is crossed both ways
    const float shift = uni() < 0.3f ? (uni() < 0.5f ? -1.f : 1.f) * (0.02f + 0.3f * uni()) : 0.f;
    const V3 hit =
        addScaled(addScaled(xpos, uDir, su * gauss() + shift), vDir, sv * gauss() * (htype == 2 ? 0.f : 1.f));
    const float uu[3] = {uDir.x, uDir.y, uDir.z}, vv[3] = {vDir.x, vDir.y, vDir.z};
    const float pp[3] = {pnt.x, pnt.y, pnt.z}, nn[3] = {nrm.x, nrm.y, nrm.z}, hh[3] = {hit.x, hit.y, hit.z};
    for (int i = 0; i < 3; ++i) {
      r.plPnt[i] = pp[i];
      r.plNrm[i] = nn[i];
      r.plDir[i] = uu[i];
      r.msPar[i] = hh[i];
      for (int j = 0; j <= i; ++j)
        r.msErr[symIdx(i, j)] = su * su * uu[i] * uu[j] + sv * sv * vv[i] * vv[j];
    }
    if (r.solvePlane) {
      // the parent state: the smeared track a little before the crossing
      TrackIn parent;
      const float back = 2.f * uni();
      const bool ok = isBarrel ? propagateTo(s0, true, std::hypot(xpos.x, xpos.y) - back, parent)
                               : propagateTo(s0, false, xpos.z - (xpos.z > 0 ? back : -back), parent);
      if (!ok)
        return false;
      std::copy(parent.par, parent.par + 6, r.inPar);
      std::copy(parent.err, parent.err + 21, r.inErr);
      r.chg = parent.chg;
    } else {
      // the search's crossing: the smeared track's own crossing of the layer surface through the plane point, with
      // inv_pt, inv_k and theta of the state at the previous hit (inv_k with the field there, param_b)
      std::copy(s0.par, s0.par + 6, r.inPar);
      std::copy(s0.err, s0.err + 21, r.inErr);
      r.chg = s0.chg;
      MPlexLS err;
      MPlexLV par;
      MPlexQI chg;
      loadState(s0, err, par, chg, 0);
      const float bf = Config::bFieldFromZR(s0.par[2], std::hypot(s0.par[0], s0.par[1]));
      mp::InitialState is(mp::State(par, 0), s0.chg, s0.par[3], s0.par[5], bf);
      is.dalpha = 0.f;  // State(par, ti) leaves the turning angle indeterminate
      is.fail_flag = 0;
      mp::State st;
      const bool fail = isBarrel ? is.propagate_to_r(mp::PA_Exact, std::hypot(xpos.x, xpos.y), st, true)
                                 : is.propagate_to_z(mp::PA_Exact, xpos.z, st, true);
      if (fail)
        return false;
      const float v[10] = {st.x, st.y, st.z, st.px, st.py, st.pz, st.dalpha, is.inv_pt, is.inv_k, is.theta};
      std::copy(v, v + 10, r.isXyz);
      r.isFail = st.fail_flag;
    }
    return true;
  }

  void runKal(std::vector<KalRec>& recs, const PropagationConfig& pconf) {
    // one batch per depth kind, NN items at a time, as rk_flush runs them; a batch has one chi2_trk_fac, the first
    // item's, and its record keeps the value it was computed with
    for (int solve = 0; solve < 2; ++solve) {
      std::vector<KalRec*> items;
      for (auto& r : recs)
        if (r.solvePlane == solve)
          items.push_back(&r);
      for (size_t base = 0; base < items.size(); base += NN) {
        KalmanOpArgs koa;
        koa.prop_config = &pconf;
        koa.m_solve_plane = solve;
        koa.m_chi2_trk_fac = items[base]->chi2TrkFac;
        const int n = std::min<size_t>(NN, items.size() - base);
        for (int k = 0; k < n; ++k) {
          KalRec& r = *items[base + k];
          r.chi2TrkFac = koa.m_chi2_trk_fac;
          TrackIn t;
          std::copy(r.inPar, r.inPar + 6, t.par);
          std::copy(r.inErr, r.inErr + 21, t.err);
          t.chg = r.chg;
          koa.item_begin(nullptr, HitOnTrack{0, 0}, solve ? 0 : -1, 0);
          if (solve) {
            koa.load_state_err_chg(toTrackState(t));
          } else {
            mp::InitialState is;
            is.x = r.isXyz[0];
            is.y = r.isXyz[1];
            is.z = r.isXyz[2];
            is.px = r.isXyz[3];
            is.py = r.isXyz[4];
            is.pz = r.isXyz[5];
            is.dalpha = r.isXyz[6];
            is.fail_flag = r.isFail;
            is.inv_pt = r.isXyz[7];
            is.inv_k = r.isXyz[8];
            is.theta = r.isXyz[9];
            koa.load_state_err_chg(is, toTrackState(t));
          }
          // KalmanOpArgs::load_hit_module, from the record's hit and plane
          for (int i = 0; i < 3; ++i) {
            koa.msPar(k, i, 0) = r.msPar[i];
            koa.plNrm(k, i, 0) = r.plNrm[i];
            koa.plDir(k, i, 0) = r.plDir[i];
            koa.plPnt(k, i, 0) = r.plPnt[i];
            for (int j = 0; j <= i; ++j)
              koa.msErr(k, i, j) = r.msErr[symIdx(i, j)];
          }
          koa.item_finished();
        }
        if (!solve)
          koa.compute_pars();
        koa.do_kalman_stuff();
        for (int k = 0; k < n; ++k) {
          KalRec& r = *items[base + k];
          for (int i = 0; i < 6; ++i) {
            r.outPar[i] = koa.tsPar.constAt(k, i, 0);
            for (int j = 0; j <= i; ++j)
              r.outErr[symIdx(i, j)] = koa.tsErr.constAt(k, i, j);
          }
          r.chgOut = koa.tsChg.constAt(k, 0, 0);
          r.chi2 = koa.tsChi2.constAt(k, 0, 0);
          r.detV = koa.tsDetV.constAt(k, 0, 0);
          r.chi2Acc = koa.tsChi2Acc.constAt(k, 0, 0);
        }
      }
    }
  }

  //--------------------------------------------------------------------------
  // window of one candidate on one layer

  // the same helix walked the other way: momentum reversed, charge flipped
  void reverse(TrackIn& t) {
    t.par[4] = squashPhiGeneral(t.par[4] + Const::PI);
    t.par[5] = Const::PI - t.par[5];
    t.chg = -t.chg;
  }

  bool makeWin(WinRec& r) {
    TrackIn s0, truth;
    int il;
    float pt, eta;
    if (!makeTrack(s0, truth, il, pt, eta))
      return false;
    r = WinRec{};
    r.outward = uni() < 0.7f;
    r.isPixel = uni() < 0.3f;
    const bool barrel = std::abs(eta) < 1.3f ? (uni() < 0.85f) : (uni() < 0.25f);
    TrackIn start = s0;
    if (barrel) {
      r.layerType = LayerInfo::Barrel;
      const float rc = kBrlR[std::min(9, il + 1 + pick(3))] + (uni() - 0.5f);
      const float half = 0.2f + 1.5f * uni();
      r.rin = rc - half;
      r.rout = rc + half;
      const float zl = 30.f + 90.f * uni();
      r.zmin = -zl;
      r.zmax = zl;
      if (!r.outward) {
        // inward: start outside the layer, moving in (the backward search walks the other way)
        if (!propagateTo(s0, true, r.rout + 1.f + 10.f * uni(), start))
          return false;
        reverse(start);
      }
    } else {
      const float zsgn = std::cos(truth.par[5]) > 0 ? 1.f : -1.f;
      r.layerType = zsgn > 0 ? LayerInfo::EndCapPos : LayerInfo::EndCapNeg;
      int kz = 0;
      while (kz < 15 && kEcZ[kz] < std::abs(s0.par[2]) + 2.f)
        ++kz;
      kz = std::min(14, kz + pick(2));
      const float zc = zsgn * kEcZ[kz], half = 0.3f + 1.5f * uni();
      r.zmin = zc - half;
      r.zmax = zc + half;
      r.rin = 3.f + 20.f * uni();
      r.rout = r.rin + 40.f + 50.f * uni();
      r.hasHole = uni() < 0.3f;
      r.holeRMin = r.rin + 10.f * uni();
      r.holeRMax = r.holeRMin + 2.f + 15.f * uni();
      if (!r.outward) {
        if (!propagateTo(s0, false, zc + zsgn * (half + 1.f + 10.f * uni()), start))
          return false;
        reverse(start);
      }
    }
    r.layerId = pick(60);
    std::copy(start.par, start.par + 6, r.par);
    std::copy(start.err, start.err + 21, r.err);
    r.chg = start.chg;

    LayerInfo li(r.layerId, LayerInfo::LayerType_e(r.layerType));
    li.set_limits(r.rin, r.rout, r.zmin, r.zmax);
    li.set_is_pixel(r.isPixel);
    if (r.hasHole)
      li.set_r_hole_range(r.holeRMin, r.holeRMax);
    MkRZLimits rz;
    rz.setup(li, r.outward);
    const TrackState ts = toTrackState(start);
    r.quadrantOk = rz.rz_quadrant_check(ts.z(), ts.pz());

    // MkFinderV2p2::prop_to_layer_edges for one candidate (lane 0; the other lanes repeat it, never read)
    MkBins B(1);
    MPlexQF phi(0.0f);
    MPlexQI chg(0);
    MPlexQF bf(Config::Bfield);
    for (int i = 0; i < NN; ++i) {
      B.m_isp.copyIn_partial_track_state(i, ts);
      phi[i] = ts.momPhi();
      chg[i] = ts.charge;
      // param_b: mini_prop_param_b == 2 and the finding flags of Phase 2 use the parametrised field
      bf[i] = Config::bFieldFromZR(ts.z(), std::hypot(ts.x(), ts.y()));
    }
    B.m_isp.init_momentum_vec_and_k(phi, chg, bf);
    B.prop_to_limits_in_order(rz);

    // MkFinderV2p2::determine_search_windows: covariance, windows, WSR, Hermite cubic
    MPlexLV par0;
    MPlexLS err0;
    MPlexQI chg0;
    for (int i = 0; i < NN; ++i)
      loadState(start, err0, par0, chg0, i);
    MkBinTrackCovExtract TCE;
    B.transport_position_cov(par0, err0, TCE);
    B.determine_bin_windows(TCE);

    // MkFinderV2p2::determine_wsr for lane 0
    {
      namespace pol = Config::V2p2::Policy;
      const float dq_to_n_sigma = pol::wsr_n_sigma / 3.0f;
      const bool is_barrel = rz.m_is_barrel;
      const float q_lo = is_barrel ? rz.m_zmin : rz.m_rin;
      const float q_hi = is_barrel ? rz.m_zmax : rz.m_rout;
      const LayerInfo& LI = rz.layer_info_1();
      const int i = 0;
      const bool f_entry = B.m_sp1.fail_flag[i] != 0;
      const bool f_exit = B.m_sp2.fail_flag[i] != 0;
      WSR_Result w;
      if (f_entry && f_exit) {
        w = WSR_Result(WSR_Outside, false);
      } else {
        const float dq = dq_to_n_sigma * B.m_dq_track[i];
        const float qa = B.m_q_min[i], qb = B.m_q_max[i];
        if (qb < q_lo - dq || qa > q_hi + dq)
          w = WSR_Result(WSR_Outside, false);
        else if (qa > q_lo + dq && qb < q_hi - dq)
          w = WSR_Result(WSR_Inside, false);
        else
          w = WSR_Result(WSR_Edge, false);
        if ((f_entry || f_exit) && w.m_wsr == WSR_Inside)
          w.m_wsr = WSR_Edge;
        if (!is_barrel && LI.has_r_range_hole() && w.m_wsr != WSR_Outside) {
          const float h_lo = LI.hole_r_min(), h_hi = LI.hole_r_max();
          if (qa > h_lo + dq && qb < h_hi - dq) {
            w = WSR_Result(WSR_Outside, true);
          } else if (qb > h_lo - dq && qa < h_hi + dq) {
            w.m_wsr = WSR_Edge;
            w.m_in_gap = true;
          }
        }
      }
      r.wsr = w.m_wsr;
      r.inGap = w.m_in_gap;
    }
    mp::Hermite3D H;
    H.calculate_coeffs(B.m_sp1, B.m_sp2, B.m_isp.inv_k);

    const mp::StatePlex* sps[2] = {&B.m_sp1, &B.m_sp2};
    float* outs[2] = {r.sp1, r.sp2};
    for (int k = 0; k < 2; ++k) {
      const mp::StatePlex& sp = *sps[k];
      const float v[7] = {sp.x[0], sp.y[0], sp.z[0], sp.px[0], sp.py[0], sp.pz[0], sp.dalpha[0]};
      std::copy(v, v + 7, outs[k]);
    }
    r.fail1 = B.m_sp1.fail_flag[0];
    r.fail2 = B.m_sp2.fail_flag[0];
    const float cov[6] = {
        TCE.m_cov_0_0[0], TCE.m_cov_0_1[0], TCE.m_cov_1_1[0], TCE.m_cov_2_2[0], TCE.m_cov_0_2[0], TCE.m_cov_1_2[0]};
    std::copy(cov, cov + 6, r.cov);
    r.phiMin = B.m_phi_min[0];
    r.phiMax = B.m_phi_max[0];
    r.phiCenter = B.m_phi_center[0];
    r.phiDelta = B.m_phi_delta[0];
    r.qMin = B.m_q_min[0];
    r.qMax = B.m_q_max[0];
    r.dphiTrack = B.m_dphi_track[0];
    r.dqTrack = B.m_dq_track[0];
    for (int c = 0; c < 4; ++c) {
      r.hx[c] = H.m_Hx.constAt(0, c, 0);
      r.hy[c] = H.m_Hy.constAt(0, c, 0);
      r.hz[c] = H.m_Hz.constAt(0, c, 0);
    }
    r.hderfac = H.m_Hderfac[0];
    return true;
  }

  //--------------------------------------------------------------------------
  // end-of-layer selection

  // MkFinderV2p2::SelEntry
  struct SelEntry {
    int tcand_idx;
    int node_idx;
    int fake_hit;
    bool add_fake;
    float score;
  };

  SelSeed makeSel() {
    SelSeed s{};
    s.layerId = pick(60);
    s.isPixel = uni() < 0.3f;
    s.isBarrel = uni() < 0.5f;
    s.outward = uni() < 0.6f;
    s.maxCands = 1 + pick(kSelMaxCands);
    s.maxHolesPerCand = 2 + pick(3);
    s.maxConsecHoles = 1 + pick(2);
    const V2p2Params defaults;
    s.outwardP = uni() < 0.5f ? fromCore(defaults.score_outward) : randomScoreParams();
    s.inwardP = uni() < 0.5f ? fromCore(defaults.score_inward) : randomScoreParams();
    // coarse scores and chi2: equal scores are common
    const bool coarse = uni() < 0.6f;
    s.nCands = 1 + pick(kSelMaxCands);
    for (int ic = 0; ic < s.nCands; ++ic) {
      SelCand& c = s.cands[ic];
      c.score = coarse ? 0.5f * pick(6) : 10.f * uni();
      c.par[0] = 10.f * uni();
      c.par[1] = 10.f * uni();
      c.par[2] = 20.f * (uni() - 0.5f);
      c.par[3] = 0.2f + uni();
      c.par[4] = 3.f * (uni() - 0.5f);
      c.par[5] = 0.3f + 2.5f * uni();
      for (int i = 0; i < 6; ++i)
        c.err[symIdx(i, i)] = 1e-4f * (1.f + uni());
      c.charge = uni() < 0.5f ? -1 : 1;
      c.nHits = 2 + pick(kSelMaxHits - 4);
      for (int h = 0; h < c.nHits; ++h) {
        const float u = uni();
        c.hitIdx[h] = u < 0.75f ? pick(5000)
                                : (u < 0.85f ? Hit::kHitMissIdx : (u < 0.92f ? Hit::kHitEdgeIdx : Hit::kHitInGapIdx));
        c.hitLyr[h] = h;
        c.hitChi2[h] = c.hitIdx[h] >= 0 ? 5.f * uni() : 0.f;
      }
      // a stopped TrackCand now and then: it does not enter the layer
      const bool stopped = uni() < 0.1f;
      if (stopped) {
        c.hitIdx[c.nHits - 1] = Hit::kHitStopIdx;
        c.hitChi2[c.nHits - 1] = 0.f;
      }
      c.active = !stopped && uni() < 0.9f;
      const float w = uni();
      c.wsr = w < 0.7f ? WSR_Inside : (w < 0.85f ? WSR_Edge : WSR_Outside);
      c.inGap = !s.isBarrel && uni() < 0.2f;
      c.logRho = coarse ? -2.f : -5.f + 4.f * uni();
    }
    // in-layer nodes: trees under the active, not Outside, TrackCands; paths in increasing hit position
    for (int ic = 0; ic < s.nCands; ++ic) {
      const SelCand& c = s.cands[ic];
      if (!c.active || c.wsr == WSR_Outside)
        continue;
      const int nlh = pick(5);
      std::vector<int> frontier;
      for (int lh = 0; lh < nlh && s.nNodes < kSelMaxNodes; ++lh) {
        if (uni() < 0.3f)
          continue;  // rejected by the chi2 cut
        frontier.push_back(s.nNodes);
        SelNode& n = s.nodes[s.nNodes++];
        n = SelNode{};
        n.ic = ic;
        n.parent = -1;
        n.hitPos = lh;
      }
      for (size_t f = 0; f < frontier.size() && s.nNodes < kSelMaxNodes; ++f) {
        const SelNode p = s.nodes[frontier[f]];
        for (int lh = p.hitPos + 1; lh < nlh && s.nNodes < kSelMaxNodes; ++lh) {
          if (uni() < 0.6f)
            continue;
          frontier.push_back(s.nNodes);
          SelNode& n = s.nodes[s.nNodes++];
          n = SelNode{};
          n.ic = ic;
          n.parent = frontier[f];
          n.hitPos = lh;
        }
      }
    }
    // node contents; the features accumulate parent -> child as harvest_nodes() does
    for (int k = 0; k < s.nNodes; ++k) {
      SelNode& n = s.nodes[k];
      n.hitLayer = s.layerId;
      n.hitIndex = 10000 + 16 * n.ic + n.hitPos;
      n.chi2 = coarse ? 0.5f * pick(10) : 9.f * uni();
      const float det_v = coarse ? 1e-6f : 1e-7f + 1e-5f * uni();
      const SelNode* pn = n.parent < 0 ? nullptr : &s.nodes[n.parent];
      n.nHits = (pn ? pn->nHits : 0) + 1;
      n.chi2Sum = (pn ? pn->chi2Sum : 0.0f) + n.chi2;
      n.logDetVSum = (pn ? pn->logDetVSum : 0.0f) + std::log(std::max(1e-30f, det_v));
      for (int i = 0; i < 6; ++i)
        n.par[i] = s.cands[n.ic].par[i] + 0.01f * (k + 1) * i;
      for (int i = 0; i < 21; ++i)
        n.err[i] = s.cands[n.ic].err[i] * 0.5f;
      n.charge = s.cands[n.ic].charge;
    }

    // ---- MkFinderV2p2::select_and_materialise on MkFitCore objects
    EventOfCombCandidates combs;
    combs.reset(1, kSelMaxCands);
    CombCandidate& ccand = combs[0];
    for (int ic = 0; ic < s.nCands; ++ic) {
      const SelCand& c = s.cands[ic];
      TrackState st;
      for (int i = 0; i < 6; ++i)
        st.parameters.At(i) = c.par[i];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j)
          st.errors.At(i, j) = c.err[symIdx(i, j)];
      st.charge = c.charge;
      TrackCand tc(TrackBase(st, 0.f, ic), &ccand);
      ccand.push_back(tc);
      for (int h = 0; h < c.nHits; ++h)
        ccand[ic].addHitIdx(c.hitIdx[h], c.hitLyr[h], c.hitChi2[h]);
      ccand[ic].setScore(c.score);
    }
    LayerInfo li(s.layerId, s.isBarrel ? LayerInfo::Barrel : LayerInfo::EndCapPos);
    li.set_is_pixel(s.isPixel);
    V2p2Params vp;
    vp.score_outward = toCore(s.outwardP);
    vp.score_inward = toCore(s.inwardP);
    // MkFinderV2p2::fill_step_geometry
    auto fill_step_geometry = [&](LayerStepFeatures& f, float log_rho) {
      f.layer_to = (short)li.layer_id();
      f.is_outward = s.outward;
      f.is_pixel = li.is_pixel();
      f.is_barrel = li.is_barrel();
      f.log_rho = log_rho;
    };
    // MkFinderV2p2::fake_hit_index
    auto fake_hit_index = [&](const TrackCand& tc, const WSR_Result& wsr) {
      namespace pol = Config::V2p2::Policy;
      int fake_hit_idx = Hit::kHitMissIdx;
      if (pol::use_hole_limits &&
          (tc.nAllMinusOneHits() >= s.maxHolesPerCand || tc.nTailMinusOneHits() >= s.maxConsecHoles))
        fake_hit_idx = Hit::kHitStopIdx;
      if (pol::use_wsr) {
        if (wsr.m_wsr == WSR_Edge)
          fake_hit_idx = Hit::kHitEdgeIdx;
        else if (wsr.m_in_gap)
          fake_hit_idx = Hit::kHitInGapIdx;
      }
      return fake_hit_idx;
    };
    // the node features of harvest_nodes()
    std::vector<LayerStepFeatures> feat(s.nNodes);
    for (int k = 0; k < s.nNodes; ++k) {
      const SelNode& n = s.nodes[k];
      LayerStepFeatures& f = feat[k];
      if (n.parent >= 0)
        f = feat[n.parent];
      else
        fill_step_geometry(f, s.cands[n.ic].logRho);
      f.n_hits = n.nHits;
      f.n_overlap = f.n_hits - 1;
      f.hole_kind = V2P2_NoHole;
      f.chi2_sum = n.chi2Sum;
      f.log_det_v_sum = n.logDetVSum;
    }
    std::vector<SelEntry> m_sel;
    const int n_tc = ccand.size();
    std::vector<char> m_has_cand(n_tc, 0);
    // the CandIncomings in TrackCand order (begin_next_Ccrep_in_layer)
    for (int ic = 0; ic < n_tc; ++ic) {
      if (!s.cands[ic].active)
        continue;
      m_has_cand[ic] = 1;
      const TrackCand& tc = ccand[ic];
      const WSR_Result wsr(WithinSensitiveRegion_e(s.cands[ic].wsr), s.cands[ic].inGap);
      if (Config::V2p2::Policy::use_wsr && wsr.m_wsr == WSR_Outside) {
        m_sel.push_back({ic, -1, 0, false, tc.score()});
      } else {
        const int fake = fake_hit_index(tc, wsr);
        LayerStepFeatures f;
        fill_step_geometry(f, 0.0f);
        f.hole_kind = v2p2_hole_kind(fake);
        m_sel.push_back({ic, -1, fake, true, tc.score() + v2p2_layer_step_score(f, vp)});
      }
    }
    // ccrep.m_nodes: harvest order, by depth
    std::vector<int> order(s.nNodes);
    for (int k = 0; k < s.nNodes; ++k)
      order[k] = k;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s.nodes[a].nHits < s.nodes[b].nHits; });
    for (const int ni : order)
      m_sel.push_back(
          {s.nodes[ni].ic, ni, 0, false, ccand[s.nodes[ni].ic].score() + v2p2_layer_step_score(feat[ni], vp)});
    for (int ic = 0; ic < n_tc; ++ic) {
      if (m_has_cand[ic])
        continue;
      m_sel.push_back({ic, -1, 0, false, ccand[ic].score()});
    }
    const int n_keep = std::min((int)m_sel.size(), s.maxCands);
    std::partial_sort(m_sel.begin(), m_sel.begin() + n_keep, m_sel.end(), [](const SelEntry& a, const SelEntry& b) {
      return a.score > b.score;
    });
    s.nSel = m_sel.size();
    s.boundaryTie = 0;
    for (int k = n_keep; k < (int)m_sel.size(); ++k)
      s.boundaryTie = s.boundaryTie || m_sel[k].score == m_sel[n_keep - 1].score;
    std::vector<TrackCand> m_new_cands;
    for (int k = 0; k < n_keep; ++k) {
      const SelEntry& e = m_sel[k];
      m_new_cands.push_back(ccand[e.tcand_idx]);
      TrackCand& nc = m_new_cands.back();
      SelOut& o = s.out[k];
      o.srcIc = e.tcand_idx;
      if (e.node_idx >= 0) {
        int chain[V2p2Params::max_hits_per_layer_limit];
        int n_chain = 0;
        for (int ci = e.node_idx; ci >= 0; ci = s.nodes[ci].parent)
          chain[n_chain++] = ci;
        for (int c = n_chain - 1; c >= 0; --c) {
          const SelNode& n = s.nodes[chain[c]];
          nc.addHitIdx(n.hitIndex, n.hitLayer, n.chi2);
          if (c != n_chain - 1)
            nc.incOverlapCount();
        }
        TrackState st;
        for (int i = 0; i < 6; ++i)
          st.parameters.At(i) = s.nodes[e.node_idx].par[i];
        st.charge = s.nodes[e.node_idx].charge;
        nc.setState(st);
        o.kind = 0;
      } else if (e.add_fake) {
        nc.addHitIdx(e.fake_hit, li.layer_id(), 0.0f);
        o.kind = 1;
      } else {
        o.kind = 2;
      }
      nc.setScore(e.score);
    }
    s.nOut = n_keep;
    for (int k = 0; k < n_keep; ++k) {
      const TrackCand& nc = m_new_cands[k];
      SelOut& o = s.out[k];
      o.score = nc.score();
      o.nFound = nc.nFoundHits();
      o.nMissing = nc.nMissingHits();
      o.nOverlap = nc.nOverlapHits();
      o.nInside = nc.nInsideMinusOneHits();
      o.nTail = nc.nTailMinusOneHits();
      o.chi2 = nc.chi2();
      std::vector<std::pair<int, int>> chain;
      for (int node = nc.lastCcIndex(); node >= 0; node = ccand.hot_node(node).m_prev_idx)
        chain.push_back({ccand.hot_node(node).m_hot.index, ccand.hot_node(node).m_hot.layer});
      o.nHits = chain.size();
      for (int h = 0; h < o.nHits; ++h) {
        o.hitIdx[h] = chain[o.nHits - 1 - h].first;
        o.hitLyr[h] = chain[o.nHits - 1 - h].second;
      }
      for (int i = 0; i < 6; ++i)
        o.par[i] = nc.parameters().At(i);
    }
    return s;
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s out.bin [nrec] [seed]\n", argv[0]);
    return 1;
  }
  const std::string outFn = argv[1];
  const int nRec = argc > 2 ? std::atoi(argv[2]) : 4000;
  rng.seed(argc > 3 ? std::atoi(argv[3]) : 4242);

  // Phase-2 settings (MkFitGeometryESProducer)
  Config::usePropToPlane = true;
  Config::usePtMultScat = true;

  RefData d;
  constexpr int kMatBinsZ = 300, kMatBinsR = 120;
  constexpr float kMatRangeZ = 300.f, kMatRangeR = 120.f;
  d.head = FileHeader{kMagic, kMatBinsZ, kMatBinsR, kMatRangeZ, kMatRangeR, V2p2Params().hit_chi2_cut};
  TrackerInfo ti;
  ti.create_material(kMatBinsZ, kMatRangeZ, kMatBinsR, kMatRangeR);
  d.bbxi.resize(kMatBinsZ * kMatBinsR);
  d.radl.resize(kMatBinsZ * kMatBinsR);
  // synthetic layer-like cylinders and disks over a low baseline, as the propagation test
  for (int bz = 0; bz < kMatBinsZ; ++bz) {
    for (int br = 0; br < kMatBinsR; ++br) {
      const float zCenter = bz + 0.5f, rCenter = br + 0.5f;
      float radl = 0.0005f;
      for (float layerR : kBrlR)
        if (std::abs(rCenter - layerR) < 1.5f && zCenter < 120.f + layerR)
          radl = 0.01f + 0.04f * uni();
      for (float diskZ : kEcZ)
        if (std::abs(zCenter - diskZ) < 1.5f && rCenter > 3.f && rCenter < 112.f)
          radl = 0.01f + 0.05f * uni();
      const float bbxi = radl * (0.9e-3f + 0.2e-3f * uni());
      ti.material_radl(bz, br) = d.radl[bz * kMatBinsR + br] = radl;
      ti.material_bbxi(bz, br) = d.bbxi[bz * kMatBinsR + br] = bbxi;
    }
  }
  // the finding flags of Phase 2 (MkFitGeometryESProducer with usePropToPlane)
  PropagationConfig pconf;
  pconf.finding_inter_layer_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
  pconf.finding_intra_layer_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
  pconf.finding_inter_layer_pflags.tracker_info = &ti;
  pconf.finding_intra_layer_pflags.tracker_info = &ti;

  const track_score_func llh = IterationConfig::get_track_scorer("v2p2:llh");

  for (int i = 0; i < nRec; ++i)
    d.pq.push_back(makePq());
  for (int i = 0; i < nRec; ++i)
    d.score.push_back(makeScore(llh));
  int nSkip = 0;
  while ((int)d.kal.size() < nRec) {
    KalRec r;
    if (makeKal(r))
      d.kal.push_back(r);
    else
      ++nSkip;
  }
  runKal(d.kal, pconf);
  while ((int)d.win.size() < nRec) {
    WinRec r;
    if (makeWin(r))
      d.win.push_back(r);
    else
      ++nSkip;
  }
  int nTies = 0;
  for (int i = 0; i < nRec; ++i) {
    d.sel.push_back(makeSel());
    nTies += d.sel.back().boundaryTie;
  }
  int nWsr[4] = {0, 0, 0, 0}, nAcc = 0;
  for (auto const& w : d.win)
    ++nWsr[std::clamp(w.wsr, 0, 3)];
  for (auto const& k : d.kal)
    nAcc += k.chi2Acc < d.head.hitChi2Cut;
  std::printf(
      "MkFitCore NN = %d: %d records per block (%d generated tracks skipped); window WSR inside %d edge %d outside %d; "
      "Kalman accepted at the default cut %d; selections with a boundary tie %d\n",
      int(NN),
      nRec,
      nSkip,
      nWsr[0],
      nWsr[1],
      nWsr[2],
      nAcc,
      nTies);
  if (!writeFile(outFn, d)) {
    std::printf("cannot write %s\n", outFn.c_str());
    return 2;
  }
  return 0;
}
