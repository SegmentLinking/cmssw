// Runs the Alpaka port of the v2p2 finder pieces on the inputs written by testMkFitAlpakaV2p2MkFitCoreRef and compares
// with the MkFitCore outputs stored in the same file (record format: test/v2p2TestIO.h).
//   pqueue + path order  pq_offer / pq_push / pq_pop and order_hits_along_path: identical heap array and path order
//   scores               V2p2Score.h and trackScore(V2p2Llh / Keep): identical group and hole kind, scores at the
//                        rounding level
//   Kalman               KalmanOpArgs<N>::do_kalman_stuff: the gate of the propagation test, acceptance agreement
//                        (harvest_accept); a slot alone and in a full batch bitwise equal, also in the backward fit
//   harvest_accept       the acceptance on negative chi2 / chi2_acc, det V <= 0 and NaN
//   density region       across phi = +-pi: the hits counted (select_hits_setup, select_hits_scan_hit) and the area
//                        (select_hits_log_rho), the area capped at the full turn
//   phi-margin cap       Window::dphi_max_rad in the fetch, the line pre-cut and the density region
//   window factors       V2p2Params' track terms of the window (default and non-default) at the same call sites
//   outside hole         outside_as_hole: WSR_Outside as a hole for short seeds (outside_hole_max_seed_hits)
//   atan2_poly           against std::atan2 over every octant, at the axes and at (0, 0)
//   window               MkRZLimits, prop_to_layer_edges, MkBins, determine_wsr, Hermite3D: the gate of the
//                        propagation test, the discrete outcomes agree almost always
//   selection            select_and_materialise: the same new TrackCands as MkFitCore up to the order of equal
//                        scores, and the same result whatever the order of the seed's node list; on the lanes of a
//                        block, with equal and NaN scores, the one-lane result
//   HoT pool             the home rows and extension slots of the seeds' HoT pool: a seed that outgrows its home rows
//                        takes a slot; with none left it fails at its home rows and is counted; the selection on a
//                        real pool (survivors past the home rows, several lanes) gives the fixed-pool result; a chain
//                        with an extension exports its hits in order
//   filter + kept rows   the in-place filter, the CPU kept-row list (written up to the capacity) and the backward fit
//                        over it: every surviving seed is fitted once, whatever the list buffer held before
//   recount              the backward fit's outlier rejection on hand-built chains: the hole and overlap counts after
//                        it (inside and tail holes, overlaps, one hole per layer crossing, seed hits)
// Usage: testMkFitAlpakaV2p2<Backend> <mkfitcore_ref.bin>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/tracks/TrackSoADeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsDeviceCollection.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Score.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2SoA.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/SeedsAlgo.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/bkfit/BkFitKernel.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PlaneSteps.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationFlags.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkFinderV2p2Structures.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkFinderV2p2.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/V2p2Kernels.h"
#include "RecoTracker/MkFitAlpaka/test/v2p2TestIO.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ::mkfitdev::v2p2test;
namespace mkd = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev;
using ::mkfitdev::prop::PropagationFlags;

static_assert(kPqCap == ::mkfitdev::kV2p2MaxPreselHits);
static_assert(kSelMaxCands == ::mkfitdev::kMaxCandsPerSeed);
static_assert(kSelMaxDepth == ::mkfitdev::kV2p2MaxHitsPerLayer);

namespace {

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  constexpr int kThreads = 128;
  constexpr int kN = 1;
#else
  constexpr int kThreads = 1;
  constexpr int kN = mkd::kNN;
#endif

  ALPAKA_FN_HOST_ACC ::mkfitdev::V2p2ScoreParams toDev(const ScoreParams& p) {
    ::mkfitdev::V2p2ScoreParams d{};
    d.hit_bonus = p.hit_bonus;
    d.overlap_bonus = p.overlap_bonus;
    d.chi2_weight = p.chi2_weight;
    d.miss_penalty = p.miss_penalty;
    d.edge_penalty = p.edge_penalty;
    d.gap_penalty = p.gap_penalty;
    d.stop_penalty = p.stop_penalty;
    d.hit_eff = p.hit_eff;
    for (int g = 0; g < ::mkfitdev::V2p2ScoreParams::n_hit_eff_grp; ++g)
      d.hit_eff_grp[g] = p.hit_eff_grp[g];
    return d;
  }

  template <typename T>
  auto toDevice(Queue& queue, const std::vector<T>& v) {
    auto h = cms::alpakatools::make_host_buffer<T[]>(queue, v.size());
    std::copy(v.begin(), v.end(), h.data());
    auto d = cms::alpakatools::make_device_buffer<T[]>(queue, v.size());
    alpaka::memcpy(queue, d, h);
    return d;
  }

  template <typename T, typename TBuf>
  std::vector<T> toHost(Queue& queue, const TBuf& d, int n) {
    auto h = cms::alpakatools::make_host_buffer<T[]>(queue, n);
    alpaka::memcpy(queue, h, d);
    alpaka::wait(queue);
    return std::vector<T>(h.data(), h.data() + n);
  }

  template <typename TKernel, typename... TArgs>
  void launch(Queue& queue, int nItems, TKernel kernel, TArgs&&... args) {
    if (nItems == 0)
      return;
    const int blocks = cms::alpakatools::divide_up_by(nItems, kThreads);
    alpaka::exec<Acc1D>(
        queue, cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads), kernel, std::forward<TArgs>(args)...);
  }

  //--------------------------------------------------------------------------
  // pqueue + path order

  struct PqOut {
    int nKept;
    uint32_t heapHit[kPqCap];
    uint32_t pathHit[kPqCap];
  };

  struct KernelPq {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, const PqRec* recs, PqOut* outs, int n) const {
      for (int32_t r : cms::alpakatools::uniform_elements(acc, n)) {
        const PqRec& in = recs[r];
        ::mkfitdev::V2p2PassedHitMatch pq[kPqCap];
        int size = 0;
        for (int i = 0; i < in.nHits; ++i) {
          ::mkfitdev::V2p2PassedHitMatch e{};
          e.ddphi = in.ddphi[i];
          e.hit_orig_index = in.hitOrig[i];
          e.hit_index = in.hitOrig[i];
          e.mixed_state.dalpha = in.dalpha[i];
          size = mkd::pq_offer(pq, size, in.maxPresel, e);
        }
        PqOut& out = outs[r];
        out.nKept = size;
        for (int k = 0; k < size; ++k)
          out.heapHit[k] = pq[k].hit_orig_index;
        // the path direction comes from the two crossings' turning angles
        mkd::V2p2CandWindow<1> w;
        w.B.m_sp1.dalpha[0] = 0.f;
        w.B.m_sp2.dalpha[0] = in.dir;
        ::mkfitdev::V2p2PassedHitMatch lh[kPqCap];
        mkd::order_hits_along_path(w, 0, pq, size, lh);
        for (int k = 0; k < size; ++k)
          out.pathHit[k] = lh[k].hit_orig_index;
      }
    }
  };

  int checkPq(Queue& queue, const std::vector<PqRec>& recs) {
    const int n = recs.size();
    auto d_in = toDevice(queue, recs);
    auto d_out = cms::alpakatools::make_device_buffer<PqOut[]>(queue, std::max(n, 1));
    launch(queue, n, KernelPq{}, d_in.data(), d_out.data(), n);
    const auto out = toHost<PqOut>(queue, d_out, n);
    int badHeap = 0, badPath = 0;
    for (int r = 0; r < n; ++r) {
      const PqRec& ref = recs[r];
      const PqOut& got = out[r];
      bool heapOk = got.nKept == ref.nKept, pathOk = heapOk;
      for (int k = 0; heapOk && k < ref.nKept; ++k)
        heapOk = got.heapHit[k] == ref.heapHit[k];
      for (int k = 0; pathOk && k < ref.nKept; ++k)
        pathOk = got.pathHit[k] == ref.pathHit[k];
      badHeap += !heapOk;
      badPath += !pathOk;
    }
    std::printf("%-44s n=%6d heap differs %d, path order differs %d%s\n",
                "pqueue (pq_offer) + order_hits_along_path",
                n,
                badHeap,
                badPath,
                badHeap + badPath ? "  OUTSIDE THE GATE (must be identical)" : "");
    return badHeap + badPath ? 1 : 0;
  }

  //--------------------------------------------------------------------------
  // scores

  struct ScoreOut {
    int group;
    float loglh;
    float step_score;
    int hole_kind;
    float llh;
    float keep;
  };

  struct KernelScore {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, const ScoreRec* recs, ScoreOut* outs, int n) const {
      for (int32_t r : cms::alpakatools::uniform_elements(acc, n)) {
        const ScoreRec& in = recs[r];
        ::mkfitdev::LayerStepFeatures f{};
        f.chi2_sum = in.chi2_sum;
        f.log_det_v_sum = in.log_det_v_sum;
        f.log_rho = in.log_rho;
        f.n_hits = in.n_hits;
        f.n_overlap = in.n_overlap;
        f.hole_kind = in.hole_kind;
        f.layer_to = in.layer_to;
        f.is_outward = in.is_outward;
        f.is_pixel = in.is_pixel;
        f.is_barrel = in.is_barrel;
        ::mkfitdev::V2p2Params vp{};
        vp.score_outward = toDev(in.outward);
        vp.score_inward = toDev(in.inward);
        ScoreOut& out = outs[r];
        out.group = ::mkfitdev::v2p2_hit_eff_group(f);
        out.loglh = ::mkfitdev::v2p2_layer_step_loglh(f, vp.score(f.is_outward));
        out.step_score = ::mkfitdev::v2p2_layer_step_score(f, vp);
        out.hole_kind = ::mkfitdev::v2p2_hole_kind(in.fake_hit);
        ::mkfitdev::TrackScoreInput ti;
        ti.n_found_hits = in.n_found;
        ti.n_tail_holes = in.n_tail;
        ti.n_overlap_hits = in.n_overlap_hits;
        ti.n_inside_holes = in.n_inside;
        ti.n_seed_hits = in.n_seed_hits;
        ti.chi2 = in.chi2;
        ti.pt = in.pt;
        ti.score_in = in.score_in;
        ti.penalize_tail_holes = true;
        out.llh = ::mkfitdev::trackScore(::mkfitdev::TrackScorer::V2p2Llh, ti);
        out.keep = ::mkfitdev::trackScore(::mkfitdev::TrackScorer::Keep, ti);
      }
    }
  };

  // scores: the same operations in the same order; a device log may differ in the last bit
  constexpr float kScoreTol = 1e-5f;

  bool closeScore(float got, float ref) { return std::abs(got - ref) <= kScoreTol * std::max(1.f, std::abs(ref)); }

  int checkScore(Queue& queue, const std::vector<ScoreRec>& recs) {
    const int n = recs.size();
    auto d_in = toDevice(queue, recs);
    auto d_out = cms::alpakatools::make_device_buffer<ScoreOut[]>(queue, std::max(n, 1));
    launch(queue, n, KernelScore{}, d_in.data(), d_out.data(), n);
    const auto out = toHost<ScoreOut>(queue, d_out, n);
    int badDiscrete = 0, badFloat = 0, bitwise = 0;
    for (int r = 0; r < n; ++r) {
      const ScoreRec& ref = recs[r];
      const ScoreOut& got = out[r];
      badDiscrete += got.group != ref.group || got.hole_kind != ref.hole_kind_out;
      badFloat += !closeScore(got.loglh, ref.loglh) || !closeScore(got.step_score, ref.step_score) ||
                  !closeScore(got.llh, ref.llh) || got.keep != ref.score_in;
      bitwise += got.loglh == ref.loglh && got.step_score == ref.step_score && got.llh == ref.llh;
    }
    std::printf("%-44s n=%6d bitwise %.5f, group / hole kind differ %d, scores outside %.0e %d%s\n",
                "V2p2Score + trackScore(V2p2Llh, Keep)",
                n,
                double(bitwise) / std::max(n, 1),
                badDiscrete,
                double(kScoreTol),
                badFloat,
                badDiscrete + badFloat ? "  OUTSIDE THE GATE" : "");
    return badDiscrete + badFloat ? 1 : 0;
  }

  //--------------------------------------------------------------------------
  // Kalman

  struct KalOut {
    float outPar[6];
    float outErr[21];
    int chgOut;
    float chi2, detV, chi2Acc;
  };

  // a batch: consecutive items of one depth kind and one chi2_trk_fac
  struct KalGroup {
    int begin, n;
  };

  struct KernelKal {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const KalRec* recs,
                                  const KalGroup* groups,
                                  KalOut* outs,
                                  int nGroups,
                                  PropagationFlags inter,
                                  PropagationFlags intra) const {
      for (int32_t g : cms::alpakatools::uniform_elements(acc, nGroups)) {
        const KalGroup grp = groups[g];
        const bool solve = recs[grp.begin].solvePlane;
        mkd::KalmanOpArgs<kN> koa(solve, recs[grp.begin].chi2TrkFac);
        koa.reset();
        for (int k = 0; k < grp.n; ++k) {
          const KalRec& r = recs[grp.begin + k];
          koa.item_begin(0, 0, 0, 0, solve ? 0 : -1, 0);
          if (solve) {
            ::mkfitdev::V2p2TrackState st{};
            for (int i = 0; i < 6; ++i)
              st.par[i] = r.inPar[i];
            for (int i = 0; i < 21; ++i)
              st.err[i] = r.inErr[i];
            st.charge = r.chg;
            koa.load_state_err_chg(st);
          } else {
            ::mkfitdev::V2p2InitialState is{};
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
            ::mkfitdev::CandState prev{};
            for (int i = 0; i < 6; ++i)
              prev.par[i] = r.inPar[i];
            for (int i = 0; i < 21; ++i)
              prev.err[i] = r.inErr[i];
            prev.charge = r.chg;
            koa.load_state_err_chg(is, prev);
          }
          // load_hit_module from the record's hit and plane
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
        koa.do_kalman_stuff(inter, intra);
        for (int k = 0; k < grp.n; ++k) {
          KalOut& o = outs[grp.begin + k];
          ::mkfitdev::V2p2TrackState st;
          koa.item_state(k, st);
          for (int i = 0; i < 6; ++i)
            o.outPar[i] = st.par[i];
          for (int i = 0; i < 21; ++i)
            o.outErr[i] = st.err[i];
          o.chgOut = st.charge;
          o.chi2 = koa.tsChi2.constAt(k, 0, 0);
          o.detV = koa.tsDetV.constAt(k, 0, 0);
          o.chi2Acc = koa.tsChi2Acc.constAt(k, 0, 0);
        }
      }
    }
  };

  // the agreement gate of the propagation test (alpaka/testProp.dev.cc), per metric median and 99 % quantile
  constexpr double kMinAgreement = 0.999;
  constexpr double kMaxMedianDiff = 1e-3;
  constexpr double kMaxQ99Diff = 1e-2;

  // per-item differences of one metric; the quantiles sort a copy
  struct Distribution {
    std::vector<double> values;
    double quantile(double fraction) const {
      if (values.empty())
        return 0.;
      std::vector<double> sorted(values);
      std::sort(sorted.begin(), sorted.end());
      return sorted[std::min<std::size_t>(sorted.size() - 1, std::size_t(fraction * sorted.size()))];
    }
    bool inGate() const { return quantile(0.5) <= kMaxMedianDiff && quantile(0.99) <= kMaxQ99Diff; }
    void print(const char* name) const {
      std::printf("    %-9s q50 %.2e q90 %.2e q99 %.2e q99.9 %.2e max %.2e\n",
                  name,
                  quantile(0.5),
                  quantile(0.9),
                  quantile(0.99),
                  quantile(0.999),
                  quantile(1.));
    }
  };

  double relDiff(double val, double ref, double minScale) {
    return std::abs(val - ref) / std::max(std::abs(ref), minScale);
  }

  double absDeltaPhi(double phi1, double phi2) { return std::abs(std::remainder(phi1 - phi2, 2 * M_PI)); }

  // the track-parameter metric of the propagation test: x, y, z relative to max(|ref|, 1 cm), 1/pT relative, phi and
  // theta absolute
  double parDiff(const float* got, const float* ref) {
    double d = 0.;
    for (int k = 0; k < 3; ++k)
      d = std::max(d, relDiff(got[k], ref[k], 1.0));
    d = std::max(d, relDiff(got[3], ref[3], 1e-6));
    d = std::max(d, absDeltaPhi(got[4], ref[4]));
    return std::max(d, double(std::abs(got[5] - ref[5])));
  }

  // relative Frobenius norm of a difference of packed matrices
  double errDiff(const float* got, const float* ref, int n) {
    double diff2 = 0, ref2 = 0;
    for (int k = 0; k < n; ++k) {
      diff2 += (double(got[k]) - ref[k]) * (double(got[k]) - ref[k]);
      ref2 += double(ref[k]) * ref[k];
    }
    return ref2 > 0 ? std::sqrt(diff2 / ref2) : std::sqrt(diff2);
  }

  // Backward fit (bkFitFitTracksV2p2) of one Kalman record per candidate: its state, one node with its hit and plane
  struct BkOut {
    float par[6];
    float err[21];
    int chg;
    float chi2;
  };
  struct BkFitRecIO {
    const KalRec* recs;
    BkOut* outs;
    const ::mkfitdev::EngineModule* mods;
    template <::mkfitdev::Matriplex::idx_t N>
    ALPAKA_FN_HOST_ACC void loadCand(
        int c, ::mkfitdev::MPlexLS<N>& err, ::mkfitdev::MPlexLV<N>& par, ::mkfitdev::MPlexQI<N>& chg, int n) const {
      par.copyIn(n, recs[c].inPar);
      err.copyIn(n, recs[c].inErr);
      chg.At(n, 0, 0) = recs[c].chg;
    }
    ALPAKA_FN_HOST_ACC int lastNode(int) const { return 0; }
    ALPAKA_FN_HOST_ACC int nodeIndex(int, int) const { return 0; }
    ALPAKA_FN_HOST_ACC int nodeLayer(int, int) const { return 0; }
    ALPAKA_FN_HOST_ACC int nodePrev(int, int) const { return -1; }
    ALPAKA_FN_HOST_ACC int doubleLayerPartner(int) const { return -1; }
    ALPAKA_FN_HOST_ACC const ::mkfitdev::EngineModule& nodeModule(int c, int) const { return mods[c]; }
    template <::mkfitdev::Matriplex::idx_t N>
    ALPAKA_FN_HOST_ACC void loadHit(int c,
                                    int,
                                    ::mkfitdev::MPlexHS<N>& msErr,
                                    ::mkfitdev::MPlexHV<N>& msPar,
                                    ::mkfitdev::MPlexHV<N>& plNrm,
                                    ::mkfitdev::MPlexHV<N>& plDir,
                                    ::mkfitdev::MPlexHV<N>& plPnt,
                                    int n) const {
      const KalRec& r = recs[c];
      for (int i = 0; i < 3; ++i) {
        msPar(n, i, 0) = r.msPar[i];
        plNrm(n, i, 0) = r.plNrm[i];
        plDir(n, i, 0) = r.plDir[i];
        plPnt(n, i, 0) = r.plPnt[i];
        for (int j = 0; j <= i; ++j)
          msErr(n, i, j) = r.msErr[symIdx(i, j)];
      }
    }
    template <::mkfitdev::Matriplex::idx_t N>
    ALPAKA_FN_HOST_ACC void storeCand(int c,
                                      const ::mkfitdev::MPlexLS<N>& err,
                                      const ::mkfitdev::MPlexLV<N>& par,
                                      const ::mkfitdev::MPlexQI<N>& chg,
                                      float chi2,
                                      int n) const {
      par.copyOut(n, outs[c].par);
      err.copyOut(n, outs[c].err);
      outs[c].chg = chg.constAt(n, 0, 0);
      outs[c].chi2 = chi2;
    }
    ALPAKA_FN_HOST_ACC float candPt(int) const { return 1.f; }
    ALPAKA_FN_HOST_ACC void markOutlier(int, int) const {}
    ALPAKA_FN_HOST_ACC void recountHolesAndOverlaps(int) const {}
  };

  struct KernelBkFitRec {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, BkFitRecIO io, const KalGroup* groups, int nGroups, PropagationFlags pf) const {
      for (int32_t g : cms::alpakatools::uniform_elements(acc, nGroups)) {
        int cand[kN];
        for (int k = 0; k < kN; ++k)
          cand[k] = groups[g].begin + (k < groups[g].n ? k : 0);
        mkd::bkFitFitTracksV2p2<kN>(io, cand, groups[g].n, pf, ::mkfitdev::bkfit::OutlierParams{});
      }
    }
  };

  // Each record alone (a partial batch on CPU backends) and in its batch: bitwise equal, for do_kalman_stuff and the
  // backward fit (computed at full width, as MkFitCore)
  int checkFullWidth(Queue& queue,
                     const std::vector<KalRec>& recs,
                     const std::vector<KalGroup>& groups,
                     const PropagationFlags& pf) {
    const int n = recs.size();
    std::vector<KalGroup> alone(n);
    std::vector<::mkfitdev::EngineModule> mods(std::max(n, 1));
    for (int r = 0; r < n; ++r)
      alone[r] = {r, 1};
    auto d_in = toDevice(queue, recs);
    auto d_mods = toDevice(queue, mods);
    int differ = 0;
    for (int fit = 0; fit < 2; ++fit) {
      std::vector<std::vector<char>> res;
      for (const std::vector<KalGroup>* gs : {&groups, static_cast<const std::vector<KalGroup>*>(&alone)}) {
        auto d_groups = toDevice(queue, *gs);
        const int ng = gs->size();
        if (fit == 0) {
          auto d_out = cms::alpakatools::make_device_buffer<KalOut[]>(queue, std::max(n, 1));
          alpaka::memset(queue, d_out, 0);
          launch(queue, ng, KernelKal{}, d_in.data(), d_groups.data(), d_out.data(), ng, pf, pf);
          const auto o = toHost<KalOut>(queue, d_out, n);
          res.emplace_back(reinterpret_cast<const char*>(o.data()), reinterpret_cast<const char*>(o.data() + n));
        } else {
          auto d_out = cms::alpakatools::make_device_buffer<BkOut[]>(queue, std::max(n, 1));
          alpaka::memset(queue, d_out, 0);
          launch(queue,
                 ng,
                 KernelBkFitRec{},
                 BkFitRecIO{d_in.data(), d_out.data(), d_mods.data()},
                 d_groups.data(),
                 ng,
                 pf);
          const auto o = toHost<BkOut>(queue, d_out, n);
          res.emplace_back(reinterpret_cast<const char*>(o.data()), reinterpret_cast<const char*>(o.data() + n));
        }
      }
      const size_t sz = fit == 0 ? sizeof(KalOut) : sizeof(BkOut);
      for (int r = 0; r < n; ++r)
        differ += std::memcmp(res[0].data() + r * sz, res[1].data() + r * sz, sz) != 0;
    }
    std::printf("%-44s n=%6d x 2: items that differ alone vs in a batch %d%s\n",
                "do_kalman_stuff + bkFitFitTracksV2p2 width",
                n,
                differ,
                differ ? "  OUTSIDE THE GATE" : "");
    return differ ? 1 : 0;
  }

  struct KernelAtan2Poly {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, const float* y, const float* x, float* out, int n) const {
      for (int32_t i : cms::alpakatools::uniform_elements(acc, n))
        out[i] = ::mkfitdev::Matriplex::internal::atan2_poly(y[i], x[i]);
    }
  };

  // Matriplex::internal::atan2_poly against std::atan2, on the host and in a kernel of the backend: within kMaxUlp ulp
  // over a full turn at several radii, bitwise at the axes and at the zeros of both signs
  int checkAtan2Poly(Queue& queue) {
    constexpr int kSteps = 100000;
    // room above the host worst case for a device's multiply-add contraction and approximate division
    constexpr int kMaxUlp = 8;
    auto ulps = [](float a, float b) {
      int32_t ia, ib;
      std::memcpy(&ia, &a, sizeof(a));
      std::memcpy(&ib, &b, sizeof(b));
      if ((ia < 0) != (ib < 0))
        return a == b ? 0 : std::numeric_limits<int>::max();
      return std::abs(ia - ib);
    };
    std::vector<float> ys, xs;
    for (float r : {1e-3f, 1.f, 37.5f, 1e4f}) {
      for (int k = 0; k < kSteps; ++k) {
        const double phi = -M_PI + 2 * M_PI * (k + 0.5) / kSteps;
        ys.push_back(r * std::sin(phi));
        xs.push_back(r * std::cos(phi));
      }
    }
    const int nSweep = ys.size();
    const float edges[][2] = {{0.f, 1.f},
                              {1.f, 0.f},
                              {0.f, -1.f},
                              {-1.f, 0.f},
                              {-0.f, -1.f},
                              {0.f, -0.f},
                              {0.f, 0.f},
                              {-0.f, 1.f},
                              {-0.f, 0.f},
                              {-0.f, -0.f}};
    for (const auto& e : edges) {
      ys.push_back(e[0]);
      xs.push_back(e[1]);
    }
    const int n = ys.size();
    auto d_y = toDevice(queue, ys);
    auto d_x = toDevice(queue, xs);
    auto d_out = cms::alpakatools::make_device_buffer<float[]>(queue, n);
    launch(queue, n, KernelAtan2Poly{}, d_y.data(), d_x.data(), d_out.data(), n);
    const auto dev = toHost<float>(queue, d_out, n);
    int worst = 0, bad = 0, badEdge = 0;
    for (int i = 0; i < n; ++i) {
      const float ref = std::atan2(ys[i], xs[i]);
      for (const float got : {::mkfitdev::Matriplex::internal::atan2_poly(ys[i], xs[i]), dev[i]}) {
        if (i < nSweep) {
          const int u = ulps(got, ref);
          worst = std::max(worst, u);
          bad += u > kMaxUlp;
        } else {
          badEdge += std::memcmp(&got, &ref, sizeof(float)) != 0;
        }
      }
    }
    std::printf("%-44s largest difference %d ulp, %d values beyond %d ulp, %d of %d edge values not bitwise equal%s\n",
                "atan2_poly vs std::atan2 (host and kernel)",
                worst,
                bad,
                kMaxUlp,
                badEdge,
                2 * (n - nSweep),
                bad + badEdge ? "  OUTSIDE THE GATE" : "");
    return bad + badEdge ? 1 : 0;
  }

  // A one-layer EventOfHits on the host for the window code: barrel, the normal phi axis, q in 1 cm bins, hits at the
  // given phi and one q, no hit extents (the window margins are the track terms alone); no bin table
  struct HostLayer {
    PortableHostCollection<::mkfitdev::LayerSoA> layers;
    PortableHostCollection<::mkfitdev::BinnedHitSoA> hits;
    PortableHostCollection<::mkfitdev::BinSoA> bins;
    HostLayer(Queue& queue, const std::vector<float>& phis, float q)
        : layers(queue, 1), hits(queue, int(phis.size())), bins(queue, 1) {
      auto lv = layers.view();
      lv.phiRMin() = -::mkfitdev::Const::PI;
      lv.phiNFac() = ::mkfitdev::kNPhiBins / ::mkfitdev::Const::TwoPI;
      const int nQBins = 32;
      lv[0].qRMin() = 0.f;
      lv[0].qNFac() = 1.f;
      lv[0].qNLbhp() = float(nQBins);
      lv[0].qLastNBin() = nQBins - 1;
      lv[0].nQBins() = nQBins;
      lv[0].isBarrel() = 1;
      lv[0].isPixel() = 0;
      lv[0].hitBegin() = 0;
      lv[0].nHits() = phis.size();
      lv[0].maxQHalfLength() = 0.f;
      lv[0].maxPhiHalfExtent() = 0.f;
      auto hv = hits.view();
      for (int h = 0; h < int(phis.size()); ++h) {
        hv[h].rank() = h;
        hv[h].phi() = phis[h];
        hv[h].q() = q;
        hv[h].qHalfLength() = 0.f;
        hv[h].qbar() = 50.f;
        hv[h].phiHalfExtent() = 0.f;
        hv[h].qbarHalfExtent() = 0.f;
      }
    }
    ::mkfitdev::LayerOfHitsAccess access() const {
      return ::mkfitdev::LayerOfHitsAccess{layers.const_view(), hits.const_view(), bins.const_view(), 0};
    }
  };

  // A one-candidate barrel window at phi = centre on the layer of HostLayer: crossings at r 49 and 51, the given half-width
  // of the crossing span and track phi term; MkBins keeps m_phi_min / m_phi_max swapped across phi = +-pi
  mkd::V2p2CandWindow<1> barrelWindow(float centre, float halfWidth, float dphiTrack, float q) {
    mkd::V2p2CandWindow<1> w{};
    w.N_proc = 1;
    w.B.m_n_proc = 1;
    w.B.m_is_barrel = true;
    const float c = std::cos(centre), s = std::sin(centre);
    w.B.m_sp1.x[0] = 49.f * c;
    w.B.m_sp1.y[0] = 49.f * s;
    w.B.m_sp1.z[0] = q;
    w.B.m_sp2.x[0] = 51.f * c;
    w.B.m_sp2.y[0] = 51.f * s;
    w.B.m_sp2.z[0] = q;
    w.B.m_phi1[0] = w.B.m_phi2[0] = centre;
    w.B.m_phi_center[0] = centre;
    w.B.m_phi_delta[0] = halfWidth;
    w.B.m_phi_min[0] = ::mkfitdev::squashPhiGeneral(centre - halfWidth);
    w.B.m_phi_max[0] = ::mkfitdev::squashPhiGeneral(centre + halfWidth);
    w.B.m_q_min[0] = w.B.m_q_max[0] = q;
    w.B.m_dq_track[0] = 0.1f;
    w.B.m_dphi_track[0] = dphiTrack;
    return w;
  }

  // The density region across phi = +-pi: select_hits_setup + select_hits_scan_hit count the hits within the window's
  // centre +- half-width across the wrap and none outside; select_hits_log_rho gives the window the area of the same window
  // away from +-pi, and the phi extent of the region stops at the full turn
  int checkDensityAcrossPi(Queue& queue) {
    const float pi = ::mkfitdev::Const::PI, halfWidth = 0.05f, q = 11.f;
    int bad = 0;
    {
      // inside: just below +pi and just above -pi; outside: further on either side, and far away
      const std::vector<float> phis{3.10f, -3.10f, 3.0f, -3.0f, 0.f};
      const HostLayer layer(queue, phis, q);
      const auto L = layer.access();
      auto w = barrelWindow(pi, halfWidth, 0.001f, q);
      mkd::select_hits_setup(w, 0, L, ::mkfitdev::V2p2Params{});
      int nScanned = 0;
      for (uint32_t h = 0; h < phis.size(); ++h)
        mkd::select_hits_scan_hit(w, 0, L, h, nScanned);
      bad += nScanned != 2;
    }
    auto logRho = [&](float centre, float rhoDphi) {
      auto w = barrelWindow(centre, halfWidth, 0.f, q);
      w.rho_dphi[0] = rhoDphi;
      w.rho_dq[0] = 0.5f;
      return mkd::select_hits_log_rho(w, 0, 7);
    };
    bad += logRho(pi, 0.01f) != logRho(0.f, 0.01f);
    bad += logRho(0.f, 4.f) != std::log(8.f / (::mkfitdev::Const::TwoPI * 51.f * 1.f));
    std::printf("%-44s %d cases: %d wrong%s\n", "density region across +-pi", 3, bad, bad ? "  OUTSIDE THE GATE" : "");
    return bad ? 1 : 0;
  }

  // The phi-margin cap at its call sites, for a track phi term far above it: the fetch (MkBins::find_bin_ranges), the line
  // pre-cut tolerance and the density region (select_hits_setup) take Window::dphi_max_rad as the margin
  int checkPhiMarginCap(Queue& queue) {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    namespace PreCut = ::mkfitdev::Config::V2p2::PreCut;
    const float cap = Window::dphi_max_rad, halfWidth = 0.05f, q = 11.f, dphiTrack = 1.f;
    const HostLayer layer(queue, {0.f}, q);
    const auto L = layer.access();
    auto w = barrelWindow(1.f, halfWidth, dphiTrack, q);
    int bad = 0;
    mkd::MkBinLimits<1> bl{};
    w.B.find_bin_ranges(L, bl);
    auto fetch = [&](float margin) { return L.phiRangeBins(w.B.m_phi_min[0] - margin, w.B.m_phi_max[0] + margin); };
    const auto capped = fetch(cap), uncapped = fetch(Window::dphi_trk_fac * dphiTrack);
    bad += bl.p1[0] != L.phiMaskApply(capped.begin) || bl.p2[0] != L.phiMaskApply(capped.end);
    bad += capped.begin == uncapped.begin && capped.end == uncapped.end;  // the case must tell the two apart
    mkd::select_hits_setup(w, 0, L, ::mkfitdev::V2p2Params{});
    bad += !w.pc_on[0] || w.pc_tol_phi[0] != PreCut::dphi_slack * cap;
    bad += w.rho_dphi[0] != cap || w.rho_phi_half[0] != halfWidth + cap;
    std::printf(
        "%-44s %d cases: %d wrong%s\n", "phi-margin cap at its call sites", 4, bad, bad ? "  OUTSIDE THE GATE" : "");
    return bad ? 1 : 0;
  }

  // The window factors of V2p2Params (windowPhiTrackFactor, windowQTrackFactor) at their call sites: the fetch
  // (MkBins::find_bin_ranges), the line pre-cut tolerance and the density region (select_hits_setup); <= 0 takes the
  // built-in Window factors
  int checkWindowFactors(Queue& queue) {
    namespace Window = ::mkfitdev::Config::V2p2::Window;
    namespace PreCut = ::mkfitdev::Config::V2p2::PreCut;
    const float halfWidth = 0.05f, q = 11.f, dphiTrack = 0.01f;
    const HostLayer layer(queue, {0.f}, q);
    const auto L = layer.access();
    int bad = 0, n = 0;
    // factors in the PSet (an offline-like non-default value), zero and negative (the built-in ones)
    for (const float fac : {3.5f, 0.f, -1.f}) {
      ::mkfitdev::V2p2Params vp{};
      vp.window_dphi_trk_fac = fac;
      vp.window_dq_trk_fac = 0.5f * fac;
      const float fphi = fac > 0.f ? fac : Window::dphi_trk_fac, fq = fac > 0.f ? 0.5f * fac : Window::dq_trk_fac;
      bad += mkd::win_dphi_trk_fac(vp) != fphi || mkd::win_dq_trk_fac(vp) != fq;
      auto w = barrelWindow(1.f, halfWidth, dphiTrack, q);
      mkd::MkBinLimits<1> bl{}, ref{};
      w.B.find_bin_ranges(L, bl, mkd::win_dphi_trk_fac(vp), mkd::win_dq_trk_fac(vp));
      const float cut_dphi = Window::capped_dphi(
          fphi * dphiTrack +
          (Window::phi_per_hit ? Window::dphi_hit_fac * L.max_hit_phi_half_extent() : Window::dphi_flat_rad));
      const auto pr = L.phiRangeBins(w.B.m_phi_min[0] - cut_dphi, w.B.m_phi_max[0] + cut_dphi);
      bad += bl.p1[0] != L.phiMaskApply(pr.begin - Window::phi_extra_bins) ||
             bl.p2[0] != L.phiMaskApply(pr.end + Window::phi_extra_bins);
      w.B.find_bin_ranges(L, ref, fac, 0.5f * fac);  // the same factors through find_bin_ranges' own fallback
      bad += bl.p1[0] != ref.p1[0] || bl.p2[0] != ref.p2[0] || bl.q1[0] != ref.q1[0] || bl.q2[0] != ref.q2[0];
      mkd::select_hits_setup(w, 0, L, vp);
      bad += w.pc_tol_q[0] != PreCut::dq_slack * fq * w.B.m_dq_track[0] * (1.0f + w.pc_gq[0] * w.pc_gq[0]);
      bad += w.pc_tol_phi[0] != PreCut::dphi_slack * Window::capped_dphi(fphi * dphiTrack);
      bad += w.rho_dq[0] != fq * w.B.m_dq_track[0] + Window::dq_hit_fac * L.max_hit_q_half_length();
      n += 6;
    }
    std::printf("%-44s %d cases: %d wrong%s\n",
                "v2p2 window factors at their call sites",
                n,
                bad,
                bad ? "  OUTSIDE THE GATE" : "");
    return bad ? 1 : 0;
  }

  // outside_as_hole: WSR_Outside is a hole for a seed with at most outside_hole_max_seed_hits hits until the candidate
  // has a hit beyond its seed; off at 0 (the default) and below
  int checkOutsideHole() {
    struct Case {
      int maxSeed, nSeed, nFound;
      bool hole;
    };
    const Case cases[] = {{0, 3, 3, false},
                          {-1, 3, 3, false},
                          {3, 3, 3, true},
                          {3, 2, 2, true},
                          {3, 3, 4, false},
                          {3, 4, 4, false},
                          {3, 1, 0, true}};
    int bad = 0;
    for (const Case& c : cases) {
      ::mkfitdev::V2p2Params vp{};
      vp.outside_hole_max_seed_hits = c.maxSeed;
      ::mkfitdev::CandBook book{};
      book.nFound = c.nFound;
      ::mkfitdev::CandState state{};
      state.nSeedHits = c.nSeed;
      bad += mkd::outside_as_hole(mkd::TrackCandView{&book, &state, nullptr}, vp) != c.hole;
    }
    std::printf("%-44s %d cases: %d wrong%s\n",
                "v2p2 outside hole of a short seed",
                int(std::size(cases)),
                bad,
                bad ? "  OUTSIDE THE GATE" : "");
    return bad ? 1 : 0;
  }

  // harvest_accept on the cases of the negative-chi2 and det V <= 0 rejections, NaN, and which of the two chi2 values the
  // cut takes
  int checkHarvestAccept() {
    ::mkfitdev::V2p2Params vp{};
    vp.hit_chi2_cut = 10.f;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    struct Case {
      float chi2, chi2Acc, detV;
      bool accept;
    };
    const Case cases[] = {{1.f, 1.f, 1.f, true},
                          {-1.f, 1.f, 1.f, false},
                          {1.f, -1.f, 1.f, false},
                          {nan, 1.f, 1.f, false},
                          {1.f, nan, 1.f, false},
                          {1.f, 11.f, 1.f, false},
                          {11.f, 1.f, 1.f, true},
                          {0.f, 0.f, 1.f, true},
                          {1.f, 1.f, 0.f, false},
                          {1.f, 1.f, -1e-6f, false},
                          {1.f, 1.f, nan, false},
                          {1.f, 1.f, 1e-30f, true}};
    int bad = 0;
    for (const Case& c : cases)
      bad += mkd::harvest_accept(c.chi2, c.chi2Acc, c.detV, vp) != c.accept;
    std::printf(
        "%-44s %d cases: %d wrong%s\n", "harvest_accept", int(std::size(cases)), bad, bad ? "  OUTSIDE THE GATE" : "");
    return bad ? 1 : 0;
  }

  int checkKal(Queue& queue, std::vector<KalRec> recs, const PropagationFlags& pf, float cut) {
    // consecutive items of one kind and one chi2_trk_fac (a batch has one); every item's result is its own
    std::stable_sort(recs.begin(), recs.end(), [](const KalRec& a, const KalRec& b) {
      return a.solvePlane != b.solvePlane ? a.solvePlane < b.solvePlane : a.chi2TrkFac < b.chi2TrkFac;
    });
    std::vector<KalGroup> groups;
    for (int i = 0; i < int(recs.size());) {
      int j = i;
      while (j < int(recs.size()) && j - i < kN && recs[j].solvePlane == recs[i].solvePlane &&
             recs[j].chi2TrkFac == recs[i].chi2TrkFac)
        ++j;
      groups.push_back({i, j - i});
      i = j;
    }
    const int n = recs.size();
    auto d_in = toDevice(queue, recs);
    auto d_groups = toDevice(queue, groups);
    auto d_out = cms::alpakatools::make_device_buffer<KalOut[]>(queue, std::max(n, 1));
    launch(
        queue, int(groups.size()), KernelKal{}, d_in.data(), d_groups.data(), d_out.data(), int(groups.size()), pf, pf);
    const auto out = toHost<KalOut>(queue, d_out, n);
    ::mkfitdev::V2p2Params vp{};
    vp.hit_chi2_cut = cut;
    int nBad = checkFullWidth(queue, recs, groups, pf);
    for (int solve = 0; solve < 2; ++solve) {
      Distribution par, err, chi2, detV;
      int nItems = 0, chgAgree = 0, accAgree = 0, nonFinite = 0;
      for (int r = 0; r < n; ++r) {
        if (recs[r].solvePlane != solve)
          continue;
        const KalRec& ref = recs[r];
        const KalOut& got = out[r];
        ++nItems;
        chgAgree += got.chgOut == ref.chgOut;
        accAgree += mkd::harvest_accept(got.chi2, got.chi2Acc, got.detV, vp) ==
                    mkd::harvest_accept(ref.chi2, ref.chi2Acc, ref.detV, vp);
        bool finite = std::isfinite(got.chi2) == std::isfinite(ref.chi2);
        for (int k = 0; k < 6; ++k)
          finite = finite && std::isfinite(got.outPar[k]) == std::isfinite(ref.outPar[k]);
        if (!finite) {
          ++nonFinite;
          continue;
        }
        par.values.push_back(parDiff(got.outPar, ref.outPar));
        err.values.push_back(errDiff(got.outErr, ref.outErr, 21));
        chi2.values.push_back(std::max(relDiff(got.chi2, ref.chi2, 1e-3), relDiff(got.chi2Acc, ref.chi2Acc, 1e-3)));
        detV.values.push_back(relDiff(got.detV, ref.detV, 1e-30));
      }
      std::string outside;
      if (nItems && double(chgAgree) / nItems < kMinAgreement)
        outside += " charge";
      if (nItems && double(accAgree) / nItems < kMinAgreement)
        outside += " acceptance";
      if (nonFinite)
        outside += " non-finite";
      if (!par.inGate())
        outside += " params";
      if (!err.inGate())
        outside += " errors";
      if (!chi2.inGate())
        outside += " chi2";
      if (!detV.inGate())
        outside += " detV";
      std::printf("%-44s n=%6d charge-agree %.5f accept-agree %.5f non-finite %d%s%s\n",
                  solve ? "KalmanOpArgs::do_kalman_stuff (solve plane)" : "KalmanOpArgs::do_kalman_stuff (depth 0)",
                  nItems,
                  double(chgAgree) / std::max(nItems, 1),
                  double(accAgree) / std::max(nItems, 1),
                  nonFinite,
                  outside.empty() ? "" : "  OUTSIDE THE GATE:",
                  outside.c_str());
      par.print("params");
      err.print("errors");
      chi2.print("chi2");
      detV.print("detV");
      nBad += !outside.empty();
    }
    return nBad;
  }

  //--------------------------------------------------------------------------
  // window

  struct WinOut {
    int quadrantOk;
    float sp1[7], sp2[7];
    int fail1, fail2;
    float cov[6];
    float phiMin, phiMax, phiCenter, phiDelta, qMin, qMax, dphiTrack, dqTrack;
    int wsr, inGap;
    float hx[4], hy[4], hz[4], hderfac;
  };

  struct KernelWin {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const WinRec* recs, ::mkfitdev::LayerInfoSoA::ConstView geo, WinOut* outs, int n) const {
      for (int32_t r : cms::alpakatools::uniform_elements(acc, n)) {
        const WinRec& in = recs[r];
        WinOut& out = outs[r];
        // the layer of record r is row r of the layer table
        mkd::V2p2LayerCtx ctx{};
        ctx.is_outward = in.outward;
        ctx.param_b = true;
        ctx.bfield = ::mkfitdev::Config::BFieldParams{};
        mkd::begin_layer(ctx, geo, r);
        ::mkfitdev::CandState st{};
        for (int i = 0; i < 6; ++i)
          st.par[i] = in.par[i];
        for (int i = 0; i < 21; ++i)
          st.err[i] = in.err[i];
        st.charge = in.chg;
        const mkd::TrackCandView tc{nullptr, &st, nullptr};
        out.quadrantOk = ctx.rz.rz_quadrant_check(tc.z(), tc.pz());
        mkd::V2p2CandWindow<1> w;
        mkd::prop_to_layer_edges(w, &tc, 1, ctx);
        mkd::MPlexLV<1> par0;
        mkd::MPlexLS<1> err0;
        for (int i = 0; i < 6; ++i) {
          par0(0, i, 0) = in.par[i];
          for (int j = 0; j <= i; ++j)
            err0(0, i, j) = in.err[symIdx(i, j)];
        }
        w.B.transport_position_cov(par0, err0, w.TCE);
        w.B.determine_bin_windows(w.TCE);
        mkd::determine_wsr(w, ctx);
        w.H.calculate_coeffs(w.B.m_sp1, w.B.m_sp2, w.B.m_isp.inv_k);
        const mkd::mini_propagators::StatePlex<1>* sps[2] = {&w.B.m_sp1, &w.B.m_sp2};
        float* dst[2] = {out.sp1, out.sp2};
        for (int k = 0; k < 2; ++k) {
          const auto& sp = *sps[k];
          const float v[7] = {sp.x[0], sp.y[0], sp.z[0], sp.px[0], sp.py[0], sp.pz[0], sp.dalpha[0]};
          for (int i = 0; i < 7; ++i)
            dst[k][i] = v[i];
        }
        out.fail1 = w.B.m_sp1.fail_flag[0];
        out.fail2 = w.B.m_sp2.fail_flag[0];
        const float cov[6] = {w.TCE.m_cov_0_0[0],
                              w.TCE.m_cov_0_1[0],
                              w.TCE.m_cov_1_1[0],
                              w.TCE.m_cov_2_2[0],
                              w.TCE.m_cov_0_2[0],
                              w.TCE.m_cov_1_2[0]};
        for (int i = 0; i < 6; ++i)
          out.cov[i] = cov[i];
        out.phiMin = w.B.m_phi_min[0];
        out.phiMax = w.B.m_phi_max[0];
        out.phiCenter = w.B.m_phi_center[0];
        out.phiDelta = w.B.m_phi_delta[0];
        out.qMin = w.B.m_q_min[0];
        out.qMax = w.B.m_q_max[0];
        out.dphiTrack = w.B.m_dphi_track[0];
        out.dqTrack = w.B.m_dq_track[0];
        out.wsr = w.wsr[0].wsr;
        out.inGap = w.wsr[0].in_gap;
        for (int c = 0; c < 4; ++c) {
          out.hx[c] = w.H.m_Hx[c][0];
          out.hy[c] = w.H.m_Hy[c][0];
          out.hz[c] = w.H.m_Hz[c][0];
        }
        out.hderfac = w.H.m_Hderfac[0];
      }
    }
  };

  int checkWin(Queue& queue, const std::vector<WinRec>& recs) {
    const int n = recs.size();
    PortableHostCollection<::mkfitdev::LayerInfoSoA> hGeo(queue, std::max(n, 1));
    for (int r = 0; r < n; ++r) {
      auto row = hGeo.view()[r];
      const WinRec& w = recs[r];
      row.layer_type() = w.layerType;
      row.rin() = w.rin;
      row.rout() = w.rout;
      row.zmin() = w.zmin;
      row.zmax() = w.zmax;
      row.propagate_to() = 0.f;
      row.q_bin() = 0.f;
      row.hole_r_min() = w.hasHole ? w.holeRMin : 0.f;
      row.hole_r_max() = w.hasHole ? w.holeRMax : 0.f;
      row.has_r_range_hole() = w.hasHole;
      row.is_pixel() = w.isPixel;
      row.has_charge() = false;
      row.module_begin() = 0;
      row.q_min() = 0.f;
      row.q_max() = 0.f;
      row.n_q() = 0;
    }
    ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<::mkfitdev::LayerInfoSoA> dGeo(queue, std::max(n, 1));
    alpaka::memcpy(queue, dGeo.buffer(), hGeo.buffer());
    auto d_in = toDevice(queue, recs);
    auto d_out = cms::alpakatools::make_device_buffer<WinOut[]>(queue, std::max(n, 1));
    launch(queue, n, KernelWin{}, d_in.data(), dGeo.const_view(), d_out.data(), n);
    const auto out = toHost<WinOut>(queue, d_out, n);
    int quadAgree = 0, failAgree = 0, wsrAgree = 0, hermNonFinite = 0;
    Distribution pos, mom, cov, win, herm;
    for (int r = 0; r < n; ++r) {
      const WinRec& ref = recs[r];
      const WinOut& got = out[r];
      quadAgree += got.quadrantOk == ref.quadrantOk;
      failAgree += (got.fail1 != 0) == (ref.fail1 != 0) && (got.fail2 != 0) == (ref.fail2 != 0);
      wsrAgree += got.wsr == ref.wsr && got.inGap == ref.inGap;
      if (got.fail1 != ref.fail1 || got.fail2 != ref.fail2)
        continue;
      double dp = 0., dm = 0.;
      for (int k = 0; k < 2; ++k) {
        const float* g = k ? got.sp2 : got.sp1;
        const float* e = k ? ref.sp2 : ref.sp1;
        const double pRef = std::sqrt(double(e[3]) * e[3] + double(e[4]) * e[4] + double(e[5]) * e[5]);
        for (int i = 0; i < 3; ++i)
          dp = std::max(dp, relDiff(g[i], e[i], 1.0));
        for (int i = 3; i < 6; ++i)
          dm = std::max(dm, std::abs(double(g[i]) - e[i]) / pRef);
        dm = std::max(dm, double(std::abs(g[6] - e[6])));
      }
      pos.values.push_back(dp);
      mom.values.push_back(dm);
      cov.values.push_back(errDiff(got.cov, ref.cov, 6));
      double dw = 0.;
      dw = std::max(dw, absDeltaPhi(got.phiMin, ref.phiMin));
      dw = std::max(dw, absDeltaPhi(got.phiMax, ref.phiMax));
      dw = std::max(dw, absDeltaPhi(got.phiCenter, ref.phiCenter));
      dw = std::max(dw, double(std::abs(got.phiDelta - ref.phiDelta)));
      dw = std::max(dw, relDiff(got.qMin, ref.qMin, 1.0));
      dw = std::max(dw, relDiff(got.qMax, ref.qMax, 1.0));
      dw = std::max(dw, relDiff(got.dphiTrack, ref.dphiTrack, 1e-6));
      dw = std::max(dw, relDiff(got.dqTrack, ref.dqTrack, 1e-6));
      win.values.push_back(dw);
      // the cubic is used only for a candidate that is not outside the layer; one through two unreachable crossings
      // has no finite derivative factor
      if (ref.wsr == ::mkfitdev::WSR_Outside)
        continue;
      if (!std::isfinite(ref.hderfac) || !std::isfinite(got.hderfac)) {
        hermNonFinite += std::isfinite(ref.hderfac) != std::isfinite(got.hderfac);
        continue;
      }
      double dh = relDiff(got.hderfac, ref.hderfac, 1e-6);
      for (int c = 0; c < 4; ++c)
        dh = std::max({dh,
                       relDiff(got.hx[c], ref.hx[c], 1.0),
                       relDiff(got.hy[c], ref.hy[c], 1.0),
                       relDiff(got.hz[c], ref.hz[c], 1.0)});
      herm.values.push_back(dh);
    }
    std::string outside;
    if (double(quadAgree) / std::max(n, 1) < kMinAgreement)
      outside += " quadrant";
    if (double(failAgree) / std::max(n, 1) < kMinAgreement)
      outside += " fail-agreement";
    if (double(wsrAgree) / std::max(n, 1) < kMinAgreement)
      outside += " wsr";
    if (hermNonFinite)
      outside += " hermite-non-finite";
    const std::pair<const Distribution*, const char*> metrics[] = {
        {&pos, " positions"}, {&mom, " momenta"}, {&cov, " covariance"}, {&win, " windows"}, {&herm, " hermite"}};
    for (auto const& [dist, name] : metrics)
      if (!dist->inGate())
        outside += name;
    std::printf("%-44s n=%6d fail-agree %.5f wsr-agree %.5f quadrant-agree %.5f%s%s\n",
                "prop_to_layer_edges + MkBins + determine_wsr",
                n,
                double(failAgree) / std::max(n, 1),
                double(wsrAgree) / std::max(n, 1),
                double(quadAgree) / std::max(n, 1),
                outside.empty() ? "" : "  OUTSIDE THE GATE:",
                outside.c_str());
    pos.print("position");
    mom.print("momentum");
    cov.print("cov");
    win.print("window");
    herm.print("hermite");
    return outside.empty() ? 0 : 1;
  }

  //--------------------------------------------------------------------------
  // end-of-layer selection

  // the seed's HoT pool: the chains of its TrackCands plus what the selection appends
  constexpr int kHotCap = kSelMaxCands * (kSelMaxHits + kSelMaxDepth + 1) + kSelMaxCands * kSelMaxDepth;
  constexpr int kChainCap = kSelMaxHits + kSelMaxDepth;

  struct SelDevOut {
    int n;
    int nan;
    int hitIdx[kSelMaxCands][kChainCap];
    int hitLyr[kSelMaxCands][kChainCap];
    int nHits[kSelMaxCands];
    float score[kSelMaxCands];
    int nFound[kSelMaxCands], nMissing[kSelMaxCands], nOverlap[kSelMaxCands], nInside[kSelMaxCands],
        nTail[kSelMaxCands];
    float chi2[kSelMaxCands];
    float par[kSelMaxCands][6];
  };

  // the seed's candidate books, HoT pool and the lanes' tops, shared by the lanes of a block
  struct SelBlock {
    ::mkfitdev::CandBook books[kSelMaxCands], outBooks[kSelMaxCands], bestShort;
    ::mkfitdev::CandState states[kSelMaxCands], outStates[kSelMaxCands], bestShortState;
    ::mkfitdev::HoTNode hots[kHotCap];
    int32_t nCands, nHots;
    int8_t state, bestShortValid;
    int16_t pickupLayer, lastBefore, insideBefore, tailBefore;
    uint32_t overflow;
    mkd::V2p2SelTop tops[mkd::kV2p2SelectSeedLanes];
  };

  // the seed's HoT pool of the selection test is fixed (kHotCap nodes in shared memory)
  struct NoHotPool {
    template <typename TAcc>
    ALPAKA_FN_ACC void reserve(TAcc const&, int) const {}
    ALPAKA_FN_HOST_ACC void attach(::mkfitdev::SeedCandsRef&) const {}
  };

  // the seeds' HoT pool in the SoAs of the build (use = 0: the fixed pool of SelBlock)
  struct SelPoolArgs {
    ::mkfitdev::SeedCandsSoA::View seeds;
    ::mkfitdev::CandHotsSoA::View hots;
    int use;
  };

  struct KernelSel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  const SelSeed* seeds,
                                  ::mkfitdev::V2p2CandSoA::ConstView cands,
                                  ::mkfitdev::V2p2NodeSoA::ConstView nodes,
                                  const int* heads,
                                  const int32_t* lists,
                                  SelDevOut* outs,
                                  int n,
                                  SelPoolArgs pa) const {
      // the threads of a block are the lanes of one seed (select_and_materialise has block barriers)
      auto& sh = alpaka::declareSharedVar<SelBlock, __COUNTER__>(acc);
      const int lane = int(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const int nLanes = int(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      for (int32_t s : cms::alpakatools::independent_groups(acc, n)) {
        const SelSeed& in = seeds[s];
        ::mkfitdev::SeedCandsRef seed{&sh.state,
                                      &sh.pickupLayer,
                                      sh.books,
                                      sh.states,
                                      &sh.nCands,
                                      &sh.bestShort,
                                      &sh.bestShortState,
                                      &sh.bestShortValid,
                                      sh.hots,
                                      nullptr,
                                      kHotCap,
                                      kHotCap,
                                      &sh.nHots,
                                      &sh.lastBefore,
                                      &sh.insideBefore,
                                      &sh.tailBefore,
                                      &sh.overflow};
        const mkd::HotPool hp = pa.use ? mkd::hotPoolOf(pa.seeds, pa.hots) : mkd::HotPool{};
        if (pa.use) {
          // the TrackCands' chains first: lane 0 reserves for them, then every lane attaches
          if (lane == 0) {
            int need = 0;
            for (int ic = 0; ic < in.nCands; ++ic)
              need += in.cands[ic].nHits;
            mkd::reserveHots(acc, hp, s, need);
          }
          alpaka::syncBlockThreads(acc);
          mkd::attachHots(hp, s, seed);
          seed.nHots = &hp.nHots[s];
          seed.overflowBits = &pa.seeds.overflowBits(s);
        }
        if (lane == 0) {
          // the seed's candidate books: the TrackCands with their HoT chains, as TrackCand::addHitIdx builds them
          sh.nCands = in.nCands;
          sh.nHots = 0;
          sh.state = ::mkfitdev::kFinding;
          sh.bestShortValid = 0;
          sh.bestShort = ::mkfitdev::CandBook{};
          sh.bestShortState = ::mkfitdev::CandState{};
          sh.pickupLayer = sh.lastBefore = sh.insideBefore = sh.tailBefore = 0;
          sh.overflow = 0;
          for (int ic = 0; ic < in.nCands; ++ic) {
            const SelCand& c = in.cands[ic];
            sh.books[ic] = ::mkfitdev::CandBook{};
            sh.books[ic].lastHitIdx = -1;
            sh.books[ic].originIndex = ic;
            sh.books[ic].overlaps.reset();
            for (int h = 0; h < c.nHits; ++h)
              ::mkfitdev::seedAddHitIdx(seed, sh.books[ic], c.hitIdx[h], c.hitLyr[h], c.hitChi2[h]);
            sh.books[ic].score = c.score;
            sh.states[ic] = ::mkfitdev::CandState{};
            for (int i = 0; i < 6; ++i)
              sh.states[ic].par[i] = c.par[i];
            for (int i = 0; i < 21; ++i)
              sh.states[ic].err[i] = c.err[i];
            sh.states[ic].charge = c.charge;
            sh.states[ic].label = ic;
          }
        }
        mkd::V2p2LayerCtx ctx{};
        ctx.is_outward = in.outward;
        ctx.rz.m_layer_id = in.layerId;
        ctx.rz.m_layer_type = in.isBarrel ? int(::mkfitdev::LayerType::Barrel) : int(::mkfitdev::LayerType::EndCapPos);
        ctx.rz.m_layer_is_pixel = in.isPixel;
        ctx.rz.m_is_barrel = in.isBarrel;
        ctx.rz.m_is_outward = in.outward;
        ctx.ip.maxCandsPerSeed = in.maxCands;
        ctx.ip.maxHolesPerCand = in.maxHolesPerCand;
        ctx.ip.maxConsecHoles = in.maxConsecHoles;
        ctx.capacity = ::mkfitdev::kMaxCandsPerSeed;
        ctx.vp.score_outward = toDev(in.outwardP);
        ctx.vp.score_inward = toDev(in.inwardP);
        alpaka::syncBlockThreads(acc);
        bool nan = false;
        // the in-layer paths as a node list (lists != nullptr) or as the seed's chain
        const auto select = [&](auto const& hotPool) {
          return mkd::select_and_materialise(acc,
                                             mkd::V2p2SelGroup{lane, nLanes, sh.tops},
                                             true,
                                             seed,
                                             hotPool,
                                             sh.outBooks,
                                             sh.outStates,
                                             cands,
                                             s,
                                             nodes,
                                             lists != nullptr ? &lists[s * kSelMaxNodes] : nullptr,
                                             lists != nullptr ? in.nNodes : 0,
                                             lists != nullptr ? -1 : heads[s],
                                             in.maxCands,
                                             ctx,
                                             nan);
        };
        const int nOut = pa.use ? select(mkd::SeedHotPool{&hp, s}) : select(NoHotPool{});
        if (lane == 0) {
          SelDevOut& out = outs[s];
          out.n = nOut;
          out.nan = nan;
          for (int k = 0; k < out.n && k < kSelMaxCands; ++k) {
            const ::mkfitdev::CandBook& b = sh.outBooks[k];
            out.score[k] = b.score;
            out.nFound[k] = b.nFound;
            out.nMissing[k] = b.nMissing;
            out.nOverlap[k] = b.nOverlap;
            out.nInside[k] = b.nInsideMinusOne;
            out.nTail[k] = b.nTailMinusOne;
            out.chi2[k] = b.chi2;
            int chain[kChainCap];
            int nc = 0;
            for (int node = b.lastHitIdx; node >= 0 && nc < kChainCap; node = seed.hot(node).prev)
              chain[nc++] = node;
            out.nHits[k] = nc;
            for (int h = 0; h < nc; ++h) {
              out.hitIdx[k][h] = seed.hot(chain[nc - 1 - h]).index;
              out.hitLyr[k][h] = seed.hot(chain[nc - 1 - h]).layer;
            }
            for (int i = 0; i < 6; ++i)
              out.par[k][i] = sh.outStates[k].par[i];
          }
        }
        // the shared books are rewritten for the block's next seed
        alpaka::syncBlockThreads(acc);
      }
    }
  };

  // the state of the real HoT pool after a run with one (runSel poolHome > 0)
  struct SelPoolInfo {
    std::vector<int> ext, nHots;
    std::vector<uint32_t> overflow;
    uint32_t nExt;
  };

  // Initial state of a HoT pool of n seed rows (as KernelSeedScan / KernelSeedInitCands leave it).
  struct KernelPoolInit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::mkfitdev::SeedCandsSoA::View seeds,
                                  ::mkfitdev::HotPoolLayout layout,
                                  int n) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        seeds.hotsPerSeed() = layout.perSeed;
        seeds.hotsHome() = layout.home;
        seeds.hotExtSlots() = layout.extSlots;
        seeds.nHotExt() = 0;
        seeds.nOverflowHots() = 0;
      }
      for (int32_t s : cms::alpakatools::uniform_elements(acc, n)) {
        seeds.nHots(s) = 0;
        seeds.hotExt(s) = -1;
        seeds.overflowBits(s) = 0;
      }
    }
  };

  std::vector<SelDevOut> runSel(Queue& queue,
                                const std::vector<SelSeed>& seeds,
                                bool reversedLists,
                                int lanes = 1,
                                bool asList = false,
                                int poolHome = 0,
                                SelPoolInfo* poolInfo = nullptr) {
    const int n = seeds.size();
    const int nCandRows = std::max(n, 1) * ::mkfitdev::kMaxCandsPerSeed;
    const int nNodeRows = std::max(n, 1) * kSelMaxNodes;
    PortableHostCollection<::mkfitdev::V2p2CandSoA> hCands(queue, nCandRows);
    PortableHostCollection<::mkfitdev::V2p2NodeSoA> hNodes(queue, nNodeRows);
    std::vector<int> heads(std::max(n, 1), -1);
    std::vector<int32_t> lists(nNodeRows, -1);
    for (int s = 0; s < n; ++s) {
      const SelSeed& in = seeds[s];
      for (int ic = 0; ic < ::mkfitdev::kMaxCandsPerSeed; ++ic) {
        auto row = hCands.view()[::mkfitdev::v2p2CandRow(s, ic)];
        const bool exists = ic < in.nCands;
        row.active() = exists && in.cands[ic].active;
        row.wsr() = exists ? in.cands[ic].wsr : 0;
        row.inGap() = exists ? in.cands[ic].inGap : 0;
        row.nLayerHits() = 0;
        row.logRho() = exists ? in.cands[ic].logRho : 0.f;
      }
      std::vector<uint32_t> keys(in.nNodes);
      for (int k = 0; k < in.nNodes; ++k) {
        const SelNode& nd = in.nodes[k];
        const uint32_t parentKey = nd.parent >= 0 ? keys[nd.parent] : 0u;
        keys[k] = ::mkfitdev::v2p2PathKey(parentKey, nd.nHits - 1, nd.hitPos);
        auto row = hNodes.view()[s * kSelMaxNodes + k];
        row.seed() = s;
        row.ic() = nd.ic;
        row.hitPos() = nd.hitPos;
        row.nHits() = nd.nHits;
        row.parentNode() = nd.parent >= 0 ? s * kSelMaxNodes + nd.parent : -1;
        row.pathKey() = keys[k];
        row.hitIndex() = nd.hitIndex;
        row.hitLayer() = nd.hitLayer;
        row.module() = k;
        row.chi2() = nd.chi2;
        row.chi2Sum() = nd.chi2Sum;
        row.logDetVSum() = nd.logDetVSum;
        ::mkfitdev::V2p2TrackState st{};
        for (int i = 0; i < 6; ++i)
          st.par[i] = nd.par[i];
        for (int i = 0; i < 21; ++i)
          st.err[i] = nd.err[i];
        st.charge = nd.charge;
        row.state() = st;
      }
      // the seed's node list, linked by push-front in harvest order or in the reverse order
      for (int j = 0; j < in.nNodes; ++j) {
        const int k = reversedLists ? in.nNodes - 1 - j : j;
        hNodes.view()[s * kSelMaxNodes + k].nextInSeed() = heads[s];
        heads[s] = s * kSelMaxNodes + k;
        lists[s * kSelMaxNodes + j] = s * kSelMaxNodes + k;
      }
    }
    ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<::mkfitdev::V2p2CandSoA> dCands(queue, nCandRows);
    ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<::mkfitdev::V2p2NodeSoA> dNodes(queue, nNodeRows);
    alpaka::memcpy(queue, dCands.buffer(), hCands.buffer());
    alpaka::memcpy(queue, dNodes.buffer(), hNodes.buffer());
    auto d_seeds = toDevice(queue, seeds.empty() ? std::vector<SelSeed>(1) : seeds);
    auto d_heads = toDevice(queue, heads);
    auto d_lists = toDevice(queue, lists);
    auto d_out = cms::alpakatools::make_device_buffer<SelDevOut[]>(queue, std::max(n, 1));
    // zeroed: the two runs are compared whole, unused ranks included
    alpaka::memset(queue, d_out, 0);
    // a real HoT pool: home rows per seed, an extension slot of the fixed pool's size for every seed
    const ::mkfitdev::HotPoolLayout layout{kHotCap, poolHome, std::max(n, 1)};
    mkd::SeedCandsDeviceCollection dPoolSeeds(queue, std::max(n, 1));
    mkd::CandHotsDeviceCollection dPoolHots(queue, poolHome > 0 ? ::mkfitdev::hotPoolRows(std::max(n, 1), layout) : 1);
    if (poolHome > 0)
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                          KernelPoolInit{},
                          dPoolSeeds.view(),
                          layout,
                          std::max(n, 1));
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(std::max(n, 1), lanes),
                        KernelSel{},
                        d_seeds.data(),
                        dCands.const_view(),
                        dNodes.const_view(),
                        d_heads.data(),
                        asList ? d_lists.data() : nullptr,
                        d_out.data(),
                        n,
                        SelPoolArgs{dPoolSeeds.view(), dPoolHots.view(), poolHome > 0});
    if (poolInfo != nullptr) {
      PortableHostCollection<::mkfitdev::SeedCandsSoA> hPoolSeeds(queue, std::max(n, 1));
      alpaka::memcpy(queue, hPoolSeeds.buffer(), dPoolSeeds.buffer());
      alpaka::wait(queue);
      const auto v = hPoolSeeds.const_view();
      poolInfo->ext.assign(n, -1);
      poolInfo->nHots.assign(n, 0);
      poolInfo->overflow.assign(n, 0);
      for (int s = 0; s < n; ++s) {
        poolInfo->ext[s] = v[s].hotExt();
        poolInfo->nHots[s] = v[s].nHots();
        poolInfo->overflow[s] = v[s].overflowBits();
      }
      poolInfo->nExt = v.nHotExt();
    }
    return toHost<SelDevOut>(queue, d_out, n);
  }

  // one survivor, both sides: the HoT chain identifies it
  bool sameChain(const SelOut& ref, const SelDevOut& got, int k) {
    if (ref.nHits != got.nHits[k])
      return false;
    for (int h = 0; h < ref.nHits; ++h)
      if (ref.hitIdx[h] != got.hitIdx[k][h] || ref.hitLyr[h] != got.hitLyr[k][h])
        return false;
    return true;
  }

  bool sameBook(const SelOut& ref, const SelDevOut& got, int k) {
    bool ok = ref.nFound == got.nFound[k] && ref.nMissing == got.nMissing[k] && ref.nOverlap == got.nOverlap[k] &&
              ref.nInside == got.nInside[k] && ref.nTail == got.nTail[k] && closeScore(got.score[k], ref.score) &&
              closeScore(got.chi2[k], ref.chi2);
    for (int i = 0; i < 6; ++i)
      ok = ok && got.par[k][i] == ref.par[i];
    return ok;
  }

  int checkSel(Queue& queue, const std::vector<SelSeed>& seeds) {
    const int n = seeds.size();
    const auto out = runSel(queue, seeds, false);
    const auto outRev = runSel(queue, seeds, true);
    int identical = 0, tieOnly = 0, bad = 0, orderDependent = 0, nan = 0;
    int shown = 0;
    for (int s = 0; s < n; ++s) {
      const SelSeed& ref = seeds[s];
      const SelDevOut& got = out[s];
      orderDependent += std::memcmp(&got, &outRev[s], sizeof(SelDevOut)) != 0;
      nan += got.nan;
      bool ok = got.n == ref.nOut;
      bool tie = false;
      for (int k = 0; ok && k < ref.nOut; ++k) {
        if (sameChain(ref.out[k], got, k)) {
          ok = sameBook(ref.out[k], got, k);
          continue;
        }
        // another survivor at this rank: allowed only at a score equal to the reference's at this rank, and it must
        // be a reference survivor with that score, or a dropped competitor at a boundary tie
        int kr = -1;
        for (int j = 0; j < ref.nOut; ++j)
          if (sameChain(ref.out[j], got, k))
            kr = j;
        const bool sameScore = closeScore(got.score[k], ref.out[k].score);
        if (kr >= 0)
          ok = sameScore && sameBook(ref.out[kr], got, k) && ref.out[kr].score == ref.out[k].score;
        else
          ok = sameScore && ref.boundaryTie && ref.out[k].score == ref.out[ref.nOut - 1].score;
        tie = tie || ok;
      }
      if (!ok) {
        ++bad;
        if (shown++ < 5)
          std::printf("    seed %d: n ref %d got %d, first ranks ref score %g / got %g\n",
                      s,
                      ref.nOut,
                      got.n,
                      ref.nOut ? ref.out[0].score : 0.f,
                      got.n ? got.score[0] : 0.f);
      } else if (tie) {
        ++tieOnly;
      } else {
        ++identical;
      }
    }
    std::printf(
        "%-44s n=%6d identical %d, equal-score order only %d, differ %d, node-list order changes %d, NaN %d%s\n",
        "select_and_materialise",
        n,
        identical,
        tieOnly,
        bad,
        orderDependent,
        nan,
        bad + orderDependent + nan ? "  OUTSIDE THE GATE" : "");
    return bad + orderDependent + nan ? 1 : 0;
  }

  // The lanes of a block (V4's group) give the one-lane result, on seeds with engineered equal node scores and NaN
  // scores, the in-layer paths as a list (split over the lanes) or as the chain (lane 0), in either order
  int checkSelLanes(Queue& queue, std::vector<SelSeed> seeds) {
    int nTied = 0, nNan = 0;
    for (size_t s = 0; s < seeds.size(); ++s) {
      SelSeed& sd = seeds[s];
      // the features of an earlier node of the same TrackCand and depth: an equal score
      for (int k = 1; k < sd.nNodes; ++k) {
        for (int j = 0; j < k; ++j) {
          if (sd.nodes[j].ic == sd.nodes[k].ic && sd.nodes[j].nHits == sd.nodes[k].nHits) {
            sd.nodes[k].chi2Sum = sd.nodes[j].chi2Sum;
            sd.nodes[k].logDetVSum = sd.nodes[j].logDetVSum;
            ++nTied;
            break;
          }
        }
      }
      if (s % 3 == 0 && sd.nNodes > 0) {
        sd.nodes[sd.nNodes - 1].chi2Sum = std::numeric_limits<float>::quiet_NaN();
        ++nNan;
      }
    }
    const int n = seeds.size();
    const auto ref = runSel(queue, seeds, false);
    int differ = 0, nanRanked = 0, runs = 0;
    for (int lanes = 1; lanes <= int(mkd::kV2p2SelectSeedLanes); lanes *= 2) {
      for (bool asList : {false, true}) {
        for (bool reversed : {false, true}) {
          const auto got = runSel(queue, seeds, reversed, lanes, asList);
          ++runs;
          for (int s = 0; s < n; ++s)
            differ += std::memcmp(&got[s], &ref[s], sizeof(SelDevOut)) != 0;
        }
      }
    }
    for (int s = 0; s < n; ++s)
      nanRanked += ref[s].nan;
    std::printf("%-44s n=%6d tied nodes %d, NaN nodes %d (ranked %d), %d runs up to %d lanes: differ %d%s\n",
                "select_and_materialise on the lanes",
                n,
                nTied,
                nNan,
                nanRanked,
                runs,
                int(mkd::kV2p2SelectSeedLanes),
                differ,
                differ || (nNan > 0 && nanRanked == 0) ? "  OUTSIDE THE GATE" : "");
    return differ || (nNan > 0 && nanRanked == 0) ? 1 : 0;
  }

  //--------------------------------------------------------------------------
  // HoT pool: home rows and extension slots

  // selAddHitIdx (the selection and the K-select extras' stops, through SeedSelIO::hot) across the home rows into an
  // extension, then past the capacity: overflow bit, node not written
  int checkSelAddHome() {
    constexpr int kHome = 2, kCap = 6;
    ::mkfitdev::HoTNode home[kHome] = {}, ext[kCap - kHome] = {};
    int32_t nHots = 0;
    uint32_t ovf = 0;
    ::mkfitdev::SeedSelIO io{};
    io.hots = home;
    io.extHots = ext;
    io.hotHome = kHome;
    io.hotCap = kCap;
    io.nHots = &nHots;
    io.overflowBits = &ovf;
    ::mkfitdev::CandBook cand{};
    cand.lastHitIdx = -1;
    for (int h = 0; h <= kCap; ++h)
      ::mkfitdev::selAddHitIdx(io, cand, 10 + h, 1, 0.f);
    bool ok = nHots == kCap && (ovf & ::mkfitdev::kOverflowHotsBit) && cand.lastHitIdx == kCap - 1;
    for (int n = 0; n < kCap; ++n)
      ok = ok && io.hot(n).index == 10 + n && io.hot(n).prev == n - 1 &&
           &io.hot(n) == (n < kHome ? &home[n] : &ext[n - kHome]);
    std::printf("%-44s %s\n", "selAddHitIdx across the home rows", ok ? "as expected" : "WRONG  OUTSIDE THE GATE");
    return ok ? 0 : 1;
  }

  // Pool of kHotSeeds seed rows, kHotHome home rows each and one extension slot of kHotPerSeed - kHotHome rows; every
  // seed appends kHotAdd nodes (past its home rows) in seed order, so seed 0 takes the slot and the others overflow.
  constexpr int kHotSeeds = 3, kHotHome = 4, kHotPerSeed = 8, kHotExtSlots = 1, kHotAdd = 6;
  constexpr ::mkfitdev::HotPoolLayout kHotPool{kHotPerSeed, kHotHome, kHotExtSlots};

  struct KernelHotPool {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::mkfitdev::SeedCandsSoA::View seeds,
                                  ::mkfitdev::CandHotsSoA::View hots) const {
      if (!cms::alpakatools::once_per_grid(acc))
        return;
      seeds.hotsPerSeed() = kHotPerSeed;
      seeds.hotsHome() = kHotHome;
      seeds.hotExtSlots() = kHotExtSlots;
      seeds.nHotExt() = 0;
      seeds.nOverflowHots() = 0;
      for (int s = 0; s < kHotSeeds; ++s) {
        seeds.nHots(s) = 0;
        seeds.hotExt(s) = -1;
        seeds.overflowBits(s) = 0;
        seeds.nCands(s) = 1;
        seeds.curBuf(s) = 0;
        seeds.bestShortValid(s) = 0;
      }
      for (int s = 0; s < kHotSeeds; ++s) {
        const mkd::HotPool pool = mkd::hotPoolOf(seeds, hots);
        mkd::reserveHots(acc, pool, s, kHotAdd);
        ::mkfitdev::SeedCandsRef ref{};
        mkd::attachHots(pool, s, ref);
        ref.nHots = &seeds.nHots(s);
        ref.overflowBits = &seeds.overflowBits(s);
        ::mkfitdev::CandBook book{};
        book.lastHitIdx = -1;
        for (int h = 0; h < kHotAdd; ++h)
          ::mkfitdev::seedAddHitIdx(ref, book, 100 * s + h, s, 0.f);
        mkd::failSeedOnHotOverflow(acc, seeds, s, 0u);
      }
    }
  };

  int checkHotPool(Queue& queue) {
    mkd::SeedCandsDeviceCollection seeds(queue, kHotSeeds);
    mkd::CandHotsDeviceCollection hots(queue, ::mkfitdev::hotPoolRows(kHotSeeds, kHotPool));
    alpaka::exec<Acc1D>(queue, cms::alpakatools::make_workdiv<Acc1D>(1, 1), KernelHotPool{}, seeds.view(), hots.view());
    PortableHostCollection<::mkfitdev::SeedCandsSoA> seedsH(queue, kHotSeeds);
    PortableHostCollection<::mkfitdev::CandHotsSoA> hotsH(queue, ::mkfitdev::hotPoolRows(kHotSeeds, kHotPool));
    alpaka::memcpy(queue, seedsH.buffer(), seeds.buffer());
    alpaka::memcpy(queue, hotsH.buffer(), hots.buffer());
    alpaka::wait(queue);
    const auto sv = seedsH.const_view();
    const auto hv = hotsH.const_view();
    // seed 0: every node, the ones past the home rows in the extension rows after all home rows
    bool ok = sv[0].nHots() == kHotAdd && sv[0].hotExt() == 0 && sv[0].overflowBits() == 0 &&
              ::mkfitdev::hotNodeRow(sv, 0, kHotHome) == kHotSeeds * kHotHome;
    for (int h = 0; h < kHotAdd; ++h)
      ok = ok && hv[::mkfitdev::hotNodeRow(sv, 0, h)].node().index == h &&
           hv[::mkfitdev::hotNodeRow(sv, 0, h)].node().prev == h - 1;
    // seeds 1, 2: no slot left, their home rows only, failed and counted
    for (int s = 1; s < kHotSeeds; ++s) {
      ok = ok && sv[s].nHots() == kHotHome && sv[s].hotExt() == -1 &&
           (sv[s].overflowBits() & ::mkfitdev::kOverflowHotsBit) && sv[s].nCands() == 0;
      for (int h = 0; h < kHotHome; ++h)
        ok = ok && hv[::mkfitdev::hotNodeRow(sv, s, h)].node().index == 100 * s + h;
    }
    ok = ok && sv.nHotExt() == uint32_t(kHotSeeds) && sv.nOverflowHots() == uint32_t(kHotSeeds - 1);
    std::printf("%-44s %d seeds, %d extension slot: %s\n",
                "HoT pool home rows + extension",
                kHotSeeds,
                kHotExtSlots,
                ok ? "as expected" : "WRONG  OUTSIDE THE GATE");
    return ok ? 0 : 1;
  }

  // The selection on a real HoT pool gives the fixed-pool result, on the lanes of a block, the paths as a list or as
  // the chain, either order: with few home rows (the TrackCands' chains already take a slot, the selection's reserve
  // finds it) and with as many as the longest initial chains (the survivors take the slot in the selection).
  int checkSelPool(Queue& queue, const std::vector<SelSeed>& seeds) {
    const int n = seeds.size();
    int maxInit = 1;
    for (const SelSeed& sd : seeds) {
      int init = 0;
      for (int ic = 0; ic < sd.nCands; ++ic)
        init += sd.cands[ic].nHits;
      maxInit = std::max(maxInit, init);
    }
    const auto ref = runSel(queue, seeds, false);
    int differ = 0, wrongPool = 0, runs = 0, slotsInSelect = 0, crossing = 0;
    for (int home : {std::min(4, maxInit), maxInit}) {
      for (int lanes = 1; lanes <= int(mkd::kV2p2SelectSeedLanes); lanes *= 2) {
        for (bool asList : {false, true}) {
          for (bool reversed : {false, true}) {
            SelPoolInfo info;
            const auto got = runSel(queue, seeds, reversed, lanes, asList, home, &info);
            ++runs;
            for (int s = 0; s < n; ++s) {
              differ += std::memcmp(&got[s], &ref[s], sizeof(SelDevOut)) != 0;
              int init = 0;
              for (int ic = 0; ic < seeds[s].nCands; ++ic)
                init += seeds[s].cands[ic].nHits;
              // a slot exactly when the seed's nodes pass its home rows, never an overflow
              wrongPool += (info.ext[s] >= 0) != (info.nHots[s] > home) || info.overflow[s] != 0;
              crossing += info.nHots[s] > home;
              slotsInSelect += home == maxInit && init <= home && info.nHots[s] > home;
            }
          }
        }
      }
    }
    const bool bad = differ || wrongPool || slotsInSelect == 0;
    std::printf(
        "%-44s n=%6d %d runs: differ %d, wrong pool state %d, past home %d (slot taken by the selection %d)%s\n",
        "select_and_materialise on a real HoT pool",
        n,
        runs,
        differ,
        wrongPool,
        crossing,
        slotsInSelect,
        bad ? "  OUTSIDE THE GATE" : "");
    return bad ? 1 : 0;
  }

  // Export of a seed whose chain passes its home rows (extension slot) and holds a missing hit, next to a seed at home.
  constexpr int kExpHome = 4, kExpPerSeed = 16, kExpHits0 = 10, kExpMiss0 = 5, kExpHits1 = 3;
  constexpr ::mkfitdev::HotPoolLayout kExpPool{kExpPerSeed, kExpHome, 1};

  struct KernelExportInit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::mkfitdev::SeedCandsSoA::View seeds,
                                  ::mkfitdev::CandSlotsSoA::View slots,
                                  ::mkfitdev::CandHotsSoA::View hots,
                                  int32_t* nKept) const {
      if (!cms::alpakatools::once_per_grid(acc))
        return;
      seeds.hotsPerSeed() = kExpPool.perSeed;
      seeds.hotsHome() = kExpPool.home;
      seeds.hotExtSlots() = kExpPool.extSlots;
      seeds.nHotExt() = 0;
      seeds.nOverflowHots() = 0;
      *nKept = 2;
      const mkd::HotPool pool = mkd::hotPoolOf(seeds, hots);
      for (int s = 0; s < 2; ++s) {
        const int nh = s == 0 ? kExpHits0 : kExpHits1;
        seeds.nHots(s) = 0;
        seeds.hotExt(s) = -1;
        seeds.overflowBits(s) = 0;
        seeds.nCands(s) = 1;
        seeds.curBuf(s) = 0;
        mkd::reserveHots(acc, pool, s, nh);
        ::mkfitdev::SeedCandsRef ref{};
        mkd::attachHots(pool, s, ref);
        ref.nHots = &seeds.nHots(s);
        ref.overflowBits = &seeds.overflowBits(s);
        ::mkfitdev::CandBook book{};
        book.lastHitIdx = -1;
        for (int h = 0; h < nh; ++h)
          ::mkfitdev::seedAddHitIdx(
              ref, book, s == 0 && h == kExpMiss0 ? ::mkfitdev::kHitMissIdx : 100 * s + h, 10 + h, 0.f);
        slots.book(::mkfitdev::candSlotRow(s, 0, 0)) = book;
        slots.state(::mkfitdev::candSlotRow(s, 0, 0)) = ::mkfitdev::CandState{};
      }
    }
  };

  int checkExportExtension(Queue& queue) {
    mkd::SeedCandsDeviceCollection seeds(queue, 2);
    mkd::CandSlotsDeviceCollection slots(queue, 2 * ::mkfitdev::kSlotsPerSeed);
    mkd::CandHotsDeviceCollection hots(queue, ::mkfitdev::hotPoolRows(2, kExpPool));
    auto nKept = cms::alpakatools::make_device_buffer<int32_t>(queue);
    auto broken = cms::alpakatools::make_device_buffer<uint32_t>(queue);
    alpaka::memset(queue, broken, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                        KernelExportInit{},
                        seeds.view(),
                        slots.view(),
                        hots.view(),
                        nKept.data());
    mkd::TrackSoADeviceCollection out(queue, 2);
    mkd::seeds::exportBestCands(
        queue, seeds.const_view(), slots.const_view(), hots.const_view(), nKept.data(), 2, out.view(), broken.data());
    PortableHostCollection<::mkfitdev::TrackSoA> outH(queue, 2);
    PortableHostCollection<::mkfitdev::SeedCandsSoA> seedsH(queue, 2);
    auto brokenH = cms::alpakatools::make_host_buffer<uint32_t>(queue);
    alpaka::memcpy(queue, outH.buffer(), out.buffer());
    alpaka::memcpy(queue, seedsH.buffer(), seeds.buffer());
    alpaka::memcpy(queue, brokenH, broken);
    alpaka::wait(queue);
    const auto ov = outH.const_view();
    const auto sv = seedsH.const_view();
    bool ok = ov.nTracks() == 2 && *brokenH.data() == 0 && sv[0].hotExt() == 0 && sv[1].hotExt() == -1 &&
              ::mkfitdev::hotNodeRow(sv, 0, kExpHome) == 2 * kExpHome;
    // the found hits in chain order, the missing one dropped (remove_missing_hits)
    ok = ok && ov[0].nTotalHits() == kExpHits0 - 1 && ov[1].nTotalHits() == kExpHits1;
    for (int h = 0, k = 0; ok && h < kExpHits0; ++h) {
      if (h == kExpMiss0)
        continue;
      ok = ov[0].hits().hot[k].index == h && ov[0].hits().hot[k].layer == 10 + h;
      ++k;
    }
    for (int h = 0; ok && h < kExpHits1; ++h)
      ok = ov[1].hits().hot[h].index == 100 + h && ov[1].hits().hot[h].layer == 10 + h;
    std::printf("%-44s %s\n", "export of a chain with an extension", ok ? "as expected" : "WRONG  OUTSIDE THE GATE");
    return ok ? 0 : 1;
  }

  // In-place filter, kept-row list and backward fit over it. Seed rows [0, kFltKept) of a capacity of kFltCap: rows
  // with s % 3 == 1 have too few hits, row kFltEmpty has no candidates, row kFltOverflow a HoT overflow; the rows past
  // kFltKept are empty (KernelSeedInitCands). The list buffer starts as all zeros, the content a reused or fresh
  // buffer may hold: a list entry past the kept seeds that is not rewritten would fit row 0 again.
  constexpr int kFltCap = 20, kFltKept = 14, kFltEmpty = 6, kFltOverflow = 9, kFltMinHits = 3;
  bool fltSurvives(int s) { return s < kFltKept && s % 3 != 1 && s != kFltEmpty && s != kFltOverflow; }

  struct KernelFilterInit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::mkfitdev::SeedCandsSoA::View seeds,
                                  ::mkfitdev::CandSlotsSoA::View slots,
                                  int32_t* nKept) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        seeds.hotsPerSeed() = 1;
        seeds.hotsHome() = 1;
        seeds.hotExtSlots() = 0;
        *nKept = kFltKept;
      }
      for (int32_t s : cms::alpakatools::uniform_elements(acc, kFltCap)) {
        const bool kept = s < kFltKept;
        seeds.nCands(s) = kept && s != kFltEmpty ? 1 : 0;
        seeds.curBuf(s) = 0;
        seeds.state(s) = ::mkfitdev::kFinished;
        seeds.overflowBits(s) = s == kFltOverflow ? ::mkfitdev::kOverflowHotsBit : 0u;
        seeds.bestShortValid(s) = 0;
        seeds.bkwRepacked(s) = 0;
        seeds.nHots(s) = 0;
        seeds.hotExt(s) = -1;
        ::mkfitdev::CandBook book{};
        book.lastHitIdx = -1;
        book.nFound = s % 3 == 1 ? kFltMinHits - 1 : kFltMinHits;
        slots.book(::mkfitdev::candSlotRow(s, 0, 0)) = book;
        slots.state(::mkfitdev::candSlotRow(s, 0, 0)) = ::mkfitdev::CandState{};
      }
    }
  };

  // Backward-fit IO that only counts the copy-outs per candidate (no hits: every candidate is done at once).
  struct BkFitCountIO {
    ::mkfitdev::SeedCandsSoA::ConstView seeds;
    int32_t* stores;
    ALPAKA_FN_HOST_ACC bool active(int c) const { return seeds[c].nCands() > 0; }
    template <::mkfitdev::Matriplex::idx_t N>
    ALPAKA_FN_HOST_ACC void loadCand(
        int, ::mkfitdev::MPlexLS<N>& err, ::mkfitdev::MPlexLV<N>& par, ::mkfitdev::MPlexQI<N>& chg, int n) const {
      const float p[6] = {1.f, 0.f, 0.f, 1.f, 0.f, 1.f};
      float e[21] = {};
      par.copyIn(n, p);
      err.copyIn(n, e);
      chg.At(n, 0, 0) = 1;
    }
    ALPAKA_FN_HOST_ACC int lastNode(int) const { return -1; }
    ALPAKA_FN_HOST_ACC int nodeIndex(int, int) const { return -1; }
    ALPAKA_FN_HOST_ACC int nodeLayer(int, int) const { return -1; }
    ALPAKA_FN_HOST_ACC int nodePrev(int, int) const { return -1; }
    template <::mkfitdev::Matriplex::idx_t N>
    ALPAKA_FN_HOST_ACC void loadHit(int,
                                    int,
                                    ::mkfitdev::MPlexHS<N>&,
                                    ::mkfitdev::MPlexHV<N>&,
                                    ::mkfitdev::MPlexHV<N>&,
                                    ::mkfitdev::MPlexHV<N>&,
                                    ::mkfitdev::MPlexHV<N>&,
                                    int) const {}
    template <::mkfitdev::Matriplex::idx_t N>
    ALPAKA_FN_HOST_ACC void storeCand(int c,
                                      const ::mkfitdev::MPlexLS<N>&,
                                      const ::mkfitdev::MPlexLV<N>&,
                                      const ::mkfitdev::MPlexQI<N>&,
                                      float,
                                      int) const {
      ++stores[c];  // one thread per candidate on GPU backends, one thread in all on CPU ones
    }
    ALPAKA_FN_HOST_ACC float candPt(int) const { return 1.f; }
    ALPAKA_FN_HOST_ACC void markOutlier(int, int) const {}
    ALPAKA_FN_HOST_ACC void recountHolesAndOverlaps(int) const {}
  };

  int checkFilterKeptRows(Queue& queue, const PropagationFlags& pf) {
    mkd::SeedCandsDeviceCollection seeds(queue, kFltCap);
    mkd::CandSlotsDeviceCollection slots(queue, kFltCap * ::mkfitdev::kSlotsPerSeed);
    mkd::CandHotsDeviceCollection hots(queue, kFltCap);
    auto nKept = cms::alpakatools::make_device_buffer<int32_t>(queue);
    auto rows = cms::alpakatools::make_device_buffer<int32_t[]>(queue, kFltCap);
    auto stores = cms::alpakatools::make_device_buffer<int32_t[]>(queue, kFltCap);
    alpaka::memset(queue, rows, 0);
    alpaka::memset(queue, stores, 0);
    launch(queue, kFltCap, KernelFilterInit{}, seeds.view(), slots.view(), nKept.data());
    launch(queue,
           kFltCap,
           mkd::KernelEngineFilter{},
           seeds.view(),
           slots.view(),
           hots.view(),
           false,
           kFltMinHits,
           ::mkfitdev::BkwSearchGate{},
           kFltCap,
           nKept.data());
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                        mkd::KernelEngineKeptRows{},
                        seeds.const_view(),
                        rows.data(),
                        kFltCap,
                        nKept.data());
    mkd::launchBkFit(queue, BkFitCountIO{seeds.const_view(), stores.data()}, kFltCap, rows.data(), pf);
    const auto rowsH = toHost<int32_t>(queue, rows, kFltCap);
    const auto storesH = toHost<int32_t>(queue, stores, kFltCap);
    std::vector<int32_t> rowsRef;
    for (int s = 0; s < kFltCap; ++s)
      if (fltSurvives(s))
        rowsRef.push_back(s);
    const int nSurv = rowsRef.size();
    rowsRef.resize(kFltCap, -1);
    int wrongRows = 0, wrongFits = 0, maxFits = 0;
    for (int k = 0; k < kFltCap; ++k) {
      wrongRows += rowsH[k] != rowsRef[k];
      wrongFits += storesH[k] != (fltSurvives(k) ? 1 : 0);
      maxFits = std::max(maxFits, int(storesH[k]));
    }
    const bool bad = wrongRows || wrongFits;
    std::printf("%-44s %d of %d rows kept: list entries wrong %d, seeds fitted a wrong number of times %d (max %d)%s\n",
                "filter + kept-row list + backward fit",
                nSurv,
                kFltKept,
                wrongRows,
                wrongFits,
                maxFits,
                bad ? "  OUTSIDE THE GATE" : "");
    return bad ? 1 : 0;
  }

  // The backward fit's outlier rejection on hand-built chains (BkFitCandsIO: markOutlier on the rejected nodes, then
  // recountHolesAndOverlaps): the expected hole and overlap counts of TrackCand::recountHolesAndOverlaps' rules.
  constexpr int kRcNodes = 20, kRcMaxOutl = 2;
  constexpr int kMiss = ::mkfitdev::kHitMissIdx, kEdge = ::mkfitdev::kHitEdgeIdx, kCCC = ::mkfitdev::kHitCCCFilterIdx,
                kGap = ::mkfitdev::kHitInGapIdx, kStop = ::mkfitdev::kHitStopIdx;
  constexpr ::mkfitdev::HotPoolLayout kRcPool{kRcNodes, kRcNodes, 0};
  struct RecountCase {
    int nSeed, n;
    int idx[kRcNodes], layer[kRcNodes];
    int nOutl, outl[kRcMaxOutl];
    int inside, tail, overlap;  // expected after the rejection
  };
  const std::vector<std::pair<const char*, RecountCase>> kRecountCases = {
      {"inside hole", {3, 7, {0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5, 6}, 1, {4}, 1, 0, 0}},
      {"tail holes", {3, 7, {0, 1, 2, 3, 4, 5, kMiss}, {0, 1, 2, 3, 4, 5, 6}, 1, {5}, 0, 2, 0}},
      {"overlap kept, tail hole", {3, 8, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 4, 5, 6}, 1, {7}, 0, 1, 1}},
      {"miss before an overlap hit of its layer",
       {3, 8, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 4, 5, 6}, 1, {4}, 0, 0, 0}},
      {"miss after an overlap hit of its layer",
       {3, 8, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 4, 5, 6}, 1, {5}, 0, 0, 0}},
      {"both hits of a layer", {3, 7, {0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 4, 5}, 2, {4, 5}, 1, 0, 0}},
      {"miss, miss on one layer", {3, 7, {0, 1, 2, 3, kMiss, kMiss, 6}, {0, 1, 2, 3, 4, 4, 5}, 0, {}, 1, 0, 0}},
      {"found, miss, miss on one layer",
       {3, 8, {0, 1, 2, 3, 4, kMiss, kMiss, 7}, {0, 1, 2, 3, 4, 4, 4, 5}, 0, {}, 0, 0, 0}},
      {"seed hit", {3, 5, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, 1, {1}, 1, 0, 0}},
      {"seed hits of one layer", {3, 7, {0, 1, 2, 3, 4, 5, 6}, {0, 1, 1, 1, 2, 2, 3}, 1, {3}, 1, 0, 1}},
      {"17-hit seed, two seed hits on one layer",
       {17,
        19,
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17},
        0,
        {},
        0,
        0,
        0}},
      {"edge, CCC, gap and stop nodes",
       {3, 9, {0, 1, 2, 3, kEdge, kCCC, 6, kGap, kStop}, {0, 1, 2, 3, 4, 5, 6, 7, 8}, 2, {3, 6}, 1, 1, 0}},
  };

  struct KernelRecountInit {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::mkfitdev::SeedCandsSoA::View seeds,
                                  ::mkfitdev::CandSlotsSoA::View slots,
                                  ::mkfitdev::CandHotsSoA::View hots,
                                  const RecountCase* cases,
                                  int n) const {
      if (!cms::alpakatools::once_per_grid(acc))
        return;
      seeds.hotsPerSeed() = kRcPool.perSeed;
      seeds.hotsHome() = kRcPool.home;
      seeds.hotExtSlots() = kRcPool.extSlots;
      const mkd::HotPool pool = mkd::hotPoolOf(seeds, hots);
      for (int s = 0; s < n; ++s) {
        seeds.nHots(s) = 0;
        seeds.hotExt(s) = -1;
        seeds.overflowBits(s) = 0;
        seeds.nCands(s) = 1;
        seeds.curBuf(s) = 0;
        ::mkfitdev::SeedCandsRef ref{};
        mkd::attachHots(pool, s, ref);
        ref.nHots = &seeds.nHots(s);
        ref.overflowBits = &seeds.overflowBits(s);
        ::mkfitdev::CandBook book{};
        book.lastHitIdx = -1;
        for (int h = 0; h < cases[s].n; ++h)
          ::mkfitdev::seedAddHitIdx(ref, book, cases[s].idx[h], cases[s].layer[h], 0.f);
        slots.book(::mkfitdev::candSlotRow(s, 0, 0)) = book;
        ::mkfitdev::CandState state{};
        state.nSeedHits = cases[s].nSeed;
        slots.state(::mkfitdev::candSlotRow(s, 0, 0)) = state;
      }
    }
  };

  struct KernelRecount {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::mkfitdev::SeedCandsSoA::ConstView seeds,
                                  ::mkfitdev::CandSlotsSoA::View slots,
                                  ::mkfitdev::CandHotsSoA::View hots,
                                  const RecountCase* cases,
                                  int n) const {
      const mkd::BkFitCandsIO io{seeds, slots, hots, ::mkfitdev::EngineHitInputs{}};
      for (int32_t s : cms::alpakatools::uniform_elements(acc, n)) {
        for (int k = 0; k < cases[s].nOutl; ++k)
          io.markOutlier(s, cases[s].outl[k]);
        io.recountHolesAndOverlaps(s);
      }
    }
  };

  int checkRecount(Queue& queue) {
    std::vector<RecountCase> cases;
    for (auto const& c : kRecountCases)
      cases.push_back(c.second);
    const int n = cases.size();
    auto d_cases = toDevice(queue, cases);
    mkd::SeedCandsDeviceCollection seeds(queue, n);
    mkd::CandSlotsDeviceCollection slots(queue, n * ::mkfitdev::kSlotsPerSeed);
    mkd::CandHotsDeviceCollection hots(queue, ::mkfitdev::hotPoolRows(n, kRcPool));
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                        KernelRecountInit{},
                        seeds.view(),
                        slots.view(),
                        hots.view(),
                        d_cases.data(),
                        n);
    launch(queue, n, KernelRecount{}, seeds.const_view(), slots.view(), hots.view(), d_cases.data(), n);
    PortableHostCollection<::mkfitdev::CandSlotsSoA> slotsH(queue, n * ::mkfitdev::kSlotsPerSeed);
    alpaka::memcpy(queue, slotsH.buffer(), slots.buffer());
    alpaka::wait(queue);
    int wrong = 0;
    for (int s = 0; s < n; ++s) {
      const ::mkfitdev::CandBook b = slotsH.const_view()[::mkfitdev::candSlotRow(s, 0, 0)].book();
      if (b.nInsideMinusOne != cases[s].inside || b.nTailMinusOne != cases[s].tail || b.nOverlap != cases[s].overlap) {
        ++wrong;
        std::printf("  recount %s: inside / tail / overlap %d / %d / %d, expected %d / %d / %d\n",
                    kRecountCases[s].first,
                    b.nInsideMinusOne,
                    b.nTailMinusOne,
                    b.nOverlap,
                    cases[s].inside,
                    cases[s].tail,
                    cases[s].overlap);
      }
    }
    std::printf("%-44s %d chains: counts wrong %d%s\n",
                "outlier rejection: hole and overlap recount",
                n,
                wrong,
                wrong ? "  OUTSIDE THE GATE" : "");
    return wrong ? 1 : 0;
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s mkfitcore_ref.bin\n", argv[0]);
    return 1;
  }
  RefData d;
  if (!readFile(argv[1], d)) {
    std::printf("cannot read %s\n", argv[1]);
    return 2;
  }
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::printf("No devices available for the %s backend, skipping.\n", EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE));
    return 0;
  }
  Queue queue(devices[0]);

  auto d_bbxi = toDevice(queue, d.bbxi);
  auto d_radl = toDevice(queue, d.radl);
  // field constants: the MkFitCore defaults of mkfit::Config::mag_* (building) and mag_fit_* (final fit)
  const ::mkfitdev::MaterialView mv{d_bbxi.data(),
                                    d_radl.data(),
                                    d.head.matBinsZ,
                                    d.head.matBinsR,
                                    d.head.matBinsZ / d.head.matRangeZ,
                                    d.head.matBinsR / d.head.matRangeR,
                                    ::mkfitdev::Config::BFieldParams{},
                                    ::mkfitdev::Config::BFieldParams{::mkfitdev::Config::mag_fit_c1,
                                                                     ::mkfitdev::Config::mag_fit_b0,
                                                                     ::mkfitdev::Config::mag_fit_b1,
                                                                     ::mkfitdev::Config::mag_fit_a}};
  // the finding flags of Phase 2, inter- and intra-layer alike
  using namespace ::mkfitdev::prop;
  const PropagationFlags pf(PF_use_param_b_field | PF_apply_material, mv);

  int nBad = 0;
  nBad += checkPq(queue, d.pq);
  nBad += checkScore(queue, d.score);
  nBad += checkKal(queue, d.kal, pf, d.head.hitChi2Cut);
  nBad += checkWin(queue, d.win);
  nBad += checkSel(queue, d.sel);
  nBad += checkSelLanes(queue, d.sel);
  nBad += checkHotPool(queue);
  nBad += checkSelAddHome();
  nBad += checkHarvestAccept();
  nBad += checkDensityAcrossPi(queue);
  nBad += checkPhiMarginCap(queue);
  nBad += checkWindowFactors(queue);
  nBad += checkOutsideHole();
  nBad += checkAtan2Poly(queue);
  nBad += checkSelPool(queue, d.sel);
  nBad += checkExportExtension(queue);
  nBad += checkFilterKeptRows(queue, pf);
  nBad += checkRecount(queue);
  std::printf("%s %s: %d block(s) outside the gate\n",
              nBad ? "FAIL" : "PASS",
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE),
              nBad);
  return nBad ? 3 : 0;
}
