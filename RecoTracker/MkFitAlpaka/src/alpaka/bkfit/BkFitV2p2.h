#ifndef RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitV2p2_h
#define RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitV2p2_h

// Backward fit of MkFinderV2p2 tracks: port of MkFinder::bkFitFitTracksV2p2 (MkFinderV2p2-BkFit.cc), which
// MkBuilder::fit_cands picks when the candidates come from findTracksStandardv2p2 (always propagation to planes; no
// PCA step: backward_fit_to_pca is refused by SupportedConfig). Input as bkFitInputTracks (errors scaled by 100, chi2
// reset); the HoT chain is walked backwards in runs of one physical layer (TrackerInfo::get_double_layers), each run
// fitted nearest module plane first; every hit is fitted, holes and repeated hits are skipped; outlier rejection and
// copy-out as bkFitFitTracksProp2Plane. Instantiated only in src/alpaka/BkFit.dev.cc (BkFitKernel.h).

#include <cmath>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/bkfit/BkFitTypes.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationMPlex.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // TIO as bkFitFitTracksProp2Plane (BkFitKernel.h), plus:
  //   int doubleLayerPartner(int layer)                     LayerInfo::double_layer_partner(), -1 = none
  //   const EngineModule& nodeModule(int c, int k)          module_info(detIDinLayer) of the hit of node k
  template <idx_t N, typename TIO>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void bkFitFitTracksV2p2(const TIO& io,
                                                              const int (&cand)[N],
                                                              const int N_proc,
                                                              const PropagationFlags& pflags,
                                                              const ::mkfitdev::bkfit::OutlierParams& outl) {
    MPlexLS<N> errC, errP;
    MPlexLV<N> parC, parP;
    MPlexQI<N> chg(0);
    MPlexQF<N> chi2;
    int curNode[N];

    // ---- MkFinder::bkFitInputTracks(EventOfCombCandidates&, beg, end) ----
    for (int i = 0; i < N_proc; ++i) {
      io.loadCand(cand[i], errC, parC, chg, i);
      curNode[i] = io.lastNode(cand[i]);
    }
    chi2.setVal(0);
    errC.scale(100.0f);

    // ---- MkFinder::bkFitFitTracksV2p2 ----
    // Computed at full width, see pad_free_lanes(); only the first N_proc slots are read out.
    ::mkfitdev::pad_free_lanes<N>(N_proc, parC, errC, chg);
    MPlexQF<N> tmp_chi2(0.0f);
    int done_flag[N];
    // The last hit fitted per lane, to skip a repeated hit on track (seed-hit merging can put the same hit on twice).
    int fit_lay[N], fit_idx[N];
    for (int i = 0; i < N; ++i) {
      done_flag[i] = 0;
      fit_lay[i] = fit_idx[i] = -1;
    }

    MPlexHV<N> plNrm(0.0f);
    MPlexHV<N> plDir(0.0f);
    MPlexHV<N> plPnt(0.0f);
    MPlexHS<N> msErr(0.0f);
    MPlexHV<N> msPar(0.0f);
    MPlexQI<N> failFlag(0);

    int n_outliers[N];
    for (int i = 0; i < N; ++i)
      n_outliers[i] = 0;

    // Path order within a physical layer: a lane coming to a physical layer takes all its hits on that layer, a run,
    // and fits them nearest module plane first, by the straight-line path from its state at the start of the run.
    constexpr int kMaxRun = 16;  // a longer run is split
    int run_node[N][kMaxRun];
    int run_n[N], run_pos[N];
    for (int i = 0; i < N; ++i)
      run_n[i] = run_pos[i] = 0;

    auto collect_run = [&](int i) {
      run_n[i] = run_pos[i] = 0;
      int dl_first = -1, dl_second = -1;
      for (; curNode[i] >= 0; curNode[i] = io.nodePrev(cand[i], curNode[i])) {
        const int hot_layer = io.nodeLayer(cand[i], curNode[i]);
        const int hot_index = io.nodeIndex(cand[i], curNode[i]);
        if (hot_index < 0)
          continue;
        // A repeat of the hit just fitted or of one in the run.
        bool repeat = hot_layer == fit_lay[i] && hot_index == fit_idx[i];
        for (int k = 0; k < run_n[i] && !repeat; ++k)
          repeat =
              io.nodeLayer(cand[i], run_node[i][k]) == hot_layer && io.nodeIndex(cand[i], run_node[i][k]) == hot_index;
        if (repeat)
          continue;
        if (dl_first < 0) {
          // TrackerInfo::get_double_layers()
          const int p = io.doubleLayerPartner(hot_layer);
          dl_first = p < 0 ? hot_layer : (hot_layer < p ? hot_layer : p);
          dl_second = p < 0 ? hot_layer : (hot_layer < p ? p : hot_layer);
        } else if ((hot_layer != dl_first && hot_layer != dl_second) || run_n[i] == kMaxRun)
          break;
        run_node[i][run_n[i]++] = curNode[i];
      }
      if (run_n[i] < 2)
        return;

      // s = (plane point - x) . n / (p^ . n): positive ahead along the momentum, negative behind.
      // The fit walks backwards, so the largest s comes first; equal s keep the chain order.
      const float x = parC.At(i, 0, 0), y = parC.At(i, 1, 0), z = parC.At(i, 2, 0);
      const float phi = parC.At(i, 4, 0), theta = parC.At(i, 5, 0);
      const float px = std::cos(phi) * std::sin(theta), py = std::sin(phi) * std::sin(theta), pz = std::cos(theta);
      float s[kMaxRun];
      for (int k = 0; k < run_n[i]; ++k) {
        const ::mkfitdev::EngineModule& mi = io.nodeModule(cand[i], run_node[i][k]);
        const float nx = mi.nrm[0], ny = mi.nrm[1], nz = mi.nrm[2];
        s[k] = ((mi.pnt[0] - x) * nx + (mi.pnt[1] - y) * ny + (mi.pnt[2] - z) * nz) / (px * nx + py * ny + pz * nz);
      }
      // insertion by strict >, as MkFitCore's swap loop (bounded: at most kMaxRun entries)
      for (int k = 1; k < run_n[i]; ++k) {
        for (int j = k; j > 0 && s[j] > s[j - 1]; --j) {
          const float ts = s[j];
          s[j] = s[j - 1];
          s[j - 1] = ts;
          const int tn = run_node[i][j];
          run_node[i][j] = run_node[i][j - 1];
          run_node[i][j - 1] = tn;
        }
      }
    };

    int done_count = 0;
    while (done_count != N_proc) {
      int fit_node[N];  // HoT node fitted in this step, -1 if none
      for (int i = 0; i < N; ++i)
        fit_node[i] = -1;
      int here_count = 0;
      for (int i = 0; i < N_proc; ++i) {
        if (done_flag[i])
          continue;

        if (run_pos[i] == run_n[i])
          collect_run(i);

        if (run_pos[i] == run_n[i]) {
          // Mark as done and copy out.
          done_flag[i] = 1;
          ++done_count;
          // Outliers became misses: the hole and overlap counts follow before the score.
          if (n_outliers[i] > 0)
            io.recountHolesAndOverlaps(cand[i]);
          io.storeCand(cand[i], errC, parC, chg, chi2.At(i, 0, 0), i);
        } else {
          // Prepare the next hit of the run and its module info. Every hit is fitted, overlap and sister hits
          // included.
          const int node = run_node[i][run_pos[i]++];
          fit_lay[i] = io.nodeLayer(cand[i], node);
          fit_idx[i] = io.nodeIndex(cand[i], node);

          io.loadHit(cand[i], node, msErr, msPar, plNrm, plDir, plPnt, i);

          fit_node[i] = node;
          ++here_count;
        }
      }

      if (done_count == N_proc)
        break;
      if (here_count == 0)
        continue;

      ::mkfitdev::pad_free_lanes<N>(N_proc, msErr, msPar, plNrm, plDir, plPnt);

      // The propagation fail flag is not checked, as in MkFitCore.
      failFlag.setVal(0);
      const MPlexQI<N> chg_prev = chg;
      propagateHelixToPlaneMPlex(errC, parC, chg, plPnt, plNrm, errP, parP, failFlag, N, pflags);
      kalmanOperationPlaneLocal(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                                errP,
                                parP,
                                chg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                errC,
                                parC,
                                tmp_chi2,
                                N);
      kalmanCheckChargeFlip(parC, chg, N);

      // Outlier rejection: the hit becomes a missing hit and the propagated state is kept. The negated comparison
      // leaves a NaN chi2 alone.
      if (outl.chi2 > 0.f) {
        for (int i = 0; i < N_proc; ++i) {
          if (fit_node[i] < 0 || !(tmp_chi2.At(i, 0, 0) > outl.chi2) || n_outliers[i] >= outl.maxOutliers ||
              io.candPt(cand[i]) < outl.minPt)
            continue;
          errC.copySlot(i, errP);
          parC.copySlot(i, parP);
          chg.At(i, 0, 0) = chg_prev.constAt(i, 0, 0);
          tmp_chi2.At(i, 0, 0) = 0.f;
          io.markOutlier(cand[i], fit_node[i]);
          ++n_outliers[i];
        }
      }

      // update chi2
      chi2.add(tmp_chi2);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
