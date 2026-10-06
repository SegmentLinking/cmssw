#ifndef RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitKernel_h
#define RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitKernel_h

// Backward fit of the best candidate of every seed: port of MkBuilder::fit_cands (usePropToPlane branch,
// backward_fit_to_pca = false as in the Phase-2 LST step) = MkFinder::bkFitInputTracks(EventOfCombCandidates&)
// + MkFinder::bkFitFitTracksProp2Plane (MkFinder.cc:1967-2070, 2450-2560 of CMSSW_20_1_0_pre2).
// Instantiated only in src/alpaka/BkFit.dev.cc; callers use BkFitLaunch.h.
//
// MkFitCore behaviour reproduced on purpose:
//   a) input errors scaled by 100 (m_Err[iC].scale(100.0f));
//   b) chi2 reset to 0 at input and re-accumulated over the backward-fit updates only (MkFinder.cc:1984, 2662): the
//      forward chi2 is replaced by the backward-fit chi2, and the score is recomputed with it ( b said the
//      chi2 stays 0; the MkFitCore dump shows it does not);
//   c) hits on the same layer are collapsed to the EARLIEST node of the run (overlap hits never enter the fit);
//   d) kalmanCheckChargeFlip after every update; the propagation fail flag is ignored.
// One Alpaka thread fits N = kNN candidates with the MkFitCore Matriplex loop (N = 1 on GPU, 8 on CPU); slots are
// independent, so the grouping does not change any result.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/Packers.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/bkfit/BkFitTypes.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationMPlex.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // ---- the MkFitCore algorithm, generic over where candidates / nodes / hits come from ----
  // TIO provides (c = candidate id, k = node index in the candidate's pool):
  //   void loadCand(int c, MPlexLS<N>&, MPlexLV<N>&, MPlexQI<N>&, int n)   state of the best candidate into slot n
  //   int lastNode(int c)                                                   TrackCand::lastCcIndex()
  //   int nodeIndex(int c, int k), nodeLayer(int c, int k), nodePrev(int c, int k)
  //   void loadHit(int c, int k, MPlexHS<N>& msErr, MPlexHV<N>& msPar, MPlexHV<N>& plNrm, MPlexHV<N>& plDir,
  //                MPlexHV<N>& plPnt, int n)                                hit + module of node k into slot n
  //   void storeCand(int c, const MPlexLS<N>&, const MPlexLV<N>&, const MPlexQI<N>&, float chi2, int n)
  //                                                                         copy-out incl. score recomputation
  //   float candPt(int c)                                                   TrackCand::pT() of the INPUT state
  //   void markOutlier(int c, int k)                                        node k -> kHitMissIdx, nFound - 1,
  //                                                                         nMissing + 1
  template <idx_t N, typename TIO>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void bkFitFitTracksProp2Plane(const TIO& io,
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
    chi2.setVal(0);      // MkFitCore: m_Chi2.setVal(0)
    errC.scale(100.0f);  // MkFitCore: m_Err[iC].scale(100.0f)

    // ---- MkFinder::bkFitFitTracksProp2Plane ----
    MPlexQF<N> tmp_chi2(0.0f);
    MPlexQI<N> failFlag(0);
    int done_flag[N];
    for (int i = 0; i < N; ++i)
      done_flag[i] = 0;
    MPlexHV<N> plNrm(0.0f);
    MPlexHV<N> plDir(0.0f);
    MPlexHV<N> plPnt(0.0f);
    MPlexHS<N> msErr(0.0f);
    MPlexHV<N> msPar(0.0f);
    int nOutliers[N];  // hits removed as outliers per slot (backward-fit outlier rejection)
    for (int i = 0; i < N; ++i)
      nOutliers[i] = 0;

    int done_count = 0;
    while (done_count != N_proc) {
      int fitNode[N];  // HoT node fitted in this step, -1 if none
      for (int i = 0; i < N; ++i)
        fitNode[i] = -1;
      int here_count = 0;
      for (int i = 0; i < N_proc; ++i) {
        if (done_flag[i])
          continue;

        // skip invalid hits
        while (curNode[i] >= 0 && io.nodeIndex(cand[i], curNode[i]) < 0) {
          curNode[i] = io.nodePrev(cand[i], curNode[i]);
        }

        if (curNode[i] < 0) {
          // Mark as done and copy out (chi2 = backward-fit chi2 only).
          done_flag[i] = 1;
          ++done_count;
          io.storeCand(cand[i], errC, parC, chg, chi2.At(i, 0, 0), i);
        } else {
          // Same-layer run collapsed to its earliest node (overlap hits skipped).
          const int layer = io.nodeLayer(cand[i], curNode[i]);
          while (io.nodePrev(cand[i], curNode[i]) >= 0 &&
                 io.nodeLayer(cand[i], io.nodePrev(cand[i], curNode[i])) == layer)
            curNode[i] = io.nodePrev(cand[i], curNode[i]);

          io.loadHit(cand[i], curNode[i], msErr, msPar, plNrm, plDir, plPnt, i);

          fitNode[i] = curNode[i];
          ++here_count;

          curNode[i] = io.nodePrev(cand[i], curNode[i]);
        }
      }

      if (done_count == N_proc)
        break;
      if (here_count == 0)
        continue;

      // PROP-FAIL-ENABLE: the propagation fail flag is not checked (MkFitCore).
      failFlag.setVal(0);
      const MPlexQI<N> chgPrev = chg;
      propagateHelixToPlaneMPlex(errC, parC, chg, plPnt, plNrm, errP, parP, failFlag, N_proc, pflags);
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
                                N_proc);
      kalmanCheckChargeFlip(parC, chg, N_proc);

      // outlier rejection (MkFinder.cc bkFitFitTracksProp2Plane): the hit becomes a missing hit and the
      // propagated state is kept. Negated comparison on purpose: a NaN chi2 is not treated as an outlier.
      if (outl.chi2 > 0.f) {
        for (int i = 0; i < N_proc; ++i) {
          if (fitNode[i] < 0 || !(tmp_chi2.At(i, 0, 0) > outl.chi2) || nOutliers[i] >= outl.maxOutliers ||
              io.candPt(cand[i]) < outl.minPt)
            continue;
          errC.copySlot(i, errP);
          parC.copySlot(i, parP);
          chg.At(i, 0, 0) = chgPrev.constAt(i, 0, 0);
          tmp_chi2.At(i, 0, 0) = 0.f;
          io.markOutlier(cand[i], fitNode[i]);
          ++nOutliers[i];
        }
      }

      // update chi2 (MkFinder.cc:2662)
      chi2.add(tmp_chi2);
    }
  }

  // TrackCand copy-out of bkFitFitTracksProp2Plane: chi2, then the score if chi2 is finite
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void bkFitScore(::mkfitdev::CandBook& book, float chi2, float par3) {
    book.chi2 = chi2;
    if (isFinite(chi2)) {
      const float pt = std::abs(1.f / par3);  // TrackBase::pT()
      book.score = ::mkfitdev::getScoreCand(book, pt);
    }
  }

  // ---- production IO: clone-engine SoAs + the engine's hit inputs ----
  // Candidate id = seed index s; the best candidate is slot (s, curBuf, 0) and its HoT chain lives in the seed's
  // pool (compactifyHitStorageForBestCand already applied). Seeds with nCands == 0 or an overflow bit are skipped.
  // Hits/modules: EngineHitInputs (interface/cands/CandEngineTypes.h), read as the engine's K3 does (engineHitRow:
  // LayerOfHits::refHit(index); module = moduleBegin(layer) + detIDinLayer = module_info(detIDinLayer)).
  struct BkFitCandsIO {
    ::mkfitdev::SeedCandsSoA::ConstView seeds;
    ::mkfitdev::CandSlotsSoA::View slots;
    ::mkfitdev::CandHotsSoA::View hots;  // written only by markOutlier
    ::mkfitdev::EngineHitInputs in;

    ALPAKA_FN_HOST_ACC bool active(int s) const { return seeds[s].nCands() > 0 && seeds[s].overflowBits() == 0; }
    ALPAKA_FN_HOST_ACC int row(int s) const { return ::mkfitdev::candSlotRow(s, seeds[s].curBuf(), 0); }
    ALPAKA_FN_HOST_ACC const ::mkfitdev::HoTNode& node(int s, int k) const {
      return hots[::mkfitdev::hotRow(s, k, seeds.hotsPerSeed())].node();
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC void loadCand(int s, MPlexLS<N>& err, MPlexLV<N>& par, MPlexQI<N>& chg, int n) const {
      const ::mkfitdev::CandState& st = slots[row(s)].state();
      err.copyIn(n, st.err);
      par.copyIn(n, st.par);
      chg.At(n, 0, 0) = st.charge;
    }
    ALPAKA_FN_HOST_ACC int lastNode(int s) const { return slots[row(s)].book().lastHitIdx; }
    // TrackCand::pT() of the candidate as stored (the input state: the copy-out happens at the end)
    ALPAKA_FN_HOST_ACC float candPt(int s) const { return std::abs(1.f / slots[row(s)].state().par[3]); }
    ALPAKA_FN_HOST_ACC void markOutlier(int s, int k) const {
      ::mkfitdev::CandHotsSoA::View hv = hots;
      hv[::mkfitdev::hotRow(s, k, seeds.hotsPerSeed())].node().index = ::mkfitdev::kHitMissIdx;
      ::mkfitdev::CandSlotsSoA::View sv = slots;
      ::mkfitdev::CandBook& b = sv[row(s)].book();
      b.nFound -= 1;
      b.nMissing += 1;
    }
    ALPAKA_FN_HOST_ACC int nodeIndex(int s, int k) const { return node(s, k).index; }
    ALPAKA_FN_HOST_ACC int nodeLayer(int s, int k) const { return node(s, k).layer; }
    ALPAKA_FN_HOST_ACC int nodePrev(int s, int k) const { return node(s, k).prev; }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC void loadHit(int s,
                                    int k,
                                    MPlexHS<N>& msErr,
                                    MPlexHV<N>& msPar,
                                    MPlexHV<N>& plNrm,
                                    MPlexHV<N>& plDir,
                                    MPlexHV<N>& plPnt,
                                    int n) const {
      const ::mkfitdev::HoTNode& hn = node(s, k);
      // shared loaders: row = engine hit row, then hit and module plane
      const int r = ::mkfitdev::pack::hitRow(in, hn.layer, hn.index);
      ::mkfitdev::pack::loadHit<N>(in, r, n, msErr, msPar);
      ::mkfitdev::pack::loadModulePlane<N>(in, hn.layer, r, n, plNrm, plDir, plPnt);
    }

    template <idx_t N>
    ALPAKA_FN_HOST_ACC void storeCand(
        int s, const MPlexLS<N>& err, const MPlexLV<N>& par, const MPlexQI<N>& chg, float chi2, int n) const {
      const int r = row(s);
      ::mkfitdev::CandSlotsSoA::View sv = slots;  // the IO object is const, the slots it points to are not
      ::mkfitdev::CandState& st = sv[r].state();
      err.copyOut(n, st.err);
      par.copyOut(n, st.par);
      st.charge = chg.constAt(n, 0, 0);
      bkFitScore(sv[r].book(), chi2, st.par[3]);
    }
  };

  // ---- kernel: one thread per group of kNN candidates ----
  template <typename TIO>
  struct BkFitKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, TIO io, int nCands, PropagationFlags pflags, ::mkfitdev::bkfit::OutlierParams outl) const {
      constexpr int N = kNN;
      const int ngroups = (nCands + N - 1) / N;
      for (int g : cms::alpakatools::uniform_elements(acc, ngroups)) {
        int cand[N];
        int N_proc = 0;
        for (int n = 0; n < N; ++n) {
          const int c = g * N + n;
          if (c < nCands && io.active(c))
            cand[N_proc++] = c;
        }
        if (N_proc == 0)
          continue;
        for (int n = N_proc; n < N; ++n)
          cand[n] = cand[0];
        bkFitFitTracksProp2Plane<N>(io, cand, N_proc, pflags, outl);
      }
    }
  };

  // block size <= 128 for heavy kernels
  constexpr int kBkFitBlockSize = 64;

  template <typename TIO>
  inline void launchBkFit(Queue& queue,
                          TIO const& io,
                          int nCands,
                          PropagationFlags const& pflags,
                          ::mkfitdev::bkfit::OutlierParams const& outl = {}) {
    if (nCands <= 0)
      return;
    const int ngroups = (nCands + kNN - 1) / kNN;
    const int threads = cms::alpakatools::requires_single_thread_per_block_v<Acc1D> ? 1 : kBkFitBlockSize;
    const int blocks = cms::alpakatools::divide_up_by(ngroups, threads);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threads);
    alpaka::exec<Acc1D>(queue, workDiv, BkFitKernel<TIO>{}, io, nCands, pflags, outl);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
