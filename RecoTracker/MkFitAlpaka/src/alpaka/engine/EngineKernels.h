#ifndef RecoTracker_MkFitAlpaka_src_alpaka_engine_EngineKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_engine_EngineKernels_h

// Clone-engine kernels of one plan step .
// MkFitCore reference: MkBuilder::find_tracks_in_layers (MkBuilder.cc:1036-1245),
// MkFinder::findCandidatesCloneEngine (MkFinder.cc:1647-1850), find_tracks_handle_missed_layers (MkBuilder.cc:706),
// MkFinder::updateWithLoadedHit (MkFinder.cc:1874) + copyOutParErr.
//   K1  KernelEngineActivate   thread per seed: unroll (activateSeedCands) with in-kernel kinematics, per-step resets
//   K3a KernelEngineChi2       thread per (seed, cand, hit): kalmanPropagateAndComputeChi2Plane -> CandHitChi2SoA
//   K3b KernelEngineOptions    thread per (seed, cand): dynamic chi2 cut, strip checks, pixel duplicate-layer check,
//                              considerHitForOverlap, options; extras (find_tracks_handle_missed_layers) by slot ic
//   K4  KernelEngineSelect     thread per seed: extras compaction + their -2 stop nodes, then selectSeedCandidates
//   K5  KernelEngineUpdate     thread per (seed, update entry): kalmanPropagateAndUpdatePlane from the parent state
// K2 (propagate to layer + selectHitIndicesV2) is the select; it fills CandSelHitsSoA for the listed cands.
// Matriplex width N = 1 per thread; on CPU backends K3a/K5 walk the seeds and run kNN = 8 rows per Matriplex.

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandOptions.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSelection.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/Packers.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/cands/CandSeedOpsKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using namespace ::mkfitdev;

  constexpr uint32_t kEngineBlock = 128;  // block size of every engine kernel (heavy kernels: <= 128)

  // Sync-free driver: grids are sized from the buffer capacity (nSeeds = capacity), the current number of
  // seed rows is read on the device from nDev.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int engineRows(int cap, const int32_t* nDev) {
    if (nDev == nullptr)
      return cap;
    const int n = *nDev;
    return n < cap ? n : cap;
  }

  // ---------------------------------------------------------------------------------------------------------
  // helpers

  // A seed whose HoT pool overflowed is failed: Finished, no candidates (excluded from export by the filters).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void failSeedOnHotOverflow(Acc1D const& acc,
                                                            SeedCandsSoA::View seeds,
                                                            int s,
                                                            uint32_t ovf0) {
    if (seeds.overflowBits(s) & kOverflowHotsBit) {
      seeds.state(s) = kFinished;
      seeds.nCands(s) = 0;
      seeds.nActive(s) = 0;
      seeds.activeMask(s) = 0;
      seeds.nUpdates(s) = 0;
      seeds.nOverlapUpdates(s) = 0;
      seeds.bestShortValid(s) = 0;
      if (!(ovf0 & kOverflowHotsBit))
        alpaka::atomicAdd(acc, &seeds.nOverflowHots(), 1u, alpaka::hierarchy::Blocks{});
    }
  }

  // Loaders: the shared set in src/alpaka/Packers.h; these names only bind N = 1, slot 0.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int engineHitRow(const EngineHitInputs& in, int layer, int hitIdx) {
    return pack::hitRow(in, layer, hitIdx);
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadCandState(const CandState& cs,
                                                         MPlexLS<1>& err,
                                                         MPlexLV<1>& par,
                                                         MPlexQI<1>& chg) {
    pack::loadCandState<1>(cs, 0, err, par, chg);
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadHit(const EngineHitInputs& in,
                                                   int layer,
                                                   int hitIdx,
                                                   MPlexHS<1>& msErr,
                                                   MPlexHV<1>& msPar,
                                                   MPlexHV<1>& nrm,
                                                   MPlexHV<1>& dir,
                                                   MPlexHV<1>& pnt) {
    const int row = pack::hitRow(in, layer, hitIdx);
    pack::loadHit<1>(in, row, 0, msErr, msPar);
    pack::loadModulePlane<1>(in, layer, row, 0, nrm, dir, pnt);
  }

  // MkFinder::getHitSelDynamicChi2Cut (MkFinder.cc:301-319) on the layer-propagated state (iP).
  // MkFitCore (GCC, x86-64-v3) contracts v[c2_0]*max_invpt + v[c2_1]*theta into an fma; written explicitly.
  // Row i of a width-N Matriplex: K3a/K5 batch kNN rows per thread (8 on CPU backends as NN, 1 on GPU).
  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadCandStateSlot(
      const CandState& cs, MPlexLS<N>& err, MPlexLV<N>& par, MPlexQI<N>& chg, int i) {
    for (int k = 0; k < 6; ++k)
      par.At(i, k, 0) = cs.par[k];
    for (int k = 0; k < 21; ++k)
      err.fArray[k * N + i] = cs.err[k];  // packed lower triangle == CandState.err order
    chg.At(i, 0, 0) = cs.charge;
  }

  template <idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void loadHitSlot(const EngineHitInputs& in,
                                                       int layer,
                                                       int hitIdx,
                                                       MPlexHS<N>& msErr,
                                                       MPlexHV<N>& msPar,
                                                       MPlexHV<N>& nrm,
                                                       MPlexHV<N>& dir,
                                                       MPlexHV<N>& pnt,
                                                       int i) {
    const int row = engineHitRow(in, layer, hitIdx);
    msPar.At(i, 0, 0) = in.x[row];
    msPar.At(i, 1, 0) = in.y[row];
    msPar.At(i, 2, 0) = in.z[row];
    msErr.fArray[0 * N + i] = in.e00[row];
    msErr.fArray[1 * N + i] = in.e10[row];
    msErr.fArray[2 * N + i] = in.e11[row];
    msErr.fArray[3 * N + i] = in.e20[row];
    msErr.fArray[4 * N + i] = in.e21[row];
    msErr.fArray[5 * N + i] = in.e22[row];
    const EngineModule& m = in.modules[in.layers[layer].moduleBegin + int(hitpack::detIDinLayer(in.packed[row]))];
    for (int k = 0; k < 3; ++k) {
      nrm.At(i, k, 0) = m.nrm[k];
      dir.At(i, k, 0) = m.dir[k];
      pnt.At(i, k, 0) = m.pnt[k];
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float engineDynamicChi2Cut(TAcc const& acc,
                                                            const EngineLayerParams& lp,
                                                            const CandPropState& ps,
                                                            float minChi2Cut) {
    const float invpt = ps.par[3];
    const float theta = std::abs(ps.par[5] - Const::PIOver2);
    const float max_invpt = alpaka::math::min(acc, invpt, 10.0f);
    if (lp.hasC2) {
      const float this_c2 = lp.c2[0] * (alpaka::math::fma(acc, lp.c2[1], max_invpt, lp.c2[2] * theta) + lp.c2[3]);
      if (this_c2 > minChi2Cut)
        return this_c2;
    }
    return minChi2Cut;
  }

  // isStripQCompatible (MkFinder.cc:1305-1330): pErr = layer-propagated errors (iP), (px, py, pz) =
  // plane-propagated parameters, msErr/msPar of the hit. Sums of products written as the fma chains GCC forms.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool engineIsStripQCompatible(TAcc const& acc,
                                                               bool isBarrel,
                                                               const float* pErr,  // 21, lower triangle
                                                               float px,
                                                               float py,
                                                               float pz,
                                                               const MPlexHS<1>& msErr,
                                                               const MPlexHV<1>& msPar) {
    if (isBarrel) {
      const float res = std::abs(msPar.constAt(0, 2, 0) - pz);
      const float hitHL = alpaka::math::sqrt(acc, msErr.constAt(0, 2, 2) * 3.f);
      const float qErr = alpaka::math::sqrt(acc, pErr[5]);  // (2,2)
      return hitHL + alpaka::math::max(acc, 3.f * qErr, 0.5f) > res;
    } else {
      const float res0 = msPar.constAt(0, 0, 0) - px;
      const float res1 = msPar.constAt(0, 1, 0) - py;
      const float hitT2 = msErr.constAt(0, 0, 0) + msErr.constAt(0, 1, 1);
      const float hitT2inv = 1.f / hitT2;
      const float proj0 = msErr.constAt(0, 0, 0) * hitT2inv;
      const float proj1 = msErr.constAt(0, 0, 1) * hitT2inv;
      const float proj2 = msErr.constAt(0, 1, 1) * hitT2inv;
      // pErr (0,0) = [0], (1,0) = [1], (1,1) = [2]
      const float q =
          alpaka::math::fma(acc, pErr[2], proj2, alpaka::math::fma(acc, pErr[0], proj0, 2.f * pErr[1] * proj1));
      const float qErr = alpaka::math::sqrt(acc, std::abs(q));
      const float rp = alpaka::math::fma(
          acc, res1 * proj2, res1, alpaka::math::fma(acc, res0 * proj0, res0, 2.f * res1 * proj1 * res0));
      const float resProj = alpaka::math::sqrt(acc, rp);
      return alpaka::math::sqrt(acc, hitT2 * 3.f) + alpaka::math::max(acc, 3.f * qErr, 0.5f) > resProj;
    }
  }

  // ---------------------------------------------------------------------------------------------------------
  // K1: unroll with in-kernel kinematics (TrackCand::pT(), posRsq() = fma(x, x, y*y) as GCC contracts it,
  // posPhi() = vdt::fast_atan2f(y, x), momPhi() = par[4]), plus per-step resets.

  class KernelEngineActivate {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::View seeds,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::View hots,
                                  CandSelHitsSoA::View sel,
                                  const int16_t* stepLayer,      // per region: layer of this step (-1: none)
                                  const int16_t* stepPrevLayer,  // per region: layer of the previous plan step
                                  bool fwdSearch,
                                  float minPtCut,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      const int hps = seeds.hotsPerSeed();
      for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
        const int reg = seeds.region(s);
        const int layer = stepLayer[reg];
        seeds.nExtras(s) = 0;
        seeds.nActive(s) = 0;
        seeds.activeMask(s) = 0;
        seeds.nUpdates(s) = 0;
        seeds.nOverlapUpdates(s) = 0;
        if (layer < 0)
          continue;  // this region's plan has no step here
        seeds.layer(s) = layer;
        const uint32_t ovf0 = seeds.overflowBits(s);
        if (ovf0 & kOverflowHotsBit)
          continue;  // failed seed (HoT pool overflow), already Finished
        SeedCandsRef r = makeSeedRef(seeds, slots, hots, s, hps);
        CandKin kin[kMaxCandsPerSeed];
        // read by the unroll only for a seed that is Finding or picked up at this step
        const bool finding = *r.state == kFinding || (*r.state == kDormant && *r.pickupLayer == stepPrevLayer[reg]);
        const int nKin = finding ? *r.nCands : 0;
        for (int ic = 0; ic < nKin; ++ic) {
          const CandState& cs = r.states[ic];
          const float x = cs.par[0], y = cs.par[1];
          kin[ic].pt = std::abs(1.f / cs.par[3]);
          kin[ic].posRsq = alpaka::math::fma(acc, x, x, y * y);
          kin[ic].posPhi = ::mkfitdev::getPhi(x, y);
          kin[ic].momPhi = cs.par[4];
        }
        int32_t active[kMaxCandsPerSeed];
        int32_t nActive = 0;
        activateSeedCands(r, kin, layer, stepPrevLayer[reg], false, fwdSearch, minPtCut, active, &nActive);
        uint8_t mask = 0;
        for (int k = 0; k < nActive; ++k)
          mask |= uint8_t(1u << active[k]);
        seeds.nActive(s) = nActive;
        seeds.activeMask(s) = mask;
        // rows of the listed candidates only (the others are not read this step); K3b writes their option slots
        // and extras rows
        for (int k = 0; k < nActive; ++k) {
          sel.sel(selRow(s, active[k])).n = 0;
          sel.sel(selRow(s, active[k])).wsr = kWsrUndef;
        }
        failSeedOnHotOverflow(acc, seeds, s, ovf0);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // K3a: chi2 of every (listed cand, selected hit), propagating from the cand's last-hit state to the hit's module
  // plane (usePropToPlane: m_Par/m_Err[iC] re-input at the last hit, MkBuilder.cc:1123).

  class KernelEngineChi2 {
    // nb <= N listed (seed, cand, hit) items in one Matriplex; rows nb..N-1 repeat row 0 (never stored).
    template <idx_t N>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void runBatch(const int (&tb)[N],
                                                        int nb,
                                                        SeedCandsSoA::ConstView seeds,
                                                        CandSlotsSoA::ConstView slots,
                                                        CandSelHitsSoA::ConstView sel,
                                                        CandHitChi2SoA::View c2,
                                                        const EngineHitInputs& in,
                                                        const prop::PropagationFlags& pf,
                                                        bool propToHit) {
      MPlexLS<N> err;
      MPlexLV<N> par;
      MPlexQI<N> chg;
      MPlexHS<N> msErr;
      MPlexHV<N> msPar, nrm, dir, pnt;
      for (int i = 0; i < N; ++i) {
        const int t = tb[i < nb ? i : 0];
        const int ih = t % kMaxHitsPerCand;
        const int ic = (t / kMaxHitsPerCand) % kMaxCandsPerSeed;
        const int s = t / (kMaxHitsPerCand * kMaxCandsPerSeed);
        loadCandStateSlot<N>(slots.state(candSlotRow(s, seeds.curBuf(s), ic)), err, par, chg, i);
        loadHitSlot<N>(in, seeds.layer(s), sel.sel(selRow(s, ic)).hit[ih], msErr, msPar, nrm, dir, pnt, i);
      }
      MPlexQF<N> outChi2;
      MPlexLV<N> propPar;
      MPlexQI<N> fail;
      for (int i = 0; i < N; ++i) {
        outChi2.At(i, 0, 0) = 0.f;
        fail.At(i, 0, 0) = 0;
      }
      kalmanPropagateAndComputeChi2Plane(
          err, par, chg, msErr, msPar, nrm, dir, pnt, outChi2, propPar, fail, nb, pf, propToHit);
      for (int i = 0; i < nb; ++i) {
        const int t = tb[i];
        const int ih = t % kMaxHitsPerCand;
        const int ic = (t / kMaxHitsPerCand) % kMaxCandsPerSeed;
        const int s = t / (kMaxHitsPerCand * kMaxCandsPerSeed);
        CandHitChi2& o = c2.c2(chi2Row(s, ic, ih));
        o.chi2 = outChi2.At(i, 0, 0);
        o.px = propPar.At(i, 0, 0);
        o.py = propPar.At(i, 1, 0);
        o.pz = propPar.At(i, 2, 0);
      }
    }

  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::ConstView seeds,
                                  CandSlotsSoA::ConstView slots,
                                  CandSelHitsSoA::ConstView sel,
                                  CandHitChi2SoA::View c2,
                                  EngineHitInputs in,
                                  prop::PropagationFlags pf,
                                  bool propToHit,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      if constexpr (kNN > 1) {
        // CPU: element = seed (the fixed (seed, cand, hit) grid is mostly inactive slots; testing them one by one
        // was ~45% of this kernel on serial). Blocks past nSeeds have no elements.
        int tb[kNN];
        int nb = 0;
        for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
          const uint32_t mask = seeds.activeMask(s);
          if (mask == 0)
            continue;
          for (int ic = 0; ic < kMaxCandsPerSeed; ++ic) {
            if (!(mask & (1u << ic)))
              continue;
            const int nh = sel.sel(selRow(s, ic)).n;
            for (int ih = 0; ih < nh; ++ih) {
              tb[nb++] = (s * kMaxCandsPerSeed + ic) * kMaxHitsPerCand + ih;
              if (nb == kNN) {
                runBatch<kNN>(tb, nb, seeds, slots, sel, c2, in, pf, propToHit);
                nb = 0;
              }
            }
          }
        }
        if (nb > 0)
          runBatch<kNN>(tb, nb, seeds, slots, sel, c2, in, pf, propToHit);
        return;
      }
      const int n = nSeeds * kMaxCandsPerSeed * kMaxHitsPerCand;
      for (int32_t t : cms::alpakatools::uniform_elements(acc, n)) {
        const int ih = t % kMaxHitsPerCand;
        const int ic = (t / kMaxHitsPerCand) % kMaxCandsPerSeed;
        const int s = t / (kMaxHitsPerCand * kMaxCandsPerSeed);
        if (!(seeds.activeMask(s) & (1u << ic)))
          continue;
        const CandSelHits& sh = sel.sel(selRow(s, ic));
        if (ih >= sh.n)
          continue;
        const int layer = seeds.layer(s);
        MPlexLS<1> err;
        MPlexLV<1> par;
        MPlexQI<1> chg;
        loadCandState(slots.state(candSlotRow(s, seeds.curBuf(s), ic)), err, par, chg);
        MPlexHS<1> msErr;
        MPlexHV<1> msPar, nrm, dir, pnt;
        loadHit(in, layer, sh.hit[ih], msErr, msPar, nrm, dir, pnt);
        MPlexQF<1> outChi2;
        outChi2.At(0, 0, 0) = 0.f;
        MPlexLV<1> propPar;
        MPlexQI<1> fail;
        fail.At(0, 0, 0) = 0;
        kalmanPropagateAndComputeChi2Plane(
            err, par, chg, msErr, msPar, nrm, dir, pnt, outChi2, propPar, fail, 1, pf, propToHit);
        CandHitChi2& o = c2.c2(chi2Row(s, ic, ih));
        o.chi2 = outChi2.At(0, 0, 0);
        o.px = propPar.At(0, 0, 0);
        o.py = propPar.At(0, 1, 0);
        o.pz = propPar.At(0, 2, 0);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // K3b: the sequential per-candidate part of findCandidatesCloneEngine (MkFinder.cc:1730-1850) and
  // find_tracks_handle_missed_layers (MkBuilder.cc:706-750) for one listed candidate.

  class KernelEngineOptions {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::ConstView seeds,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::ConstView hots,
                                  CandSelHitsSoA::ConstView sel,
                                  CandHitChi2SoA::ConstView c2,
                                  CandOptionsSoA::View opts,
                                  CandExtrasSoA::View extras,
                                  EngineHitInputs in,
                                  EngineIterParams ip,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      const int hps = seeds.hotsPerSeed();
      for (int32_t t : cms::alpakatools::uniform_elements(acc, nSeeds * kMaxCandsPerSeed)) {
        const int ic = t % kMaxCandsPerSeed;
        const int s = t / kMaxCandsPerSeed;
        if (!(seeds.activeMask(s) & (1u << ic)))
          continue;
        const int layer = seeds.layer(s);
        const EngineLayerParams lp = in.layers[layer];
        const CandSelHits sh = sel.sel(selRow(s, ic));
        const CandPropState& ps = sel.prop(selRow(s, ic));
        CandBook& book = slots.book(candSlotRow(s, seeds.curBuf(s), ic));

        // find_tracks_handle_missed_layers: Failed -> held back (barrel layers only; -2 stop node in the barrel
        // region, appended by K4, the seed's single HoT writer), treated as Outside below; Outside -> held back.
        extras.extra(extraRow(s, ic)).stateSrc = -1;  // not held back unless set below
        int wsr = sh.wsr;
        if (wsr == kWsrFailed) {
          wsr = kWsrOutside;
          if (lp.isBarrel) {
            CandExtra& x = extras.extra(extraRow(s, ic));
            x.book = book;
            x.stateSrc = ic;
            x.needsStop = seeds.region(s) == kRegBarrel ? 1 : 0;
          }
        } else if (wsr == kWsrOutside) {
          CandExtra& x = extras.extra(extraRow(s, ic));
          x.book = book;
          x.stateSrc = ic;
          x.needsStop = 0;
        }

        const float maxC2 = engineDynamicChi2Cut(acc, lp, ps, ip.chi2CutMin);
        const float pt = optionPt(ps.par[3]);
        const CandBook parent = book;  // MkFinder's copy of the bookkeeping (m_NFoundHits, m_Chi2, ...)
        int nHitsAdded = 0;
        bool tooLargeCluster = false;
        const int nh = sh.n;
        for (int ih = 0; ih < nh; ++ih) {
          CandOption& hitOpt = opts.opt(optRow(s, optSlot(ic, ih)));
          hitOpt.hitIdx = kOptEmpty;  // K4 reads the slots ih < n and the invalid-option slot of a listed cand
          const CandHitChi2 hc = c2.c2(chi2Row(s, ic, ih));
          const float chi2 = std::abs(hc.chi2);
          if (!(chi2 < maxC2))
            continue;
          const int hitIdx = sh.hit[ih];
          const int row = engineHitRow(in, layer, hitIdx);
          const uint32_t packed = in.packed[row];
          const uint32_t module = hitpack::detIDinLayer(packed);
          bool isCompatible = true;
          if (!lp.isPixel) {
            if (int(hitpack::spanRows(packed)) >= ip.maxClusterSize) {
              tooLargeCluster = true;
              isCompatible = false;
            }
            if (isCompatible) {
              MPlexHS<1> msErr;
              MPlexHV<1> msPar, nrm, dir, pnt;
              loadHit(in, layer, hitIdx, msErr, msPar, nrm, dir, pnt);
              isCompatible = engineIsStripQCompatible(acc, lp.isBarrel, ps.err, hc.px, hc.py, hc.pz, msErr, msPar);
            }
            // passStripChargePCMfromTrack not ported: has_charge is false on every Phase-2 layer.
          }
          if (!isCompatible)
            continue;
          if (lp.isPixel) {
            bool hitExists = false;
            const int maxHits = parent.nFound;
            for (int i = 0; i < maxHits && i <= 2; ++i) {
              if (hots.node(hotRow(s, i, hps)).layer == layer) {
                hitExists = true;
                break;
              }
            }
            if (hitExists)
              continue;
          }
          ++nHitsAdded;
          book.overlaps.considerHitForOverlap(hitIdx, int(module), chi2);
          hitOpt = makeHitOption(parent, ic, pt, hitIdx, module, chi2);
        }

        CandOption& invOpt = opts.opt(optRow(s, optSlot(ic, kMaxHitsPerCand)));
        if (wsr != kWsrOutside) {
          const int code = invalidHitCode(
              parent, ip.maxHolesPerCand, ip.maxConsecHoles, wsr, sh.inGap != 0, nHitsAdded, tooLargeCluster);
          invOpt = makeInvalidOption(parent, ic, pt, code);
        } else {
          invOpt.hitIdx = kOptEmpty;
        }
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // K4: per seed, compact the held-back candidates in ic order (extra_cands order), append their -2 stop nodes
  // (find_tracks_handle_missed_layers does this before the clone engine runs), then the selection.

  class KernelEngineSelect {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::View seeds,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::View hots,
                                  CandOptionsSoA::ConstView opts,
                                  CandExtrasSoA::View extras,
                                  CandUpdatesSoA::View upds,
                                  CandSelHitsSoA::ConstView sel,
                                  SeedSelParams params,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      const int hps = seeds.hotsPerSeed();
      for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
        const uint32_t mask = seeds.activeMask(s);
        if (mask == 0) {
          if constexpr (kNN == 1) {
            for (int k = 0; k < kMaxCandsPerSeed; ++k)
              upds.outSrc(updRow(s, k)) = -1;
          }
          continue;  // no listed cand: CandCloner has nothing for this seed (no options, no extras)
        }
        const int cur = seeds.curBuf(s);
        const int nxt = 1 - cur;
        const int nIn = seeds.nCands(s);
        const int layer = seeds.layer(s);

        int32_t nHots = seeds.nHots(s);
        uint32_t ovf = seeds.overflowBits(s);
        const uint32_t ovf0 = ovf;

        // extras: compact slots ic -> e in ic order; stop nodes in the same order
        int nEx = 0;
        for (int ic = 0; ic < kMaxCandsPerSeed; ++ic) {
          if (!(mask & (1u << ic)))
            continue;
          CandExtra x = extras.extra(extraRow(s, ic));
          if (x.stateSrc < 0)
            continue;
          if (x.needsStop) {
            if (nHots < hps) {
              hots.node(hotRow(s, nHots, hps)) = HoTNode{kHitStopIdx, layer, 0.f, x.book.lastHitIdx};
              x.book.lastHitIdx = nHots;
              ++nHots;
            } else {
              ovf |= kOverflowHotsBit;
            }
            x.book.addHitCounters(kHitStopIdx, 0.f);
          }
          extras.extra(extraRow(s, nEx++)) = x;
        }

        // option slots K3b wrote this step: hit slots ih < n and the invalid-option slot of every listed cand
        uint64_t liveOpts = 0;
        for (int ic = 0; ic < kMaxCandsPerSeed; ++ic) {
          if (!(mask & (1u << ic)))
            continue;
          const int nh = sel.sel(selRow(s, ic)).n;
          for (int ih = 0; ih < nh && ih < kMaxHitsPerCand; ++ih)
            liveOpts |= uint64_t(1) << optSlot(ic, ih);
          liveOpts |= uint64_t(1) << optSlot(ic, kMaxHitsPerCand);
        }

        float pt[kMaxCandsPerSeed];
        for (int ic = 0; ic < nIn; ++ic) {
          const float v = 1.f / slots.state(candSlotRow(s, cur, ic)).par[3];  // TrackState::pT()
          pt[ic] = v < 0.f ? -v : v;
        }

        int32_t outSrc[kMaxCandsPerSeed];
        int32_t nOut = 0, nUpd = 0, nOvl = 0, bsSrc = -1;
        int8_t bsValid = seeds.bestShortValid(s);

        SeedSelIO io;
        io.candsIn = &slots.book(candSlotRow(s, cur, 0));
        io.candsInPt = pt;
        io.nCandsIn = nIn;
        io.extras = &extras.extra(extraRow(s, 0));
        io.nExtras = nEx;
        io.opts = &opts.opt(optRow(s, 0));
        io.nOptSlots = kMaxOptsPerSeed;
        io.liveOpts = liveOpts;
        io.state = seeds.state(s);
        io.bestShort = &seeds.bestShort(s);
        io.bestShortValid = &bsValid;
        io.bestShortSrc = &bsSrc;
        io.hots = &hots.node(hotRow(s, 0, hps));
        io.hotOffset = 0;
        io.hotCap = hps;
        io.nHots = &nHots;
        io.candsOut = &slots.book(candSlotRow(s, nxt, 0));
        io.candsOutSrc = outSrc;
        io.nCandsOut = &nOut;
        io.upd = &upds.upd(updRow(s, 0));
        io.nUpd = &nUpd;
        io.ovl = &upds.ovl(updRow(s, 0));
        io.nOvl = &nOvl;
        io.overflowBits = &ovf;

        SeedSelParams p = params;
        p.layer = layer;
        const bool changed = selectSeedCandidates(p, io);

        if (bsSrc >= 0)
          slots.state(bestShortRow(s)) = slots.state(candSlotRow(s, cur, bsSrc));
        if constexpr (kNN == 1) {
          // GPU: K5's (seed, k) threads copy the parent states (one ~112 B state each)
          for (int k = 0; k < kMaxCandsPerSeed; ++k)
            upds.outSrc(updRow(s, k)) = (changed && k < nOut) ? outSrc[k] : -1;
        } else {
          if (changed) {
            for (int k = 0; k < nOut; ++k)
              slots.state(candSlotRow(s, nxt, k)) = slots.state(candSlotRow(s, cur, outSrc[k]));
          }
        }
        if (changed) {
          seeds.nCands(s) = nOut;
          seeds.curBuf(s) = nxt;
        }
        seeds.bestShortValid(s) = bsValid;
        seeds.nHots(s) = nHots;
        seeds.nExtras(s) = 0;
        seeds.nUpdates(s) = nUpd;
        seeds.nOverlapUpdates(s) = nOvl;
        seeds.overflowBits(s) = ovf;
        failSeedOnHotOverflow(acc, seeds, s, ovf0);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // K5: Kalman update of every updated candidate with its new hit, from the state copied from its parent
  // (MkFinder::updateWithLoadedHit: kalmanPropagateAndUpdatePlane with the inter-layer flags, then
  // copyOutParErr: par, err, charge). The overlap re-check list is not processed (recheckOverlap is false in the
  // LST step; a non-empty list is counted as an error by the driver).

  class KernelEngineUpdate {
    // nb <= N update entries (distinct candidate slots) in one Matriplex; rows nb..N-1 repeat row 0 (never stored).
    template <idx_t N>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void runBatch(const int (&tb)[N],
                                                        int nb,
                                                        SeedCandsSoA::ConstView seeds,
                                                        CandSlotsSoA::View slots,
                                                        CandUpdatesSoA::ConstView upds,
                                                        const EngineHitInputs& in,
                                                        const prop::PropagationFlags& pf,
                                                        bool propToHit) {
      MPlexLS<N> err, outErr;
      MPlexLV<N> par, outPar;
      MPlexQI<N> chg, fail;
      MPlexHS<N> msErr;
      MPlexHV<N> msPar, nrm, dir, pnt;
      int row[N];
      for (int i = 0; i < N; ++i) {
        const int t = tb[i < nb ? i : 0];
        const int u = t % kMaxCandsPerSeed;
        const int s = t / kMaxCandsPerSeed;
        const CandUpdate up = upds.upd(updRow(s, u));
        row[i] = candSlotRow(s, seeds.curBuf(s), up.cand_idx);
        loadCandStateSlot<N>(slots.state(row[i]), err, par, chg, i);
        loadHitSlot<N>(in, seeds.layer(s), up.hit_idx, msErr, msPar, nrm, dir, pnt, i);
        fail.At(i, 0, 0) = 0;
      }
      kalmanPropagateAndUpdatePlane(
          err, par, chg, msErr, msPar, nrm, dir, pnt, outErr, outPar, fail, nb, pf, propToHit);
      for (int i = 0; i < nb; ++i) {
        CandState& cs = slots.state(row[i]);
        for (int k = 0; k < 6; ++k)
          cs.par[k] = outPar.At(i, k, 0);
        for (int k = 0; k < 21; ++k)
          cs.err[k] = outErr.fArray[k * N + i];
        cs.charge = chg.At(i, 0, 0);
      }
    }

  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::ConstView seeds,
                                  CandSlotsSoA::View slots,
                                  CandUpdatesSoA::ConstView upds,
                                  EngineHitInputs in,
                                  prop::PropagationFlags pf,
                                  bool propToHit,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      if constexpr (kNN > 1) {
        // CPU: element = seed, one nUpdates read per seed (see K3a)
        int tb[kNN];
        int nb = 0;
        for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
          const int nu0 = seeds.nUpdates(s);
          const int nu = nu0 < kMaxCandsPerSeed ? nu0 : kMaxCandsPerSeed;
          for (int u = 0; u < nu; ++u) {
            tb[nb++] = s * kMaxCandsPerSeed + u;
            if (nb == kNN) {
              runBatch<kNN>(tb, nb, seeds, slots, upds, in, pf, propToHit);
              nb = 0;
            }
          }
        }
        if (nb > 0)
          runBatch<kNN>(tb, nb, seeds, slots, upds, in, pf, propToHit);
        return;
      }
      for (int32_t t : cms::alpakatools::uniform_elements(acc, nSeeds * kMaxCandsPerSeed)) {
        const int u = t % kMaxCandsPerSeed;
        const int s = t / kMaxCandsPerSeed;
        const int nu = seeds.nUpdates(s);
        // K4 copy split: when K4 changed the seed's candidates, thread (s, k = u) owns output slot k: it
        // reads the parent state from the previous buffer and writes slot k, updated if an update targets it, else
        // copied. Same operands as K4's copy followed by the in-place update.
        const int src = upds.outSrc(updRow(s, u));
        const bool split = upds.outSrc(updRow(s, 0)) >= 0;
        int ui = u;
        const CandState* srcState = nullptr;
        if (split) {
          if (src < 0)
            continue;
          const int curNow = seeds.curBuf(s);
          srcState = &slots.state(candSlotRow(s, 1 - curNow, src));
          ui = -1;
          for (int v = 0; v < nu && v < kMaxCandsPerSeed; ++v)
            if (upds.upd(updRow(s, v)).cand_idx == u)
              ui = v;
          if (ui < 0) {
            slots.state(candSlotRow(s, curNow, u)) = *srcState;
            continue;
          }
        } else if (u >= nu) {
          continue;
        }
        const CandUpdate up = upds.upd(updRow(s, ui));
        CandState& cs = slots.state(candSlotRow(s, seeds.curBuf(s), up.cand_idx));
        if (srcState == nullptr) {
          srcState = &cs;
        } else {
          cs.label = srcState->label;  // the fields the update does not write
          cs.status = srcState->status;
          cs.nSeedHits = srcState->nSeedHits;
        }
        MPlexLS<1> err, outErr;
        MPlexLV<1> par, outPar;
        MPlexQI<1> chg, fail;
        loadCandState(*srcState, err, par, chg);
        MPlexHS<1> msErr;
        MPlexHV<1> msPar, nrm, dir, pnt;
        loadHit(in, seeds.layer(s), up.hit_idx, msErr, msPar, nrm, dir, pnt);
        fail.At(0, 0, 0) = 0;
        kalmanPropagateAndUpdatePlane(
            err, par, chg, msErr, msPar, nrm, dir, pnt, outErr, outPar, fail, 1, pf, propToHit);
        for (int i = 0; i < 6; ++i)
          cs.par[i] = outPar.At(0, i, 0);
        for (int i = 0; i < 21; ++i)
          cs.err[i] = outErr.fArray[i];
        cs.charge = chg.At(0, 0, 0);
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // Stable survivor compaction (filter_comb_cands drops failing seeds keeping the order): exclusive prefix scan of
  // the pass flags in ONE block (chunks of 1024 with a carry), then a gather of the surviving seed rows.

  class KernelEngineScanPass {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, const int8_t* passed, int32_t* newIdx, int32_t* nOut, int n, const int32_t* nDev) const {
      n = engineRows(n, nDev);
      constexpr int kChunk = 1024;
      auto& buf = alpaka::declareSharedVar<int32_t[kChunk], __COUNTER__>(acc);
      auto& ws = alpaka::declareSharedVar<int32_t[32], __COUNTER__>(acc);
      auto& carry = alpaka::declareSharedVar<int32_t, __COUNTER__>(acc);
      const int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const int bdim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      if (tid == 0)
        carry = 0;
      alpaka::syncBlockThreads(acc);
      for (int off = 0; off < n; off += kChunk) {
        const int len = (n - off) < kChunk ? (n - off) : kChunk;
        for (int i = tid; i < len; i += bdim)
          buf[i] = passed[off + i] ? 1 : 0;
        alpaka::syncBlockThreads(acc);
        cms::alpakatools::blockPrefixScan(acc, buf, len, ws);  // inclusive
        alpaka::syncBlockThreads(acc);
        const int c = carry;
        for (int i = tid; i < len; i += bdim)
          newIdx[off + i] = passed[off + i] ? c + buf[i] - 1 : -1;
        alpaka::syncBlockThreads(acc);
        if (tid == 0)
          carry = c + buf[len - 1];
        alpaka::syncBlockThreads(acc);
      }
      if (tid == 0)
        *nOut = carry;
    }
  };

  class KernelEngineGatherSeeds {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::ConstView seedsIn,
                                  CandSlotsSoA::ConstView slotsIn,
                                  CandHotsSoA::ConstView hotsIn,
                                  SeedCandsSoA::View seedsOut,
                                  CandSlotsSoA::View slotsOut,
                                  CandHotsSoA::View hotsOut,
                                  const int32_t* newIdx,
                                  const int32_t* nSurvivors,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      const int cap = nSeeds;
      const int nIn = engineRows(nSeeds, nDev);
      const int nSurv = *nSurvivors;
      const int hps = seedsIn.hotsPerSeed();
      if (cms::alpakatools::once_per_grid(acc)) {
        seedsOut.hotsPerSeed() = hps;
        seedsOut.nOverflowHots() = seedsIn.nOverflowHots();
        seedsOut.nOverflowOpts() = seedsIn.nOverflowOpts();
        seedsOut.nOverflowExtras() = seedsIn.nOverflowExtras();
        seedsOut.nRepackRepeat() = seedsIn.nRepackRepeat();
      }
      for (int32_t s : cms::alpakatools::uniform_elements(acc, cap)) {
        if (s >= nSurv) {
          // destination rows past the survivors hold no seed (capacity-wide kernels must skip them)
          seedsOut.nCands(s) = 0;
          seedsOut.nActive(s) = 0;
          seedsOut.activeMask(s) = 0;
          seedsOut.nUpdates(s) = 0;
          seedsOut.nOverlapUpdates(s) = 0;
          seedsOut.nExtras(s) = 0;
          seedsOut.state(s) = kFinished;
          seedsOut.bestShortValid(s) = 0;
          seedsOut.bkwRepacked(s) = 0;
          seedsOut.overflowBits(s) = 0;
        }
        if (s >= nIn)
          continue;
        const int d = newIdx[s];
        if (d < 0)
          continue;
        seedsOut.nCands(d) = seedsIn.nCands(s);
        seedsOut.curBuf(d) = seedsIn.curBuf(s);
        seedsOut.state(d) = seedsIn.state(s);
        seedsOut.pickupLayer(d) = seedsIn.pickupLayer(s);
        seedsOut.layer(d) = seedsIn.layer(s);
        seedsOut.region(d) = seedsIn.region(s);
        seedsOut.activeMask(d) = seedsIn.activeMask(s);
        seedsOut.nActive(d) = seedsIn.nActive(s);
        seedsOut.seedOriginIdx(d) = seedsIn.seedOriginIdx(s);
        seedsOut.nHots(d) = seedsIn.nHots(s);
        seedsOut.nExtras(d) = seedsIn.nExtras(s);
        seedsOut.nUpdates(d) = seedsIn.nUpdates(s);
        seedsOut.nOverlapUpdates(d) = seedsIn.nOverlapUpdates(s);
        seedsOut.bestShort(d) = seedsIn.bestShort(s);
        seedsOut.bestShortValid(d) = seedsIn.bestShortValid(s);
        seedsOut.lastHitIdxBeforeBkw(d) = seedsIn.lastHitIdxBeforeBkw(s);
        seedsOut.nInsideMinusOneBeforeBkw(d) = seedsIn.nInsideMinusOneBeforeBkw(s);
        seedsOut.nTailMinusOneBeforeBkw(d) = seedsIn.nTailMinusOneBeforeBkw(s);
        seedsOut.overflowBits(d) = seedsIn.overflowBits(s);
        seedsOut.bkwRepacked(d) = seedsIn.bkwRepacked(s);
        // live slots only: the current buffer's candidates and a valid best-short state (the other rows are written
        // before they are read)
        const int cur = seedsIn.curBuf(s);
        const int nc = seedsIn.nCands(s);
        for (int k = 0; k < nc && k < kMaxCandsPerSeed; ++k) {
          slotsOut.book(candSlotRow(d, cur, k)) = slotsIn.book(candSlotRow(s, cur, k));
          slotsOut.state(candSlotRow(d, cur, k)) = slotsIn.state(candSlotRow(s, cur, k));
        }
        if (seedsIn.bestShortValid(s))
          slotsOut.state(bestShortRow(d)) = slotsIn.state(bestShortRow(s));
        const int nh = seedsIn.nHots(s);
        for (int k = 0; k < nh; ++k)
          hotsOut.node(hotRow(d, k, hps)) = hotsIn.node(hotRow(s, k, hps));
      }
    }
  };

  // filter_comb_cands per seed + pass flag (seeds with nCands == 0, e.g. HoT-overflow-failed ones, fail).
  class KernelEngineFilter {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::View seeds,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::View hots,
                                  int8_t* passed,
                                  bool bkwRep,
                                  int minHitsQF,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      const int hps = seeds.hotsPerSeed();
      for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
        if (seeds.nCands(s) <= 0 || (seeds.overflowBits(s) & kOverflowHotsBit)) {
          passed[s] = 0;
          continue;
        }
        SeedCandsRef r = makeSeedRef(seeds, slots, hots, s, hps);
        bool repack = bkwRep;
        if (bkwRep) {
          // repackCandPostBkwSearch must run once per seed (TrackStructures.cc:231); a second post-filter
          // on the same rows only filters, and is counted (status product: must stay 0)
          if (seeds.bkwRepacked(s)) {
            repack = false;
            alpaka::atomicAdd(acc, &seeds.nRepackRepeat(), 1u, alpaka::hierarchy::Blocks{});
          } else {
            seeds.bkwRepacked(s) = 1;
          }
        }
        passed[s] = filterSeedCands(r, repack, true, minHitsQF) ? 1 : 0;
      }
    }
  };

  // endBkwSearch: MkFitCore only clears EventOfCombCandidates::m_cands_in_backward_rep (a host-side flag here).

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
