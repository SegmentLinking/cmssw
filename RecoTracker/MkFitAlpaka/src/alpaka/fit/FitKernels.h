#ifndef RecoTracker_MkFitAlpaka_src_alpaka_fit_FitKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_fit_FitKernels_h

// mkFit final fit (MkFitFitProducer path, procModifier trackingMkFitFit) on the device, ported from MkFitCore.
// MkFitCore: MkBuilder::fittracks / fit_tracks, MkFitter fwdFitInputTracks, bkReFitInputTracks, reFitOutputTracks,
// reFitIndices, fwdFitFitTracks, bkReFitFitTracks; the propagation and update of each step are
// final_fit::propagate_update (FinalFit.cc), here kalmanPropagateAndUpdateAndChi2Plane with the final-fit flags.
// GPU: one thread per track (N = 1 Matriplex slot), the fit SPLIT into stage kernels: reFitIndices, forward pass,
// backward pass, then per outlier round: outlier removal (+ reFitIndices of the refit), forward and backward refit.
// The state between them is the track row itself (TrackSoA params/errors/chi2/outer) plus the per-track scratch
// below; every track runs exactly the operations of a single-kernel fit in the same order.
// CPU: tracks grouped by their number of hits with index >= 0 into kNN-wide Matriplex groups as MkFitCore (one kernel).
// CPE: kCpe = true applies the device PixelCPEGeneric with the track angles on every pixel hit through the measurement
// CPE hook (do_cpe / cpe_corr_func, plane::measurement_cpe); kCpe = false is the no-CPE variant.

#include <cstdint>
#include <limits>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeGeneric.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/math/HitConstants.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/PropagationFlagsAdapter.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/fit/FitTracks.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::fit {

  using ::mkfitdev::HitSoAConstView;
  using ::mkfitdev::TrackSoAView;

  // Per-thread scratch in global memory, element (h, t) at h * stride + t (coalesced across tracks).
  struct FitScratch {
    float* key;      // reFitIndices sort key, then the outlier scorer
    float* chi2f;    // per fwd step chi2 (MkFitCore chi2fwd)
    float* chi2b;    // per bkw step chi2 (MkFitCore chi2bkwd)
    int16_t* order;  // reFitIndices output: positions in the hit list, ascending key
    int32_t* meta;   // split kernels: [0, stride) n ordered, [stride, 2 stride) nFH, [2 stride, 3 stride) pass-2 flag
    int stride;
  };

  struct FitHitAccess {
    ESView es;
    HitSoAConstView hits;
    uint32_t nPixel;
    ::mkfitdev::cpe::CpeTables cpe;               // device CPE tables (used when kCpe)
    const ::mkfitdev::cpe::ClusterCpe* clusters;  // [nPixel] per pixel hit (used when kCpe)

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t row(int layer, int index) const {
      return (es.layers[layer].is_pixel() ? 0u : nPixel) + static_cast<uint32_t>(index);
    }
  };

  // MkFitter::reFitIndices for one track. Hits with index >= 0, visited from the last position down; key
  // -r2 - (z - z_minR)^2 (r2 = x^2 + y^2, z_minR = z of the first hit with the smallest r2 in that visit order).
  // MkFitCore keys a std::map<float, vector<int>>: ascending key, equal keys in insertion order. Reproduced by a stable
  // fixed-size insertion.
  // Returns the number of hits ordered (indices.size()).
  // MkFitCore is built -Ofast for x86-64-v3: GCC contracts R2 = x*x + y*y into fma(x, x, y*y) and
  // -r2 - z*z into fma(-z, z, -r2). The key decides the fit order (and near-equal keys of overlap hits swap),
  // so the contracted forms are written explicitly for every backend.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int reFitIndices(
      TAcc const& acc, FitHitAccess const& ha, TrackSoAView trk, int t, FitScratch const& sc, int& nOverflow) {
    const int nTot = trk[t].nTotalHits();
    const auto& hot = trk[t].hits().hot;
    float minR2 = std::numeric_limits<float>::max();
    float zMinR = 0.f;
    for (int m = nTot - 1; m >= 0; --m) {
      if (hot[m].index < 0)
        continue;
      const uint32_t r = ha.row(hot[m].layer, hot[m].index);
      const float x = ha.hits[r].x(), y = ha.hits[r].y(), z = ha.hits[r].z();
      const float R2 = alpaka::math::fma(acc, x, x, y * y);
      if (R2 < minR2) {
        minR2 = R2;
        zMinR = z;
      }
    }
    int n = 0;
    for (int m = nTot - 1; m >= 0; --m) {
      if (hot[m].index < 0)
        continue;
      if (n >= ::mkfitdev::kMaxTrkHits) {
        ++nOverflow;
        break;
      }
      const uint32_t r = ha.row(hot[m].layer, hot[m].index);
      const float x = ha.hits[r].x(), y = ha.hits[r].y();
      const float r2 = alpaka::math::fma(acc, x, x, y * y);  // MkFitCore: sign * R2, then its absolute value
      const float z = ha.hits[r].z() - zMinR;
      const float k = alpaka::math::fma(acc, -z, z, -r2);
      int p = n;
      while (p > 0 && sc.key[(p - 1) * sc.stride + t] > k) {
        sc.key[p * sc.stride + t] = sc.key[(p - 1) * sc.stride + t];
        sc.order[p * sc.stride + t] = sc.order[(p - 1) * sc.stride + t];
        --p;
      }
      sc.key[p * sc.stride + t] = k;
      sc.order[p * sc.stride + t] = static_cast<int16_t>(m);
      ++n;
    }
    return n;
  }

  // The CPE functor of kalmanOperationPlaneLocal: slot n with a pixel hit (do_cpe >= 0) gets the device
  // PixelCPEGeneric (x, y, exx, exy, eyy) for the local track angles.
  template <idx_t N>
  struct FitCpe {
    static constexpr bool enabled = true;
    ::mkfitdev::cpe::CpeTables tables;
    const ::mkfitdev::cpe::ClusterCpe* clusters;
    int hitRow[N];
    ALPAKA_FN_HOST_ACC bool operator()(int n, const float (&ltp)[6], float (&lh)[5]) const {
      if (hitRow[n] < 0)
        return false;
      return ::mkfitdev::cpe::cpeTrackAngles(tables, clusters[hitRow[n]], ltp[1], ltp[2], lh);
    }
  };

  template <bool kCpe, idx_t N>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto const& pickCpe(FitCpe<N> const& c, NoCpe const& z) {
    if constexpr (kCpe)
      return c;
    else
      return z;
  }

  // One fit direction over nFH hits: fwd (MkFitter::fwdFitFitTracks) visits order[n-1-h] (innermost first),
  // bkw (bkReFitFitTracks, reversed copy) visits order[h] (outermost first). Every step, the first one included,
  // propagates to the hit's plane (the start state need not lie on it); material effects are skipped on the first hit
  // of each direction (MkFitCore: index == indices.back()). The input state comes from the track row with its errors
  // scaled by 100 and chi2 = 0 (fwdFitInputTracks / bkReFitInputTracks); the charge is the track's and MkFitCore never
  // writes a flipped charge back (reFitOutputTracks copies errors, parameters and chi2 only).
  // reFitOutputTracks: when the chi2 sum is not finite the track keeps its previous state; after the backward pass,
  // which gives the final state, that is a failed fit and the chi2 is set to -1 (the converter drops the candidate).
  // N slots (tracks tl[0..nProc), all with the same nFH, as fit_tracks groups them; slots >= nProc repeat
  // slot 0 and are discarded). N = 1 on GPU and for the refits after outlier removal, N = kNN for the first fit on CPU
  // backends.
  //
  // Driven by the ES (ESConfig::refit = TrackerInfo::prop_config().final_fit_ffflags, MkFitGeometryESProducer refit*
  // parameters), as final_fit::Pass:
  //   - pf (refitKernelFlags) carries final_fit_pflags, b_field_at_mid / radial_field_corr and final_fit; eloss_by_pass
  //     / eloss_outward are set here per pass (eloss_sign_from_pass: forward loses, backward gains);
  //   - per-module material (material_per_module) from the ES module table (ModuleInfo::radl/bbxi);
  //   - backward pass: multiple scattering at the |p| of its start state (bkw_ms_fixed_momentum, ms_ref_p)
  //     and bkw_sub_steps sub-steps (propagateHelixToPlaneSubStepMPlex) where the state is off the plane.
  // kBkw is a template parameter so that the forward kernels do not carry the sub-step code (registers).
  // The state at the outermost hit (MkFitter::storeHitStates with m_storeOuter, TrackSoA outer): the forward updated
  // local state there, written by the forward pass; where it is not usable, the backward pass puts its updated state
  // at that hit in its place. Valid: finite parameters, finite and positive variances.
  template <idx_t N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool fillLocalState(
      float (&par)[5], float (&err)[15], const MPlex5V<N>& lp, const MPlex5S<N>& le, const int n) {
    bool fin = true;
    for (int i = 0; i < 5; ++i) {
      par[i] = lp.constAt(n, i, 0);
      for (int j = 0; j <= i; ++j)
        err[i * (i + 1) / 2 + j] = le.constAt(n, i, j);
      const float d = err[i * (i + 3) / 2];
      fin = fin && ::mkfitdev::isFinite(par[i]) && ::mkfitdev::isFinite(d) && d > 0.f;
    }
    return fin;
  }

  template <idx_t N, bool kCpe, bool kBkw>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fitDirection(FitHitAccess const& ha,
                                                   PropagationFlags const& pfIn,
                                                   TrackSoAView trk,
                                                   const int (&tl)[N],
                                                   const int nProc,
                                                   FitScratch const& sc,
                                                   const int (&nl)[N],
                                                   int nFH,
                                                   float* chi2Out) {
    constexpr bool bkw = kBkw;
    MPlexLS<N> errA, errB;
    MPlexLV<N> parA, parB;
    MPlexQI<N> chg, failFlag, noMat;
    MPlexQF<N> outChi2;
    MPlexHS<N> msErr;
    MPlexHV<N> msPar, norm, dir, pnt;
    MPlexQF<N> matRadl(0.0f), matBbxi(0.0f), msRefP(1.0f);
    float chi2[N];

    const ::mkfitdev::RefitConfig rc = ha.es.config->refit;
    PropagationFlags pf = pfIn;
    if (rc.elossSignFromPass) {
      pf.eloss_by_pass = true;
      pf.eloss_outward = !bkw;
    }
    const MPlexQF<N>* matRadlPtr = rc.materialPerModule ? &matRadl : nullptr;
    const MPlexQF<N>* matBbxiPtr = rc.materialPerModule ? &matBbxi : nullptr;
    const MPlexQF<N>* msRefPtr = (bkw && rc.bkwMsFixedMomentum) ? &msRefP : nullptr;

    for (int s = 0; s < N; ++s) {
      const int t = tl[s < nProc ? s : 0];
      for (int k = 0; k < 6; ++k)
        parA.At(s, k, 0) = trk[t].params().v[k];
      errA.copyIn(s, trk[t].errors().v);
      chg.At(s, 0, 0) = trk[t].charge();
      failFlag.At(s, 0, 0) = 0;
      chi2[s] = 0.f;
    }
    errA.scale(100.0f);
    if constexpr (bkw) {
      // bkReFitFitTracks ms_ref_p: |p| of the backward pass's start state (the forward result), per track
      if (rc.bkwMsFixedMomentum) {
        for (int s = 0; s < N; ++s) {
          const float ipt = parA.constAt(s, 3, 0), sT = std::sin(parA.constAt(s, 5, 0));
          msRefP.At(s, 0, 0) = (s < nProc && ipt > 0.f && sT > 0.f) ? 1.f / (ipt * sT) : 1.f;
        }
      }
    }

    using CpeT = std::conditional_t<kCpe, FitCpe<N>, NoCpe>;
    [[maybe_unused]] FitCpe<N> cpe{ha.cpe, ha.clusters, {}};
    [[maybe_unused]] const NoCpe noCpe{};
    CpeT const& cpeObj = pickCpe<kCpe>(cpe, noCpe);
    if constexpr (!bkw) {
      // fit_tracks: the track's OuterHitState is reset for every fit
      for (int s = 0; s < nProc; ++s)
        trk[tl[s]].outer() = ::mkfitdev::TrackOuterState{{}, {}, {}, {}, {}, 0.f, -1, 0, 0, 0};
    }
    LocalStatesOut<N> loc;
    MPlex5V<N> lPar;
    MPlex5S<N> lErr;
    MPlexQI<N> lPz;
    for (int h = 0; h < nFH; ++h) {
      const bool innermost = bkw && h == nFH - 1;
      const bool outermost = bkw ? h == 0 : h == nFH - 1;
      // the updated local state at the outermost hit (MkFitter keepLoc / keepBkLoc)
      loc = LocalStatesOut<N>{};
      const LocalStatesOut<N>* locPtr = nullptr;
      if (outermost) {
        loc.updPar = &lPar;
        loc.updErr = &lErr;
        loc.pzSign = &lPz;
        locPtr = &loc;
      }
      for (int s = 0; s < N; ++s) {
        const int sl = s < nProc ? s : 0;
        const int t = tl[sl], n = nl[sl];
        const auto& hot = trk[t].hits().hot;
        const int pos = bkw ? h : n - 1 - h;
        const int m = sc.order[pos * sc.stride + t];
        const int layer = hot[m].layer;
        const uint32_t r = ha.row(layer, hot[m].index);
        // = pack::loadHit + pack::loadModulePlane (src/alpaka/Packers.h, HitSoA form; same stores). Kept inline: routing
        // it through Packers changes GCC's N = 8 code generation of this kernel at rounding level.
        msPar.At(s, 0, 0) = ha.hits[r].x();
        msPar.At(s, 1, 0) = ha.hits[r].y();
        msPar.At(s, 2, 0) = ha.hits[r].z();
        msErr.At(s, 0, 0) = ha.hits[r].e00();
        msErr.At(s, 1, 0) = ha.hits[r].e10();
        msErr.At(s, 1, 1) = ha.hits[r].e11();
        msErr.At(s, 2, 0) = ha.hits[r].e20();
        msErr.At(s, 2, 1) = ha.hits[r].e21();
        msErr.At(s, 2, 2) = ha.hits[r].e22();
        const int mod = ha.es.moduleRow(layer, ::mkfitdev::hitpack::detIDinLayer(ha.hits[r].packed()));
        const auto mi = ha.es.modules[mod];
        norm.At(s, 0, 0) = mi.zdir_x();
        norm.At(s, 1, 0) = mi.zdir_y();
        norm.At(s, 2, 0) = mi.zdir_z();
        dir.At(s, 0, 0) = mi.xdir_x();
        dir.At(s, 1, 0) = mi.xdir_y();
        dir.At(s, 2, 0) = mi.xdir_z();
        pnt.At(s, 0, 0) = mi.pos_x();
        pnt.At(s, 1, 0) = mi.pos_y();
        pnt.At(s, 2, 0) = mi.pos_z();
        matRadl.At(s, 0, 0) = mi.radl();
        matBbxi.At(s, 0, 0) = mi.bbxi();
        noMat.At(s, 0, 0) = (pos == (bkw ? 0 : n - 1)) ? 1 : 0;
        if constexpr (kCpe)
          cpe.hitRow[s] = (s < nProc && ha.es.layers[layer].is_pixel()) ? int(r) : -1;  // do_cpe
      }

      bool subProp = false;
      if constexpr (bkw) {
        // final_fit::propagate_update with pass.n_sub = bkw_sub_steps > 1: the lanes whose state is off the plane
        // are propagated in sub-steps
        bool splitLane[N];
        for (int s = 0; s < N; ++s)
          splitLane[s] = false;
        if (rc.bkwSubSteps > 1) {
          for (int s = 0; s < nProc; ++s) {
            const float d = (pnt.constAt(s, 0, 0) - parA.constAt(s, 0, 0)) * norm.constAt(s, 0, 0) +
                            (pnt.constAt(s, 1, 0) - parA.constAt(s, 1, 0)) * norm.constAt(s, 1, 0) +
                            (pnt.constAt(s, 2, 0) - parA.constAt(s, 2, 0)) * norm.constAt(s, 2, 0);
            if (d != 0.f) {
              splitLane[s] = true;
              subProp = true;
            }
          }
        }
        if (subProp) {
          MPlexLS<N> propErr;
          MPlexLV<N> propPar;
          propagateHelixToPlaneSubStepMPlex(errA,
                                            parA,
                                            chg,
                                            pnt,
                                            norm,
                                            propErr,
                                            propPar,
                                            failFlag,
                                            nProc,
                                            pf,
                                            rc.bkwSubSteps,
                                            splitLane,
                                            &noMat,
                                            matRadlPtr,
                                            matBbxiPtr,
                                            msRefPtr);
          kalmanPropagateAndUpdateAndChi2Plane<N, CpeT>(propErr,
                                                        propPar,
                                                        chg,
                                                        msErr,
                                                        msPar,
                                                        norm,
                                                        dir,
                                                        pnt,
                                                        errB,
                                                        parB,
                                                        failFlag,
                                                        outChi2,
                                                        nProc,
                                                        pf,
                                                        false,  // already propagated
                                                        &noMat,
                                                        cpeObj,
                                                        matRadlPtr,
                                                        matBbxiPtr,
                                                        msRefPtr,
                                                        locPtr);
        }
      }
      if (!subProp)
        kalmanPropagateAndUpdateAndChi2Plane<N, CpeT>(errA,
                                                      parA,
                                                      chg,
                                                      msErr,
                                                      msPar,
                                                      norm,
                                                      dir,
                                                      pnt,
                                                      errB,
                                                      parB,
                                                      failFlag,
                                                      outChi2,
                                                      nProc,
                                                      pf,
                                                      true,  // every step propagates to its hit's plane
                                                      &noMat,
                                                      cpeObj,
                                                      matRadlPtr,
                                                      matBbxiPtr,
                                                      msRefPtr,
                                                      locPtr);
      if constexpr (!bkw) {
        if (outermost) {
          // the outer state: the forward updated state at the outermost hit, with its chi2 increment
          for (int s = 0; s < nProc; ++s) {
            const int t = tl[s];
            auto& o = trk[t].outer();
            o.valid = fillLocalState(o.par, o.err, lPar, lErr, s);
            o.pzSign = int8_t(lPz.constAt(s, 0, 0));
            o.kind = 1;
            o.chi2 = outChi2.At(s, 0, 0);
            o.hot = int16_t(sc.order[(nl[s] - 1 - h) * sc.stride + t]);
            for (int i = 0; i < 3; ++i) {
              o.pos[i] = pnt.constAt(s, i, 0);
              o.xdir[i] = dir.constAt(s, i, 0);
              o.zdir[i] = norm.constAt(s, i, 0);
            }
          }
        }
      } else if (outermost) {
        // the outer state, where the forward one is not usable: the backward updated state at that hit
        for (int s = 0; s < nProc; ++s) {
          auto& o = trk[tl[s]].outer();
          if (!o.valid && !innermost) {
            o.valid = fillLocalState(o.par, o.err, lPar, lErr, s);
            o.pzSign = int8_t(lPz.constAt(s, 0, 0));
            o.kind = 2;
          }
        }
      }
      errA = errB;
      parA = parB;
      for (int s = 0; s < nProc; ++s) {
        chi2[s] += outChi2.At(s, 0, 0);  // m_Chi2.add(outChi2)
        chi2Out[h * sc.stride + tl[s]] = outChi2.At(s, 0, 0);
      }
    }

    for (int s = 0; s < nProc; ++s) {
      const int t = tl[s];
      if (!::mkfitdev::isFinite(chi2[s])) {  // reFitOutputTracks
        if (bkw)
          trk[t].chi2() = -1.f;
        continue;
      }
      for (int k = 0; k < 6; ++k)
        trk[t].params().v[k] = parA.At(s, k, 0);
      errA.copyOut(s, trk[t].errors().v);
      trk[t].chi2() = chi2[s];
    }
  }

  // MkBuilder::fit_tracks outlier block for one track; returns the number of hits removed. Per hit j (fit order,
  // innermost first) chi2Fwd = forward and chi2Bkwd = backward chi2 of j, non-finite ones set to 1e6, scorer
  // chi2Fwd + chi2Bkwd; MkFitCore walks a std::multimap keyed by -scorer (descending scorer, equal scorers in j order)
  // and removes every hit that passes the cut while more than 3 hits remain. The cut: the innermost hit (j = 0) when
  // chi2Bkwd > 20, the outermost one when chi2Fwd > 20 (each tested with the pass that predicts it from all other hits;
  // kEndHitChi2Cut = the KFFittingSmoother EstimateCut), any hit when (pT > 1: scorer > 20, chi2Fwd > 8, chi2Bkwd > 8)
  // or (pT <= 1: scorer > 15, chi2Fwd > 7, chi2Bkwd > 7). An outlier that the three-hit minimum keeps on the track with
  // no other hit removed is a failed fit (chi2 = -1).
  constexpr float kEndHitChi2Cut = 20.f;
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int removeOutliers(TrackSoAView trk, int t, FitScratch const& sc, int nFH) {
    float* scorer = sc.key;  // the sort keys are no longer needed
    for (int j = 0; j < nFH; ++j) {
      float& chi2Fwd = sc.chi2f[j * sc.stride + t];
      float& chi2Bkwd = sc.chi2b[(nFH - 1 - j) * sc.stride + t];
      if (!::mkfitdev::isFinite(chi2Fwd))
        chi2Fwd = 1000000;
      if (!::mkfitdev::isFinite(chi2Bkwd))
        chi2Bkwd = 1000000;
      scorer[j * sc.stride + t] = chi2Fwd + chi2Bkwd;
    }
    float pt = 1.f / trk[t].params().v[3];  // Track::pT() of the refitted state
    pt = pt < 0.f ? -pt : pt;
    // the hits passing the cut, as a bit mask (nFH <= kMaxTrkHits = 64)
    uint64_t passMask = 0;
    for (int j = 0; j < nFH; ++j) {
      const float sj = scorer[j * sc.stride + t];
      const float chi2Fwd = sc.chi2f[j * sc.stride + t];
      const float chi2Bkwd = sc.chi2b[(nFH - 1 - j) * sc.stride + t];
      if ((j == 0 && chi2Bkwd > kEndHitChi2Cut) || (j == nFH - 1 && chi2Fwd > kEndHitChi2Cut) ||
          (pt > 1 && sj > 20 && chi2Fwd > 8 && chi2Bkwd > 8) || (pt <= 1 && sj > 15 && chi2Fwd > 7 && chi2Bkwd > 7))
        passMask |= uint64_t(1) << j;
    }
    // a passing hit is removed when fewer than nFH - 3 passing hits precede it in the walk (removals so far)
    int nRemoved = 0;
    bool blocked = false;
    for (int j = 0; j < nFH; ++j) {
      if (!(passMask >> j & 1))
        continue;
      const float kj = -scorer[j * sc.stride + t];
      int rank = 0;  // position in the multimap walk among passing entries
      for (int j2 = 0; j2 < nFH; ++j2) {
        const float k2 = -scorer[j2 * sc.stride + t];
        if ((passMask >> j2 & 1) && (k2 < kj || (k2 == kj && j2 < j)))
          ++rank;
      }
      if (rank >= nFH - 3) {
        blocked = true;  // 3 hits is the minimum
        continue;
      }
      // removeHit(sortedIdxs[nFH - 1 - j]): sortedIdxs = reFitIndices output (first nFH entries)
      const int m = sc.order[(nFH - 1 - j) * sc.stride + t];
      auto& hot = trk[t].hits().hot[m];
      hot.index = ::mkfitdev::fitOutlierIdx(hot.index);
      trk[t].nFoundHits() = trk[t].nFoundHits() - 1;
      ++nRemoved;
    }
    if (blocked && nRemoved == 0)
      trk[t].chi2() = -1.f;
    return nRemoved;
  }

  // MkBuilder::fit_tracks: TrackerInfo::prop_config().final_fit_pflags and the final fit's FinalFitFlags from the ES
  // (b_field_at_mid, radial_field_corr); final_fit selects the final fit's own sequences (final_fit:: in FinalFit.cc)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE PropagationFlags refitKernelFlags(ESView const& es) {
    const ::mkfitdev::RefitConfig& rc = es.config->refit;
    PropagationFlags pf = ::mkfitdev::prop::propagationFlags(es, ::mkfitdev::prop::PropStage::FinalFit);
    pf.b_field_at_mid = rc.bFieldAtMid;
    pf.radial_field_corr = rc.radialFieldCorr;
    pf.final_fit = true;
    return pf;
  }

  struct FitTally {
    int nOverflow = 0, nMismatch = 0;
  };

  // The hits a fit of track t uses: reFitIndices orders them (n); their number is the fit's nFoundHits, as
  // MkBuilder::fittracks groups the tracks by the hits with index >= 0. nFoundHits() also counts kHitCCCFilterIdx
  // entries, which have no hit to fit: a difference is counted.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int fitHitCount(
      TAcc const& acc, FitHitAccess const& ha, TrackSoAView trk, int t, FitScratch const& sc, FitTally& tally) {
    const int n = reFitIndices(acc, ha, trk, t, sc, tally.nOverflow);
    if (trk[t].nFoundHits() != n)
      ++tally.nMismatch;
    return n;
  }

  // The outlier rounds of one track after its fit (MkBuilder::fittracks remap loop, N = 1): remove, refit with the
  // remaining hits, and test the refit again, until a round removes no hit (every round removes at least one and a
  // track keeps at least 3, so this ends after at most nFH - 3 rounds). maxRounds bounds the rounds run here (the GPU
  // splits the first rounds into stage kernels); the return is true when the last refit was not tested yet.
  template <bool kCpe, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool outliersAndRefit(TAcc const& acc,
                                                       FitHitAccess const& ha,
                                                       PropagationFlags const& pf,
                                                       TrackSoAView trk,
                                                       int t,
                                                       FitScratch const& sc,
                                                       int nFH,
                                                       FitTally& tally,
                                                       int maxRounds = ::mkfitdev::kMaxTrkHits) {
    for (int round = 0; round < maxRounds; ++round) {
      const int nRemoved = removeOutliers(trk, t, sc, nFH);
      // remap: tracks that lost hits are refitted with nFoundHits - n_removed hits (> 2 always holds)
      if (nRemoved == 0 || nFH - nRemoved <= 2)
        return false;
      const int n = fitHitCount(acc, ha, trk, t, sc, tally);
      const int tl[1] = {t}, nl[1] = {n};
      fitDirection<1, kCpe, false>(ha, pf, trk, tl, 1, sc, nl, n, sc.chi2f);
      fitDirection<1, kCpe, true>(ha, pf, trk, tl, 1, sc, nl, n, sc.chi2b);
      nFH = n;
    }
    return true;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTally(TAcc const& acc, FitCounters* counters, FitTally const& y) {
    if (y.nOverflow)
      alpaka::atomicAdd(acc, &counters->nOverflow, y.nOverflow, alpaka::hierarchy::Blocks{});
    if (y.nMismatch)
      alpaka::atomicAdd(acc, &counters->nHitCountMismatch, y.nMismatch, alpaka::hierarchy::Blocks{});
  }

  // Row count: host value, or the device count clamped to the capacity.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int fitRowCount(int nTracks, const int32_t* nTracksDev) {
    if (nTracksDev == nullptr)
      return nTracks;
    const int n = *nTracksDev;
    return n < 0 ? 0 : (n > nTracks ? nTracks : n);
  }

  // GPU split. Stage 0: reFitIndices. Stage 1/2: forward/backward fit.
  // Stage 3: outlier removal + reFitIndices of the refit (flag). Stage 4/5: forward/backward refit of flagged tracks.
  // Per track the same calls as outliersAndRefit, in the same order; meta[] carries n, nFH and the refit flag.
  // Stages 3, 4, 5 are launched once per outlier round; round > 0 of stage 3 only looks at the tracks refitted in the
  // previous round (refit[t] != 0).
  template <int kStage, bool kCpe>
  class KernelFinalFitStage {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  FitHitAccess ha,
                                  TrackSoAView trk,
                                  int nTracks,
                                  const int32_t* nTracksDev,
                                  FitScratch sc,
                                  FitCounters* counters,
                                  int round = 0) const {
      const PropagationFlags pf = refitKernelFlags(ha.es);
      const int nRows = fitRowCount(nTracks, nTracksDev);
      for (int32_t t : cms::alpakatools::uniform_elements(acc, nRows)) {
        FitTally tally;
        int32_t* nOrd = sc.meta;
        int32_t* nFHs = sc.meta + sc.stride;
        int32_t* refit = sc.meta + 2 * sc.stride;
        if constexpr (kStage == 0 || kStage == 3) {
          if constexpr (kStage == 3) {
            if (round > 0 && refit[t] == 0)
              continue;
            const int nRemoved = removeOutliers(trk, t, sc, nFHs[t]);
            refit[t] = 0;
            if (nRemoved == 0 || nFHs[t] - nRemoved <= 2) {
              addTally(acc, counters, tally);
              continue;
            }
            refit[t] = 1;
          }
          const int n = fitHitCount(acc, ha, trk, t, sc, tally);
          nOrd[t] = n;
          nFHs[t] = n;
        }
        if constexpr (kStage != 0 && kStage != 3) {
          if ((kStage == 1 || kStage == 2) || refit[t]) {
            const int tl[1] = {t}, nl[1] = {nOrd[t]};
            constexpr bool bkw = (kStage == 2 || kStage == 5);
            fitDirection<1, kCpe, bkw>(ha, pf, trk, tl, 1, sc, nl, nFHs[t], bkw ? sc.chi2b : sc.chi2f);
          }
        }
        addTally(acc, counters, tally);
      }
    }
  };

  // GPU: the first outlier rounds run as stages 3, 4, 5; the tracks still refitted after them (refit[t] != 0, their
  // last refit not yet tested) continue in one tail kernel with the same calls per track, so the split does not
  // change the result.
  constexpr int kStagedOutlierRounds = 6;
  template <bool kCpe>
  class KernelFinalFitTail {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  FitHitAccess ha,
                                  TrackSoAView trk,
                                  int nTracks,
                                  const int32_t* nTracksDev,
                                  FitScratch sc,
                                  FitCounters* counters) const {
      const PropagationFlags pf = refitKernelFlags(ha.es);
      const int nRows = fitRowCount(nTracks, nTracksDev);
      const int32_t* nFHs = sc.meta + sc.stride;
      const int32_t* refit = sc.meta + 2 * sc.stride;
      for (int32_t t : cms::alpakatools::uniform_elements(acc, nRows)) {
        if (refit[t] == 0)
          continue;
        FitTally tally;
        outliersAndRefit<kCpe>(acc, ha, pf, trk, t, sc, nFHs[t], tally);
        addTally(acc, counters, tally);
      }
    }
  };

  // CPU backends: MkBuilder::fittracks groups tracks by their number of hits with index >= 0 (std::map, ascending) in
  // chunks of NN. Reproduced by counting: per-hit-count histogram, prefix offsets, stable scatter (track order kept
  // inside a hit count), then chunks of N. One thread (cheap: one pass over the tracks).
  struct FitGroups {
    int32_t* perm;     // [nTracks] tracks ordered by their hit count, input order inside equal counts
    int32_t* start;    // [capacity] first perm index of each group
    int32_t* count;    // [capacity] tracks in the group (<= N)
    int32_t* nGroups;  // scalar
  };
  constexpr int kMaxGroupKey = ::mkfitdev::kMaxTrkHits;

  template <int N>
  class KernelGroupTracks {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, TrackSoAView trk, int nTracksCap, const int32_t* nTracksDev, FitGroups groups) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        const int nTracks = fitRowCount(nTracksCap, nTracksDev);
        int hist[kMaxGroupKey + 2];
        for (int iKey = 0; iKey < kMaxGroupKey + 2; ++iKey)
          hist[iKey] = 0;
        auto key = [&trk](int t) {
          int k = 0;
          const auto& hot = trk[t].hits().hot;
          const int nTot = trk[t].nTotalHits();
          for (int ih = 0; ih < nTot; ++ih)
            if (hot[ih].index >= 0)
              ++k;
          return k > kMaxGroupKey ? kMaxGroupKey + 1 : k;
        };
        for (int t = 0; t < nTracks; ++t)
          ++hist[key(t)];
        int off = 0, ng = 0;
        for (int iKey = 0; iKey < kMaxGroupKey + 2; ++iKey) {
          const int nInKey = hist[iKey];
          for (int i = 0; i < nInKey; i += N) {
            groups.start[ng] = off + i;
            groups.count[ng] = (nInKey - i) < N ? (nInKey - i) : N;
            ++ng;
          }
          hist[iKey] = off;
          off += nInKey;
        }
        for (int t = 0; t < nTracks; ++t)
          groups.perm[hist[key(t)]++] = t;
        *groups.nGroups = ng;
      }
    }
  };

  template <int N, bool kCpe>
  class KernelFinalFitGrouped {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  FitHitAccess ha,
                                  TrackSoAView trk,
                                  int maxGroups,
                                  FitGroups groups,
                                  FitScratch sc,
                                  FitCounters* counters) const {
      const PropagationFlags pf = refitKernelFlags(ha.es);
      const int nGroups = *groups.nGroups;
      for (int32_t ig : cms::alpakatools::uniform_elements(acc, maxGroups)) {
        if (ig >= nGroups)
          continue;
        FitTally tally;
        const int nProc = groups.count[ig];
        int tl[N], nl[N];
        for (int s = 0; s < N; ++s) {
          const int t = groups.perm[groups.start[ig] + (s < nProc ? s : 0)];
          tl[s] = t;
          nl[s] = s < nProc ? fitHitCount(acc, ha, trk, t, sc, tally) : nl[0];
        }
        // every track of the group has the group's hit count (KernelGroupTracks key)
        const int nFH = nl[0];
        fitDirection<N, kCpe, false>(ha, pf, trk, tl, nProc, sc, nl, nFH, sc.chi2f);
        fitDirection<N, kCpe, true>(ha, pf, trk, tl, nProc, sc, nl, nFH, sc.chi2b);
        for (int s = 0; s < nProc; ++s)
          outliersAndRefit<kCpe>(acc, ha, pf, trk, tl[s], sc, nFH, tally);
        addTally(acc, counters, tally);
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::fit

#endif
