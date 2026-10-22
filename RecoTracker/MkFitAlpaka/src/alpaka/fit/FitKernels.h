#ifndef RecoTracker_MkFitAlpaka_src_alpaka_fit_FitKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_fit_FitKernels_h

// mkFit final fit (MkFitFitProducer path, procModifier trackingMkFitFit) on the device, ported from MkFitCore.
// MkFitCore: MkBuilder::fittracks / fit_tracks (MkBuilder.cc:1432-1660), MkFitter fwdFitInputTracks,
// bkReFitInputTracks, reFitOutputTracks, reFitIndices, fwdFitFitTracks, bkReFitFitTracks (MkFitter.cc:15-540).
// GPU: one thread per track (N = 1 Matriplex slot), the fit SPLIT into six kernels:
// pass-1 reFitIndices, pass-1 forward, pass-1 backward, outlier removal (+ pass-2 reFitIndices), pass-2 forward,
// pass-2 backward. The
// state between them is the track row itself (TrackSoA params/errors/chi2) plus the per-track scratch below; every
// track runs exactly the operations of a single-kernel fit in the same order.
// CPU: tracks grouped by nFoundHits into kNN-wide Matriplex groups as MkFitCore (one kernel).
// CPE: kCpe = true applies the device PixelCPEGeneric with the track angles on every pixel hit through
// the kalmanOperationPlaneLocal hook (do_cpe / cpe_corr_func, MkFitter.cc:258-275, KalmanUtilsMPlex.cc:1548-1716);
// kCpe = false is the no-CPE variant.

#include <cstdint>
#include <limits>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeGeneric.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/HitStateDev.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
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
    // storeHitStates (MkFitCore per-hit smoothed states; nullptr = off): hsFwd = forward updated local state per
    // forward step, element (h * kHsFwd + k) * stride + t (5 par, 15 err, pz sign); hs = output, t * kMaxTrkHits + pos
    float* hsFwd = nullptr;
    ::mkfitdev::fit::HitStateDev* hs = nullptr;
  };
  constexpr int kHsFwd = 21;

  struct FitHitAccess {
    ESView es;
    HitSoAConstView hits;
    uint32_t nPixel;
    ::mkfitdev::cpe::CpeTables cpe;               // device CPE tables (used when kCpe)
    const ::mkfitdev::cpe::ClusterCpe* clusters;  // [nPixel] per pixel hit (used when kCpe)
    FitOptions opt;                               // outlier switches (DEVIATION DEV-3; defaults = MkFitCore)

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
  // bkw (bkReFitFitTracks, reversed copy) visits order[h] (outermost first). Material effects are skipped on the
  // first hit of each direction (MkFitCore: index == indices.back()). The input state comes from the track row with
  // its errors scaled by 100 and chi2 = 0 (fwdFitInputTracks / bkReFitInputTracks); the charge is the track's and
  // MkFitCore never writes a flipped charge back (reFitOutputTracks copies errors, parameters and chi2 only).
  // reFitOutputTracks: when the chi2 sum is NaN the track keeps its previous state.
  // N slots (tracks tl[0..nProc), all with the same nFH, as fit_tracks groups them; slots >= nProc repeat
  // slot 0 and are discarded). N = 1 on GPU and for the pass-2 refits, N = kNN for pass 1 on CPU backends.
  //
  // driven by the ES switches
  // (ESConfig::refit = mkfit::Config::refit*, MkFitGeometryESProducer refit* parameters):
  //   - pf carries b_field_at_mid / radial_field_corr (refitKernelFlags); eloss_by_pass/eloss_outward are set here
  //     per pass (refitElossSignFromPass: forward loses, backward gains);
  //   - the first hit of each pass is an update without propagation (propHit = h != 0);
  //   - per-module material (refitMaterialPerModule) from the ES module table (ModuleInfo::radl/bbxi);
  //   - backward pass: multiple scattering at the |p| of its start state (refitBkwMsFixedMomentum, ms_ref_p)
  //     and refitBkwSubSteps sub-steps (propagateHelixToPlaneSubStepMPlex) where the state is off the plane.
  // kBkw is a template parameter so that the forward kernels do not carry the sub-step code (registers).
  // storeHitStates (kStore; MkFitter::storeHitStates): the forward pass keeps its updated local state per hit
  // (sc.hsFwd), the backward pass writes the smoothed state per HitOnTrack position (sc.hs): forward updated at the
  // outermost hit, backward updated at the innermost, two-filter combination of forward updated and backward
  // predicted in between. The validation-only fwd/bwd copies of MkFitCore (validateHitStates) are not ported.
  template <idx_t N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fillHitState(::mkfitdev::fit::HitStateDev& o,
                                                   const MPlex5V<N>& par,
                                                   const MPlex5S<N>& err,
                                                   const int n) {
    bool fin = true;
    for (int i = 0; i < 5; ++i) {
      o.par[i] = par.constAt(n, i, 0);
      for (int j = 0; j <= i; ++j)
        o.err[i * (i + 1) / 2 + j] = err.constAt(n, i, j);
      const float d = o.err[i * (i + 3) / 2];
      fin = fin && ::mkfitdev::isFinite(o.par[i]) && ::mkfitdev::isFinite(d) && d > 0.f;
    }
    o.valid = fin;
  }

  template <idx_t N, bool kCpe, bool kBkw, bool kStore = false>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fitDirection(FitHitAccess const& ha,
                                                   PropagationFlags const& pfIn,
                                                   TrackSoAView trk,
                                                   const int (&tl)[N],
                                                   const int nProc,
                                                   FitScratch const& sc,
                                                   const int (&nl)[N],
                                                   int nFH,
                                                   float* chi2Out,
                                                   int& nNaN,
                                                   bool propFirst = false) {
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
    if constexpr (kStore && !bkw) {
      // fit_tracks: hs.assign(nTotalHits, HitStateOnTrack{}) for every fit of a track (pass 1 and the refit)
      for (int s = 0; s < nProc; ++s) {
        const int t = tl[s];
        const int nTot = trk[t].nTotalHits() < ::mkfitdev::kMaxTrkHits ? trk[t].nTotalHits() : ::mkfitdev::kMaxTrkHits;
        for (int p = 0; p < nTot; ++p)
          sc.hs[t * ::mkfitdev::kMaxTrkHits + p] = ::mkfitdev::fit::HitStateDev{};
      }
    }
    [[maybe_unused]] LocalStatesOut<N> loc;
    [[maybe_unused]] MPlex5V<N> lPar;
    [[maybe_unused]] MPlex5S<N> lErr;
    [[maybe_unused]] MPlexQI<N> lPz;
    for (int h = 0; h < nFH; ++h) {
      // just update when the position is already at the hit; DEVIATION DEV-5: propFirst propagates there
      const bool propHit = h != 0 || propFirst;
      [[maybe_unused]] const bool innermost = h == nFH - 1;
      const LocalStatesOut<N>* locPtr = nullptr;
      if constexpr (kStore) {
        loc = LocalStatesOut<N>{};
        if constexpr (!bkw) {
          if (h > 0) {  // not needed at the innermost hit, where the smoothed state is the backward one
            loc.updPar = &lPar;
            loc.updErr = &lErr;
            loc.pzSign = &lPz;
            locPtr = &loc;
          }
        } else {
          if (h > 0 && !innermost) {
            loc.predPar = &lPar;
            loc.predErr = &lErr;
          }
          if (innermost) {
            loc.updPar = &lPar;
            loc.updErr = &lErr;
          }
          loc.pzSign = &lPz;
          locPtr = &loc;
        }
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
        // it through Packers changed GCC's N = 8 codegen of this kernel at rounding level
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
        // bkReFitFitTracks: refitBkwSubSteps > 1 sub-steps the lanes whose state is off the plane
        bool splitLane[N];
        for (int s = 0; s < N; ++s)
          splitLane[s] = false;
        if (propHit && rc.bkwSubSteps > 1) {
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
                                                      propHit,
                                                      &noMat,
                                                      cpeObj,
                                                      matRadlPtr,
                                                      matBbxiPtr,
                                                      msRefPtr,
                                                      locPtr);
      if constexpr (kStore) {
        if constexpr (!bkw) {
          if (h > 0) {
            for (int s = 0; s < nProc; ++s) {
              const int t = tl[s];
              float* f = sc.hsFwd;
              for (int i = 0; i < 5; ++i)
                f[(h * kHsFwd + i) * sc.stride + t] = lPar.constAt(s, i, 0);
              int k = 5;
              for (int i = 0; i < 5; ++i)
                for (int j = 0; j <= i; ++j)
                  f[(h * kHsFwd + k++) * sc.stride + t] = lErr.constAt(s, i, j);
              f[(h * kHsFwd + 20) * sc.stride + t] = float(lPz.constAt(s, 0, 0));
            }
          }
        } else {
          // MkFitter::storeHitStates for the h-th hit of the backward pass
          const int hf = nFH - 1 - h;  // the same hit in the forward pass
          const bool outermost = h == 0;
          MPlex5V<N> fPar, sPar;
          MPlex5S<N> fErr, sErr;
          MPlexQI<N> ok(1);
          if (!innermost) {
            for (int s = 0; s < N; ++s) {
              const int t = tl[s < nProc ? s : 0];
              const float* f = sc.hsFwd;
              for (int i = 0; i < 5; ++i)
                fPar.At(s, i, 0) = f[(hf * kHsFwd + i) * sc.stride + t];
              int k = 5;
              for (int i = 0; i < 5; ++i)
                for (int j = 0; j <= i; ++j)
                  fErr.At(s, i, j) = f[(hf * kHsFwd + k++) * sc.stride + t];
            }
          }
          if (!outermost && !innermost)
            smoothLocalStatesPlane(fPar, fErr, lPar, lErr, sPar, sErr, ok, nProc);
          for (int s = 0; s < nProc; ++s) {
            const int t = tl[s];
            const int m = sc.order[h * sc.stride + t];
            ::mkfitdev::fit::HitStateDev& o = sc.hs[t * ::mkfitdev::kMaxTrkHits + m];
            if (outermost) {
              fillHitState(o, fPar, fErr, s);
              o.chi2 = sc.chi2f[(nFH - 1) * sc.stride + t];
              o.pzSign = int8_t(sc.hsFwd[(hf * kHsFwd + 20) * sc.stride + t]);
              o.kind = 1;
            } else if (innermost) {
              fillHitState(o, lPar, lErr, s);
              o.chi2 = outChi2.At(s, 0, 0);
              o.pzSign = int8_t(lPz.constAt(s, 0, 0));
              o.kind = 2;
            } else {
              fillHitState(o, sPar, sErr, s);
              o.valid = o.valid && ok.constAt(s, 0, 0);
              o.chi2 = outChi2.At(s, 0, 0);
              o.pzSign = int8_t(lPz.constAt(s, 0, 0));
              o.kind = 0;
            }
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
      // DEVIATION DEV-1: MkFitCore's x != x is folded away by -Ofast (NaN state written); here it works (no
      // finite-math)
      if (chi2[s] != chi2[s]) {  // reFitOutputTracks: "trick for the nan so the track is not dead"
        ++nNaN;
        continue;
      }
      for (int k = 0; k < 6; ++k)
        trk[t].params().v[k] = parA.At(s, k, 0);
      errA.copyOut(s, trk[t].errors().v);
      trk[t].chi2() = chi2[s];
    }
  }

  // MkBuilder::fit_tracks outlier block (MkBuilder.cc:1490-1535) for one track. Returns the number of hits removed.
  // MkFitCore ranks hits through std::map<float, int> scorerAndIdx; scorerAndIdx[-scorer] = j: equal scorers collide
  // and only the LAST j survives. The map is walked in ascending key order
  // (descending scorer) and passing hits are removed while more than 3 found hits remain. The walk
  // position of a passing survivor is the number of passing survivors with a larger scorer.
  // opt.edgeOutliers (DEVIATION DEV-3, switch, default off): the first fitted hit (j = 0) is also an outlier when the
  // pass that predicts it from all other hits has chi2 > opt.edgeChi2Cut, and the last one (j = nFH - 1) likewise.
  // DEVIATION DEV-10: the same test for the opt.nearEdgeHitsInner / Outer hits nearest each end; at most
  // opt.outliersPerRound removed per round, in the map walk order (largest f + b first).
  // Two mkFit layers are one detector layer (the two sensors of a stacked module) when their mean radius (barrel) or
  // mean z (endcap) differ by less than kSameLayerTol: stacked sensors are at most 0.4 cm apart, separate layers 4 cm.
  constexpr float kSameLayerTol = 1.f;
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool sameDetectorLayer(ESView const& es, int l1, int l2) {
    if (l1 == l2)
      return true;
    const int ty = es.layers[l1].layer_type();
    if (ty != es.layers[l2].layer_type())
      return false;
    const float d = ty == static_cast<int>(::mkfitdev::LayerType::Barrel)
                        ? (es.layers[l1].rin() + es.layers[l1].rout()) - (es.layers[l2].rin() + es.layers[l2].rout())
                        : (es.layers[l1].zmin() + es.layers[l1].zmax()) - (es.layers[l2].zmin() + es.layers[l2].zmax());
    return d < 2.f * kSameLayerTol && d > -2.f * kSameLayerTol;
  }

  // Number of detector layers other than hit j's own among the fitted hits before j (inner to outer), up to `need`.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int otherLayersBefore(
      FitHitAccess const& ha, TrackSoAView trk, int t, FitScratch const& sc, int nFH, int j, int need) {
    const auto& hot = trk[t].hits().hot;
    auto layerAt = [&](int p) { return int(hot[sc.order[(nFH - 1 - p) * sc.stride + t]].layer); };
    const int lj = layerAt(j);
    int n = 0;
    for (int p = 0; p < j && n < need; ++p) {
      const int lp = layerAt(p);
      if (sameDetectorLayer(ha.es, lp, lj))
        continue;
      bool seen = false;
      for (int q = 0; q < p && !seen; ++q)
        seen = sameDetectorLayer(ha.es, layerAt(q), lp);
      if (!seen)
        ++n;
    }
    return n;
  }

  // DEVIATION DEV-10: with an outer window (nearEdgeHitsOuter > 1) an outer hit is tested only when the hits before it
  // lie on at least kNearEdgeMinLayers other detector layers (three points fix the circle).
  constexpr int kNearEdgeMinLayers = 3;
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int removeOutliers(
      FitHitAccess const& ha, TrackSoAView trk, int t, FitScratch const& sc, int nFH, int& nShadowed) {
    FitOptions const& opt = ha.opt;
    float* scorer = sc.key;  // the sort keys are no longer needed
    for (int j = 0; j < nFH; ++j) {
      float& f = sc.chi2f[j * sc.stride + t];
      float& b = sc.chi2b[(nFH - 1 - j) * sc.stride + t];
      // DEVIATION DEV-1: MkFitCore's NaN -> 1e6 substitution is folded away by -Ofast; here it works (no finite-math)
      if (f != f)
        f = 1000000;
      if (b != b)
        b = 1000000;
      scorer[j * sc.stride + t] = f + b;
    }
    float pt = 1.f / trk[t].params().v[3];  // Track::pT() of the refitted state
    pt = pt < 0.f ? -pt : pt;
    // passing survivors as a bit mask (nFH <= kMaxTrkHits = 64); windowMask: those only the outer window flags
    uint64_t passMask = 0, windowMask = 0;
    const bool outerWindow = opt.nearEdgeHitsOuter > 1;
    for (int j = 0; j < nFH; ++j) {
      const float s = scorer[j * sc.stride + t];
      bool survives = true;  // a later j with the same map key overwrites this entry
      for (int j2 = j + 1; j2 < nFH; ++j2)
        if (-scorer[j2 * sc.stride + t] == -s)
          survives = false;
      const float f = sc.chi2f[j * sc.stride + t];
      const float b = sc.chi2b[(nFH - 1 - j) * sc.stride + t];
      bool cut = (pt > 1 && s > 20 && f > 8 && b > 8) || (pt <= 1 && s > 15 && f > 7 && b > 7);
      // DEVIATION DEV-3: edge hits tested with the pass that predicts them (MkFitCore tests them as interior hits only)
      // DEVIATION DEV-10: the hits nearest each end (DEV-3: one per end), each with its well-determined side
      bool windowOnly = false;
      if (opt.edgeOutliers) {
        bool outer = j >= nFH - opt.nearEdgeHitsOuter && f > opt.edgeChi2Cut;
        if (outer && outerWindow)
          outer = otherLayersBefore(ha, trk, t, sc, nFH, j, kNearEdgeMinLayers) >= kNearEdgeMinLayers;
        const bool base = cut || (j < opt.nearEdgeHitsInner && b > opt.edgeChi2Cut) || (j == nFH - 1 && outer);
        windowOnly = !base && outer;
        cut = base || outer;
      }
      if (cut && !survives)
        ++nShadowed;  // passes the cut but lost the map collision: MkFitCore can never remove it
      if (survives && cut) {
        passMask |= uint64_t(1) << j;
        if (windowOnly)
          windowMask |= uint64_t(1) << j;
      }
    }
    // DEVIATION DEV-10: the window's own hits count only in a round in which no other hit of the track is flagged
    if (passMask & ~windowMask)
      passMask &= ~windowMask;
    // DEVIATION DEV-10: at most outliersPerRound hits per round (0 = MkFitCore: all while more than 3 remain)
    const int maxRemove = (opt.outliersPerRound > 0 && opt.outliersPerRound < nFH - 3) ? opt.outliersPerRound : nFH - 3;
    int nRemoved = 0;
    for (int j = 0; j < nFH && maxRemove > 0; ++j) {
      if (!(passMask >> j & 1))
        continue;
      const float kj = -scorer[j * sc.stride + t];
      int rank = 0;  // position in the map walk among passing entries
      for (int j2 = 0; j2 < nFH; ++j2)
        if ((passMask >> j2 & 1) && -scorer[j2 * sc.stride + t] < kj)
          ++rank;
      if (rank >= maxRemove)
        continue;
      // removeHit(sortedIdxs[nFH - 1 - j]): sortedIdxs = reFitIndices output (first nFH entries)
      const int m = sc.order[(nFH - 1 - j) * sc.stride + t];
      trk[t].hits().hot[m].index = -1;
      trk[t].nFoundHits() = trk[t].nFoundHits() - 1;
      ++nRemoved;
    }
    return nRemoved;
  }

  // MkBuilder::fit_tracks my_flags: use_param_b_field | apply_material, plus the field-model
  // switches of the final fit from the ES (refitBFieldAtMid, refitRadialFieldCorr)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE PropagationFlags refitKernelFlags(ESView const& es) {
    const ::mkfitdev::RefitConfig& rc = es.config->refit;
    return PropagationFlags(PF_use_param_b_field | PF_apply_material | (rc.bFieldAtMid ? PF_b_field_at_mid : PF_none) |
                                (rc.radialFieldCorr ? PF_radial_field_corr : PF_none),
                            es.material);
  }

  struct FitTally {
    int nOverflow = 0, nNaN = 0, nMismatch = 0, nRemoved = 0, nRefit = 0, nShadowed = 0;
  };

  // Pass-1 tail of one track (outlier removal) and its pass 2 (remap refit, N = 1).
  // DEVIATION DEV-3: opt.outlierRounds > 1 repeats removal + refit on the refitted track (MkFitCore: one refit, never
  // checked again); the last refit is not checked, as MkFitCore's remap loop with nextRemap = nullptr.
  template <bool kCpe, bool kStore = false, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void outliersAndRefit(TAcc const& acc,
                                                       FitHitAccess const& ha,
                                                       PropagationFlags const& pf,
                                                       TrackSoAView trk,
                                                       int t,
                                                       FitScratch const& sc,
                                                       int nFH,
                                                       FitTally& tally) {
    for (int round = 0; round < ha.opt.outlierRounds; ++round) {
      const int nRemoved = removeOutliers(ha, trk, t, sc, nFH, tally.nShadowed);
      tally.nRemoved += nRemoved;
      // remap: tracks that lost hits are refitted with nFoundHits - n_removed (> 2 always holds)
      if (nRemoved == 0 || nFH - nRemoved <= 2)
        return;
      ++tally.nRefit;
      const int n = reFitIndices(acc, ha, trk, t, sc, tally.nOverflow);
      int nFH2 = trk[t].nFoundHits();
      if (nFH2 != n) {
        ++tally.nMismatch;
        if (nFH2 > n)
          nFH2 = n;
      }
      const int tl[1] = {t}, nl[1] = {n};
      fitDirection<1, kCpe, false, kStore>(  // DEVIATION DEV-5: firstHitProp >= 1 propagates to the refit's first hit
          ha,
          pf,
          trk,
          tl,
          1,
          sc,
          nl,
          nFH2,
          sc.chi2f,
          tally.nNaN,
          ha.opt.firstHitProp >= 1);
      fitDirection<1, kCpe, true, kStore>(  // DEVIATION DEV-5: firstHitProp >= 1 propagates to the refit's first hit
          ha,
          pf,
          trk,
          tl,
          1,
          sc,
          nl,
          nFH2,
          sc.chi2b,
          tally.nNaN,
          ha.opt.firstHitProp >= 1);
      nFH = nFH2;
    }
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTally(TAcc const& acc, FitCounters* counters, FitTally const& y) {
    if (y.nOverflow)
      alpaka::atomicAdd(acc, &counters->nOverflow, y.nOverflow, alpaka::hierarchy::Blocks{});
    if (y.nNaN)
      alpaka::atomicAdd(acc, &counters->nNaN, y.nNaN, alpaka::hierarchy::Blocks{});
    if (y.nMismatch)
      alpaka::atomicAdd(acc, &counters->nHitCountMismatch, y.nMismatch, alpaka::hierarchy::Blocks{});
    if (y.nRemoved)
      alpaka::atomicAdd(acc, &counters->nRemovedHits, y.nRemoved, alpaka::hierarchy::Blocks{});
    if (y.nShadowed)
      alpaka::atomicAdd(acc, &counters->nShadowed, y.nShadowed, alpaka::hierarchy::Blocks{});
    if (y.nRefit)
      alpaka::atomicAdd(acc, &counters->nRefit, y.nRefit, alpaka::hierarchy::Blocks{});
  }

  // Both passes of one track with N = 1 (CPU fallback for groups with a hit-count mismatch).
  template <bool kCpe, bool kStore = false, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fitTrackSingle(TAcc const& acc,
                                                     FitHitAccess const& ha,
                                                     PropagationFlags const& pf,
                                                     TrackSoAView trk,
                                                     int t,
                                                     FitScratch const& sc,
                                                     FitTally& tally) {
    const int n = reFitIndices(acc, ha, trk, t, sc, tally.nOverflow);
    int nFH = trk[t].nFoundHits();
    if (nFH != n) {
      // MkFitCore indexes indices[] with nFoundHits steps; nFoundHits > indices.size() would read out of bounds
      ++tally.nMismatch;
      if (nFH > n)
        nFH = n;
    }
    const int tl[1] = {t}, nl[1] = {n};
    fitDirection<1, kCpe, false, kStore>(  // DEVIATION DEV-5: firstHitProp 2 propagates to the first hit in pass 1
        ha,
        pf,
        trk,
        tl,
        1,
        sc,
        nl,
        nFH,
        sc.chi2f,
        tally.nNaN,
        ha.opt.firstHitProp >= 2);
    fitDirection<1, kCpe, true, kStore>(  // DEVIATION DEV-5: firstHitProp 2 propagates to the first hit in pass 1
        ha,
        pf,
        trk,
        tl,
        1,
        sc,
        nl,
        nFH,
        sc.chi2b,
        tally.nNaN,
        ha.opt.firstHitProp >= 2);
    outliersAndRefit<kCpe, kStore>(acc, ha, pf, trk, t, sc, nFH, tally);
  }

  // Row count: host value, or the device count clamped to the capacity.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int fitRowCount(int nTracks, const int32_t* nTracksDev) {
    if (nTracksDev == nullptr)
      return nTracks;
    const int n = *nTracksDev;
    return n < 0 ? 0 : (n > nTracks ? nTracks : n);
  }

  // GPU split. Stage 0: pass-1 reFitIndices. Stage 1/2: pass-1 forward/backward fit.
  // Stage 3: outlier removal + pass-2 reFitIndices (flag). Stage 4/5: pass-2 forward/backward fit of flagged tracks.
  // Per track the same calls as fitTrackSingle, in the same order; meta[] carries n, nFH and the pass-2 flag.
  // DEVIATION DEV-3 (opt.outlierRounds > 1): stages 3, 4, 5 are launched again per round; round > 0 of stage 3 only
  // looks at the tracks refitted in the previous round (refit[t] != 0).
  template <int kStage, bool kCpe, bool kStore = false>
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
          int nFH;
          if constexpr (kStage == 0) {
            nFH = trk[t].nFoundHits();
          } else {
            if (round > 0 && refit[t] == 0)
              continue;
            const int nRemoved = removeOutliers(ha, trk, t, sc, nFHs[t], tally.nShadowed);
            tally.nRemoved += nRemoved;
            refit[t] = 0;
            if (nRemoved == 0 || nFHs[t] - nRemoved <= 2) {
              addTally(acc, counters, tally);
              continue;
            }
            refit[t] = 1;
            ++tally.nRefit;
            nFH = trk[t].nFoundHits();
          }
          const int n = reFitIndices(acc, ha, trk, t, sc, tally.nOverflow);
          if (nFH != n) {
            ++tally.nMismatch;
            if (nFH > n)
              nFH = n;
          }
          nOrd[t] = n;
          nFHs[t] = nFH;
        }
        if constexpr (kStage != 0 && kStage != 3) {
          if ((kStage == 1 || kStage == 2) || refit[t]) {
            const int tl[1] = {t}, nl[1] = {nOrd[t]};
            constexpr bool bkw = (kStage == 2 || kStage == 5);
            fitDirection<1, kCpe, bkw, kStore>(  // DEVIATION DEV-5: refits (stages 4, 5) >= 1, pass 1 >= 2
                ha,
                pf,
                trk,
                tl,
                1,
                sc,
                nl,
                nFHs[t],
                bkw ? sc.chi2b : sc.chi2f,
                tally.nNaN,
                (kStage >= 4) ? ha.opt.firstHitProp >= 1 : ha.opt.firstHitProp >= 2);
          }
        }
        addTally(acc, counters, tally);
      }
    }
  };

  // GPU: rounds launched as stages 3, 4, 5; the tracks still refitted after them (refit[t] != 0, their last refit not
  // yet checked) continue in one tail kernel with the same calls per track, so the split does not change the result.
  // ha.opt.outlierRounds is the number of rounds left.
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

  // CPU backends: MkBuilder::fittracks groups tracks by nFoundHits (std::map, ascending) in chunks of NN.
  // Reproduced by counting: per-hit-count histogram, prefix offsets, stable scatter (track order kept inside a hit
  // count), then chunks of N. One thread (cheap: one pass over the tracks).
  struct FitGroups {
    int32_t* perm;     // [nTracks] tracks ordered by nFoundHits, input order inside equal counts
    int32_t* start;    // [capacity] first perm index of each group
    int32_t* count;    // [capacity] tracks in the group (<= N)
    int32_t* nGroups;  // scalar
  };
  constexpr int kMaxGroupKey = ::mkfitdev::kMaxTrkHits;

  template <int N>
  class KernelGroupTracks {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, TrackSoAView trk, int nTracksCap, const int32_t* nTracksDev, FitGroups g) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        const int nTracks = fitRowCount(nTracksCap, nTracksDev);
        int hist[kMaxGroupKey + 2];
        for (int b = 0; b < kMaxGroupKey + 2; ++b)
          hist[b] = 0;
        auto key = [&](int t) {
          const int k = trk[t].nFoundHits();
          return k < 0 ? 0 : (k > kMaxGroupKey ? kMaxGroupKey + 1 : k);
        };
        for (int t = 0; t < nTracks; ++t)
          ++hist[key(t)];
        int off = 0, ng = 0;
        for (int b = 0; b < kMaxGroupKey + 2; ++b) {
          const int c = hist[b];
          for (int i = 0; i < c; i += N) {
            g.start[ng] = off + i;
            g.count[ng] = (c - i) < N ? (c - i) : N;
            ++ng;
          }
          hist[b] = off;
          off += c;
        }
        for (int t = 0; t < nTracks; ++t)
          g.perm[hist[key(t)]++] = t;
        *g.nGroups = ng;
      }
    }
  };

  template <int N, bool kCpe, bool kStore = false>
  class KernelFinalFitGrouped {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  FitHitAccess ha,
                                  TrackSoAView trk,
                                  int maxGroups,
                                  FitGroups g,
                                  FitScratch sc,
                                  FitCounters* counters) const {
      const PropagationFlags pf = refitKernelFlags(ha.es);
      const int nGroups = *g.nGroups;
      for (int32_t ig : cms::alpakatools::uniform_elements(acc, maxGroups)) {
        if (ig >= nGroups)
          continue;
        FitTally tally;
        const int nProc = g.count[ig];
        int tl[N], nl[N];
        bool uniform = true;
        int nFH = -1;
        for (int s = 0; s < N; ++s) {
          const int t = g.perm[g.start[ig] + (s < nProc ? s : 0)];
          tl[s] = t;
          if (s >= nProc) {
            nl[s] = nl[0];
            continue;
          }
          nl[s] = reFitIndices(acc, ha, trk, t, sc, tally.nOverflow);
          const int f = trk[t].nFoundHits();
          if (f != nl[s] || (nFH >= 0 && f != nFH))
            uniform = false;
          nFH = f;
        }
        if (!uniform) {  // never seen; keep MkFitCore semantics per track
          for (int s = 0; s < nProc; ++s)
            fitTrackSingle<kCpe, kStore>(acc, ha, pf, trk, tl[s], sc, tally);
        } else {
          fitDirection<N, kCpe, false, kStore>(  // DEVIATION DEV-5: firstHitProp 2, pass 1
              ha,
              pf,
              trk,
              tl,
              nProc,
              sc,
              nl,
              nFH,
              sc.chi2f,
              tally.nNaN,
              ha.opt.firstHitProp >= 2);
          fitDirection<N, kCpe, true, kStore>(  // DEVIATION DEV-5: firstHitProp 2, pass 1
              ha,
              pf,
              trk,
              tl,
              nProc,
              sc,
              nl,
              nFH,
              sc.chi2b,
              tally.nNaN,
              ha.opt.firstHitProp >= 2);
          for (int s = 0; s < nProc; ++s)
            outliersAndRefit<kCpe, kStore>(acc, ha, pf, trk, tl[s], sc, nFH, tally);
        }
        addTally(acc, counters, tally);
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::fit

#endif
