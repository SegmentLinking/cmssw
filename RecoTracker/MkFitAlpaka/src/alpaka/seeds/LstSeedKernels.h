#ifndef RecoTracker_MkFitAlpaka_src_alpaka_seeds_LstSeedKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_seeds_LstSeedKernels_h

// option (b): device fit of the LST seed state.
// One thread per seed row (N = 1 Matriplex slot on every backend):
//   1. class from the hit content: no OT hit -> pixel-only (pLS) -> kept;
//   2. initial helix from three of the seed's hits (first, middle, last): circle in xy -> 1/pT, charge, phi at the
//      first hit; theta from the arc length and dz between the first and the last hit (no origin, no beam spot:
//      LST seeds may be displaced);
//   3. mkFit Kalman fit over the seed's hits in their stored order (the host seed order: inner -> outer), the same
//      primitive as the device final fit (kalmanPropagateAndUpdateAndChi2Plane: helix propagation to the module
//      plane with the parametrised field and material, plane-local update), material skipped on the first hit;
//      passes = 3: forward, backward, forward again with the errors scaled by 100 between passes (mkFit final-fit
//      style), so the result forgets the 3-hit estimate;
//   4. the state at the last hit (the seed's last hit, as the host seed) replaces params / errors / charge.
// originPrior = kHostCreator (3) emulates the STRUCTURE of the host creator
// SeedFromConsecutiveHitsCreator::makeSeed with the LSTOutputConverter's default GlobalTrackingRegion (fitHostLike):
//   initialKinematic: FastHelix(hit 1, hit 0, region origin (0, 0, 0)), nominal field 3.8 T, 0.3 GeV/(T m), state at
//     the origin; initialError: diagonal curvilinear (q/p, lambda, phi, xT, yT) prior of the region (ptMin 1,
//     MinOneOverPtError 1, r 0.2 cm, half-length 22.7 cm) rotated to mkFit's CCS; ONE forward KF pass over every
//     seed hit: propagation to the module plane (mkFit helix, the refit field flags and material, applied at every
//     destination incl. the first hit, as PropagatorWithMaterial), checkHit (true: no seed comparitor), update; the
//     state at the last hit. Rejections as the host's: a step with |dphi| > MaxDPhi (1.6 rad, PropagatorWithMaterial),
//     a step against the momentum (alongMomentum propagator), a non-finite / non-positive state.

#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/LstSeedFit.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/fit/FitKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds {

  using ::mkfitdev::ESView;
  using ::mkfitdev::HitSoAConstView;
  using ::mkfitdev::SeedSoAView;
  using namespace ::mkfitdev::lstseeds;

  struct LstSeedHitRow {
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static uint32_t row(ESView const& es, uint32_t nPixel, int layer, int index) {
      return (es.layers[layer].is_pixel() ? 0u : nPixel) + static_cast<uint32_t>(index);
    }
  };

  // Outcome of the host-structure fit (fitHostLike).
  enum LstHostLikeStatus : int { kHLOk = 0, kHLRejDPhi = 1, kHLRejBackward = 2, kHLRejInvalid = 3 };

  // initialKinematic + initialError of the host creator in slot `lane` of an N-wide state; false if the first two
  // seed hits are missing
  template <typename TAcc, idx_t N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hostLikeStart(TAcc const& acc,
                                                    ESView const& es,
                                                    HitSoAConstView hits,
                                                    uint32_t nPixel,
                                                    const ::mkfitdev::HitOnTrack* hot,
                                                    LstSeedFitConfig const& cfg,
                                                    const bool safePrior,
                                                    MPlexLV<N>& parA,
                                                    MPlexLS<N>& errA,
                                                    MPlexQI<N>& chg,
                                                    const int lane,
                                                    const double vxRegion,
                                                    const double vyRegion) {
    if (hot[0].index < 0 || hot[1].index < 0)
      return false;
    // ---- initialKinematic: FastHelix(outer = hit 1, middle = hit 0, vertex = region origin), double as the host
    const uint32_t r0 = LstSeedHitRow::row(es, nPixel, hot[0].layer, hot[0].index);
    const uint32_t r1 = LstSeedHitRow::row(es, nPixel, hot[1].layer, hot[1].index);
    const double mx = hits[r0].x(), my = hits[r0].y(), mz = hits[r0].z();  // middle
    const double ox = hits[r1].x(), oy = hits[r1].y(), oz = hits[r1].z();  // outer
    // region origin: GlobalTrackingRegion() (0, 0, 0) for the LST seeds; the proto track's vertex for the pixel seeds
    const double vx = vxRegion, vy = vyRegion;
    constexpr double kTesla0 = 3.8;  // 0.1 * MagneticField::nominalValue()
    constexpr double kCm2GeV = 0.01 * 0.3 * kTesla0;
    constexpr double kMaxPt = 10000.;
    constexpr double kMaxRho = kMaxPt / kCm2GeV;
    // FastCircle through (outer, middle, vertex): exact circle through three points (FastCircle's Riemann-sphere
    // construction is exact for three points; the direct form differs at rounding level)
    const double ax = mx - vx, ay = my - vy, bx = ox - vx, by = oy - vy;
    const double cr = ax * by - ay * bx;
    double x0 = 0., y0 = 0., rho = 0.;
    bool circle =
        alpaka::math::abs(acc, cr) > 1e-12 * (ax * ax + ay * ay) * (bx * bx + by * by) / (1. + ax * ax + ay * ay);
    if (circle) {
      const double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by;
      x0 = vx + (by * a2 - ay * b2) / (2. * cr);
      y0 = vy + (ax * b2 - bx * a2) / (2. * cr);
      rho = alpaka::math::sqrt(acc, (x0 - vx) * (x0 - vx) + (y0 - vy) * (y0 - vy));
    }
    double px, py, pz, zv;
    int q = 1;
    bool helix = circle && rho < kMaxRho;
    double dcphi = 0.;
    if (helix) {
      dcphi = ((ox - x0) * (mx - x0) + (oy - y0) * (my - y0)) / (rho * rho);
      helix = alpaka::math::abs(acc, dcphi) < 1.;
    }
    if (helix) {  // FastHelix::helixStateAtVertex
      const double pt = kCm2GeV * rho;
      px = -kCm2GeV * (vy - y0);
      py = kCm2GeV * (vx - x0);
      if (px * (mx - vx) + py * (my - vy) < 0.) {
        px = -px;
        py = -py;
      }
      const double dzdrphi = (oz - mz) / (rho * alpaka::math::acos(acc, dcphi));
      pz = pt * dzdrphi;
      if (x0 * py - y0 * px < 0)
        q = -q;
      zv = mz;
      double ds = ((vx - x0) * (mx - x0) + (vy - y0) * (my - y0)) / (rho * rho);
      if (alpaka::math::abs(acc, ds) < 1.) {
        ds = rho * alpaka::math::acos(acc, ds);
        zv -= ds * dzdrphi;
      } else {
        const double dmv2 = (mx - vx) * (mx - vx) + (my - vy) * (my - vy);
        const double dom2 = (ox - mx) * (ox - mx) + (oy - my) * (oy - my);
        zv -= alpaka::math::sqrt(acc, dmv2 / dom2) * (oz - mz);
      }
    } else {  // FastHelix::straightLineStateAtVertex (pT = 10 TeV along the vertex -> middle chord, charge +1)
      const double cl = alpaka::math::sqrt(acc, ax * ax + ay * ay);
      px = kMaxPt * ax / cl;
      py = kMaxPt * ay / cl;
      const double rm = alpaka::math::sqrt(acc, mx * mx + my * my), ro = alpaka::math::sqrt(acc, ox * ox + oy * oy);
      const double dzdr = (oz - mz) / (ro - rm);
      pz = kMaxPt * dzdr;
      zv = mz - rm * dzdr;
    }
    const double pt2 = px * px + py * py;
    const double ptv = alpaka::math::sqrt(acc, pt2);
    const double p = alpaka::math::sqrt(acc, pt2 + pz * pz);
    const float sinL = pz / p, cosL = ptv / p, sinF = py / ptv, cosF = px / ptv;
    const float ipt = 1. / ptv;
    parA.At(lane, 0, 0) = vx;
    parA.At(lane, 1, 0) = vy;
    parA.At(lane, 2, 0) = zv;
    parA.At(lane, 3, 0) = ipt;
    parA.At(lane, 4, 0) = alpaka::math::atan2(acc, py, px);
    parA.At(lane, 5, 0) = alpaka::math::atan2(acc, ptv, pz);
    chg.At(lane, 0, 0) = q;

    // ---- initialError: curvilinear diag(q/p, lambda, phi, xT, yT) -> CCS (x, y, z, 1/pT, phi, theta)
    //      J: x = -sinF xT - sinL cosF yT, y = cosF xT - sinL sinF yT, z = cosL yT,
    //         1/pT = q (q/p) / cosL (d/dlambda = 1/pT tanL), phi = phi, theta = pi/2 - lambda
    const float sin2th = cosL * cosL;
    float c00 =
        alpaka::math::max(acc, sin2th / (cfg.hlPtMin * cfg.hlPtMin), cfg.hlMinOneOverPtErr * cfg.hlMinOneOverPtErr);
    float cL = 1.f, cF = 1.f;       // "no good reason. no bad reason...."
    const float cT = cfg.originR2;  // (OriginTransverseErrorMultiplier * originRBound)^2
    float cY = cfg.originZ2 * sin2th + cT * (1.f - sin2th);
    if (safePrior) {
      // retry after a float failure (the host does this in double): the uninformative parts of the prior capped so
      // that one update never needs > ~1e5 of dynamic range: q/p sigma (1/pT + 0.01) sin(theta), angles 0.1 rad,
      // yT 5 cm; the informative transverse 0.2 cm is kept
      const float sq = (ipt + 0.01f) * cosL;
      c00 = alpaka::math::min(acc, c00, sq * sq);
      cL = cfg.hlSafeAngVar;
      cF = cfg.hlSafeAngVar;
      cY = alpaka::math::min(acc, cY, cfg.hlSafeYTVar);
    }
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j)
        errA.At(lane, i, j) = 0.f;
    errA.At(lane, 0, 0) = sinF * sinF * cT + sinL * sinL * cosF * cosF * cY;
    errA.At(lane, 1, 0) = -sinF * cosF * cT + sinL * sinL * sinF * cosF * cY;
    errA.At(lane, 1, 1) = cosF * cosF * cT + sinL * sinL * sinF * sinF * cY;
    errA.At(lane, 2, 0) = -sinL * cosL * cosF * cY;
    errA.At(lane, 2, 1) = -sinL * cosL * sinF * cY;
    errA.At(lane, 2, 2) = cosL * cosL * cY;
    const float dIptdL = ipt * sinL / cosL;
    errA.At(lane, 3, 3) = c00 / (cosL * cosL) + dIptdL * dIptdL * cL;
    errA.At(lane, 4, 4) = cF;
    errA.At(lane, 5, 3) = -dIptdL * cL;
    errA.At(lane, 5, 5) = cL;
    return true;
  }

  // propagation flags of the host-structure fit
  ALPAKA_FN_ACC ALPAKA_FN_INLINE PropagationFlags hostLikeFlags(ESView const& es, LstSeedFitConfig const& cfg) {
    PropagationFlags pf = ::ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::fit::refitKernelFlags(es);
    if (cfg.passes == 1) {
      // field model as the host's AnalyticalPropagator: B at the START of each step, no radial-field kick
      // (the runtime field constants stay); passes = 3 (default) = the final fit's refitKernelFlags (mid-point B + kick)
      pf.b_field_at_mid = false;
      pf.radial_field_corr = false;
    }
    if (es.config->refit.elossSignFromPass) {
      pf.eloss_by_pass = true;
      pf.eloss_outward = true;  // along the momentum: energy lost
    }
    return pf;
  }

  // option (b'): SeedFromConsecutiveHitsCreator::makeSeed (initialKinematic + initialError + buildSeed) with
  // mkFit's propagation / material model. On kHLOk par/err/chg hold the updated state at the last seed hit.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int fitHostLike(TAcc const& acc,
                                                 ESView const& es,
                                                 HitSoAConstView hits,
                                                 uint32_t nPixel,
                                                 const ::mkfitdev::HitOnTrack* hot,
                                                 const int nh,
                                                 LstSeedFitConfig const& cfg,
                                                 const bool safePrior,
                                                 MPlexLV<1>& parA,
                                                 MPlexLS<1>& errA,
                                                 MPlexQI<1>& chg,
                                                 const double vxRegion = 0.,
                                                 const double vyRegion = 0.) {
    if (!hostLikeStart(acc, es, hits, nPixel, hot, cfg, safePrior, parA, errA, chg, 0, vxRegion, vyRegion))
      return kHLRejInvalid;

    // ---- buildSeed: one forward pass, propagate (with material at the destination) + checkHit + update per hit
    const ::mkfitdev::RefitConfig rc = es.config->refit;
    const PropagationFlags pf = hostLikeFlags(es, cfg);
    PropagationFlags pfNoMat = pf;
    pfNoMat.apply_material = false;
    MPlexLS<1> errP, errQ;
    MPlexLV<1> parP, parQ;
    MPlexQI<1> failFlag, noMat;
    MPlexQF<1> outChi2, matRadl, matBbxi;
    MPlexHS<1> msErr;
    MPlexHV<1> msPar, norm, dir, pnt;
    noMat.At(0, 0, 0) = 0;
    const MPlexQF<1>* matRadlPtr = rc.materialPerModule ? &matRadl : nullptr;
    const MPlexQF<1>* matBbxiPtr = rc.materialPerModule ? &matBbxi : nullptr;
    for (int h = 0; h < nh; ++h) {
      if (hot[h].index < 0)
        continue;
      const int layer = hot[h].layer;
      const uint32_t r = LstSeedHitRow::row(es, nPixel, layer, hot[h].index);
      msPar.At(0, 0, 0) = hits[r].x();
      msPar.At(0, 1, 0) = hits[r].y();
      msPar.At(0, 2, 0) = hits[r].z();
      msErr.At(0, 0, 0) = hits[r].e00();
      msErr.At(0, 1, 0) = hits[r].e10();
      msErr.At(0, 1, 1) = hits[r].e11();
      msErr.At(0, 2, 0) = hits[r].e20();
      msErr.At(0, 2, 1) = hits[r].e21();
      msErr.At(0, 2, 2) = hits[r].e22();
      const int mod = es.moduleRow(layer, ::mkfitdev::hitpack::detIDinLayer(hits[r].packed()));
      const auto mi = es.modules[mod];
      norm.At(0, 0, 0) = mi.zdir_x();
      norm.At(0, 1, 0) = mi.zdir_y();
      norm.At(0, 2, 0) = mi.zdir_z();
      dir.At(0, 0, 0) = mi.xdir_x();
      dir.At(0, 1, 0) = mi.xdir_y();
      dir.At(0, 2, 0) = mi.xdir_z();
      pnt.At(0, 0, 0) = mi.pos_x();
      pnt.At(0, 1, 0) = mi.pos_y();
      pnt.At(0, 2, 0) = mi.pos_z();
      matRadl.At(0, 0, 0) = mi.radl();
      matBbxi.At(0, 0, 0) = mi.bbxi();
      failFlag.At(0, 0, 0) = 0;
      const MPlexLV<1>* src = &parA;
      const MPlexLS<1>* srcE = &errA;
      if (h == 0) {
        // the long origin -> first-hit step: a geometry-only step to the plane first (mkFit's plane propagation does
        // nSStepsInProp2Plane path refinements), then the material step from there (Jacobians chain)
        propagateHelixToPlaneMPlex<1>(errA, parA, chg, pnt, norm, errQ, parQ, failFlag, 1, pfNoMat, &noMat);
        if (failFlag.At(0, 0, 0))
          return kHLRejInvalid;
        src = &parQ;
        srcE = &errQ;
      }
      propagateHelixToPlaneMPlex<1>(
          *srcE, *src, chg, pnt, norm, errP, parP, failFlag, 1, pf, &noMat, matRadlPtr, matBbxiPtr, nullptr);
      if (failFlag.At(0, 0, 0))
        return kHLRejInvalid;
      // AnalyticalPropagator: |dphi| of the step > MaxDPhi fails; PropagationDirection alongMomentum
      const float dphi = ::mkfitdev::squashPhiGeneral(parP.At(0, 4, 0) - parA.At(0, 4, 0));
      if (alpaka::math::abs(acc, dphi) > cfg.hlMaxDPhi)
        return kHLRejDPhi;
      const float st = alpaka::math::sin(acc, parA.At(0, 5, 0));
      const float dx = parP.At(0, 0, 0) - parA.At(0, 0, 0), dy = parP.At(0, 1, 0) - parA.At(0, 1, 0),
                  dz = parP.At(0, 2, 0) - parA.At(0, 2, 0);
      const float fwd = dx * alpaka::math::cos(acc, parA.At(0, 4, 0)) * st +
                        dy * alpaka::math::sin(acc, parA.At(0, 4, 0)) * st +
                        dz * alpaka::math::cos(acc, parA.At(0, 5, 0));
      if (fwd < -cfg.hlBackwardTol)
        return kHLRejBackward;
      // checkHit: SeedFromConsecutiveHitsCreator::checkHit = filter->compatible or true; the LST converter passes no
      // seed comparitor -> true. KFUpdator: the plane-local update of the propagated state.
      kalmanPropagateAndUpdateAndChi2Plane<1>(
          errP, parP, chg, msErr, msPar, norm, dir, pnt, errA, parA, failFlag, outChi2, 1, pf, false, &noMat);
      // KFUpdator returns an invalid state only when R = V + H C H^T cannot be inverted: per step only finite
      // parameters are required (as the device final fit); the covariance is checked once, at the end
      bool ok = true;
      for (int k = 0; k < 6; ++k)
        ok = ok && ::mkfitdev::isFinite(parA.At(0, k, 0));
      if (!ok)
        return kHLRejInvalid;
    }
    bool ok = true;
    for (int k = 0; k < 6; ++k)
      ok = ok && ::mkfitdev::isFinite(errA.At(0, k, k)) && errA.At(0, k, k) > 0.f;
    return ok ? kHLOk : kHLRejInvalid;
  }

  // the host-structure seed after its first fit (status st, state par / err / chg): the capped-prior retry, the
  // pixel-seed fallback of failed pT3 / pT5 seeds, the seed row
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void lstSeedHostLikeStore(Acc1D const& acc,
                                                           ESView const& es,
                                                           HitSoAConstView hits,
                                                           uint32_t nPixel,
                                                           SeedSoAView seeds,
                                                           const int32_t s,
                                                           const ::mkfitdev::HitOnTrack* hot,
                                                           const int nh,
                                                           const int nOT,
                                                           LstSeedFitConfig const& cfg,
                                                           PixelSeedRows const& pix,
                                                           LstSeedFitCounters* cnt,
                                                           int8_t* stOut,
                                                           int st,
                                                           MPlexLV<1>& parH,
                                                           MPlexLS<1>& errH,
                                                           MPlexQI<1>& chgH) {
    if (st == kHLRejInvalid) {  // float failure of the exact prior: retry with the capped prior
      alpaka::atomicAdd(acc, &cnt->nSafeRetry, 1, alpaka::hierarchy::Blocks{});
      st = fitHostLike(acc, es, hits, nPixel, hot, nh, cfg, true, parH, errH, chgH);
    }
    int8_t status = kStOk;
    if (st != kHLOk) {
      alpaka::atomicAdd(acc, &cnt->nFailed, 1, alpaka::hierarchy::Blocks{});
      if (st == kHLRejDPhi)
        alpaka::atomicAdd(acc, &cnt->nRejDPhi, 1, alpaka::hierarchy::Blocks{});
      else if (st == kHLRejBackward)
        alpaka::atomicAdd(acc, &cnt->nRejBackward, 1, alpaka::hierarchy::Blocks{});
      status = st == kHLRejDPhi ? kStRejDPhi : (st == kHLRejBackward ? kStRejBackward : kStRejInvalid);
      if (!cfg.dropFailed) {  // the input (host creator) state is kept
        if (stOut)
          stOut[s] = status;
        return;
      }
      // light seeds (placeholder input state): emulate the host. T5/T4 (no pixel track): the host skips the TC ->
      // drop. pT3/pT5: the host keeps the pixel seed (seed = *pixelSeed) -> the pixel track's seed hits and state.
      const int32_t e = pix.pixIdx != nullptr ? pix.pixIdx[s] : -1;
      const bool hasPixelSeed = cfg.hlPixelFallback && e >= 0 && e < pix.nStates;
      const bool fallback = hasPixelSeed && pix.states[e].charge() != 0 && pix.states[e].nTotalHits() >= 2;
      if (hasPixelSeed) {
        alpaka::atomicAdd(acc, fallback ? &cnt->nFallback : &cnt->nFallbackFailed, 1, alpaka::hierarchy::Blocks{});
        status = fallback ? kStFallbackOk : kStFallbackFailed;
      } else if (nOT == nh) {
        status = kStDroppedOT;
      }
      if (stOut)
        stOut[s] = status;
      if (!fallback) {
        seeds[s].errors().v[0] = std::numeric_limits<float>::quiet_NaN();  // seed_post_cleaning removes it
        return;
      }
      // the pixel seed's hits as the hand-off writes a pLS row (capped at the seed capacity)
      const auto ps = pix.states[e];
      const int np = ps.nTotalHits() < ::mkfitdev::kMaxSeedHits ? ps.nTotalHits() : ::mkfitdev::kMaxSeedHits;
      for (int k = 0; k < np; ++k)
        seeds[s].hits().hot[k] = ps.hits().hot[k];
      seeds[s].nHits() = static_cast<int16_t>(np);
      const uint32_t rl = LstSeedHitRow::row(es, nPixel, ps.hits().hot[np - 1].layer, ps.hits().hot[np - 1].index);
      seeds[s].lastX() = hits[rl].x();
      seeds[s].lastY() = hits[rl].y();
      seeds[s].lastZ() = hits[rl].z();
      // DEVIATION DEV-7: the pixel seed state from the device creator emulation (as the pLS seeds)
      seeds[s].params() = ps.params();
      seeds[s].errors() = ps.errors();
      seeds[s].charge() = ps.charge();
      return;
    }
    if (stOut)
      stOut[s] = status;
    // DEVIATION DEV-2: seed state from the device fit of the seed hits (the menu: the CMSSW seed creator on the
    // host)
    if (cfg.errScale != 1.f)
      errH.scale(cfg.errScale);
    for (int k = 0; k < 6; ++k)
      seeds[s].params().v[k] = parH.At(0, k, 0);
    errH.copyOut(0, seeds[s].errors().v);
    if (chgH.At(0, 0, 0) != seeds[s].charge())
      alpaka::atomicAdd(acc, &cnt->nChargeFlip, 1, alpaka::hierarchy::Blocks{});
    seeds[s].charge() = static_cast<int16_t>(chgH.At(0, 0, 0));
    alpaka::atomicAdd(acc, &cnt->nFitted, 1, alpaka::hierarchy::Blocks{});
  }

  class KernelLstSeedFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ESView es,
                                  HitSoAConstView hits,
                                  uint32_t nPixel,
                                  SeedSoAView seeds,
                                  int32_t n,
                                  LstSeedFitConfig cfg,
                                  PixelSeedRows pix,
                                  LstSeedFitCounters* cnt,
                                  int8_t* cls,
                                  int8_t* stOut) const {
      const PropagationFlags pf(PF_use_param_b_field | PF_apply_material, es.material);
      for (int32_t s : cms::alpakatools::uniform_elements(acc, n)) {
        const int nh = seeds[s].nHits();
        const auto& hot = seeds[s].hits().hot;
        int nOT = 0;
        for (int h = 0; h < nh; ++h)
          if (hot[h].index >= 0 && !es.layers[hot[h].layer].is_pixel())
            ++nOT;
        if (nOT == 0) {
          if (cls)
            cls[s] = kClsPixelOnly;
          alpaka::atomicAdd(acc, &cnt->nPixelOnly, 1, alpaka::hierarchy::Blocks{});
          continue;
        }
        if (cfg.originPrior == kHostCreator) {
          if (nh < 2) {
            if (cls)
              cls[s] = kClsTooFew;
            alpaka::atomicAdd(acc, &cnt->nTooFewHits, 1, alpaka::hierarchy::Blocks{});
            continue;
          }
          if (cls)
            cls[s] = (nOT == nh) ? kClsOTOnly : kClsMixed;
          MPlexLV<1> parH;
          MPlexLS<1> errH;
          MPlexQI<1> chgH;
          int st = fitHostLike(acc, es, hits, nPixel, hot, nh, cfg, false, parH, errH, chgH);
          lstSeedHostLikeStore(
              acc, es, hits, nPixel, seeds, s, hot, nh, nOT, cfg, pix, cnt, stOut, st, parH, errH, chgH);
          continue;
        }
        if (nh < 3) {
          if (cls)
            cls[s] = kClsTooFew;
          alpaka::atomicAdd(acc, &cnt->nTooFewHits, 1, alpaka::hierarchy::Blocks{});
          continue;
        }
        if (cls)
          cls[s] = (nOT == nh) ? kClsOTOnly : kClsMixed;

        // 2. initial helix from the first, middle and last hit
        const uint32_t r0 = LstSeedHitRow::row(es, nPixel, hot[0].layer, hot[0].index);
        const uint32_t r1 = LstSeedHitRow::row(es, nPixel, hot[nh / 2].layer, hot[nh / 2].index);
        const uint32_t r2 = LstSeedHitRow::row(es, nPixel, hot[nh - 1].layer, hot[nh - 1].index);
        const float x0 = hits[r0].x(), y0 = hits[r0].y(), z0 = hits[r0].z();
        const float ax = hits[r1].x() - x0, ay = hits[r1].y() - y0;
        const float bx = hits[r2].x() - x0, by = hits[r2].y() - y0;
        const float cross = ax * by - ay * bx;
        const float a2 = ax * ax + ay * ay, b2 = bx * bx + by * by;
        float invPt, phi;
        int charge;
        const float kPtPerCm = ::mkfitdev::Const::sol_over_100 * ::mkfitdev::Config::Bfield;  // pT = k R
        if (alpaka::math::abs(acc, cross) < 1e-6f * a2 * b2 / (1.f + a2 + b2) || cross == 0.f) {
          // straight in xy: 1/pT ~ 0, charge +1 (the fit may flip it)
          invPt = 1e-3f;
          charge = 1;
          phi = alpaka::math::atan2(acc, by, bx);
        } else {
          const float d = 2.f * cross;
          const float ux = (by * a2 - ay * b2) / d, uy = (ax * b2 - bx * a2) / d;  // centre relative to hit 0
          const float R = alpaka::math::sqrt(acc, ux * ux + uy * uy);
          invPt = 1.f / (kPtPerCm * R);
          charge = cross > 0.f ? -1 : 1;  // counter-clockwise in xy = negative for Bz > 0
          // tangent at hit 0, perpendicular to the radius (hit0 - centre) = (-ux, -uy), oriented toward hit 1
          float tx = uy, ty = -ux;
          if (tx * ax + ty * ay < 0.f) {
            tx = -tx;
            ty = -ty;
          }
          phi = alpaka::math::atan2(acc, ty, tx);
        }
        // theta from the transverse arc length between hit 0 and the last hit
        const float chord = alpaka::math::sqrt(acc, b2);
        float sArc = chord;
        if (invPt > 1e-3f) {
          const float R = 1.f / (kPtPerCm * invPt);
          const float q = chord / (2.f * R);
          sArc = 2.f * R * alpaka::math::asin(acc, q < 1.f ? q : 1.f);
        }
        const float theta = alpaka::math::atan2(acc, sArc, hits[r2].z() - z0);

        // origin prior (host seed creator style): circle through the beam line point (0, 0), hit 0 and the last hit;
        // the KF starts AT the beam line with the creator's prior and material is applied on every hit
        const bool usePrior = (cfg.originPrior == 1 && nOT == nh) || cfg.originPrior == 2;
        float priorPar[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        int priorChg = 1;
        if (usePrior) {
          const float px = x0, py = y0, qx = hits[r2].x(), qy = hits[r2].y();
          const float p2 = px * px + py * py, q2 = qx * qx + qy * qy;
          const float cr = px * qy - py * qx;
          float ipt = 1e-3f, ph = alpaka::math::atan2(acc, py, px);
          float s0 = alpaka::math::sqrt(acc, p2), sl = alpaka::math::sqrt(acc, q2);
          if (alpaka::math::abs(acc, cr) > 1e-6f * p2 * q2 / (1.f + p2 + q2) && cr != 0.f) {
            const float d = 2.f * cr;
            const float ux = (qy * p2 - py * q2) / d, uy = (px * q2 - qx * p2) / d;  // centre relative to the origin
            const float R = alpaka::math::sqrt(acc, ux * ux + uy * uy);
            ipt = 1.f / (kPtPerCm * R);
            priorChg = cr > 0.f ? -1 : 1;
            float tx = uy, ty = -ux;
            if (tx * px + ty * py < 0.f) {
              tx = -tx;
              ty = -ty;
            }
            ph = alpaka::math::atan2(acc, ty, tx);
            const float a0 = s0 / (2.f * R), al = sl / (2.f * R);
            s0 = 2.f * R * alpaka::math::asin(acc, a0 < 1.f ? a0 : 1.f);
            sl = 2.f * R * alpaka::math::asin(acc, al < 1.f ? al : 1.f);
          }
          const float ds = (sl - s0) > 1e-3f ? (sl - s0) : 1e-3f;
          const float cotT = (hits[r2].z() - z0) / ds;
          priorPar[2] = z0 - s0 * cotT;
          priorPar[3] = ipt;
          priorPar[4] = ph;
          priorPar[5] = alpaka::math::atan2(acc, 1.f, cotT);
        }

        MPlexLS<1> errA, errB;
        MPlexLV<1> parA, parB;
        MPlexQI<1> chg, failFlag, noMat;
        MPlexQF<1> outChi2;
        MPlexHS<1> msErr;
        MPlexHV<1> msPar, norm, dir, pnt;
        parA.At(0, 0, 0) = x0;
        parA.At(0, 1, 0) = y0;
        parA.At(0, 2, 0) = z0;
        parA.At(0, 3, 0) = invPt;
        parA.At(0, 4, 0) = phi;
        parA.At(0, 5, 0) = theta;
        for (int i = 0; i < 6; ++i)
          for (int j = 0; j <= i; ++j)
            errA.At(0, i, j) = 0.f;
        errA.At(0, 0, 0) = cfg.posVar;
        errA.At(0, 1, 1) = cfg.posVar;
        errA.At(0, 2, 2) = cfg.posVar;
        const float sIpt = cfg.relInvPtErr * invPt + cfg.invPtErrFloor;
        errA.At(0, 3, 3) = sIpt * sIpt;
        errA.At(0, 4, 4) = cfg.angVar;
        errA.At(0, 5, 5) = cfg.angVar;
        chg.At(0, 0, 0) = charge;
        failFlag.At(0, 0, 0) = 0;
        if (usePrior) {
          for (int k = 0; k < 6; ++k)
            parA.At(0, k, 0) = priorPar[k];
          errA.At(0, 0, 0) = cfg.originR2;
          errA.At(0, 1, 1) = cfg.originR2;
          errA.At(0, 2, 2) = cfg.originZ2;
          errA.At(0, 3, 3) = 1.f;  // host: C(q/p) >= MinOneOverPtError^2 = 1, angles 1 rad^2
          errA.At(0, 4, 4) = 1.f;
          errA.At(0, 5, 5) = 1.f;
          chg.At(0, 0, 0) = priorChg;
        }

        // 3. Kalman passes over the seed hits (pass 0 forward, pass 1 backward, pass 2 forward, ...)
        const int nPass = cfg.passes >= 3 ? 3 : 1;
        bool failed = false;
        float chi2 = 0.f;
        for (int p = 0; p < nPass && !failed; ++p) {
          const bool bkw = (p & 1);
          if (p > 0)
            errA.scale(100.0f);
          chi2 = 0.f;
          for (int h = 0; h < nh; ++h) {
            const int m = bkw ? nh - 1 - h : h;
            if (hot[m].index < 0)
              continue;
            const int layer = hot[m].layer;
            const uint32_t r = LstSeedHitRow::row(es, nPixel, layer, hot[m].index);
            msPar.At(0, 0, 0) = hits[r].x();
            msPar.At(0, 1, 0) = hits[r].y();
            msPar.At(0, 2, 0) = hits[r].z();
            msErr.At(0, 0, 0) = hits[r].e00();
            msErr.At(0, 1, 0) = hits[r].e10();
            msErr.At(0, 1, 1) = hits[r].e11();
            msErr.At(0, 2, 0) = hits[r].e20();
            msErr.At(0, 2, 1) = hits[r].e21();
            msErr.At(0, 2, 2) = hits[r].e22();
            const int mod = es.moduleRow(layer, ::mkfitdev::hitpack::detIDinLayer(hits[r].packed()));
            const auto mi = es.modules[mod];
            norm.At(0, 0, 0) = mi.zdir_x();
            norm.At(0, 1, 0) = mi.zdir_y();
            norm.At(0, 2, 0) = mi.zdir_z();
            dir.At(0, 0, 0) = mi.xdir_x();
            dir.At(0, 1, 0) = mi.xdir_y();
            dir.At(0, 2, 0) = mi.xdir_z();
            pnt.At(0, 0, 0) = mi.pos_x();
            pnt.At(0, 1, 0) = mi.pos_y();
            pnt.At(0, 2, 0) = mi.pos_z();
            noMat.At(0, 0, 0) = (h == 0 && !(usePrior && p == 0)) ? 1 : 0;
            kalmanPropagateAndUpdateAndChi2Plane<1>(
                errA, parA, chg, msErr, msPar, norm, dir, pnt, errB, parB, failFlag, outChi2, 1, pf, true, &noMat);
            if (failFlag.At(0, 0, 0)) {
              failed = true;
              break;
            }
            errA = errB;
            parA = parB;
            chi2 += outChi2.At(0, 0, 0);
          }
        }
        bool finite = (chi2 == chi2);
        for (int k = 0; k < 6; ++k)
          finite = finite && ::mkfitdev::isFinite(parA.At(0, k, 0)) && errA.At(0, k, k) > 0.f;
        if (failed || !finite) {
          alpaka::atomicAdd(acc, &cnt->nFailed, 1, alpaka::hierarchy::Blocks{});
          if (cfg.dropFailed)
            seeds[s].errors().v[0] = std::numeric_limits<float>::quiet_NaN();  // seed_post_cleaning removes it
          continue;
        }
        // DEVIATION DEV-2: seed state from the device mkFit fit of the seed hits (the menu: the CMSSW seed creator on
        // the host) 4. state at the last hit -> the seed row
        if (cfg.errScale != 1.f)
          errA.scale(cfg.errScale);
        for (int k = 0; k < 6; ++k)
          seeds[s].params().v[k] = parA.At(0, k, 0);
        errA.copyOut(0, seeds[s].errors().v);
        if (chg.At(0, 0, 0) != seeds[s].charge())
          alpaka::atomicAdd(acc, &cnt->nChargeFlip, 1, alpaka::hierarchy::Blocks{});
        seeds[s].charge() = static_cast<int16_t>(chg.At(0, 0, 0));
        alpaka::atomicAdd(acc, &cnt->nFitted, 1, alpaka::hierarchy::Blocks{});
      }
    }
  };

#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED))
  // CPU: fitHostLike for nl seeds with nh valid hits each, one per slot of an N-wide Matriplex (as the CPU final fit);
  // same arithmetic per slot. Slots nl..N-1 repeat slot 0. st[i]: the LstHostLikeStatus of slot i.
  template <idx_t N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fitHostLikeGroup(Acc1D const& acc,
                                                       ESView const& es,
                                                       HitSoAConstView hits,
                                                       uint32_t nPixel,
                                                       const ::mkfitdev::HitOnTrack* const* hot,
                                                       const int nh,
                                                       const int nl,
                                                       LstSeedFitConfig const* cfg,
                                                       const double* vx,
                                                       const double* vy,
                                                       MPlexLV<N>& parA,
                                                       MPlexLS<N>& errA,
                                                       MPlexQI<N>& chg,
                                                       int* st) {
    bool active[N];
    for (int i = 0; i < N; ++i) {
      const int l = i < nl ? i : 0;
      active[i] = hostLikeStart(acc, es, hits, nPixel, hot[l], cfg[l], false, parA, errA, chg, i, vx[l], vy[l]);
      st[i] = active[i] ? kHLOk : kHLRejInvalid;
    }
    const PropagationFlags pf = hostLikeFlags(es, cfg[0]);
    PropagationFlags pfNoMat = pf;
    pfNoMat.apply_material = false;
    MPlexLS<N> errP, errQ;
    MPlexLV<N> parP, parQ;
    MPlexQI<N> failFlag, noMat(0);
    MPlexQF<N> outChi2, matRadl, matBbxi;
    MPlexHS<N> msErr;
    MPlexHV<N> msPar, norm, dir, pnt;
    const bool perModule = es.config->refit.materialPerModule;
    const MPlexQF<N>* matRadlPtr = perModule ? &matRadl : nullptr;
    const MPlexQF<N>* matBbxiPtr = perModule ? &matBbxi : nullptr;
    auto stop = [&](int i, int status) {
      active[i] = false;
      st[i] = status;
    };
    for (int h = 0; h < nh; ++h) {
      for (int i = 0; i < N; ++i) {
        const ::mkfitdev::HitOnTrack ht = hot[i < nl ? i : 0][h];
        const uint32_t r = LstSeedHitRow::row(es, nPixel, ht.layer, ht.index);
        msPar.At(i, 0, 0) = hits[r].x();
        msPar.At(i, 1, 0) = hits[r].y();
        msPar.At(i, 2, 0) = hits[r].z();
        msErr.At(i, 0, 0) = hits[r].e00();
        msErr.At(i, 1, 0) = hits[r].e10();
        msErr.At(i, 1, 1) = hits[r].e11();
        msErr.At(i, 2, 0) = hits[r].e20();
        msErr.At(i, 2, 1) = hits[r].e21();
        msErr.At(i, 2, 2) = hits[r].e22();
        const int mod = es.moduleRow(ht.layer, ::mkfitdev::hitpack::detIDinLayer(hits[r].packed()));
        const auto mi = es.modules[mod];
        norm.At(i, 0, 0) = mi.zdir_x();
        norm.At(i, 1, 0) = mi.zdir_y();
        norm.At(i, 2, 0) = mi.zdir_z();
        dir.At(i, 0, 0) = mi.xdir_x();
        dir.At(i, 1, 0) = mi.xdir_y();
        dir.At(i, 2, 0) = mi.xdir_z();
        pnt.At(i, 0, 0) = mi.pos_x();
        pnt.At(i, 1, 0) = mi.pos_y();
        pnt.At(i, 2, 0) = mi.pos_z();
        matRadl.At(i, 0, 0) = mi.radl();
        matBbxi.At(i, 0, 0) = mi.bbxi();
        failFlag.At(i, 0, 0) = 0;
      }
      const MPlexLV<N>* src = &parA;
      const MPlexLS<N>* srcE = &errA;
      if (h == 0) {
        // the origin -> first-hit step: geometry first, then the material step from the plane (as fitHostLike)
        propagateHelixToPlaneMPlex<N>(errA, parA, chg, pnt, norm, errQ, parQ, failFlag, N, pfNoMat, &noMat);
        for (int i = 0; i < nl; ++i)
          if (active[i] && failFlag.At(i, 0, 0))
            stop(i, kHLRejInvalid);
        src = &parQ;
        srcE = &errQ;
      }
      propagateHelixToPlaneMPlex<N>(
          *srcE, *src, chg, pnt, norm, errP, parP, failFlag, N, pf, &noMat, matRadlPtr, matBbxiPtr, nullptr);
      for (int i = 0; i < nl; ++i) {
        if (!active[i])
          continue;
        if (failFlag.At(i, 0, 0)) {
          stop(i, kHLRejInvalid);
          continue;
        }
        const float dphi = ::mkfitdev::squashPhiGeneral(parP.At(i, 4, 0) - parA.At(i, 4, 0));
        if (alpaka::math::abs(acc, dphi) > cfg[i].hlMaxDPhi) {
          stop(i, kHLRejDPhi);
          continue;
        }
        const float sth = alpaka::math::sin(acc, parA.At(i, 5, 0));
        const float dx = parP.At(i, 0, 0) - parA.At(i, 0, 0), dy = parP.At(i, 1, 0) - parA.At(i, 1, 0),
                    dz = parP.At(i, 2, 0) - parA.At(i, 2, 0);
        const float fwd = dx * alpaka::math::cos(acc, parA.At(i, 4, 0)) * sth +
                          dy * alpaka::math::sin(acc, parA.At(i, 4, 0)) * sth +
                          dz * alpaka::math::cos(acc, parA.At(i, 5, 0));
        if (fwd < -cfg[i].hlBackwardTol)
          stop(i, kHLRejBackward);
      }
      kalmanPropagateAndUpdateAndChi2Plane<N>(
          errP, parP, chg, msErr, msPar, norm, dir, pnt, errA, parA, failFlag, outChi2, N, pf, false, &noMat);
      int nActive = 0;
      for (int i = 0; i < nl; ++i) {
        if (!active[i])
          continue;
        bool ok = true;
        for (int k = 0; k < 6; ++k)
          ok = ok && ::mkfitdev::isFinite(parA.At(i, k, 0));
        if (ok)
          ++nActive;
        else
          stop(i, kHLRejInvalid);
      }
      if (nActive == 0)
        return;
    }
    for (int i = 0; i < nl; ++i) {
      if (!active[i])
        continue;
      bool ok = true;
      for (int k = 0; k < 6; ++k)
        ok = ok && ::mkfitdev::isFinite(errA.At(i, k, k)) && errA.At(i, k, k) > 0.f;
      st[i] = ok ? kHLOk : kHLRejInvalid;
    }
  }

  // slot i of an N-wide state into a one-slot state
  template <idx_t N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void hostLikeSlot(MPlexLV<N> const& parG,
                                                   MPlexLS<N> const& errG,
                                                   MPlexQI<N> const& chgG,
                                                   const int i,
                                                   MPlexLV<1>& par,
                                                   MPlexLS<1>& err,
                                                   MPlexQI<1>& chg) {
    for (int k = 0; k < 6; ++k) {
      par.At(0, k, 0) = parG.constAt(i, k, 0);
      for (int j = 0; j <= k; ++j)
        err.At(0, k, j) = errG.constAt(i, k, j);
    }
    chg.At(0, 0, 0) = chgG.constAt(i, 0, 0);
  }

  // true if one of the first nh hits is missing (such seeds take the one-seed fit)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hostLikeMissingHit(const ::mkfitdev::HitOnTrack* hot, const int nh) {
    for (int h = 0; h < nh; ++h)
      if (hot[h].index < 0)
        return true;
    return false;
  }

  // the OT hit count of a seed; pixel-only seeds are counted and kept as they are: -1
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int lstSeedOTHits(Acc1D const& acc,
                                                   ESView const& es,
                                                   const ::mkfitdev::HitOnTrack* hot,
                                                   const int nh,
                                                   const int32_t s,
                                                   LstSeedFitCounters* cnt,
                                                   int8_t* cls) {
    int nOT = 0;
    for (int h = 0; h < nh; ++h)
      if (hot[h].index >= 0 && !es.layers[hot[h].layer].is_pixel())
        ++nOT;
    if (nOT == 0) {
      if (cls)
        cls[s] = kClsPixelOnly;
      alpaka::atomicAdd(acc, &cnt->nPixelOnly, 1, alpaka::hierarchy::Blocks{});
      return -1;
    }
    return nOT;
  }

  // host-structure class of a seed with OT hits; seeds with fewer than two hits are counted and kept: false
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool lstSeedHostLikeClass(
      Acc1D const& acc, LstSeedFitCounters* cnt, int8_t* cls, const int32_t s, const int nh, const int nOT) {
    if (nh < 2) {
      if (cls)
        cls[s] = kClsTooFew;
      alpaka::atomicAdd(acc, &cnt->nTooFewHits, 1, alpaka::hierarchy::Blocks{});
      return false;
    }
    if (cls)
      cls[s] = (nOT == nh) ? kClsOTOnly : kClsMixed;
    return true;
  }

  // CPU, host-structure mode: the seeds of a block grouped by hit count into N-wide fits (KernelLstSeedFit per seed)
  template <idx_t N>
  class KernelLstSeedFitGrouped {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ESView es,
                                  HitSoAConstView hits,
                                  uint32_t nPixel,
                                  SeedSoAView seeds,
                                  int32_t n,
                                  LstSeedFitConfig cfg,
                                  PixelSeedRows pix,
                                  LstSeedFitCounters* cnt,
                                  int8_t* cls,
                                  int8_t* stOut) const {
      int32_t group[::mkfitdev::kMaxSeedHits + 1][N];
      int nOTs[::mkfitdev::kMaxSeedHits + 1][N];
      int fill[::mkfitdev::kMaxSeedHits + 1] = {};
      LstSeedFitConfig cfgs[N];
      double vx[N] = {}, vy[N] = {};
      for (int i = 0; i < N; ++i)
        cfgs[i] = cfg;
      auto fitGroup = [&](const int nh) {
        const int nl = fill[nh];
        const ::mkfitdev::HitOnTrack* hot[N];
        for (int i = 0; i < N; ++i)
          hot[i] = seeds[group[nh][i < nl ? i : 0]].hits().hot;
        MPlexLV<N> parG;
        MPlexLS<N> errG;
        MPlexQI<N> chgG;
        int st[N];
        fitHostLikeGroup<N>(acc, es, hits, nPixel, hot, nh, nl, cfgs, vx, vy, parG, errG, chgG, st);
        for (int i = 0; i < nl; ++i) {
          MPlexLV<1> parH;
          MPlexLS<1> errH;
          MPlexQI<1> chgH;
          hostLikeSlot(parG, errG, chgG, i, parH, errH, chgH);
          const int32_t s = group[nh][i];
          lstSeedHostLikeStore(acc,
                               es,
                               hits,
                               nPixel,
                               seeds,
                               s,
                               seeds[s].hits().hot,
                               nh,
                               nOTs[nh][i],
                               cfg,
                               pix,
                               cnt,
                               stOut,
                               st[i],
                               parH,
                               errH,
                               chgH);
        }
        fill[nh] = 0;
      };
      for (int32_t s : cms::alpakatools::uniform_elements(acc, n)) {
        const int nh = seeds[s].nHits();
        const auto& hot = seeds[s].hits().hot;
        const int nOT = lstSeedOTHits(acc, es, hot, nh, s, cnt, cls);
        if (nOT < 0 || !lstSeedHostLikeClass(acc, cnt, cls, s, nh, nOT))
          continue;
        if (hostLikeMissingHit(hot, nh)) {
          MPlexLV<1> parH;
          MPlexLS<1> errH;
          MPlexQI<1> chgH;
          const int st = fitHostLike(acc, es, hits, nPixel, hot, nh, cfg, false, parH, errH, chgH);
          lstSeedHostLikeStore(
              acc, es, hits, nPixel, seeds, s, hot, nh, nOT, cfg, pix, cnt, stOut, st, parH, errH, chgH);
          continue;
        }
        group[nh][fill[nh]] = s;
        nOTs[nh][fill[nh]] = nOT;
        if (++fill[nh] == N)
          fitGroup(nh);
      }
      for (int nh = 0; nh <= ::mkfitdev::kMaxSeedHits; ++nh)
        if (fill[nh] > 0)
          fitGroup(nh);
    }
  };
#endif

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds

#endif
