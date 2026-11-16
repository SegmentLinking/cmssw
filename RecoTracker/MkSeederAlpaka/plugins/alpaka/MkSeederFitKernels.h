#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederFitKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederFitKernels_h

// The seed fits on the device: the fit of the quads and triplets, the refit of seeds with more hits and the beam-line
// state. The propagation to a module plane (with the material of mkFit's (z, r) grid) and the Kalman update in the
// plane's local frame are the portable mkFit of RecoTracker/MkFitAlpaka (kalmanPropagateAndUpdateAndChi2Plane,
// propagateHelixToPlaneMPlex) with MkFitCore's backward-fit propagation flags. A work item fits kLanes seeds at once,
// the Matriplex width of the backend (mkfitdev::kNN: 1 on a GPU, 8 on a CPU backend); a lane without a seed, or whose
// seed has no more hits, computes on another lane's input and its result is not used.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpakaMath/interface/alpaka/prop/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitAlpakaMath/interface/alpaka/prop/PropagationMPlex.h"

#include "MkSeederBackend.h"
#include "MkSeederDeviceTypes.h"

// On the CPU backends a lane step (propagation, update) is flattened into one function: MkFitAlpaka's vdt helpers are
// then inlined into the per-slot loops, which vectorize over the kLanes seeds
#if defined(MKSEEDER_CPU_BUILD) && defined(__GNUC__) && !defined(__clang__)
#define MKSEEDER_FIT_FLATTEN __attribute__((flatten))
#else
#define MKSEEDER_FIT_FLATTEN
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::fit {

  namespace mkf = ::ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev;
  using PropagationFlags = ::mkfitdev::prop::PropagationFlags;
  using mkf::idx_t;

  // the seeds of a work item
  constexpr idx_t kLanes = mkf::kNN;

  // packed lower triangle of a symmetric matrix (mkfit::TrackState's covariance, mkfit::Hit's)
  ALPAKA_FN_HOST_ACC inline int sidx(int i, int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }

  // MkFitCore's backward-fit propagation flags (parametrized field, material) with the seeder's material grid and
  // mkfit::Config's field constants; withMaterial false: no material (the last step onto the hit's plane)
  ALPAKA_FN_HOST_ACC inline PropagationFlags propagationFlags(const FitInputs &F, bool withMaterial) {
    const FitParameters &P = F.parameters;
    PropagationFlags pf;
    pf.material = ::mkfitdev::MaterialView{F.material.metadata().addressOf_bbxi(),
                                           F.material.metadata().addressOf_radiationLength(),
                                           F.material.nBinsZ(),
                                           F.material.nBinsR(),
                                           F.material.binsPerCmZ(),
                                           F.material.binsPerCmR(),
                                           {P.fieldC1, P.fieldB0, P.fieldB1, P.fieldA}};
    pf.use_param_b_field = P.useParametrizedField;
    pf.apply_material = withMaterial && P.applyMaterial;
    return pf;
  }

  // the states of the seeds of a work item
  template <idx_t N>
  struct Lanes {
    mkf::MPlexLV<N> par;
    mkf::MPlexLS<N> err;
    mkf::MPlexQI<N> chg;
    ALPAKA_FN_HOST_ACC void load(int n, const float *p, const float *e, int q) {
      for (int d = 0; d < 6; ++d)
        par(n, d, 0) = p[d];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j)
          err.At(n, i, j) = e[sidx(i, j)];
      chg(n, 0, 0) = q;
    }
    ALPAKA_FN_HOST_ACC void store(int n, float *p, float *e, int &q) const {
      for (int d = 0; d < 6; ++d)
        p[d] = par.constAt(n, d, 0);
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j)
          e[sidx(i, j)] = err.constAt(n, i, j);
      q = chg.constAt(n, 0, 0);
    }
    ALPAKA_FN_HOST_ACC void copySlot(int n, const Lanes &o) {
      par.copySlot(n, o.par), err.copySlot(n, o.err), chg.copySlot(n, o.chg);
    }
    // lane n := lane m
    ALPAKA_FN_HOST_ACC void copyLane(int n, int m) {
      par.copyIn(n, par, m), err.copyIn(n, err, m), chg.copyIn(n, chg, m);
    }
    // finite parameters and diagonal; with kPositive also a positive diagonal from 1/pt on, with kFitStep also the
    // position's not below -1e-6
    static constexpr int kFinite = 0, kPositive = 1, kFitStep = 2;
    ALPAKA_FN_HOST_ACC bool finite(int n, int mode) const {
      bool ok = true;
      for (int d = 0; d < 6; ++d) {
        const float e = err.constAt(n, d, d);
        ok = ok && std::isfinite(par.constAt(n, d, 0)) && std::isfinite(e) &&
             (mode == kFinite || (d >= 3 ? e > 0.f : (mode != kFitStep || e >= -1e-6f)));
      }
      return ok;
    }
  };

  // the module planes of the hits h[n] (layers ly[n])
  template <idx_t N>
  ALPAKA_FN_HOST_ACC inline void modulePlanes(const FitInputs &F,
                                              const int16_t *ly,
                                              const uint32_t *h,
                                              mkf::MPlexHV<N> &pnt,
                                              mkf::MPlexHV<N> &nrm,
                                              mkf::MPlexHV<N> &dir) {
    for (int n = 0; n < N; ++n) {
      float m[9];
      modulePlane(F.modules, F.layerGeometry[ly[n]].firstModule() + F.hits.detIdInLayer(h[n]), m);
      for (int d = 0; d < 3; ++d)
        pnt(n, d, 0) = m[d], nrm(n, d, 0) = m[3 + d], dir(n, d, 0) = m[6 + d];
    }
  }

  // the propagation onto the hits h[n] and the update with them (kalmanPropagateAndUpdateAndChi2Plane, the charge
  // flipped where 1/pt turns negative); only the lanes with on[n] change
  template <idx_t N>
  MKSEEDER_FIT_FLATTEN ALPAKA_FN_HOST_ACC inline void propagateAndUpdate(const FitInputs &F,
                                                                         const PropagationFlags &pf,
                                                                         const int16_t *ly,
                                                                         const uint32_t *h,
                                                                         const bool *on,
                                                                         Lanes<N> &s,
                                                                         mkf::MPlexQF<N> &chi2) {
    mkf::MPlexHV<N> pnt, nrm, dir, msPar;
    mkf::MPlexHS<N> msErr;
    modulePlanes<N>(F, ly, h, pnt, nrm, dir);
    for (int n = 0; n < N; ++n) {
      msPar(n, 0, 0) = F.hits.x(h[n]), msPar(n, 1, 0) = F.hits.y(h[n]), msPar(n, 2, 0) = F.hits.z(h[n]);
      float c[6];
      hitCovariance(F.hits, h[n], c);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j)
          msErr.At(n, i, j) = c[sidx(i, j)];
    }
    Lanes<N> out;
    out.chg = s.chg;
    mkf::MPlexQI<N> fail;
    mkf::kalmanPropagateAndUpdateAndChi2Plane<N>(
        s.err, s.par, out.chg, msErr, msPar, nrm, dir, pnt, out.err, out.par, fail, chi2, N, pf, true);
    for (int n = 0; n < N; ++n)
      if (on[n])
        s.copySlot(n, out);
  }

  // SeedHelix(N): the circle through the first, a middle and the last of the n hits dh, the line z(s) through all; the
  // state at the first hit
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool computeHelixState(
      const TAcc &acc, const FitInputs &fitInputs, const uint32_t *dh, int n, float *par, int &chg) {
    auto X = [&](int k) { return fitInputs.hits.x(dh[k]); };
    auto Y = [&](int k) { return fitInputs.hits.y(dh[k]); };
    auto Z = [&](int k) { return fitInputs.hits.z(dh[k]); };
    auto R = [&](int k) { return alpaka::math::sqrt(acc, X(k) * X(k) + Y(k) * Y(k)); };
    const float ax = X(0), ay = Y(0), dx = X(n - 1), dy = Y(n - 1);
    const float rm = 0.5f * (R(0) + R(n - 1));
    int im = 1;
    for (int k = 2; k < n - 1; ++k)
      if (!(alpaka::math::abs(acc, R(im) - rm) < alpaka::math::abs(acc, R(k) - rm)))
        im = k;
    const float bx = X(im) - ax, by = Y(im) - ay, ex = dx - ax, ey = dy - ay;
    const float dd = 2.f * (bx * ey - by * ex);
    const float b2 = bx * bx + by * by, e2 = ex * ex + ey * ey;
    if (alpaka::math::abs(acc, dd) < 1e-9f * e2)
      return false;
    const float ux = (ey * b2 - by * e2) / dd, uy = (bx * e2 - ex * b2) / dd;
    const float cx = ax + ux, cy = ay + uy, Rc = std::hypot(ux, uy);
    const int turn = dd > 0 ? 1 : -1;
    float S = 0, Zs = 0, SS = 0, SZ = 0;
    for (int k = 0; k < n; ++k) {
      const float chord = std::hypot(X(k) - ax, Y(k) - ay);
      const float s = 2.f * Rc * alpaka::math::asin(acc, std::min(1.f, chord / (2.f * Rc)));
      S += s, Zs += Z(k), SS += s * s, SZ += s * Z(k);
    }
    const float fn = n;
    const float den = fn * SS - S * S;
    if (den <= 0)
      return false;
    const float cot = (fn * SZ - S * Zs) / den;
    if (!alpaka::math::isfinite(acc, cot) || !alpaka::math::isfinite(acc, Rc))
      return false;
    constexpr int kh = 0;
    const float rx = X(kh) - cx, ry = Y(kh) - cy;
    const float tx = turn > 0 ? -ry : ry, ty = turn > 0 ? rx : -rx;
    chg = turn > 0 ? -1 : 1;
    const float pt = ::mkfitdev::Const::sol_over_100 * fitInputs.parameters.bField * Rc;
    par[0] = X(kh), par[1] = Y(kh), par[2] = Z(kh);
    par[3] = 1.f / pt, par[4] = alpaka::math::atan2(acc, ty, tx);
    par[5] = ::mkfitdev::Const::PIOver2 - alpaka::math::atan(acc, cot);
    return true;
  }

  // the fit of up to N seeds (lane l: n[l] hits lay[l][k], dh[l][k] in fit order; n[l] = 0: no seed): the helix
  // state at the first hit with the prior covariance, then the propagation to and the update with each hit; a lane
  // stops at a step whose state is not finite. ok, par, err, chg, chi2 per lane with a seed
  template <idx_t N, int kMaxHits, typename TAcc>
  ALPAKA_FN_ACC inline void fitSeeds(const TAcc &acc,
                                     const FitInputs &F,
                                     const int16_t (*lay)[kMaxHits],
                                     const uint32_t (*dh)[kMaxHits],
                                     const int *n,
                                     bool *ok,
                                     float (*parOut)[6],
                                     float (*errOut)[21],
                                     int *chgOut,
                                     float *chi2Out) {
    const FitParameters &P = F.parameters;
    Lanes<N> s;
    bool live[N];
    int l0 = -1, nMax = 0;
    for (int l = 0; l < N; ++l) {
      float p[6] = {0, 0, 0, 0, 0, 0}, e[21];
      int q = 0;
      live[l] = n[l] > 0 && computeHelixState(acc, F, dh[l], n[l], p, q);
      ok[l] = live[l];
      for (int i = 0; i < 21; ++i)
        e[i] = 0.f;
      if (live[l]) {
        for (int a = 0; a < 6; ++a)
          e[sidx(a, a)] = (P.priorScale * P.priorSigma[a]) * (P.priorScale * P.priorSigma[a]);
        e[sidx(3, 3)] *= p[3] * p[3];
        nMax = std::max(nMax, n[l]);
        if (l0 < 0)
          l0 = l;
      }
      s.load(l, p, e, q);
    }
    if (l0 < 0)
      return;
    // the lanes without a fit compute on the first one's input
    for (int l = 0; l < N; ++l)
      if (!live[l])
        s.copyLane(l, l0);
    const PropagationFlags pf = propagationFlags(F, true);
    mkf::MPlexQF<N> chi2(0.f);
    for (int k = 0; k < nMax; ++k) {
      bool on[N];
      int16_t ly[N];
      uint32_t h[N];
      for (int l = 0; l < N; ++l) {
        on[l] = live[l] && k < n[l];
        const int ll = on[l] ? l : l0, kk = on[l] ? k : 0;
        ly[l] = lay[ll][kk], h[l] = dh[ll][kk];
      }
      mkf::MPlexQF<N> c2;
      propagateAndUpdate<N>(F, pf, ly, h, on, s, c2);
      for (int l = 0; l < N; ++l) {
        if (!on[l])
          continue;
        if (s.finite(l, Lanes<N>::kFitStep))
          chi2(l, 0, 0) += c2(l, 0, 0);
        else
          live[l] = false;
      }
    }
    for (int l = 0; l < N; ++l) {
      if (n[l] <= 0)
        continue;
      ok[l] = ok[l] && live[l];
      s.store(l, parOut[l], errOut[l], chgOut[l]);
      chi2Out[l] = chi2(l, 0, 0);
    }
  }

  // the propagations onto the last hit's plane that put the beam-line state on it to float precision (more solves of
  // the path than mkFit's two): the TrajectorySeed made of it then needs no propagation onto the module's surface
  // (MkSeederTrajectorySeedConverter propagateToSurface false)
  constexpr int kBeamlineLandings = 2;

  // the beam-line state of up to N seeds (n[l] >= 2 hits; 0: no seed) from their state at the last hit (par0, err0
  // with the error rescaled, chg0): the backward fit through the hits to the first one, then onto the last hit's plane.
  // ok and the state per lane with a seed
  template <idx_t N, int kMaxHits>
  MKSEEDER_FIT_FLATTEN ALPAKA_FN_HOST_ACC inline void beamlines(const FitInputs &F,
                                                                const int16_t (*lay)[kMaxHits],
                                                                const uint32_t (*dh)[kMaxHits],
                                                                const int *n,
                                                                const float (*par0)[6],
                                                                const float (*err0)[21],
                                                                const int *chg0,
                                                                bool *ok,
                                                                float (*parOut)[6],
                                                                float (*errOut)[21],
                                                                int *chgOut) {
    int l0 = 0;
    while (l0 < N && n[l0] <= 0)
      ++l0;
    if (l0 == N)
      return;
    Lanes<N> s;
    int nMax = 0;
    bool bad[N];
    for (int l = 0; l < N; ++l) {
      const int ll = n[l] > 0 ? l : l0;
      float e[21];
      for (int i = 0; i < 21; ++i)
        e[i] = err0[ll][i] * F.parameters.beamlineErrorRescale;
      s.load(l, par0[ll], e, chg0[ll]);
      nMax = std::max(nMax, n[l]);
      bad[l] = false;
    }
    const PropagationFlags pf = propagationFlags(F, true);
    for (int st = 0; st < nMax; ++st) {
      bool on[N];
      int16_t ly[N];
      uint32_t h[N];
      for (int l = 0; l < N; ++l) {
        on[l] = st < n[l];
        const int ll = n[l] > 0 ? l : l0, k = on[l] ? n[l] - 1 - st : 0;
        ly[l] = lay[ll][k], h[l] = dh[ll][k];
      }
      mkf::MPlexQF<N> c2;
      propagateAndUpdate<N>(F, pf, ly, h, on, s, c2);
      for (int l = 0; l < N; ++l)
        bad[l] = bad[l] || (on[l] && !s.finite(l, Lanes<N>::kFinite));
    }
    int16_t ly[N];
    uint32_t h[N];
    for (int l = 0; l < N; ++l) {
      const int ll = n[l] > 0 ? l : l0;
      ly[l] = lay[ll][n[ll] - 1], h[l] = dh[ll][n[ll] - 1];
    }
    mkf::MPlexHV<N> pnt, nrm, dir;
    modulePlanes<N>(F, ly, h, pnt, nrm, dir);
    const PropagationFlags pfNoMaterial = propagationFlags(F, false);
    for (int r = 0; r < kBeamlineLandings; ++r) {
      Lanes<N> out;
      out.chg = s.chg;
      mkf::MPlexQI<N> fail;
      mkf::propagateHelixToPlaneMPlex<N>(s.err, s.par, s.chg, pnt, nrm, out.err, out.par, fail, N, pfNoMaterial);
      s.par = out.par, s.err = out.err;
    }
    for (int l = 0; l < N; ++l) {
      if (n[l] <= 0)
        continue;
      ok[l] = !bad[l] && s.finite(l, Lanes<N>::kPositive);
      s.store(l, parOut[l], errOut[l], chgOut[l]);
    }
  }

  // the number of work items of n seeds
  ALPAKA_FN_HOST_ACC inline uint32_t laneGroups(uint32_t n) { return (n + kLanes - 1) / kLanes; }

  // the fit of the kept quads and triplet seeds: their hits in layer order
  // rows (nullptr: all): the kept seeds to fit, the first *nRows entries (a subset of the n, see KernelFitRowList); the
  // lanes take them in that order and write their row's state. Each lane's fit is independent of the other lanes'
  // (element-wise MPlex arithmetic), so a seed's fit does not depend on the seeds that share its lane group.
  class KernelFitQuads {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  FitInputs fitInputs,
                                  KeptSeedsConstView quads,
                                  SeedStatesView out,
                                  uint32_t n,
                                  const uint32_t *rows,
                                  const uint32_t *nRows) const {
      const uint32_t m = rows ? *nRows : n;
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, laneGroups(n))) {
        if (g * kLanes >= m)
          continue;
        int16_t lay[kLanes][4];
        uint32_t dh[kLanes][4];
        int nh[kLanes];
        uint32_t row[kLanes];
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          nh[l] = 0;
          if (i >= m)
            continue;
          row[l] = rows ? rows[i] : i;
          const auto q = quads[row[l]];
          nh[l] = q.nHits();
          for (int k = 0; k < nh[l]; ++k)
            lay[l][k] = q.layers()(k), dh[l][k] = q.hitRows()(k);
        }
        bool ok[kLanes];
        float par[kLanes][6], err[kLanes][21], chi2[kLanes];
        int chg[kLanes];
        fitSeeds<kLanes, 4>(acc, fitInputs, lay, dh, nh, ok, par, err, chg, chi2);
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          if (i >= m)
            break;
          out.fitOk(row[l]) = ok[l] ? 1 : 0;
          storeState(out, row[l], par[l], err[l], chg[l], chi2[l]);
        }
      }
    }
  };

  // the kept seeds to fit, the ones not flagged by skip (KernelMarkDroppedProbes): flags[k] 1 to fit; a skipped seed
  // gets fitOk 0 and a zero state
  class KernelFitRowFlags {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, const uint32_t *skip, uint32_t n, uint32_t *flags, SeedStatesView out) const {
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, n)) {
        flags[k] = skip[k] ? 0u : 1u;
        if (skip[k]) {
          const float p0[6] = {}, e0[21] = {};
          out.fitOk(k) = 0;
          storeState(out, k, p0, e0, 0, 0.f);
        }
      }
    }
  };
  // the list of the seeds to fit in their order (pos: the inclusive scan of the flags), *nRows its length
  class KernelFitRowList {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  const uint32_t *flags,
                                  const uint32_t *pos,
                                  uint32_t n,
                                  uint32_t *rows,
                                  uint32_t *nRows) const {
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, n)) {
        if (flags[k])
          rows[pos[k] - 1] = k;
        if (k == n - 1)
          *nRows = pos[k];
      }
    }
  };
  // per probe item (the first min(*nProbes, cap)) that the probe's extension did not keep: skip[its label] = 1
  class KernelMarkDroppedProbes {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, SeedHitsConstView probe, const uint32_t *nProbes, uint32_t cap, uint32_t *skip) const {
      const uint32_t np = *nProbes < cap ? *nProbes : cap;
      for (uint32_t p : cms::alpakatools::uniform_elements(acc, np))
        if (!probe.isKept(p))
          skip[probe.label(p)] = 1;
    }
  };

  // the seeds' hits of a work item (the first n of seeds; the lanes beyond: no hits)
  template <typename TSeedHits>
  ALPAKA_FN_HOST_ACC inline void loadLaneHits(const TSeedHits &seeds,
                                              uint32_t g,
                                              uint32_t n,
                                              int16_t (*lay)[::mkseeder::kMaxSeedHits],
                                              uint32_t (*dh)[::mkseeder::kMaxSeedHits],
                                              int *nh) {
    for (int l = 0; l < kLanes; ++l) {
      const uint32_t i = g * kLanes + l;
      nh[l] = i < n ? loadSeedHits(seeds, i, lay[l], dh[l]) : 0;
    }
  }

  // the refit of seeds: at least 4 hits
  class KernelFitSeeds {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, FitInputs fitInputs, SeedHitsConstView seeds, SeedStatesView out, uint32_t n) const {
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, laneGroups(n))) {
        int16_t lay[kLanes][::mkseeder::kMaxSeedHits];
        uint32_t dh[kLanes][::mkseeder::kMaxSeedHits];
        int nh[kLanes];
        loadLaneHits(seeds, g, n, lay, dh, nh);
        for (int l = 0; l < kLanes; ++l)
          nh[l] = nh[l] >= 4 ? nh[l] : 0;
        bool ok[kLanes];
        float par[kLanes][6], err[kLanes][21], chi2[kLanes];
        int chg[kLanes];
        fitSeeds<kLanes, ::mkseeder::kMaxSeedHits>(acc, fitInputs, lay, dh, nh, ok, par, err, chg, chi2);
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          if (i >= n)
            break;
          const bool good = nh[l] > 0 && ok[l];
          out.fitOk(i) = good ? 1 : 0;
          if (good)
            storeState(out, i, par[l], err[l], chg[l], chi2[l]);
        }
      }
    }
  };

  // the refit and, when it works, the beam-line state of the refitted seed
  class KernelRefitBl {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, FitInputs fitInputs, SeedHitsConstView seeds, SeedStatesView out, uint32_t n) const {
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, laneGroups(n))) {
        int16_t lay[kLanes][::mkseeder::kMaxSeedHits];
        uint32_t dh[kLanes][::mkseeder::kMaxSeedHits];
        int nh[kLanes];
        loadLaneHits(seeds, g, n, lay, dh, nh);
        for (int l = 0; l < kLanes; ++l)
          nh[l] = nh[l] >= 4 ? nh[l] : 0;
        bool ok[kLanes], bl[kLanes];
        float par[kLanes][6], err[kLanes][21], chi2[kLanes], blPar[kLanes][6], blErr[kLanes][21];
        int chg[kLanes], blChg[kLanes], nBl[kLanes];
        fitSeeds<kLanes, ::mkseeder::kMaxSeedHits>(acc, fitInputs, lay, dh, nh, ok, par, err, chg, chi2);
        for (int l = 0; l < kLanes; ++l)
          nBl[l] = nh[l] > 0 && ok[l] ? nh[l] : 0, bl[l] = false;
        beamlines<kLanes, ::mkseeder::kMaxSeedHits>(fitInputs, lay, dh, nBl, par, err, chg, bl, blPar, blErr, blChg);
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          if (i >= n)
            break;
          const bool refit = nBl[l] > 0, hasBl = refit && bl[l];
          out.wasRefitted(i) = refit ? 1 : 0, out.hasBeamlineState(i) = hasBl ? 1 : 0;
          if (refit)
            storeState(out, i, par[l], err[l], chg[l], chi2[l]);
          if (hasBl)
            storeBeamlineState(out, i, blPar[l], blErr[l], blChg[l]);
        }
      }
    }
  };

  // The refit of the seeds whose hits changed (4 hits at least) of the first *count (nullptr: all) of cap; if it
  // fails and the seed was extended and filled, the refit of its own hits (the input's: backup) and the extension's
  // (one seed at a time); if that fails too, the seed as it was (its input hits and state).
  class KernelRefitSeeds {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  FitInputs F,
                                  SeedHitsView seeds,
                                  SeedHitListView backup,
                                  SeedStatesView states,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = count && *count < cap ? *count : cap;
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, laneGroups(n))) {
        int16_t lay[kLanes][::mkseeder::kMaxSeedHits];
        uint32_t dh[kLanes][::mkseeder::kMaxSeedHits];
        int nh[kLanes], nFit[kLanes];
        bool refit[kLanes];
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          nh[l] = nFit[l] = 0, refit[l] = false;
          if (i >= n)
            continue;
          auto pf = states[i];
          pf.wasRefitted() = 0, pf.hasBeamlineState() = 0;
          const auto row = seeds[i];
          if (!row.isKept() || !(row.isExtended() || row.isFilled()))
            continue;
          refit[l] = true;
          nh[l] = loadSeedHits(seeds, i, lay[l], dh[l]);
          nFit[l] = nh[l] >= 4 ? nh[l] : 0;
        }
        bool ok[kLanes];
        float par[kLanes][6], err[kLanes][21], chi2[kLanes];
        int chg[kLanes];
        for (int l = 0; l < kLanes; ++l)
          ok[l] = false;
        fitSeeds<kLanes, ::mkseeder::kMaxSeedHits>(acc, F, lay, dh, nFit, ok, par, err, chg, chi2);
        for (int l = 0; l < kLanes; ++l) {
          if (!refit[l])
            continue;
          const uint32_t i = g * kLanes + l;
          const auto row = seeds[i];
          const auto s0 = backup[i];
          bool good = nFit[l] > 0 && ok[l];
          if (!good && row.nExtensionHits() > 0 && row.isFilled()) {
            int16_t lay2[1][::mkseeder::kMaxSeedHits];
            uint32_t dh2[1][::mkseeder::kMaxSeedHits];
            int n2 = 0;
            for (int k = 0; k < s0.nHits() && n2 < ::mkseeder::kMaxSeedHits; ++k)
              lay2[0][n2] = s0.hitLayers()(k), dh2[0][n2] = s0.hitRows()(k), ++n2;
            for (int k = 0; k < row.nExtensionHits() && n2 < ::mkseeder::kMaxSeedHits; ++k)
              lay2[0][n2] = row.extensionLayers()(k), dh2[0][n2] = row.extensionHits()(k), ++n2;
            if (n2 >= 4) {
              // zero-initialized: fitSeeds writes them only for a successful fit (read only then)
              bool ok2[1] = {false};
              float p2[1][6] = {}, e2[1][21] = {}, c2[1] = {};
              int q2[1] = {};
              fitSeeds<1, ::mkseeder::kMaxSeedHits>(acc, F, lay2, dh2, &n2, ok2, p2, e2, q2, c2);
              good = ok2[0];
              if (good) {
                for (int d = 0; d < 6; ++d)
                  par[l][d] = p2[0][d];
                for (int d = 0; d < 21; ++d)
                  err[l][d] = e2[0][d];
                chg[l] = q2[0], chi2[l] = c2[0];
                auto out = seeds[i];
                for (int k = 0; k < n2; ++k)
                  out.hitLayers()(k) = lay2[0][k], out.hitRows()(k) = dh2[0][k];
                out.nHits() = n2;
              }
            }
          }
          if (good) {
            states.wasRefitted(i) = 1;
            storeState(states, i, par[l], err[l], chg[l], chi2[l]);
          } else {
            auto out = seeds[i];
            for (int k = 0; k < s0.nHits(); ++k)
              out.hitLayers()(k) = s0.hitLayers()(k), out.hitRows()(k) = s0.hitRows()(k);
            out.nHits() = s0.nHits();
          }
        }
      }
    }
  };

  // The beam-line state of each kept seed's final hits and state (the first *count, nullptr: all, of cap)
  class KernelComputeBeamlineStates {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  FitInputs F,
                                  SeedHitsConstView seeds,
                                  SeedStatesView states,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = count && *count < cap ? *count : cap;
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, laneGroups(n))) {
        int16_t lay[kLanes][::mkseeder::kMaxSeedHits];
        uint32_t dh[kLanes][::mkseeder::kMaxSeedHits];
        int nh[kLanes];
        float par[kLanes][6], err[kLanes][21];
        int chg[kLanes];
        bool any = false;
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          nh[l] = 0;
          if (i >= n || !seeds.isKept(i) || seeds.nHits(i) < 2)
            continue;
          nh[l] = loadSeedHits(seeds, i, lay[l], dh[l]);
          float chi2;
          loadState(states, i, par[l], err[l], chg[l], chi2);
          any = true;
        }
        if (!any)
          continue;
        bool ok[kLanes];
        float blPar[kLanes][6], blErr[kLanes][21];
        int blChg[kLanes];
        beamlines<kLanes, ::mkseeder::kMaxSeedHits>(F, lay, dh, nh, par, err, chg, ok, blPar, blErr, blChg);
        for (int l = 0; l < kLanes; ++l) {
          if (nh[l] == 0)
            continue;
          const uint32_t i = g * kLanes + l;
          states.hasBeamlineState(i) = ok[l] ? 1 : 0;
          if (ok[l])
            storeBeamlineState(states, i, blPar[l], blErr[l], blChg[l]);
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::fit

#endif
