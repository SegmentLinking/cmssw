#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPostKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPostKernels_h

// The seeder's steps after the seed fit, on the device, one thread per seed (ported from mkFit's SeederExtend.cc): the
// outward extension into OT1-P, OT2-P and OT3-P and the pixel fill. The output is each seed's final hit list (and
// whether it is kept); the seeds whose hits changed are refit afterwards (MkSeederFitKernels.h). The hits are the layers' binned hits
// (MkSeederHitsSoA), with their global x, y, the module id (pixel layers) and the covariance (OT1-3 P,
// MkSeederOTHitsSoA).

#include <bit>
#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/MkSeederAlpaka/interface/MkSeederPSMark.h"
#include "MkSeederDeviceTypes.h"
#include "MkSeederKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {
  // the post step's scratch: the seeds' input hits (with the refits: the fallback); the pixel fill's split steps
  // on a GPU, per seed its FillPreparation, per seed and pixel layer with hits its FillCandidate (each read whole by its
  // threads: one struct column)
  GENERATE_SOA_LAYOUT(MkSeederFillPrepsLayout, SOA_COLUMN(FillPreparation, prep))
  GENERATE_SOA_LAYOUT(MkSeederFillCandidatesLayout, SOA_COLUMN(FillCandidate, candidate))
  GENERATE_SOA_BLOCKS(MkSeederPostScratchLayout,
                      SOA_BLOCK(hitsBackup, ::mkseeder::MkSeederSeedHitListLayout),
                      SOA_BLOCK(preps, MkSeederFillPrepsLayout),
                      SOA_BLOCK(fillCandidates, MkSeederFillCandidatesLayout))
  using MkSeederFillPrepsSoA = MkSeederFillPrepsLayout<>;
  using MkSeederFillCandidatesSoA = MkSeederFillCandidatesLayout<>;
  using MkSeederPostScratchSoA = MkSeederPostScratchLayout<>;

  using FillPrepsView = ScratchView<MkSeederFillPrepsSoA>;
  using FillCandidatesView = ScratchView<MkSeederFillCandidatesSoA>;

  // the global position of a hit
  struct SeedHitAccessor {
    const ChainFinderInputs *v;
    const PostInputs *p;
    ALPAKA_FN_HOST_ACC float x(uint32_t i) const { return p->hits.x(i); }
    ALPAKA_FN_HOST_ACC float y(uint32_t i) const { return p->hits.y(i); }
    ALPAKA_FN_HOST_ACC float z(uint32_t i) const { return v->hits.z(i); }
    // the CPU backends read per-event precomputed hit / module / layer quantities (PostInputs::cpuHits, cpuOTHitFrames,
    // cpuLayers); the GPU backends compute them where they are used
    template <typename TAcc>
    ALPAKA_FN_ACC float r(const TAcc &acc, uint32_t i) const {
      if constexpr (kCpuAcc<TAcc>)
        if (p->hasCpuCache)
          return p->cpuHits.rGlobal(i);
      return alpaka::math::sqrt(acc, p->hits.x(i) * p->hits.x(i) + p->hits.y(i) * p->hits.y(i));
    }
    template <typename TAcc>
    ALPAKA_FN_ACC float phi(const TAcc &acc, uint32_t i) const {
      if constexpr (kCpuAcc<TAcc>)
        if (p->hasCpuCache)
          return p->cpuHits.phiGlobal(i);
      return libm::atan2(acc, p->hits.y(i), p->hits.x(i));
    }
  };

  // the module frame of mkseeder::extensionLocalResidual for the hit (hx, hy) with covariance E, the same
  // expressions: F = hr, ux[0..1], uy[0..2], nn[0..2]
  template <typename TAcc>
  ALPAKA_FN_ACC inline void extensionLocalFrame(const TAcc &acc, float hx, float hy, const float *E, float *F) {
    const float hr = alpaka::math::sqrt(acc, hx * hx + hy * hy), cr = hx / hr, sr = hy / hr;
    const float vr = cr * cr * E[0] + sr * sr * E[1] + 2 * cr * sr * E[2];
    const float crz = cr * E[3] + sr * E[4];
    const float al = 0.5f * libm::atan2(acc, 2 * crz, vr - E[5]);
    const float ux[3] = {-sr, cr, 0.f},
                uy[3] = {alpaka::math::cos(acc, al) * cr, alpaka::math::cos(acc, al) * sr, alpaka::math::sin(acc, al)};
    const float nn[3] = {ux[1] * uy[2] - ux[2] * uy[1], ux[2] * uy[0] - ux[0] * uy[2], ux[0] * uy[1] - ux[1] * uy[0]};
    F[0] = hr, F[1] = ux[0], F[2] = ux[1], F[3] = uy[0], F[4] = uy[1], F[5] = uy[2], F[6] = nn[0], F[7] = nn[1],
    F[8] = nn[2];
  }
  // the frame of OT1-3 P hit c (PostInputs::cpuOTHitFrames: every backend) as extensionLocalFrame()'s F
  ALPAKA_FN_HOST_ACC inline void otHitFrame(const OTHitFramesView &frames, uint32_t c, float *F) {
    const auto f = frames[c];
    F[0] = f.hitR(), F[1] = f.localXDirectionX(), F[2] = f.localXDirectionY(), F[3] = f.localYDirectionX();
    F[4] = f.localYDirectionY(), F[5] = f.localYDirectionZ(), F[6] = f.normalX(), F[7] = f.normalY(),
    F[8] = f.normalZ();
  }
  // the frames' columns in extensionLocalFrame()'s F order
  inline void otHitFrameColumns(const OTHitFramesView &frames, const float **columns) {
    columns[0] = frames.hitR().data(), columns[1] = frames.localXDirectionX().data();
    columns[2] = frames.localXDirectionY().data(), columns[3] = frames.localYDirectionX().data();
    columns[4] = frames.localYDirectionY().data(), columns[5] = frames.localYDirectionZ().data();
    columns[6] = frames.normalX().data(), columns[7] = frames.normalY().data(), columns[8] = frames.normalZ().data();
  }
  // extensionLocalResidual with the frame F precomputed (PostInputs::cpuOTHitFrames): the rest of it, the same expressions
  template <typename TAcc, typename Helix>
  ALPAKA_FN_ACC inline bool extensionLocalResidualWithFrame(
      const TAcc &acc, const Helix &H, float hx, float hy, float hz, const float *F, float &dx, float &dy) {
    const float hr = F[0], ux[2] = {F[1], F[2]}, uy[3] = {F[3], F[4], F[5]}, nn[3] = {F[6], F[7], F[8]};
    float ax, ay, aq, as, bx, by, bq, bs;
    if (!H.positionAtR(acc, hr - 1.f, ax, ay, aq, as) || !H.positionAtR(acc, hr + 1.f, bx, by, bq, bs))
      return false;
    const float fa = (ax - hx) * nn[0] + (ay - hy) * nn[1] + (aq - hz) * nn[2];
    const float fb = (bx - hx) * nn[0] + (by - hy) * nn[1] + (bq - hz) * nn[2];
    if (fa == fb)
      return false;
    const float t = fa / (fa - fb);
    const float d[3] = {ax + t * (bx - ax) - hx, ay + t * (by - ay) - hy, aq + t * (bq - aq) - hz};
    dx = d[0] * ux[0] + d[1] * ux[1], dy = d[0] * uy[0] + d[1] * uy[1] + d[2] * uy[2];
    return true;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool extensionLocal(const TAcc &acc,
                                           const PostInputs &pv,
                                           const HelixF &Hx,
                                           const SeedHitAccessor &H,
                                           int l,
                                           uint32_t i,
                                           uint32_t ih,
                                           float hz,
                                           float &dx,
                                           float &dy) {
    const uint32_t c = pv.layers.firstOTHit(l) + i;
    if (pv.hasOTFrames) {
      float F[9];
      otHitFrame(pv.cpuOTHitFrames, c, F);
      return extensionLocalResidualWithFrame(acc, Hx, H.x(ih), H.y(ih), hz, F, dx, dy);
    }
    float E[6];
    hitCovariance(pv.otHits, c, E);
    return mkseeder::extensionLocalResidual(acc, Hx, H.x(ih), H.y(ih), hz, E, dx, dy);
  }

  // Hit::phi()-ordered window helpers: the hits' phi bins are in beam-relative phi; a window in global phi is
  // widened by kPhiPad for the bins and then cut exactly
  constexpr float kPhiPad = 0.01f;
  // the PS-module confirmation of an extension candidate (ps_ext, ps_curv): true if the OT1-3 P hit i of layer l (its
  // index in the layer, at radius hr) is rejected: unmarked (ps_ext), or marked with a q/pT estimate away from the
  // helix's (ps_curv; HelixF::k > 0 turns counter-clockwise: q / pT = -k / kNominalPtPerRadius); v2 (ps_alt bits): a
  // candidate failing ps_curv is still taken if 1: it has an S match in the helix's q/pT range (psCounts), 2: the
  // second-best S match's q/pT estimate passes, 4: its best S hit is merged
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool psExtensionReject(
      const TAcc &acc, const PostInputs &pv, int l, uint32_t i, const HelixF &Hx, float hr) {
    if (!pv.psExtension && !(pv.psCurvatureSigma > 0))
      return false;
    const int c0 = pv.layers.firstOTHit(l);
    if (c0 < 0)
      return false;
    if (pv.psExtension >= 2) {
      // ps_ext 2 (3): an S match in the helix's q/pT range (or a neighbouring one)
      const uint32_t c = uint32_t(c0) + i;
      return ::mkseeder::psCountInRange(pv.otHits.psCounts(c),
                                        pv.otHits.psCountsLow(c),
                                        pv.psLow,
                                        -Hx.k / kNominalPtPerRadius,
                                        pv.psExtension == 3) == 0;
    }
    const uint8_t m = pv.otHits.psMatches(uint32_t(c0) + i);
    if (pv.psExtension && m == 0)
      return true;
    if (pv.psCurvatureSigma > 0 && m > 0) {
      const float qpt = -Hx.k / kNominalPtPerRadius;
      const float tol = pv.psCurvatureSigma * pv.psCurvatureResolution * 23.f / std::max(hr, 1.f);
      const uint32_t c = uint32_t(c0) + i;
      if (!(alpaka::math::abs(acc, pv.otHits.psQOverPt(c) - qpt) > tol))
        return false;
      const int alt = pv.psAlternatives;
      if (alt == 0)
        return true;
      const uint16_t w = pv.otHits.psCounts(c);
      if ((alt & 1) && ::mkseeder::psCounts::count(w, ::mkseeder::psCounts::range(qpt)) > 0)
        return false;
      if ((alt & 2) && m >= 2 && !(alpaka::math::abs(acc, pv.otHits.psQOverPt2(c) - qpt) > tol))
        return false;
      if ((alt & 4) && (w & ::mkseeder::psCounts::kBestMerged))
        return false;
      return true;
    }
    return false;
  }

  // a layer's phi pad of the windows: kPhiPad + the beam spot's angle at its inner radius (PostInputs::ppad on the CPU
  // backends, the same expression once per event)

  template <typename TAcc>
  ALPAKA_FN_ACC inline float layerPhiPadExpression(const TAcc &acc, const PostInputs &pv, int l) {
    return kPhiPad + libm::asin(acc, std::min(1.f, pv.beamSpotRadius / std::max(1.f, pv.layerGeometry[l].rIn())));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline float layerPhiPad(const TAcc &acc, const PostInputs &pv, int l) {
    if constexpr (kCpuAcc<TAcc>)
      if (pv.hasCpuCache)
        return pv.cpuLayers.phiWindowPad(l);
    return layerPhiPadExpression(acc, pv, l);
  }

  ALPAKA_FN_HOST_ACC inline bool isInPhiRange(float ph, float lo, float hi) {
    // ph in [lo, hi] over the wrapped window (lo > hi: two ranges)
    return lo <= hi ? (ph >= lo && ph <= hi) : (ph >= lo || ph <= hi);
  }

#ifdef MKSEEDER_CPU_AVX2
  // the CPU backends, one thread, extensionLocal with the frames (PostInputs::cpuOTHitFrames): the window run [b, e) of layer l 8 hits
  // at a time up to the score, the operations of extendWithBranching's loop (positionAtR, extensionLocalResidualWithFrame with the
  // compiler's FMA contractions); consume(ih, sc) for the hits that pass, in hit order
  template <typename F>
  inline void scanExtensionHits8(const ChainFinderInputs &v,
                                 const PostInputs &pv,
                                 const avx2::Helix8 &Hx,
                                 int l,
                                 const EventLayer &L,
                                 uint32_t b,
                                 uint32_t e,
                                 float zlo,
                                 float zhi,
                                 float lo,
                                 float hi,
                                 float wx,
                                 float wy,
                                 F &&consume) {
    const __m256 vzlo = _mm256_set1_ps(zlo), vzhi = _mm256_set1_ps(zhi), vlo = _mm256_set1_ps(lo),
                 vhi = _mm256_set1_ps(hi);
    const __m256 vwx = _mm256_set1_ps(wx), vwy = _mm256_set1_ps(wy), one = _mm256_set1_ps(1.0f);
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const bool inw = lo <= hi;
    for (uint32_t i0 = b; i0 < e; i0 += 8) {
      const uint32_t nk = std::min(8u, e - i0), ih0 = L.firstHit() + i0;
      __m256 keep = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32((int)nk), lane));
      const __m256 hz = _mm256_loadu_ps(v.hits.z().data() + ih0);
      keep = _mm256_and_ps(keep, _mm256_cmp_ps(hz, vzlo, _CMP_NLT_UQ));
      keep = _mm256_and_ps(keep, _mm256_cmp_ps(hz, vzhi, _CMP_NGT_UQ));
      const __m256 ph = _mm256_loadu_ps(pv.cpuHits.phiGlobal().data() + ih0);
      const __m256 ge = _mm256_cmp_ps(ph, vlo, _CMP_GE_OQ), le = _mm256_cmp_ps(ph, vhi, _CMP_LE_OQ);
      keep = _mm256_and_ps(keep, inw ? _mm256_and_ps(ge, le) : _mm256_or_ps(ge, le));
      // ps_ext: the unmarked OT1-3 P hits out before the helix (psExtensionReject() rejects them)
      if (pv.psExtension == 1 && pv.layers.firstOTHit(l) >= 0 && _mm256_movemask_ps(keep)) {
        const uint8_t *mk = pv.otHits.psMatches().data() + pv.layers.firstOTHit(l) + i0;
        alignas(32) int32_t mm[8];
        for (uint32_t k = 0; k < 8; ++k)
          mm[k] = k < nk && mk[k] != 0 ? -1 : 0;
        keep = _mm256_and_ps(keep, _mm256_castsi256_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(mm))));
      }
      __m256 ok, wr, px, py, qp, s3;
      if (_mm256_movemask_ps(keep)) {
        avx2::positionAtR8(Hx, _mm256_loadu_ps(pv.cpuHits.rGlobal().data() + ih0), keep, ok, wr, px, py, qp, s3);
        keep = ok;
      }
      __m256 sc = _mm256_setzero_ps();
      if (_mm256_movemask_ps(keep)) {
        const float *Fr[9];
        otHitFrameColumns(pv.cpuOTHitFrames, Fr);
        const uint32_t c0 = pv.layers.firstOTHit(l) + i0;
        __m256 Fv[9];
        for (int k = 0; k < 9; ++k)
          Fv[k] = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), Fr[k] + c0, lane, keep, 4);
        __m256 oka, okb, ax, ay, aq, as, bx, by, bq, bs, w1, w2;
        avx2::positionAtR8(Hx, _mm256_sub_ps(Fv[0], one), keep, oka, w1, ax, ay, aq, as);
        avx2::positionAtR8(Hx, _mm256_add_ps(Fv[0], one), keep, okb, w2, bx, by, bq, bs);
        keep = _mm256_and_ps(keep, _mm256_and_ps(oka, okb));
        const __m256 hx = _mm256_loadu_ps(pv.hits.x().data() + ih0), hy = _mm256_loadu_ps(pv.hits.y().data() + ih0);
        const __m256 fa =
            avx2::fma(_mm256_sub_ps(aq, hz),
                      Fv[8],
                      avx2::fma(_mm256_sub_ps(ax, hx), Fv[6], _mm256_mul_ps(_mm256_sub_ps(ay, hy), Fv[7])));
        const __m256 fb =
            avx2::fma(_mm256_sub_ps(bq, hz),
                      Fv[8],
                      avx2::fma(_mm256_sub_ps(bx, hx), Fv[6], _mm256_mul_ps(_mm256_sub_ps(by, hy), Fv[7])));
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(fa, fb, _CMP_NEQ_UQ));
        const __m256 tt = _mm256_div_ps(fa, _mm256_sub_ps(fa, fb));
        const __m256 d0 = _mm256_sub_ps(avx2::fma(_mm256_sub_ps(bx, ax), tt, ax), hx);
        const __m256 d1 = _mm256_sub_ps(avx2::fma(_mm256_sub_ps(by, ay), tt, ay), hy);
        const __m256 d2 = _mm256_sub_ps(avx2::fma(_mm256_sub_ps(bq, aq), tt, aq), hz);
        const __m256 dx = avx2::fma(d0, Fv[1], _mm256_mul_ps(d1, Fv[2]));
        const __m256 dy = avx2::fma(d2, Fv[5], avx2::fma(d0, Fv[3], _mm256_mul_ps(d1, Fv[4])));
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(_mm256_and_ps(dx, absm), vwx, _CMP_NGE_UQ));
        keep = _mm256_and_ps(keep, _mm256_cmp_ps(_mm256_and_ps(dy, absm), vwy, _CMP_NGE_UQ));
        const __m256 rx = _mm256_div_ps(dx, vwx), ry = _mm256_div_ps(dy, vwy);
        sc = avx2::fma(rx, rx, _mm256_mul_ps(ry, ry));
      }
      alignas(32) float scs[8];
      _mm256_store_ps(scs, sc);
      const unsigned int km = _mm256_movemask_ps(keep);
      for (unsigned int mm = km; mm; mm &= mm - 1) {
        const int ll = std::countr_zero(mm);
        consume(ih0 + ll, scs[ll]);
      }
    }
  }
#endif

  // ---- the OT extension with its options (extension branching, redo, minimum OT hits, triplet MLP on the extension
  // hits), for a seed that passed extendBlockParallel()'s entry checks. Branching K > 1: up to K hits per layer (the
  // best K of the window, by score then orig index), followed depth first; the path with the most hits wins, then the
  // lowest sum of scores. The lanes share each window scan: every lane keeps its own best K, and K rounds of
  // reduce() give all lanes the same merged list (the control flow depends only on reduced values).
  constexpr int kExtMaxBranch = 4;
  template <typename TAcc, typename Reduce>
  ALPAKA_FN_ACC inline bool extendWithBranching(const TAcc &acc,
                                                const ChainFinderInputs &v,
                                                const PostInputs &pv,
                                                const SeedHitAccessor &H,
                                                WorkingSeed &sd,
                                                uint32_t t,
                                                uint32_t W,
                                                Reduce &&reduce,
                                                int npix,
                                                bool needs,
                                                bool trip3,
                                                bool tfail) {
    const PostParameters &P = pv.parameters;
    constexpr int kLay[3] = {4, 6, 8};
    const bool conf = P.extensionMinHits > 0;
    // ext_branch_maxpix > 0: the branching only for seeds with at most that many pixel hits (K 1: the best hit per
    // layer, the same path as the plain extension)
    const int K = P.extensionBranching < 1 ||
                          (P.extensionBranchingMaxPixelLayers > 0 && npix > P.extensionBranchingMaxPixelLayers)
                      ? 1
                      : (P.extensionBranching > kExtMaxBranch ? kExtMaxBranch : P.extensionBranching);
    const int nlay = P.nExtensionLayers < 3 ? P.nExtensionLayers : 3;
    const int n0 = sd.nHits;
    int16_t l0[::mkseeder::kMaxSeedHits];
    uint32_t h0[::mkseeder::kMaxSeedHits];
    for (int k = 0; k < n0; ++k)
      l0[k] = sd.layers[k], h0[k] = sd.hitRows[k];
    // the window of layer j for the path in sd: -1 stop, 0 not looked at, 1 looked at (csc / ckb: nc candidates)
    auto scan_layer = [&](int j, float *csc, int *ckb, int &nc) -> int {
      nc = 0;
      const int l = kLay[j];
      const int last = sd.layers[sd.nHits - 1];
      if (v.layers[l].nHits() == 0 || ((last <= 4 || (conf && last < 16)) && l <= last))
        return 0;
      const uint32_t ia = sd.hitRows[sd.nHits - 3], ib = sd.hitRows[sd.nHits - 2], ic = sd.hitRows[sd.nHits - 1];
      HelixF Hx;
      Hx.make(acc, H.x(ia), H.y(ia), H.z(ia), H.x(ib), H.y(ib), H.x(ic), H.y(ic), H.z(ic));
      if (!Hx.ok)
        return -1;
      const auto li = pv.layerGeometry[l];
      float x0, y0, q0, s0, x1, y1, q1, s1;
      if (!Hx.positionAtR(acc, li.rIn(), x0, y0, q0, s0) || !Hx.positionAtR(acc, li.rOut(), x1, y1, q1, s1))
        return -1;
      const float pt = std::max(0.5f, Hx.pt(acc)), sphi = 0.0005f + 3.2e-3f / pt, sz = 0.075f + 0.0316f / pt;
      const float p0 = libm::atan2(acc, y0, x0), dpp = wrapPhiRound(acc, libm::atan2(acc, y1, x1) - p0);
      const float esig = P.extensionSigma;
      const float wx = P.extensionWindowX[0] + P.extensionWindowX[1] / pt,
                  wy = P.extensionWindowY[0] + P.extensionWindowY[1] / pt;
      const float pc = p0 + 0.5f * dpp,
                  w = 0.5f * alpaka::math::abs(acc, dpp) + (P.extensionLocal ? 1.5f * wx / li.rIn() : esig * sphi);
      const float thr = P.extensionLocal ? 1.f : esig * esig;
      const float zpad = (P.extensionLocal ? wy : esig * sz) + 1.f;
      const float zlo = std::min(q0, q1) - zpad, zhi = std::max(q0, q1) + zpad;
      const float lo = wrapPhiRound(acc, pc - w), hi = wrapPhiRound(acc, pc + w);
      const EventLayer L = v.layers[l];
      const float ppad = layerPhiPad(acc, pv, l);
      // this lane's best K, by score then orig index
      float lsc[kExtMaxBranch];
      int lkb[kExtMaxBranch];
      int ln = 0;
      auto before = [&](float sa, int ka, float sb, int kb) {
        return sa < sb || (sa == sb && v.hits.clusterIndex(ka) < v.hits.clusterIndex(kb));
      };
      // a hit with its score: this lane's best K
      auto consume = [&](uint32_t ih, float sc) {
        if (!(sc < thr))
          return;
        if (psExtensionReject(acc, pv, l, ih - L.firstHit(), Hx, H.r(acc, ih)))
          return;
        bool dup = false;
        for (int a = 0; a < ln; ++a)
          dup |= lkb[a] == (int)ih;
        if (dup || (ln == K && !before(sc, (int)ih, lsc[K - 1], lkb[K - 1])))
          return;
        int p = ln < K ? ln++ : K - 1;
        for (; p > 0 && before(sc, (int)ih, lsc[p - 1], lkb[p - 1]); --p)
          lsc[p] = lsc[p - 1], lkb[p] = lkb[p - 1];
        lsc[p] = sc, lkb[p] = (int)ih;
      };
#ifdef MKSEEDER_CPU_AVX2
      // one thread, extensionLocal with the frames: the window's hits 8 at a time up to the score (the same values), then
      // consume() in hit order
      const bool vec = W == 1 && P.extensionLocal && pv.hasOTFrames && pv.hasCpuCache;
      const avx2::Helix8 Hx8 = avx2::bcast(Hx);
#endif
      forEachBinRun(
          v,
          L,
          phiBinRangeAround(acc, pv.layerGeometry[l], pc, w + ppad),
          qBinRangePadded(acc, pv.layerGeometry[l], zlo, zhi),
          [&](uint32_t b, uint32_t e) {
#ifdef MKSEEDER_CPU_AVX2
            if (vec) {
              scanExtensionHits8(v, pv, Hx8, l, L, b, e, zlo, zhi, lo, hi, wx, wy, consume);
              return;
            }
#endif
            for (uint32_t i = b + t; i < e; i += W) {
              const uint32_t ih = L.firstHit() + i;
              const float hz = H.z(ih);
              if (hz < zlo || hz > zhi)
                continue;
              const float hph = H.phi(acc, ih);
              if (!isInPhiRange(hph, lo, hi))
                continue;
              // ps_ext: an unmarked OT1-3 P hit out before the helix (psExtensionReject() rejects it)
              if (pv.psExtension == 1 && L.firstOTHit() >= 0 && pv.otHits.psMatches(uint32_t(L.firstOTHit()) + i) == 0)
                continue;
              float px, py, qp, s3;
              const float hr = H.r(acc, ih);
              if (!Hx.positionAtR(acc, hr, px, py, qp, s3))
                continue;
              float sc;
              if (P.extensionLocal) {
                float dx,
                    dy;  // the covariance xx, yy, xy, xz, yz, zz at pv.otHits row pv.layers.firstOTHit(l) + i
                if (!extensionLocal(acc, pv, Hx, H, l, i, ih, hz, dx, dy))
                  continue;
                if (alpaka::math::abs(acc, dx) >= wx || alpaka::math::abs(acc, dy) >= wy)
                  continue;
                sc = (dx / wx) * (dx / wx) + (dy / wy) * (dy / wy);
              } else {
                const float dp = wrapPhiRound(acc, hph - libm::atan2(acc, py, px)), dq = hz - qp;
                sc = (dp / sphi) * (dp / sphi) + (dq / sz) * (dq / sz);
              }
              consume(ih, sc);
            }
          });
      // K rounds: the best of all lanes' heads; the lane holding it moves on
      int ptr = 0;
      for (int r = 0; r < K; ++r) {
        float bb = ptr < ln ? lsc[ptr] : thr;
        int kk = ptr < ln ? lkb[ptr] : -1;
        reduce(bb, kk);
        if (kk < 0)
          break;
        csc[nc] = bb, ckb[nc] = kk, ++nc;
        if (ptr < ln && lkb[ptr] == kk)
          ++ptr;
      }
      return 1;
    };  // scan_layer
    // the extension of the path in sd (from its hits on): depth first over the candidates; sd ends as the best path
    auto run_ext = [&]() {
      const int ns = sd.nHits;
      int16_t blay[::mkseeder::kMaxSeedHits];
      uint32_t bdh[::mkseeder::kMaxSeedHits];
      int bn = ns;
      float bsum = 0.f;
      float fsc[3][kExtMaxBranch], fss[3];
      int fkb[3][kExtMaxBranch], fnc[3], fci[3];
      int j = 0;
      float ssum = 0.f;
      for (;;) {
        // down: the first candidate of each layer
        for (;;) {
          if (j >= nlay)
            break;
          const int st = sd.nHits >= ::mkseeder::kMaxSeedHits ? -1 : scan_layer(j, fsc[j], fkb[j], fnc[j]);
          if (st < 0)
            break;
          if (st == 0 || fnc[j] == 0) {
            fnc[j] = 0;  // a hole: the next layer from the same hits
            ++j;
            continue;
          }
          fci[j] = 0, fss[j] = ssum;
          sd.layers[sd.nHits] = kLay[j], sd.hitRows[sd.nHits] = (uint32_t)fkb[j][0], ++sd.nHits;
          ssum += fsc[j][0];
          ++j;
        }
        // a complete path: the most hits, then the lowest sum of scores (the first found on ties); ext_path_bonus B
        // > 0: the lowest sum of (score - B) over the added hits, then the most hits
        const float pb = P.extensionPathBonus;
        const bool better = pb > 0.f
                                ? (ssum - pb * float(sd.nHits - ns) < bsum - pb * float(bn - ns) ||
                                   (ssum - pb * float(sd.nHits - ns) == bsum - pb * float(bn - ns) && sd.nHits > bn))
                                : (sd.nHits > bn || (sd.nHits == bn && sd.nHits > ns && ssum < bsum));
        if (better) {
          for (int k = ns; k < sd.nHits; ++k)
            blay[k] = sd.layers[k], bdh[k] = sd.hitRows[k];
          bn = sd.nHits, bsum = ssum;
        }
        // up: the next candidate of the deepest layer that has one
        bool more = false;
        for (--j; j >= 0; --j) {
          if (fnc[j] == 0)
            continue;
          --sd.nHits;
          if (++fci[j] < fnc[j]) {
            sd.layers[sd.nHits] = kLay[j], sd.hitRows[sd.nHits] = (uint32_t)fkb[j][fci[j]], ++sd.nHits;
            ssum = fss[j] + fsc[j][fci[j]];
            ++j;
            more = true;
            break;
          }
          ssum = fss[j];
        }
        if (!more)
          break;
      }
      for (int k = ns; k < bn; ++k)
        sd.layers[k] = blay[k], sd.hitRows[k] = bdh[k];
      sd.nHits = bn;
    };
    auto restore = [&]() {
      for (int k = 0; k < n0; ++k)
        sd.layers[k] = l0[k], sd.hitRows[k] = h0[k];
      sd.nHits = n0;
    };
    run_ext();
    bool redone = false;
    // ext_redo: a seed of 3 pixel hits whose OT hits all come from the quad and got none here: again from its pixel
    // hits alone; that replaces it if it has more hits
    if (P.extensionRedo && npix == 3 && n0 > 3 && sd.nHits == n0) {
      int m = 0;
      for (int k = 0; k < n0; ++k)
        if (pv.layerGeometry[l0[k]].isPixel())
          sd.layers[m] = l0[k], sd.hitRows[m] = h0[k], ++m;
      if (m == 3) {
        sd.nHits = 3;
        run_ext();
        if (sd.nHits > n0)
          redone = true;
        else
          restore();
      } else
        restore();
    }
    // ext_min_ot: a seed of 3 pixel hits keeps the added OT hits only with >= ext_min_ot OT hits
    // ext_min_ot_mk 1: only the marked OT1-3 P hits (PS mark) and the OT hits of other layers count
    int nOTCounted = sd.nHits - 3;
    if (P.extensionMinOTMarked && P.extensionMinOTHits > 0 && npix == 3 && sd.nHits > n0) {
      nOTCounted = 0;
      for (int k = 0; k < sd.nHits; ++k) {
        const int l = sd.layers[k];
        if (pv.layerGeometry[l].isPixel())
          continue;
        const int c0 = pv.layers.firstOTHit(l);
        nOTCounted += c0 < 0 || pv.otHits.psMatches(uint32_t(c0) + (sd.hitRows[k] - v.layers[l].firstHit())) > 0;
      }
    }
    if (P.extensionMinOTHits > 0 && npix == 3 && sd.nHits > n0 && nOTCounted < P.extensionMinOTHits)
      restore(), redone = false;
    if (tfail && sd.nHits - npix < P.tripletMlpExtensionHits)
      return false;
    if (needs && sd.nHits < P.extensionMinHits) {
      if (!trip3)
        return false;
      restore();  // a kept triplet stays without the OT hits
      sd.nExtensionHits = 0;
      return true;
    }
    // the added hits (for the fallback refit); a redone seed has none (it keeps its own hits then)
    sd.nExtensionHits = 0;
    if (!redone)
      for (int k = n0; k < sd.nHits && sd.nExtensionHits < 3; ++k)
        sd.extensionLayers[sd.nExtensionHits] = sd.layers[k], sd.extensionHits[sd.nExtensionHits] = sd.hitRows[k],
        ++sd.nExtensionHits;
    return true;
  }  // extendWithBranching()

  // ---- the OT extension of one seed; hv / nh in out; returns false if the seed is dropped.
  // Lane t of W scans every W-th hit of the windows and reduce(best, kb) makes every lane's best the best of all
  // (the lowest score, ties by the lowest orig index: the same hit in any order); W = 1: one thread.
  template <typename TAcc, typename Reduce>
  ALPAKA_FN_ACC inline bool extendBlockParallel(const TAcc &acc,
                                                const ChainFinderInputs &v,
                                                const PostInputs &pv,
                                                const SeedHitAccessor &H,
                                                WorkingSeed &sd,
                                                uint32_t t,
                                                uint32_t W,
                                                Reduce &&reduce) {
    const PostParameters &P = pv.parameters;
    if (P.extendIntoOT <= 0)
      return true;
    constexpr int kLay[3] = {4, 6, 8};
    const bool conf = P.extensionMinHits > 0;
    bool barrel = true;
    int npix = 0;
    for (int k = 0; k < sd.nHits; ++k) {
      const int l = sd.layers[k];
      const bool lay_ok = l <= 4 || (conf && (l == 6 || l == 8)) ||
                          (P.extensionAnyLayer && ((l >= 16 && l <= 27) || (l >= 38 && l <= 49)));
      barrel &= lay_ok;
      npix += pv.layerGeometry[l].isPixel();
    }
    const bool needs = conf && npix <= 3;
    const bool trip3 = P.keepTripletsWithoutOT && npix == 3 && sd.nHits == 3;
    const bool tfail = P.tripletMlpExtensionHits > 0 && sd.tripletFailedMlp;
    if (!barrel || sd.nHits < 3)
      return !((needs && sd.nHits < P.extensionMinHits && !trip3) || tfail);
    if (P.extensionBranching > 1 || P.extensionRedo || P.extensionMinOTHits > 0 || tfail)
      return extendWithBranching(acc, v, pv, H, sd, t, W, reduce, npix, needs, trip3, tfail);
    const int n0 = sd.nHits;
    for (int j = 0; j < (P.nExtensionLayers < 3 ? P.nExtensionLayers : 3); ++j) {
      const int l = kLay[j];
      const int last = sd.layers[sd.nHits - 1];
      if (v.layers[l].nHits() == 0 || ((last <= 4 || (conf && last < 16)) && l <= last))
        continue;
      const uint32_t ia = sd.hitRows[sd.nHits - 3], ib = sd.hitRows[sd.nHits - 2], ic = sd.hitRows[sd.nHits - 1];
      HelixF Hx;
      Hx.make(acc, H.x(ia), H.y(ia), H.z(ia), H.x(ib), H.y(ib), H.x(ic), H.y(ic), H.z(ic));
      if (!Hx.ok)
        break;
      const auto li = pv.layerGeometry[l];
      float x0, y0, q0, s0, x1, y1, q1, s1;
      if (!Hx.positionAtR(acc, li.rIn(), x0, y0, q0, s0) || !Hx.positionAtR(acc, li.rOut(), x1, y1, q1, s1))
        break;
      const float pt = std::max(0.5f, Hx.pt(acc)), sphi = 0.0005f + 3.2e-3f / pt, sz = 0.075f + 0.0316f / pt;
      const float p0 = libm::atan2(acc, y0, x0), dpp = wrapPhiRound(acc, libm::atan2(acc, y1, x1) - p0);
      const float esig = P.extensionSigma;
      const float wx = P.extensionWindowX[0] + P.extensionWindowX[1] / pt,
                  wy = P.extensionWindowY[0] + P.extensionWindowY[1] / pt;
      const float pc = p0 + 0.5f * dpp,
                  w = 0.5f * alpaka::math::abs(acc, dpp) + (P.extensionLocal ? 1.5f * wx / li.rIn() : esig * sphi);
      float best = P.extensionLocal ? 1.f : esig * esig;
      int kb = -1;
      const float zpad = (P.extensionLocal ? wy : esig * sz) + 1.f;
      const float zlo = std::min(q0, q1) - zpad, zhi = std::max(q0, q1) + zpad;
      const float lo = wrapPhiRound(acc, pc - w), hi = wrapPhiRound(acc, pc + w);
      const EventLayer L = v.layers[l];
      const float ppad = layerPhiPad(acc, pv, l);
      forEachBinRun(
          v,
          L,
          phiBinRangeAround(acc, pv.layerGeometry[l], pc, w + ppad),
          qBinRangePadded(acc, pv.layerGeometry[l], zlo, zhi),
          [&](uint32_t b, uint32_t e) {
            for (uint32_t i = b + t; i < e; i += W) {
              const uint32_t ih = L.firstHit() + i;
              const float hz = H.z(ih);
              if (hz < zlo || hz > zhi)
                continue;
              const float hph = H.phi(acc, ih);
              if (!isInPhiRange(hph, lo, hi))
                continue;
              // ps_ext: an unmarked OT1-3 P hit out before the helix (psExtensionReject() rejects it)
              if (pv.psExtension == 1 && L.firstOTHit() >= 0 && pv.otHits.psMatches(uint32_t(L.firstOTHit()) + i) == 0)
                continue;
              float px, py, qp, s3;
              const float hr = H.r(acc, ih);
              if (!Hx.positionAtR(acc, hr, px, py, qp, s3))
                continue;
              float sc;
              if (P.extensionLocal) {
                float dx,
                    dy;  // the covariance xx, yy, xy, xz, yz, zz at pv.otHits row pv.layers.firstOTHit(l) + i
                if (!extensionLocal(acc, pv, Hx, H, l, i, ih, hz, dx, dy))
                  continue;
                if (alpaka::math::abs(acc, dx) >= wx || alpaka::math::abs(acc, dy) >= wy)
                  continue;
                sc = (dx / wx) * (dx / wx) + (dy / wy) * (dy / wy);
              } else {
                const float dp = wrapPhiRound(acc, hph - libm::atan2(acc, py, px)), dq = hz - qp;
                sc = (dp / sphi) * (dp / sphi) + (dq / sz) * (dq / sz);
              }
              if (sc < best && psExtensionReject(acc, pv, l, i, Hx, hr))
                continue;
              // ties: the lowest hit (orig) index
              if (sc < best || (sc == best && kb >= 0 && v.hits.clusterIndex(ih) < v.hits.clusterIndex(kb)))
                best = sc, kb = (int)ih;
            }
          });
      reduce(best, kb);
      if (kb < 0)
        continue;
      if (sd.nHits >= ::mkseeder::kMaxSeedHits)
        break;
      sd.layers[sd.nHits] = l, sd.hitRows[sd.nHits] = (uint32_t)kb, ++sd.nHits;
      if (sd.nExtensionHits < 3)
        sd.extensionLayers[sd.nExtensionHits] = l, sd.extensionHits[sd.nExtensionHits] = (uint32_t)kb,
        ++sd.nExtensionHits;
    }
    if (needs && sd.nHits < P.extensionMinHits)
      return trip3 ? (sd.nHits = n0, sd.nExtensionHits = 0, true) : false;  // a kept triplet stays without the OT hits
    return true;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC inline bool extendSeedIntoOuterTracker(
      const TAcc &acc, const ChainFinderInputs &v, const PostInputs &pv, const SeedHitAccessor &H, WorkingSeed &sd) {
    return extendBlockParallel(acc, v, pv, H, sd, 0, 1, [](float &, int &) {});
  }

  // ---- the pixel fill of one seed, the hits in their final order. In three steps, so that
  // the device can search the layers in parallel: prepareFill (the seed's circle and line), findFillCandidates (the best hits
  // of one pixel layer), applyFill (the hits added, in order).
  template <typename TAcc>
  ALPAKA_FN_ACC inline void prepareFill(
      const TAcc &acc, const PostInputs &pv, const SeedHitAccessor &H, const WorkingSeed &sd, FillPreparation &F) {
    const PostParameters &P = pv.parameters;
    F.hasWork = 0;
    if (P.pixelFillMode <= 0)
      return;
    int np = 0, no = 0;
    for (int k = 0; k < sd.nHits; ++k) {
      if (pv.layerGeometry[sd.layers[k]].isPixel())
        F.pixelLayers[np] = sd.layers[k], F.pixelHits[np++] = sd.hitRows[k];
      else
        F.outerLayers[no] = sd.layers[k], F.outerHits[no++] = sd.hitRows[k];
    }
    F.nPixelHits = np, F.nOuterHits = no;
    if (np < 3 || np + no >= ::mkseeder::kMaxSeedHits)
      return;
    const uint32_t iA = F.pixelHits[0], iB = F.pixelHits[np / 2], iC = F.pixelHits[np - 1];
    const float ax = H.x(iA), ay = H.y(iA);
    F.zAtFirstHit = H.z(iA);
    F.circle.make(acc, ax, ay, H.x(iB), H.y(iB), H.x(iC), H.y(iC));
    if (!F.circle.ok)
      return;
    const float s_last = F.circle.arcLengthOf(acc, H.x(iC), H.y(iC));
    if (s_last <= 0)
      return;
    F.cot = (H.z(iC) - F.zAtFirstHit) / s_last;
    F.s_max = P.pixelFillMode >= 2 ? 1e4f : s_last;
    const float pt = F.circle.lin ? 100.f : std::max(0.3f, kNominalPtPerRadius * F.circle.R);
    F.srp = P.pixelFillWeights[0] + P.pixelFillWeights[1] / pt,
    F.sq = P.pixelFillWeights[2] + P.pixelFillWeights[3] / pt;
    F.sig2 = P.pixelFillSigma * P.pixelFillSigma;
    F.rA = alpaka::math::sqrt(acc, ax * ax + ay * ay);
    F.hasWork = 1;
  }

#ifdef MKSEEDER_CPU_AVX2
  namespace avx2 {
    // std::round (half away from zero) for 8 lanes
    inline __m256 round8(__m256 x) {
      const __m256 t = _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
      const __m256 up = _mm256_cmp_ps(_mm256_and_ps(_mm256_sub_ps(x, t), absm), _mm256_set1_ps(0.5f), _CMP_GE_OQ);
      const __m256 one = _mm256_or_ps(_mm256_set1_ps(1.0f), _mm256_and_ps(x, _mm256_set1_ps(-0.0f)));
      return _mm256_add_ps(t, _mm256_and_ps(up, one));
    }
    // FillCircle::arcLengthOf (the circle) for 8 points
    inline __m256 arcLengthOnCircle8(const mkseeder::FillCircle &fc, __m256 x, __m256 y, __m256 lanes) {
      const __m256 d = _mm256_sub_ps(
          atan2f8(_mm256_sub_ps(y, _mm256_set1_ps(fc.yc)), _mm256_sub_ps(x, _mm256_set1_ps(fc.xc)), lanes),
          _mm256_set1_ps(fc.ang0));
      const __m256 w = _mm256_fnmadd_ps(
          round8(_mm256_mul_ps(d, _mm256_set1_ps(mkseeder::kInvTwoPi))), _mm256_set1_ps(mkseeder::kTwoPi), d);
      return _mm256_mul_ps(_mm256_mul_ps(w, _mm256_set1_ps(fc.sgn)), _mm256_set1_ps(fc.R));
    }
  }  // namespace avx2
  // the CPU backends, one thread: findFillCandidates's hit loop over the run [b, e) 8 hits at a time up to the score (the
  // operations of the scalar loop, FillCircle::positionAtR / positionAtArcLength with the compiler's FMA contractions; positionAtArcLength of a circle on
  // a disc by the scalar function); consume(ih, sc, s) for the hits that pass, in hit order
  template <typename TAcc, typename F>
  inline void scanFillHits8(const TAcc &acc,
                            const ChainFinderInputs &v,
                            const PostInputs &pv,
                            const mkseeder::FillCircle &fc,
                            const EventLayer &L,
                            uint32_t b,
                            uint32_t e,
                            bool disc,
                            float qc,
                            float qpad,
                            float lo,
                            float hi,
                            float Az,
                            float cot,
                            float s_lo,
                            float s_max,
                            float srp,
                            float sq,
                            float sig2,
                            F &&consume) {
    using namespace avx2;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)), zero = _mm256_setzero_ps();
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256 vqc = _mm256_set1_ps(qc), vqpad = _mm256_set1_ps(qpad), vlo = _mm256_set1_ps(lo),
                 vhi = _mm256_set1_ps(hi);
    const __m256 vslo = _mm256_set1_ps(s_lo), vsmax = _mm256_set1_ps(s_max), vAz = _mm256_set1_ps(Az),
                 vcot = _mm256_set1_ps(cot);
    const __m256 vax = _mm256_set1_ps(fc.ax), vay = _mm256_set1_ps(fc.ay), vux = _mm256_set1_ps(fc.ux),
                 vuy = _mm256_set1_ps(fc.uy);
    const bool inw = lo <= hi;
    for (uint32_t i0 = b; i0 < e; i0 += 8) {
      const uint32_t nk = std::min(8u, e - i0), ih0 = L.firstHit() + i0;
      __m256 keep = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32((int)nk), lane));
      const __m256 hr = _mm256_loadu_ps(pv.cpuHits.rGlobal().data() + ih0),
                   hz = _mm256_loadu_ps(v.hits.z().data() + ih0);
      keep = _mm256_and_ps(keep,
                           _mm256_cmp_ps(_mm256_and_ps(_mm256_sub_ps(disc ? hr : hz, vqc), absm), vqpad, _CMP_NGT_UQ));
      const __m256 ph = _mm256_loadu_ps(pv.cpuHits.phiGlobal().data() + ih0);
      const __m256 ge = _mm256_cmp_ps(ph, vlo, _CMP_GE_OQ), le = _mm256_cmp_ps(ph, vhi, _CMP_LE_OQ);
      keep = _mm256_and_ps(keep, inw ? _mm256_and_ps(ge, le) : _mm256_or_ps(ge, le));
      if (!_mm256_movemask_ps(keep))
        continue;
      __m256 px, py, ps, dq;
      if (disc) {
        ps = _mm256_div_ps(_mm256_sub_ps(hz, vAz), vcot);
        keep = _mm256_and_ps(
            keep, _mm256_and_ps(_mm256_cmp_ps(ps, vslo, _CMP_NLE_UQ), _mm256_cmp_ps(ps, vsmax, _CMP_NGE_UQ)));
        if (!_mm256_movemask_ps(keep))
          continue;
        if (fc.lin) {
          px = fma(ps, vux, vax), py = fma(ps, vuy, vay);
        } else {
          alignas(32) float pss[8], xs[8], ys[8];
          _mm256_store_ps(pss, ps);
          for (unsigned int mm = _mm256_movemask_ps(keep); mm; mm &= mm - 1) {
            const int l = std::countr_zero(mm);
            fc.positionAtArcLength(acc, pss[l], xs[l], ys[l]);
          }
          px = _mm256_load_ps(xs), py = _mm256_load_ps(ys);
        }
        dq = _mm256_sub_ps(hr, _mm256_sqrt_ps(fma(px, px, _mm256_mul_ps(py, py))));
      } else {
        __m256 c0x, c0y, c1x, c1y, s0, s1;
        if (fc.lin) {
          const __m256 vb = fma(vax, vux, _mm256_mul_ps(vay, vuy));
          const __m256 q = _mm256_fnmadd_ps(hr, hr, _mm256_set1_ps(fc.a2));
          const __m256 dsc = fms(vb, vb, q);
          keep = _mm256_andnot_ps(_mm256_cmp_ps(zero, dsc, _CMP_GT_OQ), keep);
          const __m256 sqv = _mm256_sqrt_ps(dsc);
          const __m256 t0 = _mm256_sub_ps(_mm256_xor_ps(vb, _mm256_set1_ps(-0.0f)), sqv), t1 = _mm256_sub_ps(sqv, vb);
          c0x = fma(vux, t0, vax), c0y = fma(t0, vuy, vay), c1x = fma(vux, t1, vax), c1y = fma(t1, vuy, vay);
          s0 = fma(_mm256_sub_ps(c0x, vax), vux, _mm256_mul_ps(_mm256_sub_ps(c0y, vay), vuy));
          s1 = fma(_mm256_sub_ps(c1x, vax), vux, _mm256_mul_ps(_mm256_sub_ps(c1y, vay), vuy));
        } else {
          const float xc = fc.xc, yc = fc.yc;
          const float dcs = alpaka::math::sqrt(acc, xc * xc + yc * yc);
          if (dcs < 1e-6f)
            return;
          const __m256 vxc = _mm256_set1_ps(xc), vyc = _mm256_set1_ps(yc);
          const __m256 dc = _mm256_sqrt_ps(fma(vxc, vxc, _mm256_mul_ps(vyc, vyc)));
          const __m256 vR = _mm256_set1_ps(fc.R);
          const __m256 aa = _mm256_div_ps(fma(dc, dc, fms(hr, hr, _mm256_mul_ps(vR, vR))), _mm256_add_ps(dc, dc));
          const __m256 h2 = fms(hr, hr, _mm256_mul_ps(aa, aa));
          keep = _mm256_andnot_ps(_mm256_cmp_ps(zero, h2, _CMP_GT_OQ), keep);
          const __m256 ex = _mm256_div_ps(vxc, dc), h = _mm256_sqrt_ps(h2), ey = _mm256_div_ps(vyc, dc);
          const __m256 hey = _mm256_mul_ps(h, ey), hex = _mm256_mul_ps(h, ex);
          c0x = fms(aa, ex, hey), c1x = fma(ex, aa, hey), c0y = fma(aa, ey, hex), c1y = fms(aa, ey, hex);
          s0 = arcLengthOnCircle8(fc, c0x, c0y, keep);
          s1 = arcLengthOnCircle8(fc, c1x, c1y, keep);
        }
        const __m256 v0 = _mm256_and_ps(_mm256_cmp_ps(s0, vslo, _CMP_GT_OQ), _mm256_cmp_ps(s0, vsmax, _CMP_LT_OQ));
        const __m256 v1 = _mm256_and_ps(_mm256_cmp_ps(s1, vslo, _CMP_GT_OQ), _mm256_cmp_ps(s1, vsmax, _CMP_LT_OQ));
        const __m256 take1 = _mm256_and_ps(v1,
                                           _mm256_or_ps(_mm256_xor_ps(v0, _mm256_castsi256_ps(_mm256_set1_epi32(-1))),
                                                        _mm256_cmp_ps(s1, s0, _CMP_LT_OQ)));
        keep = _mm256_and_ps(keep, _mm256_or_ps(v0, v1));
        px = sel(take1, c1x, c0x), py = sel(take1, c1y, c0y), ps = sel(take1, s1, s0);
        dq = _mm256_sub_ps(hz, fma(ps, vcot, vAz));
      }
      const __m256 rp = _mm256_sqrt_ps(fma(px, px, _mm256_mul_ps(py, py)));
      const __m256 gx = _mm256_loadu_ps(pv.hits.x().data() + ih0), gy = _mm256_loadu_ps(pv.hits.y().data() + ih0);
      const __m256 num = fms(px, gy, _mm256_mul_ps(py, gx));
      const __m256 drp = _mm256_div_ps(_mm256_mul_ps(num, hr), _mm256_mul_ps(hr, rp));
      const __m256 rs = _mm256_div_ps(drp, _mm256_set1_ps(srp)), rq = _mm256_div_ps(dq, _mm256_set1_ps(sq));
      const __m256 sc = fma(rs, rs, _mm256_mul_ps(rq, rq));
      keep = _mm256_and_ps(keep, _mm256_cmp_ps(sc, _mm256_set1_ps(sig2), _CMP_LT_OQ));
      alignas(32) float scs[8], pss[8];
      _mm256_store_ps(scs, sc), _mm256_store_ps(pss, ps);
      const unsigned int km = _mm256_movemask_ps(keep);
      for (unsigned int mm = km; mm; mm &= mm - 1) {
        const int ll = std::countr_zero(mm);
        consume(ih0 + ll, scs[ll], pss[ll]);
      }
    }
  }
#endif

  template <typename TAcc>
  ALPAKA_FN_ACC inline void findFillCandidates(const TAcc &acc,
                                               const ChainFinderInputs &v,
                                               const PostInputs &pv,
                                               const SeedHitAccessor &H,
                                               const FillPreparation &F,
                                               int l,
                                               FillCandidate &c) {
    const PostParameters &P = pv.parameters;
    c.bestHit = -1, c.secondBestHit = -1;
    const auto li = pv.layerGeometry[l];
    if (!li.isPixel() || v.layers[l].nHits() == 0)
      return;
    const mkseeder::FillCircle &fc = F.circle;
    const float Az = F.zAtFirstHit, cot = F.cot, s_max = F.s_max, rA = F.rA, srp = F.srp, sq = F.sq;
    bool has = false;
    int mod = -1;
    for (int k = 0; k < F.nPixelHits; ++k)
      if (F.pixelLayers[k] == l)
        has = true, mod = (int)pv.hits.detIdInLayer(F.pixelHits[k]);
    if (has && !P.pixelFillOverlaps)
      return;
    const float s_lo = has ? -2.f : 0.f;
    auto positionAtArcLength = [&](float s, float &x, float &y) { fc.positionAtArcLength(acc, s, x, y); };
    auto positionAtR = [&](float r, float &x, float &y, float &s) {
      return fc.positionAtR(acc, r, s_lo, s_max, x, y, s);
    };
    const bool disc = !li.isBarrel();
    // pix_fill_sig_disc > 0: the score cut of the disc layers
    const float sig2 = disc && P.pixelFillSigmaDisc > 0.f ? P.pixelFillSigmaDisc * P.pixelFillSigmaDisc : F.sig2;
    // the straight-line pre-check
    if (disc) {
      if (alpaka::math::abs(acc, cot) > 1e-6f) {
        const float rl = rA + (li.zMean() - Az) / cot;
        if (rl < li.rIn() - 3.f || rl > li.rOut() + 3.f)
          return;
      }
    } else {
      const float zl = Az + cot * (0.5f * (li.rIn() + li.rOut()) - rA);
      if (zl < li.zMin() - 3.f || zl > li.zMax() + 3.f)
        return;
    }
    float x, y, s;
    if (disc) {
      if (alpaka::math::abs(acc, cot) < 1e-6f)
        return;
      s = (li.zMean() - Az) / cot;
      if (!(s > s_lo && s < s_max))
        return;
      positionAtArcLength(s, x, y);
      const float r = alpaka::math::sqrt(acc, x * x + y * y);
      if (r < li.rIn() - 1.f || r > li.rOut() + 1.f)
        return;
    } else {
      if (!positionAtR(0.5f * (li.rIn() + li.rOut()), x, y, s))
        return;
      const float z = Az + cot * s;
      if (z < li.zMin() - 1.f || z > li.zMax() + 1.f)
        return;
    }
    const float rp = alpaka::math::sqrt(acc, x * x + y * y), pc = libm::atan2(acc, y, x);
    float qc, qpad;
    if (disc) {
      qc = rp;
      qpad =
          0.5f * (li.zMax() - li.zMin()) / std::max(alpaka::math::abs(acc, cot), 0.05f) + P.pixelFillSigma * sq + 0.5f;
    } else {
      qc = Az + cot * s;
      qpad = 0.5f * (li.rOut() - li.rIn()) * alpaka::math::abs(acc, cot) * 1.2f + P.pixelFillSigma * sq + 0.5f;
    }
    const float wphi =
        (P.pixelFillSigma * srp + 0.5f * alpaka::math::abs(acc, li.rOut() - li.rIn()) * 0.05f) / std::max(1.f, rp) +
        0.01f;
    const float lo = wrapPhiRound(acc, pc - wphi), hi = wrapPhiRound(acc, pc + wphi);
    // the best hit (no seed hit in the layer) and the best on another module (overlaps); ties: the lowest hit
    // (orig) index
    float best = sig2, best2 = sig2;
    int kb = -1, kb2 = -1, used = mod;
    float sb = s, sb2 = s;
    int kbMod = -1;
    // pass 1: the best hit overall (when the seed has none here); pass 2: the best on another module than `used`
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 0 && has)
        continue;
      if (pass == 1 && (!P.pixelFillOverlaps || used < 0))
        break;
      const EventLayer L = v.layers[l];
      // the q range in the hits' q (beam-relative r for the discs): widened by 0.1 cm
      const float ppad = layerPhiPad(acc, pv, l), rpad = 0.1f + pv.beamSpotRadius;
      // a hit that passed with its score and s
      auto consume = [&](uint32_t ih, float sc, float ps) {
        const int m = (int)pv.hits.detIdInLayer(ih);
        if (pass == 0) {
          if (sc < best || (sc == best && kb >= 0 && v.hits.clusterIndex(ih) < v.hits.clusterIndex(kb)))
            best = sc, kb = (int)ih, sb = ps, kbMod = m;
        } else if (m != used) {
          if (sc < best2 || (sc == best2 && kb2 >= 0 && v.hits.clusterIndex(ih) < v.hits.clusterIndex(kb2)))
            best2 = sc, kb2 = (int)ih, sb2 = ps;
        }
      };
      forEachBinRun(
          v,
          L,
          phiBinRangeAround(acc, pv.layerGeometry[l], pc, wphi + ppad),
          qBinRangePadded(acc, pv.layerGeometry[l], qc - qpad - (disc ? rpad : 0.1f), qc + qpad + (disc ? rpad : 0.1f)),
          [&](uint32_t b, uint32_t e) {
#ifdef MKSEEDER_CPU_AVX2
            if (pv.hasCpuCache) {
              scanFillHits8(
                  acc, v, pv, fc, L, b, e, disc, qc, qpad, lo, hi, Az, cot, s_lo, s_max, srp, sq, sig2, consume);
              return;
            }
#endif
            for (uint32_t i = b; i < e; ++i) {
              const uint32_t ih = L.firstHit() + i;
              const float hr = H.r(acc, ih), hz = H.z(ih);
              if (alpaka::math::abs(acc, (disc ? hr : hz) - qc) > qpad)
                continue;
              if (!isInPhiRange(H.phi(acc, ih), lo, hi))
                continue;
              float px, py, ps, dq;
              if (disc) {
                ps = (hz - Az) / cot;
                if (ps <= s_lo || ps >= s_max)
                  continue;
                positionAtArcLength(ps, px, py);
                dq = hr - alpaka::math::sqrt(acc, px * px + py * py);
              } else {
                if (!positionAtR(hr, px, py, ps))
                  continue;
                dq = hz - (Az + cot * ps);
              }
              const float drp = hr * (px * H.y(ih) - py * H.x(ih)) / (hr * alpaka::math::sqrt(acc, px * px + py * py));
              const float sc = (drp / srp) * (drp / srp) + (dq / sq) * (dq / sq);
              if (!(sc < sig2))
                continue;
              consume(ih, sc, ps);
            }
          });
      if (pass == 0 && kb >= 0)
        used = kbMod;
    }
    c.bestHit = kb, c.secondBestHit = kb2, c.bestPathLength = sb, c.secondBestPathLength = sb2;
  }  // findFillCandidates()

  // cand(j): the FillCandidate of the j-th layer id in increasing order, for j < ncand
  template <typename TAcc, typename FC>
  ALPAKA_FN_ACC inline void applyFill(const TAcc &acc,
                                      const ChainFinderInputs &v,
                                      const PostInputs &pv,
                                      const SeedHitAccessor &H,
                                      const FillPreparation &F,
                                      WorkingSeed &sd,
                                      int ncand,
                                      FC &&cand) {
    const PostParameters &P = pv.parameters;
    const int np = F.nPixelHits, no = F.nOuterHits;
    // the hits to add (s, layer, hit, an overlap flag), at most room of them (prepareFill: room >= 1): in the
    // candidates' order while they fit; once there are more, the room lowest in (s, orig) order, sorted. That is the
    // host's choice (all candidates sorted by s, the first room kept), with a stack of ::mkseeder::kMaxSeedHits entries instead of
    // 2 x kMaxLayerIds (the GPU reserves the largest per-thread stack of any kernel for every resident thread)
    const int room = ::mkseeder::kMaxSeedHits - (np + no);
    float as_[::mkseeder::kMaxSeedHits + 1];
    int16_t al[::mkseeder::kMaxSeedHits + 1];
    uint32_t ah[::mkseeder::kMaxSeedHits + 1];
    bool aov[::mkseeder::kMaxSeedHits + 1];
    int na = 0, nk = 0;  // na: the candidates seen, nk: those kept
    auto before = [&](int b, int a) {
      return as_[b] < as_[a] || (as_[b] == as_[a] && v.hits.clusterIndex(ah[b]) < v.hits.clusterIndex(ah[a]));
    };
    auto swapAdd = [&](int a, int b) {
      std::swap(as_[a], as_[b]), std::swap(al[a], al[b]), std::swap(ah[a], ah[b]), std::swap(aov[a], aov[b]);
    };
    auto add = [&](float s, int l, uint32_t h, bool ov) {
      if (na >= 2 * kMaxLayerIds)
        return;
      ++na;
      as_[nk] = s, al[nk] = l, ah[nk] = h, aov[nk] = ov, ++nk;
      if (na <= room)
        return;
      // the first overflow: the insertion sort of all so far; then the insertion of each new one
      for (int a = na == room + 1 ? 1 : nk - 1; a < nk; ++a)
        for (int b = a; b > 0 && before(b, b - 1); --b)
          swapAdd(b, b - 1);
      nk = room;
    };
    for (int j = 0; j < ncand; ++j) {
      int l;
      const FillCandidate c = cand(j, l);
      if (c.bestHit >= 0)
        add(c.bestPathLength, l, (uint32_t)c.bestHit, false);
      if (c.secondBestHit >= 0)
        add(c.secondBestPathLength, l, (uint32_t)c.secondBestHit, true);
    }
    if (na == 0)
      return;
    na = nk;
    // the pixel hits in the order of s (stable: the seed's first, then the added in their order), then the others
    float ss[::mkseeder::kMaxSeedHits];
    int16_t sl[::mkseeder::kMaxSeedHits];
    uint32_t sh[::mkseeder::kMaxSeedHits];
    bool sov[::mkseeder::kMaxSeedHits];
    int n = 0;
    for (int k = 0; k < np; ++k)
      ss[n] = F.circle.arcLengthOf(acc, H.x(F.pixelHits[k]), H.y(F.pixelHits[k])), sl[n] = F.pixelLayers[k],
      sh[n] = F.pixelHits[k], sov[n] = false, ++n;
    for (int a = 0; a < na; ++a)
      ss[n] = as_[a], sl[n] = al[a], sh[n] = ah[a], sov[n] = aov[a], ++n;
    // stable insertion sort by s
    for (int a = 1; a < n; ++a)
      for (int b = a; b > 0 && ss[b] < ss[b - 1]; --b) {
        std::swap(ss[b], ss[b - 1]), std::swap(sl[b], sl[b - 1]), std::swap(sh[b], sh[b - 1]),
            std::swap(sov[b], sov[b - 1]);
      }
    // pix_fill_ov 2: no overlap hit before the third distinct layer
    int m = 0, nlay = 0, prev = -1, ndrop = 0;
    for (int a = 0; a < n; ++a) {
      if (sov[a] && P.pixelFillOverlaps >= 2 && nlay < 3) {
        ++ndrop;
        continue;
      }
      if (sl[a] != prev)
        ++nlay, prev = sl[a];
      sd.layers[m] = sl[a], sd.hitRows[m] = sh[a], ++m;
    }
    if (m == np && ndrop > 0 && na == ndrop) {
      // only dropped overlaps: nothing added; restore the input order (pixel then others)
      for (int k = 0; k < np; ++k)
        sd.layers[k] = F.pixelLayers[k], sd.hitRows[k] = F.pixelHits[k];
      for (int k = 0; k < no; ++k)
        sd.layers[np + k] = F.outerLayers[k], sd.hitRows[np + k] = F.outerHits[k];
      sd.nHits = np + no;
      return;
    }
    for (int k = 0; k < no; ++k)
      sd.layers[m] = F.outerLayers[k], sd.hitRows[m] = F.outerHits[k], ++m;
    sd.nHits = m;
    sd.isFilled = 1;
  }

  // the three steps in one go
  template <typename TAcc>
  ALPAKA_FN_ACC inline void fillPixelHits(
      const TAcc &acc, const ChainFinderInputs &v, const PostInputs &pv, const SeedHitAccessor &H, WorkingSeed &sd) {
    FillPreparation F;
    prepareFill(acc, pv, H, sd, F);
    if (!F.hasWork)
      return;
    applyFill(acc, v, pv, H, F, sd, pv.nPixelLayersWithHits, [&](int j, int &l) {
      l = pv.layers.pixelLayerWithHits(j);
      FillCandidate c;
      findFillCandidates(acc, v, pv, H, F, l, c);
      return c;
    });
  }

  // the seeds of the post step's kernels: the first *count (the device's number; nullptr: all) of cap
  ALPAKA_FN_ACC inline uint32_t seedCount(uint32_t cap, const uint32_t *count) {
    return count && *count < cap ? *count : cap;
  }

  // The extension and the pixel fill of the first *count (nullptr: all) of cap seeds, one thread per seed (the CPU
  // backends; a GPU: KernelExtendBlock, then the fill in KernelFillLayers and KernelFillFinish)
  class KernelExtendAndFillSeeds {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  ChainFinderInputs v,
                                  PostInputs pv,
                                  SeedHitsView seeds,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = seedCount(cap, count);
      const SeedHitAccessor H{&v, &pv};
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        WorkingSeed sd;
        loadSeed(seeds, i, sd);
        sd.isFilled = 0, sd.nExtensionHits = 0;
        const int n0 = sd.nHits;
        sd.isKept = extendSeedIntoOuterTracker(acc, v, pv, H, sd) ? 1 : 0;
        sd.isExtended = sd.nHits > n0;
        if (sd.isKept)
          fillPixelHits(acc, v, pv, H, sd);
        storeSeed(seeds, i, sd);
      }
    }
  };

  // The extension with a block of threads per seed (the windows' hits shared among them)
  // (a block is one 32-wide warp on CUDA, one 64-wide wavefront on HIP; there is no warp-synchronous code, the
  // reductions use the shared arrays and block synchronisations, and the result does not depend on the number of lanes)
  template <typename TAcc>
  constexpr uint32_t kExtLanes = alpaka::accMatchesTags<TAcc, alpaka::TagGpuHipRt> ? 64 : 32;
  class KernelExtendBlock {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  ChainFinderInputs v,
                                  PostInputs pv,
                                  SeedHitsView seeds,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = seedCount(cap, count);
      const SeedHitAccessor H{&v, &pv};
      auto &sbest = alpaka::declareSharedVar<float[kExtLanes<TAcc>], __COUNTER__>(acc);
      auto &skb = alpaka::declareSharedVar<int[kExtLanes<TAcc>], __COUNTER__>(acc);
      const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const uint32_t W = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      for (uint32_t i : cms::alpakatools::independent_groups(acc, n)) {
        WorkingSeed sd;
        loadSeed(seeds, i, sd);
        sd.isFilled = 0, sd.nExtensionHits = 0;
        const int n0 = sd.nHits;
        auto reduce = [&](float &best, int &kb) {
          sbest[t] = best, skb[t] = kb;
          alpaka::syncBlockThreads(acc);
          float bb = 0.f;
          int kk = -1;
          for (uint32_t u = 0; u < W; ++u) {
            const int k = skb[u];
            if (k >= 0 &&
                (kk < 0 || sbest[u] < bb || (sbest[u] == bb && v.hits.clusterIndex(k) < v.hits.clusterIndex(kk))))
              bb = sbest[u], kk = k;
          }
          alpaka::syncBlockThreads(acc);
          if (kk >= 0)
            best = bb;
          kb = kk;
        };
        const bool keep = extendBlockParallel(acc, v, pv, H, sd, t, W, reduce);
        if (t == 0) {
          sd.isKept = keep ? 1 : 0;
          sd.isExtended = sd.nHits > n0;
          storeSeed(seeds, i, sd);
        }
      }
    }
  };

  // a GPU: the fill's first step (prepareFill) of the kept seeds after KernelExtendBlock, one thread per seed (out of
  // KernelExtendBlock, whose registers it raised)
  class KernelPrepareFills {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  ChainFinderInputs v,
                                  PostInputs pv,
                                  SeedHitsConstView seeds,
                                  FillPrepsView preps,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = seedCount(cap, count);
      const SeedHitAccessor H{&v, &pv};
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        FillPreparation &F = preps.prep(i);
        F.hasWork = 0;
        if (!seeds[i].isKept())
          continue;
        WorkingSeed sd;
        loadSeed(seeds, i, sd);
        prepareFill(acc, pv, H, sd, F);
      }
    }
  };

  class KernelFillLayers {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  ChainFinderInputs v,
                                  PostInputs pv,
                                  FillPrepsView preps,
                                  FillCandidatesView cands,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = seedCount(cap, count);
      const SeedHitAccessor H{&v, &pv};
      const uint32_t nl = pv.nPixelLayersWithHits;
      for (uint32_t t : cms::alpakatools::uniform_elements(acc, n * nl)) {
        const uint32_t i = t / nl, j = t % nl;
        FillCandidate c;
        c.bestHit = -1, c.secondBestHit = -1;
        const FillPreparation &F = preps.prep(i);
        if (F.hasWork)
          findFillCandidates(acc, v, pv, H, F, pv.layers.pixelLayerWithHits(j), c);
        cands.candidate(t) = c;
      }
    }
  };

  class KernelFillFinish {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  ChainFinderInputs v,
                                  PostInputs pv,
                                  SeedHitsView seeds,
                                  FillPrepsView preps,
                                  FillCandidatesView cands,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = seedCount(cap, count);
      const SeedHitAccessor H{&v, &pv};
      const uint32_t nl = pv.nPixelLayersWithHits;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const FillPreparation &F = preps.prep(i);
        if (!F.hasWork)
          continue;
        WorkingSeed sd;
        loadSeed(seeds, i, sd);
        applyFill(acc, v, pv, H, F, sd, nl, [&](int j, int &l) {
          l = pv.layers.pixelLayerWithHits(j);
          return cands.candidate(i * nl + j);
        });
        storeSeed(seeds, i, sd);
      }
    }
  };

  // the seeds' input hits (the refit's fallback)
  class KernelBackupSeedHits {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, SeedHitsConstView seeds, SeedHitListView backup, uint32_t cap, const uint32_t *count) const {
      const uint32_t n = seedCount(cap, count);
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const auto s = seeds[i];
        auto b = backup[i];
        b.nHits() = s.nHits();
        for (int k = 0; k < s.nHits(); ++k)
          b.hitLayers()(k) = s.hitLayers()(k), b.hitRows()(k) = s.hitRows()(k);
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
