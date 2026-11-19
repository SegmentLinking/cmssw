#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederKernels_h

// The kernels of the chain finder (ported from mkFit's SeedChainFinder), one z side at a time (see
// MkSeederDeviceTypes.h):
//   KernelFindStartDoublets   stage b: one work item per (start pair, a hit), the doublets that pass the start tests,
//                 routed onto the Q2 queue of their next crossed position;
//   KernelExtendToTriplets  stage c at position p: one item per Q2[p] candidate, triplets onto Q3, misses onto Q2;
//   KernelExtendToQuads  stage d at position p: one item per Q3[p] candidate, quads and triplet seeds out, misses
//                 onto Q3.
// Positions run in chain order (a candidate only moves to later positions), so every producer of a queue has
// run before the queue is consumed.

#include <bit>
#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederBackend.h"
#include "MkSeederPhysicsConstants.h"
#include "MkSeederDeviceTypes.h"
#include "MkSeederQueues.h"
#include "MkSeederPostMath.h"
#include "MkSeederStageCNet.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  constexpr double kTwoPiD = 2 * 3.14159265358979323846;

  // wrap d into [-pi, pi) (floor form)
  template <typename TAcc>
  ALPAKA_FN_ACC inline float wrapPhiFloor(const TAcc &acc, float d) {
    return d - kTwoPi * alpaka::math::floor(acc, d * kInvTwoPi + 0.5f);
  }
  // wrap p once into [-pi, pi]
  ALPAKA_FN_HOST_ACC inline float wrapPhiOnce(float p) {
    if (p > kPi)
      p -= 2 * kPi;
    else if (p < -kPi)
      p += 2 * kPi;
    return p;
  }

  struct BinRange {
    uint32_t first, last;
  };

  // a layer's static description: its kind, extent and binning axes (MkSeederLayerGeometrySoA)
  using LayerAxes = LayerGeometryView::const_element;
  // a chain position's layer and crossing envelope (MkSeederChainPositionsSoA)
  using ChainPosition = ChainPositionsView::const_element;

  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange phiBinRangeFromEdges(const TAcc &acc, const LayerAxes &G, float lo, float hi) {
    lo = wrapPhiOnce(lo), hi = wrapPhiOnce(hi);
    const uint32_t m = ::mkseeder::kNPhiBins - 1;
    const uint32_t b = ((uint32_t)(uint16_t)alpaka::math::floor(acc, (lo - G.phiAxisMin()) * G.phiAxisScale())) & m;
    const uint32_t e =
        ((uint32_t)(uint16_t)alpaka::math::floor(acc, (hi - G.phiAxisMin()) * G.phiAxisScale()) + 1u) & m;
    return {b, e};
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline uint32_t qBinClamped(const TAcc &acc, const LayerAxes &G, float r) {
    return r <= G.qAxisMin()
               ? 0u
               : (r >= G.qAxisLastBinHighEdge()
                      ? (uint32_t)G.qLastBin()
                      : (uint32_t)(uint16_t)alpaka::math::floor(acc, (r - G.qAxisMin()) * G.qAxisScale()));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange qBinRangeFromEdges(const TAcc &acc, const LayerAxes &G, float lo, float hi) {
    return {qBinClamped(acc, G, lo), qBinClamped(acc, G, hi) + 1u};
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange phiBinRangeAround(const TAcc &acc, const LayerAxes &G, float c, float w) {
    w = std::min(w, 0.9f * kPi);
    return phiBinRangeFromEdges(acc, G, c - w, c + w);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange qBinRangePadded(const TAcc &acc, const LayerAxes &G, float lo, float hi) {
    return qBinRangeFromEdges(acc, G, lo - 1e-4f, hi + 1e-4f);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange phiBinRangeAroundDouble(const TAcc &acc, const LayerAxes &G, double c, double w) {
    w = std::min(w, 0.9 * kTwoPiD / 2);
    return phiBinRangeFromEdges(acc, G, (float)(c - w), (float)(c + w));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange qBinRangePaddedDouble(const TAcc &acc, const LayerAxes &G, double lo, double hi) {
    return qBinRangeFromEdges(acc, G, (float)(lo - 1e-4), (float)(hi + 1e-4));
  }

  // calls f(begin, end) for each contiguous run of the layer's own hit indices in the (phi, q) window
  template <typename F>
  ALPAKA_FN_HOST_ACC inline void forEachBinRun(
      const ChainFinderInputs &finder, const EventLayer &L, BinRange p, BinRange q, F &&f) {
    const uint32_t np = ::mkseeder::kNPhiBins;
    const uint32_t *st = finder.binStarts.start().data() + L.firstBinStart();
    for (uint32_t qi = q.first; qi < q.last; ++qi) {
      if (p.first < p.last || p.last == 0) {
        f(st[qi * np + p.first], st[qi * np + (p.last == 0 ? np : p.last)]);
      } else if (p.first > p.last) {
        f(st[qi * np + p.first], st[qi * np + np]);
        f(st[qi * np + 0], st[qi * np + p.last]);
      }
    }
  }

  // GPU: a per-block copy of the kernel argument v in shared memory (all threads of the block must call it); a CPU
  // backend: v itself. For the lambdas of a kernel that the GPU compiler does not inline: a non-inlined lambda that
  // captured the by-value kernel argument by reference would need its address, i.e. a copy of the whole argument
  // (ChainFinderInputs, ~2 kB) on every thread's stack, which the driver reserves for all resident threads.
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC inline const T &blockCopyOfArgument(const TAcc &acc, const T &v) {
    if constexpr (kGpuAcc<TAcc>) {
      auto &s = alpaka::declareSharedVar<T, __COUNTER__>(acc);
      if (cms::alpakatools::once_per_block(acc))
        s = v;
      alpaka::syncBlockThreads(acc);
      return s;
    } else {
      return v;
    }
  }

  // the crossing class (states) of a line with the end points x1, x2 for a chain position
  ALPAKA_FN_HOST_ACC inline int crossingClass(const ChainPosition &e, float x1, float x2) {
    const float xl = std::min(x1, x2), xh = std::max(x1, x2);
    return ((xh > e.outerLow()) & (xl < e.outerHigh())) + ((xl > e.innerLow()) & (xh < e.innerHigh()));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline int state(const TAcc &acc, const ChainPosition &e, float z0, float c) {
    if (!e.isDisc())
      return crossingClass(e, z0 + c * e.positionLow(), z0 + c * e.positionHigh());
    const float cs = alpaka::math::abs(acc, c) >= 1e-9f ? c : 1.0f,
                ic = alpaka::math::abs(acc, c) >= 1e-9f ? 1.0f / cs : 0.0f;
    const float zr = -z0 * ic;
    const int valid = (ic != 0) & (e.position() * ic + zr > 0);
    return valid * crossingClass(e, e.positionLow() * ic + zr, e.positionHigh() * ic + zr);
  }
  // the features and score of the triplet a, b, c (hit rows), the weights from the ES
  static_assert(kTripletMlpNFeatures == MkSeederTripletMlpWeights::kNFeatures &&
                kTripletMlpNHidden1 == MkSeederTripletMlpWeights::kNHidden1 &&
                kTripletMlpNHidden2 == MkSeederTripletMlpWeights::kNHidden2);
  template <typename TAcc>
  ALPAKA_FN_ACC inline float scoreTriplet(
      const TAcc &acc, const ChainFinderInputs &finder, uint32_t ia, uint32_t ib, uint32_t ic) {
    const uint32_t ih[3] = {ia, ib, ic};
    float x[3], y[3], z[3], f[kTripletMlpNFeatures];
    int rows[3], cols[3];
    for (int i = 0; i < 3; ++i)
      x[i] = finder.hits.x(ih[i]), y[i] = finder.hits.y(ih[i]), z[i] = finder.hits.z(ih[i]),
      rows[i] = finder.hits.clusterRows(ih[i]), cols[i] = finder.hits.clusterCols(ih[i]);
    if (!mkseeder::computeTripletFeatures(acc,
                                          x,
                                          y,
                                          z,
                                          rows,
                                          cols,
                                          finder.chainEvent.beamSpotX,
                                          finder.chainEvent.beamSpotY,
                                          finder.chainEvent.beamSpotZ,
                                          f))
      return -1.f;
    const MkSeederTripletMlpWeights &w = *finder.tripletMlpWeights;
    foldTripletFeatures(f, w.zFold);
    return mkseeder::evaluateTripletMlp(acc,
                                        f,
                                        w.featureMean,
                                        w.featureScale,
                                        w.hidden1Weights,
                                        w.hidden1Bias,
                                        w.hidden2Weights,
                                        w.hidden2Bias,
                                        w.outputWeights,
                                        &w.outputBias);
  }

  // nextCrossedPosition(): the next crossed position after p (a non-pixel one only with allow), and its state; with ot_max_pix,
  // the outer-tracker layers beyond OT1-P only for a line that definitely crosses at most ot_max_pix pixel layers
  template <typename TAcc>
  ALPAKA_FN_ACC inline int nextCrossedPosition(
      const TAcc &acc, const ChainFinderInputs &finder, int p, float z0, float c, int allow, int &st) {
    int xok = -1;
    for (int q = p + 1; q < finder.chain.nPositions; ++q) {
      const ChainPosition e = finder.chainPositions[q];
      if (!e.envelopeValid())
        continue;
      const int s = state(acc, e, z0, c);
      if ((s > 0) & (e.isPixel() | allow)) {
        if (e.isExtraOuterTracker() && finder.chain.maxPixelLayersForExtraOuterTracker > 0) {
          if (xok < 0) {
            int np = 0;
            for (int r = 0; r < finder.chain.nPositions; ++r) {
              const ChainPosition er = finder.chainPositions[r];
              np += er.envelopeValid() && er.isPixel() && state(acc, er, z0, c) == 2;
            }
            xok = np <= finder.chain.maxPixelLayersForExtraOuterTracker;
          }
          if (!xok)
            continue;
        }
        st = s;
        return q;
      }
    }
    st = 0;
    return -1;
  }

  // the d0 of stage c's windows at chain position p (an outer-tracker position: d0_max_ot)
  ALPAKA_FN_HOST_ACC inline float stageCD0Max(const ChainFinderInputs &finder, int p) {
    return finder.chainPositions[p].isPixel() ? finder.chain.d0Max : finder.chain.d0MaxOuterTracker;
  }
  // the pixel hits a quad with its third hit at chain position p needs (min_pix; min_pix_ot for an outer-tracker
  // third hit); 0: no requirement
  ALPAKA_FN_HOST_ACC inline int minPixelHitsOfThird(const ChainFinderInputs &finder, int p) {
    return finder.chain.minPixelHitsOuterTrackerThird > 0 && !finder.chainPositions[p].isPixel()
               ? finder.chain.minPixelHitsOuterTrackerThird
               : finder.chain.minPixelHits;
  }
  // min_pix_ot: a doublet that found a third hit at its position does not go on to an outer-tracker position nq
  ALPAKA_FN_HOST_ACC inline bool skipOuterTrackerMiss(const ChainFinderInputs &finder, bool found, int nq) {
    return found && finder.chain.minPixelHitsOuterTrackerThird > 0 && !finder.chainPositions[nq].isPixel();
  }
  // miss_gap_dz > 0: a TBPX L1-L2 doublet (c, line z0 + cot r) that misses TBPX L3 (position p) goes on (to L4: the
  // L1-L2-L4 patterns) only if its line at L3's mid radius, within +- miss_gap_dz, touches a z gap between the ladders'
  // modules (the gap map's z gaps of L3; outside the map: it goes on)
  ALPAKA_FN_HOST_ACC inline bool skipMissOutsideGap(
      const ChainFinderInputs &finder, const Candidate &c, int p, float z0, float cot) {
    const float dz = finder.chain.missGapDz;
    if (!(dz > 0.f) || c.positions[2] != 0 || finder.chainPositions[p].layerId() != 2 ||
        finder.chainPositions[c.positions[0]].layerId() != 0 || finder.chainPositions[c.positions[1]].layerId() != 1)
      return false;
    const auto G = finder.gapMapLayers[2];
    const int nzg = G.nZGaps();
    if (!G.valid() || nzg <= 0)
      return false;
    const float z = z0 + cot * G.rMid();
    int k1 = (int)std::floor((z - dz - G.zGap0()) * (1.0f / 0.005f)),
        k2 = (int)std::floor((z + dz - G.zGap0()) * (1.0f / 0.005f));
    if (k1 < 0 || k2 >= nzg)
      return false;
    const uint8_t *zg = finder.gapMapZGaps.inGap().data() + G.firstZGap();
    for (int k = k1; k <= k2; ++k)
      if (zg[k])
        return false;
    return true;
  }

  // the shape band of chain position p's layer (a barrel pixel layer), or -1: none
  ALPAKA_FN_HOST_ACC inline int shapeOfPosition(const ChainFinderInputs &finder, int p) {
    const int l = finder.chainPositions[p].layerId();
    return finder.chain.useShapeCut && l >= 0 && l < ::mkseeder::kNShapeLayers && finder.shapeLayers[l].valid() ? l
                                                                                                                : -1;
  }
  ALPAKA_FN_HOST_ACC inline int shapeBin(const ChainFinderInputs &finder, int l, float acot) {
    return std::min((int)(acot * finder.shapeLayers[l].invBinWidth()), finder.shapeLayers[l].nBins() - 1);
  }
  // the band of layer l's |cot| bin b: its row in the bands
  ALPAKA_FN_HOST_ACC inline int shapeBand(const ChainFinderInputs &finder, int l, int b) {
    return finder.shapeLayers[l].firstBin() + b;
  }
  // a hit tagged for its charge as a merged-cluster candidate passes the band's upper limit (mrg_shape)
  ALPAKA_FN_HOST_ACC inline int shapeExempt(const ChainFinderInputs &finder, uint32_t ih) {
    return finder.chain.mergedShapeExempt && (finder.hits.mergedTag(ih) & 1);
  }
  ALPAKA_FN_HOST_ACC inline int passesShapeCut(const ChainFinderInputs &finder, int l, uint32_t ih, float acot) {
    const int span = finder.hits.clusterCols(ih);
    const int band = shapeBand(finder, l, shapeBin(finder, l, acot));
    return (span >= finder.shapeBands[band].spanLow()) &
           ((span <= finder.shapeBands[band].spanHigh()) | shapeExempt(finder, ih));
  }

  // SensorGapMap::segmentInGap on gap map layer lid
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool segmentInGap(
      const TAcc &acc, const ChainFinderInputs &finder, int lid, const float a[3], const float b[3], float phi) {
    constexpr float kPiM = (float)M_PI;
    constexpr int kGapMapNPhiBins = ::mkseeder::kGapMapNPhiBins;
    int bin = (int)((phi + kPiM) * (kGapMapNPhiBins / (2 * kPiM)));
    bin = std::min(std::max(bin, 0), kGapMapNPhiBins - 1);
    const auto layer = finder.gapMapLayers[lid];
    const float mg = layer.margin();
    const int firstPhiStart = layer.firstPhiStart(), firstLadderIndex = layer.firstLadderIndex();
    const int kEnd = finder.gapMapPhiStarts[firstPhiStart + bin + 1].firstEntry();
    for (int k = finder.gapMapPhiStarts[firstPhiStart + bin].firstEntry(); k < kEnd; ++k) {
      const uint32_t li = uint32_t(layer.firstLadder()) + finder.gapMapLadderIndex[firstLadderIndex + k].ladder();
      const auto q = finder.gapMapLadders[li];
      const float fa = q.normalX() * a[0] + q.normalY() * a[1] - q.distance(),
                  fb = q.normalX() * b[0] + q.normalY() * b[1] - q.distance();
      if ((fa > 0) == (fb > 0))
        continue;
      const float s = fa / (fa - fb);
      const float x = a[0] + (b[0] - a[0]) * s, y = a[1] + (b[1] - a[1]) * s, z = a[2] + (b[2] - a[2]) * s;
      if (alpaka::math::abs(acc, q.alongX() * x + q.alongY() * y - q.alongOffset()) > q.halfWidth() - mg)
        continue;
      const int kz = (int)alpaka::math::floor(acc, (z - q.zFirstBin()) * (1.0f / 0.005f));
      const int nz = q.nZBins();
      if (kz >= 0 && kz < nz && ((finder.gapMapBits[q.firstBitWord() + (kz >> 6)].word() >> (kz & 63)) & 1ull))
        return false;
    }
    return true;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC inline float asinOx(const TAcc &acc, float x) {
    const float x2 = x * x;
    if (x2 < 0.04f)
      return 1 + x2 * (1.f / 6 + x2 * (3.f / 40 + x2 * (5.f / 112 + x2 * (35.f / 1152 + x2 * (63.f / 2816)))));
    return x >= 1 ? kPi / 2 : libm::asin(acc, x) / x;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline void sincCosSin(const TAcc &acc, float h, float &sc, float &c, float &s) {
    const float h2 = h * h;
    if (h2 < 0.04f) {
      sc = 1 - h2 * (1.f / 6 - h2 * (1.f / 120 - h2 * (1.f / 5040)));
      c = 1 - h2 * (0.5f - h2 * (1.f / 24 - h2 * (1.f / 720 - h2 * (1.f / 40320))));
      s = h * sc;
    } else {
      s = alpaka::math::sin(acc, h), c = alpaka::math::cos(acc, h);
      sc = s / h;
    }
  }

  // seedchain::HelixF
  struct HelixF {
    float k = 0, tx = 0, ty = 0, cx = 0, cy = 0, cz = 0, cot = 0;
    float ux = 0, uy = 0, im = 0, c2 = 0, cn = 0;
    bool ok = false;
    // positionAtR / positionAtZ's per-helix factors, once (the same expressions, so the same values): used on the CPU backends,
    // computed where they are needed on a GPU (fewer registers); one layout everywhere
    float hk = 0, ahk = 0, sq = 0;

    template <typename TAcc>
    ALPAKA_FN_ACC void make(
        const TAcc &acc, float ax, float ay, float az, float bx, float by, float cx_, float cy_, float cz_) {
      cx = cx_, cy = cy_, cz = cz_;
      const float vx0 = bx - ax, vy0 = by - ay, vx = cx - bx, vy = cy - by;
      const float lu = alpaka::math::sqrt(acc, vx0 * vx0 + vy0 * vy0), lv = alpaka::math::sqrt(acc, vx * vx + vy * vy);
      const float wx = cx - ax, wy = cy - ay, lw = alpaka::math::sqrt(acc, wx * wx + wy * wy);
      ok = lu >= 1e-6f && lv >= 1e-6f;
      if (!ok)
        return;
      k = 2 * (vx0 * vy - vy0 * vx) / (lu * lv * lw);
      const float sn = std::min(std::max(0.5f * k * lv, -1.0f), 1.0f), cs = alpaka::math::sqrt(acc, 1 - sn * sn);
      const float ex = vx / lv, ey = vy / lv;
      tx = ex * cs - ey * sn;
      ty = ex * sn + ey * cs;
      const float ak = alpaka::math::abs(acc, k);
      const float s_ab = lu * asinOx(acc, std::min(1.0f, 0.5f * ak * lu)),
                  s_bc = lv * asinOx(acc, std::min(1.0f, 0.5f * ak * lv));
      cot = (cz - az) / (s_ab + s_bc);
      const float mx = k * cx - ty, my = k * cy + tx;
      im = 1.0f / alpaka::math::sqrt(acc, mx * mx + my * my);
      ux = mx * im, uy = my * im;
      c2 = cx * cx + cy * cy;
      cn = -cx * ty + cy * tx;
      if constexpr (kCpuAcc<TAcc>)
        hk = 0.5f * k, ahk = 0.5f * alpaka::math::abs(acc, k), sq = alpaka::math::sqrt(acc, 1 + cot * cot);
    }
    template <typename TAcc>
    ALPAKA_FN_ACC float pt(const TAcc &acc) const {
      return alpaka::math::abs(acc, k) > 1e-12f ? kNominalPtPerRadius / alpaka::math::abs(acc, k) : 1e9f;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC bool positionAtR(const TAcc &acc, float r, float &px, float &py, float &q, float &s3) const {
      const float A = (kCpuAcc<TAcc> ? (hk * (r * r + c2) + cn) : (0.5f * k * (r * r + c2) + cn)) * im,
                  g2 = r * r - A * A;
      if (g2 < 0)
        return false;
      const float g = alpaka::math::sqrt(acc, g2);
      const float p1x = A * ux - g * uy, p1y = A * uy + g * ux, p2x = A * ux + g * uy, p2y = A * uy - g * ux;
      const float d1x = p1x - cx, d1y = p1y - cy, d2x = p2x - cx, d2y = p2y - cy;
      const bool a1 = d1x * tx + d1y * ty > 0, a2 = d2x * tx + d2y * ty > 0;
      const float L1 = d1x * d1x + d1y * d1y, L2 = d2x * d2x + d2y * d2y;
      float L2s;
      if (a1 && (!a2 || L1 <= L2))
        px = p1x, py = p1y, L2s = L1;
      else if (a2)
        px = p2x, py = p2y, L2s = L2;
      else
        return false;
      const float L = alpaka::math::sqrt(acc, L2s);
      const float s = L * asinOx(acc, std::min(1.0f, kCpuAcc<TAcc> ? ahk * L : 0.5f * alpaka::math::abs(acc, k) * L));
      q = cz + cot * s;
      s3 = kCpuAcc<TAcc> ? s * sq : s * alpaka::math::sqrt(acc, 1 + cot * cot);
      return s > 0;
    }
    template <typename TAcc>
    ALPAKA_FN_ACC bool positionAtZ(const TAcc &acc, float u, float &px, float &py, float &q, float &s3) const {
      if (alpaka::math::abs(acc, cot) < 1e-9f)
        return false;
      const float ds = (u - cz) / cot;
      if (ds <= 0)
        return false;
      float sc, ch, sh;
      sincCosSin(acc, kCpuAcc<TAcc> ? hk * ds : 0.5f * k * ds, sc, ch, sh);
      const float f = ds * sc;
      px = cx + f * (ch * tx - sh * ty);
      py = cy + f * (sh * tx + ch * ty);
      q = alpaka::math::sqrt(acc, px * px + py * py);
      s3 = kCpuAcc<TAcc> ? ds * sq : ds * alpaka::math::sqrt(acc, 1 + cot * cot);
      return true;
    }
    template <typename TAcc>
    ALPAKA_FN_ACC bool at(const TAcc &acc, bool disc, float u, float &px, float &py, float &q, float &s3) const {
      return disc ? positionAtZ(acc, u, px, py, q, s3) : positionAtR(acc, u, px, py, q, s3);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

// HelixF for 8 lanes on the CPU backends
#include "MkSeederHelix8.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // the PS-module confirmation: true if the hit row ih of mkFit layer l is an OT1-3 P hit without a confirming S hit
  ALPAKA_FN_HOST_ACC inline bool psUnmarked(const ChainFinderInputs &finder, int l, uint32_t ih) {
    if (l != 4 && l != 6 && l != 8)
      return false;
    const int c0 = finder.layers[l].firstOTHit();
    return c0 >= 0 && finder.otHits.psMatches(uint32_t(c0) + (ih - finder.layers[l].firstHit())) == 0;
  }
  // ps_c2: true if stage c rejects the hit row ih at chain position p for the candidate c: an unmarked OT1-3 P hit as
  // the third hit of a two-pixel start whose three hits' circle (in the beam frame) has pT >= ps_seed_pt
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool psRejectStageC2(
      const TAcc &acc, const ChainFinderInputs &finder, const Candidate &c, int p, uint32_t ih) {
    if (!finder.psStageC2 || !finder.chainPositions[c.positions[0]].isPixel() ||
        !finder.chainPositions[c.positions[1]].isPixel() || !psUnmarked(finder, finder.chainPositions[p].layerId(), ih))
      return false;
    const uint32_t ia = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()].firstHit() + c.hits[0],
                   ib = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()].firstHit() + c.hits[1];
    const float ux = finder.hits.xBeam(ib) - finder.hits.xBeam(ia), uy = finder.hits.yBeam(ib) - finder.hits.yBeam(ia);
    const float vx = finder.hits.xBeam(ih) - finder.hits.xBeam(ib), vy = finder.hits.yBeam(ih) - finder.hits.yBeam(ib);
    const float wx = ux + vx, wy = uy + vy;
    // pT = kNominalPtPerRadius R, R = |u| |v| |w| / (2 |u x v|)
    const float cr2 = 2.f * alpaka::math::abs(acc, ux * vy - uy * vx);
    const float l3 = alpaka::math::sqrt(acc, (ux * ux + uy * uy) * (vx * vx + vy * vy) * (wx * wx + wy * wy));
    return kNominalPtPerRadius * l3 >= finder.psSeedPtMin * cr2;
  }
  // ps_seed: true if stage d drops the quad of the candidate c with the fourth hit ih at position p, its helix's
  // curvature k and pT pte: it has OT1-3 P hits (with ps_seed 2: and two pixel hits), pte >= ps_seed_pt, and none of its
  // OT1-3 P hits is marked with a q/pT estimate within ps_seed_curv sigma of the helix's (0: marked only)
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool psRejectQuad(
      const TAcc &acc, const ChainFinderInputs &finder, const Candidate &c, int p, uint32_t ih, float k, float pte) {
    if (!finder.psSeed || !(pte >= finder.psSeedPtMin))
      return false;
    int npix = 0, nps = 0;
    for (int m = 0; m < 4; ++m) {
      const int pos = m < 3 ? c.positions[m] : p;
      const auto P = finder.chainPositions[pos];
      if (P.isPixel()) {
        ++npix;
        continue;
      }
      const int l = P.layerId();
      if (l != 4 && l != 6 && l != 8)
        continue;
      ++nps;
      const auto L = finder.layers[l];
      const uint32_t h = m < 3 ? L.firstHit() + c.hits[m] : ih;
      const uint32_t row = uint32_t(L.firstOTHit()) + (h - L.firstHit());
      if (finder.psSeedRange > 0) {
        // ps_seed_rng: an S match in the helix's q/pT range (or a neighbouring one); below the ranges: confirmed
        if (::mkseeder::psCountInRange(finder.otHits.psCounts(row),
                                       finder.otHits.psCountsLow(row),
                                       finder.psLow,
                                       -k / kNominalPtPerRadius,
                                       finder.psSeedRange == 2) != 0)
          return false;
        continue;
      }
      if (finder.otHits.psMatches(row) == 0)
        continue;
      if (!(finder.psSeedSigma > 0) ||
          !(alpaka::math::abs(acc, finder.otHits.psQOverPt(row) + k / kNominalPtPerRadius) >
            finder.psSeedSigma * finder.psCurvatureResolution * 23.f / std::max(finder.hits.r(h), 1.f)))
        return false;  // a confirmed OT1-3 P hit
    }
    return nps > 0 && (finder.psSeed != 2 || npix == 2);
  }

  // The hit on barrel layer LP closest to the helix H by (dphi / sphi)^2 + (dz / sz)^2,
  // among the hits within fphi, fz of the prediction at the layer's two edges; -1 none, -2 the helix misses the layer;
  // psMask (the PS-module confirmation): the layer's OT hits' marks, hits without one skipped
  template <typename TAcc>
  ALPAKA_FN_ACC inline int next_hit(const TAcc &acc,
                                    const ChainFinderInputs &finder,
                                    const EventLayer &LP,
                                    const LayerAxes &GP,
                                    const HelixF &H,
                                    float pte,
                                    float fphi,
                                    float fz,
                                    float &zmid,
                                    float &dphi,
                                    float &dz,
                                    float &score,
                                    const uint8_t *psMask = nullptr) {
    float x0, y0, q0, s0, x1, y1, q1, s1;
    const bool ok0 = H.ok && H.positionAtR(acc, LP.qbarLo(), x0, y0, q0, s0),
               ok1 = H.ok && H.positionAtR(acc, LP.qbarHi(), x1, y1, q1, s1);
    if (!ok0 && !ok1)
      return -2;
    if (!ok0)
      x0 = x1, y0 = y1, q0 = q1;
    if (!ok1)
      x1 = x0, y1 = y0, q1 = q0;
    zmid = 0.5f * (q0 + q1);
    if (std::min(q0, q1) > LP.qHi() || std::max(q0, q1) < LP.qLo())
      return -2;
    const float pt = std::max(0.5f, pte), sphi = 0.0005f + 3.2e-3f / pt, sz = 0.075f + 0.0316f / pt;
    const float p0 = libm::atan2(acc, y0, x0), p1 = libm::atan2(acc, y1, x1), dpp = wrapPhiFloor(acc, p1 - p0);
    float best = 1e30f;
    int kb = -1;
    forEachBinRun(finder,
                  LP,
                  phiBinRangeAroundDouble(acc, GP, p0 + 0.5f * dpp, 0.5f * alpaka::math::abs(acc, dpp) + fphi),
                  qBinRangePaddedDouble(acc, GP, std::min(q0, q1) - fz, std::max(q0, q1) + fz),
                  [&](uint32_t b, uint32_t e) {
#ifdef MKSEEDER_CPU_AVX2
                    // 8 hits at a time (the hits' columns have kHitsPadding rows after the last hit), the minimum in hit order
                    if (b < e) {
                      const avx2::Helix8 H8v = avx2::bcast(H);
                      const __m256 vsphi = _mm256_set1_ps(sphi), vsz = _mm256_set1_ps(sz);
                      for (uint32_t k0 = b; k0 < e; k0 += 8) {
                        const uint32_t ih0 = LP.firstHit() + k0, nk = std::min(8u, e - k0);
                        const __m256 m = _mm256_castsi256_ps(
                            _mm256_cmpgt_epi32(_mm256_set1_epi32((int)nk), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
                        __m256 ok, wr, px, py, qp, s3;
                        avx2::positionAtR8(
                            H8v, _mm256_loadu_ps(finder.hits.r().data() + ih0), m, ok, wr, px, py, qp, s3);
                        const unsigned int okm = _mm256_movemask_ps(ok);
                        if (!okm)
                          continue;
                        const __m256 dp = avx2::wrapPhiFloor8(
                            _mm256_sub_ps(_mm256_loadu_ps(finder.hits.phi().data() + ih0), avx2::atan2f8(py, px, ok)));
                        const __m256 dq = _mm256_sub_ps(_mm256_loadu_ps(finder.hits.z().data() + ih0), qp);
                        const __m256 rp = _mm256_div_ps(dp, vsphi), rz = _mm256_div_ps(dq, vsz);
                        const __m256 sc = _mm256_fmadd_ps(rp, rp, _mm256_mul_ps(rz, rz));
                        alignas(32) float scs[8], dps[8], dqs[8];
                        _mm256_store_ps(scs, sc), _mm256_store_ps(dps, dp), _mm256_store_ps(dqs, dq);
                        for (unsigned int mm = okm; mm; mm &= mm - 1) {
                          const int l = std::countr_zero(mm);
                          if (psMask && psMask[k0 + l] == 0)
                            continue;
                          if (scs[l] < best)
                            best = scs[l], kb = k0 + l, dphi = dps[l], dz = dqs[l];
                        }
                      }
                    }
                    return;
#endif
                    for (uint32_t k = b; k < e; ++k) {
                      const uint32_t ih = LP.firstHit() + k;
                      float px, py, qp, s3;
                      if (!H.positionAtR(acc, finder.hits.r(ih), px, py, qp, s3))
                        continue;
                      const float dp = wrapPhiFloor(acc, finder.hits.phi(ih) - libm::atan2(acc, py, px)),
                                  dq = finder.hits.z(ih) - qp;
                      const float sc = (dp / sphi) * (dp / sphi) + (dq / sz) * (dq / sz);
                      if (psMask && psMask[k] == 0)
                        continue;
                      if (sc < best)
                        best = sc, kb = k, dphi = dp, dz = dq;
                    }
                  });
    score = best;
    return kb;
  }

  // fk_ot2_local: whether barrel layer LP (OT2-P) has a hit within the local windows (the extension's residual in the
  // hit's module frame) of the helix H; 1 yes, 0 no, -1 the helix misses the layer
  template <typename TAcc>
  ALPAKA_FN_ACC inline int localWindowHit(
      const TAcc &acc, const ChainFinderInputs &finder, int layer, const HelixF &H, float pte) {
    const EventLayer LP = finder.layers[layer];
    const LayerAxes GP = finder.layerGeometry[layer];
    float x0, y0, q0, s0, x1, y1, q1, s1;
    const bool ok0 = H.ok && H.positionAtR(acc, LP.qbarLo(), x0, y0, q0, s0),
               ok1 = H.ok && H.positionAtR(acc, LP.qbarHi(), x1, y1, q1, s1);
    if (!ok0 && !ok1)
      return -1;
    if (!ok0)
      x0 = x1, y0 = y1, q0 = q1;
    if (!ok1)
      x1 = x0, y1 = y0, q1 = q0;
    if (std::min(q0, q1) > LP.qHi() || std::max(q0, q1) < LP.qLo() ||
        alpaka::math::abs(acc, 0.5f * (q0 + q1)) >= LP.qHi() - 2)
      return -1;
    const MkSeederChainParameters &C = finder.chain;
    const float pt = std::max(0.5f, pte);
    const float wx = C.ot2LocalWindowScale * (C.ot2LocalWindowX[0] + C.ot2LocalWindowX[1] / pt),
                wy = C.ot2LocalWindowScale * (C.ot2LocalWindowY[0] + C.ot2LocalWindowY[1] / pt);
    const float p0 = libm::atan2(acc, y0, x0), dpp = wrapPhiFloor(acc, libm::atan2(acc, y1, x1) - p0);
    int found = 0;
    forEachBinRun(
        finder,
        LP,
        phiBinRangeAroundDouble(acc, GP, p0 + 0.5f * dpp, 0.5f * alpaka::math::abs(acc, dpp) + 1.5f * wx / LP.qbarLo()),
        qBinRangePaddedDouble(acc, GP, std::min(q0, q1) - wy - 1.f, std::max(q0, q1) + wy + 1.f),
        [&](uint32_t b, uint32_t e) {
          for (uint32_t k = b; k < e && !found; ++k) {
            const uint32_t ih = LP.firstHit() + k;
            float E[6], dx, dy;
            hitCovariance(finder.otHits, LP.firstOTHit() + k, E);
            if (extensionLocalResidual(
                    acc, H, finder.hits.xBeam(ih), finder.hits.yBeam(ih), finder.hits.z(ih), E, dx, dy) &&
                alpaka::math::abs(acc, dx) < wx && alpaka::math::abs(acc, dy) < wy)
              found = 1;
          }
        });
    return found;
  }

  // fk_ot1_lsig / fk_ot2_lsig: the smallest local chi2 (dx / sx)^2 + (dy / sy)^2 (extensionLocalResidual: the helix H
  // onto the hit's module plane, the module frame) of the hits of barrel layer `layer` within S sigma; best = 1e30 none;
  // -1: the helix misses the layer (as localWindowHit), else 0. With stop > 0 the scan ends at the first hit with a chi2
  // below stop (best is then that hit's chi2, not the smallest: for a caller that only tests best < stop)
  template <typename TAcc>
  ALPAKA_FN_ACC inline int localChi2Hit(const TAcc &acc,
                                        const ChainFinderInputs &finder,
                                        int layer,
                                        const HelixF &H,
                                        float pte,
                                        float S,
                                        float sy,
                                        float &best,
                                        float stop = 0.f) {
    best = 1e30f;
    const EventLayer LP = finder.layers[layer];
    const LayerAxes GP = finder.layerGeometry[layer];
    if (LP.firstOTHit() < 0)
      return -1;
    float x0, y0, q0, s0, x1, y1, q1, s1;
    const bool ok0 = H.ok && H.positionAtR(acc, LP.qbarLo(), x0, y0, q0, s0),
               ok1 = H.ok && H.positionAtR(acc, LP.qbarHi(), x1, y1, q1, s1);
    if (!ok0 && !ok1)
      return -1;
    if (!ok0)
      x0 = x1, y0 = y1, q0 = q1;
    if (!ok1)
      x1 = x0, y1 = y0, q1 = q0;
    if (std::min(q0, q1) > LP.qHi() || std::max(q0, q1) < LP.qLo() ||
        alpaka::math::abs(acc, 0.5f * (q0 + q1)) >= LP.qHi() - 2)
      return -1;
    const MkSeederChainParameters &C = finder.chain;
    const float pt = std::max(0.5f, pte);
    const float sx =
        (C.otLocalSigmaX[0] + C.otLocalSigmaX[1] / pt) * (1.f + C.otLocalSigmaX[2] * alpaka::math::abs(acc, H.cot));
    const float wx = S * sx, wy = S * sy;
    const float p0 = libm::atan2(acc, y0, x0), dpp = wrapPhiFloor(acc, libm::atan2(acc, y1, x1) - p0);
    forEachBinRun(
        finder,
        LP,
        phiBinRangeAroundDouble(acc, GP, p0 + 0.5f * dpp, 0.5f * alpaka::math::abs(acc, dpp) + 1.5f * wx / LP.qbarLo()),
        qBinRangePaddedDouble(acc, GP, std::min(q0, q1) - wy - 1.f, std::max(q0, q1) + wy + 1.f),
        [&](uint32_t b, uint32_t e) {
          for (uint32_t k = b; k < e && !(best < stop); ++k) {
            const uint32_t ih = LP.firstHit() + k;
            float E[6], dx, dy;
            hitCovariance(finder.otHits, LP.firstOTHit() + k, E);
            if (!extensionLocalResidual(
                    acc, H, finder.hits.xBeam(ih), finder.hits.yBeam(ih), finder.hits.z(ih), E, dx, dy))
              continue;
            const float c2 = (dx / sx) * (dx / sx) + (dy / sy) * (dy / sy);
            if (c2 < best)
              best = c2;
          }
        });
    return 0;
  }

  // a candidate's r-z line, as the stage that queued it computed it
  template <typename TAcc>
  ALPAKA_FN_ACC inline void line(
      const TAcc &acc, const ChainFinderInputs &finder, const Candidate &c, float &z0, float &cot) {
    const uint32_t ia = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()].firstHit() + c.hits[0];
    const float za = finder.hits.z(ia), ra = finder.hits.r(ia);
    if (c.positions[2] == 0) {
      const uint32_t ib = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()].firstHit() + c.hits[1];
      const float dr = finder.hits.r(ib) - ra, dz = finder.hits.z(ib) - za;
      cot = dz / (dr > 0.1f ? dr : 1.0f);
    } else {
      const uint32_t ic = finder.layers[finder.chainPositions[c.positions[2]].binnedLayerId()].firstHit() + c.hits[2];
      cot = (finder.hits.z(ic) - za) / (finder.hits.r(ic) - ra);
    }
    z0 = za - cot * ra;
  }

  // ---- the queues
  ALPAKA_FN_HOST_ACC inline int takeChunkSerial(const Queues &candidateQueues);
  template <typename TAcc>
  ALPAKA_FN_ACC inline void pushAt(
      const TAcc &acc, const Queues &candidateQueues, int queue, uint32_t s, const Candidate &c);
  template <typename TAcc>
  ALPAKA_FN_ACC inline void push(const TAcc &acc, const Queues &candidateQueues, int queue, const Candidate &c) {
    if constexpr (alpaka::accMatchesTags<TAcc, alpaka::TagCpuSerial>) {
      // the serial backend runs the grid in one thread: the same sequence as below without the atomics and the
      // handshake (the same chunks in the same order)
      const uint32_t s = candidateQueues.queues.count(queue)++;
      const uint32_t ci = s / Queues::kCandidateChunkSize, slot = s % Queues::kCandidateChunkSize;
      if (ci >= candidateQueues.maxChunksPerQueue) {
        ++candidateQueues.counters->nLost;
        return;
      }
      int &t = candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci);
      if (slot == 0)
        t = takeChunkSerial(candidateQueues);
      const int cid = t;
      if (cid < 0) {
        ++candidateQueues.counters->nLost;
        return;
      }
      storeCandidate(candidateQueues.pool, (uint32_t)cid * Queues::kCandidateChunkSize + slot, c);
      return;
    }
    pushAt(acc,
           candidateQueues,
           queue,
           alpaka::atomicAdd(acc, &candidateQueues.queues.count(queue), 1u, alpaka::hierarchy::Blocks{}),
           c);
  }
  // a GPU: candidate c into slot s of a queue (s reserved on its count by the caller)
  template <typename TAcc>
  ALPAKA_FN_ACC inline void pushAt(
      const TAcc &acc, const Queues &candidateQueues, int queue, uint32_t s, const Candidate &c) {
    const uint32_t ci = s / Queues::kCandidateChunkSize, slot = s % Queues::kCandidateChunkSize;
    if (ci >= candidateQueues.maxChunksPerQueue) {
      alpaka::atomicAdd(acc, &candidateQueues.counters->nLost, 1u, alpaka::hierarchy::Blocks{});
      return;
    }
    int *t = &candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci);
    int cid;
    if (slot == 0) {
      // a freed chunk if there is one (the free list only grows between the kernels that push), else a fresh one
      cid = -2;
      uint32_t fc = alpaka::atomicAdd(acc, &candidateQueues.counters->nFreeChunks, 0u, alpaka::hierarchy::Blocks{});
      while (fc > 0) {
        const uint32_t prev =
            alpaka::atomicCas(acc, &candidateQueues.counters->nFreeChunks, fc, fc - 1, alpaka::hierarchy::Blocks{});
        if (prev == fc) {
          cid = candidateQueues.freeChunks.chunk(fc - 1);
          break;
        }
        fc = prev;
      }
      if (cid < 0) {
        const uint32_t nc =
            alpaka::atomicAdd(acc, &candidateQueues.counters->nextFreshChunk, 1u, alpaka::hierarchy::Blocks{});
        cid = nc < candidateQueues.poolChunks ? (int)nc : -2;
      }
      alpaka::atomicExch(acc, t, cid, alpaka::hierarchy::Blocks{});
    } else {
      // the chunk is being handed out by the push that took its slot 0
      do {
        cid = alpaka::atomicCas(acc, t, -1, -1, alpaka::hierarchy::Blocks{});
      } while (cid == -1);
    }
    if (cid < 0) {
      alpaka::atomicAdd(acc, &candidateQueues.counters->nLost, 1u, alpaka::hierarchy::Blocks{});
      return;
    }
    storeCandidate(candidateQueues.pool, (uint32_t)cid * Queues::kCandidateChunkSize + slot, c);
  }
  // the i-th candidate of a queue; false if it was lost
  ALPAKA_FN_HOST_ACC inline bool readCandidate(const Queues &candidateQueues, int queue, uint32_t i, Candidate &c) {
    const uint32_t ci = i / Queues::kCandidateChunkSize;
    if (ci >= candidateQueues.maxChunksPerQueue)
      return false;
    const int cid = candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci);
    if (cid < 0)
      return false;
    loadCandidate(
        candidateQueues.pool, (uint32_t)cid * Queues::kCandidateChunkSize + i % Queues::kCandidateChunkSize, c);
    return true;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline void emit(const TAcc &acc, const Queues &candidateQueues, const Quad &q) {
    if constexpr (alpaka::accMatchesTags<TAcc, alpaka::TagCpuSerial>) {
      const uint32_t o = candidateQueues.counters->nQuads++;  // one thread on the serial backend
      if (o < candidateQueues.quadCapacity)
        storeQuad(candidateQueues.quads, o, q);
      else
        ++candidateQueues.counters->nLost;
      return;
    }
    const uint32_t o = alpaka::atomicAdd(acc, &candidateQueues.counters->nQuads, 1u, alpaka::hierarchy::Blocks{});
    if (o < candidateQueues.quadCapacity)
      storeQuad(candidateQueues.quads, o, q);
    else
      alpaka::atomicAdd(acc, &candidateQueues.counters->nLost, 1u, alpaka::hierarchy::Blocks{});
  }
  // queue ids: Q2[p] = p, Q3[p] = n + p
  ALPAKA_FN_HOST_ACC inline int q2(const ChainFinderInputs &finder, int p) { return p; }
  ALPAKA_FN_HOST_ACC inline int q3(const ChainFinderInputs &finder, int p) { return finder.chain.nPositions + p; }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

// the CPU backends' paths of the kernels below
#include "MkSeederKernelsCpu.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // stage b with zv_outer_cotmin (zcot > 0): the q bins of the second layer B (within fq) that can hold the second hit of
  // a doublet from the inner hit (raf, zaf) with z0 in the narrow zv range [zVertexLow, zVertexHigh] or with |cot| >= zcot,
  // in q order and disjoint (out[0..n-1], n returned). The bounds use the layer's extent (EventLayer: it holds every
  // hit) and margins well above the float rounding of the doublet's cot and z0, so no such hit is outside them.
  template <typename TAcc>
  ALPAKA_FN_ACC inline int steepOrNarrowBins(const TAcc &acc,
                                             const ChainFinderInputs &finder,
                                             const EventLayer &B,
                                             const LayerAxes &Bg,
                                             BinRange fq,
                                             float raf,
                                             float zaf,
                                             float zcot,
                                             BinRange *out) {
    constexpr double kMargin = 0.01, kRelative = 1e-4;  // cm; relative
    const double ra = raf, za = zaf, z0lo = finder.chainEvent.zVertexLow - kMargin,
                 z0hi = finder.chainEvent.zVertexHigh + kMargin;
    BinRange rs[3];
    int n = 0;
    if (!Bg.isDisc()) {
      // q = z: the narrow range (the lines from z0 in [z0lo, z0hi] through the inner hit at r in the layer's extent)
      const double rbs[2] = {B.qbarLo(), B.qbarHi()}, z0s[2] = {z0lo, z0hi};
      double lo = 1e30, hi = -1e30;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
          const double zz = za + (za - z0s[j]) * (rbs[i] - ra) / ra;
          lo = std::min(lo, zz), hi = std::max(hi, zz);
        }
      // |cot| >= zcot: |z - za| >= zcot dr with dr >= max(r extent low - ra, 0.1) (stage b needs dr > 0.1)
      const double g = zcot * std::max(double(B.qbarLo()) - ra, 0.1) * (1 - kRelative) - kMargin;
      if (g <= 0) {
        out[0] = fq;
        return 1;
      }
      rs[n++] = qBinRangePaddedDouble(acc, Bg, -1e30, za - g);
      rs[n++] = qBinRangePaddedDouble(acc, Bg, lo - kMargin, hi + kMargin);
      rs[n++] = qBinRangePaddedDouble(acc, Bg, za + g, 1e30);
    } else {
      // q = r: the narrow range as the wide one in KernelFindStartDoublets (the layer's z extent zb0..zb1)
      const double s = (B.qbarLo() + B.qbarHi()) > 0 ? 1 : -1;
      const double zas = s * za, zb0 = std::min(s * B.qbarLo(), s * B.qbarHi()),
                   zb1 = std::max(s * B.qbarLo(), s * B.qbarHi());
      const double z0min = std::min(s * z0lo, s * z0hi), z0max = std::max(s * z0lo, s * z0hi);
      // |cot| >= zcot: dr <= |dz| / zcot <= max |z - za| over the extent / zcot
      const double dzmax =
          std::max(alpaka::math::abs(acc, double(B.qbarLo()) - za), alpaka::math::abs(acc, double(B.qbarHi()) - za));
      rs[n++] = qBinRangePaddedDouble(acc, Bg, -1e30, ra + dzmax / zcot * (1 + kRelative) + kMargin);
      if (zas - z0min > 0) {
        const double lo = ra + ra * (zb0 - zas) / (zas - z0min),
                     hi = zas - z0max > 0 ? ra + ra * (zb1 - zas) / (zas - z0max) : double(B.qHi());
        rs[n++] = qBinRangePaddedDouble(acc, Bg, lo - kMargin, hi + kMargin);
      }
    }
    // clip to fq, sort by the first bin, merge the overlapping / adjacent ones
    int m = 0;
    for (int i = 0; i < n; ++i) {
      BinRange r{std::max(rs[i].first, fq.first), std::min(rs[i].last, fq.last)};
      if (r.first >= r.last)
        continue;
      int j = m;
      while (j > 0 && out[j - 1].first > r.first) {
        out[j] = out[j - 1];
        --j;
      }
      out[j] = r;
      ++m;
    }
    int k = 0;
    for (int i = 0; i < m; ++i) {
      if (k > 0 && out[i].first <= out[k - 1].last)
        out[k - 1].last = std::max(out[k - 1].last, out[i].last);
      else
        out[k++] = out[i];
    }
    return k;
  }

  // ---- a GPU's pieces of long hit windows Stage b: the pieces of kPieceHits hits of the work items with a long
  // window, beyond the first kItemHits hits that the item's own thread scans, a piece per warp
  // (KernelFindStartDoubletPieces); stage d: the (candidate, hit) pairs of KernelScanQuadWindows. data[0] counts them,
  // then the (work item, piece) pairs (at most capacity)
  struct WorkPieces {
    uint32_t *data;
    uint32_t capacity;
    ALPAKA_FN_HOST_ACC uint32_t &item(uint32_t i) const { return data[1 + 2 * i]; }
    ALPAKA_FN_HOST_ACC uint32_t &piece(uint32_t i) const { return data[2 + 2 * i]; }
  };
  // the position of the k-th set bit of m (k < popcount(m))
  template <typename TAcc>
  ALPAKA_FN_ACC inline uint32_t nthSetBit(const TAcc &acc, uint32_t m, uint32_t k) {
    uint32_t q = 0;
    for (uint32_t t = 16; t > 0; t >>= 1)
      if ((uint32_t)alpaka::popcount(acc, m & ((1u << (q + t)) - 1u)) <= k)
        q += t;
    return q;
  }
  // CUDA, a whole warp: the hits b + (g - o) of a bin run for g = f0 + lane, f0 + lane + 32, ... < f1 tested by pass()
  // (one per lane); the ones that pass are gathered in lane order (pend on the lanes 0..np-1) and handed to take() 32
  // at a time, so its (longer) work runs with all lanes busy; the caller hands the last np to take() on the lanes below
  // np
  template <typename TAcc, typename P, typename T>
  ALPAKA_FN_ACC inline void warpGatheredScan(const TAcc &acc,
                                             uint32_t lane,
                                             uint32_t b,
                                             uint32_t o,
                                             uint32_t f0,
                                             uint32_t f1,
                                             uint32_t &pend,
                                             uint32_t &np,
                                             P &&pass,
                                             T &&take) {
    constexpr uint32_t kWarp = 32;
    for (uint32_t g = f0; g < f1; g += kWarp) {
      const uint32_t k = b + (g + lane - o);
      const uint32_t m = warpBallotAll(acc, g + lane < f1 && pass(k));
      const uint32_t c = (uint32_t)alpaka::popcount(acc, m);
      if (c == 0)
        continue;
      const uint32_t v = warpShflAll(acc, k, (int)nthSetBit(acc, m, (lane - np) % kWarp));
      if (lane >= np && lane < np + c)
        pend = v;
      if (np + c >= kWarp) {
        take(pend);
        const uint32_t v2 = warpShflAll(acc, k, (int)nthSetBit(acc, m, std::min(kWarp - np + lane, c - 1)));
        np = np + c - kWarp;
        if (lane < np)
          pend = v2;
      } else {
        np += c;
      }
    }
  }

  // ---- stage b
  class KernelFindStartDoublets {
  public:
    static constexpr uint32_t kItemHits = 64, kPieceHits = 256;
    static_assert(kPieceHits % 32 == 0);
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  ChainFinderInputs finder,
                                  Queues candidateQueues,
                                  WorkPieces pieces) const {
      cpu::RouterOf<TAcc> router;  // nextCrossedPosition() on the CPU backends (MkSeederKernelsCpu.h)
      router.init(finder);
      // the serial backend: the doublets' lines and crossed positions into the pool (CandidateLine)
      bool ml = false;
      if constexpr (kSerialAcc<TAcc>)
        ml = candidateQueues.hasLines && router.use32;
      for (uint32_t w : cms::alpakatools::uniform_elements(acc, (uint32_t)finder.chainEvent.nWorkItems))
        workItem(acc, finder, candidateQueues, router, ml, w, 0u, kItemHits, 1u, &pieces);
    }
    // work item w (the a hit of a start pair): its window on B, then its doublets. A GPU scans the window's hits
    // hitLo, hitLo + hitStep, ... < hitHi in the bin runs' order; with pieces != nullptr (0, kItemHits, 1) it puts the rest of
    // the window onto the piece list. A CPU backend scans the whole window.
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void workItem(const TAcc &acc,
                                                        const ChainFinderInputs &finder,
                                                        const Queues &candidateQueues,
                                                        cpu::RouterOf<TAcc> &router,
                                                        bool ml,
                                                        uint32_t w,
                                                        uint32_t hitLo,
                                                        uint32_t hitHi,
                                                        uint32_t hitStep,
                                                        const WorkPieces *pieces) {
      const MkSeederChainParameters &chainParameters = finder.chain;
      {
        // the start pair of this work item: the last with a0 <= w
        int lo = 0, hi = chainParameters.nStartPairs - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) / 2;
          if (finder.startPairs.firstWorkItem(mid) <= w)
            lo = mid;
          else
            hi = mid - 1;
        }
        const int si = lo;
        const auto startPair = finder.chainStarts[si];
        const uint32_t ka = w - finder.startPairs.firstWorkItem(si);
        const EventLayer A = finder.layers[finder.chainPositions[startPair.positionA()].binnedLayerId()],
                         B = finder.layers[finder.chainPositions[startPair.positionB()].binnedLayerId()];
        const LayerAxes Bg = finder.layerGeometry[finder.chainPositions[startPair.positionB()].binnedLayerId()];
        const uint32_t ia = A.firstHit() + ka;
        // b_fetch (double)
        const float raf = finder.hits.r(ia), paf = finder.hits.phi(ia), zaf = finder.hits.z(ia);
        const double ra_d = raf, pa_d = paf, za_d = zaf;
        // zv_outer: the wider z0 range for an inner hit at r >= zv_outer_rmin (study gates: an inner disk layer; a hit
        // whose lines to the wider z0 range reach |cot| >= zv_outer_cotmin, and then only the doublets with |cot| >= it)
        const float zcot = chainParameters.zVertexOuterMinAbsCot;
        const bool zOuter =
            chainParameters.zVertexHalfWidthOuter > 0 && raf >= chainParameters.zVertexOuterMinRadius &&
            (!chainParameters.zVertexOuterDiscOnly ||
             finder.layerGeometry[finder.chainPositions[startPair.positionA()].binnedLayerId()].isDisc()) &&
            (zcot <= 0 ||
             alpaka::math::abs(acc, zaf - finder.chainEvent.beamSpotZ) + chainParameters.zVertexHalfWidthOuter >=
                 zcot * raf);
        const bool zOuterCot = zOuter && zcot > 0;
        const double zvw = zOuter ? chainParameters.zVertexHalfWidthOuter : chainParameters.zVertexHalfWidth;
        const double zlo_d = finder.chainEvent.beamSpotZ - zvw, zhi_d = finder.chainEvent.beamSpotZ + zvw;
        const float zvLow = zOuter ? finder.chainEvent.zVertexLowOuter : finder.chainEvent.zVertexLow,
                    zvHigh = zOuter ? finder.chainEvent.zVertexHighOuter : finder.chainEvent.zVertexHigh;
        auto b_window = [&](double ra, double rb) {
          const double Rmin = chainParameters.startPtMin / kNominalPtPerRadiusD;
          return (rb - ra) / (2 * Rmin) + chainParameters.startD0Max * (1 / ra - 1 / rb);
        };
        const double rbmax = Bg.isDisc() ? B.qHi() : B.qbarHi();
        const double w_ab = b_window(ra_d, rbmax) + chainParameters.doubletMargin;
        double bq_lo, bq_hi;
        if (!Bg.isDisc()) {
          double zz[4];
          int nz = 0;
          const double rbs[2] = {B.qbarLo(), B.qbarHi()}, z0s[2] = {zlo_d, zhi_d};
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
              zz[nz++] = za_d + (za_d - z0s[j]) * (rbs[i] - ra_d) / ra_d;
          bq_lo = std::min(std::min(zz[0], zz[1]), std::min(zz[2], zz[3]));
          bq_hi = std::max(std::max(zz[0], zz[1]), std::max(zz[2], zz[3]));
        } else {
          const double s = (B.qbarLo() + B.qbarHi()) > 0 ? 1 : -1;
          const double za = s * za_d, zb0 = std::min(s * B.qbarLo(), s * B.qbarHi()),
                       zb1 = std::max(s * B.qbarLo(), s * B.qbarHi());
          const double z0min = std::min(s * zlo_d, s * zhi_d), z0max = std::max(s * zlo_d, s * zhi_d);
          if (za - z0min <= 0)
            return;
          bq_lo = ra_d + ra_d * (zb0 - za) / (za - z0min);
          bq_hi = za - z0max > 0 ? ra_d + ra_d * (zb1 - za) / (za - z0max) : B.qHi();
        }
        if (bq_hi < B.qLo() || bq_lo > B.qHi())
          return;
        BinRange fp = phiBinRangeAroundDouble(acc, Bg, pa_d, w_ab), fq = qBinRangePaddedDouble(acc, Bg, bq_lo, bq_hi);
        // the q range the start pair's cot range allows on B
        const float ulo = startPair.cotMin(), uhi = startPair.cotMax();
        const bool cot_q = Bg.isDisc() ? (ulo > 0 || uhi < 0) : (ulo > -1e29f && uhi < 1e29f);
        if (cot_q) {
          float qlo, qhi;
          if (!Bg.isDisc()) {
            const float d0 = B.qbarLo() - raf, d1 = B.qbarHi() - raf;
            const float e0 = ulo * d0, e1 = ulo * d1, e2 = uhi * d0, e3 = uhi * d1;
            qlo = zaf + std::min(std::min(e0, e1), std::min(e2, e3));
            qhi = zaf + std::max(std::max(e0, e1), std::max(e2, e3));
          } else {
            const float i0 = 1.0f / ulo, i1 = 1.0f / uhi, d0 = B.qbarLo() - zaf, d1 = B.qbarHi() - zaf;
            const float e0 = i0 * d0, e1 = i0 * d1, e2 = i1 * d0, e3 = i1 * d1;
            qlo = raf + std::min(std::min(e0, e1), std::min(e2, e3));
            qhi = raf + std::max(std::max(e0, e1), std::max(e2, e3));
          }
          const BinRange nq = qBinRangePaddedDouble(acc, Bg, qlo - 0.01f, qhi + 0.01f);
          fq.first = std::max(fq.first, nq.first), fq.last = std::min(fq.last, nq.last);
          if (fq.first >= fq.last)
            return;
          if (Bg.isDisc() && qhi + 0.01f < B.qHi())
            fp = phiBinRangeAroundDouble(acc, Bg, pa_d, b_window(ra_d, qhi + 0.01f) + chainParameters.doubletMargin);
        }
        // the q bins of B to scan: fq; with zv_outer_cotmin (zOuterCot) only the bins that can hold a doublet with z0 in
        // the narrow zv range or one with |cot| >= zcot (testHitB rejects the other doublets with z0 outside the narrow
        // range): up to 3 ranges in q order, so the same hits in the same order as the scan of fq
        BinRange qrs[3] = {fq, {0, 0}, {0, 0}};
        int nqr = 1;
        if (zOuterCot)
          nqr = steepOrNarrowBins(acc, finder, B, Bg, fq, raf, zaf, zcot, qrs);
        const float inva = 1.0f / raf, side = chainParameters.side;
        const int g_lid = startPair.gapLayer();
        const int shA = shapeOfPosition(finder, startPair.positionA()),
                  shB = shapeOfPosition(finder, startPair.positionB());
        const int lead_only = chainParameters.leadOnly & !startPair.gapAllowed();
        // the doublet cuts of hit kb of B (its dr, dp, cot)
        auto doubletCuts = [&](uint32_t kb, float &dr, float &dp, float &cot) {
          const uint32_t ib = B.firstHit() + kb;
          dr = finder.hits.r(ib) - raf;
          const float dz = finder.hits.z(ib) - zaf;
          dp = finder.hits.phi(ib) - paf;
          dp = dp > kPi ? dp - kTwoPi : dp;
          dp = dp < -kPi ? dp + kTwoPi : dp;
          const float wd = dr * chainParameters.doubletInverseTwoRadius +
                           chainParameters.doubletD0Max * (inva - finder.hits.invR(ib)) +
                           chainParameters.doubletPhiMargin;
          const float num = zaf * dr - raf * dz;
          const int cut = (dr > 0.1f) & (alpaka::math::abs(acc, dp) <= wd) & (num >= zvLow * dr) & (num <= zvHigh * dr);
          const int sd = !((dz * side < 0) | ((dz == 0) & (side < 0)));
          cot = dz / (dr > 0.1f ? dr : 1.0f);
          const int in0 = (cot >= startPair.cotMin()) & (cot <= startPair.cotMax());
          return bool(cut & sd & in0);
        };
        // pre: a CPU backend's scan already applied the cuts below, the values dr, dp, cot, z0 (cpu::ScanStart)
        auto testHitB = [&](uint32_t kb, const float *pre) {
          {
            const uint32_t ib = B.firstHit() + kb;
            float dr, dp, cot, z0;
            if (pre) {
              dr = pre[0], dp = pre[1], cot = pre[2], z0 = pre[3];
            } else {
              if (!doubletCuts(kb, dr, dp, cot))
                return;
              z0 = zaf - cot * raf;
            }
            if (zOuterCot && alpaka::math::abs(acc, cot) < zcot &&
                (z0 < finder.chainEvent.zVertexLow || z0 > finder.chainEvent.zVertexHigh))
              return;
            // (a CPU backend with AVX2: done by the scan, cpu::ScanStart)
            if (g_lid >= 0 && !kScanGapRay<TAcc>) {
              constexpr int kGapRayNPhiBins = ::mkseeder::kGapRayNPhiBins;
              const auto G = finder.gapMapLayers[g_lid];
              const float g_pfac = kGapRayNPhiBins / kTwoPi, g_izc = 1.0f / 0.005f;
              float ph = paf + dp * (G.rMid() - raf) / (dr > 0.1f ? dr : 1.0f);
              ph = ph > kPi ? ph - kTwoPi : ph;
              ph = ph < -kPi ? ph + kTwoPi : ph;
              int pbn = (int)((ph + kPi) * g_pfac);
              pbn = pbn < 0 ? 0 : pbn >= kGapRayNPhiBins ? kGapRayNPhiBins - 1 : pbn;
              const float *r1 = finder.gapRays.rInner().data() + g_lid * kGapRayNPhiBins,
                          *r2 = finder.gapRays.rOuter().data() + g_lid * kGapRayNPhiBins;
              const uint8_t *zg = finder.gapMapZGaps.inGap().data() + G.firstZGap();
              const float z1 = zaf + cot * (r1[pbn] - raf), z2 = zaf + cot * (r2[pbn] - raf);
              const int k1 = (int)alpaka::math::floor(acc, (z1 - G.zGap0()) * g_izc),
                        k2 = (int)alpaka::math::floor(acc, (z2 - G.zGap0()) * g_izc);
              const int nzg = G.nZGaps();
              const int in1g = (k1 < 0) | (k1 >= nzg) || zg[k1], in2g = (k2 < 0) | (k2 >= nzg) || zg[k2];
              if (!(finder.gapRays.nearEdge(g_lid * kGapRayNPhiBins + pbn) | in1g | in2g))
                return;
            }
            if constexpr (kSerialAcc<TAcc>) {
              if (ml) {
                // the shape and the gap map first (the gap map's states of its few positions by Router::state1), then
                // the states of every position once (the sweep) and the tests below from its masks; every test only
                // rejects, so the same decisions in any order
                if (shA >= 0 || shB >= 0) {
                  const float ac = alpaka::math::abs(acc, cot);
                  if (!((shA < 0 || passesShapeCut(finder, shA, ia, ac)) &
                        (shB < 0 || passesShapeCut(finder, shB, ib, ac))))
                    return;
                }
                if (startPair.gapAllowed() && chainParameters.useGapMap && g_lid < 0) {
                  const float za = zaf, zb = finder.hits.z(ib);
                  const float a[3] = {finder.hits.xBeam(ia) + finder.chainEvent.beamSpotX +
                                          finder.chainEvent.beamSpotDxDz * (za - finder.chainEvent.beamSpotZ),
                                      finder.hits.yBeam(ia) + finder.chainEvent.beamSpotY +
                                          finder.chainEvent.beamSpotDyDz * (za - finder.chainEvent.beamSpotZ),
                                      za};
                  const float bb[3] = {finder.hits.xBeam(ib) + finder.chainEvent.beamSpotX +
                                           finder.chainEvent.beamSpotDxDz * (zb - finder.chainEvent.beamSpotZ),
                                       finder.hits.yBeam(ib) + finder.chainEvent.beamSpotY +
                                           finder.chainEvent.beamSpotDyDz * (zb - finder.chainEvent.beamSpotZ),
                                       zb};
                  for (int h = 0; h < startPair.nHoles(); ++h) {
                    const auto H = finder.chainHoles[startPair.firstHole() + h];
                    const int lid = finder.chainPositions[H.position()].layerId();
                    if (!H.isBetweenStartHits() || lid < 0 || lid > 3 || !finder.gapMapLayers[lid].valid())
                      continue;
                    if (router.state1(acc, H.position(), z0, cot) != 2)
                      continue;
                    const float rm = finder.gapMapLayers[lid].rMid(), rb = finder.hits.r(ib);
                    const float pm = paf + wrapPhiOnce(finder.hits.phi(ib) - paf) * (rm - raf) / (rb - raf);
                    if (!segmentInGap(acc, finder, lid, a, bb, wrapPhiOnce(pm)))
                      return;
                  }
                }
                CandidateLine L{z0, cot, 0u, 0u};
                router.sweep(acc, z0, cot, L.crossedPositions, L.definitePositions);
                int holes = 0, bt = 0;
                for (int h = 0; h < startPair.nHoles(); ++h) {
                  const auto H = finder.chainHoles[startPair.firstHole() + h];
                  const int d = (L.definitePositions >> H.position()) & 1u;
                  holes += d, bt |= d & H.isBetweenStartHits();
                }
                const int sa = (L.crossedPositions >> startPair.positionA()) & 1u,
                          sb = (L.crossedPositions >> startPair.positionB()) & 1u;
                if (!(!(lead_only & bt) & (holes <= chainParameters.maxStartHoles) & sa & sb))
                  return;
                int st;
                const int nq = router.nextCrossedPositionFromMasks(
                    startPair.positionB(), L, chainParameters.allowAnyOuterTracker, st);
                if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0)
                  return;
                Candidate c;
                c.hits[0] = ka, c.hits[1] = kb, c.hits[2] = 0;
                c.positions[0] = startPair.positionA(), c.positions[1] = startPair.positionB(), c.positions[2] = 0;
                c.holes = holes, c.innerHoles = startPair.gapAllowed() & bt, c.stage = st;
                c.score = 0;
                cpu::pushCandidateWithLine(candidateQueues, q2(finder, nq), c, L);
                return;
              }
            }
            // flush_start: holes before b, the start layers crossed, the gap map, lead-only, the shape
            int holes = 0, bt = 0;
            for (int h = 0; h < startPair.nHoles(); ++h) {
              const auto H = finder.chainHoles[startPair.firstHole() + h];
              const int d = state(acc, finder.chainPositions[H.position()], z0, cot) == 2;
              holes += d, bt |= d & H.isBetweenStartHits();
            }
            const int sa = state(acc, finder.chainPositions[startPair.positionA()], z0, cot),
                      sb = state(acc, finder.chainPositions[startPair.positionB()], z0, cot);
            int gv = 1;
            if (startPair.gapAllowed() && chainParameters.useGapMap && g_lid < 0) {
              const float za = zaf, zb = finder.hits.z(ib);
              const float a[3] = {finder.hits.xBeam(ia) + finder.chainEvent.beamSpotX +
                                      finder.chainEvent.beamSpotDxDz * (za - finder.chainEvent.beamSpotZ),
                                  finder.hits.yBeam(ia) + finder.chainEvent.beamSpotY +
                                      finder.chainEvent.beamSpotDyDz * (za - finder.chainEvent.beamSpotZ),
                                  za};
              const float bb[3] = {finder.hits.xBeam(ib) + finder.chainEvent.beamSpotX +
                                       finder.chainEvent.beamSpotDxDz * (zb - finder.chainEvent.beamSpotZ),
                                   finder.hits.yBeam(ib) + finder.chainEvent.beamSpotY +
                                       finder.chainEvent.beamSpotDyDz * (zb - finder.chainEvent.beamSpotZ),
                                   zb};
              for (int h = 0; h < startPair.nHoles() && gv; ++h) {
                const auto H = finder.chainHoles[startPair.firstHole() + h];
                const int lid = finder.chainPositions[H.position()].layerId();
                if (!H.isBetweenStartHits() || lid < 0 || lid > 3 || !finder.gapMapLayers[lid].valid())
                  continue;
                if (state(acc, finder.chainPositions[H.position()], z0, cot) != 2)
                  continue;
                const float rm = finder.gapMapLayers[lid].rMid(), rb = finder.hits.r(ib);
                const float pm = paf + wrapPhiOnce(finder.hits.phi(ib) - paf) * (rm - raf) / (rb - raf);
                if (!segmentInGap(acc, finder, lid, a, bb, wrapPhiOnce(pm)))
                  gv = 0;
              }
            }
            const int inner = startPair.gapAllowed() & bt;
            int shp = !(lead_only & bt);
            if (shA >= 0 || shB >= 0) {
              const float ac = alpaka::math::abs(acc, cot);
              shp &=
                  (shA < 0 || passesShapeCut(finder, shA, ia, ac)) & (shB < 0 || passesShapeCut(finder, shB, ib, ac));
            }
            if (!(shp & gv & (holes <= chainParameters.maxStartHoles) & (sa > 0) & (sb > 0)))
              return;
            int st;
            const int nq = router.nextCrossedPosition(
                acc, finder, startPair.positionB(), z0, cot, chainParameters.allowAnyOuterTracker, st, [&] {
                  return nextCrossedPosition(
                      acc, finder, startPair.positionB(), z0, cot, chainParameters.allowAnyOuterTracker, st);
                });
            if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0)
              return;
            Candidate c;
            c.hits[0] = ka, c.hits[1] = kb, c.hits[2] = 0;
            c.positions[0] = startPair.positionA(), c.positions[1] = startPair.positionB(), c.positions[2] = 0;
            c.holes = holes, c.innerHoles = inner, c.stage = st;
            c.score = 0;
            push(acc, candidateQueues, q2(finder, nq), c);
          }  // block of testHitB
        };  // testHitB
        if constexpr (kCpuAcc<TAcc>) {
          cpu::ScanStart scan(
              finder, B, raf, paf, zaf, inva, chainParameters, startPair.cotMin(), startPair.cotMax(), zOuter, g_lid);
          for (int r = 0; r < nqr; ++r)
            forEachBinRun(finder, B, fp, qrs[r], [&](uint32_t b, uint32_t e) {
              // the doublet cuts, loose and 8 hits at a time (MkSeederKernelsCpu.h); testHitB() repeats them exactly
              scan(b, e, testHitB);
            });
        } else {
          // the hits f, f + step, ... < l of the window (in the bin runs' order); returns the window's number of hits
          auto scanHits = [&](uint32_t f, uint32_t l, uint32_t step) {
            uint32_t o = 0;
            for (int r = 0; r < nqr; ++r)
              forEachBinRun(finder, B, fp, qrs[r], [&](uint32_t b, uint32_t e) {
                for (; f < l && f < o + (e - b); f += step)
                  testHitB(b + (f - o), nullptr);
                o += e - b;
              });
            return o;
          };
          if (hitStep != 1) {
            // a warp's piece: hitStep = the warp size, hitLo - kItemHits = the lane; the lanes test consecutive hits,
            // the hits that pass the doublet cuts are gathered in lane order (pend on lanes 0..np-1) and tested by
            // testHitB 32 at a time, so its long tail runs with all lanes busy
            if constexpr (kCudaAcc<TAcc>) {
              constexpr uint32_t kWarp = 32;
              const uint32_t lane = (hitLo - kItemHits) % kWarp, lo = hitLo - lane;
              uint32_t pend = 0, np = 0, o = 0;
              for (int r = 0; r < nqr; ++r)
                forEachBinRun(finder, B, fp, qrs[r], [&](uint32_t b, uint32_t e) {
                  warpGatheredScan(
                      acc,
                      lane,
                      b,
                      o,
                      std::max(o, lo),
                      std::min(o + (e - b), hitHi),
                      pend,
                      np,
                      [&](uint32_t kb) {
                        float dr, dp, cot;
                        return doubletCuts(kb, dr, dp, cot);
                      },
                      [&](uint32_t kb) { testHitB(kb, nullptr); });
                  o += e - b;
                });
              if (lane < np)
                testHitB(pend, nullptr);
            }
            return;
          }
          const uint32_t n = scanHits(hitLo, hitHi, hitStep);
          if (pieces && n > kItemHits) {
            const uint32_t k = (n - kItemHits + kPieceHits - 1) / kPieceHits;
            const uint32_t i0 = alpaka::atomicAdd(acc, pieces->data, k, alpaka::hierarchy::Blocks{});
            for (uint32_t i = 0; i < k; ++i) {
              if (i0 + i < pieces->capacity) {
                pieces->item(i0 + i) = w, pieces->piece(i0 + i) = i;
              } else {
                scanHits(kItemHits + i * kPieceHits, kItemHits + (i + 1) * kPieceHits, 1u);
              }
            }
          }
        }
      }  // work item w
    }  // KernelFindStartDoublets::workItem()
  };  // class KernelFindStartDoublets

  // a GPU: the pieces of the long stage-b windows (WorkPieces) after KernelFindStartDoublets, one per warp (its lanes
  // take consecutive hits)
  class KernelFindStartDoubletPieces {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  ChainFinderInputs finder,
                                  Queues candidateQueues,
                                  WorkPieces pieces) const {
      if constexpr (kGpuAcc<TAcc>) {
        cpu::RouterOf<TAcc> router;
        // a GPU other than CUDA: a piece per thread
        constexpr uint32_t kWarp = kCudaAcc<TAcc> ? 32 : 1, kPieceHits = KernelFindStartDoublets::kPieceHits;
        const uint32_t n = std::min(pieces.data[0], pieces.capacity);
        const uint32_t t = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u],
                       nt = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u];
        const uint32_t lane = t % kWarp;
        for (uint32_t i = t / kWarp; i < n; i += nt / kWarp) {
          const uint32_t lo = KernelFindStartDoublets::kItemHits + pieces.piece(i) * kPieceHits;
          KernelFindStartDoublets::workItem(
              acc, finder, candidateQueues, router, false, pieces.item(i), lo + lane, lo + kPieceHits, kWarp, nullptr);
        }
      }
    }
  };

  // a GPU block (all its threads call it; it syncs them): thread t's candidate m pushed onto queue qbase + nq (nq < 0:
  // none). The block's candidates counted per target queue in shared memory and their slots reserved with one atomicAdd
  // per (block, queue); the chunks that start in the block's slots taken at once (one take from the free list, the rest
  // fresh) and published; the chunk of a queue's first slot, if it starts in another block's slots, waited for by one
  // thread (its block reserved before this one and publishes before it waits for anything); the candidates written. The
  // slots of a queue in another order than push()'s (the same contents).
  template <typename TAcc>
  ALPAKA_FN_ACC inline void blockPush(
      const TAcc &acc, const Queues &candidateQueues, int n, int qbase, int nq, const Candidate &m) {
    constexpr int kMaxPositions = 64;
    constexpr uint32_t kChunk = Queues::kCandidateChunkSize;
    ALPAKA_ASSERT_ACC(n <= kMaxPositions);
    // per target position: the block's candidates, its first slot, its first chunk it takes and their number (in the
    // block's list of taken chunks), the chunk id of its first slot
    auto &cnt = alpaka::declareSharedVar<uint32_t[kMaxPositions], __COUNTER__>(acc);
    auto &first = alpaka::declareSharedVar<uint32_t[kMaxPositions], __COUNTER__>(acc);
    auto &ownFirst = alpaka::declareSharedVar<uint32_t[kMaxPositions], __COUNTER__>(acc);
    auto &firstChunk = alpaka::declareSharedVar<int[kMaxPositions], __COUNTER__>(acc);
    // the block's chunks: their number, how many come from the free list (from freeTop on), the first fresh one
    auto &nOwn = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    auto &nFree = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    auto &freeTop = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    auto &fresh = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    const uint32_t nth = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    const uint32_t mcq = candidateQueues.maxChunksPerQueue;
    MkSeederFinderCounters *counters = candidateQueues.counters;
    for (uint32_t q = t; q < uint32_t(n); q += nth)
      cnt[q] = 0;
    if (t == 0)
      nOwn = 0;
    alpaka::syncBlockThreads(acc);
    const uint32_t rank = nq >= 0 ? alpaka::atomicAdd(acc, &cnt[nq], 1u, alpaka::hierarchy::Threads{}) : 0u;
    alpaka::syncBlockThreads(acc);
    // the slots per target queue and the chunks that start in them
    for (uint32_t q = t; q < uint32_t(n); q += nth) {
      if (cnt[q] == 0)
        continue;
      const uint32_t s0 =
          alpaka::atomicAdd(acc, &candidateQueues.queues.count(qbase + int(q)), cnt[q], alpaka::hierarchy::Blocks{});
      const uint32_t c0 = (s0 + kChunk - 1) / kChunk, c1 = std::min((s0 + cnt[q] - 1) / kChunk + 1, mcq);
      first[q] = s0;
      ownFirst[q] = alpaka::atomicAdd(acc, &nOwn, c1 > c0 ? c1 - c0 : 0u, alpaka::hierarchy::Threads{});
    }
    alpaka::syncBlockThreads(acc);
    if (t == 0 && nOwn > 0) {
      // the free chunks first (the free list only grows between the kernels that take chunks), then fresh ones
      uint32_t fc = alpaka::atomicAdd(acc, &counters->nFreeChunks, 0u, alpaka::hierarchy::Blocks{}), k = 0;
      while (fc > 0) {
        k = std::min(nOwn, fc);
        const uint32_t prev = alpaka::atomicCas(acc, &counters->nFreeChunks, fc, fc - k, alpaka::hierarchy::Blocks{});
        if (prev == fc)
          break;
        fc = prev, k = 0;
      }
      nFree = k, freeTop = fc - k;
      fresh = k < nOwn ? alpaka::atomicAdd(acc, &counters->nextFreshChunk, nOwn - k, alpaka::hierarchy::Blocks{}) : 0u;
    }
    alpaka::syncBlockThreads(acc);
    for (uint32_t q = t; q < uint32_t(n); q += nth) {
      if (cnt[q] == 0)
        continue;
      const uint32_t s0 = first[q];
      const uint32_t c0 = (s0 + kChunk - 1) / kChunk, c1 = std::min((s0 + cnt[q] - 1) / kChunk + 1, mcq);
      for (uint32_t ci = c0; ci < c1; ++ci) {
        const uint32_t o = ownFirst[q] + ci - c0;
        const uint32_t f = fresh + (o - nFree);
        const int cid =
            o < nFree ? candidateQueues.freeChunks.chunk(freeTop + o) : (f < candidateQueues.poolChunks ? int(f) : -2);
        alpaka::atomicExch(
            acc, &candidateQueues.chunkTable.chunk((qbase + int(q)) * mcq + ci), cid, alpaka::hierarchy::Blocks{});
      }
    }
    alpaka::syncBlockThreads(acc);
    for (uint32_t q = t; q < uint32_t(n); q += nth) {
      if (cnt[q] == 0 || first[q] % kChunk == 0 || first[q] / kChunk >= mcq)
        continue;
      // the chunk of the first slot: taken by the block (or the earlier kernel) that reserved its slot 0
      int *tb = &candidateQueues.chunkTable.chunk((qbase + int(q)) * mcq + first[q] / kChunk);
      int cid;
      do {
        cid = alpaka::atomicCas(acc, tb, -1, -1, alpaka::hierarchy::Blocks{});
      } while (cid == -1);
      firstChunk[q] = cid;
    }
    alpaka::syncBlockThreads(acc);
    if (nq >= 0) {
      const uint32_t s = first[nq] + rank, ci = s / kChunk;
      const int cid = ci >= mcq ? -2
                      : ci == first[nq] / kChunk && first[nq] % kChunk != 0
                          ? firstChunk[nq]
                          : candidateQueues.chunkTable.chunk((qbase + nq) * mcq + ci);
      if (cid < 0)
        alpaka::atomicAdd(acc, &counters->nLost, 1u, alpaka::hierarchy::Blocks{});
      else
        storeCandidate(candidateQueues.pool, (uint32_t)cid * kChunk + s % kChunk, m);
    }
    alpaka::syncBlockThreads(acc);
  }

  // CUDA: a warp's pushes staged in shared memory (up to kCap per warp): add() from any lane at any time (a shared
  // atomicAdd; a full stage: the lane routes and pushes the candidate itself), flush() from all the warp's lanes
  // together: each lane routes a staged candidate (route(m) -> its target position, -1: dropped; it may set m's fields),
  // then per target queue one atomicAdd on its count for the lanes' candidates and pushAt(). Another GPU: add() routes
  // and pushes. A push() from a divergent loop waits for its atomicAdd on a count that many threads increment, and the
  // routing runs on the few lanes that found something: the stage runs both with the warp's lanes together.
  template <uint32_t kCap, uint32_t kMaxWarps>
  struct WarpPushStage {
    Candidate *cand;
    uint32_t *count;
    template <typename TAcc>
    ALPAKA_FN_ACC static WarpPushStage make(const TAcc &acc) {
      auto &c = alpaka::declareSharedVar<Candidate[kCap * kMaxWarps], __COUNTER__>(acc);
      auto &n = alpaka::declareSharedVar<uint32_t[kMaxWarps], __COUNTER__>(acc);
      const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u], w = t / 32u;
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] <= 32u * kMaxWarps));
      if (t % 32u == 0)
        n[w] = 0;
      alpaka::syncBlockThreads(acc);
      return {c + w * kCap, n + w};
    }
    template <typename TAcc, typename R>
    ALPAKA_FN_ACC void add(
        const TAcc &acc, const Queues &candidateQueues, int qbase, const Candidate &m, R &&route) const {
      if constexpr (kCudaAcc<TAcc>) {
        const uint32_t k = alpaka::atomicAdd(acc, count, 1u, alpaka::hierarchy::Threads{});
        if (k < kCap) {
          cand[k] = m;
          return;
        }
      }
      Candidate o = m;
      const int nq = route(o);
      if (nq >= 0)
        push(acc, candidateQueues, qbase + nq, o);
    }
    template <typename TAcc, typename R>
    ALPAKA_FN_ACC void flush(const TAcc &acc, const Queues &candidateQueues, int qbase, R &&route) const {
      if constexpr (kCudaAcc<TAcc>) {
#if defined(__CUDA_ARCH__)
        __syncwarp();
        const uint32_t ns = std::min(*count, kCap), lane = threadIdx.x % 32u;
        for (uint32_t e0 = 0; e0 < ns; e0 += 32u) {
          const uint32_t nl = std::min(ns - e0, 32u);
          if (lane < nl) {
            Candidate o = cand[e0 + lane];
            const int nq = route(o);
            const uint32_t valid = __ballot_sync(nl == 32u ? 0xffffffffu : (1u << nl) - 1u, nq >= 0);
            if (nq >= 0) {
              const int q = qbase + nq;
              uint32_t *cnt = &candidateQueues.queues.count(q);
              const uint32_t peers = __match_any_sync(valid, (unsigned long long)cnt);
              const uint32_t leader = __ffs(peers) - 1;
              uint32_t base = 0;
              if (lane == leader)
                base = atomicAdd(cnt, (uint32_t)__popc(peers));
              base = __shfl_sync(peers, base, leader);
              pushAt(acc, candidateQueues, q, base + __popc(peers & ((1u << lane) - 1u)), o);
            }
          }
        }
        __syncwarp();
        if (lane == 0)
          *count = 0;
        __syncwarp();
#endif
      }
    }
  };

  // ---- stage c at position p
  // One work item per (candidate, q-bin row): items r < kRows - 1 take row r of the candidate's window, the last one
  // the rest, so a wide window (barrel doublets on the first discs) is spread over kRows threads. A candidate with a
  // triplet sets its found flag; KernelExtendToTripletsMiss forwards the others.
  class KernelExtendToTriplets {
  public:
    static constexpr uint32_t kRows = 4, kStage = 64;
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p) const {
      if constexpr (kCpuAcc<TAcc>) {
        cpu::extendDoubletsToTriplets(acc, finder, candidateQueues, p);  // with the misses (MkSeederKernelsCpu.h)
        return;
      }
      const MkSeederChainParameters &chainParameters = finder.chain;
      const int n = chainParameters.nPositions;
      const EventLayer targetLayer = finder.layers[finder.chainPositions[p].binnedLayerId()];
      const LayerAxes Tg = finder.layerGeometry[finder.chainPositions[p].binnedLayerId()];
      const bool disc = Tg.isDisc();
      const float u0 = targetLayer.qbarLo(), u1 = targetLayer.qbarHi();
      const int shT = shapeOfPosition(finder, p);
      const float d0c = stageCD0Max(finder, p);
      const uint32_t nc = std::min(candidateQueues.queues.count(q2(finder, p)),
                                   candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
      // the triplets' pushes staged per warp (WarpPushStage), flushed between the rounds of items: a block-uniform loop
      // (every thread of a block runs the same rounds), a block takes its next round of items from a grid-wide counter
      // (the blocks beyond the ones that fit on the device at once do not run their own share after them)
      const auto stage = WarpPushStage<kStage, 4>::make(acc);
      // a triplet t3: its a-c line, its next crossed position from p (-1: none) and its state there
      auto route = [&](Candidate &t3) {
        const uint32_t ia =
            finder.layers[finder.chainPositions[t3.positions[0]].binnedLayerId()].firstHit() + t3.hits[0];
        const uint32_t ih = targetLayer.firstHit() + t3.hits[2];
        const float za = finder.hits.z(ia), ra = finder.hits.r(ia), zc = finder.hits.z(ih), rc = finder.hits.r(ih);
        const float cot = (zc - za) / (rc - ra), z0 = za - cot * ra;
        int st;
        const int nq =
            nextCrossedPosition(acc, finder, p, z0, cot, t3.holes <= chainParameters.maxHolesForOuterTracker, st);
        if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0)
          return -1;
        t3.stage = st;
        return nq;
      };
      const uint32_t tb = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const uint32_t nth = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const uint32_t nItems = nc * kRows;
      auto &roundFirst = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
      // this thread's item of the block's next round (the round's first item >= nItems: none left)
      auto nextItem = [&]() {
        alpaka::syncBlockThreads(acc);
        if (tb == 0)
          roundFirst = alpaka::atomicAdd(acc, &candidateQueues.counters->stageCItems, nth, alpaka::hierarchy::Blocks{});
        alpaka::syncBlockThreads(acc);
        return roundFirst + tb;
      };
      for (uint32_t wj = nextItem(); wj - tb < nItems; wj = nextItem()) {
        stage.flush(acc, candidateQueues, q3(finder, 0), route);
        if (wj >= nItems)
          continue;
        const uint32_t j = wj / kRows, row = wj % kRows;
        Candidate c;
        if (!readCandidate(candidateQueues, q2(finder, p), j, c))
          continue;
        float cz0, ccot;
        line(acc, finder, c, cz0, ccot);
        const int ix = finder.windowTableByTriplet[(c.positions[0] * n + c.positions[1]) * n + p].windowTable();
        const float fks_c =
            chainParameters.fakeScoreCutForward > 0 && alpaka::math::abs(acc, ccot) >= chainParameters.forwardMinAbsCot
                ? chainParameters.fakeScoreCutForward
                : chainParameters.fakeScoreCut;
        if (searchesCombination(chainParameters, ix)) {
          const auto w = finder.windowTables[ix >= 0 ? ix : 0];
          const EventLayer La = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()],
                           Lb = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()];
          const uint32_t ia = La.firstHit() + c.hits[0], ib = Lb.firstHit() + c.hits[1];
          const float ua = disc ? finder.hits.z(ia) : finder.hits.r(ia),
                      ub = disc ? finder.hits.z(ib) : finder.hits.r(ib);
          const float qa = disc ? finder.hits.r(ia) : finder.hits.z(ia),
                      qb = disc ? finder.hits.r(ib) : finder.hits.z(ib);
          const float pa = finder.hits.phi(ia), dp = wrapPhiFloor(acc, finder.hits.phi(ib) - pa);
          const float iA = finder.hits.invR(ia), iB = finder.hits.invR(ib), idu = 1.0f / (ub - ua);
          const float t0_ = (u0 - ua) * idu, t1_ = (u1 - ua) * idu;
          if (std::max(t0_, t1_) > 1) {
            const float cq0 = qa + (qb - qa) * t0_, cq1 = qa + (qb - qa) * t1_;
            const float cp0 = pa + dp * t0_, cp1 = pa + dp * t1_;
            const bool far0 = alpaka::math::abs(acc, t0_) > alpaka::math::abs(acc, t1_);
            const float ufar = far0 ? u0 : u1, tfar = far0 ? t0_ : t1_;
            const float rfar = std::max(0.5f, disc ? qa + (qb - qa) * tfar : ufar);
            const float wcphi = d0c * alpaka::math::abs(acc, 1 / rfar - (iA + (iB - iA) * tfar)) + w.phiWindowC();
            BinRange qc =
                qBinRangePadded(acc, Tg, std::min(cq0, cq1) - w.qWindowC(), std::max(cq0, cq1) + w.qWindowC());
            // this item's rows
            qc.first += row;
            if (row + 1 < kRows)
              qc.last = std::min(qc.last, qc.first + 1);
            if (qc.first >= qc.last)
              continue;
            const float dcp = wrapPhiFloor(acc, cp1 - cp0), cmid = cp0 + 0.5f * dcp,
                        chalf = 0.5f * alpaka::math::abs(acc, dcp);
            const float qcw = w.qWindowC(), pcw = w.phiWindowC(), dqab = qb - qa, dib = iB - iA, iqcw = 1.0f / qcw;
            int slo = 0, shi = 1 << 30;
            if (shT >= 0) {
              const int band = shapeBand(finder, shT, shapeBin(finder, shT, alpaka::math::abs(acc, dqab * idu)));
              slo = finder.shapeBands[band].spanLow(), shi = finder.shapeBands[band].spanHigh();
            }
            forEachBinRun(finder,
                          targetLayer,
                          phiBinRangeAround(acc, Tg, cmid, chalf + wcphi + 1e-6f),
                          qc,
                          [&](uint32_t b, uint32_t e) {
                            for (uint32_t i = b; i < e; ++i) {
                              const uint32_t ih = targetLayer.firstHit() + i;
                              const float hu = disc ? finder.hits.z(ih) : finder.hits.r(ih),
                                          hq = disc ? finder.hits.r(ih) : finder.hits.z(ih);
                              const float t = (hu - ua) * idu;
                              const float dq = hq - (qa + dqab * t);
                              if (!(t > 1) || !(alpaka::math::abs(acc, dq) <= qcw))
                                continue;
                              const float dph = wrapPhiFloor(acc, wrapPhiFloor(acc, finder.hits.phi(ih) - pa) - dp * t);
                              const float wph =
                                  d0c * alpaka::math::abs(acc, finder.hits.invR(ih) - (iA + dib * t)) + pcw;
                              if (!(alpaka::math::abs(acc, dph) <= wph))
                                continue;
                              float sc = 0;
                              if (chainParameters.useFakeScore) {
                                const float rq = dq * iqcw, rp = dph / wph;
                                sc = rq * rq + rp * rp;
                                if (sc >= fks_c || finder.hits.clusterCols(ih) < slo ||
                                    (finder.hits.clusterCols(ih) > shi && !shapeExempt(finder, ih)))
                                  continue;
                              }
                              if ((finder.psMode & 4) && psUnmarked(finder, finder.chainPositions[p].layerId(), ih))
                                continue;
                              if (psRejectStageC2(acc, finder, c, p, ih))
                                continue;
                              candidateQueues.tripletStage.foundTriplet(j) = 1;
                              // the triplet, routed by route() (at the stage's flush)
                              Candidate t3 = c;
                              t3.hits[2] = i, t3.positions[2] = p, t3.score = sc;
                              stage.add(acc, candidateQueues, q3(finder, 0), t3, route);
                            }
                          });
          }
        }
      }
      stage.flush(acc, candidateQueues, q3(finder, 0), route);
    }
  };

  // a chunk for the pool (a freed one first), outside the concurrent pushes: one thread only
  ALPAKA_FN_HOST_ACC inline int takeChunkSerial(const Queues &candidateQueues) {
    if (candidateQueues.counters->nFreeChunks > 0)
      return candidateQueues.freeChunks.chunk(--candidateQueues.counters->nFreeChunks);
    const uint32_t nc = candidateQueues.counters->nextFreshChunk++;
    return nc < candidateQueues.poolChunks ? (int)nc : -2;
  }

  // ---- a GPU with trip_net_c: the stage-c network (MkSeederStageCNet.h) over the triplet queue of position pf before its
  // stage d, in the launch of KernelExtendToTripletsMiss at the position before pf (stage c there was the last to push
  // triplets onto pf's queue; a pass of its own: inside KernelExtendToTriplets the network's registers slowed stage c by
  // ~6 ms/ev even when off; in a kernel of its own the launches cost more than the pass saves in stage d). A thread per
  // triplet, the kept ones pushed onto the spare queue (blockPush); the grid's last block then gives pf's triplet queue
  // the spare queue's chunks and count (its own chunks back to the free list; no block takes chunks any more) and empties
  // the spare queue. The logit with the serial backend's arithmetic (stageCNetLogitOps): the same triplets pass as in
  // its stage c. The queue also holds the triplets that stage d forwarded from an earlier position: they passed there,
  // with the same logit (the same three hits) they pass again; stage d's later forwards onto pf's queue passed too.
  template <typename TAcc>
  ALPAKA_FN_ACC inline void stageCNetFilterQueue(
      const TAcc &acc, const ChainFinderInputs &finder, const Queues &candidateQueues, int pf, int spare) {
    const int queue = q3(finder, pf);
    const uint32_t mcq = candidateQueues.maxChunksPerQueue;
    const uint32_t nc = std::min(candidateQueues.queues.count(queue), mcq * Queues::kCandidateChunkSize);
    const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    const uint32_t nth = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    const uint32_t nb = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    for (uint32_t j0 = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] * nth; j0 < nc; j0 += nb * nth) {
      const uint32_t j = j0 + t;
      Candidate c{};
      int keep = -1;
      if (j < nc && readCandidate(candidateQueues, queue, j, c)) {
        float hx[3], hy[3], hz[3], rows[3], cols[3];
        for (int i = 0; i < 3; ++i) {
          const uint32_t h =
              finder.layers[finder.chainPositions[c.positions[i]].binnedLayerId()].firstHit() + c.hits[i];
          hx[i] = finder.hits.x(h), hy[i] = finder.hits.y(h), hz[i] = finder.hits.z(h);
          rows[i] = float(std::min(32, int(finder.hits.clusterRows(h))));
          cols[i] = float(std::min(32, int(finder.hits.clusterCols(h))));
        }
        float lg, acot;
        const bool ok = stageCNetLogitOps<libm::DeviceOps>(hx,
                                                           hy,
                                                           hz,
                                                           rows,
                                                           cols,
                                                           finder.chainEvent.beamSpotX,
                                                           finder.chainEvent.beamSpotY,
                                                           finder.chainEvent.beamSpotZ,
                                                           lg,
                                                           acot);
        keep = ok && lg < stageCNetCut(finder.chain.stageCNetCut, acot) ? -1 : 0;
      }
      blockPush(acc, candidateQueues, 1, spare, keep, c);
    }
    // the last block to finish swaps the queues
    auto &last = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    alpaka::syncBlockThreads(acc);
    if (t == 0) {
      alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
      last =
          alpaka::atomicAdd(acc, &candidateQueues.counters->stageCNetBlocks, 1u, alpaka::hierarchy::Blocks{}) == nb - 1;
    }
    alpaka::syncBlockThreads(acc);
    if (!last)
      return;
    alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
    // the other blocks' counts and chunk entries through atomics (not a stale cached copy)
    const uint32_t nSpare =
        alpaka::atomicAdd(acc, &candidateQueues.queues.count(spare), 0u, alpaka::hierarchy::Blocks{});
    auto chunks = [&](uint32_t count) {
      return std::min((count + Queues::kCandidateChunkSize - 1) / Queues::kCandidateChunkSize, mcq);
    };
    const uint32_t nOld = chunks(candidateQueues.queues.count(queue)), nNew = chunks(nSpare);
    for (uint32_t i = t; i < std::max(nOld, nNew); i += nth) {
      int &o = candidateQueues.chunkTable.chunk(queue * mcq + i);
      if (i < nOld && o >= 0)
        candidateQueues.freeChunks.chunk(
            alpaka::atomicAdd(acc, &candidateQueues.counters->nFreeChunks, 1u, alpaka::hierarchy::Blocks{})) = o;
      int *s = &candidateQueues.chunkTable.chunk(spare * mcq + i);
      o = i < nNew ? alpaka::atomicExch(acc, s, -1, alpaka::hierarchy::Blocks{}) : -1;
    }
    alpaka::syncBlockThreads(acc);
    if (t == 0) {
      candidateQueues.queues.count(queue) = nSpare;
      candidateQueues.queues.count(spare) = 0;
      candidateQueues.counters->stageCNetBlocks = 0;
    }
  }

  // ---- stage c's misses at position p, forwarded on their own line (blockPush), a block of candidates at a time; with
  // trip_net_c (filterPosition >= 0) also the stage-c network's pass over the next position's triplets
  // (stageCNetFilterQueue)
  class KernelExtendToTripletsMiss {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p, int filterPosition, int spare) const {
      const MkSeederChainParameters &chainParameters = finder.chain;
      const int n = chainParameters.nPositions;
      const uint32_t nc = std::min(candidateQueues.queues.count(q2(finder, p)),
                                   candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
      // stage c's item counter for its next launch (KernelExtendToTriplets has finished)
      if (cms::alpakatools::once_per_grid(acc))
        candidateQueues.counters->stageCItems = 0;
      const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const uint32_t nth = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const uint32_t nb = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      for (uint32_t j0 = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] * nth; j0 < nc; j0 += nb * nth) {
        const uint32_t j = j0 + t;
        Candidate m;
        int nq = -1;
        if (j < nc) {
          const bool found = candidateQueues.tripletStage.foundTriplet(j);
          candidateQueues.tripletStage.foundTriplet(j) = 0;  // for the next position
          Candidate c;
          if (readCandidate(candidateQueues, q2(finder, p), j, c) && !(found && !chainParameters.alwaysForwardMisses)) {
            const int h = c.holes + (c.stage == 2);
            if (h <= chainParameters.maxHoles) {
              float cz0, ccot;
              line(acc, finder, c, cz0, ccot);
              int st;
              nq = nextCrossedPosition(acc, finder, p, cz0, ccot, chainParameters.allowAnyOuterTracker, st);
              if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0 || skipOuterTrackerMiss(finder, found, nq) ||
                  skipMissOutsideGap(finder, c, p, cz0, ccot))
                nq = -1;
              else {
                m = c;
                m.innerHoles += h > c.holes;
                m.holes = h, m.stage = st;
              }
            }
          }
        }
        blockPush(acc, candidateQueues, n, q2(finder, 0), nq, m);
      }
      if constexpr (kGpuAcc<TAcc>)
        if (filterPosition >= 0)
          stageCNetFilterQueue(acc, finder, candidateQueues, filterPosition, spare);
    }
  };

  // ---- stage d at position p
  // stage d: the gates of a triplet seed of candidate c with no fourth hit (its |cot| range; minPixelHits: a triplet of
  // pixel positions has 3)
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool tripletSeedGates(const TAcc &acc,
                                             const ChainFinderInputs &finder,
                                             const Candidate &c,
                                             float ccot) {
    const MkSeederChainParameters &chainParameters = finder.chain;
    if (!chainParameters.emitTriplets)
      return false;
    const float ac = alpaka::math::abs(acc, ccot);
    if (ac < chainParameters.tripletMinAbsCot || ac >= chainParameters.tripletMaxAbsCot)
      return false;
    return !(chainParameters.minPixelHits > 0 && finder.chainPositions[c.positions[0]].isPixel() +
                                                         finder.chainPositions[c.positions[1]].isPixel() +
                                                         finder.chainPositions[c.positions[2]].isPixel() <
                                                     chainParameters.minPixelHits);
  }
  // stage d, a candidate past tripletSeedGates() (not the serial backend's batched MLP): the triplet MLP and the triplet
  // seed's emission
  template <typename TAcc>
  ALPAKA_FN_ACC inline void emitTripletSeed(const TAcc &acc,
                                            const ChainFinderInputs &finder,
                                            const Queues &candidateQueues,
                                            const Candidate &c,
                                            uint32_t ia,
                                            uint32_t ib,
                                            uint32_t ic) {
    const MkSeederChainParameters &chainParameters = finder.chain;
    bool tfail = false;
    if (chainParameters.tripletMlpCut > 0.f) {
      float ts;
      if constexpr (kCpuAcc<TAcc>)
        ts = cpu::scoreTriplet(acc, finder, ia, ib, ic);  // the same score, the MLP in SIMD (MkSeederKernelsCpu.h)
      else
        ts = scoreTriplet(acc, finder, ia, ib, ic);
      if (ts < chainParameters.tripletMlpCut) {
        if (chainParameters.tripletMlpRescueOuterTrackerHits > 0 && ts >= chainParameters.tripletMlpRescueMinScore)
          tfail = true;  // trip_dnn_ext: kept for the extension's probe
        else
          return;
      }
    }
    Quad o;
    o.layers[0] = finder.chainPositions[c.positions[0]].layerId(),
    o.layers[1] = finder.chainPositions[c.positions[1]].layerId(),
    o.layers[2] = finder.chainPositions[c.positions[2]].layerId();
    o.layers[3] = -1;
    o.hits[0] = finder.hits.clusterIndex(ia), o.hits[1] = finder.hits.clusterIndex(ib),
    o.hits[2] = finder.hits.clusterIndex(ic), o.hits[3] = 0;
    o.score = c.score, o.fakeScore = c.score, o.tripletFailedMlp = tfail;
    emit(acc, candidateQueues, o);
  }

  // kWhole (a CPU backend, and a GPU at the other positions): one launch does everything. A GPU at the positions where
  // splitAt() (the long windows) runs KernelScanQuadWindows (the hits that can pass, as (candidate, hit) pairs), then
  // kPairs (a thread per pair: the candidate's prologue, the hit's tests, take(); found into the candidate's flag in
  // tripletStage.foundTriplet, which KernelExtendToTripletsMiss left cleared: 1 found, 4 its pairs did not fit; then a
  // thread per candidate with flag 4: its whole window) and KernelForwardQuadCandidates (a thread per candidate: the
  // forwarding / triplet seed; the flags cleared).
  class KernelExtendToQuads {
  public:
    // kWhole: the candidates' windows scanned and the candidates forwarded (a CPU backend; a GPU at the positions not
    // split); kPairs (a GPU at a split position): the (candidate, hit) pairs of KernelScanQuadWindows, then the whole windows
    // of the candidates whose pairs did not fit (flag 4), their found flags for KernelForwardQuadCandidates
    static constexpr int kWhole = 0, kPairs = 1;
    // whether a GPU splits the launch at a position of mkFit layer l: where it is faster (tt5k PU200, with the stage-c
    // network: L4, OT1-P, OT2-P, the pixel discs from the third on; not L3 (no triplets reach it), the first two discs,
    // OT3-P (few candidates))
    static constexpr bool splitAt(int l) {
      return l == 3 || l == 4 || l == 6 || (l >= 18 && l <= 27) || (l >= 40 && l <= 49);
    }
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p, WorkPieces pieces, int mode) const {
      const MkSeederChainParameters &chainParameters = finder.chain;
      const int n = chainParameters.nPositions;
      const EventLayer targetLayer = finder.layers[finder.chainPositions[p].binnedLayerId()];
      const LayerAxes Tg = finder.layerGeometry[finder.chainPositions[p].binnedLayerId()];
      const bool disc = Tg.isDisc();
      const float u0 = targetLayer.qbarLo(), u1 = targetLayer.qbarHi();
      const int shT = shapeOfPosition(finder, p);
      const float fks = chainParameters.fakeScoreCut;
      // take() (the 4th-hit tests, the OT confirmations, the emission) reads the inputs and the queues through fin and qs
      // (blockCopyOfArgument)
      const ChainFinderInputs &fin = blockCopyOfArgument(acc, finder);
      const Queues &qs = blockCopyOfArgument(acc, candidateQueues);
      cpu::RouterOf<TAcc> router;
      router.init(finder);
      // the serial backend: the candidates' lines and crossed positions from the pool (CandidateLine)
      bool ml = false;
      if constexpr (kSerialAcc<TAcc>)
        ml = candidateQueues.hasLines && router.use32;
      // the serial backend: the triplet MLP of 4 triplets at a time, the emits in their order (MkSeederKernelsCpu.h)
      [[maybe_unused]] std::conditional_t<kSerialAcc<TAcc>, cpu::PendingEmits, int> pend;
      const uint32_t nc = std::min(candidateQueues.queues.count(q3(finder, p)),
                                   candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
      // the serial backend with the CandidateLine: the prologues of kDB candidates at a time (cpu::doubletPrologue)
      constexpr uint32_t kDB = cpu::kDProBlock;
      [[maybe_unused]] std::conditional_t<kSerialAcc<TAcc>, cpu::DPro[kDB], int> dpro;
      // a GPU's kPairs: a (candidate, hit) pair per thread, then (if the pair list overflowed) a candidate per thread
      const bool pc = kGpuAcc<TAcc> && mode == kPairs;
      const uint32_t npairs = pc ? std::min(pieces.data[0], pieces.capacity) : 0u;
      const uint32_t nIt = pc ? npairs + (pieces.data[0] > pieces.capacity ? nc : 0u) : nc;
      for (uint32_t e : cms::alpakatools::uniform_elements(acc, nIt)) {
        const bool pairItem = pc && e < npairs;
        const uint32_t j = pairItem ? pieces.item(e) : e - npairs;
        // kPairs: a pair of a candidate with flag 4 is skipped, a candidate with flag 4 scanned
        bool scan = !pairItem;
        if constexpr (kGpuAcc<TAcc>) {
          if (pc && (pairItem == bool(candidateQueues.tripletStage.foundTriplet(j) & 4)))
            continue;
        }
        Candidate c;
        [[maybe_unused]] CandidateLine L;
        const cpu::DPro *dp = nullptr;
        if constexpr (kSerialAcc<TAcc>) {
          if (ml) {
            if (j % kDB == 0)
              cpu::doubletPrologue(acc, finder, candidateQueues, p, disc, u0, u1, j, std::min(nc, j + kDB), dpro);
            dp = &dpro[j % kDB];
            if (!dp->got)
              continue;
            c = dp->c, L = dp->L;
          } else if (!readCandidate(candidateQueues, q3(finder, p), j, c))
            continue;
        } else if (!readCandidate(candidateQueues, q3(finder, p), j, c))
          continue;
        float cz0, ccot;
        if (ml)
          cz0 = L.z0, ccot = L.cot;
        else
          line(acc, finder, c, cz0, ccot);
        const EventLayer La = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()],
                         Lb = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()],
                         Lc = finder.layers[finder.chainPositions[c.positions[2]].binnedLayerId()];
        const uint32_t ia = La.firstHit() + c.hits[0], ib = Lb.firstHit() + c.hits[1], ic = Lc.firstHit() + c.hits[2];
        HelixF Hloc;
        if (!kCpuAcc<TAcc> && (scan || pairItem))
          Hloc.make(acc,
                    finder.hits.xBeam(ia),
                    finder.hits.yBeam(ia),
                    finder.hits.z(ia),
                    finder.hits.xBeam(ib),
                    finder.hits.yBeam(ib),
                    finder.hits.xBeam(ic),
                    finder.hits.yBeam(ic),
                    finder.hits.z(ic));
        const int ix =
            dp ? dp->ix
               : finder.windowTableByQuad[((c.positions[0] * n + c.positions[1]) * n + c.positions[2]) * n + p]
                     .windowTable();
        bool found = false;
        const bool inner_ok =
            !chainParameters.innerOuterTrackerOnly || !c.innerHoles || !finder.chainPositions[p].isPixel();
        // a CPU backend: the helix only for the candidates that search the layer (H is not used otherwise)
        if constexpr (kCpuAcc<TAcc>)
          if (!dp && searchesCombination(chainParameters, ix) && inner_ok)
            Hloc.make(acc,
                      finder.hits.xBeam(ia),
                      finder.hits.yBeam(ia),
                      finder.hits.z(ia),
                      finder.hits.xBeam(ib),
                      finder.hits.yBeam(ib),
                      finder.hits.xBeam(ic),
                      finder.hits.yBeam(ic),
                      finder.hits.z(ic));
        const HelixF &H = dp ? dp->H : Hloc;
        auto tripletSeed = [&]() {
          if (!tripletSeedGates(acc, finder, c, ccot))
            return;
          if constexpr (kSerialAcc<TAcc>) {
            if (chainParameters.tripletMlpCut > 0.f) {
              Quad o;
              o.layers[0] = finder.chainPositions[c.positions[0]].layerId(),
              o.layers[1] = finder.chainPositions[c.positions[1]].layerId(),
              o.layers[2] = finder.chainPositions[c.positions[2]].layerId();
              o.layers[3] = -1;
              o.hits[0] = finder.hits.clusterIndex(ia), o.hits[1] = finder.hits.clusterIndex(ib),
              o.hits[2] = finder.hits.clusterIndex(ic), o.hits[3] = 0;
              o.score = c.score, o.fakeScore = c.score;
              pend.trip(acc, finder, candidateQueues, o, ia, ib, ic);
              return;
            }
          }
          emitTripletSeed(acc, finder, candidateQueues, c, ia, ib, ic);
        };
        if ((scan || pairItem) && searchesCombination(chainParameters, ix) && H.ok && inner_ok) {
          const auto w = finder.windowTables[ix >= 0 ? ix : 0];
          float aphi = w.phiWindowA(), bphi = w.phiWindowB(), aq = w.qWindowA(), bq = w.qWindowB();
          if (dp)
            aphi = dp->aphi, bphi = dp->bphi, aq = dp->aq, bq = dp->bq;
          else if (w.nEtaWindows() > 0) {
            const float ae = alpaka::math::asinh(acc, alpaka::math::abs(acc, H.cot));
            for (int i = 0; i < w.nEtaWindows(); ++i) {
              const auto E = finder.etaWindows[w.firstEtaWindow() + i];
              if (ae >= E.absEtaLow() && ae < E.absEtaHigh()) {
                aphi = E.phiWindowA(), bphi = E.phiWindowB(), aq = E.qWindowA(), bq = E.qWindowB();
                break;
              }
            }
          }
          // a merged-cluster candidate among the triplet's hits: the windows widened (mrg_win; the serial
          // backend's prologue did it already)
          if (!dp && chainParameters.mergedWindowScale > 0.f &&
              (::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ia)) ||
               ::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ib)) ||
               ::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ic)))) {
            const float s = chainParameters.mergedWindowScale;
            aphi *= s, bphi *= s, aq *= s, bq *= s;
          }
          float x0, y0, q0, s0 = -1, x1, y1, q1, s1 = -1;
          bool ok0, ok1;
          if (dp)
            ok0 = dp->ok0, ok1 = dp->ok1, x0 = dp->x0, y0 = dp->y0, q0 = dp->q0, s0 = dp->s0, x1 = dp->x1, y1 = dp->y1,
            q1 = dp->q1, s1 = dp->s1;
          else
            ok0 = H.at(acc, disc, u0, x0, y0, q0, s0), ok1 = H.at(acc, disc, u1, x1, y1, q1, s1);
          if (ok0 || ok1) {
            if (!dp) {
              if (!ok0)
                x0 = x1, y0 = y1, q0 = q1;
              if (!ok1)
                x1 = x0, y1 = y0, q1 = q0;
            }
            const float p0 = dp ? dp->p0 : libm::atan2(acc, y0, x0), p1 = dp ? dp->p1 : libm::atan2(acc, y1, x1);
            const float pte = H.pt(acc), ipt = 1.0f / std::max(chainParameters.ptMin, pte);
            const float isr = w.referencePathLength() > 0 ? 1.0f / w.referencePathLength() : 0.0f;
            const float smax = std::max(s0, s1), lmax = isr > 0 && smax > 0 ? smax * isr : 1.0f;
            const float wpd = aphi + lmax * bphi * ipt, wqd = aq + lmax * bq * ipt;
            const BinRange qd = dp ? dp->qd : qBinRangePadded(acc, Tg, std::min(q0, q1) - wqd, std::max(q0, q1) + wqd);
            const float dpp = wrapPhiFloor(acc, p1 - p0), dmid = p0 + 0.5f * dpp,
                        dhalf = 0.5f * alpaka::math::abs(acc, dpp);
            const float bqi = bq * ipt, bpi = bphi * ipt;
            const float sw = dp ? dp->sw : alpaka::math::sin(acc, wpd), sw2 = sw * sw;
            const float fkd = (chainParameters.fakeScoreCutForward > 0 &&
                                       alpaka::math::abs(acc, ccot) >= chainParameters.forwardMinAbsCot
                                   ? chainParameters.fakeScoreCutForward
                                   : fks) -
                              c.score;
            int slo = 0, shi = 1 << 30;
            if (shT >= 0) {
              const int band = shapeBand(finder, shT, shapeBin(finder, shT, alpaka::math::abs(acc, ccot)));
              slo = finder.shapeBands[band].spanLow(), shi = finder.shapeBands[band].spanHigh();
            }
            // minPixelHits (min_pix_ot for an outer-tracker third hit): the quads of this candidate at p are dropped
            const int mp = minPixelHitsOfThird(finder, c.positions[2]);
            const bool minPixFail = mp > 0 && finder.chainPositions[c.positions[0]].isPixel() +
                                                      finder.chainPositions[c.positions[1]].isPixel() +
                                                      finder.chainPositions[c.positions[2]].isPixel() +
                                                      finder.chainPositions[p].isPixel() <
                                                  mp;
            auto take = [&](uint32_t ih, float dq, float wq, float c2, float sn, float s2) {
              if ((fin.psMode & 4) && psUnmarked(fin, fin.chainPositions[p].layerId(), ih))
                return;
              if (fin.psCurvatureSigma > 0) {
                const int lp = fin.chainPositions[p].layerId();
                if (lp == 4 || lp == 6 || lp == 8) {
                  const uint32_t c = uint32_t(fin.layers[lp].firstOTHit()) + (ih - fin.layers[lp].firstHit());
                  if (fin.otHits.psMatches(c) > 0 &&
                      alpaka::math::abs(acc, fin.otHits.psQOverPt(c) + H.k / kNominalPtPerRadius) >
                          fin.psCurvatureSigma * fin.psCurvatureResolution * 23.f / std::max(fin.hits.r(ih), 1.f))
                    return;
                }
              }
              const float rq = dq / wq, fk_d = rq * rq + c2 / sn;
              if (fin.chain.useFakeScore) {
                if (fk_d >= fkd || fin.hits.clusterCols(ih) < slo ||
                    (fin.hits.clusterCols(ih) > shi && !shapeExempt(fin, ih)))
                  return;
              }
              // a quad that min_pix or ps_seed drops (below) is not emitted; the OT confirmations then only decide
              // `found`, which an earlier hit may have set already: no confirmation search for it
              const bool dropped = minPixFail || psRejectQuad(acc, fin, c, p, ih, H.k, pte);
              if (dropped && found)
                return;
              if (fin.chain.ot2LayerId >= 0 && fin.chainPositions[p].layerId() == 4) {
                // OT2-P for the helix through b, c, d
                const EventLayer L2 = fin.layers[fin.chain.ot2LayerId];
                HelixF H3;
                H3.make(acc,
                        fin.hits.xBeam(ib),
                        fin.hits.yBeam(ib),
                        fin.hits.z(ib),
                        fin.hits.xBeam(ic),
                        fin.hits.yBeam(ic),
                        fin.hits.xBeam(ih),
                        fin.hits.yBeam(ih),
                        fin.hits.z(ih));
                if (fin.chain.ot2LocalWindowScale > 0 && fin.chainPositions[c.positions[0]].isPixel() &&
                    fin.chainPositions[c.positions[1]].isPixel() && fin.chainPositions[c.positions[2]].isPixel()) {
                  if (localWindowHit(acc, fin, fin.chain.ot2LayerId, H3, pte) == 0)
                    return;
                } else if (fin.chain.ot2LocalSigma > 0) {
                  // fk_ot2_lsig: the local chi2; a quad exempted by fk_ot2_gate passes without a hit (fk_ot2_lgate 0: no
                  // search) or needs one within fk_ot2_lgate sigma
                  const bool gated = c.score + fk_d < fin.chain.ot2ConfirmationGate;
                  const float sc = gated ? fin.chain.ot2LocalGateSigma : fin.chain.ot2LocalSigma;
                  float best;
                  const float sc2 = sc * sc;
                  if (sc > 0 &&
                      localChi2Hit(acc, fin, fin.chain.ot2LayerId, H3, pte, sc, fin.chain.ot2LocalSigmaY, best, sc2) ==
                          0 &&
                      !(best < sc2))
                    return;
                } else {
                  float zm = 0, dpb = 0, dzb = 0, scb = 0;
                  const float ip2 = 1.0f / std::max(0.9f, pte);
                  const float wp2 =
                      fin.chain.ot2WindowScale *
                      std::max(fin.chain.ot2PhiWindowA + fin.chain.ot2PhiWindowB * ip2, fin.chain.ot2PhiWindowMin);
                  const float wz2 = fin.chain.ot2WindowScale * (fin.chain.ot2QWindowA + fin.chain.ot2QWindowB * ip2);
                  const int kn = next_hit(acc,
                                          fin,
                                          L2,
                                          fin.layerGeometry[fin.chain.ot2LayerId],
                                          H3,
                                          pte,
                                          wp2,
                                          wz2,
                                          zm,
                                          dpb,
                                          dzb,
                                          scb,
                                          (fin.psMode & 2) && L2.firstOTHit() >= 0
                                              ? fin.otHits.psMatches().data() + L2.firstOTHit()
                                              : nullptr);
                  if (kn != -2 && alpaka::math::abs(acc, zm) < L2.qHi() - 2) {
                    const bool pass =
                        kn >= 0 && alpaka::math::abs(acc, dpb) < wp2 && alpaka::math::abs(acc, dzb) < wz2 &&
                        (fin.chain.ot2ConfirmationSigma <= 0 || c.score + fk_d < fin.chain.ot2ConfirmationGate ||
                         scb < fin.chain.ot2ConfirmationSigma * fin.chain.ot2ConfirmationSigma);
                    if (!pass)
                      return;
                  }
                }
              }
              {
                const int ld = fin.chainPositions[p].layerId();
                if (fin.chain.ot1LayerId >= 0 && (ld <= 3 || (ld >= 16 && ld <= 27) || (ld >= 38 && ld <= 49)) &&
                    alpaka::math::abs(acc, ccot) < fin.chain.ot1MaxAbsCot &&
                    c.score + fk_d >= fin.chain.ot1ConfirmationGate) {
                  // OT1-P for the helix through b, c, d of a quad of four pixel hits
                  const EventLayer L1 = fin.layers[fin.chain.ot1LayerId];
                  HelixF H3;
                  H3.make(acc,
                          fin.hits.xBeam(ib),
                          fin.hits.yBeam(ib),
                          fin.hits.z(ib),
                          fin.hits.xBeam(ic),
                          fin.hits.yBeam(ic),
                          fin.hits.xBeam(ih),
                          fin.hits.yBeam(ih),
                          fin.hits.z(ih));
                  if (fin.chain.ot1LocalSigma > 0) {
                    // fk_ot1_lsig: the local chi2
                    float best;
                    const float ot1s2 = fin.chain.ot1LocalSigma * fin.chain.ot1LocalSigma;
                    if (localChi2Hit(acc,
                                     fin,
                                     fin.chain.ot1LayerId,
                                     H3,
                                     pte,
                                     fin.chain.ot1LocalSigma,
                                     fin.chain.ot1LocalSigmaY,
                                     best,
                                     ot1s2) == 0 &&
                        !(best < ot1s2))
                      return;
                  } else {
                    float zm = 0, dpb = 0, dzb = 0, scb = 0;
                    const float pt1 = std::max(0.5f, pte);
                    const float wp1 = fin.chain.ot1WindowScale * (0.0005f + 3.2e-3f / pt1),
                                wz1 = fin.chain.ot1WindowScale * (0.075f + 0.0316f / pt1);
                    const int kn = next_hit(acc,
                                            fin,
                                            L1,
                                            fin.layerGeometry[fin.chain.ot1LayerId],
                                            H3,
                                            pte,
                                            wp1,
                                            wz1,
                                            zm,
                                            dpb,
                                            dzb,
                                            scb,
                                            (fin.psMode & 1) && L1.firstOTHit() >= 0
                                                ? fin.otHits.psMatches().data() + L1.firstOTHit()
                                                : nullptr);
                    if (kn != -2 && alpaka::math::abs(acc, zm) < L1.qHi() - 2) {
                      if (kn < 0 || scb >= fin.chain.ot1WindowScale * fin.chain.ot1WindowScale)
                        return;
                    }
                  }
                }
              }
              found = true;
              if (dropped)  // min_pix or ps_seed (found stays set)
                return;
              Quad o;
              o.layers[0] = fin.chainPositions[c.positions[0]].layerId(),
              o.layers[1] = fin.chainPositions[c.positions[1]].layerId(),
              o.layers[2] = fin.chainPositions[c.positions[2]].layerId();
              o.layers[3] = fin.chainPositions[p].layerId();
              o.hits[0] = fin.hits.clusterIndex(ia), o.hits[1] = fin.hits.clusterIndex(ib),
              o.hits[2] = fin.hits.clusterIndex(ic), o.hits[3] = fin.hits.clusterIndex(ih);
              // the cleaning score with the pattern's own windows
              const bool dc = fin.layerGeometry[fin.chainPositions[c.positions[2]].binnedLayerId()].isDisc();
              const float ua = dc ? fin.hits.z(ia) : fin.hits.r(ia), ub = dc ? fin.hits.z(ib) : fin.hits.r(ib),
                          uc = dc ? fin.hits.z(ic) : fin.hits.r(ic);
              const float qa = dc ? fin.hits.r(ia) : fin.hits.z(ia), qb = dc ? fin.hits.r(ib) : fin.hits.z(ib),
                          qcc = dc ? fin.hits.r(ic) : fin.hits.z(ic);
              const float rqc = (qcc - (qa + (qb - qa) * (uc - ua) / (ub - ua))) / w.qWindowC();
              const float wpc = w.phiWindowA() + w.phiWindowB() * ipt, wqc = w.qWindowA() + w.qWindowB() * ipt;
              const float dph2 = s2 * (1 + s2 * (1.0f / 3));
              const float sc = rqc * rqc + dph2 / (wpc * wpc) + (dq / wqc) * (dq / wqc);
              o.score = alpaka::math::isfinite(acc, sc) ? sc : 1e30f;
              o.fakeScore = c.score + fk_d;
              if constexpr (kSerialAcc<TAcc>)
                pend.quad(acc, fin, qs, o);
              else
                emit(acc, qs, o);
            };
            // the q pre-filter of the hit loop
            float xm, ym, qm = 0, sm = -1;
            const float um = 0.5f * (u0 + u1), ihh = 2.0f / (u1 - u0);
            bool pre;
            if (dp)
              pre = dp->pre, qm = dp->qm;
            else
              pre = ok0 && ok1 && H.at(acc, disc, um, xm, ym, qm, sm);
            const float qa1 = 0.5f * (q1 - q0), qa2 = 0.5f * (q0 + q1) - qm, wq_pre = 1.1f * wqd + 0.002f;
            auto hitd = [&](uint32_t i) {
              const uint32_t ih = targetLayer.firstHit() + i;
              const float hu = disc ? finder.hits.z(ih) : finder.hits.r(ih),
                          hq = disc ? finder.hits.r(ih) : finder.hits.z(ih);
              if (pre) {
                const float t = (hu - um) * ihh;
                if (alpaka::math::abs(acc, hq - (qm + t * (qa1 + t * qa2))) > wq_pre)
                  return;
              }
              float px = 0, py = 0, qp = 0, s3 = -1;
              const bool ok = H.at(acc, disc, hu, px, py, qp, s3);
              float wq = wqd, sp2 = sw2;
              if (isr > 0) {
                const float lv = s3 > 0 ? s3 * isr : 1.0f, wp = aphi + lv * bpi, sp = alpaka::math::sin(acc, wp);
                wq = aq + lv * bqi, sp2 = sp * sp;
              }
              const float hxx = finder.hits.xBeam(ih), hyy = finder.hits.yBeam(ih);
              const float cr = px * hyy - py * hxx, dt = px * hxx + py * hyy;
              const float n2 = (px * px + py * py) * (hxx * hxx + hyy * hyy);
              const float dq = hq - qp, c2 = cr * cr, sn = sp2 * n2;
              if (ok && alpaka::math::abs(acc, dq) <= wq && dt > 0 && c2 <= sn)
                take(ih, dq, wq, c2, sn, c2 / n2);
            };
            const BinRange pb = dp ? dp->pb : phiBinRangeAround(acc, Tg, dmid, dhalf + wpd + 1e-6f);
            if constexpr (kCpuAcc<TAcc>) {
              cpu::ScanPre scan(um, ihh, qm, qa1, qa2, wq_pre);
              forEachBinRun(finder, targetLayer, pb, qd, [&](uint32_t b, uint32_t e) {
                // the q pre-filter, loose and 8 hits at a time (MkSeederKernelsCpu.h); hitd() repeats it exactly
                if (pre) {
                  scan((disc ? finder.hits.z() : finder.hits.r()).data() + targetLayer.firstHit(),
                       (disc ? finder.hits.r() : finder.hits.z()).data() + targetLayer.firstHit(),
                       b,
                       e,
                       hitd);
                  return;
                }
                for (uint32_t i = b; i < e; ++i)
                  hitd(i);
              });
            } else if (pairItem) {
              // the pair's hit if it is in the window (in q bin qi: the hits from st[qi np] on; in its phi bins)
              const uint32_t i = pieces.piece(e), np = ::mkseeder::kNPhiBins;
              const uint32_t *st = finder.binStarts.start().data() + targetLayer.firstBinStart();
              if (qd.first < qd.last && i >= st[qd.first * np] && i < st[qd.last * np]) {
                uint32_t lo = qd.first, hi = qd.last - 1;
                while (lo < hi) {
                  const uint32_t mid = (lo + hi + 1) / 2;
                  if (st[mid * np] <= i)
                    lo = mid;
                  else
                    hi = mid - 1;
                }
                const uint32_t *sq = st + lo * np;
                const bool in =
                    pb.first < pb.last || pb.last == 0
                        ? i >= sq[pb.first] && i < sq[pb.last == 0 ? np : pb.last]
                        : pb.first > pb.last && ((i >= sq[pb.first] && i < sq[np]) || (i >= sq[0] && i < sq[pb.last]));
                if (in)
                  hitd(i);
              }
            } else {
              forEachBinRun(finder, targetLayer, pb, qd, [&](uint32_t b, uint32_t e) {
                for (uint32_t i = b; i < e; ++i)
                  hitd(i);
              });
            }
          }  // if (ok0 || ok1)
        }  // if (window tables and hit at the layer)
        if constexpr (kGpuAcc<TAcc>) {
          if (mode == kPairs) {
            // the pairs of a candidate (1) or its whole window (5: flag 4 kept for its pairs still to be skipped)
            if (found)
              candidateQueues.tripletStage.foundTriplet(j) = pairItem ? 1 : 5;
            continue;
          }
        }
        if (!found || chainParameters.alwaysForwardMisses) {
          const int h = c.holes + (c.stage == 2);
          if (h <= chainParameters.maxHoles) {
            int st;
            const int allow = h <= chainParameters.maxHolesForOuterTracker;
            const int nq = ml ? router.nextCrossedPositionFromMasks(p, L, allow, st)
                              : router.nextCrossedPosition(acc, finder, p, cz0, ccot, allow, st, [&] {
                                  return nextCrossedPosition(acc, finder, p, cz0, ccot, allow, st);
                                });
            if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0) {
              tripletSeed();
            } else {
              Candidate m = c;
              m.innerHoles += h > c.holes;
              m.holes = h, m.stage = st;
              if constexpr (kSerialAcc<TAcc>) {
                if (ml) {
                  cpu::pushCandidateWithLine(candidateQueues, q3(finder, nq), m, L);
                  continue;
                }
              }
              push(acc, candidateQueues, q3(finder, nq), m);
            }
          } else if (!found)
            tripletSeed();
        }
      }  // for (e : candidates or pieces)
      if constexpr (kSerialAcc<TAcc>)
        pend.flush(acc, finder, candidateQueues);
    }  // KernelExtendToQuads::operator()
  };  // class KernelExtendToQuads

  // a GPU, stage d at a position where KernelExtendToQuads::splitAt(): per candidate its window (the prologue's
  // expressions, one bin wider on each side) and the hits that pass looser versions of hitd()'s tests, as (candidate,
  // hit) pairs onto the pair list for KernelExtendToQuads kPairs, which repeats every test exactly (a candidate whose
  // pairs do not fit: flag 4, its whole window scanned by kPairs). A light kernel (71 registers; no take(), no forwarding): many warps
  // in flight for the hits' loads, where KernelExtendToQuads (255 registers) waited for them.
  class KernelScanQuadWindows {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p, WorkPieces pairs) const {
      if constexpr (kGpuAcc<TAcc>) {
        const MkSeederChainParameters &chainParameters = finder.chain;
        const int n = chainParameters.nPositions;
        const EventLayer targetLayer = finder.layers[finder.chainPositions[p].binnedLayerId()];
        const LayerAxes Tg = finder.layerGeometry[finder.chainPositions[p].binnedLayerId()];
        const bool disc = Tg.isDisc();
        const float u0 = targetLayer.qbarLo(), u1 = targetLayer.qbarHi();
        const uint32_t np = ::mkseeder::kNPhiBins, m = np - 1;
        const float *hus = (disc ? finder.hits.z() : finder.hits.r()).data() + targetLayer.firstHit(),
                    *hqs = (disc ? finder.hits.r() : finder.hits.z()).data() + targetLayer.firstHit();
        const uint32_t nc = std::min(candidateQueues.queues.count(q3(finder, p)),
                                     candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
        for (uint32_t j : cms::alpakatools::uniform_elements(acc, nc)) {
          Candidate c;
          if (!readCandidate(candidateQueues, q3(finder, p), j, c))
            continue;
          const EventLayer La = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()],
                           Lb = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()],
                           Lc = finder.layers[finder.chainPositions[c.positions[2]].binnedLayerId()];
          const uint32_t ia = La.firstHit() + c.hits[0], ib = Lb.firstHit() + c.hits[1], ic = Lc.firstHit() + c.hits[2];
          HelixF H;
          H.make(acc,
                 finder.hits.xBeam(ia),
                 finder.hits.yBeam(ia),
                 finder.hits.z(ia),
                 finder.hits.xBeam(ib),
                 finder.hits.yBeam(ib),
                 finder.hits.xBeam(ic),
                 finder.hits.yBeam(ic),
                 finder.hits.z(ic));
          const int ix = finder.windowTableByQuad[((c.positions[0] * n + c.positions[1]) * n + c.positions[2]) * n + p]
                             .windowTable();
          const bool inner_ok =
              !chainParameters.innerOuterTrackerOnly || !c.innerHoles || !finder.chainPositions[p].isPixel();
          if (!(searchesCombination(chainParameters, ix) && H.ok && inner_ok))
            continue;
          const auto w = finder.windowTables[ix >= 0 ? ix : 0];
          float aphi = w.phiWindowA(), bphi = w.phiWindowB(), aq = w.qWindowA(), bq = w.qWindowB();
          if (w.nEtaWindows() > 0) {
            const float ae = alpaka::math::asinh(acc, alpaka::math::abs(acc, H.cot));
            for (int i = 0; i < w.nEtaWindows(); ++i) {
              const auto E = finder.etaWindows[w.firstEtaWindow() + i];
              if (ae >= E.absEtaLow() && ae < E.absEtaHigh()) {
                aphi = E.phiWindowA(), bphi = E.phiWindowB(), aq = E.qWindowA(), bq = E.qWindowB();
                break;
              }
            }
          }
          if (chainParameters.mergedWindowScale > 0.f &&
              (::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ia)) ||
               ::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ib)) ||
               ::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ic)))) {
            const float sc = chainParameters.mergedWindowScale;
            aphi *= sc, bphi *= sc, aq *= sc, bq *= sc;
          }
          float x0, y0, q0, s0 = -1, x1, y1, q1, s1 = -1;
          const bool ok0 = H.at(acc, disc, u0, x0, y0, q0, s0), ok1 = H.at(acc, disc, u1, x1, y1, q1, s1);
          if (!(ok0 || ok1))
            continue;
          if (!ok0)
            x0 = x1, y0 = y1, q0 = q1;
          if (!ok1)
            x1 = x0, y1 = y0, q1 = q0;
          const float p0 = libm::atan2(acc, y0, x0), p1 = libm::atan2(acc, y1, x1);
          const float pte = H.pt(acc), ipt = 1.0f / std::max(chainParameters.ptMin, pte);
          const float isr = w.referencePathLength() > 0 ? 1.0f / w.referencePathLength() : 0.0f;
          const float smax = std::max(s0, s1), lmax = isr > 0 && smax > 0 ? smax * isr : 1.0f;
          const float wpd = aphi + lmax * bphi * ipt, wqd = aq + lmax * bq * ipt;
          BinRange qd = qBinRangePadded(acc, Tg, std::min(q0, q1) - wqd, std::max(q0, q1) + wqd);
          const float dpp = wrapPhiFloor(acc, p1 - p0), dmid = p0 + 0.5f * dpp,
                      dhalf = 0.5f * alpaka::math::abs(acc, dpp);
          BinRange pb = phiBinRangeAround(acc, Tg, dmid, dhalf + wpd + 1e-6f);
          // one bin wider on each side (a superset of KernelExtendToQuads's window whatever the rounding)
          qd.first = qd.first > 0 ? qd.first - 1 : 0u;
          qd.last = std::min(qd.last + 1, uint32_t(Tg.qLastBin()) + 1u);
          if (((pb.last - pb.first) & m) + 3 >= np)
            pb = {0u, 0u};
          else
            pb = {(pb.first + m) & m, (pb.last + 1) & m};
          // the q pre-filter, looser
          float xm, ym, qm = 0, sm = -1;
          const float um = 0.5f * (u0 + u1), ihh = 2.0f / (u1 - u0);
          const bool pre = ok0 && ok1 && H.at(acc, disc, um, xm, ym, qm, sm);
          const float qa1 = 0.5f * (q1 - q0), qa2 = 0.5f * (q0 + q1) - qm,
                      wq_loose = (1.1f * wqd + 0.002f) * 1.001f + 1e-3f;
          const float bqi = bq * ipt, bpi = bphi * ipt, sw = alpaka::math::sin(acc, wpd), sw2 = sw * sw;
          bool full = false;
          forEachBinRun(finder, targetLayer, pb, qd, [&](uint32_t b, uint32_t e) {
            for (uint32_t i = b; i < e && !full; ++i) {
              const float hu = hus[i], hq = hqs[i], t = (hu - um) * ihh;
              if (pre && !(alpaka::math::abs(acc, hq - (qm + t * (qa1 + t * qa2))) <= wq_loose))
                continue;
              // the hit's window tests (KernelExtendToQuads's hitd), looser; kept if the helix misses (hitd decides)
              float px = 0, py = 0, qp = 0, s3 = -1;
              if (H.at(acc, disc, hu, px, py, qp, s3)) {
                float wq = wqd, sp2 = sw2;
                if (isr > 0) {
                  const float lv = s3 > 0 ? s3 * isr : 1.0f, sp = alpaka::math::sin(acc, aphi + lv * bpi);
                  wq = aq + lv * bqi, sp2 = sp * sp;
                }
                const uint32_t ih = targetLayer.firstHit() + i;
                const float hxx = finder.hits.xBeam(ih), hyy = finder.hits.yBeam(ih);
                const float cr = px * hyy - py * hxx, dt = px * hxx + py * hyy;
                const float n2 = (px * px + py * py) * (hxx * hxx + hyy * hyy);
                if (!(alpaka::math::abs(acc, hq - qp) <= wq * 1.001f + 1e-3f &&
                      dt > -1e-3f * alpaka::math::sqrt(acc, n2) && cr * cr <= sp2 * n2 * 1.002f + 1e-6f * n2))
                  continue;
              }
              const uint32_t k = alpaka::atomicAdd(acc, pairs.data, 1u, alpaka::hierarchy::Blocks{});
              if (k < pairs.capacity)
                pairs.item(k) = j, pairs.piece(k) = i;
              else
                full = true;
            }
          });
          if (full)
            candidateQueues.tripletStage.foundTriplet(j) = 4;
        }
      }
    }
  };

  // a GPU, stage d at a position where KernelExtendToQuads::splitAt(), after its kPairs: KernelExtendToQuads's forwarding
  // (a candidate with no quad, or any with alwaysForwardMisses, to its next crossed position, else its triplet seed), a
  // thread per candidate, the pushes per block (blockPush). A light kernel: KernelExtendToQuads (255 registers) kept few
  // warps in flight for these loads. Clears the found flags and the pair list. The same steps as KernelExtendToQuads's forwarding: keep them
  // equal.
  class KernelForwardQuadCandidates {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p, WorkPieces pairs) const {
      if constexpr (kGpuAcc<TAcc>) {
        const MkSeederChainParameters &chainParameters = finder.chain;
        const uint32_t nc = std::min(candidateQueues.queues.count(q3(finder, p)),
                                     candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
        const uint32_t t = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        const uint32_t nth = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
        const uint32_t nb = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
        for (uint32_t j0 = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] * nth; j0 < nc; j0 += nb * nth) {
          const uint32_t j = j0 + t;
          Candidate m;
          int nq = -1;
          if (j < nc) {
            const bool found = candidateQueues.tripletStage.foundTriplet(j) & 1;
            candidateQueues.tripletStage.foundTriplet(j) = 0;
            Candidate c;
            if ((!found || chainParameters.alwaysForwardMisses) &&
                readCandidate(candidateQueues, q3(finder, p), j, c)) {
              float cz0, ccot;
              line(acc, finder, c, cz0, ccot);
              auto tripletSeed = [&]() {
                if (!tripletSeedGates(acc, finder, c, ccot))
                  return;
                const uint32_t ia =
                    finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()].firstHit() + c.hits[0];
                const uint32_t ib =
                    finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()].firstHit() + c.hits[1];
                const uint32_t ic =
                    finder.layers[finder.chainPositions[c.positions[2]].binnedLayerId()].firstHit() + c.hits[2];
                emitTripletSeed(acc, finder, candidateQueues, c, ia, ib, ic);
              };
              const int h = c.holes + (c.stage == 2);
              if (h <= chainParameters.maxHoles) {
                int st;
                const int allow = h <= chainParameters.maxHolesForOuterTracker;
                nq = nextCrossedPosition(acc, finder, p, cz0, ccot, allow, st);
                if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0) {
                  nq = -1;
                  tripletSeed();
                } else {
                  m = c;
                  m.innerHoles += h > c.holes;
                  m.holes = h, m.stage = st;
                }
              } else if (!found)
                tripletSeed();
            }
          }
          blockPush(acc, candidateQueues, chainParameters.nPositions, q3(finder, 0), nq, m);
        }
        // the pair list empty for the next split position (after its last reader, kPairs)
        if (cms::alpakatools::once_per_grid(acc))
          pairs.data[0] = 0;
      }
    }
  };

  // ---- the chunks of the queues of position p (consumed) back to the free list
  class KernelReleaseConsumedChunks {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p) const {
      for (int k = 0; k < 2; ++k) {
        const int queue = k ? q3(finder, p) : q2(finder, p);
        const uint32_t nch = std::min(
            (candidateQueues.queues.count(queue) + Queues::kCandidateChunkSize - 1) / Queues::kCandidateChunkSize,
            candidateQueues.maxChunksPerQueue);
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, nch)) {
          const int cid = candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + i);
          if (cid >= 0)
            candidateQueues.freeChunks.chunk(
                alpaka::atomicAdd(acc, &candidateQueues.counters->nFreeChunks, 1u, alpaka::hierarchy::Blocks{})) = cid;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
