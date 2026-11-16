#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederPSMark_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederPSMark_h

// The PS-module confirmation ("mark") of the OT1-3 P-side hits (mkFit layers 4, 6, 8: the lower sensors of the PS
// modules of the barrel layers 1-3) by the S-side hits (mkFit layers 5, 7, 9) of the same module: a P hit is
// confirmed when an S hit of the partner sensor lies, after the parallax correction for a straight line from the beam
// spot (as the VectorHit builder), within a local-x window that holds the bend of a prompt track of pT >= ps_pt, and
// within the S strip's length along the strip. Per P hit: the number of confirming S hits (capped at 255) and the
// q/pT estimate of the best one (smallest corrected local-x residual) from its bend: the hit pair's local direction
// in the module's transverse plane relative to the radial line, tan(alpha) = -q r k / pT with k = 0.0015 Bz (Bz in T).
// The module pairs and frames (buildPSMarkGeometry(), host) are static tables (MkSeederModulesSoA psPartner,
// psFrame); the device marks the P hits in the hits' binning (markPSHit()).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <alpaka/alpaka.hpp>

namespace mkfit {
  class TrackerInfo;
}
class TrackerTopology;

namespace mkseeder {

  // the mark's settings (MkSeederConfig, JSON keys at the members)
  struct PSMarkParameters {
    int mode = 0;                // ps_mark: 0 off, 1 compute the mark
    float ptMin = 0.8f;          // ps_pt, GeV: the bend window's pT (with ps_d0: the window's slope)
    float d0Max = 0.0f;          // ps_d0, cm: the bend window also holds a line missing the beam line by this
    float residualPad = 0.015f;  // ps_w0, cm: added to the local-x window (resolution)
    float halfLength = 1.25f;    // ps_hl, cm: half the S strip length (the along-strip test)
    float lengthPad = 0.2f;      // ps_dv, cm: added to the along-strip window
    // v2: the per-range counts (psCounts) take the S hits within the bend window of pT >= countPtMin (same pads)
    float countPtMin = 0.5f;  // ps_cnt_pt, GeV
    int mergedExcess = 2;     // ps_mrg_n: an S cluster is "merged" when wider than the P cluster by >= this many strips
    // > 0: the low-pT counts (psCountsLow) take the S hits within the bend window of pT >= lowPtMin (same pads)
    float lowPtMin = 0.f;  // ps_low_pt, GeV
  };

  // v2 (psCounts, uint16): per P hit the number of confirming S hits per q/pT range of each pair's bend estimate,
  // 2 bits per range (0, 1, 2, 3 = 3 or more): range 0 = pT > 3 GeV (either charge), 1/2/3 = q > 0 with pT 1.5-3,
  // 0.9-1.5, < 0.9 (down to ps_cnt_pt), 4/5/6 = q < 0 with the same pT ranges; bit 14: the best (the mark's) S hit is
  // merged (its cluster wider than the P cluster by >= ps_mrg_n strips), bit 15: some S hit in the count window is
  // merged. The ranges are of the estimate, whose resolution is ~0.2 / 0.14 / 0.11 1/GeV on OT1 / 2 / 3.
  namespace psCounts {
    constexpr int kNRanges = 7;
    constexpr uint16_t kBestMerged = 1u << 14, kAnyMerged = 1u << 15;
    // the range of a q/pT estimate (1/GeV)
    constexpr int range(float qOverPt) {
      const float a = qOverPt < 0 ? -qOverPt : qOverPt;
      if (a < 1.f / 3.f)
        return 0;
      const int k = a < 1.f / 1.5f ? 1 : a < 1.f / 0.9f ? 2 : 3;
      return qOverPt > 0 ? k : k + 3;
    }
    constexpr int count(uint16_t w, int r) { return (w >> (2 * r)) & 3; }
    constexpr uint16_t add(uint16_t w, int r) { return count(w, r) < 3 ? uint16_t(w + (1u << (2 * r))) : w; }
    // the counts summed over all ranges (each capped at 3)
    constexpr int total(uint16_t w) {
      int n = 0;
      for (int r = 0; r < kNRanges; ++r)
        n += count(w, r);
      return n;
    }
  }  // namespace psCounts

  // ps_low_pt > 0 (psCountsLow, uint16): the confirming S hits per low-pT range of each pair's bend estimate, 2 bits
  // per range: 0/1/2 = q > 0 with pT 0.6-0.9, 0.4-0.6, 0.3-0.4 (down to ps_low_pt), 3/4/5 = q < 0 with the same pT
  namespace psCountsLow {
    constexpr int kNRanges = 6;
    // the range of a q/pT estimate (1/GeV); -1: outside the low ranges
    constexpr int range(float qOverPt) {
      const float a = qOverPt < 0 ? -qOverPt : qOverPt;
      if (a < 1.f / 0.9f || a >= 1.f / 0.3f)
        return -1;
      const int k = a < 1.f / 0.6f ? 0 : a < 1.f / 0.4f ? 1 : 2;
      return qOverPt > 0 ? k : k + 3;
    }
  }  // namespace psCountsLow

  // The confirming S hits of a P hit in the q/pT range of a hypothesis qOverPt (1/GeV; the ranges per charge: pT > 3
  // (either charge), 1.5-3, 0.9-1.5, then with the low counts (low) 0.6-0.9, 0.4-0.6, 0.3-0.4, without them < 0.9
  // (psCounts range 3 / 6)); adjacent: also the neighbouring ranges of the same charge (both charges next to pT > 3);
  // -1: the hypothesis is below the ranges (pT < 0.3 with the low counts, never without)
  constexpr int psCountInRange(uint16_t counts, uint16_t countsLow, bool low, float qOverPt, bool adjacent) {
    const float a = qOverPt < 0 ? -qOverPt : qOverPt;
    const int u = a < 1.f / 3.f    ? 0
                  : a < 1.f / 1.5f ? 1
                  : a < 1.f / 0.9f ? 2
                  : !low           ? 3
                  : a < 1.f / 0.6f ? 3
                  : a < 1.f / 0.4f ? 4
                  : a < 1.f / 0.3f ? 5
                                   : 6;
    if (u == 6)
      return -1;
    const int umax = low ? 5 : 3;
    auto cnt = [&](int v, bool pos) -> int {
      if (v == 0)
        return psCounts::count(counts, 0);
      if (v <= 2 || !low)
        return psCounts::count(counts, pos ? v : v + 3);
      return psCounts::count(countsLow, pos ? v - 3 : v);
    };
    const bool pos = qOverPt > 0;
    int n = cnt(u, pos);
    if (adjacent) {
      if (u == 0)
        n += cnt(1, true) + cnt(1, false);
      else {
        n += u == 1 ? cnt(0, pos) : cnt(u - 1, pos);
        if (u < umax)
          n += cnt(u + 1, pos);
      }
    }
    return n;
  }

  // per P layer (OT1-3 P): per module (Hit::detIDinLayer()) its partner S module in the S layer (-1: none) and the P
  // module's frame (u: local x, the measurement direction, v: along the strip, n: the normal) and the S sensor's
  // signed distance along n
  struct PSMarkLayer {
    int pLayer = -1, sLayer = -1;
    std::vector<int> partner;
    std::vector<std::array<float, 9>> frame;  // ux uy uz vx vy vz nx ny nz
    std::vector<float> separation;
  };
  struct PSMarkGeometry {
    std::array<PSMarkLayer, 3> layers;
    int nPartnered = 0;  // P modules with an S partner
  };
  // the geometry from mkFit's TrackerInfo (the modules' planes, detids) and the tracker topology (partnerDetId); the
  // S module index is TrackerInfo's short id of the partner's detid in the S layer
  PSMarkGeometry buildPSMarkGeometry(const mkfit::TrackerInfo &ti, const TrackerTopology &topo);

  // The mark of one P hit at (px, py, pz) with cluster width pSize (Hit::spanRows()) on a module with the frame f (u, v,
  // n as PSMarkLayer::frame) against the S hits q in [qb, qe) of its partner module, in their cluster order (sHit(q, x,
  // y, z, size) gives one): an S hit confirms when its residual to the straight line from the beam spot, in local x
  // (resU) and along the strip (resV), is within the bend window of pT >= ps_pt (+ ps_d0, ps_w0) and the strip's
  // length (ps_hl + ps_dv). Out: the confirming S hits (capped at 255), the q/pT estimates of the best (smallest
  // |resU|, the first on ties) and of the second best (0: fewer than 2), the per-range counts (psCounts). The
  // estimate: the pair's transverse angle to the radial line, sin(alpha) = -q r k / pT, k = 0.0015 Bz (Bz in T).
  // Host and device: the device marks the hits in the binning (MkSeederBinKernels.h).
  struct PSMarkOut {
    uint8_t nMatches = 0;
    uint16_t counts = 0, countsLow = 0;
    float qOverPt = 0.f, qOverPt2 = 0.f;
  };
  template <typename SHit>
  ALPAKA_FN_HOST_ACC inline PSMarkOut markPSHit(const float *f,
                                                const PSMarkParameters &par,
                                                float beamX,
                                                float beamY,
                                                float beamZ,
                                                float bz,
                                                float px,
                                                float py,
                                                float pz,
                                                int pSize,
                                                uint32_t qb,
                                                uint32_t qe,
                                                const SHit &sHit) {
    PSMarkOut o;
    const float k = 0.5f * 0.00299792458f * bz;  // sin(alpha) = k r / pT (r in cm, pT in GeV): r / (2 R)
    // the line from the beam spot
    const float dx = px - beamX, dy = py - beamY, dz = pz - beamZ;
    const float dn = dx * f[6] + dy * f[7] + dz * f[8];
    if (dn <= 1e-3f)
      return o;
    const float du = dx * f[0] + dy * f[1] + dz * f[2], dv = dx * f[3] + dy * f[4] + dz * f[5];
    const float r = std::sqrt(dx * dx + dy * dy);
    // sin(alpha) from the corrected residual over the separation: the transverse angle of a line along d rotated
    // by alpha in the transverse plane, (t.u) / (t.n) ~ sin(alpha) |d_xy| / (d.n)
    const float toSin = dn / r;
    const float sinMax = std::min(0.95f, r * k / par.ptMin + par.d0Max / r);
    const float sinMaxCnt = std::min(0.95f, r * k / std::min(par.ptMin, par.countPtMin) + par.d0Max / r);
    const float qptScale = -toSin / (r * k);  // q/pT of a pair = resU / en x qptScale
    const float sinMaxLow = par.lowPtMin > 0.f ? std::min(0.95f, r * k / par.lowPtMin + par.d0Max / r) : 0.f;
    float bestAbs = 1e30f, bestAbs2 = 1e30f, sin1 = 0.f, sin2 = 0.f;
    for (uint32_t q = qb; q < qe; ++q) {
      float sx, sy, sz;
      int sSize;
      sHit(q, sx, sy, sz, sSize);
      const float ex = sx - px, ey = sy - py, ez = sz - pz;
      const float en = ex * f[6] + ey * f[7] + ez * f[8];
      if (std::abs(en) < 1e-3f)
        continue;
      const float eu = ex * f[0] + ey * f[1] + ez * f[2], ev = ex * f[3] + ey * f[4] + ez * f[5];
      const float resU = eu - en * du / dn, resV = ev - en * dv / dn;
      if (std::abs(resV) > par.halfLength + par.lengthPad)
        continue;
      // the low-pT counts (ps_low_pt) over their own (widest) window
      if (par.lowPtMin > 0.f && std::abs(resU) <= std::abs(en) * sinMaxLow / toSin + par.residualPad) {
        const int rl = psCountsLow::range(resU / en * qptScale);
        if (rl >= 0)
          o.countsLow = psCounts::add(o.countsLow, rl);
      }
      const float wc = std::abs(en) * sinMaxCnt / toSin + par.residualPad;
      if (std::abs(resU) > wc)
        continue;
      // v2: the per-range counts and the merged flag over the (wider) count window
      const bool merged = sSize - pSize >= par.mergedExcess;
      o.counts = psCounts::add(o.counts, psCounts::range(resU / en * qptScale));
      if (merged)
        o.counts |= psCounts::kAnyMerged;
      const float w = std::abs(en) * sinMax / toSin + par.residualPad;
      if (std::abs(resU) > w)
        continue;
      if (o.nMatches < 255)
        ++o.nMatches;
      if (std::abs(resU) < bestAbs) {
        bestAbs2 = bestAbs, sin2 = sin1;
        bestAbs = std::abs(resU);
        sin1 = resU / en * toSin;
        o.counts = merged ? (o.counts | psCounts::kBestMerged) : uint16_t(o.counts & ~psCounts::kBestMerged);
      } else if (std::abs(resU) < bestAbs2) {
        bestAbs2 = std::abs(resU), sin2 = resU / en * toSin;
      }
    }
    if (o.nMatches >= 2)
      o.qOverPt2 = -sin2 / (r * k);
    if (o.nMatches > 0)
      o.qOverPt = -sin1 / (r * k);
    return o;
  }

}  // namespace mkseeder

#endif
