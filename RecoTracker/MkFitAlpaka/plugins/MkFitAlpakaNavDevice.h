#ifndef RecoTracker_MkFitAlpaka_plugins_MkFitAlpakaNavDevice_h
#define RecoTracker_MkFitAlpaka_plugins_MkFitAlpakaNavDevice_h
// Steps (a) + (b) of the device missing-hit navigation:
// SimpleNavigationSchool's compatibleLayers(layer, fts, dir) as a per-track device function.
//   (a) a table per DetLayer: the candidate lists of SimpleBarrel/ForwardNavigableLayer::nextLayers, rebuilt on the
//       host from the school's public static lists + TkLayerLess, as layer indices;
//   (b) the branch bits of nextLayers, the stop rule of SimpleNavigableLayer::wellInside (thickness-extended bounds,
//       crossing-side test) with a double helix, field z at the start point and AnalyticalPropagator's maxDPhi 1.6;
//   the closure as compatibleLayers (<= 150 rounds) on a 64-bit layer set.
#include <cmath>
#include <cstdint>

#include <alpaka/core/Common.hpp>

#include "RecoTracker/MkFitAlpaka/interface/math/HelixCrossings.h"  // MKFITDEV_NAV_INLINE

namespace mkfitdev::nav {
  constexpr int kMaxLayers = 64;
  constexpr int kMaxIdx = 4096;
  // barrel lists
  enum { kNegOuter, kPosOuter, kNegInner, kPosInner, kIB, kIL, kIR, kOB, kOL, kOR, kNLists };
  // forward lists (same slots): sOut, sIn, inner forward, outer barrel, inner barrel, outer forward
  enum { kFOut = 0, kFIn = 1, kFIF = 2, kFOB = 3, kFIB = 4, kFOF = 5 };
  struct Layer {
    int barrel;
    float rz;  // barrel: radius; disk: z
    float halfLength, thickness, rin, rout;
    int off[kNLists], len[kNLists];
  };
  struct Table {
    int nLayers;
    int nIdx;
    Layer layers[kMaxLayers];
    int idx[kMaxIdx];
  };
  // guarded sets. A set is MARGINAL (the host asks the navigation school instead) when a decision taken while
  // building it is within these margins of its threshold, so that the device start state (equal to the host's up to
  // float rounding, ~1e-7 relative) or the device helix could decide it differently from the host propagator.
  // Positions (crossing z / r vs the bounds +- the thickness term, the helix reach, the crossing path length, the
  // crossing-side products, |z| of the start): 20 um = 5x the largest helix vs AnalyticalPropagator crossing distance
  // measured on real tracks.
  constexpr double kSetMarginPos = 20e-4;
  // relative (the direction signs x px + y py and pz, the maxDPhi limit): the det-test margin (1e-4, ~1000x
  // the float rounding of the start state).
  constexpr double kSetMarginRel = 1e-4;
  struct TrackIn {  // the error-free state at the first hit + the start layers
    float x, y, z, px, py, pz;
    int charge;
    float bz;      // Tesla, at (x, y, z)
    int start[2];  // [0] innermost-hit layer (oppositeToMomentum), [1] outermost-hit layer (alongMomentum)
  };

  // crossing of the cylinder r = RZ (barrel) or the plane z = RZ (disk); false if not reachable within maxDPhi
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool helixTo(
      TrackIn const& t, bool barrel, double RZ, bool along, double out[6], bool& marg) {
    const double pt = sqrt(double(t.px) * t.px + double(t.py) * t.py);
    const double kappa = -2.99792458e-3 * t.charge * t.bz / pt;
    const double phi0 = atan2(double(t.py), double(t.px));
    const double sgn = along ? 1. : -1.;
    double s = 0;
    if (!barrel) {
      if (t.pz == 0)
        return false;
      s = (RZ - t.z) * pt / t.pz;
      marg |= fabs(RZ - t.z) <= kSetMarginPos;
      if (s * sgn <= 0)
        return false;
    } else {
      const double xc = t.x - sin(phi0) / kappa, yc = t.y + cos(phi0) / kappa;
      const double rho = 1. / fabs(kappa), d = sqrt(xc * xc + yc * yc);
      marg |= fabs(d - (RZ + rho)) <= kSetMarginPos || fabs(d - fabs(RZ - rho)) <= kSetMarginPos;
      if (d > RZ + rho || d < fabs(RZ - rho))
        return false;
      const double a = (RZ * RZ - rho * rho + d * d) / (2 * d);
      const double h = sqrt(fmax(0., RZ * RZ - a * a));
      const double ux = xc / d, uy = yc / d;
      const double period = 2 * M_PI / fabs(kappa);
      double best = 1e30;
      for (int k = 0; k < 2; ++k) {
        const double hh = k ? -h : h;
        const double px = a * ux - hh * uy, py = a * uy + hh * ux;
        const double a0 = atan2(t.y - yc, t.x - xc), a1 = atan2(py - yc, px - xc);
        double sk = (a1 - a0) / kappa;
        while (sk * sgn <= 0)
          sk += sgn * period;
        while (sk * sgn > period)
          sk -= sgn * period;
        marg |= fabs(sk) <= kSetMarginPos || fabs(sk) >= period - kSetMarginPos;
        if (fabs(sk) < fabs(best))
          best = sk;
      }
      if (best == 1e30)
        return false;
      s = best;
    }
    marg |= fabs(fabs(kappa * s) - 1.6) <= kSetMarginRel * 1.6;
    if (fabs(kappa * s) > 1.6)
      return false;
    const double phi = phi0 + kappa * s;
    out[0] = t.x + (sin(phi) - sin(phi0)) / kappa;
    out[1] = t.y - (cos(phi) - cos(phi0)) / kappa;
    out[2] = t.z + s * t.pz / pt;
    out[3] = cos(phi);
    out[4] = sin(phi);
    out[5] = t.pz / pt;
    return true;
  }

  // SimpleNavigableLayer::wellInside for one candidate: pushed / well inside
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void decide(
      Layer const& L, TrackIn const& t, bool along, bool& pushed, bool& well, bool& marg) {
    pushed = well = false;
    double b[6];
    if (!helixTo(t, L.barrel != 0, L.rz, along, b, marg))
      return;
    {  // |x . b| <= margin |x|, squared (no square root: registers)
      const double rT2 = double(t.x) * t.x + double(t.y) * t.y, r2 = rT2 + double(t.z) * t.z;
      const double dT = t.x * b[0] + t.y * b[1], d = dT + t.z * b[2];
      constexpr double m2 = kSetMarginPos * kSetMarginPos;
      marg |= dT * dT <= m2 * rT2 || d * d <= m2 * r2;
    }
    if ((t.x * b[0] + t.y * b[1]) < 0 || (t.x * b[0] + t.y * b[1] + t.z * b[2]) < 0)
      return;
    const double tperp = sqrt(b[3] * b[3] + b[4] * b[4]);
    if (L.barrel) {
      const float deltaZ = 0.5f * L.thickness * fabs(b[5]) / tperp;
      const double az = fabs(b[2]);
      pushed = az < L.halfLength + deltaZ;
      well = az < L.halfLength - deltaZ;
      marg |=
          fabs(az - (L.halfLength + deltaZ)) <= kSetMarginPos || fabs(az - (L.halfLength - deltaZ)) <= kSetMarginPos;
    } else {
      const float rpos = sqrt(b[0] * b[0] + b[1] * b[1]);
      const float deltaR = 0.5f * L.thickness * tperp / fabs(b[5]);
      pushed = L.rin - deltaR < rpos && rpos < L.rout + deltaR;
      well = L.rin + deltaR < rpos && rpos < L.rout - deltaR;
      marg |= fabs(rpos - (L.rin - deltaR)) <= kSetMarginPos || fabs(rpos - (L.rout + deltaR)) <= kSetMarginPos ||
              fabs(rpos - (L.rin + deltaR)) <= kSetMarginPos || fabs(rpos - (L.rout - deltaR)) <= kSetMarginPos;
    }
  }

  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool wellInsideList(
      Table const& T, Layer const& from, int list, TrackIn const& t, bool along, uint64_t& result, bool& marg) {
    for (int i = 0; i < from.len[list]; ++i) {
      const int c = T.idx[from.off[list] + i];
      bool pushed, well;
      decide(T.layers[c], t, along, pushed, well, marg);
      if (pushed)
        result |= (uint64_t(1) << c);
      if (well)
        return true;
    }
    return false;
  }

  // SimpleBarrel/ForwardNavigableLayer::nextLayers(fts, dir) as a layer set
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE uint64_t
  nextLayers(Table const& T, int l, TrackIn const& t, bool along, bool& marg) {
    Layer const& L = T.layers[l];
    const bool inOutB = t.x * t.px + t.y * t.py > 0;
    const bool inOutF = t.pz * t.z > 0;
    const bool xb = (along && inOutB) || (!along && !inOutB);
    const bool xf = (along && inOutF) || (!along && !inOutF);
    // the candidate lists of the branch, in the host order, searched at ONE call site (everything is inlined
    // into the kernel on the device; an out-of-line copy in two device TUs breaks the sm_100 / sm_120 device link)
    int lists[3], nLists = 0;
    if (L.barrel) {
      const bool signZ = ((t.pz > 0) && !along) || (!(t.pz > 0) && along);
      if (xb && xf)
        lists[nLists++] = signZ ? kNegOuter : kPosOuter;
      else if (!xb && !xf)
        lists[nLists++] = signZ ? kPosInner : kNegInner;
      else if (!xb && xf) {
        lists[nLists++] = kIB;
        lists[nLists++] = signZ ? kIL : kIR;
        lists[nLists++] = signZ ? kOL : kOR;
      } else {
        lists[nLists++] = signZ ? kIL : kIR;
        lists[nLists++] = kOB;
      }
    } else {
      if (xf && xb)
        lists[nLists++] = kFOut;
      else if (!xf && !xb)
        lists[nLists++] = kFIn;
      else if (!xf && xb) {
        lists[nLists++] = kFIF;
        lists[nLists++] = kFOB;
      } else {
        lists[nLists++] = kFIB;
        lists[nLists++] = kFOF;
      }
    }
    uint64_t r = 0;
    for (int i = 0; i < nLists; ++i)
      wellInsideList(T, L, lists[i], t, along, r, marg);
    return r;
  }

  // compatibleLayers: the closure of nextLayers (SimpleNavigableLayer::compatibleLayers, <= 150 rounds)
  // marg: set when a decision was within the kSetMargin* of its threshold, or the 150-round limit was reached (the host
  // search then returns an empty list)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE uint64_t
  compatibleLayers(Table const& T, int start, TrackIn const& t, bool along, bool& marg) {
    // one nextLayers call site: the first pass (i = -1) is the host call on the start layer (someLayers), the rounds
    // follow as on the host (counter, the collected set)
    // the direction signs of nextLayers (the same for every layer of the closure)
    marg |= fabs(double(t.x) * t.px + double(t.y) * t.py) <= kSetMarginRel * (fabs(t.x * t.px) + fabs(t.y * t.py)) ||
            fabs(t.z) <= kSetMarginPos ||
            double(t.pz) * t.pz <=
                kSetMarginRel * kSetMarginRel * (double(t.px) * t.px + double(t.py) * t.py + double(t.pz) * t.pz);
    uint64_t collect = 0, toTry = 0;
    int counter = 0;
    bool first = true;
    while (first || (toTry != 0 && (counter++) <= 150)) {
      uint64_t next = 0;
      for (int i = first ? -1 : 0; i < (first ? 0 : T.nLayers); ++i) {
        if (i >= 0) {
          const uint64_t bit = uint64_t(1) << i;
          if (!(toTry & bit) || (collect & bit))
            continue;
          collect |= bit;
        }
        next |= nextLayers(T, i < 0 ? start : i, t, along, marg);
      }
      toTry = next;
      first = false;
    }
    marg |= counter >= 150;
    return collect;
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE uint64_t compatibleLayers(Table const& T,
                                                                   int start,
                                                                   TrackIn const& t,
                                                                   bool along) {
    bool marg = false;
    return compatibleLayers(T, start, t, along, marg);
  }
}  // namespace mkfitdev::nav

#endif
