#ifndef RecoTracker_MkFitAlpaka_interface_math_NavSearch_h
#define RecoTracker_MkFitAlpaka_interface_math_NavSearch_h

// the compatibleDets search of the OT endcap layers (Phase2EndcapLayer =
// tkDetUtil::groupedCompatibleDetsV over Phase2EndcapRing, brothers = the upper sensors) on FLAT tables, portable
// (ALPAKA_FN_HOST_ACC), fixed capacities with overflow flags. Transliterated from the TkDetLayers code
// (tkDetUtil::groupedCompatibleDetsV, Phase2EndcapRing / Phase2EndcapSingleRing::groupedCompatibleDetsV,
// DetGroupMerger, the barrel rod searches); identical per call to the TkDetLayers search. The per-det test is
// navdev::detTest from the device start state.

#include <cmath>
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/DetPlaneTest.h"
#include "RecoTracker/MkFitAlpaka/interface/math/HelixCrossings.h"
#include "RecoTracker/MkFitAlpaka/interface/math/PcaToBeamLine.h"

namespace mkfitdev::navdev {

  constexpr int kNavMaxGroups = 8;     // groups per search
  constexpr int kNavMaxDetTests = 64;  // det tests per search
  constexpr int kNavMaxRings = 16;     // rings per endcap layer
  enum NavOverflow : int {
    kNavOvfGroups = 2,
    kNavOvfDetTests = 4,
    kNavOvfRings = 8,
    kNavUnsupported = 16,
    kNavOvfArena = 32,
    kNavMarginal = 64  // a det test within rounding of its bounds (boundsMarginal): the call goes to the host
  };
  constexpr int kNavArena = 24;       // group lists per call in the caller's scratch (the deepest search nests 11)
  constexpr int kNavMaxSubDisks = 8;  // sub-disks per pixel double disk

  // ---- flat tables (built on the host from the TkDetLayers objects; an ES product later) ----
  struct NavDet {
    DetPlane plane;              // position, axes, rectangular half sizes
    float phi;                   // surface().phi()
    float phiSpanLo, phiSpanHi;  // surface().phiSpan()
    float posZ;                  // position().z()
    float posPerp;               // position().perp()
    float normalZ;               // surface().normalVector().z()
    float originLx, originLy;    // surface().toLocal(GlobalPoint(0, 0, 0))
  };
  struct NavRing {  // a Phase2EndcapRing (tkDetUtil ring parameters + its two sub-layers)
    float diskZ, ringR, thetaMin, thetaMax;
    float subZ[2];  // sub-layer plane z (ForwardRingDiskBuilderFromDet)
    float phiOffset[2], invPhiStep[2];
    int n[2];    // dets per sub-layer
    int sub[2];  // offsets into the det-index list (lower sensors, in phi order)
    int bro[2];  // offsets of the brothers (upper sensors), -1 = none
    int single;  // Phase2EndcapSingleRing (pixel double disks): sub-layer 0 only, crossing at diskZ
  };
  struct NavSubDisk {  // a sub-disk of a Phase2EndcapLayerDoubleDisk: its single rings
    float z;
    int ring0, nRing;
  };
  struct NavRod {  // a Phase2OTBarrelRod (stacked) or a PixelRod
    int stacked;
    int n[2], sub[2], bro[2];         // stacked: sub-rods (lower sensors in z order) + brothers; pixel: n[0], sub[0]
    float zOffset[2], zStep[2];       // GenericBinFinderInZ (stacked) / PeriodicBinFinderInZ (pixel, [0])
    int zs[2];                        // GenericBinFinderInZ: offsets into the float list (n bins, then n - 1 borders)
    DetPlane plane[2];                // sub-rod planes (stacked) / the rod plane ([0], pixel)
    float phi, phiSpanLo, phiSpanHi;  // the rod surface (TBPLayer neighbours)
  };
  struct NavBarrel {  // a TBPLayer (inner / outer rods) + the tilted rings of a Phase2OTtiltedBarrelLayer
    int stacked;
    float cylR[2];
    float phiOffset[2], phiStep[2], invPhiStep[2];  // PeriodicBinFinderInPhi over the rods
    int nRod[2], rod0[2];                           // rods in phi order: rod rows [rod0, rod0 + nRod)
    int ring0[2], nRing[2];                         // tilted rings per z side (negative, positive)
  };
  struct NavTables {
    NavDet const* dets;
    int const* idx;
    NavRing const* rings;
    NavSubDisk const* subDisks;
    NavRod const* rods = nullptr;
    NavBarrel const* barrels = nullptr;
    float const* zs = nullptr;
  };

  // ---- groups: a DetGroup reduced to what the merges and the converter read: index, indexSize, the number of dets
  // and the FIRST det (front() of the group; DetGroupMerger only appends, so it never changes). Exact for the result
  // order and its front(); 16 bytes per group, so one thread's search state stays small. ----
  struct NavGroup {
    int index = 0, indexSize = 1, n = 0, first = -1;
  };
  struct NavGroups {
    int n = 0;
    NavGroup g[kNavMaxGroups];
  };
  struct NavCtx {
    NavTables t;
    DetTestStart const* st;
    bool along;
    double maxSagitta, minTolerance2, nSigma, maxDisplacement;
    int detTests = 0;
    int overflow = 0;
    NavGroups* arena =
        nullptr;  // kNavArena group lists of scratch (global memory on the device): keeps the stack small
    int arenaTop = 0;
    // the det-test memo of memoLayerSearch (kNavMemo entries of scratch; required on the device, where the
    // searches never run a det test themselves). Entries [0, nDone) hold results, [nDone, nMemo) are requested dets.
    DetTestResult* memo = nullptr;
    int* memoDet = nullptr;
    int nMemo = 0, nDone = 0;
    bool incomplete = false;  // this pass met a det without a result: its outcome is discarded
  };
  constexpr int kNavMemo = kNavMaxDetTests;  // distinct det tests per search in the memo
  // a group list from the scratch, released at the end of the scope (stack order)
  struct NavSlot {
    NavCtx& c;
    NavGroups* p;
    ALPAKA_FN_HOST_ACC explicit NavSlot(NavCtx& ctx) : c(ctx) {
      if (c.arenaTop >= kNavArena) {
        c.overflow |= kNavOvfArena;  // results unreliable: the caller falls back to the host
        p = c.arena + kNavArena - 1;
      } else
        p = c.arena + c.arenaTop;
      ++c.arenaTop;
      p->n = 0;
    }
    ALPAKA_FN_HOST_ACC ~NavSlot() { --c.arenaTop; }
    NavSlot(NavSlot const&) = delete;
    NavSlot& operator=(NavSlot const&) = delete;
  };

  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float barePhiF(float x, float y) { return ::mkfitdev::pca::atan2F(y, x); }
  // reco::reducePhiRange / deltaPhi (float) and Geom::phiLess
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float deltaPhiF(float a, float b) {
    const float x = a - b;
    constexpr float o2pi = 1. / (2. * M_PI);
    if (std::abs(x) <= float(M_PI))
      return x;
    const float n = std::round(x * o2pi);
    return x - n * float(2. * M_PI);
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool phiLessF(float a, float b) { return deltaPhiF(a, b) < 0; }

  // Binning guards: a bin / closest / window decision within kNavMarginAbs (kNavMarginAbs / r as an angle) of its edge
  // sends the call to the host search (kNavMarginal), like the det tests' boundsMarginal.
  constexpr float kNavGuardPos = float(kNavMarginAbs);
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void guardNear(float a, float b, float margin, NavCtx& c) {
    if (std::abs(a - b) <= margin)
      c.overflow |= kNavMarginal;
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void guardPhiNear(float a, float b, float margin, NavCtx& c) {
    if (std::abs(deltaPhiF(a, b)) <= margin)
      c.overflow |= kNavMarginal;
  }
  // int(tmp) of a bin finder: tmp within margin (in bin units) of an integer = a bin edge
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void guardBinEdge(float tmp, float margin, NavCtx& c) {
    const float f = tmp - std::floor(tmp);
    if (f <= margin || f >= 1.f - margin)
      c.overflow |= kNavMarginal;
  }
  // the angle of kNavGuardPos at the transverse radius of (x, y)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float guardPhiAt(float x, float y) {
    return kNavGuardPos / std::sqrt(x * x + y * y);
  }

  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void pushGroup(NavGroups& r, NavGroup const& g, NavCtx& c) {
    if (r.n < kNavMaxGroups)
      r.g[r.n++] = g;
    else
      c.overflow |= kNavOvfGroups;
  }
  // the det test of one det: from the memo (memoLayerSearch; always on the device) or run here (host, no memo)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE DetTestResult testDet(int det, NavCtx& c) {
    DetTestResult r;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    if (c.memo == nullptr)
      r = detTest(*c.st, c.t.dets[det].plane, c.along, c.maxSagitta, c.minTolerance2, c.nSigma);
    else
#endif
    {
      int i = 0;
      while (i < c.nMemo && c.memoDet[i] != det)
        ++i;
      if (i >= c.nDone) {  // no result yet: requested for the next pass; a placeholder (no crossing) for this one
        c.incomplete = true;
        if (i == c.nMemo && c.nMemo < kNavMemo)
          c.memoDet[c.nMemo++] = det;
        return r;
      }
      r = c.memo[i];
    }
    if (r.marginal)
      c.overflow |= kNavMarginal;
    return r;
  }
  // the single-group result of one det test (res.front().el.push_back); r = the det's state
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool addDet(int det, NavGroups& res, NavCtx& c, DetTestResult& r) {
    if (c.detTests >= kNavMaxDetTests)
      c.overflow |= kNavOvfDetTests;
    ++c.detTests;
    r = testDet(det, c);
    if (!r.compatible)
      return false;
    if (res.n == 0) {
      res.n = 1;
      res.g[0] = NavGroup{0, 1, 0, det};
    }
    ++res.g[0].n;
    return true;
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool addDet(int det, NavGroups& res, NavCtx& c) {
    DetTestResult r;
    return addDet(det, res, c, r);
  }
  // DetGroupMerger::mergeTwoLevels (appended to result)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void mergeTwoLevels(NavGroups const& one,
                                                             NavGroups const& two,
                                                             NavGroups& result,
                                                             NavCtx& c) {
    const int s1 = one.g[0].indexSize, s2 = two.g[0].indexSize;
    for (int i = 0; i < one.n; ++i) {
      NavGroup g = one.g[i];
      g.indexSize = s1 + s2;
      pushGroup(result, g, c);
    }
    for (int i = 0; i < two.n; ++i) {
      NavGroup g = two.g[i];
      g.index += s1;
      g.indexSize += s1;
      pushGroup(result, g, c);
    }
  }
  // DetGroupMerger::orderAndMergeTwoLevels (one half empty: result is REPLACED, as on the host)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void orderAndMerge(
      NavGroups const& one, NavGroups const& two, NavGroups& result, int firstIndex, int firstCrossed, NavCtx& c) {
    if (one.n == 0 && two.n == 0)
      return;
    if (one.n == 0 || two.n == 0) {
      result = one.n == 0 ? two : one;
      const int s = result.g[0].indexSize;
      const bool inc = (one.n == 0) == (firstIndex == firstCrossed);  // incrementAndDoubleSize
      for (int i = 0; i < result.n; ++i) {
        if (inc) {
          result.g[i].index += s;
          result.g[i].indexSize += s;
        } else
          result.g[i].indexSize = 2 * s;  // doubleIndexSize
      }
    } else if (firstIndex == firstCrossed)
      mergeTwoLevels(one, two, result, c);
    else
      mergeTwoLevels(two, one, result, c);
  }
  // DetGroupMerger::addSameLevel
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void addSameLevel(NavGroups const& gvec, NavGroups& result, NavCtx& c) {
    for (int k = 0; k < gvec.n; ++k) {
      NavGroup const& ig = gvec.g[k];
      int pos = result.n;
      bool merged = false;
      for (int i = 0; i < result.n; ++i) {
        if (ig.index == result.g[i].index) {
          result.g[i].n += ig.n;  // appended after the group's first det
          merged = true;
          break;
        } else if (ig.index < result.g[i].index) {
          pos = i;
          break;
        }
      }
      if (merged)
        continue;
      if (result.n >= kNavMaxGroups) {
        c.overflow |= kNavOvfGroups;
        continue;
      }
      for (int i = result.n; i > pos; --i)
        result.g[i] = result.g[i - 1];
      result.g[pos] = ig;
      ++result.n;
    }
  }

  // Chi2MeasurementEstimatorBase::maximalLocalDisplacement on a det-test state
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void maxLocalDisplacement(DetTestResult const& r,
                                                                   NavCtx const& c,
                                                                   float& mx,
                                                                   float& my) {
    const float emax = c.maxDisplacement;
    mx = float(std::fmin(emax, std::sqrt(float(r.sxx))) * c.nSigma);
    my = float(std::fmin(emax, std::sqrt(float(r.syy))) * c.nSigma);
  }
  // Rounding bound of the window w = acos(sp): each float sp carries <= 4.25 ulps (7 roundings) and dw = dsp / sin(w);
  // 16 ulps covers the host and the device with a factor 2 margin
  constexpr float kNavWindowSpErr = 16.f * 5.9604645e-8f;  // 16 * 2^-24
  // tkDetUtil::computeWindowSize = calculatePhiWindow with the sign of the x displacement; wErr (if given) = the
  // window's rounding bound, added to the guard margin of the overlap tests that use the window
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float phiWindow(NavDet const& d,
                                                         DetTestResult const& r,
                                                         NavCtx const& c,
                                                         float* wErr = nullptr) {
    float ix, iy;
    maxLocalDisplacement(r, c, ix, iy);
    const float mdx = std::abs(ix), mdy = std::abs(iy);
    constexpr float tolerance = 1.e-6;
    const float sx = float(r.lx), sy = float(r.ly);
    float w = 0;
    if (std::abs(1.f - std::abs(d.normalZ)) < tolerance) {
      const float xc = std::abs(sx - d.originLx);
      const float yc = std::abs(sy - d.originLy);
      if (yc < mdy && xc < mdx)
        w = M_PI;
      else {
        const bool hori = yc > mdy;
        const float y0 = hori ? yc + std::copysign(mdy, xc - mdx) : xc - mdx;
        const float x0 = hori ? xc - mdx : -yc - mdy;
        const float y1 = hori ? yc - mdy : xc - mdx;
        const float x1 = hori ? xc + mdx : -yc + mdy;
        float sp = (x0 * x1 + y0 * y1) / std::sqrt((x0 * x0 + y0 * y0) * (x1 * x1 + y1 * y1));
        sp = std::fmin(std::fmax(sp, -1.f), 1.f);
        w = float(std::acos(double(sp)));
        if (wErr)  // acos near sp = 1 amplifies the rounding of sp by 1 / sin(w)
          *wErr = kNavWindowSpErr / std::sqrt(std::fmax(1.f - sp * sp, 1.e-12f));
      }
    } else {
      DetPlane const& p = d.plane;
      auto cornerPhi = [&](float lx, float ly) MKFITDEV_NAV_LAMBDA_INLINE {
        const float gx = float(p.ax[0]) * lx + float(p.ay[0]) * ly + float(p.pos[0]);
        const float gy = float(p.ax[1]) * lx + float(p.ay[1]) * ly + float(p.pos[1]);
        return barePhiF(gx, gy);
      };
      const float corners[4] = {cornerPhi(sx + mdx, sy + mdy),
                                cornerPhi(sx - mdx, sy + mdy),
                                cornerPhi(sx - mdx, sy - mdy),
                                cornerPhi(sx + mdx, sy - mdy)};
      float phimin = corners[0], phimax = phimin;
      for (int i = 1; i < 4; i++) {
        if (phiLessF(corners[i], phimin))
          phimin = corners[i];
        if (phiLessF(phimax, corners[i]))
          phimax = corners[i];
      }
      w = phimax - phimin;
      if (w < 0.)
        w += 2. * M_PI;
    }
    return std::copysign(w, ix);
  }

  // PeriodicBinFinderInPhi<float>::binIndex; dPhi = the guard angle (a bin edge within dPhi: marginal)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE int ringBin(NavRing const& R, int s, float phi, float dPhi, NavCtx& c) {
    constexpr float kTwoPi = 2 * float(3.141592653589793238);
    const int n = R.n[s];
    float tmp = std::fmod((phi - R.phiOffset[s]), kTwoPi) * R.invPhiStep[s];
    if (tmp < 0)
      tmp += n;
    guardBinEdge(tmp, dPhi * R.invPhiStep[s], c);
    const int b = int(tmp);
    return b < n - 1 ? b : n - 1;
  }

  // Phase2EndcapRing::groupedCompatibleDetsV
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void ringGroups(NavRing const& R, NavCtx& c, NavGroups& result) {
    DetTestStart const& st = *c.st;
    const float curv = float(st.rho);
    const Crossing fp = forwardPlaneCrossing(st.t.x, st.t.p, curv, c.along, R.subZ[0]);
    if (fp.status != kXingOk)
      return;
    const Crossing bp = forwardPlaneCrossing(st.t.x, st.t.p, curv, c.along, R.subZ[1]);
    if (bp.status != kXingOk)
      return;
    const float px[2] = {fp.x, bp.x}, py[2] = {fp.y, bp.y};
    float pphi[2];
    int idx[2];
    float dist[2];
    const float dPhi = guardPhiAt(px[0], py[0]);
    for (int s = 0; s < 2; ++s) {
      pphi[s] = barePhiF(px[s], py[s]);
      idx[s] = ringBin(R, s, pphi[s], dPhi, c);
      dist[s] = std::abs(deltaPhiF(pphi[s], c.t.dets[c.t.idx[R.sub[s] + idx[s]]].phi));
    }
    guardNear(dist[0], dist[1], dPhi, c);
    const int cs = dist[0] < dist[1] ? 0 : 1, os = 1 - cs;
    const bool hasBro = R.bro[0] >= 0 && R.bro[1] >= 0;
    auto subDet = [&](int s, int i) MKFITDEV_NAV_LAMBDA_INLINE { return c.t.idx[R.sub[s] + i]; };
    auto broDet = [&](int s, int i) MKFITDEV_NAV_LAMBDA_INLINE { return c.t.idx[R.bro[s] + i]; };
    NavSlot closestResSlot(c);
    NavSlot closestBroSlot(c);
    NavGroups &closestRes = *closestResSlot.p, &closestBro = *closestBroSlot.p;
    DetTestResult cg;  // the closest det's state (closestRes.front().el.front())
    addDet(subDet(cs, idx[cs]), closestRes, c, cg);
    if (hasBro)
      addDet(broDet(cs, idx[cs]), closestBro, c);
    if (closestRes.n == 0)
      return;
    const int cgDet = closestRes.g[0].first;
    // LayerCrossingSide::endcapSide on the closest det's state
    const bool outwards = cg.tz * cg.gz > 0;
    const int side = c.along == outwards ? 0 : 1;
    float wErr = 0;
    const float window = phiWindow(c.t.dets[cgDet], cg, c, &wErr);
    auto searchNeighbors =
        [&](int s, int ci, NavGroups& res, NavGroups& bres, bool checkClosest) MKFITDEV_NAV_LAMBDA_INLINE {
          const int n = R.n[s];
          const float cphi = pphi[s];
          int negStart = ci - 1, posStart = ci + 1;
          if (checkClosest) {
            guardPhiNear(cphi, c.t.dets[subDet(s, ci)].phi, dPhi, c);
            if (phiLessF(cphi, c.t.dets[subDet(s, ci)].phi))
              posStart = ci;
            else
              negStart = ci;
          }
          auto wrap = [n](int i) {
            const int ind = i % n;
            return ind < 0 ? ind + n : ind;
          };
          auto overlapInPhi = [&](int det) MKFITDEV_NAV_LAMBDA_INLINE {
            const float lo = cphi - window, hi = cphi + window;
            NavDet const& d = c.t.dets[det];
            guardPhiNear(d.phiSpanHi, lo, dPhi + wErr, c);
            guardPhiNear(hi, d.phiSpanLo, dPhi + wErr, c);
            return !(phiLessF(d.phiSpanHi, lo) | phiLessF(hi, d.phiSpanLo));
          };
          const int halfN = n / 2;
          for (int i = negStart; i >= negStart - halfN; i--) {
            if (!overlapInPhi(subDet(s, wrap(i))) || !addDet(subDet(s, wrap(i)), res, c) || !hasBro)
              break;
            addDet(broDet(s, wrap(i)), bres, c);
          }
          for (int i = posStart; i < posStart + halfN; i++) {
            if (!overlapInPhi(subDet(s, wrap(i))) || !addDet(subDet(s, wrap(i)), res, c) || !hasBro)
              break;
            addDet(broDet(s, wrap(i)), bres, c);
          }
        };
    searchNeighbors(cs, idx[cs], closestRes, closestBro, false);
    NavSlot closestCompleteSlot(c);
    NavSlot nextResSlot(c);
    NavSlot nextBroSlot(c);
    NavSlot nextCompleteSlot(c);
    NavGroups &closestComplete = *closestCompleteSlot.p, &nextRes = *nextResSlot.p, &nextBro = *nextBroSlot.p,
              &nextComplete = *nextCompleteSlot.p;
    orderAndMerge(closestRes, closestBro, closestComplete, 0, side, c);
    searchNeighbors(os, idx[os], nextRes, nextBro, true);
    orderAndMerge(nextRes, nextBro, nextComplete, 0, side, c);
    orderAndMerge(closestComplete, nextComplete, result, cs, side, c);
    // TkDetLayers: sort by |z| of each group's first det (DetGroupElementZLess; <= 4 groups): insertion, only with
    // brothers
    for (int i = 1; hasBro && i < result.n; ++i)
      for (int j = i;
           j > 0 && std::abs(c.t.dets[result.g[j].first].posZ) < std::abs(c.t.dets[result.g[j - 1].first].posZ);
           --j) {
        const NavGroup tmp = result.g[j];
        result.g[j] = result.g[j - 1];
        result.g[j - 1] = tmp;
      }
  }

  // Phase2EndcapSingleRing::groupedCompatibleDetsV (one sub-layer, no brothers, addSameLevel into result)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void singleRingGroups(NavRing const& R, NavCtx& c, NavGroups& result) {
    DetTestStart const& st = *c.st;
    const Crossing x = forwardPlaneCrossing(st.t.x, st.t.p, float(st.rho), c.along, R.diskZ);
    if (x.status != kXingOk)
      return;
    const float cphi = barePhiF(x.x, x.y);
    const float dPhi = guardPhiAt(x.x, x.y);
    const int idx = ringBin(R, 0, cphi, dPhi, c);
    const int n = R.n[0];
    auto subDet = [&](int i) MKFITDEV_NAV_LAMBDA_INLINE { return c.t.idx[R.sub[0] + i]; };
    NavSlot closestResSlot(c);
    NavGroups& closestRes = *closestResSlot.p;
    DetTestResult cg;
    addDet(subDet(idx), closestRes, c, cg);
    if (closestRes.n == 0)
      return;
    float wErr = 0;
    const float window = phiWindow(c.t.dets[closestRes.g[0].first], cg, c, &wErr);
    auto overlapInPhi = [&](int det) MKFITDEV_NAV_LAMBDA_INLINE {
      const float lo = cphi - window, hi = cphi + window;
      NavDet const& d = c.t.dets[det];
      guardPhiNear(d.phiSpanHi, lo, dPhi + wErr, c);
      guardPhiNear(hi, d.phiSpanLo, dPhi + wErr, c);
      return !(phiLessF(d.phiSpanHi, lo) | phiLessF(hi, d.phiSpanLo));
    };
    auto wrap = [n](int i) {
      const int ind = i % n;
      return ind < 0 ? ind + n : ind;
    };
    const int halfN = n / 2;
    for (int i = idx - 1; i >= idx - 1 - halfN; i--)
      if (!overlapInPhi(subDet(wrap(i))) || !addDet(subDet(wrap(i)), closestRes, c))
        break;
    for (int i = idx + 1; i < idx + 1 + halfN; i++)
      if (!overlapInPhi(subDet(wrap(i))) || !addDet(subDet(wrap(i)), closestRes, c))
        break;
    addSameLevel(closestRes, result, c);
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void anyRingGroups(NavRing const& R, NavCtx& c, NavGroups& result) {
    if (R.single)
      singleRingGroups(R, c, result);
    else
      ringGroups(R, c, result);
  }

  // tkDetUtil::groupedCompatibleDetsV over the rings [ring0, ring0 + nR) of a Phase2EndcapLayer; false = not supported
  // (the caller falls back to the host)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool endcapLayerSearch(int ring0, int nR, NavCtx& c, NavGroups& result) {
    if (nR <= 0 || nR > kNavMaxRings) {
      c.overflow |= kNavOvfRings;
      return false;
    }
    DetTestStart const& st = *c.st;
    const float curv = float(st.rho);
    float rr[kNavMaxRings];  // |perp(crossing) - ringR|
    for (int i = 0; i < nR; ++i) {
      NavRing const& R = c.t.rings[ring0 + i];
      const Crossing x = forwardPlaneCrossing(st.t.x, st.t.p, curv, c.along, R.diskZ);
      const float px = x.status == kXingOk ? x.x : 0.f, py = x.status == kXingOk ? x.y : 0.f;
      rr[i] = std::abs(std::sqrt(px * px + py * py) - R.ringR);
    }
    // tkDetUtil::findThreeClosest
    int bins[3] = {0, -1, -1};
    {
      float r0 = rr[0], r1 = -1., r2 = -1.;
      for (int i = 1; i < nR; i++) {
        const float t = rr[i];
        guardNear(t, r0, kNavGuardPos, c);
        if (r1 >= 0)
          guardNear(t, r1, kNavGuardPos, c);
        if (r2 >= 0)
          guardNear(t, r2, kNavGuardPos, c);
        if (t < r0) {
          r2 = r1;
          r1 = r0;
          r0 = t;
          bins[2] = bins[1];
          bins[1] = bins[0];
          bins[0] = i;
        } else if (r1 < 0 || t < r1) {
          r2 = r1;
          r1 = t;
          bins[2] = bins[1];
          bins[1] = i;
        } else if (r2 < 0 || t < r2) {
          r2 = t;
          bins[2] = i;
        }
      }
    }
    if (bins[0] == -1 || bins[1] == -1 || bins[2] == -1)
      return true;  // TkDetLayers: LogError + empty
    int ringOrder[kNavMaxRings];
    for (int i = 0; i < nR; ++i)
      ringOrder[i] = 1;
    if (nR > 1) {
      const float z0 = std::abs(c.t.rings[ring0].diskZ), z1 = std::abs(c.t.rings[ring0 + 1].diskZ);
      if (z0 < z1) {
        for (int i = 0; i < nR; i++)
          if (i % 2 == 0)
            ringOrder[i] = 0;
      } else if (z0 > z1) {
        for (int i = 0; i < nR; i++)
          ringOrder[i] = i % 2 == 0 ? 1 : 0;
      } else {
        c.overflow |= kNavUnsupported;
        return false;
      }
    }
    auto index = [&](int i) MKFITDEV_NAV_LAMBDA_INLINE { return ringOrder[bins[i]]; };
    auto ring = [&](int i) MKFITDEV_NAV_LAMBDA_INLINE -> NavRing const& { return c.t.rings[ring0 + bins[i]]; };
    NavSlot closestResSlot(c);
    NavGroups& closestRes = *closestResSlot.p;
    anyRingGroups(ring(0), c, closestRes);
    if (closestRes.n == 0) {
      anyRingGroups(ring(1), c, result);
      return true;
    }
    // the state of closestRes.front().el.front(): the same det test again (deterministic; not counted)
    DetTestResult const cg = testDet(closestRes.g[0].first, c);
    float mx, my;
    maxLocalDisplacement(cg, c, mx, my);
    const double rWindow = my;
    auto overlapInR = [&](int i) MKFITDEV_NAV_LAMBDA_INLINE {
      const float tsRadius = std::sqrt(cg.gx * cg.gx + cg.gy * cg.gy);
      const float thetamin = (std::fmax(0., tsRadius - rWindow)) / (std::abs(cg.gz) + 10.f);
      const float thetamax = (tsRadius + rWindow) / (std::abs(cg.gz) - 10.f);
      const float dTheta = kNavGuardPos / (std::abs(cg.gz) - 10.f);  // the radius margin as a "theta" here
      guardNear(thetamin, ring(i).thetaMax, dTheta, c);
      guardNear(ring(i).thetaMin, thetamax, dTheta, c);
      return !(thetamin > ring(i).thetaMax || ring(i).thetaMin > thetamax);
    };
    const bool ring1ok = overlapInR(1);
    bool ring2ok = overlapInR(2);
    int direction = 0;
    if (st.t.x.z * st.t.p.z > 0)
      direction = c.along ? 0 : 1;
    else
      direction = c.along ? 1 : 0;
    if (index(0) == index(1) && index(0) == index(2))
      ring2ok = false;
    if (index(0) == index(1)) {
      if (ring1ok) {
        NavSlot r1Slot(c);
        NavGroups& r1 = *r1Slot.p;
        anyRingGroups(ring(1), c, r1);
        addSameLevel(r1, closestRes, c);
      }
      if (ring2ok) {
        NavSlot r2Slot(c);
        NavGroups& r2 = *r2Slot.p;
        anyRingGroups(ring(2), c, r2);
        orderAndMerge(closestRes, r2, result, index(0), direction, c);
      } else
        result = closestRes;
    } else if (index(0) == index(2)) {
      if (ring2ok) {
        NavSlot r2Slot(c);
        NavGroups& r2 = *r2Slot.p;
        anyRingGroups(ring(2), c, r2);
        addSameLevel(r2, closestRes, c);
      }
      if (ring1ok) {
        NavSlot r1Slot(c);
        NavGroups& r1 = *r1Slot.p;
        anyRingGroups(ring(1), c, r1);
        orderAndMerge(closestRes, r1, result, index(0), direction, c);
      } else
        result = closestRes;
    } else {
      NavSlot r12Slot(c);
      NavGroups& r12 = *r12Slot.p;
      if (ring1ok)
        anyRingGroups(ring(1), c, r12);
      if (ring2ok) {
        NavSlot r2Slot(c);
        NavGroups& r2 = *r2Slot.p;
        anyRingGroups(ring(2), c, r2);
        addSameLevel(r2, r12, c);
      }
      if (r12.n > 0)
        orderAndMerge(closestRes, r12, result, index(0), direction, c);
      else
        result = closestRes;
    }
    return true;
  }

  // Phase2EndcapLayerDoubleDisk::groupedCompatibleDetsV over the sub-disks [sub0, sub0 + nS) (tkDetUtil over each
  // sub-disk's single rings)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool pixEndcapSearch(int sub0, int nS, NavCtx& c, NavGroups& result) {
    if (nS <= 0 || nS > kNavMaxSubDisks) {
      c.overflow |= kNavUnsupported;
      return false;
    }
    DetTestStart const& st = *c.st;
    const float curv = float(st.rho);
    float sz[kNavMaxSubDisks];
    for (int i = 0; i < nS; ++i) {
      const Crossing x = forwardPlaneCrossing(st.t.x, st.t.p, curv, c.along, c.t.subDisks[sub0 + i].z);
      sz[i] = x.status == kXingOk ? x.z : 0.f;
    }
    // findTwoClosest (|z| of the sub-disk surface)
    int bins[2] = {0, -1};
    {
      float z0 = std::abs(sz[0] - std::abs(c.t.subDisks[sub0].z)), z1 = -1.;
      for (int i = 1; i < nS; i++) {
        const float t = std::abs(sz[i] - std::abs(c.t.subDisks[sub0 + i].z));
        guardNear(t, z0, kNavGuardPos, c);
        if (z1 >= 0)
          guardNear(t, z1, kNavGuardPos, c);
        if (t < z0) {
          z1 = z0;
          z0 = t;
          bins[1] = bins[0];
          bins[0] = i;
        } else if (z1 < 0 || t < z1) {
          z1 = t;
          bins[1] = i;
        }
      }
    }
    int order[kNavMaxSubDisks];
    for (int i = 0; i < nS; ++i)
      order[i] = 1;
    if (nS > 1) {
      const float a = std::abs(c.t.subDisks[sub0].z), b = std::abs(c.t.subDisks[sub0 + 1].z);
      if (a < b) {
        for (int i = 0; i < nS; i++)
          if (i % 2 == 0)
            order[i] = 0;
      } else if (a > b) {
        for (int i = 0; i < nS; i++)
          order[i] = i % 2 == 0 ? 1 : 0;
      } else {
        c.overflow |= kNavUnsupported;
        return false;
      }
    }
    auto subGroups = [&](int i, NavGroups& res) MKFITDEV_NAV_LAMBDA_INLINE {
      NavSubDisk const& d = c.t.subDisks[sub0 + i];
      endcapLayerSearch(d.ring0, d.nRing, c, res);  // the return value is ignored, as the transliteration does
    };
    NavSlot closestResSlot(c);
    NavGroups& closestRes = *closestResSlot.p;
    subGroups(bins[0], closestRes);
    if (closestRes.n == 0) {
      subGroups(bins[1], result);
      return true;
    }
    const bool sub1ok = bins[1] != -1;
    int direction = 0;
    if (st.t.x.z * st.t.p.z > 0)
      direction = c.along ? 0 : 1;
    else
      direction = c.along ? 1 : 0;
    if (!sub1ok || order[bins[0]] == order[bins[1]]) {  // bins[1] is always set for nS >= 2
      if (sub1ok) {
        NavSlot r1Slot(c);
        NavGroups& r1 = *r1Slot.p;
        subGroups(bins[1], r1);
        addSameLevel(r1, closestRes, c);
        result = closestRes;
      }
    } else {
      NavSlot r1Slot(c);
      NavGroups& r1 = *r1Slot.p;
      if (sub1ok)
        subGroups(bins[1], r1);
      if (r1.n > 0)
        orderAndMerge(closestRes, r1, result, order[bins[0]], direction, c);
      else
        result = closestRes;
    }
    return true;
  }

  // ---- barrel (TBPLayer + Phase2OTBarrelRod / PixelRod + Phase2OTtiltedBarrelLayer rings) ----
  // local y of a global point on a det plane (Surface::toLocal, float)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float localY(DetPlane const& p, float x, float y, float z) {
    const float dx = x - float(p.pos[0]), dy = y - float(p.pos[1]), dz = z - float(p.pos[2]);
    return float(p.ay[0]) * dx + float(p.ay[1]) * dy + float(p.ay[2]) * dz;
  }
  // LayerCrossingSide::barrelSide on a det-test state
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE int barrelSide(DetTestResult const& r, bool along) {
    const bool outwards = (r.gx * float(r.tx) + r.gy * float(r.ty)) > 0;
    return along == outwards ? 0 : 1;
  }
  // GenericBinFinderInZ<float, GeomDet>::binIndex
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE int genericBinZUnguarded(NavRod const& R, int s, float const* zs, float z) {
    const int n = R.n[s];
    float const* bins = zs + R.zs[s];
    float const* borders = bins + n;
    auto clampBin = [n](int i) { return i < 0 ? 0 : (i > n - 1 ? n - 1 : i); };
    int bin = clampBin(int((z - R.zOffset[s]) / R.zStep[s]) + 1);
    if (bin > 0) {
      if (z < borders[bin - 1]) {
        for (int i = bin - 1;; i--) {
          if (i <= 0)
            return 0;
          if (z > borders[i - 1])
            return i;
        }
      }
    } else
      return 0;
    if (bin < n - 1) {
      if (z > borders[bin]) {
        for (int i = bin + 1;; i++) {
          if (i >= n - 1)
            return n - 1;
          if (z < borders[i])
            return i;
        }
      }
    } else
      return n - 1;
    return bin;
  }
  // the same, guarded: the result is fixed by the two borders around z (the first guess only starts the walk)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE int genericBinZ(NavRod const& R, int s, float const* zs, float z, NavCtx& c) {
    const int b = genericBinZUnguarded(R, s, zs, z);
    float const* borders = zs + R.zs[s] + R.n[s];
    if (b > 0)
      guardNear(z, borders[b - 1], kNavGuardPos, c);
    if (b < R.n[s] - 1)
      guardNear(z, borders[b], kNavGuardPos, c);
    return b;
  }
  // Phase2OTBarrelRod::groupedCompatibleDetsV
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void stackRodGroups(NavRod const& R, NavCtx& c, NavGroups& result) {
    DetTestStart const& st = *c.st;
    auto planeX = [&](int s) MKFITDEV_NAV_LAMBDA_INLINE {
      DetPlane const& p = R.plane[s];
      return barrelPlaneCrossing(st.t.x,
                                 st.t.p,
                                 float(st.rho),
                                 c.along,
                                 ::mkfitdev::pca::F3{float(p.pos[0]), float(p.pos[1]), float(p.pos[2])},
                                 ::mkfitdev::pca::F3{float(p.az[0]), float(p.az[1]), float(p.az[2])});
    };
    const Crossing op = planeX(1);
    if (op.status != kXingOk)
      return;
    const Crossing ip = planeX(0);
    if (ip.status != kXingOk)
      return;
    const float px[2] = {ip.x, op.x}, py[2] = {ip.y, op.y}, pz[2] = {ip.z, op.z};
    int idx[2];
    float dist[2];
    for (int s = 0; s < 2; ++s) {
      idx[s] = genericBinZ(R, s, c.t.zs, pz[s], c);
      dist[s] = std::abs(c.t.zs[R.zs[s] + idx[s]] - pz[s]);
    }
    guardNear(dist[0], dist[1], kNavGuardPos, c);
    const int cs = dist[0] < dist[1] ? 0 : 1, os = 1 - cs;
    auto subDet = [&](int s, int i) MKFITDEV_NAV_LAMBDA_INLINE { return c.t.idx[R.sub[s] + i]; };
    auto broDet = [&](int s, int i) MKFITDEV_NAV_LAMBDA_INLINE { return c.t.idx[R.bro[s] + i]; };
    auto overlapZ = [&](int s, int det, float window) MKFITDEV_NAV_LAMBDA_INLINE {
      NavDet const& d = c.t.dets[det];
      constexpr float relativeMargin = 1.01;
      const float lhs = std::abs(localY(d.plane, px[s], py[s], pz[s])) - window;
      const float rhs = relativeMargin * 0.5f * float(2 * d.plane.halfLength);
      guardNear(lhs, rhs, kNavGuardPos, c);
      return lhs < rhs;
    };
    auto searchNeighbors = [&](int s, float window, NavGroups& res, NavGroups& bres, bool checkClosest)
                               MKFITDEV_NAV_LAMBDA_INLINE {
                                 int negStart = idx[s] - 1, posStart = idx[s] + 1;
                                 if (checkClosest) {
                                   guardNear(pz[s], c.t.dets[subDet(s, idx[s])].posZ, kNavGuardPos, c);
                                   if (pz[s] < c.t.dets[subDet(s, idx[s])].posZ)
                                     posStart = idx[s];
                                   else
                                     negStart = idx[s];
                                 }
                                 for (int i = negStart; i >= 0; i--) {
                                   if (!overlapZ(s, subDet(s, i), window) || !addDet(subDet(s, i), res, c))
                                     break;
                                   addDet(broDet(s, i), bres, c);
                                 }
                                 for (int i = posStart; i < R.n[s]; i++) {
                                   if (!overlapZ(s, subDet(s, i), window) || !addDet(subDet(s, i), res, c))
                                     break;
                                   addDet(broDet(s, i), bres, c);
                                 }
                               };
    NavSlot closestResSlot(c);
    NavSlot closestBroSlot(c);
    NavGroups &closestRes = *closestResSlot.p, &closestBro = *closestBroSlot.p;
    DetTestResult cg;
    addDet(subDet(cs, idx[cs]), closestRes, c, cg);
    addDet(broDet(cs, idx[cs]), closestBro, c);
    if (closestRes.n == 0) {
      NavSlot nextResSlot(c);
      NavSlot nextBroSlot(c);
      NavGroups &nextRes = *nextResSlot.p, &nextBro = *nextBroSlot.p;
      DetTestResult ng;
      addDet(subDet(os, idx[os]), nextRes, c, ng);
      addDet(broDet(os, idx[os]), nextBro, c);
      if (nextRes.n == 0)
        return;
      const int side = barrelSide(ng, c.along);
      NavSlot ccSlot(c);
      NavSlot ncSlot(c);
      NavGroups &cc = *ccSlot.p, &nc = *ncSlot.p;
      orderAndMerge(closestRes, closestBro, cc, 0, side, c);
      orderAndMerge(nextRes, nextBro, nc, 0, side, c);
      orderAndMerge(cc, nc, result, cs, side, c);
    } else {
      const int side = barrelSide(cg, c.along);
      float mx, my;
      maxLocalDisplacement(cg, c, mx, my);
      const float window = my;
      searchNeighbors(cs, window, closestRes, closestBro, false);
      NavSlot ccSlot(c);
      NavSlot nextResSlot(c);
      NavSlot nextBroSlot(c);
      NavSlot ncSlot(c);
      NavGroups &cc = *ccSlot.p, &nextRes = *nextResSlot.p, &nextBro = *nextBroSlot.p, &nc = *ncSlot.p;
      orderAndMerge(closestRes, closestBro, cc, 0, side, c);
      searchNeighbors(os, window, nextRes, nextBro, true);
      orderAndMerge(nextRes, nextBro, nc, 0, side, c);
      orderAndMerge(cc, nc, result, cs, side, c);
    }
    // TkDetLayers: sort by the first det's perp (DetGroupElementPerpLess): insertion
    for (int i = 1; i < result.n; ++i)
      for (int j = i; j > 0 && c.t.dets[result.g[j].first].posPerp < c.t.dets[result.g[j - 1].first].posPerp; --j) {
        const NavGroup tmp = result.g[j];
        result.g[j] = result.g[j - 1];
        result.g[j - 1] = tmp;
      }
  }
  // PixelRod::compatibleDetsV (one group): first det and the number of dets appended
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE int pixelRodDets(NavRod const& R, NavCtx& c, int& first) {
    first = -1;
    // AnalyticalPropagator to the rod plane: OptimalHelixPlaneCrossing picks HelixBarrelPlaneCrossingByCircle for a
    // plane parallel to z (|n_z| < 1e-6), then the maxDPhi limit (float, as propagateWithPath)
    float rodZ;
    {
      DetPlane const& p = R.plane[0];
      DetTestStart const& st = *c.st;
      if (std::abs(float(p.az[2])) < 1.e-6f) {
        const Crossing x = barrelPlaneCrossing(st.t.x,
                                               st.t.p,
                                               float(st.rho),
                                               c.along,
                                               ::mkfitdev::pca::F3{float(p.pos[0]), float(p.pos[1]), float(p.pos[2])},
                                               ::mkfitdev::pca::F3{float(p.az[0]), float(p.az[1]), float(p.az[2])});
        if (x.status != kXingOk)
          return 0;
        float dphi2 = float(x.s) * float(st.rho);
        dphi2 = dphi2 * dphi2 * perp2F(st.t.p.x, st.t.p.y);
        const float mag2 = st.t.p.x * st.t.p.x + st.t.p.y * st.t.p.y + st.t.p.z * st.t.p.z;
        if (std::abs(dphi2 - 1.6f * 1.6f * mag2) <= kNavMarginRelDPhi * 1.6f * 1.6f * mag2)
          c.overflow |= kNavMarginal;
        if (dphi2 > 1.6f * 1.6f * mag2)
          return 0;
        rodZ = x.z;
      } else {
        const PlaneHit h = helixToPlane(
            HelixStart{{st.t.x.x, st.t.x.y, st.t.x.z}, {st.t.p.x, st.t.p.y, st.t.p.z}, st.rho}, p, c.along, 1.6);
        if (!h.valid)
          return 0;
        rodZ = float(h.x[2]);
      }
    }
    const int n = R.n[0];
    const float zBin = (rodZ - R.zOffset[0]) / R.zStep[0];
    guardBinEdge(zBin, kNavGuardPos / std::abs(R.zStep[0]), c);  // the closest-det decision of the rod
    int closest = int(zBin);
    closest = closest < 0 ? 0 : (closest > n - 1 ? n - 1 : closest);
    auto det = [&](int i) MKFITDEV_NAV_LAMBDA_INLINE { return c.t.idx[R.sub[0] + i]; };
    int count = 0;
    if (c.detTests >= kNavMaxDetTests)
      c.overflow |= kNavOvfDetTests;
    ++c.detTests;
    const DetTestResult cs = testDet(det(closest), c);
    if (cs.compatible) {
      first = det(closest);
      ++count;
    } else if (!cs.crossed)
      return 0;
    float mx, my;
    maxLocalDisplacement(cs, c, mx, my);
    const float detHalfLen = float(2 * c.t.dets[det(closest)].plane.halfLength) / 2.f;
    auto addOne = [&](int i) MKFITDEV_NAV_LAMBDA_INLINE {
      if (c.detTests >= kNavMaxDetTests)
        c.overflow |= kNavOvfDetTests;
      ++c.detTests;
      if (!testDet(det(i), c).compatible)
        return false;
      if (first < 0)
        first = det(i);
      ++count;
      return true;
    };
    for (int i = closest + 1; i < n; i++) {
      const float ly = localY(c.t.dets[det(i)].plane, cs.gx, cs.gy, cs.gz);
      guardNear(std::abs(ly), detHalfLen + my, kNavGuardPos, c);
      if (!(std::abs(ly) < detHalfLen + my) || !addOne(i))
        break;
    }
    for (int i = closest - 1; i >= 0; i--) {
      const float ly = localY(c.t.dets[det(i)].plane, cs.gx, cs.gy, cs.gz);
      guardNear(std::abs(ly), detHalfLen + my, kNavGuardPos, c);
      if (!(std::abs(ly) < detHalfLen + my) || !addOne(i))
        break;
    }
    return count;
  }
  // rod kinds a barrel search is compiled for
  enum NavRodKinds : int { kNavRodsAny = 0, kNavRodsStacked = 1, kNavRodsPixel = 2 };
  // CompatibleDetToGroupAdder::add for a rod; false = nothing compatible
  template <int kRods = kNavRodsAny>
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool rodAdd(int rod, NavCtx& c, NavGroups& result) {
    NavRod const& R = c.t.rods[rod];
    if ((kRods == kNavRodsStacked && !R.stacked) || (kRods == kNavRodsPixel && R.stacked)) {
      c.overflow |= kNavUnsupported;  // a layer with both rod kinds: the host searches it
      return false;
    }
    if constexpr (kRods != kNavRodsPixel) {
      if (kRods == kNavRodsStacked || R.stacked) {
        NavSlot tmpSlot(c);
        NavGroups& tmp = *tmpSlot.p;
        stackRodGroups(R, c, tmp);
        if (tmp.n == 0)
          return false;
        if (result.n == 0)
          result = tmp;
        else
          addSameLevel(tmp, result, c);
        return true;
      }
    }
    if constexpr (kRods != kNavRodsStacked) {
      int first;
      const int k = pixelRodDets(R, c, first);
      if (k == 0)
        return false;
      if (result.n == 0) {
        result.n = 1;
        result.g[0] = NavGroup{0, 1, 0, first};
      }
      result.g[0].n += k;
      return true;
    }
    return false;
  }
  // TBLayer::groupedCompatibleDetsV with TBPLayer's indexes / window / neighbours, then the tilted rings
  // parts of a barrel search (the device runs the rods and the tilted rings of an OT barrel layer in two
  // kernels; the rings part keeps the rods' scratch slot taken, as the whole search does)
  enum NavBarrelParts : int { kNavPartAll = 0, kNavPartRods = 1, kNavPartRings = 2 };
  template <int kRods = kNavRodsAny, int kPart = kNavPartAll>
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool barrelLayerSearch(int b, NavCtx& c, NavGroups& result) {
    NavBarrel const& B = c.t.barrels[b];
    DetTestStart const& st = *c.st;
    NavSlot rodsResSlot(c);
    NavGroups& rodsRes = *rodsResSlot.p;
    if constexpr (kPart != kNavPartRings)
      [&]() MKFITDEV_NAV_LAMBDA_INLINE {
        const float startPerp = std::sqrt(perp2F(st.t.x.x, st.t.x.y));
        const bool inBetween = startPerp < B.cylR[1] && startPerp > B.cylR[0];
        const Crossing ix = barrelCylinderCrossing(st.t.x, st.t.p, double(float(st.rho)), c.along, B.cylR[0]);
        if (ix.status == kXingStraight)
          c.overflow |= kNavUnsupported;
        if (!inBetween && ix.status != kXingOk)
          return;
        const Crossing ox = barrelCylinderCrossing(st.t.x, st.t.p, double(float(st.rho)), c.along, B.cylR[1]);
        if (ox.status == kXingStraight)
          c.overflow |= kNavUnsupported;
        if (ix.status != kXingOk && ox.status != kXingOk)
          return;
        float px[2] = {ix.x, ox.x}, py[2] = {ix.y, ox.y};
        if (ix.status != kXingOk) {
          px[0] = px[1];
          py[0] = py[1];
        } else if (ox.status != kXingOk) {
          px[1] = px[0];
          py[1] = py[0];
        }
        // TBPLayer::computeIndexes
        int idx[2];
        float dist[2], gphi[2];
        const float dPhi = guardPhiAt(px[0], py[0]);
        for (int s = 0; s < 2; ++s) {
          gphi[s] = barePhiF(px[s], py[s]);
          constexpr float kTwoPi = 2 * float(3.141592653589793238);
          float tmp = std::fmod((gphi[s] - B.phiOffset[s]), kTwoPi) * B.invPhiStep[s];
          if (tmp < 0)
            tmp += B.nRod[s];
          guardBinEdge(tmp, dPhi * B.invPhiStep[s], c);
          idx[s] = int(tmp) < B.nRod[s] - 1 ? int(tmp) : B.nRod[s] - 1;
          const float binPos = B.phiOffset[s] + B.phiStep[s] * (float(idx[s]) + 0.5f);
          dist[s] = binPos - gphi[s];
          dist[s] *= phiLessF(binPos, gphi[s]) ? -1.f : 1.f;
          if (dist[s] < 0.f)
            dist[s] += kTwoPi;
        }
        guardNear(dist[0], dist[1], dPhi, c);
        const int cs = dist[0] < dist[1] ? 0 : 1, os = 1 - cs;
        auto rodAt = [&](int s, int i) MKFITDEV_NAV_LAMBDA_INLINE {
          const int n = B.nRod[s];
          const int ind = i % n;
          return B.rod0[s] + (ind < 0 ? ind + n : ind);
        };
        NavSlot closestResSlot(c);
        NavGroups& closestRes = *closestResSlot.p;
        rodAdd<kRods>(rodAt(cs, idx[cs]), c, closestRes);
        if (closestRes.n == 0) {
          rodAdd<kRods>(rodAt(os, idx[os]), c, rodsRes);
          return;
        }
        const DetTestResult cg = testDet(closestRes.g[0].first, c);  // its state again (not counted)
        NavDet const& cd = c.t.dets[closestRes.g[0].first];
        // barrelUtil::computeWindowSize
        float mx, my;
        maxLocalDisplacement(cg, c, mx, my);
        float window;
        {
          DetPlane const& p = cd.plane;
          auto phiAt = [&](float lx, float ly) MKFITDEV_NAV_LAMBDA_INLINE {
            const float gx = float(p.ax[0]) * lx + float(p.ay[0]) * ly + float(p.pos[0]);
            const float gy = float(p.ax[1]) * lx + float(p.ay[1]) * ly + float(p.pos[1]);
            return barePhiF(gx, gy);
          };
          const float phi1 = phiAt(float(cg.lx) + mx, float(cg.ly));
          const float phi2 = phiAt(float(cg.lx) - mx, float(cg.ly));
          const float phiStart = barePhiF(cg.gx, cg.gy);
          window = std::fmin(std::abs(phiStart - phi1), std::abs(phiStart - phi2));
        }
        auto searchNeighbors = [&](int s, NavGroups& res, bool checkClosest) MKFITDEV_NAV_LAMBDA_INLINE {
          int negStart = idx[s] - 1, posStart = idx[s] + 1;
          if (checkClosest) {
            guardPhiNear(gphi[s], c.t.rods[rodAt(s, idx[s])].phi, dPhi, c);
            if (phiLessF(gphi[s], c.t.rods[rodAt(s, idx[s])].phi))
              posStart = idx[s];
            else
              negStart = idx[s];
          }
          auto overlap = [&](int rod) MKFITDEV_NAV_LAMBDA_INLINE {  // barrelUtil::overlap
            constexpr float phiOffset = 0.00034;
            const float w = window + phiOffset;
            const float lo = gphi[s] - w, hi = gphi[s] + w;
            NavRod const& r = c.t.rods[rod];
            guardPhiNear(r.phiSpanHi, lo, dPhi, c);
            guardPhiNear(hi, r.phiSpanLo, dPhi, c);
            return !(phiLessF(r.phiSpanHi, lo) | phiLessF(hi, r.phiSpanLo));
          };
          const int quarter = B.nRod[s] / 4;
          for (int i = negStart; i >= negStart - quarter; i--)
            if (!overlap(rodAt(s, i)) || !rodAdd<kRods>(rodAt(s, i), c, res))
              break;
          for (int i = posStart; i < posStart + quarter; i++)
            if (!overlap(rodAt(s, i)) || !rodAdd<kRods>(rodAt(s, i), c, res))
              break;
        };
        searchNeighbors(cs, closestRes, false);
        NavSlot nextResSlot(c);
        NavGroups& nextRes = *nextResSlot.p;
        searchNeighbors(os, nextRes, true);
        const int side = barrelSide(cg, c.along);
        orderAndMerge(closestRes, nextRes, rodsRes, cs, side, c);
      }();
    if constexpr (kPart == kNavPartRods) {
      result = rodsRes;
      return true;
    }
    NavSlot ringsResSlot(c);
    NavGroups& ringsRes = *ringsResSlot.p;
    const int zs = st.t.x.z < 0 ? 0 : 1;
    if constexpr (kRods == kNavRodsPixel) {
      if (B.nRing[zs] > 0)
        c.overflow |= kNavUnsupported;  // tilted rings in a pixel layer: not expected, the host searches it
    } else {
      for (int i = 0; i < B.nRing[zs]; ++i)
        ringGroups(c.t.rings[B.ring0[zs] + i], c, ringsRes);
    }
    if constexpr (kPart == kNavPartRings) {
      result = ringsRes;  // the caller appends it to the rods part (pushGroup)
      return true;
    }
    result = rodsRes;
    for (int i = 0; i < ringsRes.n; ++i)
      pushGroup(result, ringsRes.g[i], c);
    return true;
  }

  // the search of one layer (kind 2 barrel, 1 pixel double disk, 0 OT endcap) with its det tests answered
  // from the call's memo (c.memo / c.memoDet, kNavMemo entries). Each pass runs the transliterated search; a det without
  // a result gets a placeholder and is requested, the requested dets are tested here (the only det-test site of the
  // device code: one inlined copy per kernel, no calls) and the search runs again, until a pass needs no new test. That
  // last pass is the host search with every det test exact; a full memo goes to the host (kNavOvfDetTests).
  // kKind: 0 OT endcap, 1 pixel double disk, 2 barrel (any rods), 3 barrel with stacked (OT) rods, 4 barrel with pixel rods,
  // kKind 5 / 6: the rods / the tilted rings of a barrel layer with stacked rods (detTests0: the rods part's det tests)
  template <int kKind>
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool memoLayerSearch(
      int first, int count, NavCtx& c, NavGroups& res, int detTests0 = 0) {
    c.nMemo = c.nDone = 0;
    while (true) {
      c.detTests = detTests0;
      c.overflow = 0;
      c.arenaTop = 0;
      c.incomplete = false;
      res.n = 0;
      bool ok;
      if constexpr (kKind == 2)
        ok = barrelLayerSearch(first, c, res);
      else if constexpr (kKind == 3)
        ok = barrelLayerSearch<kNavRodsStacked>(first, c, res);
      else if constexpr (kKind == 4)
        ok = barrelLayerSearch<kNavRodsPixel>(first, c, res);
      else if constexpr (kKind == 5)
        ok = barrelLayerSearch<kNavRodsStacked, kNavPartRods>(first, c, res);
      else if constexpr (kKind == 6)
        ok = barrelLayerSearch<kNavRodsStacked, kNavPartRings>(first, c, res);
      else if constexpr (kKind == 1)
        ok = pixEndcapSearch(first, count, c, res);
      else
        ok = endcapLayerSearch(first, count, c, res);
      if (!c.incomplete)
        return ok;
      if (c.nMemo == c.nDone) {  // the memo is full: nothing new could be requested
        c.overflow |= kNavOvfDetTests;
        return false;
      }
      for (; c.nDone < c.nMemo; ++c.nDone)
        c.memo[c.nDone] =
            detTest(*c.st, c.t.dets[c.memoDet[c.nDone]].plane, c.along, c.maxSagitta, c.minTolerance2, c.nSigma);
    }
  }

}  // namespace mkfitdev::navdev

#endif
