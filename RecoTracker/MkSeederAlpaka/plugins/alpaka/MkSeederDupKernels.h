#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDupKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDupKernels_h

// The kernels of the duplicate cleanings (MkSeederClean.h); the decision rule is described at dupStep(). Call 1 runs on
// the seeds of the kept seeds' fits (in the order of the quads, then the triplet seeds, each by label; the probe's
// dropped and the failed fits are out) and gives the post step's input; call 2 runs on the post step's kept seeds and
// gives the final seeds. Hits are identified by their bin-order row (MkSeederHitsSoA).

#include <cmath>
#include <cstdint>
#include <cstring>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederClean.h"
#include "MkSeederCleanKernels.h"
#include "MkSeederDeviceTypes.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::dup {

  using SortKeysView = ::mkseeder::MkSeederSortKeysSoA::View;
  using DupInputsView = ScratchView<::mkseeder::MkSeederDupInputsSoA>;
  using DupSeedsView = ScratchView<::mkseeder::MkSeederDupSeedsSoA>;
  using ::mkseeder::MkSeederItemLists;

  // a duplicate cleaning's inputs and outputs with its state (nK + 1 rows per input / seed); per seed its hits,
  // helix, merged hits and refit; per hit, per no-share bin (nBins) and per owner the seeds by rank
  struct DupState : DupInputs {
    mutable DupInputsView inputs;
    mutable DupSeedsView seeds;
    mutable SeedHitsView seedHits;
    mutable SeedHelicesView seedHelices;
    mutable SeedHitsView merged;  // per seed: the merged hits to refit (nHits = 0: none)
    mutable SeedStatesView refits;
    MkSeederItemLists *hitSeeds, *binSeeds, *ownerSeeds;
  };
  // the kept sharers a seed collects on the stack; more: a merge over the hits' lists (the same order)
#ifndef MKSEEDER_DUP_MAX_CANDIDATES
#define MKSEEDER_DUP_MAX_CANDIDATES 160
#endif
  constexpr int kMaxDupCandidates = MKSEEDER_DUP_MAX_CANDIDATES;

  // a seed's helix (a row of MkSeederSeedHelicesSoA)
  struct SeedHelixParameters {
    float qpt, cot, phi0, z0, eta, v_qpt, v_cot;
  };
  template <typename THelices>
  ALPAKA_FN_ACC inline SeedHelixParameters loadHelix(const THelices &h, uint32_t i) {
    const auto row = h[i];
    return {row.qOverPt(), row.cot(), row.phi0(), row.z0(), row.eta(), row.varQOverPt(), row.varCot()};
  }

  template <typename TAcc>
  ALPAKA_FN_ACC inline bool compatW(
      const TAcc &acc, const SeedHelixParameters &a, const SeedHelixParameters &b, float s2, float wphi, float wz) {
    const float dq = a.qpt - b.qpt, dc = a.cot - b.cot;
    float dphi = a.phi0 - b.phi0;
    dphi -= kTwoPi * alpaka::math::round(acc, dphi / kTwoPi);
    return dq * dq < s2 * (a.v_qpt + b.v_qpt) && dc * dc < s2 * (a.v_cot + b.v_cot) &&
           alpaka::math::abs(acc, dphi) < wphi && alpaka::math::abs(acc, a.z0 - b.z0) < wz;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline float maxPt(const TAcc &acc, const SeedHelixParameters &a, const SeedHelixParameters &b) {
    return 1.f / std::max(alpaka::math::abs(acc, a.qpt), alpaka::math::abs(acc, b.qpt));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool compat(const TAcc &acc,
                                   const DupParameters &C,
                                   const SeedHelixParameters &a,
                                   const SeedHelixParameters &b) {
    if (C.highPtThreshold > 0) {
      const float pt = maxPt(acc, a, b);
      if (pt > C.highPtThreshold)
        return compatW(acc,
                       a,
                       b,
                       C.maxSigma2,
                       std::min(C.dphi, C.highPtMaxDPhi0 + C.highPtMaxDPhi1 / pt),
                       std::min(C.dz, C.highPtMaxDz));
    }
    return compatW(acc, a, b, C.maxSigma2, C.dphi, C.dz);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool compatNs(const TAcc &acc,
                                     const DupParameters &C,
                                     const SeedHelixParameters &x,
                                     const SeedHelixParameters &y) {
    if (C.highPtThreshold > 0) {
      const float pt = maxPt(acc, x, y);
      if (pt > C.highPtThreshold)
        return compatW(acc,
                       x,
                       y,
                       C.noShareMaxSigma2,
                       std::min(C.noShareDphi, C.highPtMaxDPhi0 + C.highPtMaxDPhi1 / pt),
                       std::min(C.noShareDz, C.highPtMaxDz));
    }
    return compatW(acc, x, y, C.noShareMaxSigma2, C.noShareDphi, C.noShareDz);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline int ipRaw(const TAcc &acc, const DupParameters &C, const SeedHelixParameters &p) {
    return (int)alpaka::math::floor(acc, (p.phi0 + kPi) / C.phiBinWidth);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline int ieOf(const TAcc &acc, const DupParameters &C, const SeedHelixParameters &p) {
    return (int)alpaka::math::floor(acc, p.eta / C.deta);
  }
  // the no-share bin (dphi_bin, deta_bin) around p: eta bin x nbphi + phi bin
  template <typename TAcc>
  ALPAKA_FN_ACC inline int64_t hostBinOf(
      const TAcc &acc, const DupParameters &C, const SeedHelixParameters &p, int db, int de) {
    const int ip = (ipRaw(acc, C, p) + db + 2 * C.nPhiBins) % C.nPhiBins;
    return (int64_t(ieOf(acc, C, p)) + de) * C.nPhiBins + ip;
  }
  // its list: the bins of the eta range [ieMin, ieMin + nIe); outside, the overflow bin (the last), whose seeds are
  // filtered by their host bin
  ALPAKA_FN_ACC inline uint32_t binIndex(const DupParameters &C, int64_t hostBin) {
    const int64_t b = hostBin - int64_t(C.firstEtaBin) * C.nPhiBins, nb = int64_t(C.nEtaBins) * C.nPhiBins;
    return uint32_t(b >= 0 && b < nb ? b : nb);
  }
  ALPAKA_FN_ACC inline bool isOverflowBin(const DupParameters &C, uint32_t b) {
    return b == uint32_t(C.nEtaBins) * uint32_t(C.nPhiBins);
  }
  // on_circle(j, i, tol): every hit of seed i that seed j lacks lies within tol of j's circle through its first,
  // middle and last hit (double)
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool onCircle(const TAcc &acc, const DupState &v, uint32_t j, uint32_t i, float tol) {
    const int nj = v.seedHits.nHits(j), ni = v.seedHits.nHits(i);
    const auto dj = v.seedHits.hitRows(j), di = v.seedHits.hitRows(i);
    if (nj < 3)
      return true;
    const double Ax = v.hits.x(dj(0)), Ay = v.hits.y(dj(0));
    const double bx = double(v.hits.x(dj(nj / 2))) - Ax, by = double(v.hits.y(dj(nj / 2))) - Ay;
    const double cx = double(v.hits.x(dj(nj - 1))) - Ax, cy = double(v.hits.y(dj(nj - 1))) - Ay;
    const double d = 2 * (bx * cy - by * cx);
    if (alpaka::math::abs(acc, d) < 1e-12)
      return true;
    const double b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
    const double xc = (b2 * cy - c2 * by) / d, yc = (c2 * bx - b2 * cx) / d,
                 R = alpaka::math::sqrt(acc, xc * xc + yc * yc);
    for (int k = 0; k < ni; ++k) {
      bool shared = false;
      for (int q = 0; q < nj; ++q)
        shared |= dj(q) == di(k);
      if (shared)
        continue;
      const double dx = double(v.hits.x(di(k))) - Ax - xc, dy = double(v.hits.y(di(k))) - Ay - yc;
      if (alpaka::math::abs(acc, alpaka::math::sqrt(acc, dx * dx + dy * dy) - R) > tol)
        return false;
    }
    return true;
  }

  // ---- call 1: the seeds from the kept seeds
  // per probe item: its label dropped
  class KernelDropProbeLabels {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v) const {
      const uint32_t np = v.cleanCounters->nProbes < v.probeCap ? v.cleanCounters->nProbes : v.probeCap;
      for (uint32_t p : cms::alpakatools::uniform_elements(acc, np))
        if (!v.probe.isKept(p))
          v.inputs.probeDropped(v.probe.label(p)) = 1;
    }
  };
  // per label k: kept by the probe; a quad / triplet seed with a working fit
  class KernelSetKeptSeedFlags {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v) const {
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, v.nK)) {
        const bool live = !v.inputs.probeDropped(k);
        const bool ok = live && v.fits.fitOk(k);
        const bool four = v.kept.nHits(k) == 4;
        v.inputs.isLive(k) = live, v.inputs.isQuadSeed(k) = ok && four, v.inputs.isTripletSeed(k) = ok && !four;
      }
    }
  };
  // per label k with a seed: the seed (index i: the quads first), its hits, helix, flags, rank key
  class KernelBuildSeedItemsFromKept {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v, SortKeysView keys) const {
      const uint32_t n4 = v.inputs.quadSeedPosition(v.nK);
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, v.nK)) {
        if (k == 0)
          v.counters->nSeeds = n4 + v.inputs.tripletSeedPosition(v.nK);
        const bool s4 = v.inputs.isQuadSeed(k), s3 = v.inputs.isTripletSeed(k);
        if (!s4 && !s3)
          continue;
        const uint32_t i = s4 ? v.inputs.quadSeedPosition(k) : n4 + v.inputs.tripletSeedPosition(k);
        const auto K = v.kept[k];
        v.seeds.input(i) = k;
        const int nh = K.nHits();
        auto S = v.seedHits[i];
        S.nHits() = nh;
        int tier = 0;
        for (int q = 0; q < nh; ++q) {
          S.hitLayers()(q) = K.layers()(q);
          S.hitRows()(q) = K.hitRows()(q);
          tier += !v.layerGeometry[K.layers()(q)].isPixel();
        }
        const SeedHelixParameters h = loadHelix(v.helices, k);
        auto H = v.seedHelices[i];
        H.qOverPt() = h.qpt, H.cot() = h.cot, H.phi0() = h.phi0, H.z0() = h.z0, H.eta() = h.eta,
        H.varQOverPt() = h.v_qpt;
        H.varCot() = h.v_cot;
        const uint8_t tf = K.tripletFailedMlp() ? 1 : 0;
        S.tripletFailedMlp() = tf;
        v.seeds.score(i) = K.score();
        keys.keyHigh(i) = uint64_t(tf) << 7 | uint64_t(tier);
        keys.keyLow(i) = clean::fkey(K.score());
        keys.index(i) = i;
      }
    }
  };

  // ---- call 2: the seeds from the post step's kept seeds (all in the first group: isQuadSeed, quadSeedPosition)
  class KernelSetPostSeedFlags {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v) const {
      const uint32_t n = *v.postCount;
      for (uint32_t p : cms::alpakatools::uniform_elements(acc, v.nK))
        v.inputs.isQuadSeed(p) = p < n && v.postHits.isKept(p) ? 1 : 0;
    }
  };
  class KernelBuildSeedItemsFromPost {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v, SortKeysView keys) const {
      for (uint32_t p : cms::alpakatools::uniform_elements(acc, v.nK)) {
        if (p == 0)
          v.counters->nSeeds = v.inputs.quadSeedPosition(v.nK);
        if (!v.inputs.isQuadSeed(p))
          continue;
        const uint32_t i = v.inputs.quadSeedPosition(p);
        const auto P = v.postHits[p];
        v.seeds.input(i) = p;
        const int nh = P.nHits();
        auto S = v.seedHits[i];
        S.nHits() = nh;
        int tier = 0;
        for (int q = 0; q < nh; ++q) {
          S.hitLayers()(q) = P.hitLayers()(q);
          S.hitRows()(q) = P.hitRows()(q);
          tier += !v.layerGeometry[P.hitLayers()(q)].isPixel();
        }
        const SeedHelixParameters h = loadHelix(v.helices, p);
        auto H = v.seedHelices[i];
        H.qOverPt() = h.qpt, H.cot() = h.cot, H.phi0() = h.phi0, H.z0() = h.z0, H.eta() = h.eta,
        H.varQOverPt() = h.v_qpt;
        H.varCot() = h.v_cot;
        S.tripletFailedMlp() = 0;
        const float sc = v.postQuality.score(p);
        v.seeds.score(i) = sc;
        keys.keyHigh(i) = uint64_t(tier);
        keys.keyLow(i) = clean::fkey(sc);
        keys.index(i) = i;
      }
    }
  };

  // per rank: the maps, the state, the hits' and the bins' counts
  class KernelDupFields {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  DupState v,
                                  ::mkseeder::MkSeederSortKeysSoA::ConstView rankOrder) const {
      const uint32_t n = v.counters->nSeeds;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, v.nK)) {
        if (r >= n)
          break;
        const uint32_t i = rankOrder.index(r);
        v.seeds.seedOfRank(r) = i, v.seeds.rankOfSeed(i) = r;
        v.seeds.state(i) = 0, v.seeds.owner(i) = -1;
        const auto S = v.seedHits[i];
        for (int q = 0; q < S.nHits(); ++q)
          v.hitSeeds->count(acc, S.hitRows()(q));
        const SeedHelixParameters p = loadHelix(v.seedHelices, i);
        uint32_t b = 0xffffffffu;
        if (v.parameters.noShare && alpaka::math::abs(acc, p.eta) >= v.parameters.noShareEta - v.parameters.deta) {
          b = binIndex(v.parameters, hostBinOf(acc, v.parameters, p, 0, 0));
          v.binSeeds->count(acc, b);
        }
        v.seeds.binKey(i) = b;
      }
    }
  };
  class KernelDupFill {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v) const {
      const uint32_t n = v.counters->nSeeds;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, v.nK)) {
        if (r >= n)
          break;
        const uint32_t i = v.seeds.seedOfRank(r);
        const auto S = v.seedHits[i];
        for (int q = 0; q < S.nHits(); ++q)
          v.hitSeeds->fill(acc, S.hitRows()(q), r);
        if (v.seeds.binKey(i) != 0xffffffffu)
          v.binSeeds->fill(acc, v.seeds.binKey(i), r);
      }
    }
  };

  // The greedy's decision for the undecided seed of rank r. The seeds are ranked by (demote, tier, score), ties by
  // index; a seed is decided once every better-ranked seed that shares a hit with it, or lies in its 3 x 3 no-share
  // bins, is (in rounds, then the completion in one block). A seed is removed if a kept sharer owns it (the first
  // compatible one by index), else if a kept seed in the no-share bins does (searched in (dphi, deta, kept) order,
  // with the on-circle test). The merges: per owner the removed seeds' hits in rank order, sorted by r^2 + z^2, are
  // refit (call 2 with the beam-line state); then the kept seeds follow: call 1 the post step's input, call 2 the
  // final seeds.
  // Returns false when decided (kept or removed, with its owner),
  // true when a better-ranked seed that could own it is still undecided (one that is not compatible with it can never
  // own it, whether kept or not: it does not block). The states only go from undecided to final, so the latest
  // states of the others (other threads' included) can be read
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool dupStep(const TAcc &acc, const DupState &v, uint32_t r) {
    const DupParameters &C = v.parameters;
    const uint32_t i = v.seeds.seedOfRank(r);
    const int nh = v.seedHits.nHits(i);
    const SeedHelixParameters a = loadHelix(v.seedHelices, i);
    const bool ns = C.noShare && alpaka::math::abs(acc, a.eta) >= C.noShareEta && (nh != 3 || C.noShareTriplets);
    // the no-share neighbours of a in the bin order (dphi, deta): f(j) per better-ranked seed j of each bin
    // until it returns true
    auto forNoShareNeighbours = [&](auto &&f) {
      for (int bp = -1; bp <= 1; ++bp)
        for (int be = -1; be <= 1; ++be) {
          const int64_t hb = hostBinOf(acc, C, a, bp, be);
          const uint32_t b = binIndex(C, hb);
          const bool overflow = isOverflowBin(C, b);
          for (const uint32_t *e = v.binSeeds->begin(b), *e1 = v.binSeeds->end(b); e < e1; ++e) {
            const uint32_t r2 = *e;
            if (r2 >= r)
              break;
            const uint32_t j = v.seeds.seedOfRank(r2);
            if (overflow && hostBinOf(acc, C, loadHelix(v.seedHelices, j), 0, 0) != hb)
              continue;
            if (f(j))
              return true;
          }
        }
      return false;
    };
    bool blocked = false;
    uint32_t cand[kMaxDupCandidates];
    int nc = 0;
    bool over = false;
    for (int q = 0; q < nh && !blocked; ++q) {
      const uint32_t h = v.seedHits.hitRows(i)(q);
      if (C.mergedNoShare && ::mkseeder::isStrongMergedCandidate(v.hits.mergedTag(h)))
        continue;
      for (const uint32_t *e = v.hitSeeds->begin(h), *e1 = v.hitSeeds->end(h); e < e1; ++e) {
        const uint32_t r2 = *e;
        if (r2 >= r)
          break;
        const uint32_t j = v.seeds.seedOfRank(r2);
        const uint8_t s2 = v.seeds.state(j);
        if (s2 == 0) {
          if (compat(acc, C, a, loadHelix(v.seedHelices, j))) {
            blocked = true;
            break;
          }
          continue;
        }
        if (s2 == 1) {
          if (nc < kMaxDupCandidates)
            cand[nc++] = j;
          else
            over = true;
        }
      }
    }
    if (!blocked && ns)
      blocked = forNoShareNeighbours([&](uint32_t j) {
        if (v.seeds.state(j) != 0)
          return false;
        const SeedHelixParameters hj = loadHelix(v.seedHelices, j);
        return alpaka::math::abs(acc, a.eta - hj.eta) < C.deta && compatNs(acc, C, a, hj);
      });
    if (blocked)
      return true;
    // the kept sharers by index, with their shared-hit counts
    bool drop = false;
    int own = -1;
    const int minShared = nh == 3 && C.sharedTriplet > 0 ? C.sharedTriplet : C.shared;
    auto sharer = [&](uint32_t j, int shared) {
      if (shared < minShared)
        return;
      const SeedHelixParameters b = loadHelix(v.seedHelices, j);
      drop = compat(acc, C, a, b);
      if (drop)
        own = int(j);
    };
    if (!over) {
      for (int x = 1; x < nc; ++x)
        for (int y = x; y > 0 && cand[y] < cand[y - 1]; --y) {
          const uint32_t t = cand[y];
          cand[y] = cand[y - 1], cand[y - 1] = t;
        }
      for (int c = 0; c < nc && !drop;) {
        const uint32_t j = cand[c];
        int e = c;
        while (e < nc && cand[e] == j)
          ++e;
        sharer(j, e - c);
        c = e;
      }
    } else {
      // more than kMaxDupCandidates: the kept sharers by index from the lists, the next one each time
      int64_t last = -1;
      while (!drop) {
        uint32_t jMin = 0xffffffffu;
        int shared = 0;
        for (int q = 0; q < nh; ++q) {
          const uint32_t h = v.seedHits.hitRows(i)(q);
          if (C.mergedNoShare && ::mkseeder::isStrongMergedCandidate(v.hits.mergedTag(h)))
            continue;
          for (const uint32_t *e = v.hitSeeds->begin(h), *e1 = v.hitSeeds->end(h); e < e1; ++e) {
            const uint32_t r2 = *e;
            if (r2 >= r)
              break;
            const uint32_t j = v.seeds.seedOfRank(r2);
            if (v.seeds.state(j) != 1 || int64_t(j) <= last)
              continue;
            if (j < jMin)
              jMin = j, shared = 1;
            else if (j == jMin)
              ++shared;
          }
        }
        if (jMin == 0xffffffffu)
          break;
        last = jMin;
        sharer(jMin, shared);
      }
    }
    if (!drop && ns)
      forNoShareNeighbours([&](uint32_t j) {
        if (v.seeds.state(j) != 1)
          return false;
        const SeedHelixParameters b = loadHelix(v.seedHelices, j);
        if (alpaka::math::abs(acc, a.eta - b.eta) < C.deta && compatNs(acc, C, a, b) &&
            (C.noShareRes <= 0 || onCircle(acc, v, j, i, C.noShareRes))) {
          drop = true;
          own = C.noShareMerge ? int(j) : -1;
          return true;
        }
        return false;
      });
    v.seeds.owner(i) = own;
    v.seeds.state(i) = drop ? 2 : 1;
    return false;
  }  // dupStep()

  // one round of the greedy
  class KernelDupRound {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v, int round) const {
      if (round > 0 && v.counters->undecidedAfterRound[round - 1] == 0)
        return;
      const uint32_t n = v.counters->nSeeds;
      for (uint32_t r : cms::alpakatools::uniform_elements(acc, v.nK)) {
        if (r >= n)
          break;
        if (v.seeds.state(v.seeds.seedOfRank(r)) == 0 && dupStep(acc, v, r))
          alpaka::atomicAdd(acc, &v.counters->undecidedAfterRound[round], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // after the rounds (one block): the seeds they left undecided (clean::completeInOneBlock)
  class KernelDupComplete {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v, int rounds) const {
      if (rounds > 0 && v.counters->undecidedAfterRound[rounds - 1] == 0)
        return;
      clean::completeInOneBlock(acc, 0u, v.counters->nSeeds, [&](uint32_t r) {
        return v.seeds.state(v.seeds.seedOfRank(r)) == 0 && dupStep(acc, v, r);
      });
    }
  };

  // per removed seed with an owner (not demoted): the owner's count / list
  class KernelOwnerCount {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v, int fill) const {
      const uint32_t n = v.counters->nSeeds;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, v.nK)) {
        if (i >= n)
          break;
        const int j = v.seeds.owner(i);
        if (v.seeds.state(i) != 2 || j < 0 || v.seedHits.tripletFailedMlp(i))
          continue;
        if (!fill)
          v.ownerSeeds->count(acc, j);
        else
          v.ownerSeeds->fill(acc, j, v.seeds.rankOfSeed(i));
      }
    }
  };

  // dup_merge_ot (MkSeederConfig.h): whether the OT hit h of mkFit layer l of the removed seed i may be merged into its
  // owner j
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool mergeOuterTrackerHit(
      const TAcc &acc, const DupState &v, uint32_t j, uint32_t i, int l, uint32_t h) {
    const DupParameters &C = v.parameters;
    const int mode = C.mergeOT;
    if (mode & 8)
      return false;
    if ((mode & 4) && !(v.seeds.score(i) <= C.mergeOTMaxScore))
      return false;
    const SeedHelixParameters o = loadHelix(v.seedHelices, j);
    const float hx = v.hits.x(h), hy = v.hits.y(h), hz = v.hits.z(h);
    const float hr = alpaka::math::sqrt(acc, hx * hx + hy * hy);
    if ((mode & 2) && (l == 4 || l == 6 || l == 8) && v.layers.firstOTHit(l) >= 0) {
      const uint32_t row = uint32_t(v.layers.firstOTHit(l)) + (h - v.layers.firstHit(l));
      if (v.otHits.psMatches(row) == 0)
        return false;
      if (C.psCurvatureSigma > 0 && alpaka::math::abs(acc, v.otHits.psQOverPt(row) - o.qpt) >
                                        C.psCurvatureSigma * C.psCurvatureResolution * 23.f / std::max(hr, 1.f))
        return false;
    }
    if (mode & 1) {
      // the owner's circle through its first, middle and last hit; its r-z slope from its last hit on
      const int nj = v.seedHits.nHits(j);
      const auto dj = v.seedHits.hitRows(j);
      if (nj < 3)
        return true;
      const double Ax = v.hits.x(dj(0)), Ay = v.hits.y(dj(0));
      const double bx = double(v.hits.x(dj(nj / 2))) - Ax, by = double(v.hits.y(dj(nj / 2))) - Ay;
      const double cx = double(v.hits.x(dj(nj - 1))) - Ax, cy = double(v.hits.y(dj(nj - 1))) - Ay;
      const double d = 2 * (bx * cy - by * cx);
      if (alpaka::math::abs(acc, d) < 1e-12)
        return true;
      const double b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
      const double xc = (b2 * cy - c2 * by) / d, yc = (c2 * bx - b2 * cx) / d,
                   R = alpaka::math::sqrt(acc, xc * xc + yc * yc);
      const double ex = double(hx) - Ax - xc, ey = double(hy) - Ay - yc;
      const float dT = float(alpaka::math::sqrt(acc, ex * ex + ey * ey) - R);
      const float xl = v.hits.x(dj(nj - 1)), yl = v.hits.y(dj(nj - 1)), zl = v.hits.z(dj(nj - 1));
      const float rl = alpaka::math::sqrt(acc, xl * xl + yl * yl);
      const float twoR = float(2 * R);
      auto arc = [&](float r) { return twoR * float(alpaka::math::asin(acc, std::min(1.f, r / twoR))); };
      const float dz = hz - (zl + o.cot * (arc(hr) - arc(rl)));
      const float pt = std::max(0.5f, 1.f / std::max(alpaka::math::abs(acc, o.qpt), 1e-6f));
      const float sT = hr * (0.0005f + 3.2e-3f / pt), sz = 0.075f + 0.0316f / pt;
      if (!((dT / sT) * (dT / sT) + (dz / sz) * (dz / sz) < C.mergeOTSigma2))
        return false;
    }
    return true;
  }

  // per owner: its hits and those of the seeds removed for it (in rank order), by r^2 + z^2: the refit's input
  // (nHits = 0: no refit, the owner gains no hit)
  class KernelMergeOwnerHits {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v) const {
      constexpr int kSeedHitSlots = ::mkseeder::kMaxSeedHits;
      const uint32_t n = v.counters->nSeeds;
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, v.nK)) {
        auto f = v.merged[j];
        f.nHits() = 0;
        if (j >= n)
          continue;
        const uint32_t *e0 = v.ownerSeeds->begin(j), *e1 = v.ownerSeeds->end(j);
        if (e0 == e1)
          continue;
        int16_t lay[kSeedHitSlots];
        uint32_t dh[kSeedHitSlots];
        int m = 0;
        const int nj = v.seedHits.nHits(j);
        for (int q = 0; q < nj; ++q)
          lay[m] = v.seedHits.hitLayers(j)(q), dh[m] = v.seedHits.hitRows(j)(q), ++m;
        for (const uint32_t *e = e0; e < e1; ++e) {
          const uint32_t i = v.seeds.seedOfRank(*e);
          const auto S = v.seedHits[i];
          for (int q = 0; q < S.nHits() && m < kSeedHitSlots; ++q) {
            const int16_t l = S.hitLayers()(q);
            const uint32_t h = S.hitRows()(q);
            bool d = false;
            for (int o = 0; o < m && !d; ++o) {
              if (lay[o] != l)
                continue;
              d = v.parameters.merge >= 2 || dh[o] == h || v.hits.detIdInLayer(dh[o]) == v.hits.detIdInLayer(h);
            }
            if (!d && v.parameters.mergeOT && !v.layerGeometry[l].isPixel() &&
                !mergeOuterTrackerHit(acc, v, j, i, l, h))
              d = true;
            if (!d)
              lay[m] = l, dh[m] = h, ++m;
          }
        }
        if (m == nj)
          continue;
        float key[kSeedHitSlots];
        for (int q = 0; q < m; ++q) {
          const float x = v.hits.x(dh[q]), y = v.hits.y(dh[q]), z = v.hits.z(dh[q]);
          const float rr = alpaka::math::sqrt(acc, x * x + y * y);
          key[q] = rr * rr + z * z;
        }
        // stable insertion sort
        for (int a = 1; a < m; ++a)
          for (int b = a; b > 0 && key[b] < key[b - 1]; --b) {
            const float tk = key[b];
            key[b] = key[b - 1], key[b - 1] = tk;
            const int16_t tl = lay[b];
            lay[b] = lay[b - 1], lay[b - 1] = tl;
            const uint32_t td = dh[b];
            dh[b] = dh[b - 1], dh[b - 1] = td;
          }
        f.nHits() = m;
        for (int q = 0; q < m; ++q)
          f.hitLayers()(q) = lay[q], f.hitRows()(q) = dh[q];
      }
    }
  };

  // per seed: kept (the scan's input)
  class KernelFlagKeptSeeds {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v) const {
      const uint32_t n = v.counters->nSeeds;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, v.nK))
        v.seeds.isKept(i) = i < n && v.seeds.state(i) == 1 ? 1 : 0;
    }
  };

  // per kept seed: call 1 the post step's input (with its cot), call 2 the final seed; with its quality
  class KernelDupOut {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, DupState v) const {
      const uint32_t n = v.counters->nSeeds;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, v.nK)) {
        if (i == 0)
          v.counters->nOutput = v.seeds.keptPosition(v.nK);
        if (i >= n || v.seeds.state(i) != 1)
          continue;
        const uint32_t p = v.seeds.keptPosition(i);
        const auto mh = v.merged[i];
        auto out = v.outStates[p];
        bool refit;
        WorkingSeed ps;
        std::memset(&ps, 0, sizeof(WorkingSeed));
        if (v.parameters.pass == 1) {
          const uint32_t k = v.seeds.input(i);
          refit = mh.nHits() > 0 && v.refits.fitOk(i);
          const auto o = refit ? SeedStatesConstView(v.refits)[i] : v.fits[k];
          out.charge() = o.charge(), out.chi2() = o.chi2();
          for (int d = 0; d < 6; ++d)
            out.parameters()(d) = o.parameters()(d);
          for (int d = 0; d < 21; ++d)
            out.covariance()(d) = o.covariance()(d);
          out.fitOk() = 1, out.wasRefitted() = 0, out.hasBeamlineState() = 0, out.beamlineCharge() = 0;
          const auto K = v.kept[k];
          auto Q = v.outQuality[p];
          Q.score() = K.score(), Q.fakeScore() = K.fakeScore(), Q.nAmbiguous() = K.nAmbiguous();
          // the post step's r-z slope: 1 / tan(theta) with libm::tan
          ps.cot = 1.f / libm::tan(acc, o.parameters()(5));
          ps.tripletFailedMlp = v.seedHits.tripletFailedMlp(i);
          v.outHits.label(p) = v.inputs.newLabel(k);
        } else {
          const uint32_t q = v.seeds.input(i);
          refit = mh.nHits() > 0 && v.refits.wasRefitted(i);
          const auto o = refit ? SeedStatesConstView(v.refits)[i] : v.postStates[q];
          out.fitOk() = o.fitOk(), out.wasRefitted() = o.wasRefitted(), out.hasBeamlineState() = o.hasBeamlineState();
          out.charge() = o.charge(), out.chi2() = o.chi2(), out.beamlineCharge() = o.beamlineCharge();
          for (int d = 0; d < 6; ++d)
            out.parameters()(d) = o.parameters()(d), out.beamlineParameters()(d) = o.beamlineParameters()(d);
          for (int d = 0; d < 21; ++d)
            out.covariance()(d) = o.covariance()(d), out.beamlineCovariance()(d) = o.beamlineCovariance()(d);
          ps.isKept = 1;
          ps.cot = v.postHits.cot(q);
          v.outHits.label(p) = v.postHits.label(q);
          const auto K = v.postQuality[q];
          auto Q = v.outQuality[p];
          Q.score() = K.score(), Q.fakeScore() = K.fakeScore(), Q.nAmbiguous() = K.nAmbiguous();
        }
        if (refit) {
          ps.nHits = mh.nHits();
          for (int q = 0; q < ps.nHits; ++q)
            ps.layers[q] = mh.hitLayers()(q), ps.hitRows[q] = mh.hitRows()(q);
        } else {
          const auto S = v.seedHits[i];
          ps.nHits = S.nHits();
          for (int q = 0; q < ps.nHits; ++q)
            ps.layers[q] = S.hitLayers()(q), ps.hitRows[q] = S.hitRows()(q);
        }
        storeSeed(v.outHits, p, ps);
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::dup

#endif
