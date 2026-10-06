#ifndef RecoTracker_MkFitAlpaka_src_alpaka_clean_CleanFunctions_h
#define RecoTracker_MkFitAlpaka_src_alpaka_clean_CleanFunctions_h

// Host+device transliterations of the LST-step quality filter and duplicate-cleaner pair decision
// (MkFitCMS/src/MkStdSeqs.cc, MkFitCMS/src/runFunctions.cc). Same operations, same order, float arithmetic.

#include <cmath>
#include <cstdint>
#include <cstring>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace mkfitdev::clean {

  // MkFitCore Const (Config.h) and the Config:: globals of the duplicate cleaners (Config.cc): one copy in
  // interface/math/Config.h.
  constexpr float kPI = ::mkfitdev::Const::PI;
  constexpr float kTwoPI = ::mkfitdev::Const::TwoPI;
  constexpr float kMaxdcth = ::mkfitdev::Config::maxdcth;
  constexpr float kMaxdphi = ::mkfitdev::Config::maxdphi;
  constexpr float kMaxcthOb = ::mkfitdev::Config::maxcth_ob;
  constexpr float kMaxcthFw = ::mkfitdev::Config::maxcth_fw;
  constexpr float kMaxd1pt = ::mkfitdev::Config::maxd1pt;

  // IterationConfig dc_* values of the iteration (from the iteration JSON).
  // pixelPriority: phase2:clean_duplicates_sharedhits_pixelpriority instead of
  // phase1:clean_duplicates_sharedhits_pixelseed; pixelLayers = TrackerInfo is_pixel() per layer (bit l of word l / 64).
  struct DupCleanParams {
    float fracSharedHits;
    float drthCentral;
    float drthObarrel;
    float drthForward;
    int pixelPriority = 0;
    uint64_t pixelLayers[4] = {0, 0, 0, 0};
  };

  inline DupCleanParams makeDupCleanParams(const float dc[4], const uint64_t* pixelPriorityLayers) {
    DupCleanParams p{dc[0], dc[1], dc[2], dc[3]};
    if (pixelPriorityLayers) {
      p.pixelPriority = 1;
      for (int w = 0; w < 4; ++w)
        p.pixelLayers[w] = pixelPriorityLayers[w];
    }
    return p;
  }

  ALPAKA_FN_HOST_ACC inline bool isPixelLayer(DupCleanParams const& p, int layer) {
    return layer >= 0 && layer < 256 && ((p.pixelLayers[layer >> 6] >> (layer & 63)) & 1u);
  }

  // clean_duplicates_sharedhits_pixelseed_impl<true> (MkStdSeqs.cc): hasPixel = >= 2 found hits on pixel layers
  ALPAKA_FN_HOST_ACC inline bool hasPixelHits(TrackSoAConstView t, int i, DupCleanParams const& p) {
    const auto& h = t[i].hits().hot;
    const int n = t[i].nTotalHits();
    int npix = 0;
    for (int a = 0; a < n; ++a)
      if (h[a].index >= 0 && isPixelLayer(p, h[a].layer))
        ++npix;
    return npix >= 2;
  }

  // sharePixelHit of the same function: a valid pixel hit of track i equal (index and layer) to any hit of track j
  ALPAKA_FN_HOST_ACC inline bool sharePixelHit(TrackSoAConstView t, int i, int j, DupCleanParams const& p) {
    const auto& h1 = t[i].hits().hot;
    const auto& h2 = t[j].hits().hot;
    const int n1 = t[i].nTotalHits(), n2 = t[j].nTotalHits();
    for (int a = 0; a < n1; ++a) {
      if (h1[a].index < 0 || !isPixelLayer(p, h1[a].layer))
        continue;
      for (int b = 0; b < n2; ++b)
        if (h2[b].index == h1[a].index && h2[b].layer == h1[a].layer)
          return true;
    }
    return false;
  }

  // mkfit::squashPhiMinimal (Hit.h), mkfit::isFinite: interface/math/MathUtils.h
  using ::mkfitdev::squashPhiMinimal;

  ALPAKA_FN_HOST_ACC inline bool isFiniteBits(float x) { return ::mkfitdev::isFinite(x); }

  // packed 32-bit word of a HitOnTrack (index:24, layer:8)
  ALPAKA_FN_HOST_ACC inline uint32_t hotKey(::mkfitdev::HitOnTrack h) {
    return __builtin_bit_cast(uint32_t, h);  // not a union pun
  }

  // TrackBase::hasNanNSillyValues: negative diagonal or non-finite element of the 6x6 symmetric error matrix.
  ALPAKA_FN_HOST_ACC inline bool hasNanNSillyValues(const float* err) {
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j <= i; ++j) {
        const float e = err[i * (i + 1) / 2 + j];
        if ((i == j && e < 0) || !isFiniteBits(e))
          return true;
      }
    }
    return false;
  }

  // StdSeq::qfilter_n_hits_pixseed
  ALPAKA_FN_HOST_ACC inline bool qfilterNHitsPixseed(int nFoundHits, int minHitsQF) { return nFoundHits >= minHitsQF; }

  // The LST-step pre- and post-backward-fit filter as run_OneIteration composes it when the backward fit runs:
  // m_{pre,post}_bkfit_filter (= phase1:qfilter_n_hits_pixseed) && qfilter_nan_n_silly.
  ALPAKA_FN_HOST_ACC inline bool lstStepCandFilter(int nFoundHits, const float* err, int minHitsQF) {
    return qfilterNHitsPixseed(nFoundHits, minHitsQF) && !hasNanNSillyValues(err);
  }

  // One pair decision of StdSeq::clean_duplicates_sharedhits_pixelseed (p.pixelPriority = 0) or
  // clean_duplicates_sharedhits_pixelpriority (p.pixelPriority = 1) for itrack = i < jtrack = j.
  // ct = 1/tan(theta) precomputed per track as MkFitCore; hasPix = hasPixelHits per track (pixelPriority only;
  // MkFitCore also computes it once per track). Returns the index to flag as duplicate, or -1.
  // Decisions read only immutable inputs, so the union of flagged indices is independent of pair order.
  ALPAKA_FN_HOST_ACC inline int dupPairDecision(
      TrackSoAConstView t, const float* ct, const int8_t* hasPix, int i, int j, DupCleanParams const& p) {
    if (t[i].label() == t[j].label())
      return -1;

    const float phi1 = t[i].params().v[4];
    const float invpt1 = t[i].params().v[3];
    const float ctheta1 = ct[i];

    const float dctheta = std::abs(ct[j] - ctheta1);
    if (dctheta > kMaxdcth)
      return -1;

    const float dphi = std::abs(squashPhiMinimal(phi1 - t[j].params().v[4]));
    if (dphi > kMaxdphi)
      return -1;

    // pixelpriority: two tracks with pixel hits but no shared pixel hit come from different pixel seeds
    bool hasPix1 = false, hasPix2 = false;
    if (p.pixelPriority) {
      hasPix1 = hasPix[i] != 0;
      hasPix2 = hasPix[j] != 0;
      if (hasPix1 && hasPix2 && !sharePixelHit(t, i, j, p))
        return -1;
    }
    // pixelOrScoreWins(i, j): with pixelPriority a track with pixel hits beats a pixel-less one, else the higher score
    const bool iWins = (p.pixelPriority && hasPix1 != hasPix2) ? hasPix1 : (t[i].score() > t[j].score());

    float maxdRSquared = p.drthCentral * p.drthCentral;
    if (std::abs(ctheta1) > kMaxcthFw)
      maxdRSquared = p.drthForward * p.drthForward;
    else if (std::abs(ctheta1) > kMaxcthOb)
      maxdRSquared = p.drthObarrel * p.drthObarrel;
    const float dr2 = dphi * dphi + dctheta * dctheta;
    if (dr2 < maxdRSquared) {
      // keep track with best score
      return iWins ? j : i;
    }

    if (std::abs(t[j].params().v[3] - invpt1) > kMaxd1pt)
      return -1;

    int sharedCount = 0;
    const int nF1 = t[i].nFoundHits(), nF2 = t[j].nFoundHits();
    const int minFoundHits = nF1 < nF2 ? nF1 : nF2;
    const auto& h1 = t[i].hits().hot;
    const auto& h2 = t[j].hits().hot;
    const int n1 = t[i].nTotalHits(), n2 = t[j].nTotalHits();
    const float fraction = p.fracSharedHits;

    // MkFitCore counts every (a, b) pair of equal valid hits (its in-loop 'continue's are no-ops) and sets sharedFirst
    // when the first entries of both tracks are valid and equal. Same counts here: the packed (index, layer) words are
    // equal iff both fields are, and a valid hit (index >= 0) can never equal an invalid entry, so track 2 needs no
    // validity test. sharedCount only grows, so stopping once the threshold is met is exact.
    const int sharedFirst =
        (n1 > 0 && n2 > 0 && h1[0].index >= 0 && h2[0].index >= 0 && hotKey(h1[0]) == hotKey(h2[0])) ? 1 : 0;
    for (int a = 0; a < n1; ++a) {
      if (h1[a].index < 0)
        continue;
      const uint32_t key = hotKey(h1[a]);
      int c = 0;
      for (int b = 0; b < n2; ++b)
        c += (hotKey(h2[b]) == key) ? 1 : 0;
      sharedCount += c;
      if ((sharedCount - sharedFirst) >= ((minFoundHits - sharedFirst) * fraction))
        break;
    }

    if ((sharedCount - sharedFirst) >= ((minFoundHits - sharedFirst) * fraction)) {
      return iWins ? j : i;
    }
    return -1;
  }

}  // namespace mkfitdev::clean

#endif
