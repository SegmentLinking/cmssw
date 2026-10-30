#ifndef RecoTracker_MkFitAlpaka_src_alpaka_clean_CleanFunctions_h
#define RecoTracker_MkFitAlpaka_src_alpaka_clean_CleanFunctions_h

// Host+device transliterations of the LST-step quality filter and duplicate-cleaner pair decision
// (MkFitCMS/src/MkStdSeqs.cc, MkFitCMS/src/runFunctions.cc). Same operations, same order, float arithmetic.

#include <cmath>
#include <cstdint>
#include <cstring>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/DupCleanParams.h"
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

  ALPAKA_FN_HOST_ACC inline bool isPixelLayer(DupCleanParams const& params, int layer) {
    return layer >= 0 && layer < 256 && ((params.pixelLayers[layer >> 6] >> (layer & 63)) & 1u);
  }

  // clean_duplicates_sharedhits_pixelseed_impl<true> (MkStdSeqs.cc): hasPixel = >= 2 found hits on pixel layers
  ALPAKA_FN_HOST_ACC inline bool hasPixelHits(TrackSoAConstView tracks, int i, DupCleanParams const& params) {
    const auto& hots = tracks[i].hits().hot;
    const int nHits = tracks[i].nTotalHits();
    int npix = 0;
    for (int ih = 0; ih < nHits; ++ih)
      if (hots[ih].index >= 0 && isPixelLayer(params, hots[ih].layer))
        ++npix;
    return npix >= 2;
  }

  // sharePixelHit of the same function: a valid pixel hit of track i equal (index and layer) to any hit of track j
  ALPAKA_FN_HOST_ACC inline bool sharePixelHit(TrackSoAConstView tracks, int i, int j, DupCleanParams const& params) {
    const auto& h1 = tracks[i].hits().hot;
    const auto& h2 = tracks[j].hits().hot;
    const int n1 = tracks[i].nTotalHits(), n2 = tracks[j].nTotalHits();
    for (int ih1 = 0; ih1 < n1; ++ih1) {
      if (h1[ih1].index < 0 || !isPixelLayer(params, h1[ih1].layer))
        continue;
      for (int ih2 = 0; ih2 < n2; ++ih2)
        if (h2[ih2].index == h1[ih1].index && h2[ih2].layer == h1[ih1].layer)
          return true;
    }
    return false;
  }

  // mkfit::squashPhiMinimal (Hit.h), mkfit::isFinite: interface/math/MathUtils.h
  using ::mkfitdev::isFinite;
  using ::mkfitdev::squashPhiMinimal;

  // packed 32-bit word of a HitOnTrack (index:24, layer:8)
  ALPAKA_FN_HOST_ACC inline uint32_t hotKey(::mkfitdev::HitOnTrack hot) {
    return __builtin_bit_cast(uint32_t, hot);  // not a union pun
  }

  // TrackBase::hasNanNSillyValues: interface/cands/CandSeedOps.h
  using ::mkfitdev::hasNanNSillyValues;

  // StdSeq::qfilter_n_hits_pixseed
  ALPAKA_FN_HOST_ACC inline bool qfilterNHitsPixseed(int nFoundHits, int minHitsQF) { return nFoundHits >= minHitsQF; }

  // The LST-step pre- and post-backward-fit filter as run_OneIteration composes it when the backward fit runs:
  // m_{pre,post}_bkfit_filter (= phase1:qfilter_n_hits_pixseed) && qfilter_nan_n_silly.
  ALPAKA_FN_HOST_ACC inline bool lstStepCandFilter(int nFoundHits, const float* err, int minHitsQF) {
    return qfilterNHitsPixseed(nFoundHits, minHitsQF) && !hasNanNSillyValues(err);
  }

  // One pair decision of StdSeq::clean_duplicates_sharedhits_pixelseed (params.pixelPriority = 0) or
  // clean_duplicates_sharedhits_pixelpriority (params.pixelPriority = 1) for itrack = i < jtrack = j.
  // ct = 1/tan(theta) precomputed per track as MkFitCore; hasPix = hasPixelHits per track (pixelPriority only;
  // MkFitCore also computes it once per track). Returns the index to flag as duplicate, or -1.
  // Decisions read only immutable inputs, so the union of flagged indices is independent of pair order.
  ALPAKA_FN_HOST_ACC inline int dupPairDecision(
      TrackSoAConstView tracks, const float* ct, const int8_t* hasPix, int i, int j, DupCleanParams const& params) {
    if (tracks[i].label() == tracks[j].label())
      return -1;

    const float phi1 = tracks[i].params().v[4];
    const float invpt1 = tracks[i].params().v[3];
    const float ctheta1 = ct[i];

    const float dctheta = std::abs(ct[j] - ctheta1);
    if (dctheta > kMaxdcth)
      return -1;

    const float dphi = std::abs(squashPhiMinimal(phi1 - tracks[j].params().v[4]));
    if (dphi > kMaxdphi)
      return -1;

    // pixelpriority: two tracks with pixel hits but no shared pixel hit come from different pixel seeds
    bool hasPix1 = false, hasPix2 = false;
    if (params.pixelPriority) {
      hasPix1 = hasPix[i] != 0;
      hasPix2 = hasPix[j] != 0;
      if (hasPix1 && hasPix2 && !sharePixelHit(tracks, i, j, params))
        return -1;
    }
    // pixelOrScoreWins(i, j): with pixelPriority a track with pixel hits beats a pixel-less one, else the higher score
    const bool iWins = (params.pixelPriority && hasPix1 != hasPix2) ? hasPix1 : (tracks[i].score() > tracks[j].score());

    float maxdRSquared = params.drthCentral * params.drthCentral;
    if (std::abs(ctheta1) > kMaxcthFw)
      maxdRSquared = params.drthForward * params.drthForward;
    else if (std::abs(ctheta1) > kMaxcthOb)
      maxdRSquared = params.drthObarrel * params.drthObarrel;
    const float dr2 = dphi * dphi + dctheta * dctheta;
    if (dr2 < maxdRSquared) {
      // keep track with best score
      return iWins ? j : i;
    }

    if (std::abs(tracks[j].params().v[3] - invpt1) > kMaxd1pt)
      return -1;

    int sharedCount = 0;
    const int nF1 = tracks[i].nFoundHits(), nF2 = tracks[j].nFoundHits();
    const int minFoundHits = nF1 < nF2 ? nF1 : nF2;
    const auto& h1 = tracks[i].hits().hot;
    const auto& h2 = tracks[j].hits().hot;
    const int n1 = tracks[i].nTotalHits(), n2 = tracks[j].nTotalHits();
    const float fraction = params.fracSharedHits;

    // MkFitCore counts every (ih1, ih2) pair of equal valid hits (its in-loop 'continue's are no-ops) and sets sharedFirst
    // when the first entries of both tracks are valid and equal. Same counts here: the packed (index, layer) words are
    // equal iff both fields are, and a valid hit (index >= 0) can never equal an invalid entry, so track 2 needs no
    // validity test. sharedCount only grows, so stopping once the threshold is met is exact.
    const int sharedFirst =
        (n1 > 0 && n2 > 0 && h1[0].index >= 0 && h2[0].index >= 0 && hotKey(h1[0]) == hotKey(h2[0])) ? 1 : 0;
    for (int ih1 = 0; ih1 < n1; ++ih1) {
      if (h1[ih1].index < 0)
        continue;
      const uint32_t key = hotKey(h1[ih1]);
      int nMatch = 0;
      for (int ih2 = 0; ih2 < n2; ++ih2)
        nMatch += (hotKey(h2[ih2]) == key) ? 1 : 0;
      sharedCount += nMatch;
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
