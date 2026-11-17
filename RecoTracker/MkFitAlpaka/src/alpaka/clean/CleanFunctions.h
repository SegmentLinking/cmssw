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
  constexpr float kMindetaP2 = ::mkfitdev::Config::mindeta_p2;
  constexpr float kMaxetaObP2 = ::mkfitdev::Config::maxeta_ob_p2;
  constexpr float kMaxetaFwP2 = ::mkfitdev::Config::maxeta_fw_p2;

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

  // A valid hit of track i (a pixel hit if pixelOnly) equal (index and layer) to any hit of track j: sharePixelHit of
  // the same function (pixelOnly), and the pairs of the pT-adaptive cleaner's hit -> tracks index
  ALPAKA_FN_HOST_ACC inline bool shareHit(
      TrackSoAConstView tracks, int i, int j, DupCleanParams const& params, bool pixelOnly) {
    const auto& h1 = tracks[i].hits().hot;
    const auto& h2 = tracks[j].hits().hot;
    const int n1 = tracks[i].nTotalHits(), n2 = tracks[j].nTotalHits();
    for (int ih1 = 0; ih1 < n1; ++ih1) {
      if (h1[ih1].index < 0 || (pixelOnly && !isPixelLayer(params, h1[ih1].layer)))
        continue;
      for (int ih2 = 0; ih2 < n2; ++ih2)
        if (h2[ih2].index == h1[ih1].index && h2[ih2].layer == h1[ih1].layer)
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

  // Per-track inputs of the pair decision, computed once per track as clean_duplicates_sharedhits_pixelseed_impl:
  // polar = cot(theta), eta for the pT-adaptive cleaner; maxDPolar = the polar window of the track, maxdcth, or
  // max(maxdcth / cosh(eta), mindeta_p2); phi = momentum phi, at the point of closest approach to the beam spot for the
  // pT-adaptive cleaner; absD0 = |d0| to the beam spot (pT-adaptive cleaner only).
  struct TrackDirection {
    float polar;
    float maxDPolar;
    float phi;
    float absD0;
  };

  ALPAKA_FN_HOST_ACC inline TrackDirection trackDirection(TrackSoAConstView tracks,
                                                          int i,
                                                          DupCleanParams const& params) {
    const float* par = tracks[i].params().v;
    TrackDirection dir;
    if (!params.ptAdaptive) {
      dir.polar = 1.f / std::tan(par[5]);
      dir.maxDPolar = kMaxdcth;
      dir.phi = par[4];
      dir.absD0 = 0.f;
      return dir;
    }
    dir.polar = getEta(par[5]);  // Track::momEta()
    const float window = kMaxdcth / std::cosh(dir.polar);
    dir.maxDPolar = (window < kMindetaP2) ? kMindetaP2 : window;  // std::max(window, mindeta_p2)
    // perpendicular to the line from the helix center to the beam spot
    const HelixCenter center = helixCenter(par, tracks[i].charge());
    const float ux = params.bsX - center.x;
    const float uy = params.bsY - center.y;
    dir.phi = center.k > 0.f ? std::atan2(ux, -uy) : std::atan2(-ux, uy);
    dir.absD0 = std::abs(d0BeamSpot(par, tracks[i].charge(), params.bsX, params.bsY));
    return dir;
  }

  // The pT-adaptive cleaner's low-pT fraction for the pair, or the standard one: below dc_lowPtRampStart of the harder
  // track if the two tracks are consistent with one particle, linear in 1/pT up to dc_lowPtRampEnd.
  ALPAKA_FN_HOST_ACC inline float sharedHitFraction(
      TrackSoAConstView tracks, const TrackDirection* dir, int i, int j, float dr2, DupCleanParams const& params) {
    const float fraction = params.fracSharedHits;
    if (!params.ptAdaptive || !(params.fracSharedHitsLowPt >= 0.f))
      return fraction;
    const float invpt1 = tracks[i].params().v[3];
    const float invpt2 = tracks[j].params().v[3];
    const float invptHard = (invpt2 < invpt1) ? invpt2 : invpt1;  // std::min(invpt1, invpt2)
    const float invptSoft = (invpt1 < invpt2) ? invpt2 : invpt1;  // std::max(invpt1, invpt2)
    const float maxDR = params.lowPtMaxDRTimesPt * invptHard;
    // MkFitCore computes absD0 only if lowPtMaxD0 > 0 and the d0 condition is off otherwise
    const bool sameParticle =
        (!params.lowPtRequireSameCharge || tracks[i].charge() == tracks[j].charge()) &&
        (params.lowPtMaxRelDiffInvPt <= 0.f || std::abs(invpt1 - invpt2) <= params.lowPtMaxRelDiffInvPt * invptSoft) &&
        (!(params.lowPtMaxD0 > 0.f) || (dir[i].absD0 < params.lowPtMaxD0 && dir[j].absD0 < params.lowPtMaxD0)) &&
        (params.lowPtMaxDRTimesPt <= 0.f || dr2 < maxDR * maxDR);
    if (!sameParticle)
      return fraction;
    // std::clamp(ramp, 0.f, 1.f)
    const float ramp = (invptHard - params.invPtRampEnd) / (params.invPtRampStart - params.invPtRampEnd);
    const float lowPtScale = (ramp < 0.f) ? 0.f : ((1.f < ramp) ? 1.f : ramp);
    return fraction + (params.fracSharedHitsLowPt - fraction) * lowPtScale;
  }

  // One pair decision of StdSeq::clean_duplicates_sharedhits_pixelseed (params.pixelPriority = 0),
  // clean_duplicates_sharedhits_pixelpriority (params.pixelPriority = 1) or
  // clean_duplicates_sharedhits_pixelpriority_ptadaptive (params.ptAdaptive = 1) for itrack = i < jtrack = j.
  // dir = trackDirection per track; hasPix = hasPixelHits per track (pixelPriority only; MkFitCore also computes both
  // once per track). Returns the index to flag as duplicate, or -1.
  // Decisions read only immutable inputs, so the union of flagged indices is independent of pair order.
  ALPAKA_FN_HOST_ACC inline int dupPairDecision(TrackSoAConstView tracks,
                                                const TrackDirection* dir,
                                                const int8_t* hasPix,
                                                int i,
                                                int j,
                                                DupCleanParams const& params) {
    if (tracks[i].label() == tracks[j].label())
      return -1;

    const float phi1 = dir[i].phi;
    const float invpt1 = tracks[i].params().v[3];
    const float polar1 = dir[i].polar;

    const float dpolar = std::abs(dir[j].polar - polar1);
    // std::max(maxdeta[itrack], maxdeta[jtrack]) of the pT-adaptive cleaner, maxdcth otherwise
    const float maxDPolar = (dir[i].maxDPolar < dir[j].maxDPolar) ? dir[j].maxDPolar : dir[i].maxDPolar;
    if (dpolar > maxDPolar)
      return -1;

    const float dphi = std::abs(squashPhiMinimal(phi1 - dir[j].phi));
    if (dphi > kMaxdphi)
      return -1;

    // pixelpriority: two tracks with pixel hits but no shared pixel hit come from different pixel seeds
    bool hasPix1 = false, hasPix2 = false;
    if (params.pixelPriority) {
      hasPix1 = hasPix[i] != 0;
      hasPix2 = hasPix[j] != 0;
      if (hasPix1 && hasPix2 && !shareHit(tracks, i, j, params, true))
        return -1;
    }
    if (params.sharedHitPairsOnly && !shareHit(tracks, i, j, params, false))
      return -1;
    // pixelOrScoreWins(i, j): with pixelPriority a track with pixel hits beats a pixel-less one, else the higher score
    const bool iWins = (params.pixelPriority && hasPix1 != hasPix2) ? hasPix1 : (tracks[i].score() > tracks[j].score());

    const float maxPolarOb = params.ptAdaptive ? kMaxetaObP2 : kMaxcthOb;
    const float maxPolarFw = params.ptAdaptive ? kMaxetaFwP2 : kMaxcthFw;
    float maxdRSquared = params.drthCentral * params.drthCentral;
    if (std::abs(polar1) > maxPolarFw)
      maxdRSquared = params.drthForward * params.drthForward;
    else if (std::abs(polar1) > maxPolarOb)
      maxdRSquared = params.drthObarrel * params.drthObarrel;
    const float dr2 = dphi * dphi + dpolar * dpolar;
    if (dr2 < maxdRSquared) {
      // keep track with best score
      return iWins ? j : i;
    }

    if (std::abs(tracks[j].params().v[3] - invpt1) > kMaxd1pt)
      return -1;

    const float fraction = sharedHitFraction(tracks, dir, i, j, dr2, params);
    // the unique-hit rule of the pT-adaptive cleaner reads the full count
    const bool countAll = params.ptAdaptive && params.minUniqueHitsToKeep > 0;

    int sharedCount = 0;
    const int nF1 = tracks[i].nFoundHits(), nF2 = tracks[j].nFoundHits();
    const int minFoundHits = nF1 < nF2 ? nF1 : nF2;
    const auto& h1 = tracks[i].hits().hot;
    const auto& h2 = tracks[j].hits().hot;
    const int n1 = tracks[i].nTotalHits(), n2 = tracks[j].nTotalHits();

    // MkFitCore counts every (ih1, ih2) pair of equal valid hits (its in-loop 'continue's are no-ops) and sets sharedFirst
    // when the first entries of both tracks are valid and equal. Same counts here: the packed (index, layer) words are
    // equal iff both fields are, and a valid hit (index >= 0) can never equal an invalid entry, so track 2 needs no
    // validity test. sharedCount only grows, so stopping once the threshold is met is exact unless countAll.
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
      if (!countAll && (sharedCount - sharedFirst) >= ((minFoundHits - sharedFirst) * fraction))
        break;
    }

    if ((sharedCount - sharedFirst) >= ((minFoundHits - sharedFirst) * fraction)) {
      const int loser = iWins ? j : i;
      // pT-adaptive cleaner: keep a high-pT would-be duplicate that still has many hits of its own
      if (countAll && std::abs(1.f / tracks[loser].params().v[3]) >= params.minPtUniqueHitsToKeep &&
          tracks[loser].nFoundHits() - sharedCount >= params.minUniqueHitsToKeep)
        return -1;
      return loser;
    }
    return -1;
  }

}  // namespace mkfitdev::clean

#endif
