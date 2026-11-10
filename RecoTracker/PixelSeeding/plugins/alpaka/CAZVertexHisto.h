#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAZVertexHisto_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAZVertexHisto_h

// z-vertex finding inside the inner-tracker-only CA (pixelTopology::Phase2, zVertexFilter), used to restrict the
// short tracks of the less pure categories to the leading vertices of the event.
//
// 1. Kernel_zvtxFill: every track with >= 4 hits and quality >= highPurity (after the classification, the
//    hit-expectation tests and the duplicate removers) fills a histogram of its z at the point of closest approach
//    to the beam line (zip), kNBins bins of kBinWidth over +-kZRange. Each track adds the weight
//    trackWeight x min(pT, ptCap)^ptPower, spread over the bins as a Gaussian of its sigma(zip); a track with a
//    sigma below one bin fills one bin.
// 2. Kernel_zvtxFind (one block): a sliding sum over +-peakHalfWidth bins, its local maxima above threshold, and
//    the nMax largest of them separated by at least minSeparation; each vertex is the weighted mean z of its window.
// 3. Kernel_zvtxFilterTriplets: a 3- or 4-hit track of quality >= strict is demoted to dup unless its zip is
//    within dzTable[c] of one of the vertices and its |tip| is below tipTable[c] (negative = not restricted), for
//    the category c = 3 * (2 * class + dxyClass) + region of a 3-hit track (class: see
//    caHoleCheck::tripletClass), and c = 18 + 3 * dxyClass + region of a 4-hit track; dxyClass = |tip| >= dxyKnee,
//    region = |eta| < 0.9, 0.9-1.6, >= 1.6. A 3-hit track that failed the hit-expectation tests (suspect) gets its
//    quality back if its zip is within rescueDz of a vertex and |tip| < rescueMaxTip. Without a vertex the tracks
//    are left untouched.

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

#include "CAStructures.h"
#include "CATripletHoleCheck.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caZVertex {

  constexpr int kNBins = 3000;
  constexpr float kZRange = 15.f;                      // cm
  constexpr float kBinWidth = 2.f * kZRange / kNBins;  // 100 um
  constexpr int kMaxVtx = 8;

  struct ZVertexHisto {
    float bins[kNBins];
    float z[kMaxVtx];
    int n;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int binOf(TAcc const& acc, float z) {
    return int(alpaka::math::floor(acc, (z + kZRange) / kBinWidth));
  }

  class Kernel_zvtxFill {
  public:
    template <typename TAcc, typename HitContainer>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::reco::TrackSoAConstView tracks_view,
                                  HitContainer const* __restrict__ foundNtuplets,
                                  ZVertexHisto* __restrict__ hist,
                                  caStructures::ZVertexParams p) const {
      using Quality = ::pixelTrack::Quality;
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        const auto nh = foundNtuplets->size(it);
        if (nh == 0)
          break;
        if (nh < 4 || tracks_view[it].quality() < Quality::highPurity)
          continue;
        const float z = ::reco::zip(tracks_view, it);
        const float var = tracks_view[it].covariance()(14);
        if (!(alpaka::math::abs(acc, z) < kZRange) || !(var >= 0.f))
          continue;
        const float w =
            p.trackWeight * alpaka::math::pow(acc, alpaka::math::min(acc, tracks_view[it].pt(), p.ptCap), p.ptPower);
        const float s = alpaka::math::sqrt(acc, var);
        const int b0 = binOf(acc, z);
        if (s < kBinWidth) {
          alpaka::atomicAdd(acc, &hist->bins[b0], w, alpaka::hierarchy::Blocks{});
          continue;
        }
        const int nb = alpaka::math::min(acc, int(3.f * s / kBinWidth) + 1, 50);
        const float inv = 1.f / (1.41421356f * s);
        const float zLo = -kZRange + (b0 - nb) * kBinWidth;
        const float norm = alpaka::math::erf(acc, (zLo + (2 * nb + 1) * kBinWidth - z) * inv) -
                           alpaka::math::erf(acc, (zLo - z) * inv);
        if (!(norm > 0.f))
          continue;
        float e0 = alpaka::math::erf(acc, (zLo - z) * inv);
        for (int b = b0 - nb; b <= b0 + nb; ++b) {
          const float e1 = alpaka::math::erf(acc, (-kZRange + (b + 1) * kBinWidth - z) * inv);
          if (b >= 0 && b < kNBins)
            alpaka::atomicAdd(acc, &hist->bins[b], w * (e1 - e0) / norm, alpaka::hierarchy::Blocks{});
          e0 = e1;
        }
      }
    }
  };

  class Kernel_zvtxFind {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ZVertexHisto* __restrict__ hist,
                                  caStructures::ZVertexParams p) const {
      // Launched as a single block. The sliding sums S(c), c = -1 .. kNBins, and the local-maximum flags are computed
      // in parallel into shared memory (each sum in the order of a serial loop); the order-dependent greedy peak
      // selection then runs on one thread over the flagged bins only.
      auto& sums = alpaka::declareSharedVar<float[kNBins + 2], __COUNTER__>(acc);  // sums[c + 1] = S(c)
      auto& isMax = alpaka::declareSharedVar<uint8_t[kNBins], __COUNTER__>(acc);
      const int hw = p.peakHalfWidth;
      for (int i : cms::alpakatools::independent_group_elements(acc, kNBins + 2)) {
        const int c = i - 1;
        float s = 0.f;
        for (int b = alpaka::math::max(acc, 0, c - hw); b <= alpaka::math::min(acc, kNBins - 1, c + hw); ++b)
          s += hist->bins[b];
        sums[i] = s;
      }
      alpaka::syncBlockThreads(acc);
      for (int c : cms::alpakatools::independent_group_elements(acc, kNBins)) {
        const float s = sums[c + 1];
        isMax[c] = (s < p.threshold || s < sums[c] || s <= sums[c + 2]) ? 0 : 1;
      }
      alpaka::syncBlockThreads(acc);
      if (!cms::alpakatools::once_per_block(acc))
        return;
      // candidate peaks: local maxima of the sliding sum above threshold, kept sorted by weight (top nMax)
      float cw[kMaxVtx];
      int cb[kMaxVtx];
      int nc = 0;
      const int minSep = int(p.minSeparation / kBinWidth + 0.5f);
      for (int c = 0; c < kNBins; ++c) {
        if (!isMax[c])
          continue;
        const float s = sums[c + 1];
        // separation from the kept candidates: replace a weaker close one, or drop this one
        int close = -1;
        for (int i = 0; i < nc; ++i)
          if (alpaka::math::abs(acc, cb[i] - c) < minSep)
            close = i;
        if (close >= 0) {
          if (cw[close] >= s)
            continue;
          for (int i = close; i + 1 < nc; ++i) {
            cw[i] = cw[i + 1];
            cb[i] = cb[i + 1];
          }
          --nc;
        }
        // insert sorted (descending)
        int pos = nc;
        while (pos > 0 && cw[pos - 1] < s)
          --pos;
        if (pos >= p.nMax)
          continue;
        const int last = alpaka::math::min(acc, nc, p.nMax - 1);
        for (int i = last; i > pos; --i) {
          cw[i] = cw[i - 1];
          cb[i] = cb[i - 1];
        }
        cw[pos] = s;
        cb[pos] = c;
        nc = alpaka::math::min(acc, nc + 1, p.nMax);
      }
      // each vertex at the weighted mean z of its window
      for (int i = 0; i < nc; ++i) {
        float sw = 0.f, swz = 0.f;
        for (int b = alpaka::math::max(acc, 0, cb[i] - hw); b <= alpaka::math::min(acc, kNBins - 1, cb[i] + hw); ++b) {
          sw += hist->bins[b];
          swz += hist->bins[b] * (-kZRange + (b + 0.5f) * kBinWidth);
        }
        hist->z[i] = sw > 0.f ? swz / sw : -kZRange + (cb[i] + 0.5f) * kBinWidth;
      }
      hist->n = nc;
    }
  };

  template <typename TrackerTraits>
  class Kernel_zvtxFilterTriplets {
  public:
    using HitsView = caStructures::HitsViewT<TrackerTraits>;

    template <typename TAcc, typename HitContainer>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::reco::TrackSoAView tracks_view,
                                  HitContainer const* __restrict__ foundNtuplets,
                                  HitsView hh,
                                  ::reco::CALayersSoAConstView cl,
                                  ZVertexHisto const* __restrict__ hist,
                                  caStructures::ZVertexParams p,
                                  uint8_t const* __restrict__ suspect) const {
      using Quality = ::pixelTrack::Quality;
      if (hist->n == 0)
        return;
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        const auto nh = foundNtuplets->size(it);
        if (nh == 0)
          break;
        // vertex rescue of a 3-hit track that failed the hit-expectation tests
        if (suspect && suspect[it] != 0 && nh == 3) {
          if (p.rescueDz >= 0.f && alpaka::math::abs(acc, ::reco::tip(tracks_view, it)) < p.rescueMaxTip) {
            const float zr = ::reco::zip(tracks_view, it);
            for (int v = 0; v < hist->n; ++v)
              if (alpaka::math::abs(acc, zr - hist->z[v]) < p.rescueDz) {
                tracks_view[it].quality() = static_cast<Quality>(suspect[it] - 1);
                break;
              }
          }
          continue;
        }
        if ((nh != 3 && nh != 4) || tracks_view[it].quality() < Quality::strict)
          continue;
        const float tip = alpaka::math::abs(acc, ::reco::tip(tracks_view, it));
        const int dxyClass = tip >= p.dxyKnee ? 1 : 0;
        const float absCotTheta = alpaka::math::abs(acc, tracks_view[it].state()(3));
        const int region = absCotTheta < 1.0265f ? 0 : (absCotTheta < 2.3756f ? 1 : 2);  // |eta| 0.9, 1.6
        int category;
        if (nh == 3) {
          auto hp = foundNtuplets->begin(it);
          const int cls = caHoleCheck::tripletClass(caHoleCheck::layerOfModule(cl, hh[hp[0]].detectorIndex()),
                                                    caHoleCheck::layerOfModule(cl, hh[hp[1]].detectorIndex()),
                                                    caHoleCheck::layerOfModule(cl, hh[hp[2]].detectorIndex()));
          category = 3 * (2 * cls + dxyClass) + region;
        } else {
          category = 18 + 3 * dxyClass + region;
        }
        const float dzTol = p.dzTable[category];
        const float tipMax = p.tipTable[category];
        if (dzTol < 0.f)
          continue;
        const float z = ::reco::zip(tracks_view, it);
        bool inVertex = false;
        for (int v = 0; v < hist->n && !inVertex; ++v)
          inVertex = alpaka::math::abs(acc, z - hist->z[v]) < dzTol;
        if (!inVertex || (tipMax >= 0.f && tip >= tipMax))
          tracks_view[it].quality() = Quality::dup;
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caZVertex

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAZVertexHisto_h
