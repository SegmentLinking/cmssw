#ifndef RecoTracker_LSTCore_interface_alpaka_TCSeedStateSelection_h
#define RecoTracker_LSTCore_interface_alpaka_TCSeedStateSelection_h

// The per-candidate work of the seed-state fit of the track candidates: which hits are fitted, in which order, and
// on which hit the state sits. Compiled only in CMSSW, as the fit (TCSeedStateFit.h).
#ifndef LST_STANDALONE

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "RecoTracker/LSTCore/interface/OTHitCovariance.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentParameters.h"
#include "RecoTracker/LSTCore/interface/SeedStateFitTypes.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/SeedStateFit.h"
#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStateFit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState {

  // Two hits of one layer slot whose transverse positions are closer than this are degenerate for the circle fit.
  // It happens to the two sensors of a 2S endcap module crossed by a straight enough track;
  // the line fit is not affected, the two hits still being apart in z.
  // The outer hit of such a pair is not fitted. The value, squared in cm^2, sits between the coincidences
  // left by the float rounding of the global transforms of the two sensors (below 1e-5 cm) and the genuine
  // separations of the two sensors (from a few 1e-3 cm).
  constexpr float kMaxDegenerateSeparation2 = 1e-4f * 1e-4f;

  // stands for "not an OT hit" in the search of duplicate OT hits
  constexpr uint32_t kNoOTHit = std::numeric_limits<uint32_t>::max();

  // the most hits a track candidate can have: two per layer slot
  constexpr uint32_t kMaxCandidateHits = ::lst::Params_TC::kHits;

  // The pixel hits of pT3 and pT5 are fitted with their OT hits; a T5 or a T4 has none,
  // and a pLS is not fitted here (its seed state is the one of its pixel track).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr bool usesPixelSlots(::lst::LSTObjType type) {
    return type == ::lst::LSTObjType::pT3 || type == ::lst::LSTObjType::pT5;
  }

  // The pixelSeedIndex of a row: the candidate's for the types with a pixel seed (pT3, pT5, pLS), none otherwise.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t rowPixelSeedIndex(::lst::LSTObjType type,
                                                                           uint32_t candidatePixelSeedIndex) {
    return usesPixelSlots(type) || type == ::lst::LSTObjType::pLS ? candidatePixelSeedIndex : ::lst::kInvalidU32Idx;
  }

  // The hits of the LST input as the fit reads them.
  // - A hit index h < nHitsOT is an OT row of `hits`, with its position there. Its global covariance is computed from
  //   row h of the OT rechits and its module frame (otHits), as the device LST input computes it for the fits of the
  //   pixel tracks: the LST input does not store it, since the fit needs it for few of the OT rows.
  // - A hit index h >= nHitsOT is a pLS hit slot: its row of `hits` holds pLS kinematics
  //   (see RecoTracker/LST/plugins/alpaka/LSTInputDeviceKernels.dev.cc, FillPixelSegmentsKernel),
  //   so its position and covariance are read from hitsIT[h - nHitsOT], the values that entered
  //   the fit of the pixel track; its detid tells a pixel hit from an OT hit of the pixel track.
  struct InputHits {
    ::lst::HitsBaseConst hits;
    ::lst::HitsITConst hitsIT;
    ::lst::PixelSeedsConst pixelSeeds;
    uint32_t nHitsOT;
    ::lst::OTHitCovarianceInputs otHits;

    ALPAKA_FN_ACC bool isPixel(uint32_t h) const { return h >= nHitsOT && hits.detid()[h] == ::lst::kPixelModuleId; }

    // The LSTOTHits index of an OT hit, kNoOTHit for a pixel hit. The OT rows of `hits` are the OT rechits in the
    // order of LSTOTHits, so an OT row is its own index; an OT hit of a pLS slot refers to its OT rechit row by
    // hitsIT.idxs.
    ALPAKA_FN_ACC uint32_t otIndex(uint32_t h) const {
      if (h < nHitsOT)
        return h;
      return isPixel(h) ? kNoOTHit : hitsIT.idxs()[h - nHitsOT];
    }

    ALPAKA_FN_ACC void position(uint32_t h, float& x, float& y, float& z) const {
      if (h < nHitsOT) {
        x = hits.xs()[h];
        y = hits.ys()[h];
        z = hits.zs()[h];
      } else {
        x = hitsIT.xs()[h - nHitsOT];
        y = hitsIT.ys()[h - nHitsOT];
        z = hitsIT.zs()[h - nHitsOT];
      }
    }

    ALPAKA_FN_ACC ::lst::FitHit fitHit(uint32_t h) const {
      ::lst::FitHit hit;
      position(h, hit.x, hit.y, hit.z);
      if (h < nHitsOT) {
        otHits.covariance(h, hit.cov);
      } else {
        auto const& cov = hitsIT.cov()[h - nHitsOT];
        for (int k = 0; k < 6; ++k)
          hit.cov[k] = cov[k];
      }
      return hit;
    }

    // The encoded hit a state on hit h sits on:
    // - an OT hit by its LSTOTHits index;
    // - a pixel hit by its position k in its pixel track,
    //   from its slot in its pLS and the hit count of the pixel track,
    //   as the LST input fills the slots.
    ALPAKA_FN_ACC uint32_t stateHit(uint32_t h) const {
      if (!isPixel(h))
        return ::lst::encodeStateHitOT(otIndex(h));
      // the pLS whose slots hold h: the last one whose first slot is not above h (firstHit increases with the pLS)
      uint32_t low = 0, high = pixelSeeds.metadata().size();
      while (high - low > 1) {
        const uint32_t middle = (low + high) / 2;
        if (pixelSeeds.firstHit()[middle] <= h)
          low = middle;
        else
          high = middle;
      }
      const uint32_t nHits = pixelSeeds.nHits()[low];
      const uint32_t nSlots = nHits < ::lst::kMaxPLSHitsInHitsSoA ? nHits : ::lst::kMaxPLSHitsInHitsSoA;
      return ::lst::encodeStateHitPixel(::lst::pixelSegmentHitOfSlot(h - pixelSeeds.firstHit()[low], nSlots, nHits));
    }
  };

  // The hits of a track candidate kept by collectHits, with their transverse position
  // and squared 3D distance from the origin.
  struct Collected {
    uint32_t n;
    uint32_t hits[kMaxCandidateHits];
    float x[kMaxCandidateHits], y[kMaxCandidateHits], d2[kMaxCandidateHits];
  };

  // The hits of the fit of a track candidate.
  struct Selection {
    uint32_t nCollected;         // hits kept by collectHits
    uint32_t nFit;               // hits of the fit, 0 if too few
    uint32_t hits[kMaxFitHits];  // their indices, inside out; the last one is the hit the state sits on
  };

  // The hits of a track candidate, slot by slot (for a pT3 or a pT5 the two pLS slots first, then the OT slots):
  // - an OT hit already collected is skipped, so that the first copy wins: a pixel track's OT hit can be an OT hit of
  //   the candidate's outer object too, and LSTOutputConverter skips such a shared hit when it builds the seed's hits.
  // - of the two hits of a slot, the one closer to the origin in 3D comes first, and the other one is dropped if the
  //   pair is degenerate (kMaxDegenerateSeparation2), so that the inner hit of a degenerate outermost pair is the last
  //   hit; when one hit of a slot is skipped as a copy, the other one is tested against that copy the same way;
  // - a slot always holds two hit indices; an empty second index is read as the first one.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void collectHits(InputHits const& input,
                                                  ::lst::Params_TC::ArrayUxHits const& slots,
                                                  bool pixelSlots,
                                                  Collected& collected) {
    uint32_t seenOT[kMaxCandidateHits], seenAt[kMaxCandidateHits];
    uint32_t nSeen = 0;
    uint32_t n = 0;
    auto entryOf = [&](uint32_t otHit) -> int {
      if (otHit == kNoOTHit)
        return -1;
      for (uint32_t i = 0; i < nSeen; ++i)
        if (seenOT[i] == otHit)
          return static_cast<int>(seenAt[i]);
      return -1;
    };
    auto set = [&](uint32_t entry, uint32_t h, uint32_t otHit, float x, float y, float z) {
      collected.hits[entry] = h;
      collected.x[entry] = x;
      collected.y[entry] = y;
      collected.d2[entry] = x * x + y * y + z * z;
      if (otHit != kNoOTHit) {
        seenOT[nSeen] = otHit;
        seenAt[nSeen] = entry;
        ++nSeen;
      }
    };
    auto push = [&](uint32_t h, uint32_t otHit, float x, float y, float z) { set(n++, h, otHit, x, y, z); };
    for (int slot = pixelSlots ? 0 : ::lst::Params_TC::kPixelLayerSlots; slot < ::lst::Params_TC::kLayers; ++slot) {
      const uint32_t h0 = slots[slot][0];
      if (h0 == ::lst::kTCEmptyHitIdx)
        continue;
      const uint32_t h1 = slots[slot][1] == ::lst::kTCEmptyHitIdx ? h0 : slots[slot][1];
      const uint32_t ot0 = input.otIndex(h0), ot1 = input.otIndex(h1);
      const int copy0 = entryOf(ot0), copy1 = entryOf(ot1);
      const bool duplicate0 = copy0 >= 0;
      const bool duplicate1 = copy1 >= 0 || (!duplicate0 && (h0 == h1 || (ot0 != kNoOTHit && ot0 == ot1)));
      if (duplicate0 && duplicate1)
        continue;
      float x0, y0, z0, x1, y1, z1;
      input.position(h0, x0, y0, z0);
      input.position(h1, x1, y1, z1);
      if (duplicate0 || duplicate1) {
        // One hit of the slot is already collected, or is the other one listed twice.
        const bool keep1 = duplicate0;
        const uint32_t h = keep1 ? h1 : h0, ot = keep1 ? ot1 : ot0;
        const float x = keep1 ? x1 : x0, y = keep1 ? y1 : y0, z = keep1 ? z1 : z0;
        const int copy = keep1 ? copy0 : copy1;
        if (copy >= 0) {
          const float dx = x - collected.x[copy], dy = y - collected.y[copy];
          if (dx * dx + dy * dy < kMaxDegenerateSeparation2) {
            if (x * x + y * y + z * z < collected.d2[copy])
              set(static_cast<uint32_t>(copy), h, ot, x, y, z);
            continue;
          }
        }
        push(h, ot, x, y, z);
        continue;
      }
      const bool swap = x0 * x0 + y0 * y0 + z0 * z0 > x1 * x1 + y1 * y1 + z1 * z1;
      if (swap)
        push(h1, ot1, x1, y1, z1);
      else
        push(h0, ot0, x0, y0, z0);
      const float dx = x1 - x0, dy = y1 - y0;
      if (dx * dx + dy * dy < kMaxDegenerateSeparation2)
        continue;
      if (swap)
        push(h0, ot0, x0, y0, z0);
      else
        push(h1, ot1, x1, y1, z1);
    }
    collected.n = n;
  }

  // the collected hits ordered by 3D distance from the origin, inside out (a stable insertion sort)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void sortInsideOut(Collected& collected) {
    for (uint32_t i = 1; i < collected.n; ++i) {
      const uint32_t h = collected.hits[i];
      const float x = collected.x[i], y = collected.y[i], d2 = collected.d2[i];
      uint32_t j = i;
      while (j > 0 && collected.d2[j - 1] > d2) {
        collected.hits[j] = collected.hits[j - 1];
        collected.x[j] = collected.x[j - 1];
        collected.y[j] = collected.y[j - 1];
        collected.d2[j] = collected.d2[j - 1];
        --j;
      }
      collected.hits[j] = h;
      collected.x[j] = x;
      collected.y[j] = y;
      collected.d2[j] = d2;
    }
  }

  // m of the nSrc hits of src, in their order: every pixel hit, and the OT hits by a uniform stride rounded to the
  // nearest hit with the outermost OT hit forced, for its lever arm Requires nSrc >= m and at most m pixel hits.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void strideKeepingPixels(
      InputHits const& input, uint32_t const* src, uint32_t nSrc, uint32_t m, uint32_t* dst) {
    int nPixel = 0;
    for (uint32_t i = 0; i < nSrc; ++i)
      nPixel += input.isPixel(src[i]) ? 1 : 0;
    const int nOT = static_cast<int>(nSrc) - nPixel;
    const int nWantOT = static_cast<int>(m) - nPixel;
    float increment = nWantOT > 0 ? static_cast<float>(nOT) / static_cast<float>(nWantOT) : 0.f;
    if (increment < 1.f)
      increment = 1.f;
    // the position among the OT hits of the next OT hit to keep; the last one kept is the outermost
    int nextOT = nWantOT > 0 ? (nWantOT == 1 ? nOT - 1 : 0) : -1;
    int nOut = 0, iOT = 0, kOT = 0;
    for (uint32_t i = 0; i < nSrc && nOut < static_cast<int>(m); ++i) {
      if (input.isPixel(src[i])) {
        dst[nOut++] = src[i];
        continue;
      }
      // also keep every remaining OT hit once only as many are left as are still wanted
      if (iOT == nextOT || nOT - iOT <= nWantOT - kOT) {
        dst[nOut++] = src[i];
        ++kOT;
        if (kOT >= nWantOT)
          nextOT = -1;
        else if (kOT == nWantOT - 1)
          nextOT = nOT - 1;
        else
          nextOT = static_cast<int>(static_cast<float>(kOT) * increment + 0.5f);
      }
      ++iOT;
    }
  }

  // The hits of the fit among the collected hits, ordered inside out (fitSize):
  // - as many as the fit size: all of them;
  // - one more: the hit at the smallest transverse separation from the previous hit is dropped, as the hit that adds
  //   the least to the circle fit; never a pixel hit, and never the outermost hit, on which the state must sit (a seed
  //   puts its state on its last hit); without such a hit, the stride below;
  // - more: kMaxFitHits by strideKeepingPixels.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void chooseFitHits(InputHits const& input,
                                                    Collected const& collected,
                                                    Selection& selection) {
    const uint32_t n = collected.n;
    const uint32_t nFit = fitSize(n);
    selection.nCollected = n;
    selection.nFit = nFit;
    if (nFit == 0)
      return;
    if (n == nFit) {
      for (uint32_t i = 0; i < n; ++i)
        selection.hits[i] = collected.hits[i];
      return;
    }
    if (n == nFit + 1) {
      int drop = -1;
      float smallest = std::numeric_limits<float>::max();
      // i = n - 1 is not a candidate: the outermost hit stays the last node
      for (uint32_t i = 1; i + 1 < n; ++i) {
        if (input.isPixel(collected.hits[i]))
          continue;
        const float dx = collected.x[i] - collected.x[i - 1], dy = collected.y[i] - collected.y[i - 1];
        const float separation2 = dx * dx + dy * dy;
        if (separation2 < smallest) {
          smallest = separation2;
          drop = static_cast<int>(i);
        }
      }
      if (drop >= 0) {
        for (uint32_t i = 0; i < nFit; ++i)
          selection.hits[i] = collected.hits[i < static_cast<uint32_t>(drop) ? i : i + 1];
        return;
      }
    }
    strideKeepingPixels(input, collected.hits, n, nFit, selection.hits);
  }

  // What prepareCandidate decided for a track candidate.
  enum class Preparation : uint8_t { kFit, kPLS, kTooFewHits };

  // The work of the prepare kernel for one track candidate.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Preparation prepareCandidate(InputHits const& input,
                                                              ::lst::Params_TC::ArrayUxHits const& slots,
                                                              ::lst::LSTObjType type,
                                                              Collected& collected,
                                                              Selection& selection,
                                                              FitInput& fitInput,
                                                              ::lst::FittedState& state) {
    collected.n = 0;
    selection.nCollected = 0;
    selection.nFit = 0;
    if (type == ::lst::LSTObjType::pLS) {
      setUnfit(state, ::lst::kInvalidU32Idx);
      return Preparation::kPLS;
    }
    collectHits(input, slots, usesPixelSlots(type), collected);
    sortInsideOut(collected);
    chooseFitHits(input, collected, selection);
    if (selection.nFit == 0) {
      setUnfit(state, ::lst::kInvalidU32Idx);
      return Preparation::kTooFewHits;
    }
    for (uint32_t i = 0; i < selection.nFit; ++i)
      fitInput.hits[i] = input.fitHit(selection.hits[i]);
    setUnfit(state, input.stateHit(selection.hits[selection.nFit - 1]));
    return Preparation::kFit;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState

#endif  // LST_STANDALONE

#endif
