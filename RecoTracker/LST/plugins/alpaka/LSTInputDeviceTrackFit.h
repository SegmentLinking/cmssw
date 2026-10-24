#ifndef RecoTracker_LST_plugins_alpaka_LSTInputDeviceTrackFit_h
#define RecoTracker_LST_plugins_alpaka_LSTInputDeviceTrackFit_h

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <cmath>
#include <cstdint>
#include <numbers>

#include <alpaka/alpaka.hpp>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/LST/interface/PixelSegmentClosestApproach.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentParameters.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/SeedStateFit.h"

#include "LSTInputDeviceKernels.h"

// The per-track work of the device LST input: the hits of the fits of a pixel track, the Broken Line Fit with its
// state at the last fitted hit, the checks of that state, and the pLS computed from it.
namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  using ::lstInputDevice::FitHit;
  using ::lstInputDevice::kITFitHits;
  using ::lstInputDevice::kMaxFitHits;
  using ::lstInputDevice::kMaxTrackHits;
  using ::lstInputDevice::kMinFitHits;

  using ::ALPAKA_ACCELERATOR_NAMESPACE::lst::checkState;
  using ::ALPAKA_ACCELERATOR_NAMESPACE::lst::fitState;
  using ::ALPAKA_ACCELERATOR_NAMESPACE::lst::setUnfit;

  // A hit of a pixel track as prepareTrack reads it: what the fit uses, and what the LST input records.
  struct TrackHit {
    FitHit fit;
    uint32_t index;  // pixel rechit or OT rechit row
    uint32_t detId;
    uint16_t clusterSize;
  };

  // The fits prepareTrack set up for a track.
  struct TrackPreparation {
    uint32_t nFit;     // hits of the full fit
    uint32_t nITHits;  // inner-tracker hits of the track
    bool itCandidate;  // the output converter may rebuild the seed without its OT hits
    bool itFit;        // an inner-tracker-only candidate with the kITFitHits inner-tracker hits of that fit
  };

  // the global covariance (xx, xy, yy, xz, yz, zz) of a hit with local variances varX and varY, uncorrelated, on a
  // module of frame `frame`
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void hitCovariance(SOAFrame<float> const& frame,
                                                         float varX,
                                                         float varY,
                                                         float cov[6]) {
    frame.toGlobal(varX, 0.f, varY, cov);
  }

  // whether a pixel track of nHits hits is fitted
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr bool isFitted(uint32_t nHits) {
    return nHits >= kMinFitHits && nHits <= kMaxTrackHits;
  }

  // the number of hits of the full fit of a track of nHits hits: the largest fit size not above nHits, 0 below the
  // smallest one. The fit sizes are the even numbers from kMinFitHits to kMaxFitHits.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t nFitHits(uint32_t nHits) {
    return nHits >= kMaxFitHits ? kMaxFitHits : (nHits < kMinFitHits ? 0 : nHits - nHits % 2);
  }

  static_assert(kMinFitHits % 2 == 0 && kMaxFitHits % 2 == 0 && kMinFitHits <= kMaxFitHits,
                "the fit sizes are the even numbers from kMinFitHits to kMaxFitHits");
  static_assert(::lstInputDevice::kNumFitSizes == (kMaxFitHits - kMinFitHits) / 2 + 1,
                "kNumFitSizes counts the even numbers from kMinFitHits to kMaxFitHits");
  static_assert(nFitHits(kMaxFitHits) == kMaxFitHits && nFitHits(kMinFitHits) == kMinFitHits &&
                    nFitHits(kMinFitHits - 1) == 0,
                "nFitHits at the ends of the set of fit sizes");

  // whether nFitHits maps each fit size, and the hit count one above it, to that fit size
  constexpr bool fitSizesAreFixedPoints() {
    for (uint32_t n = kMinFitHits; n <= kMaxFitHits; n += 2)
      if (nFitHits(n) != n || (n + 1 <= kMaxFitHits && nFitHits(n + 1) != n))
        return false;
    return true;
  }
  static_assert(fitSizesAreFixedPoints(), "nFitHits maps each fit size, and the hit count above it, to the fit size");

  // The positions in a track of nHits hits of its nFit fitted hits, inside out, as the pixel-track fit takes them
  // (RecoTracker/PixelSeeding/plugins/alpaka/BrokenLineFit.h): a uniform stride of max(1, nHits / nFit) rounded to the
  // nearest hit, from the first hit, and the last hit forced as the last node.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void fitHitPositions(uint32_t nHits, uint32_t nFit, uint32_t* positions) {
    const float ratio = float(nHits) / float(nFit);
    const float increment = ratio > 1.f ? ratio : 1.f;
    float n = 0.f;
    for (uint32_t i = 0; i < nFit; ++i) {
      positions[i] = i + 1 == nFit ? nHits - 1 : static_cast<uint32_t>(int(n + 0.5f));
      n += increment;
    }
  }

  // The fits of a pixel track of nHits hits:
  // - the hits of the full fit (nFitHits of them, at fitHitPositions) put into input.hits;
  // - for an inner-tracker-only candidate with kITFitHits inner-tracker hits, hits are put into input.itHits;
  // A track is an inner-tracker-only candidate when, as LSTOutputConverter decides the rebuild of a pLS seed
  // without its OT hits, its last hit is in the outer tracker and it has at most maxITHitsToDropOTHits
  // (at most kITFitHits) inner-tracker hits. A hit used by several fits or slots is read once.
  template <typename Hits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE TrackPreparation prepareTrack(Hits const& hits,
                                                               uint32_t nHits,
                                                               uint32_t maxITHitsToDropOTHits,
                                                               FitInput& input,
                                                               PixelSegmentCandidate& candidate,
                                                               FittedState& state,
                                                               FittedState& itState) {
    TrackPreparation result{nFitHits(nHits), 0, false, false};
    uint32_t positions[kMaxFitHits];
    fitHitPositions(nHits, result.nFit, positions);

    const uint32_t nSlots = nHits < ::lst::kMaxPLSHitsInHitsSoA ? nHits : ::lst::kMaxPLSHitsInHitsSoA;
    auto setSlot = [&candidate](uint32_t slot, TrackHit const& hit) {
      candidate.slotDetId[slot] = hit.detId;
      candidate.slotClusterSize[slot] = hit.clusterSize;
      candidate.slotHitIndex[slot] = hit.index;
      candidate.slotHits[slot] = hit.fit;
    };
    bool slotSet[::lst::kMaxPLSHitsInHitsSoA] = {};
    for (uint32_t i = 0; i < result.nFit; ++i) {
      const TrackHit hit = hits.hit(positions[i]);
      input.hits[i] = hit.fit;
      for (uint32_t slot = 0; slot < nSlots; ++slot)
        if (::lst::pixelSegmentHitOfSlot(slot, nSlots, nHits) == positions[i]) {
          setSlot(slot, hit);
          slotSet[slot] = true;
        }
    }
    for (uint32_t slot = 0; slot < nSlots; ++slot)
      if (!slotSet[slot])
        setSlot(slot, hits.hit(::lst::pixelSegmentHitOfSlot(slot, nSlots, nHits)));

    uint32_t lastITHit = 0;
    for (uint32_t k = 0; k < nHits; ++k)
      if (!hits.isOT(k)) {
        ++result.nITHits;
        lastITHit = k;
      }
    const uint32_t nBits = nHits < ::lst::kMaxPLSHitBitsInHitsSoA ? nHits : ::lst::kMaxPLSHitBitsInHitsSoA;
    uint8_t hitDetBits = 0;
    for (uint32_t bit = 0; bit < nBits; ++bit)
      hitDetBits |= hits.isOT(::lst::pixelSegmentHitOfSlot(bit, nBits, nHits)) << bit;

    candidate.pixelType = ::lst::PixelType::kInvalid;
    candidate.nHits = nHits;
    candidate.nITHits = result.nITHits;
    candidate.nSlots = nSlots;
    candidate.hitDetBits = hitDetBits;
    setUnfit(state, ::lst::encodeStateHitPixel(nHits - 1));
    setUnfit(itState, result.nITHits > 0 ? ::lst::encodeStateHitPixel(lastITHit) : ::lst::kInvalidU32Idx);

    result.itCandidate = hits.isOT(nHits - 1) && result.nITHits <= maxITHitsToDropOTHits;
    result.itFit = result.itCandidate && result.nITHits == kITFitHits;
    if (result.itFit) {
      uint32_t itHit = 0;
      for (uint32_t k = 0; k < nHits && itHit < kITFitHits; ++k) {
        if (hits.isOT(k))
          continue;
        int fitted = -1;
        for (uint32_t i = 0; i < result.nFit; ++i)
          if (positions[i] == k)
            fitted = i;
        input.itHits[itHit++] = fitted >= 0 ? input.hits[fitted] : hits.hit(k).fit;
      }
    }
    return result;
  }

  // The closest approach to the beam line and the pLS of a full state of status kOk,
  // as LSTInputProducer computes them from a seed:
  // - the quantities on the last hit from the stored state,
  // - those at the closest approach from lst::closestApproach in the uniform field of the fit.
  // A failed closest approach keeps the pLS with zero closest-approach quantities and charge.
  // candidate.pixelType is kInvalid when the track gives no pLS; the other pLS fields are then not set.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void makePixelSegment(FittedState const& state,
                                                       Settings const& settings,
                                                       PixelSegmentCandidate& candidate) {
    auto const& row = state.row;
    const beamLineClosestApproach::BeamLine beamLine{{settings.beamSpotX, settings.beamSpotY, settings.beamSpotZ},
                                                     {settings.beamSlopeX, settings.beamSlopeY, 1.f}};
    const ::lst::PixelSegmentClosestApproach pca =
        ::lst::closestApproach(row, ::lst::UniformField(settings.bField), beamLine);
    candidate.pcaStatus = pca.status;

    const float ptIn = settings.bField / std::abs(row.kappa);
    float deltaPhi = std::atan2(row.y, row.x) - row.phi;
    if (deltaPhi > std::numbers::pi_v<float>)
      deltaPhi -= 2.f * std::numbers::pi_v<float>;
    else if (deltaPhi <= -std::numbers::pi_v<float>)
      deltaPhi += 2.f * std::numbers::pi_v<float>;
    const auto pixelType = ::lst::pixelSegmentType(ptIn, pca.ptErr, deltaPhi, settings.ptCut);
    candidate.pixelType = pixelType;
    if (pixelType == ::lst::PixelType::kInvalid)
      return;

    ::lst::PixelSegmentKinematics& kinematics = candidate.kinematics;
    ::lst::pixelSegmentPCAPosition(
        pca.px, pca.py, pca.pz, pca.dxy, pca.dz, kinematics.pcaX, kinematics.pcaY, kinematics.pcaZ);
    kinematics.ptPCA = std::sqrt(pca.px * pca.px + pca.py * pca.py);
    kinematics.etaPCA = kinematics.ptPCA > 0.f ? std::asinh(pca.pz / kinematics.ptPCA) : 0.f;
    kinematics.phiPCA = std::atan2(pca.py, pca.px);
    kinematics.lastHitX = row.x;
    kinematics.lastHitY = row.y;
    kinematics.lastHitZ = row.z;
    kinematics.dxy = pca.dxy;
    kinematics.dz = pca.dz;
    candidate.charge = pca.charge;
    candidate.superbin = ::lst::pixelSegmentSuperbin(kinematics.etaPCA, kinematics.phiPCA, pca.dz);
    candidate.ptIn = ptIn;
    candidate.ptErr = pca.ptErr;
    candidate.px = ptIn * std::cos(row.phi);
    candidate.py = ptIn * std::sin(row.phi);
    candidate.pz = ptIn * row.cotTheta;
    candidate.etaErr = pca.etaErr;
    candidate.eta = std::asinh(row.cotTheta);
    candidate.phi = row.phi;
    candidate.deltaPhi = deltaPhi;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice

#endif
