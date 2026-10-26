#ifndef RecoTracker_LST_plugins_alpaka_LSTInputDeviceKernels_h
#define RecoTracker_LST_plugins_alpaka_LSTInputDeviceKernels_h

#include <cstdint>

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/LST/interface/LSTSeedFitModulesHost.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentParameters.h"
#include "RecoTracker/LSTCore/interface/SeedStateFitTypes.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"

namespace lstInputDevice {

  // the fit types are shared with LSTCore's fit
  using ::lst::FitHit;
  using ::lst::FittedState;

  // The Broken Line Fit of a pixel track (the full fit) uses 4, 6, 8, 10, 12 or 14 hits, one fit kernel per size: a
  // track is fitted with the largest size not above its hit count, its hits taken with a uniform stride that keeps
  // the first and the last hit. Tracks with fewer than kMinFitHits hits, the fewest a pixel track has, are not
  // fitted. Even sizes only, because each fit kernel costs GPU time whatever the number of tracks it fits, while a
  // track whose hit count lies between two sizes loses only one hit to the stride.
  constexpr uint32_t kNumFitSizes = 6;
  constexpr uint32_t kMinFitHits = 4;
  constexpr uint32_t kMaxFitHits = 14;
  // the inner-tracker-only fit, for the seeds that the output converter may rebuild without their OT hits
  constexpr uint32_t kITFitHits = 3;
  // the largest pixel-track hit count: the hit counts are 8-bit columns
  constexpr uint32_t kMaxTrackHits = 255;

  // One list of tracks per fit: list 0 (kITFitList) holds the inner-tracker-only fits, then one list per fit size,
  // the smallest first: fitList(nFit) holds the full fits of nFit hits.
  constexpr uint32_t kITFitList = 0;
  constexpr uint32_t kNumFitLists = kNumFitSizes + 1;
  // Precondition: nFit is a fit size, an even number from kMinFitHits to kMaxFitHits; nothing is checked at run time.
  // The static_asserts below check every fit size.
  ALPAKA_FN_HOST_ACC constexpr uint32_t fitList(uint32_t nFit) { return 1 + (nFit - kMinFitHits) / 2; }
  static_assert(kMinFitHits > kITFitHits && kMaxFitHits == kMinFitHits + 2 * (kNumFitSizes - 1),
                "the fit sizes are kNumFitSizes even numbers from kMinFitHits, all above kITFitHits");
  static_assert(fitList(kMinFitHits) == 1 && fitList(kMaxFitHits) == kNumFitSizes,
                "the lists of the full fits are 1 to kNumFitSizes");

  // whether fitList maps the fit sizes, in order, to 1, 2, ..., kNumFitSizes
  constexpr bool fitListsHaveNoGaps() {
    uint32_t list = kITFitList;
    for (uint32_t n = kMinFitHits; n <= kMaxFitHits; n += 2) {
      if (fitList(n) != ++list)
        return false;
    }
    return list == kNumFitSizes;
  }
  static_assert(fitListsHaveNoGaps(), "fitList maps the fit sizes to 1 to kNumFitSizes without gaps");

  // The hits of the fits of one pixel track.
  struct FitInput {
    FitHit hits[kMaxFitHits];
    FitHit itHits[kITFitHits];
  };

  // The pLS of one pixel track, computed from its fitted state as LSTInputProducer computes it from a seed, and its
  // hit slots in the LST input.
  struct PixelSegmentCandidate {
    int8_t pixelType;  // lst::PixelType; kInvalid: no pLS
    int8_t charge;
    uint8_t nHits;
    uint8_t nITHits;
    uint8_t nSlots;  // pLS hit slots
    uint8_t hitDetBits;
    uint8_t pcaStatus;  // lst::PixelSegmentClosestApproach::Status
    int32_t superbin;
    float ptIn, ptErr, px, py, pz, etaErr, eta, phi, deltaPhi;
    ::lst::PixelSegmentKinematics kinematics;
    uint32_t slotDetId[::lst::kMaxPLSHitsInHitsSoA];
    uint16_t slotClusterSize[::lst::kMaxPLSHitsInHitsSoA];
    uint32_t slotHitIndex[::lst::kMaxPLSHitsInHitsSoA];
    // computed as for the fit (identical values); a slot hit may not be a fitted hit when the fit drops hits
    FitHit slotHits[::lst::kMaxPLSHitsInHitsSoA];
  };

  // the beam line and the event-wide settings of the per-track kernels
  struct Settings {
    float beamSpotX, beamSpotY, beamSpotZ, beamSlopeX, beamSlopeY;
    float ptCut;
    float bField;                    // GeV/cm, 1 / MagneticField::inverseBzAtOriginInGeV()
    uint32_t nPixelHits;             // pixel-track hit indices below this are pixel hits, then the CA's OT hits
    uint32_t nCAOTModules;           // OT modules of the CA (caModuleStart has nCAOTModules + 1 entries)
    uint32_t maxITHitsToDropOTHits;  // at most kITFitHits
  };

  // per-event totals: seeds, pLS and pLS hits
  struct Totals {
    uint32_t nSeeds, nSegments, nSegmentHits;
  };

  // The sizes of the fit lists and the per-event counters of the seed-state product.
  struct Counters {
    uint32_t listSize[kNumFitLists];
    uint32_t nNotFitted, nNonFinite, nChargeMismatch, nNonPositiveVariance;
    uint32_t nITCandidates, nITTooFewHits, nITNonFinite, nITChargeMismatch, nITNonPositiveVariance;
    uint32_t nPCANotConverged, nPCANonFinite;
  };

}  // namespace lstInputDevice

namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  using ::lstInputDevice::Counters;
  using ::lstInputDevice::FitInput;
  using ::lstInputDevice::FittedState;
  using ::lstInputDevice::PixelSegmentCandidate;
  using ::lstInputDevice::Settings;
  using ::lstInputDevice::Totals;

  // Per pixel track: an unfit state, and for a track with an EDM track (edmIndex >= 0) and a supported hit count,
  // the hits of its fits, its pLS hit slots and its entry in the lists of its fits.
  // Lists holds kNumFitLists lists of nTracks entries; counters must be zero.
  void prepareTracks(Queue& queue,
                     ::reco::TrackBlocksConstView tracks,
                     uint32_t nTracks,
                     int32_t const* edmIndex,
                     ::reco::TrackingRecHitConstView pixelHits,
                     uint32_t const* caModuleStart,
                     uint32_t const* caRecHitStart,
                     ::reco::Phase2OTRecHitsConstView otRecHits,
                     ::lst::LSTSeedFitModulesConstView modules,
                     Settings const& settings,
                     FitInput* inputs,
                     PixelSegmentCandidate* candidates,
                     FittedState* states,
                     FittedState* itStates,
                     uint32_t* lists,
                     Counters* counters);

  // The Broken Line Fit of the tracks in list fitList(N): the full fits of N hits into states. One instantiation per
  // fit size, each in its own compilation unit.
  template <uint32_t N>
  void fitTracks(Queue& queue,
                 uint32_t nTracks,
                 FitInput const* inputs,
                 float bField,
                 uint32_t const* lists,
                 Counters* counters,
                 FittedState* states);

  // The Broken Line Fit of the tracks in list kITFitList: the inner-tracker-only fits into itStates, in a
  // compilation unit of their own.
  void fitInnerTrackerTracks(Queue& queue,
                             uint32_t nTracks,
                             FitInput const* inputs,
                             float bField,
                             uint32_t const* lists,
                             Counters* counters,
                             FittedState* itStates);

  // Every fit of the producer: the inner-tracker-only fits, then the full fits of each fit size, the smallest first.
  void fitAllTracks(Queue& queue,
                    uint32_t nTracks,
                    FitInput const* inputs,
                    float bField,
                    uint32_t const* lists,
                    Counters* counters,
                    FittedState* states,
                    FittedState* itStates);

  // Per track with a full state of status kOk: the closest approach to the beam line and the pLS.
  void makePixelSegments(Queue& queue,
                         uint32_t nTracks,
                         Settings const& settings,
                         FittedState const* states,
                         PixelSegmentCandidate* candidates,
                         Counters* counters);

  // the seed index, pLS index and first pLS hit of each track in track order, and the totals
  void indexPixelSegments(Queue& queue,
                          FittedState const* states,
                          PixelSegmentCandidate const* candidates,
                          uint32_t nTracks,
                          uint32_t* seedIndex,
                          uint32_t* segmentIndex,
                          uint32_t* hitOffset,
                          Totals* totals);

  // the LST input: every OT rechit (OT row r from OT rechit r), then the pLS and their hits
  void fillLSTInput(Queue& queue,
                    ::lst::LSTInputView input,
                    ::reco::Phase2OTRecHitsConstView otRecHits,
                    uint32_t nOTHits,
                    float bField,
                    PixelSegmentCandidate const* candidates,
                    uint32_t nTracks,
                    uint32_t const* seedIndex,
                    uint32_t const* segmentIndex,
                    uint32_t const* hitOffset);

  // the seed-state product: one row per seed at its seed index, and the per-event counters
  void fillSeedStates(Queue& queue,
                      ::lst::PixelSeedStatesView seedStates,
                      float bField,
                      FittedState const* states,
                      FittedState const* itStates,
                      PixelSegmentCandidate const* candidates,
                      uint32_t nTracks,
                      uint32_t const* seedIndex,
                      Counters const* counters);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice

#endif
