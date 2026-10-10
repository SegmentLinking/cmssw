#ifndef RecoTracker_LST_plugins_alpaka_LSTInputDeviceKernels_h
#define RecoTracker_LST_plugins_alpaka_LSTInputDeviceKernels_h

#include <cstdint>

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/ParametrizedEngine/interface/BCyl.h"
#include "RecoTracker/LST/interface/LSTSeedFitModulesHost.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentParameters.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHitsFit.h"

namespace lstInputDevice {

  // hits per pixel track in the seed fit; longer tracks get no seed and no pLS
  constexpr int kMaxSeedHits = 32;

  // the seed fit and the pLS of one pixel track
  struct PixelSegmentCandidate {
    int8_t seedStatus;  // seedFromConsecutiveHits::Status, or kNoSeed
    int8_t pixelType;   // lst::PixelType; kInvalid: no pLS
    int8_t charge;
    uint8_t nHits;
    uint8_t nSlots;  // pLS hit slots
    uint8_t hitDetBits;
    int32_t superbin;
    float ptIn, ptErr, px, py, pz, etaErr, eta, phi, deltaPhi;
    lst::PixelSegmentKinematics kinematics;
    uint32_t slotDetId[lst::kMaxPLSHitsInHitsSoA];
    uint16_t slotClusterSize[lst::kMaxPLSHitsInHitsSoA];
    uint32_t slotHitIndex[lst::kMaxPLSHitsInHitsSoA];
    seedFromConsecutiveHits::State state;
  };
  constexpr int8_t kNoSeed = -1;

  // the beam line and the event-wide settings of the per-track kernel
  struct Settings {
    float beamSpotX, beamSpotY, beamSpotZ, beamSlopeX, beamSlopeY;
    float ptCut;
    float originRadius;  // GlobalTrackingRegion of the pixel-seed creator
    float originHalfLength;
    uint32_t nPixelHits;    // pixel-track hit indices below this are pixel hits, then the CA's OT hits
    uint32_t nCAOTModules;  // OT modules of the CA (caModuleStart has nCAOTModules + 1 entries)
    seedFromConsecutiveHits::Settings fit;
  };

  // per-event totals: seeds, pLS and pLS hits
  struct Totals {
    uint32_t nSeeds, nSegments, nSegmentHits;
  };

}  // namespace lstInputDevice

namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  using ::lstInputDevice::PixelSegmentCandidate;
  using ::lstInputDevice::Settings;
  using ::lstInputDevice::Totals;

  // per pixel track with an EDM track (edmIndex >= 0): the seed fit on its hits, the state at the beam line and the
  // pLS fields as LSTInputProducer computes them from the seed
  void fitPixelTracks(Queue& queue,
                      ::reco::TrackBlocksConstView tracks,
                      uint32_t nTracks,
                      int32_t const* edmIndex,
                      ::reco::TrackingRecHitConstView pixelHits,
                      uint32_t const* caModuleStart,
                      uint32_t const* caRecHitStart,
                      ::reco::Phase2OTRecHitsConstView otRecHits,
                      ::lst::LSTSeedFitModulesConstView modules,
                      magfieldparam::BCycl<float> const& field,
                      Settings const& settings,
                      PixelSegmentCandidate* candidates);

  // the seed index, pLS index and first pLS hit of each track in track order, and the totals
  void indexPixelSegments(Queue& queue,
                          PixelSegmentCandidate const* candidates,
                          uint32_t nTracks,
                          uint32_t* seedIndex,
                          uint32_t* segmentIndex,
                          uint32_t* hitOffset,
                          Totals* totals);

  // the LST input: every OT rechit, then the pLS and their hits
  void fillLSTInput(Queue& queue,
                    ::lst::LSTInputView input,
                    ::reco::Phase2OTRecHitsConstView otRecHits,
                    uint32_t nOTHits,
                    PixelSegmentCandidate const* candidates,
                    uint32_t nTracks,
                    uint32_t const* seedIndex,
                    uint32_t const* segmentIndex,
                    uint32_t const* hitOffset);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice

#endif
