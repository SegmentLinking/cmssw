#ifndef RecoTracker_LSTCore_interface_alpaka_TCSeedStateFit_h
#define RecoTracker_LSTCore_interface_alpaka_TCSeedStateFit_h

// The Broken Line Fit seed state of every LST track candidate, on the device, after the track candidates are built.
// Compiled only in CMSSW: the fit uses Eigen, which the standalone build of LSTCore does not use, and the standalone
// does not run it.
#ifndef LST_STANDALONE

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "RecoTracker/LSTCore/interface/SeedStateFitTypes.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState {

  // The numbers of hits the fit uses, one fit kernel each: 14, 12, 10, 8, 6 and 5. A track candidate whose hit
  // selection keeps nSelected hits is fitted with all of them when nSelected is one of these numbers, with one hit
  // less when nSelected - 1 is, and with kMaxFitHits hits taken by a stride above kMaxFitHits + 1; below kMinFitHits
  // it is not fitted. The even sizes are the hit counts of the candidates whose two hits per layer all survive (T4,
  // T5, pT3, pT5); dropping one hit of an odd count keeps the number of fit kernels small, and the size 5 lets a
  // candidate left with 5 hits be fitted.
  constexpr uint32_t kNumFitSizes = 6;
  constexpr uint32_t kMaxFitHits = 14;
  constexpr uint32_t kMinFitHits = 5;

  // the number of hits of the fit of nSelected selected hits: the largest fit size not above nSelected, 0 below the
  // smallest one
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t fitSize(uint32_t nSelected) {
    if (nSelected >= 14)
      return 14;
    if (nSelected >= 12)
      return 12;
    if (nSelected >= 10)
      return 10;
    if (nSelected >= 8)
      return 8;
    if (nSelected >= 6)
      return 6;
    if (nSelected >= 5)
      return 5;
    return 0;
  }

  // the list of the fits of nFit hits, one per fit size: 0 for 14 hits, then 12, 10, 8, 6, and 5 for 5 hits
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t fitList(uint32_t nFit) {
    return nFit == kMinFitHits ? kNumFitSizes - 1 : (kMaxFitHits - nFit) / 2;
  }

  static_assert(fitSize(kMaxFitHits) == kMaxFitHits && fitSize(kMinFitHits) == kMinFitHits &&
                fitSize(kMinFitHits - 1) == 0);
  static_assert(fitList(14) == 0 && fitList(12) == 1 && fitList(10) == 2 && fitList(8) == 3 && fitList(6) == 4 &&
                fitList(5) == kNumFitSizes - 1);

  // The hits of the fit of one track candidate, inside out.
  struct FitInput {
    ::lst::FitHit hits[kMaxFitHits];
  };

  // The sizes of the fit lists and the per-event counters of the product, filled with atomics.
  struct Counters {
    uint32_t listSize[kNumFitSizes];
    uint32_t nNotFittedPLS, nTooFewHits, nNonFinite, nChargeMismatch, nNonPositiveVariance;
  };

  // The device buffers the fit works in, allocated by the caller for nTCs track candidates: inputs and states hold
  // nTCs entries, lists kNumFitSizes lists of nTCs entries, counters one entry.
  struct Scratch {
    FitInput* inputs;
    ::lst::FittedState* states;
    uint32_t* lists;
    Counters* counters;
  };

  // The seed states of the first nTCs track candidates of `candidates` into `seedStates` (nTCs rows),
  // with the hits of `input`.
  void fitSeedStates(Queue& queue,
                     ::lst::LSTInputConstView input,
                     ::lst::TrackCandidatesBaseConst candidates,
                     uint32_t nTCs,
                     ::lst::TCSeedStatesView seedStates,
                     Scratch const& scratch);

  // The steps of fitSeedStates.
  // Per track candidate: the hit selection, the hits of its fit and its entry in the list of its fit size, or for a
  // candidate that is not fitted an unfit state and its counter. The counters must be zero.
  void prepareFits(Queue& queue,
                   ::lst::LSTInputConstView input,
                   ::lst::TrackCandidatesBaseConst candidates,
                   uint32_t nTCs,
                   Scratch const& scratch);

  // The Broken Line Fit of the track candidates in the list of N hits, with the counters of its rejections.
  // One instantiation per compilation unit.
  template <uint32_t N>
  void fitCandidates(Queue& queue, ::lst::HitsBaseConst hits, uint32_t nTCs, Scratch const& scratch);

  // The rows and the per-event scalars of the product.
  void fillSeedStates(Queue& queue,
                      ::lst::HitsBaseConst hits,
                      ::lst::TrackCandidatesBaseConst candidates,
                      uint32_t nTCs,
                      ::lst::TCSeedStatesView seedStates,
                      Scratch const& scratch);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState

#endif  // LST_STANDALONE

#endif
