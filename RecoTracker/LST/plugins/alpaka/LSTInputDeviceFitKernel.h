#ifndef RecoTracker_LST_plugins_alpaka_LSTInputDeviceFitKernel_h
#define RecoTracker_LST_plugins_alpaka_LSTInputDeviceFitKernel_h

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"

#include "LSTInputDeviceKernels.h"
#include "LSTInputDeviceTrackFit.h"

// The fit kernel of the device LST input for one number of fitted hits; each compilation unit that includes this
// header instantiates fitTracks for one N, so that the fits of different sizes are compiled separately.
namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  // one thread per track of the list: the full fit of N hits, or for N = kITFitHits the inner-tracker-only fit, and
  // the count of its rejections by reason
  template <uint32_t N>
  struct FitTracksKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  FitInput const* inputs,
                                  float bField,
                                  uint32_t const* list,
                                  Counters* counters,
                                  FittedState* states) const {
      constexpr bool kITOnly = N == ::lstInputDevice::kITFitHits;
      const uint32_t size = counters->listSize[::lstInputDevice::fitList(N)];
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, size)) {
        const uint32_t track = list[i];
        FittedState& state = states[track];
        ::lstInputDevice::FitHit const* hits = kITOnly ? &inputs[track].itHits[0] : &inputs[track].hits[0];
        fitState<N>(acc, hits, bField, state);
        uint32_t* counter = nullptr;
        switch (static_cast<::lst::SeedStateStatus>(state.status)) {
          case ::lst::SeedStateStatus::kNonFinite:
            counter = kITOnly ? &counters->nITNonFinite : &counters->nNonFinite;
            break;
          case ::lst::SeedStateStatus::kChargeMismatch:
            counter = kITOnly ? &counters->nITChargeMismatch : &counters->nChargeMismatch;
            break;
          case ::lst::SeedStateStatus::kNonPositiveVariance:
            counter = kITOnly ? &counters->nITNonPositiveVariance : &counters->nNonPositiveVariance;
            break;
          default:
            break;
        }
        if (counter != nullptr)
          alpaka::atomicAdd(acc, counter, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <uint32_t N>
  void fitTracks(Queue& queue,
                 uint32_t nTracks,
                 FitInput const* inputs,
                 float bField,
                 uint32_t const* lists,
                 Counters* counters,
                 FittedState* states,
                 FittedState* itStates) {
    if (nTracks == 0)
      return;
    // The list sizes are known on the device only: the grid covers every track, and the threads beyond the size of
    // the list return at once. One track per thread: the fit keeps its matrices in registers and local memory.
    constexpr uint32_t kThreads = 64;
    constexpr bool kITOnly = N == ::lstInputDevice::kITFitHits;
    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTracks, kThreads), kThreads),
        FitTracksKernel<N>{},
        inputs,
        bField,
        lists + ::lstInputDevice::fitList(N) * nTracks,
        counters,
        kITOnly ? itStates : states);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice

#endif
