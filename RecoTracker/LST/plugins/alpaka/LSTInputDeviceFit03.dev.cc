#include "LSTInputDeviceFitKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  // the inner-tracker-only fits (3 hits), in a compilation unit of their own
  static_assert(::lstInputDevice::kITFitHits == 3);
  void fitInnerTrackerTracks(Queue& queue,
                             uint32_t nTracks,
                             FitInput const* inputs,
                             float bField,
                             uint32_t const* lists,
                             Counters* counters,
                             FittedState* itStates) {
    launchFits<::lstInputDevice::kITFitHits, true>(
        queue, nTracks, inputs, bField, ::lstInputDevice::kITFitList, lists, counters, itStates);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice
