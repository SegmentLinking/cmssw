#include "LSTInputDeviceFitKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  // the full fits of 8 hits, in a compilation unit of their own
  template void fitTracks<8>(Queue& queue,
                             uint32_t nTracks,
                             FitInput const* inputs,
                             float bField,
                             uint32_t const* lists,
                             Counters* counters,
                             FittedState* states,
                             FittedState* itStates);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice
