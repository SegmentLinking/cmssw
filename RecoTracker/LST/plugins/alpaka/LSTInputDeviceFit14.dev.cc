#include "LSTInputDeviceFitKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  // the full fits of 14 hits, in a compilation unit of their own
  template void fitTracks<14>(Queue& queue,
                              uint32_t nTracks,
                              FitInput const* inputs,
                              float bField,
                              uint32_t const* lists,
                              Counters* counters,
                              FittedState* states);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice
