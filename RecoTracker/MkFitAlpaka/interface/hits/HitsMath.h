#ifndef RecoTracker_MkFitAlpaka_interface_hits_HitsMath_h
#define RecoTracker_MkFitAlpaka_interface_hits_HitsMath_h

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

namespace mkfitdev::hitsmath {

  // axis_base::from_R_to_?_bin returns std::floor(...) converted to unsigned short; gcc/x86 converts through a
  // 32-bit int (cvttss2si) and keeps the low 16 bits. Do the same explicitly so the device matches out-of-range cases.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint16_t toBin(float value) {
    return static_cast<uint16_t>(static_cast<uint32_t>(static_cast<int32_t>(floorf(value))));
  }

}  // namespace mkfitdev::hitsmath

#endif
