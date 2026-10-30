#ifndef RecoTracker_MkFitAlpaka_interface_navdev_NavResultSoA_h
#define RecoTracker_MkFitAlpaka_interface_navdev_NavResultSoA_h

// The device missing-hit navigation per (track, direction, layer): row (2 * track + direction) *
// nLayers + layer (direction 0 = inner / oppositeToMomentum, 1 = outer / alongMomentum; layer = the index in
// GeometricSearchTracker::allLayers()). det = the flat det index of compatibleDets().front(), or a NavResultCode.
// capacity * 2 * nLayers such rows, then 2 * capacity HEADER rows, row capacity * 2 * nLayers + 2 * track +
// direction: the start layer index of the device compatible-layer set when the set is exact (its layers are the rows
// of that (track, direction) other than kNavResultNotComputed), kNavResultHost when a decision of the set was within
// its margin (MkFitAlpakaNavDevice.h), kNavResultNotComputed when the device has no set. Outward from the final
// fit's outer state (TrackSoA outer) instead of the fit row: the start layer + kNavOuterTag, or kNavResultHostOuter.
#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkfitdev {

  enum NavResultCode : int32_t {
    kNavResultEmpty = -1,        // compatibleDets is empty: no entry
    kNavResultNotComputed = -2,  // layer not in the device's compatible set, or start outside the field box
    kNavResultHost = -3,         // search overflow / unsupported layout / a set layer not searched here: host search
    kNavResultHostOuter = -4     // header row: as kNavResultHost, the start state was the fit's outer state
  };
  // header row: start layer + kNavOuterTag = the device searched outward from the fit's outer state
  constexpr int32_t kNavOuterTag = 1 << 20;

  GENERATE_SOA_LAYOUT(NavResultLayout, SOA_COLUMN(int32_t, det))

  using NavResultSoA = NavResultLayout<>;
  using NavResultSoAView = NavResultSoA::View;
  using NavResultSoAConstView = NavResultSoA::ConstView;

}  // namespace mkfitdev

#endif
