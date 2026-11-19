#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederPatternsSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederPatternsSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // The patterns' order of the cleaning: one row per configured layer combination (each pattern, then its mirror on
  // the -z side), sorted by layerKey, the four mkFit layers + 1 packed in bytes (the first layer in the top byte, -1
  // for the missing fourth layer of a triplet); patternIndex is its position in the configured order. A combination
  // not in the table ranks after all of them, at nPatternPositions.
  GENERATE_SOA_LAYOUT(MkSeederPatternsLayout,
                      SOA_COLUMN(uint32_t, layerKey),
                      SOA_COLUMN(int32_t, patternIndex),
                      SOA_SCALAR(int32_t, nPatternPositions))

  using MkSeederPatternsSoA = MkSeederPatternsLayout<>;

}  // namespace mkseeder

#endif
