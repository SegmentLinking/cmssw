#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederShapeSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederShapeSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // The shape cut of the barrel pixel layers (mkFit layers 0-3): a hit's cluster length along z (Hit::spanCols())
  // must be within the band of its |cot theta| bin.

  // Four rows, one per barrel pixel layer: the band's bins (rows firstBin .. firstBin + nBins of the bands) and their
  // width in |cot| (bin = min(|cot| * invBinWidth, nBins - 1)); valid false: no cut on that layer
  GENERATE_SOA_LAYOUT(MkSeederShapeLayersLayout,
                      SOA_COLUMN(bool, valid),
                      SOA_COLUMN(int32_t, firstBin),
                      SOA_COLUMN(int32_t, nBins),
                      SOA_COLUMN(float, invBinWidth))

  // One row per |cot| bin: the allowed cluster lengths
  GENERATE_SOA_LAYOUT(MkSeederShapeBandsLayout, SOA_COLUMN(int32_t, spanLow), SOA_COLUMN(int32_t, spanHigh))

  GENERATE_SOA_BLOCKS(MkSeederShapeLayout,
                      SOA_BLOCK(layers, MkSeederShapeLayersLayout),
                      SOA_BLOCK(bands, MkSeederShapeBandsLayout))

  using MkSeederShapeLayersSoA = MkSeederShapeLayersLayout<>;
  using MkSeederShapeBandsSoA = MkSeederShapeBandsLayout<>;
  using MkSeederShapeSoA = MkSeederShapeLayout<>;

  // the barrel pixel layers with a shape band
  constexpr int kNShapeLayers = 4;

}  // namespace mkseeder

#endif
