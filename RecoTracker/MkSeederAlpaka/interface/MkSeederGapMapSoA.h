#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederGapMapSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederGapMapSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // The gap map of the barrel pixel layers (mkFit layers 0-3; mkfit::SensorGapMap): the ladders as planes, and per
  // ladder the z bins of its sensitive area, so that a line crossing a layer can be found to pass through a gap
  // between ladders or sensors (the layer then does not count as a hole).

  // the gap map's phi bins per layer (the ladders by phi bin), and the per-event ray tables' phi bins
  constexpr int kGapMapNPhiBins = 256;
  constexpr int kGapRayNPhiBins = 2048;
  constexpr int kGapMapNLayers = 4;

  // Four rows, one per barrel pixel layer: valid false without a map. The layer's mid radius and the edge margin;
  // its z gaps (rows firstZGap .. firstZGap + nZGaps of the z gaps, bins of 50 um from zGap0); its ladders (rows
  // firstLadder .. firstLadder + nLadders of the ladders); its phi bins' ladder lists (kGapMapNPhiBins + 1 rows from
  // firstPhiStart of the phi starts, each the first row of its list in the ladder index from firstLadderIndex)
  GENERATE_SOA_LAYOUT(MkSeederGapMapLayersLayout,
                      SOA_COLUMN(bool, valid),
                      SOA_COLUMN(float, rMid),
                      SOA_COLUMN(float, zGap0),
                      SOA_COLUMN(float, margin),
                      SOA_COLUMN(int32_t, nZGaps),
                      SOA_COLUMN(int32_t, firstZGap),
                      SOA_COLUMN(int32_t, firstLadder),
                      SOA_COLUMN(int32_t, nLadders),
                      SOA_COLUMN(int32_t, firstPhiStart),
                      SOA_COLUMN(int32_t, firstLadderIndex))

  // One row per ladder: its plane normalX * x + normalY * y = distance, the coordinate along it alongX * x + alongY *
  // y - alongOffset within +- halfWidth, and its sensitive z bins (nZBins bins of 50 um from zFirstBin, one bit each
  // in the bit words from firstBitWord)
  GENERATE_SOA_LAYOUT(MkSeederGapMapLaddersLayout,
                      SOA_COLUMN(float, normalX),
                      SOA_COLUMN(float, normalY),
                      SOA_COLUMN(float, distance),
                      SOA_COLUMN(float, alongX),
                      SOA_COLUMN(float, alongY),
                      SOA_COLUMN(float, alongOffset),
                      SOA_COLUMN(float, halfWidth),
                      SOA_COLUMN(float, zFirstBin),
                      SOA_COLUMN(int32_t, nZBins),
                      SOA_COLUMN(int32_t, firstBitWord))

  // the layers' z gap flags (1: a gap at that z bin)
  GENERATE_SOA_LAYOUT(MkSeederGapMapZGapsLayout, SOA_COLUMN(uint8_t, inGap))
  // the ladders' sensitive z bins, 64 per word
  GENERATE_SOA_LAYOUT(MkSeederGapMapBitsLayout, SOA_COLUMN(uint64_t, word))
  // per layer and phi bin: the first entry of its ladder list in the ladder index
  GENERATE_SOA_LAYOUT(MkSeederGapMapPhiStartsLayout, SOA_COLUMN(int32_t, firstEntry))
  // the ladder lists: the ladder (relative to the layer's firstLadder)
  GENERATE_SOA_LAYOUT(MkSeederGapMapLadderIndexLayout, SOA_COLUMN(int32_t, ladder))

  GENERATE_SOA_BLOCKS(MkSeederGapMapLayout,
                      SOA_BLOCK(layers, MkSeederGapMapLayersLayout),
                      SOA_BLOCK(ladders, MkSeederGapMapLaddersLayout),
                      SOA_BLOCK(zGaps, MkSeederGapMapZGapsLayout),
                      SOA_BLOCK(bits, MkSeederGapMapBitsLayout),
                      SOA_BLOCK(phiStarts, MkSeederGapMapPhiStartsLayout),
                      SOA_BLOCK(ladderIndex, MkSeederGapMapLadderIndexLayout))

  using MkSeederGapMapLayersSoA = MkSeederGapMapLayersLayout<>;
  using MkSeederGapMapLaddersSoA = MkSeederGapMapLaddersLayout<>;
  using MkSeederGapMapZGapsSoA = MkSeederGapMapZGapsLayout<>;
  using MkSeederGapMapBitsSoA = MkSeederGapMapBitsLayout<>;
  using MkSeederGapMapPhiStartsSoA = MkSeederGapMapPhiStartsLayout<>;
  using MkSeederGapMapLadderIndexSoA = MkSeederGapMapLadderIndexLayout<>;
  using MkSeederGapMapSoA = MkSeederGapMapLayout<>;

}  // namespace mkseeder

#endif
