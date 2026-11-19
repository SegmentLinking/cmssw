#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederEventTablesSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederEventTablesSoA_h

// The seeder's small per-event tables, filled on the host: the layers' hit ranges and extents, the gap map's ray
// tables in the beam spot's frame, and the start pairs' work items. And the cluster tables of the classifier's
// features.

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // One row per mkFit layer id (MkSeederLayerGeometrySoA's rows): the layer's hits in MkSeederHitsSoA, its bin starts
  // in MkSeederBinStartsSoA (only a binned layer has hits), its first row in MkSeederOTHitsSoA (OT1-3 P; -1 else),
  // and the extents of its hits (qbar is r on a barrel layer and z on a disc, q the other;
  // they depend on the beam spot)
  GENERATE_SOA_LAYOUT(MkSeederLayersLayout,
                      SOA_COLUMN(uint32_t, firstHit),
                      SOA_COLUMN(uint32_t, nHits),
                      SOA_COLUMN(uint32_t, firstBinStart),
                      SOA_COLUMN(int32_t, firstOTHit),
                      SOA_COLUMN(double, qbarLo),
                      SOA_COLUMN(double, qbarHi),
                      SOA_COLUMN(double, qLo),
                      SOA_COLUMN(double, qHi),
                      // the pixel layers with hits in increasing id, in the first nPixelLayersWithHits rows (the
                      // layers the pixel fill searches)
                      SOA_COLUMN(int32_t, pixelLayerWithHits),
                      SOA_SCALAR(int32_t, nPixelLayersWithHits))

  // kGapMapNLayers * kGapRayNPhiBins rows (layer-major): per phi bin of a ray from the beam spot in the transverse
  // plane, the radii at which it crosses the layer's ladder planes (rInner <= rOuter; equal with one ladder), and
  // whether it passes near a ladder's long edge or crosses no ladder (mkfit::SensorGapMap::ray_table)
  GENERATE_SOA_LAYOUT(MkSeederGapRaysLayout,
                      SOA_COLUMN(float, rInner),
                      SOA_COLUMN(float, rOuter),
                      SOA_COLUMN(uint8_t, nearEdge))

  // One row per start pair of a chain: its first work item of stage b (the work items are the hits of the start
  // pairs' first layers, start pair after start pair)
  GENERATE_SOA_LAYOUT(MkSeederStartPairWorkLayout, SOA_COLUMN(uint32_t, firstWorkItem))

  GENERATE_SOA_BLOCKS(MkSeederEventTablesLayout,
                      SOA_BLOCK(layers, MkSeederLayersLayout),
                      SOA_BLOCK(gapRays, MkSeederGapRaysLayout),
                      // the start pairs of the +z side's chain (MkSeederESData::chains[0]), the -z side's
                      SOA_BLOCK(startPairsPlusZ, MkSeederStartPairWorkLayout),
                      SOA_BLOCK(startPairsMinusZ, MkSeederStartPairWorkLayout))

  // The clusters of the event's hits, by cluster index (the index in the pixel or outer-tracker HitVec): the
  // classifier's cluster features, and the position of the cluster's hit in the cluster order of MkSeederHitsSoA
  // (the binning's input; MkSeederClusterOrderSoA::binOrderPosition then gives its bin-order row; set only for the
  // clusters of the binned layers)
  GENERATE_SOA_LAYOUT(MkSeederPixelClustersLayout,
                      SOA_COLUMN(float, charge),
                      SOA_COLUMN(uint16_t, sizeX),
                      SOA_COLUMN(uint16_t, sizeY),
                      SOA_COLUMN(uint32_t, hitPosition))
  GENERATE_SOA_LAYOUT(MkSeederOTClustersLayout, SOA_COLUMN(uint16_t, size), SOA_COLUMN(uint32_t, hitPosition))

  GENERATE_SOA_BLOCKS(MkSeederClustersLayout,
                      SOA_BLOCK(pixelClusters, MkSeederPixelClustersLayout),
                      SOA_BLOCK(otClusters, MkSeederOTClustersLayout))

  using MkSeederLayersSoA = MkSeederLayersLayout<>;
  using MkSeederGapRaysSoA = MkSeederGapRaysLayout<>;
  using MkSeederStartPairWorkSoA = MkSeederStartPairWorkLayout<>;
  using MkSeederEventTablesSoA = MkSeederEventTablesLayout<>;
  using MkSeederPixelClustersSoA = MkSeederPixelClustersLayout<>;
  using MkSeederOTClustersSoA = MkSeederOTClustersLayout<>;
  using MkSeederClustersSoA = MkSeederClustersLayout<>;

}  // namespace mkseeder

#endif
