#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederEventHitsSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederEventHitsSoA_h

// The event's hits as the seeder reads them: the hits of the layers it bins (MkSeederLayerGeometrySoA::isBinned),
// layer after layer in increasing mkFit layer id (each layer's rows: MkSeederLayersSoA::firstHit, nHits), and the
// starts of the layers' (q, phi) bins.

#include <cstdint>

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // rows after the last hit: the CPU backends scan a layer's hits 8 at a time and read up to 7 rows past its end
  // (the rows are zeroed and never selected)
  constexpr int kHitsPadding = 16;

  // a hit's covariance: the 6 entries of the symmetric 3 x 3 matrix in a column order given by the layout
  using HitCovariance = Eigen::Matrix<float, 6, 1>;

  // One row per hit. Within a layer the rows are in the layer's bin order; the
  // binning's input has them in cluster order (the order of the layer's clusters in its HitVec).
  GENERATE_SOA_LAYOUT(MkSeederHitsLayout,
                      // the binning's coordinates: the azimuth, r and 1 / r, x and y relative to the beam line at
                      // the hit's z; z itself is global
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, z),
                      SOA_COLUMN(float, r),
                      SOA_COLUMN(float, invR),
                      SOA_COLUMN(float, xBeam),
                      SOA_COLUMN(float, yBeam),
                      // the global x and y (mkfit::Hit)
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      // the hit's index in its HitVec (pixel or outer tracker), the cluster index
                      SOA_COLUMN(uint32_t, clusterIndex),
                      // the cluster's length in rows and columns (Hit::spanRows(), spanCols(); spanCols() is the
                      // length along z in the barrel pixels, the shape cut's input)
                      SOA_COLUMN(int32_t, clusterRows),
                      SOA_COLUMN(int32_t, clusterCols),
                      // the module in the layer (Hit::detIDinLayer()): MkSeederLayerGeometrySoA::firstModule + it is
                      // the row of MkSeederModulesSoA
                      SOA_COLUMN(uint32_t, detIdInLayer),
                      // bin order: the hit's bin, q bin * kNPhiBins + phi bin; the binning's input: its sort key,
                      // mkFit layer id << 24 | q bin << 16 | phi M-bin (16 bits, 5 more than the phi bin)
                      SOA_COLUMN(uint32_t, binIndex),
                      // the fits: the hit's covariance as mkfit::Hit::errArray(), the packed lower triangle xx,
                      // xy, yy, xz, yz, zz
                      SOA_EIGEN_COLUMN(HitCovariance, covariance),
                      // a pixel hit's merged-cluster candidate bits (MkSeederConfig mrg_*; 0 when off): charge per path
                      // length (bit 0), size excess along the local x (bit 1), along the local y (bit 2)
                      SOA_COLUMN(uint8_t, mergedTag),
                      SOA_SCALAR(uint32_t, nHits))

  // One row per hit in cluster order: the hit's row in the bin order
  GENERATE_SOA_LAYOUT(MkSeederClusterOrderLayout, SOA_COLUMN(uint32_t, binOrderPosition))

  // One row per hit of the outer-tracker layers OT1-P, OT2-P, OT3-P (mkFit layers 4, 6, 8), in the order of the hits
  // (MkSeederLayersSoA::firstOTHit): the covariance as the extension's local residual reads it, xx, yy, xy, xz, yz, zz;
  // the PS-module confirmation (MkSeederPSMark.h; zeros when ps_mark is off): the confirming S hits and the best
  // one's q/pT estimate (1/GeV); v2: the confirming S hits per q/pT range (2 bits each) and the merged-S flags
  // (MkSeederPSMark.h, namespace psCounts), the q/pT estimate of the second-best S hit (0: fewer than 2 matches), the
  // low-pT counts (ps_low_pt; namespace psCountsLow)
  GENERATE_SOA_LAYOUT(MkSeederOTHitsLayout,
                      SOA_EIGEN_COLUMN(HitCovariance, covariance),
                      SOA_COLUMN(uint8_t, psMatches),
                      SOA_COLUMN(float, psQOverPt),
                      SOA_COLUMN(uint16_t, psCounts),
                      SOA_COLUMN(float, psQOverPt2),
                      SOA_COLUMN(uint16_t, psCountsLow))

  // One row per bin start: a layer's kNPhiBins * nQBins + 1 rows from MkSeederLayersSoA::firstBinStart, the
  // layer's first hit (relative to its firstHit) in each bin (q bin * kNPhiBins + phi bin), then its nHits
  GENERATE_SOA_LAYOUT(MkSeederBinStartsLayout, SOA_COLUMN(uint32_t, start))

  // the event's hits: the hits in bin order, their cluster-order rows, the
  // outer-tracker covariances, the bin starts
  GENERATE_SOA_BLOCKS(MkSeederEventHitsLayout,
                      SOA_BLOCK(hits, MkSeederHitsLayout),
                      SOA_BLOCK(clusterOrder, MkSeederClusterOrderLayout),
                      SOA_BLOCK(otHits, MkSeederOTHitsLayout),
                      SOA_BLOCK(binStarts, MkSeederBinStartsLayout))

  // a merged-cluster candidate with both an excess charge and an excess size (MkSeederHitsSoA::mergedTag)
  constexpr bool isStrongMergedCandidate(uint8_t tag) { return (tag & 1) && (tag & 6); }

  // The S hits of the PS modules of OT1-3 (mkFit layers 5, 7, 9) for the PS-module confirmation of the P hits
  // (ps_mark; MkSeederPSMark.h): per S layer its hits grouped by module (in cluster order within a module), the global
  // position and the cluster's width (Hit::spanRows())
  GENERATE_SOA_LAYOUT(MkSeederPSStripHitsLayout,
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_COLUMN(uint16_t, size))
  // per S layer its modules' first rows of MkSeederPSStripHitsSoA (n modules + 1 rows; the layers one after the other)
  GENERATE_SOA_LAYOUT(MkSeederPSStripStartsLayout, SOA_COLUMN(uint32_t, start))

  // the binning's input: the hits and the outer-tracker covariances in cluster order, and the S hits of the PS
  // modules, filled on the host
  GENERATE_SOA_BLOCKS(MkSeederUnbinnedHitsLayout,
                      SOA_BLOCK(hits, MkSeederHitsLayout),
                      SOA_BLOCK(otHits, MkSeederOTHitsLayout),
                      SOA_BLOCK(psStripHits, MkSeederPSStripHitsLayout),
                      SOA_BLOCK(psStripStarts, MkSeederPSStripStartsLayout))

  // The CPU backends' per-event caches of values the GPU computes where it uses them

  // per hit (the rows of MkSeederHitsSoA): the global azimuth and r, and the azimuth around the beam spot (the
  // triplet MLP's features)
  GENERATE_SOA_LAYOUT(MkSeederHitsCpuCacheLayout,
                      SOA_COLUMN(float, phiGlobal),
                      SOA_COLUMN(float, rGlobal),
                      SOA_COLUMN(float, phiBeam))
  // per outer-tracker hit (the rows of MkSeederOTHitsSoA): the frame of the extension's local residual, the hit's r,
  // the local x direction (in the transverse plane), the local y direction and the normal
  GENERATE_SOA_LAYOUT(MkSeederOTHitFramesLayout,
                      SOA_COLUMN(float, hitR),
                      SOA_COLUMN(float, localXDirectionX),
                      SOA_COLUMN(float, localXDirectionY),
                      SOA_COLUMN(float, localYDirectionX),
                      SOA_COLUMN(float, localYDirectionY),
                      SOA_COLUMN(float, localYDirectionZ),
                      SOA_COLUMN(float, normalX),
                      SOA_COLUMN(float, normalY),
                      SOA_COLUMN(float, normalZ))
  // per mkFit layer: the phi pad of the post steps' windows
  GENERATE_SOA_LAYOUT(MkSeederLayersCpuCacheLayout, SOA_COLUMN(float, phiWindowPad))

  GENERATE_SOA_BLOCKS(MkSeederCpuCacheLayout,
                      SOA_BLOCK(hits, MkSeederHitsCpuCacheLayout),
                      SOA_BLOCK(otHitFrames, MkSeederOTHitFramesLayout),
                      SOA_BLOCK(layers, MkSeederLayersCpuCacheLayout))

  using MkSeederHitsSoA = MkSeederHitsLayout<>;
  using MkSeederClusterOrderSoA = MkSeederClusterOrderLayout<>;
  using MkSeederOTHitsSoA = MkSeederOTHitsLayout<>;
  using MkSeederBinStartsSoA = MkSeederBinStartsLayout<>;
  using MkSeederEventHitsSoA = MkSeederEventHitsLayout<>;
  using MkSeederUnbinnedHitsSoA = MkSeederUnbinnedHitsLayout<>;
  using MkSeederPSStripHitsSoA = MkSeederPSStripHitsLayout<>;
  using MkSeederPSStripStartsSoA = MkSeederPSStripStartsLayout<>;
  using MkSeederHitsCpuCacheSoA = MkSeederHitsCpuCacheLayout<>;
  using MkSeederOTHitFramesSoA = MkSeederOTHitFramesLayout<>;
  using MkSeederLayersCpuCacheSoA = MkSeederLayersCpuCacheLayout<>;
  using MkSeederCpuCacheSoA = MkSeederCpuCacheLayout<>;

}  // namespace mkseeder

#endif
