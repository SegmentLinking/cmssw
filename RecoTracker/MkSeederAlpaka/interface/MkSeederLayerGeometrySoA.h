#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederLayerGeometrySoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederLayerGeometrySoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // The number of phi bins of every binned layer (a power of two: the bin index wraps with a mask)
  constexpr int kNPhiBins = 2048;

  // The seeder's static description of the mkFit layers, one row per mkFit layer id: the kind of layer, its extent,
  // the axes of the (phi, q) binning of its hits (q is z on a barrel layer and r on a disc), and its first module in
  // MkSeederModulesSoA (the modules of a layer are consecutive, indexed by Hit::detIDinLayer()).
  GENERATE_SOA_LAYOUT(MkSeederLayerGeometryLayout,
                      // the seeder bins the hits of this layer (a layer of a pattern or of a chain)
                      SOA_COLUMN(bool, isBinned),
                      // the hits are in the pixel HitVec (mkfit::LayerInfo::is_pixel())
                      SOA_COLUMN(bool, isPixel),
                      SOA_COLUMN(bool, isBarrel),
                      // a pixel layer for the chains' crossing envelopes (SeedLayerEnvelopes::is_pix())
                      SOA_COLUMN(bool, isPixelEnvelope),
                      // the binning's q is r (a disc) or z (a barrel layer); only for a binned layer
                      SOA_COLUMN(bool, isDisc),
                      // the q axis: bin = q <= qAxisMin ? 0 : q >= qAxisLastBinHighEdge ? qLastBin
                      //                   : floor((q - qAxisMin) * qAxisScale)
                      SOA_COLUMN(int32_t, nQBins),
                      SOA_COLUMN(int32_t, qLastBin),
                      SOA_COLUMN(float, qAxisMin),
                      SOA_COLUMN(float, qAxisScale),
                      SOA_COLUMN(float, qAxisLastBinHighEdge),
                      // the phi axis: bin = floor((phi - phiAxisMin) * phiAxisScale) & (kNPhiBins - 1)
                      SOA_COLUMN(float, phiAxisMin),
                      SOA_COLUMN(float, phiAxisScale),
                      // the layer's extent (mkfit::LayerInfo)
                      SOA_COLUMN(float, rIn),
                      SOA_COLUMN(float, rOut),
                      SOA_COLUMN(float, zMin),
                      SOA_COLUMN(float, zMax),
                      SOA_COLUMN(float, zMean),
                      SOA_COLUMN(int32_t, firstModule))

  using MkSeederLayerGeometrySoA = MkSeederLayerGeometryLayout<>;

}  // namespace mkseeder

#endif
