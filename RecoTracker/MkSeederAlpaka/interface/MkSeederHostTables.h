#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederHostTables_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederHostTables_h

#include <cstdint>
#include <vector>

namespace mkseeder {

  // A binned layer's (phi, q) axes on the host: the hits' sort keys use the fine bins (256 per q bin, 32 per phi bin;
  // the device tables, MkSeederLayerGeometrySoA, have the bins). The axes are mkFit's binnor axes
  // (axis_pow2_u1<float, unsigned short, 16, 11> in phi over [-pi, pi), axis<float, unsigned short, 16, 8> in q).
  struct MkSeederLayerBinning {
    int layerId = -1;
    bool isDisc = false;  // q is r (a disc) or z (a barrel layer)
    int nQBins = 0;
    float phiAxisMin = 0, phiFineScale = 0;
    float qAxisMin = 0, qFineScale = 0, qFineLastBinHighEdge = 0;
    uint16_t qLastFineBin = 0;
    // the layer's nominal extent: qbar (r of a barrel layer, z of a disc) and q
    double qbarLowNominal = 0, qbarHighNominal = 0, qLowNominal = 0, qHighNominal = 0;
  };

  // The host-only part of the static tables (MkSeederESData::hostTables): what the module needs on the host per event
  struct MkSeederHostTables {
    std::vector<MkSeederLayerBinning> binnedLayers;  // by increasing mkFit layer id
  };

}  // namespace mkseeder

#endif
