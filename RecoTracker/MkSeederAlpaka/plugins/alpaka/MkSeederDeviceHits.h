#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDeviceHits_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDeviceHits_h

// The seeder's cluster-order hits made on the device from RecoTracker/MkFitAlpaka's device EventOfHits (its HitSoA
// block: one row per pixel cluster key, then one per OT cluster key, the hits of the RecoTracker/MkFit hit converters
// bitwise), instead of the host MkFitHitWrappers (MkSeederProducer with 'eventOfHits', a GPU backend; the CPU
// backends fill on the host from the same rows). The host counts the hits per layer from the cluster detsets (one row
// per cluster, the detset's mkFit layer) and gives per detset of a binned layer its first row in the cluster-order
// hits; a thread per HitSoA row writes the hit's values, its sort key and its OT covariance (as the host fill:
// MkSeederProducer::acquire, computeHitBinning) and widens its layer's extent. A row whose layer is not the one
// counted on the host (or an S hit whose module is not its detset's) is counted in 'mismatches' (the producer throws).

#include <cstdint>
#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"

#include "MkSeederDeviceTypes.h"

namespace mkseeder {

  // per mkFit layer: its binning (MkSeederLayerBinning) and nominal extent; binned = 0 for the layers not binned
  struct MkSeederDeviceHitsLayer {
    double qbarLowNominal = 0, qbarHighNominal = 0, qLowNominal = 0, qHighNominal = 0;
    float phiAxisMin = 0, phiFineScale = 0, qAxisMin = 0, qFineScale = 0, qFineLastBinHighEdge = 0;
    uint32_t qLastFineBin = 0;
    int32_t binned = 0, isDisc = 0;
  };

  // per cluster detset of a binned layer: its HitSoA rows [firstRow, firstRow + n), the cluster-order row of its first
  // hit, its first OT1-3 P covariance row (-1: none) and its mkFit layer; per detset of an OT1-3 S layer with ps_mark
  // (binned or not): its first row in the S hits grouped by module (MkSeederPSStripHitsSoA; -1: none) and its module
  struct MkSeederDeviceHitsDetSet {
    uint32_t firstRow = 0, n = 0, dest = 0;
    int32_t otHit = -1, layer = -1;
    int32_t psStrip = -1;
    uint32_t psModule = 0;
  };

  // the event's beam spot (MkSeederBeamSpot)
  struct MkSeederDeviceHitsBeamSpot {
    float x = 0, y = 0, z = 0, dxdz = 0, dydz = 0;
  };

}  // namespace mkseeder

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // The cluster-order hits (out: nTot rows, its nHits set) and OT1-3 P covariances (outOT) of the binned layers, and
  // the S hits of the detsets with a psStrip row (outPSStrip), from the HitSoA rows eoh (nPixelRows pixel rows first), the layers' extents widened in eventLayers (their nominal extents
  // set by the host); detSets in increasing firstRow. mismatches (device, zeroed here) counts the rows whose layer
  // differs from the detsets'.
  void fillHitsFromEventOfHits(Queue &queue,
                               ::mkfitdev::HitSoA::ConstView eoh,
                               uint32_t nPixelRows,
                               std::vector<::mkseeder::MkSeederDeviceHitsLayer> const &layers,
                               std::vector<::mkseeder::MkSeederDeviceHitsDetSet> const &detSets,
                               ::mkseeder::MkSeederDeviceHitsBeamSpot const &beamSpot,
                               mkseeder::MutableTableView<::mkseeder::MkSeederHitsSoA> out,
                               mkseeder::MutableTableView<::mkseeder::MkSeederOTHitsSoA> outOT,
                               mkseeder::MutableTableView<::mkseeder::MkSeederPSStripHitsSoA> outPSStrip,
                               uint32_t nTot,
                               mkseeder::MutableTableView<::mkseeder::MkSeederLayersSoA> eventLayers,
                               uint32_t *mismatches);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
