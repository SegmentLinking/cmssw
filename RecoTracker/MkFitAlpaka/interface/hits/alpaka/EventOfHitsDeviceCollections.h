#ifndef RecoTracker_MkFitAlpaka_interface_hits_alpaka_EventOfHitsDeviceCollections_h
#define RecoTracker_MkFitAlpaka_interface_hits_alpaka_EventOfHitsDeviceCollections_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsHostCollections.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using HitsDeviceCollection = PortableCollection<::mkfitdev::HitSoA>;
  using LayersDeviceCollection = PortableCollection<::mkfitdev::LayerSoA>;
  using BinnedHitsDeviceCollection = PortableCollection<::mkfitdev::BinnedHitSoA>;
  using BinsDeviceCollection = PortableCollection<::mkfitdev::BinSoA>;

  // Device EventOfHits: the input hits plus the MkFitCore-LayerOfHits-equivalent binning of every layer.
  //  hits       : HitSoA, one row per input hit (pixel wrapper, then strip wrapper)
  //  layers     : LayerSoA, one row per mkFit layer (axes, bin-table offset, hitBegin/nHits of the binned order)
  //  binnedHits : BinnedHitSoA, MkFitCore internal order of every layer (rank = original index, phi/q/half-length/qbar)
  //  bins       : BinSoA, MkFitCore bin table (packed first|count) and dead flags of every layer
  // Views of the four tables, from an EventOfHitsDevice or from the blocks of the EventOfHits product.
  struct EventOfHitsViews {
    ::mkfitdev::HitSoA::View hits;
    ::mkfitdev::LayerSoA::View layers;
    ::mkfitdev::BinnedHitSoA::View binnedHits;
    ::mkfitdev::BinSoA::View bins;
  };

  struct EventOfHitsDevice {
    HitsDeviceCollection hits;
    LayersDeviceCollection layers;
    BinnedHitsDeviceCollection binnedHits;
    BinsDeviceCollection bins;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
