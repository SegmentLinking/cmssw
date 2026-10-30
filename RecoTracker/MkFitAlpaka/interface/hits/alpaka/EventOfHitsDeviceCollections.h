#ifndef RecoTracker_MkFitAlpaka_interface_hits_alpaka_EventOfHitsDeviceCollections_h
#define RecoTracker_MkFitAlpaka_interface_hits_alpaka_EventOfHitsDeviceCollections_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // Views of the four tables of the device EventOfHits (the blocks of the EventOfHits product): the input hits plus
  // the binning of every layer, equivalent to MkFitCore's LayerOfHits.
  //  hits       : HitSoA, one row per input hit (pixel wrapper, then strip wrapper)
  //  layers     : LayerSoA, one row per mkFit layer (axes, bin-table offset, hitBegin/nHits of the binned order)
  //  binnedHits : BinnedHitSoA, MkFitCore internal order of every layer (rank = original index, phi/q/half-length/qbar)
  //  bins       : BinSoA, MkFitCore bin table (packed first|count) and dead flags of every layer
  struct EventOfHitsViews {
    ::mkfitdev::HitSoA::View hits;
    ::mkfitdev::LayerSoA::View layers;
    ::mkfitdev::BinnedHitSoA::View binnedHits;
    ::mkfitdev::BinSoA::View bins;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
