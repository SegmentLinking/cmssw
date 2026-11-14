#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaOTCAHitsKernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaOTCAHitsKernels_h

// The CA OT layers' hit SoA (= Phase2OTRecHitsSoAConverter's product) selected on the
// device from the OT rechit SoA.
#include <cstdint>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTRecHitSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits {

  struct CAHitsParams {
    double bsx, bsy, bsz;     // beam spot (reco::BeamSpot x0/y0/z0)
    int32_t firstIndex;       // first OT GeomDet index (OTCpeModule table)
    uint32_t nOT;             // OT rows
    uint16_t modulesInPixel;  // detectorIndex = modulesInPixel + P-module offset
  };

  // for every OT row on a P module of the OT barrel (pOffset[module - firstIndex] >= 0):
  // row idx = hitStart[off] + (key - keyStart[off]) of the CA hit SoA
  void runOTCAHits(Queue& queue,
                   ::mkfitdev::OTRecHitSoA::ConstView ot,
                   ::mkfitdev::OTCpeModule const* table,
                   int32_t const* pOffset,
                   uint32_t const* hitStart,
                   uint32_t const* keyStart,
                   CAHitsParams const& p,
                   ::reco::TrackingRecHitView hits);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits

#endif
