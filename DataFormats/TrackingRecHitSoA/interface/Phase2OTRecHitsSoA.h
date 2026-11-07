#ifndef DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  // Phase-2 Outer Tracker rechits: one row per Phase2TrackerCluster1D, in the order of the cluster collection (which is
  // also the order of the Phase2TrackerRecHit1D collection made from it)
  GENERATE_SOA_LAYOUT(Phase2OTRecHitsLayout,
                      SOA_COLUMN(uint32_t, detId),          // raw DetId of the module
                      SOA_COLUMN(uint32_t, detectorIndex),  // GeomDetUnit::index() of the module
                      SOA_COLUMN(uint16_t, clusterSize),
                      SOA_COLUMN(float, xLocal),
                      SOA_COLUMN(float, yLocal),
                      SOA_COLUMN(float, xerrLocal),  // local position errors xx and yy (xy is zero)
                      SOA_COLUMN(float, yerrLocal),
                      SOA_COLUMN(float, xGlobal),
                      SOA_COLUMN(float, yGlobal),
                      SOA_COLUMN(float, zGlobal))

  using Phase2OTRecHitsSoA = Phase2OTRecHitsLayout<>;
  using Phase2OTRecHitsView = Phase2OTRecHitsSoA::View;
  using Phase2OTRecHitsConstView = Phase2OTRecHitsSoA::ConstView;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsSoA_h
