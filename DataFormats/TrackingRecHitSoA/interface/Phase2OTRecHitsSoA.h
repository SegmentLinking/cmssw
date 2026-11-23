#ifndef DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  // Phase-2 OT rechits made on the device from the clusters: one row per cluster of every OT module, the row being the
  // cluster's index in the cluster collection (= the Phase2TrackerRecHit1D index). Unlike OTRecHitsSoA: no per-stack
  // ordering and no local position or error columns. moduleIndex is the module's row in the Phase2OTModuleParams table
  // (GeomDetUnit::index() - firstModuleIndex), which holds the DetId, the local position errors and the surface; it is
  // meaningful only with the same tracker geometry.
  GENERATE_SOA_LAYOUT(Phase2OTRecHitsLayout,
                      SOA_COLUMN(float, xGlobal),
                      SOA_COLUMN(float, yGlobal),
                      SOA_COLUMN(float, zGlobal),
                      SOA_COLUMN(uint16_t, moduleIndex),
                      SOA_COLUMN(uint16_t, clusterSize))

  using Phase2OTRecHitsSoA = Phase2OTRecHitsLayout<>;
  using Phase2OTRecHitsView = Phase2OTRecHitsSoA::View;
  using Phase2OTRecHitsConstView = Phase2OTRecHitsSoA::ConstView;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsSoA_h
