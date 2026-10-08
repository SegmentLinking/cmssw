#ifndef DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsHost_h
#define DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"

namespace reco {

  using Phase2OTRecHitsHost = PortableHostCollection<Phase2OTRecHitsSoA>;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_Phase2OTRecHitsHost_h
