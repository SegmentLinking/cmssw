#ifndef DataFormats_TrackingRecHitSoA_interface_alpaka_Phase2OTRecHitsSoACollection_h
#define DataFormats_TrackingRecHitSoA_interface_alpaka_Phase2OTRecHitsSoACollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using Phase2OTRecHitsSoACollection = PortableCollection<::reco::Phase2OTRecHitsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::Phase2OTRecHitsSoACollection, ::reco::Phase2OTRecHitsHost);

#endif  // DataFormats_TrackingRecHitSoA_interface_alpaka_Phase2OTRecHitsSoACollection_h
