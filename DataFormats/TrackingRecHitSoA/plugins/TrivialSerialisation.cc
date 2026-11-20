#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsHost.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::TrackingRecHitHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::VectorHitsHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::OTRecHitsHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::Phase2OTRecHitsHost);
