#ifndef RecoTracker_FinalTrackSelectors_TrackFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_TrackFeaturesSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(TrackFeaturesSoALayout,
                    SOA_COLUMN(float, dxyBeamSpot),
                    SOA_COLUMN(float, dzBeamSpot),
                    SOA_COLUMN(float, dxyError),
                    SOA_COLUMN(float, dzError),
                    SOA_COLUMN(float, normalizedChi2),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, phi),
                    SOA_COLUMN(float, etaError),
                    SOA_COLUMN(float, phiError),
                    SOA_COLUMN(float, ndof),
                    SOA_COLUMN(float, lostInnerHits),
                    SOA_COLUMN(float, lostOuterHits),
                    SOA_COLUMN(float, layersWithoutMeas),
                    SOA_COLUMN(float, validPixelHits),
                    SOA_COLUMN(float, validStripHits))

using TrackFeaturesSoA = TrackFeaturesSoALayout<>;

// Define the SoA layout for track scores (output)
GENERATE_SOA_LAYOUT(TrackScoresSoALayout, SOA_COLUMN(float, score))

using TrackScoresSoA = TrackScoresSoALayout<>;

#endif  // RecoTracker_FinalTrackSelectors_TrackFeaturesSoA_h
