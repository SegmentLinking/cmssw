#ifndef RecoTracker_FinalTrackSelectors_TrackFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_TrackFeaturesSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(TrackFeaturesSoALayout,
                    SOA_COLUMN(float, pt),
                    SOA_COLUMN(float, innerMomentumX),
                    SOA_COLUMN(float, innerMomentumY),
                    SOA_COLUMN(float, innerMomentumZ),
                    SOA_COLUMN(float, innerMomentumRho),
                    SOA_COLUMN(float, outerMomentumX),
                    SOA_COLUMN(float, outerMomentumY),
                    SOA_COLUMN(float, outerMomentumZ),
                    SOA_COLUMN(float, outerMomentumRho),
                    SOA_COLUMN(float, ptError),
                    SOA_COLUMN(float, dxyBestVertex),
                    SOA_COLUMN(float, dzBestVertex),
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
                    SOA_COLUMN(float, layersOffInner),
                    SOA_COLUMN(float, layersOffOuter),
                    SOA_COLUMN(float, layersWithoutMeas),
                    SOA_COLUMN(float, validPixelHits),
                    SOA_COLUMN(float, validStripHits))

using TrackFeaturesSoA = TrackFeaturesSoALayout<>;

// Define the SoA layout for track scores (output)
GENERATE_SOA_LAYOUT(TrackScoresSoALayout, SOA_COLUMN(float, score))

using TrackScoresSoA = TrackScoresSoALayout<>;

#endif  // RecoTracker_FinalTrackSelectors_TrackFeaturesSoA_h
