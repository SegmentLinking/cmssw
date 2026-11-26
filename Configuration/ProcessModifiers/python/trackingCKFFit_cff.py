import FWCore.ParameterSet.Config as cms

# Phase 2: the CKF fit (TrackProducer) in place of the mkFit final fit, in the steps where mkFit builds the track
# candidates (by default these use the mkFit fit)
trackingCKFFit = cms.Modifier()
