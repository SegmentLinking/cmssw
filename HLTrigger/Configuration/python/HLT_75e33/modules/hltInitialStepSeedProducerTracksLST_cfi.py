import FWCore.ParameterSet.Config as cms

hltInitialStepSeedTracksLST = cms.EDProducer(
    "TrackFromSeedProducer",
    src = cms.InputTag("initialStepSeeds"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    TTRHBuilder = cms.string("WithoutRefit")
)

