import FWCore.ParameterSet.Config as cms

hltHighPtTripletStepSeedTracksLST = cms.EDProducer(
    "TrackFromSeedProducer",
    src = cms.InputTag("highPtTripletStepSeeds"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    TTRHBuilder = cms.string("WithoutRefit")
)

