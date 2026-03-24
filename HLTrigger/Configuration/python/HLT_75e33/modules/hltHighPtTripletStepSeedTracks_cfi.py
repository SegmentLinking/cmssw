import FWCore.ParameterSet.Config as cms

hltHighPtTripletStepSeedTracks = cms.EDProducer(
    "TrackFromSeedProducer",
    src = cms.InputTag("hltHighPtTripletStepSeeds"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    TTRHBuilder = cms.string("hltESPTTRHBuilderWithoutRefit")
)
