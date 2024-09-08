import FWCore.ParameterSet.Config as cms

hltPixelSeedInputLST = cms.EDProducer('LSTPixelSeedInputProducer',
    beamSpot = cms.InputTag('offlineBeamSpot'),
    seedTracks = cms.VInputTag(
        'initialStepSeedTracksLST',
        'highPtTripletStepSeedTracksLST'
    )
    mightGet = cms.optional.untracked.vstring
)

