import FWCore.ParameterSet.Config as cms

hltInitialStepTrackFeatureExtractor = cms.EDProducer("TrackFeatureExtractor@alpaka",
    src = cms.InputTag("hltInitialStepTracks"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    vertices = cms.InputTag("hltPhase2PixelVertices"),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
