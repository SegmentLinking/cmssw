import FWCore.ParameterSet.Config as cms

hltInputLST = cms.EDProducer('LSTInputDeviceProducer@alpaka',
    ptCut = cms.double(0.8),
    originRadius = cms.double(0.2),
    originHalfLength = cms.double(0.2),
    pixelTracksSoA = cms.InputTag('hltPhase2PixelTrackTorchHighPuritySelector'),
    pixelTracks = cms.InputTag('hltPhase2PixelTracks'),
    pixelRecHitsSoA = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    caOTHits = cms.InputTag('hltPhase2OtRecHitsSoA'),
    otRecHitsSoA = cms.InputTag('hltSiPhase2RecHitsSoA'),
    otRecHits = cms.InputTag('hltSiPhase2RecHits'),
    beamSpot = cms.InputTag('hltOnlineBeamSpot'),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
