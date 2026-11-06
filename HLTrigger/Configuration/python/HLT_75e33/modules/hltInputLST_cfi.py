import FWCore.ParameterSet.Config as cms

hltInputLST = cms.EDProducer('LSTInputDeviceProducer@alpaka',
    ptCut = cms.double(0.8),
    maxITHitsToDropOTHitsPurePLS = cms.int32(3),
    pixelTracksSoA = cms.InputTag('hltPhase2PixelTrackTorchHighPuritySelector'),
    pixelTracks = cms.InputTag('hltPhase2PixelTracks'),
    pixelRecHitsSoA = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    caOTHits = cms.InputTag('hltPhase2OtRecHitsSoA'),
    otRecHitsSoA = cms.InputTag('hltSiPhase2RecHitsSoA'),
    otRecHits = cms.InputTag('hltSiPhase2RecHits'),
    beamSpot = cms.InputTag('hltOnlineBeamSpot'),
    produceSeeds = cms.bool(False),
    propagator = cms.ESInputTag('', 'AnyDirectionAnalyticalPropagator'),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)

# With trackingLST, the converters in makeSeed mode read the pixel seeds of hltInputLST.
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(trackingLST | ngtScouting).toModify(hltInputLST, produceSeeds = True)
