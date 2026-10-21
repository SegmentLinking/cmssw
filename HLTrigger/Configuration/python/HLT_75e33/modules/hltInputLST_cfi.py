import FWCore.ParameterSet.Config as cms

hltInputLST = cms.EDProducer('LSTInputProducer@alpaka',
    ptCut = cms.double(0.8),
    phase2OTRecHits = cms.InputTag('hltSiPhase2RecHits'),
    beamSpot = cms.InputTag('hltOnlineBeamSpot'),
    pixelSeeds = cms.VInputTag('hltInitialStepSeeds'),
    produceSeedHitPositions = cms.bool(False),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)


from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toModify(hltInputLST, produceSeedHitPositions = True)
