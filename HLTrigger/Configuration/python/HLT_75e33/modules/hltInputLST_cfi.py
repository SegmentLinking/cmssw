import FWCore.ParameterSet.Config as cms

hltInputLST = cms.EDProducer('LSTInputProducer@alpaka',
    ptCut = cms.double(0.8),
    phase2OTRecHits = cms.InputTag('hltSiPhase2RecHits'),
    beamSpot = cms.InputTag('hltOnlineBeamSpot'),
    pixelSeeds = cms.VInputTag('hltInitialStepSeeds'),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)

from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
# The mkFit seeder's seeds carry outer-tracker hits besides the pixel ones. The pixel line segments are built from all
# the seed's hits (the OT hit constrains the segment), and the segments of low-pT seeds start from the seed's state
# propagated back to the last pixel hit, since the multiple scattering in the material in between makes the state at
# the OT hit a poor estimate of the direction at the pixel hits.
phase2MkSeeder.toModify(hltInputLST, pixelSeeds = ['hltMkFitSeederSeedsClassified'],
    stateAtLastPixelHit = cms.bool(True), stateAtLastPixelHitMaxPt = cms.double(2.0))
