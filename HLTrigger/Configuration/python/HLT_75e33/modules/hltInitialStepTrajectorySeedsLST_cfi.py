import FWCore.ParameterSet.Config as cms

hltInitialStepTrajectorySeedsLST = cms.EDProducer('LSTOutputConverter',
    lstOutput = cms.InputTag('hltLST'),
    lstInput = cms.InputTag('hltInputLST'),
    includeT5s = cms.bool(True),
    includeNonpLSTSs = cms.bool(True),
    dropOTHitsPurePLS = cms.bool(True),
    maxITHitsToDropOTHitsPurePLS = cms.int32(3),
    produceSeeds = cms.bool(True),
    produceTrackCandidates = cms.bool(False),
    propagatorAlong = cms.ESInputTag('', 'PropagatorWithMaterial'),
    propagatorOpposite = cms.ESInputTag('', 'PropagatorWithMaterialOpposite'),
    seedsFromStates = cms.bool(True),
    pixelSeedStates = cms.InputTag('hltInputLST'),
    tcSeedStates = cms.InputTag('hltLST'),
    pixelTrackHits = cms.InputTag('hltInputLST'),
    propagator = cms.ESInputTag('', 'AnyDirectionAnalyticalPropagator'),
    SeedCreatorPSet = cms.PSet(
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        propagator = cms.string('PropagatorWithMaterial'),
        SeedMomentumForBOFF = cms.double(5),
        OriginTransverseErrorMultiplier = cms.double(1),
        MinOneOverPtError = cms.double(1),
        magneticField = cms.string(''),
        TTRHBuilder = cms.string('hltESPTTRHBuilderWithTrackAngle'),
        forceKinematicWithRegionDirection = cms.bool(False)
    )
)

# Under these menus hltLST makes no track-candidate seed states and hltInputLST makes the pixel seeds makeSeed reads.
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(trackingLST | ngtScouting).toModify(hltInitialStepTrajectorySeedsLST, seedsFromStates = False)
