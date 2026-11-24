import FWCore.ParameterSet.Config as cms

hltInitialStepTrackCandidates = cms.EDProducer('MkFitOutputConverter',
    batchSize = cms.int32(16),
    candMVASel = cms.bool(False),
    candCutSel = cms.bool(True),
    candMinNHitsCut = cms.int32(4),
    candMinPtCut = cms.double(0.85),
    candMinPtRelaxedCut = cms.double(0.8),
    candMinAbsEtaForRelaxedCut = cms.double(1.4),
    candWP = cms.double(0),
    doErrorRescale = cms.bool(True),
    mightGet = cms.optional.untracked.vstring,
    mkFitEventOfHits = cms.InputTag("hltMkFitEventOfHits"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitSeeds = cms.InputTag("hltInitialStepMkFitSeeds"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    propagatorAlong = cms.ESInputTag("","PropagatorWithMaterial"),
    propagatorOpposite = cms.ESInputTag("","PropagatorWithMaterialOpposite"),
    qualityMaxInvPt = cms.double(100),
    qualityMaxPosErr = cms.double(100),
    qualityMaxR = cms.double(120),
    qualityMaxZ = cms.double(280),
    qualityMinTheta = cms.double(0.01),
    qualitySignPt = cms.bool(True),
    seeds = cms.InputTag("hltInitialStepTrajectorySeedsLST"),
    tfDnnLabel = cms.string('trackSelectionTf'),
    tracks = cms.InputTag("hltInitialStepTrackCandidatesMkFit"),
    ttrhBuilder = cms.ESInputTag("","hltESPTTRHBuilderWithTrackAngle")
)


_hltInitialStepTrackCandidatesLegacy = cms.EDProducer("CkfTrackCandidateMaker",
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('initialStepTrajectoryBuilder')
    ),
    TrajectoryCleaner = cms.string('hltESPInitialStepTrajectoryCleanerBySharedHits'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        numberMeasurementsForFit = cms.int32(4),
        propagatorAlongTISE = cms.string('PropagatorWithMaterialParabolicMf'),
        propagatorOppositeTISE = cms.string('hltOppositeMaterialPropagatorParabolicMF')
    ),
    cleanTrajectoryAfterInOut = cms.bool(True),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(100000),
    maxSeedsBeforeCleaning = cms.uint32(1000),
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    reverseTrajectories = cms.bool(False),
    src = cms.InputTag("hltInitialStepSeeds"),
    useHitsSplitting = cms.bool(False)
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
# inner-tracker-only pixel CA (phase2ITonlyCA): keep the mkFit candidates down to pT 0.8 GeV (0.7 above |eta| 1.2),
# recovering the displaced and low-pT tracks the inner-tracker-only pixel seeds find
from Configuration.ProcessModifiers.phase2ITonlyCA_cff import phase2ITonlyCA
(phase2ITonlyCA & ~hltPhase2LegacyTracking).toModify(hltInitialStepTrackCandidates,
    candMinPtCut = 0.8,
    candMinPtRelaxedCut = 0.7,
    candMinAbsEtaForRelaxedCut = 1.2
)

hltPhase2LegacyTracking.toReplaceWith(hltInitialStepTrackCandidates, _hltInitialStepTrackCandidatesLegacy)


_hltInitialStepTrackCandidatesLST = cms.EDProducer('LSTOutputConverter',
    lstOutput = cms.InputTag('hltLST'),
    lstInput = cms.InputTag('hltInputLST'),
    includeT5s = cms.bool(True),
    includeNonpLSTSs = cms.bool(False),
    dropOTHitsPurePLS = cms.bool(True),
    maxITHitsToDropOTHitsPurePLS = cms.int32(3),
    produceSeeds = cms.bool(False),
    produceTrackCandidates = cms.bool(True),
    propagatorAlong = cms.ESInputTag('', 'PropagatorWithMaterial'),
    propagatorOpposite = cms.ESInputTag('', 'PropagatorWithMaterialOpposite'),
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

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(hltInitialStepTrackCandidates, _hltInitialStepTrackCandidatesLST)
