import FWCore.ParameterSet.Config as cms

hltInitialStepTracks = cms.EDProducer("TrackProducer",
    AlgorithmName = cms.string('initialStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    GeometricInnerState = cms.bool(False),
    MeasurementTracker = cms.string(''),
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    SimpleMagneticField = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    TrajectoryInEvent = cms.bool(False),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    clusterRemovalInfo = cms.InputTag(""),
    src = cms.InputTag("hltInitialStepTrackCandidates"),
    useHitsSplitting = cms.bool(False),
    useSimpleMF = cms.bool(False)
)


_hltInitialStepTracksMkFitFit = cms.EDProducer("MkFitOutputTrackConverter",
    mkFitEventOfHits = cms.InputTag("hltMkFitEventOfHits"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    mkFitSeeds = cms.InputTag("hltInitialStepMkFitSeeds"),
    src = cms.InputTag("hltInitialStepTrackCandidatesMkFitFit"),
    seeds = cms.InputTag("hltInitialStepTrajectorySeedsLST"),
    ttrhBuilder = cms.ESInputTag('', 'WithTrackAngle'),
    propagatorAlong = cms.ESInputTag('', 'PropagatorWithMaterial'),
    propagatorOpposite = cms.ESInputTag('', 'PropagatorWithMaterialOpposite'),
    qualityMaxInvPt = cms.double(100),
    qualityMinTheta = cms.double(0.01),
    qualityMaxR = cms.double(120),
    qualityMaxZ = cms.double(280),
    qualityMaxPosErr = cms.double(100),
    qualitySignPt = cms.bool(True),
    calibrate = cms.bool(True),
    calibBinCenter = cms.vdouble(
      0.1704,
      0.6028,
      1.0188,
      1.2898,
      1.439,
      1.4908,
      1.55
    ),
    calibBinCoeff = cms.vdouble(
      1,
      1.0004,
      1.00014,
      1.0027,
      1.0029,
      1.0009,
      0.9999
    ),
    calibBinOffset = cms.vdouble(
      0.0016,
      0.0032,
      0.0033,
      0.0045,
      0.0005,
      0.0012,
      0.0003
    ),
    NavigationSchool = cms.ESInputTag('', 'SimpleNavigationSchool'),
    measurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    mightGet = cms.optional.untracked.vstring
)

from Configuration.ProcessModifiers.trackingMkFitFit_cff import trackingMkFitFit
trackingMkFitFit.toReplaceWith(hltInitialStepTracks, _hltInitialStepTracksMkFitFit)


_hltInitialStepTracksTrackProducer = hltInitialStepTracks.clone()

_hltInitialStepTracksLST = cms.EDProducer('LSTOutputConverter',
    lstOutput = cms.InputTag('hltLST'),
    lstInput = cms.InputTag('hltInputLST'),
    lstPixelSeeds = cms.InputTag('hltInputLST'),
    includeT5s = cms.bool(True),
    includeNonpLSTSs = cms.bool(False),
    produceSeeds = cms.bool(False),
    produceTrackCandidates = cms.bool(False),
    produceBLFTracks = cms.bool(True),
    lstBLFFitOutput = cms.InputTag('hltLST'),
    propagatorAlong = cms.ESInputTag('', 'PropagatorWithMaterial'),
    propagatorOpposite = cms.ESInputTag('', 'PropagatorWithMaterialOpposite'),
    SeedCreatorPSet = cms.PSet(
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        propagator = cms.string('PropagatorWithMaterial'),
        SeedMomentumForBOFF = cms.double(5),
        OriginTransverseErrorMultiplier = cms.double(1),
        MinOneOverPtError = cms.double(1),
        magneticField = cms.string(''),
        TTRHBuilder = cms.string('WithTrackAngle'),
        forceKinematicWithRegionDirection = cms.bool(False)
    )
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(hltInitialStepTracks, _hltInitialStepTracksLST)
