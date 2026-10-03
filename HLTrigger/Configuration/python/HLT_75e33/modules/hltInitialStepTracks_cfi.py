import FWCore.ParameterSet.Config as cms

hltInitialStepTracks = cms.EDProducer("TrackProducer",
    AlgorithmName = cms.string('initialStep'),
    Fitter = cms.string('hltESPFlexibleKFFittingSmoother'),
    GeometricInnerState = cms.bool(False),
    MeasurementTracker = cms.string('hltESPMeasurementTracker'),
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    Propagator = cms.string('hltESPRungeKuttaTrackerPropagator'),
    SimpleMagneticField = cms.string(''),
    TTRHBuilder = cms.string('hltESPTTRHBuilderWithTrackAngle'),
    TrajectoryInEvent = cms.bool(False),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    clusterRemovalInfo = cms.InputTag(""),
    src = cms.InputTag("hltInitialStepTrackCandidates"),
    useHitsSplitting = cms.bool(False),
    useSimpleMF = cms.bool(False)
)

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
mtd_at_hlt.toModify(hltInitialStepTracks, TrajectoryInEvent = True)

_hltInitialStepTracksMkFitFit = cms.EDProducer("MkFitOutputTrackConverter",
    mkFitEventOfHits = cms.InputTag("hltMkFitEventOfHits"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    mkFitSeeds = cms.InputTag("hltInitialStepMkFitSeeds"),
    src = cms.InputTag("hltInitialStepTrackCandidatesMkFitFit"),
    seeds = cms.InputTag("hltInitialStepTrajectorySeedsLST"),
    ttrhBuilder = cms.ESInputTag('', 'hltESPTTRHBuilderWithTrackAngle'),
    propagatorAlong = cms.ESInputTag('', 'PropagatorWithMaterial'),
    propagatorOpposite = cms.ESInputTag('', 'PropagatorWithMaterialOpposite'),
    qualityMaxInvPt = cms.double(100),
    qualityMinTheta = cms.double(0.01),
    qualityMaxR = cms.double(120),
    qualityMaxZ = cms.double(280),
    qualityMaxPosErr = cms.double(100),
    qualitySignPt = cms.bool(True),
    NavigationSchool = cms.ESInputTag('', 'SimpleNavigationSchool'),
    TrajectoryInEvent = cms.bool(False),
    measurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    mightGet = cms.optional.untracked.vstring
)

from Configuration.ProcessModifiers.trackingMkFitFit_cff import trackingMkFitFit
trackingMkFitFit.toReplaceWith(hltInitialStepTracks, _hltInitialStepTracksMkFitFit)
(trackingMkFitFit & mtd_at_hlt).toModify(hltInitialStepTracks, TrajectoryInEvent = True)
