import FWCore.ParameterSet.Config as cms

#hltSeedTracks = cms.EDProducer("TrackCollectionFilterCloner",
#    copyExtras = cms.untracked.bool(True),
#    copyTrajectories = cms.untracked.bool(False),
#    minQuality = cms.string('highPurity'),
#    originalMVAVals = cms.InputTag("hltPhase2PixelTracksCutClassifier","MVAValues"),
#    originalQualVals = cms.InputTag("hltPhase2PixelTracksCutClassifier","QualityMasks"),
#    originalSource = cms.InputTag("hltPhase2PixelTracksCAExtension")
#)
hltSeedTracks = cms.EDProducer(
    "TrackFromSeedProducer",
    src = cms.InputTag("hltInitialStepTrajectorySeedsLST"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    TTRHBuilder = cms.string("hltESPTTRHBuilderWithoutRefit")
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
_hltSeedTracksLegacy = cms.EDProducer("TrackFromSeedProducer",
    src = cms.InputTag("hltMergedSeeds"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    TTRHBuilder = cms.string("hltESPTTRHBuilderWithoutRefit")
)
hltPhase2LegacyTracking.toReplaceWith(hltSeedTracks, _hltSeedTracksLegacy)

from Configuration.ProcessModifiers.hltPhase2LegacyTrackingPatatrackQuadsChain_cff import hltPhase2LegacyTrackingPatatrackQuads
_hltSeedTracksLegacyPatatrack = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    trackSrc = cms.InputTag("hltPhase2PixelTracksSoA"),
    outerTrackerRecHitSrc = cms.InputTag(""),
    useOTExtension = cms.bool(False),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
)
(hltPhase2LegacyTracking & hltPhase2LegacyTrackingPatatrackQuads).toReplaceWith(hltSeedTracks, _hltSeedTracksLegacyPatatrack)
