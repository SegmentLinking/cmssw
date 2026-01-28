import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracks = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("hltPhase2PixelTracksCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("hltPhase2PixelTracksCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("hltPhase2PixelTracksCAExtension")
)

from Configuration.ProcessModifiers.phase2LegacyTracking_cff import phase2LegacyTracking
_hltPhase2PixelTracksLegacy = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('pixelTrackCleanerBySharedHits'),
    Filter = cms.InputTag("hltPhase2PixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("hltPhase2PixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("hltPhase2PixelTracksHitSeeds"),
    mightGet = cms.optional.untracked.vstring,
    passLabel = cms.string('hltPhase2PixelTracks')
)
phase2LegacyTracking.toReplaceWith(hltPhase2PixelTracks, _hltPhase2PixelTracksLegacy)
