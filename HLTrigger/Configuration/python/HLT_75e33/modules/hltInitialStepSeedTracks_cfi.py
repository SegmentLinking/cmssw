import FWCore.ParameterSet.Config as cms

hltInitialStepSeedTracks = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('pixelTrackCleanerBySharedHits'),
    Filter = cms.InputTag("hltPhase2PixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("hltPhase2PixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("hltPhase2PixelTracksHitSeeds"),
    mightGet = cms.optional.untracked.vstring,
    passLabel = cms.string('hltInitialStepSeedTracks')
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
_hltInitialStepSeedTracksLegacy = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('pixelTrackCleanerBySharedHits'),
    Filter = cms.InputTag("hltPhase2PixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("hltPhase2PixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("hltPhase2PixelTracksHitSeeds"),
    mightGet = cms.optional.untracked.vstring,
    passLabel = cms.string('hltInitialStepSeedTracks')
)
hltPhase2LegacyTracking.toReplaceWith(hltInitialStepSeedTracks, _hltInitialStepSeedTracksLegacy)
