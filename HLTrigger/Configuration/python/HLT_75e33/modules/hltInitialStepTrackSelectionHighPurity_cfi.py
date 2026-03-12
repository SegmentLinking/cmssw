import FWCore.ParameterSet.Config as cms

hltInitialStepTrackSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("hltInitialStepTrackCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("hltInitialStepTrackCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("hltInitialStepTracks")
)

from Configuration.ProcessModifiers.trackTorchClassifier_cff import trackTorchClassifier
trackTorchClassifier.toModify(hltInitialStepTrackSelectionHighPurity, originalSource = cms.InputTag("hltInitialStepTrackTorchClassifierOutput"))
