import FWCore.ParameterSet.Config as cms

hltInitialStepTrackTorchClassifier = cms.EDProducer("TrackTorchClassifierAlpaka@alpaka",
    modelPath = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/best_model.pt'),
    features = cms.InputTag("hltInitialStepTrackFeatureExtractor"),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
