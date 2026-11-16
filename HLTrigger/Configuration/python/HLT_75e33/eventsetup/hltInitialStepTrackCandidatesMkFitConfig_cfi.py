import FWCore.ParameterSet.Config as cms

hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
    ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
    appendToDataLabel = cms.string(''),
    config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-lstStep.json'),
    backwardSearchMinPixelLayers = cms.int32(2),
    backwardSearchPromptMaxD0 = cms.double(2.0),
    duplicateCleaningLowPtFraction = cms.double(0.10),
    duplicateCleaningLowPtRampStart = cms.double(5.0),
    duplicateCleaningLowPtRampEnd = cms.double(10.0),
    duplicateCleaningLowPtMaxRelDiffInvPt = cms.double(0.10),
    duplicateCleaningLowPtRequireSameCharge = cms.bool(True),
    duplicateCleaningLowPtMaxD0 = cms.double(2.0),
    duplicateCleaningLowPtMaxDRTimesPt = cms.double(0.10),
    duplicateCleaningMinUniqueHitsToKeep = cms.int32(9),
    duplicateCleaningMinPtUniqueHitsToKeep = cms.double(10.0),
    maxClusterSize = cms.uint32(8),
    minPt = cms.double(0.9)
)

# with the mkFit seeder's seeds: mkFit's seed cleaner (phase1:default) on, the rest as mkfit-phase2-lstStep.json
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
phase2MkSeeder.toModify(hltInitialStepTrackCandidatesMkFitConfig,
    config = 'RecoTracker/MkFit/data/mkfit-phase2-lstStep-mkseeder.json'
)
