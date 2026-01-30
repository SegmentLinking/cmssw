import FWCore.ParameterSet.Config as cms

mkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
    appendToDataLabel = cms.string('')
)

def _addProcesshltInitialStepMkFitConfig(process):
    process.hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-initialStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0.8)
    )

def _addProcesshltLSTStepMkFitConfig(process):
    process.hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-lstStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0.9)
    )

from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
modifyConfigurationForTrackingMkFithltInitialStepMkFitConfig_ = (~seedingLST).makeProcessModifier(_addProcesshltInitialStepMkFitConfig)
modifyConfigurationForTrackingMkFithltLSTStepMkFitConfig_ = seedingLST.makeProcessModifier(_addProcesshltLSTStepMkFitConfig)
