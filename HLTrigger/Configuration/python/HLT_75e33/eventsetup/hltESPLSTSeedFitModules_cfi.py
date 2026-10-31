import FWCore.ParameterSet.Config as cms

hltESPLSTSeedFitModules = cms.ESProducer('LSTSeedFitModulesESProducer@alpaka',
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
