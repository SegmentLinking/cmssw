import FWCore.ParameterSet.Config as cms

hltESPPhase2StripCPEParams = cms.ESProducer('Phase2StripCPEParamsESProducerAlpaka@alpaka',
    Phase2StripCPE = cms.ESInputTag('hltESPPhase2StripCPE', 'Phase2StripCPEHLT'),
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
