import FWCore.ParameterSet.Config as cms

hltPhase2OTHitsInputLST = cms.EDProducer('LSTPhase2OTHitsInputProducer',
    phase2OTRecHits = cms.InputTag('siPhase2RecHits'),
    mightGet = cms.optional.untracked.vstring
)

