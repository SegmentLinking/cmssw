import FWCore.ParameterSet.Config as cms

hltSiPhase2RecHitsSoA = cms.EDProducer('Phase2TrackerRecHitsAlpaka@alpaka',
    src = cms.InputTag('hltSiPhase2Clusters'),
    cpeParams = cms.ESInputTag('', ''),
    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
