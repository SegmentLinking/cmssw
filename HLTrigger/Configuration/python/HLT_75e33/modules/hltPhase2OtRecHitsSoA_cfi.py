import FWCore.ParameterSet.Config as cms

hltPhase2OtRecHitsSoA = cms.EDProducer('Phase2OTCAHitsAlpaka@alpaka',
  otRecHitsSoA = cms.InputTag('hltSiPhase2RecHitsSoA'),
  otClusters = cms.InputTag('hltSiPhase2Clusters'),
  pixelRecHitSoASource = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
  beamSpot = cms.InputTag('hltOnlineBeamSpot'),
  mightGet = cms.optional.untracked.vstring,
  alpaka = cms.untracked.PSet(
    backend = cms.untracked.string('')
  )
)
