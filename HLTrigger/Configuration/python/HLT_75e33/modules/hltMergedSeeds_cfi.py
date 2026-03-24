import FWCore.ParameterSet.Config as cms

hltMergedSeeds = cms.EDProducer("SeedCombiner",
    seedCollections = cms.VInputTag(
        cms.InputTag("hltInitialStepSeeds"),
        cms.InputTag("hltHighPtTripletStepSeeds"),
    )
)
