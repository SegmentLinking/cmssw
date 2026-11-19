import FWCore.ParameterSet.Config as cms

# the trees of the mkFit seeder's seed classifier (hltMkFitSeederSeedsClassifiedDevice)
def _addMkSeederForestESProducer(process):
    process.hltMkSeederForestESProducer = cms.ESProducer("MkSeederForestESProducer@alpaka",
        model = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-seed-classifier-6l-v3-n100.json'),
        appendToDataLabel = cms.string('')
    )

from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
modifyHLTForMkSeederForestESProducer_ = phase2MkSeeder.makeProcessModifier(_addMkSeederForestESProducer)
