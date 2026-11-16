import FWCore.ParameterSet.Config as cms

# the configuration of the mkFit seeder (its windows, layer patterns and chain, fake cuts, cleanings, outer-tracker
# extension, pixel fill and seed fit): the working point of the phase-2 HLT seeding
def _addMkSeederConfig(process):
    process.hltMkSeederConfig = cms.ESProducer("MkSeederConfigESProducer",
        ComponentName = cms.string('hltMkSeederConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-seeder-6l-l6c10-effB.json')
    )

from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
modifyHLTForMkSeederConfig_ = phase2MkSeeder.makeProcessModifier(_addMkSeederConfig)
