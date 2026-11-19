import FWCore.ParameterSet.Config as cms

# the static tables of the mkFit seeder on alpaka (the chains, the layers' binning, the gap map, the modules and the
# material of the seed fits, the patterns' order, the triplet MLP), from its configuration (hltMkSeederConfig), the
# geometry and the triplet MLP's weights
def _addMkSeederESProducer(process):
    process.hltMkSeederESProducer = cms.ESProducer("MkSeederESProducer@alpaka",
        config = cms.ESInputTag("", "hltMkSeederConfig"),
        tripletMlpWeights = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-seeder-triplet-mlp.json'),
        appendToDataLabel = cms.string('')
    )

from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
modifyHLTForMkSeederESProducer_ = phase2MkSeeder.makeProcessModifier(_addMkSeederESProducer)
