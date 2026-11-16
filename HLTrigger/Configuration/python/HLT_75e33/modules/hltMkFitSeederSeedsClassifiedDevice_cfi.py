import FWCore.ParameterSet.Config as cms

# the seed classifier of the mkFit seeder (phase2MkSeeder): the scores of the seeder's final seeds (their features,
# hltMkFitSeederSeedsMkFitDevice) with the model's trees (hltMkSeederForestESProducer), and the selection (per |eta|
# bin thresholds, a separate one for 3-hit seeds); the seeds with fewer than 4 hits do not count in the sharing
# features (nsShareMinHits of hltMkFitSeederSeedsMkFitDevice, 4)
hltMkFitSeederSeedsClassifiedDevice = cms.EDProducer("MkSeederSeedClassifier@alpaka",
    seeds = cms.InputTag("hltMkFitSeederSeedsMkFitDevice"),
    forest = cms.ESInputTag("", ""),
    zFold = cms.bool(True),  # the model (v3) is trained on z-folded features
    threshold = cms.double(0.2),
    thresholdAbsEtaEdges = cms.vdouble(1.7, 2.4, 3.2),
    thresholdsVsAbsEta = cms.vdouble(0.34, 0.325, 0.34, 0.385),
    thresholdTriplets = cms.double(0.16),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)

# added to the process only under the modifier (the menu without it is unchanged), as hltMkFitSeederSeedsMkFitDevice
hltMkFitSeederSeedsClassifiedDevice.setLabel("hltMkFitSeederSeedsClassifiedDevice")
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
def _addMkFitSeederSeedsClassifiedDevice(process):
    process.hltMkFitSeederSeedsClassifiedDevice = hltMkFitSeederSeedsClassifiedDevice
hltMkFitSeederSeedsClassifiedDeviceModifier = cms.ProcessModifier(phase2MkSeeder, _addMkFitSeederSeedsClassifiedDevice)
