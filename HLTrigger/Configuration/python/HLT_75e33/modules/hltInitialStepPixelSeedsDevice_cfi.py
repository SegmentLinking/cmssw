import FWCore.ParameterSet.Config as cms

# The mkFit seeder's selected seeds as the device pixel seeds of the LST input on the device (phase2MkSeeder with
# trackingMkFitAlpaka): RecoTracker/MkFitAlpaka's customizeHLTforMkFitAlpaka gives them to hltInputLSTDevice and the
# device seed hand-off in place of the host LST input and seed path. Row k is the seed of pixel track k of
# hltPhase2PixelTracks (the same selection); the pLS of low-pT seeds that end on an outer-tracker hit start from the
# state at the last pixel hit ('plsStates': stateAtLastPixelHit, the settings of hltInputLST under phase2MkSeeder,
# which the customise checks)
hltInitialStepPixelSeedsDevice = cms.EDProducer("MkSeederDevicePixelSeedsProducer@alpaka",
    seeds = cms.InputTag("hltMkFitSeederSeedsMkFitDevice"),
    scores = cms.InputTag("hltMkFitSeederSeedsClassifiedDevice"),
    stateAtLastPixelHit = cms.bool(True),
    stateAtLastPixelHitMaxPt = cms.double(2.0),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)

# the module is added to the process only under the modifiers (the menus without them are unchanged); the sequence
# (HLTPhase2PixelTracksAndVerticesSequence) schedules it (labelled here: the paths are checked when they are loaded,
# before the process modifier adds it)
hltInitialStepPixelSeedsDevice.setLabel("hltInitialStepPixelSeedsDevice")
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
from Configuration.ProcessModifiers.trackingMkFitAlpaka_cff import trackingMkFitAlpaka
def _addInitialStepPixelSeedsDevice(process):
    process.hltInitialStepPixelSeedsDevice = hltInitialStepPixelSeedsDevice
hltInitialStepPixelSeedsDeviceModifier = cms.ProcessModifier(phase2MkSeeder & trackingMkFitAlpaka,
                                                             _addInitialStepPixelSeedsDevice)
