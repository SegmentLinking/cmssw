import FWCore.ParameterSet.Config as cms

# the mkFit seeder (phase2MkSeeder): the binning, the chain finder, the cleanings, the seed fits, the post step and the
# classifier's records on the device (any alpaka backend), the final seeds as device products. The device buffers'
# starting sizes: ttbar PU200 needs up to ~1.1M live candidates and ~0.22M quads; an event that overflows them
# runs again with larger ones (the stream then keeps 1.25 x its largest use).
hltMkFitSeederSeedsMkFitDevice = cms.EDProducer("MkSeederProducer@alpaka",
    pixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    stripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    config = cms.ESInputTag("", "hltMkSeederConfig"),
    tables = cms.ESInputTag("", ""),
    minCandidates = cms.uint32(1572864),
    minQuads = cms.uint32(262144),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)

# the module is added to the process only under the modifier (the menu without it is unchanged); the sequence
# (HLTPhase2PixelTracksAndVerticesSequence) schedules it (labelled here: the paths are checked when they are loaded,
# before the process modifier adds it)
hltMkFitSeederSeedsMkFitDevice.setLabel("hltMkFitSeederSeedsMkFitDevice")
# with trackingMkFitAlpaka the seeds go to the LST input on the device (hltInitialStepPixelSeedsDevice), which takes the
# low-pT seeds' states at their last pixel hit from here
from Configuration.ProcessModifiers.trackingMkFitAlpaka_cff import trackingMkFitAlpaka
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder as _phase2MkSeeder
# and the seeder reads its hits from the device EventOfHits (hltMkFitEventOfHits, the hits of the mkFit hit converters
# made on the device), so the host hit converters and the legacy OT rechits leave the menu
(_phase2MkSeeder & trackingMkFitAlpaka).toModify(hltMkFitSeederSeedsMkFitDevice,
                                                 produceLastPixelStates = cms.bool(True),
                                                 eventOfHits = cms.InputTag("hltMkFitEventOfHits"),
                                                 # set by customizeHLTforMkFitAlpaka when the EventOfHits' pixel
                                                 # rows are in pixel rechit SoA order
                                                 pixelSoA = cms.InputTag(""),
                                                 pixelHits = cms.InputTag(""),
                                                 stripHits = cms.InputTag(""))
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
def _addMkFitSeederSeedsMkFitDevice(process):
    process.hltMkFitSeederSeedsMkFitDevice = hltMkFitSeederSeedsMkFitDevice
hltMkFitSeederSeedsMkFitDeviceModifier = cms.ProcessModifier(phase2MkSeeder, _addMkFitSeederSeedsMkFitDevice)
