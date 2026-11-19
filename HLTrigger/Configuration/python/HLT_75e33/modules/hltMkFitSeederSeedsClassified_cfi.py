import FWCore.ParameterSet.Config as cms

# The mkFit seeder's seeds that pass the seed classifier (hltMkFitSeederSeedsClassifiedDevice), as TrajectorySeeds, for
# hltInputLST (phase2MkSeeder); 'mkFitIndices': their rows in the seeder's products (the pixel tracks'), 'scores': the
# scores of all seeds
hltMkFitSeederSeedsClassified = cms.EDProducer("MkSeederTrajectorySeedConverter",
    seeds = cms.InputTag("hltMkFitSeederSeedsMkFitDevice"),
    scores = cms.InputTag("hltMkFitSeederSeedsClassifiedDevice"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    # without material: the step onto the last hit's module has zero length, and LST's input takes the seed's state
    # back to the beam line, where the module's material would make the errors too large
    propagatorAlong = cms.ESInputTag("", "hltESPAnalyticalPropagator"),
    propagatorOpposite = cms.ESInputTag("", "hltESPBwdAnalyticalPropagator"),
    # the seeder's states are on the last hit's module plane: no propagation onto its surface
    propagateToSurface = cms.bool(False)
)

# the module is added to the process only under the modifier (the menu without it is unchanged); the sequence
# (HLTPhase2PixelTracksAndVerticesSequence) schedules it (labelled here: the paths are checked when they are loaded,
# before the process modifier adds it)
hltMkFitSeederSeedsClassified.setLabel("hltMkFitSeederSeedsClassified")
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
def _addMkFitSeederSeedsClassified(process):
    process.hltMkFitSeederSeedsClassified = hltMkFitSeederSeedsClassified
hltMkFitSeederSeedsClassifiedModifier = cms.ProcessModifier(phase2MkSeeder, _addMkFitSeederSeedsClassified)
