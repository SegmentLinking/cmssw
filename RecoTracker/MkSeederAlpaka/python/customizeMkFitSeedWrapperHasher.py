# Adds the validation hasher (MkFitSeedWrapperHasher) to a phase2MkSeeder HLT process.
# Use as a cmsDriver customisation:
#   --customise RecoTracker/MkSeederAlpaka/customizeMkFitSeedWrapperHasher.addHasher
# (or --customise_command with the same module), after the HLT sequence has been built. The mkFit seeds
# (hltMkFitSeederSeedsMkFit: MkSeederSeedConverter, not in the menu) are made for it only.
import FWCore.ParameterSet.Config as cms


def addHasher(process):
    process.hltMkFitSeederSeedsMkFit = cms.EDProducer("MkSeederSeedConverter",
        src = cms.InputTag("hltMkFitSeederSeedsMkFitDevice"),
        algorithm = cms.string("initialStep"),
        maxNSeeds = cms.uint32(500000)
    )
    process.mkFitSeedHasher = cms.EDAnalyzer("MkFitSeedWrapperHasher",
        wrappers = cms.VInputTag("hltMkFitSeederSeedsMkFit", "hltMkFitSeederSeedsMkFit:beamline"),
        floatVectors = cms.VInputTag("hltMkFitSeederSeedsClassified:scores"),
        intVectors = cms.VInputTag("hltMkFitSeederSeedsClassified:mkFitIndices"),
        trajectorySeeds = cms.VInputTag("hltMkFitSeederSeedsClassified"),
        tracks = cms.VInputTag("hltPhase2PixelTracks"),
        seeder = cms.VInputTag("hltMkFitSeederSeedsMkFitDevice"),
    )
    process.mkFitSeedHasherPath = cms.EndPath(process.mkFitSeedHasher, cms.Task(process.hltMkFitSeederSeedsMkFit))
    process.schedule.append(process.mkFitSeedHasherPath)
    return process
