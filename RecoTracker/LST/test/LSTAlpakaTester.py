import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('RECO',Phase2C17I13M9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## Timing
#process.Timing = cms.Service("Timing")
#process.options.wantSummary = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/home/users/evourlio/TrackingNTupleProduction/CMSSW_12_6_0_pre2/src/step2_DIGI_L1TrackTrigger_L1_DIGI2RAW_HLT.root'),
    #fileNames = cms.untracked.vstring('file:/home/users/phchang/work/lst/samples/CMSSW_12_5_0/src/step2_DIGI_L1TrackTrigger_L1_DIGI2RAW_HLT_PU.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('LSTAlpakaTesterOutput.root'),
    outputCommands = cms.untracked.vstring('keep *_lst*_*_*')
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

# Tasks and Sequences for LST inputs
process.load("RecoTracker.LST.lst_cff")
process.load("Validation.RecoTrack.trackingNtuple_cff")
lstInputTask = cms.Task(process.trackingNtupleSeedSelectors,process.siPhase2RecHits)
lstInputSequence = cms.Sequence(lstInputTask)

# Main LST Producer
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('RecoTracker.LST.alpaka_cuda_asyncLSTProducer_cfi')
### Enable alpaka-based heterogeneous modules
process.AlpakaServiceCudaAsync = cms.Service('AlpakaServiceCudaAsync')
process.AlpakaServiceSerialSync = cms.Service('AlpakaServiceSerialSync')
process.lstProducer = process.alpaka_cuda_asyncLSTProducer.clone()

# Track Fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
process.lstTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    useSimpleMF = cms.bool(True),
    SimpleMagneticField = cms.string("ParabolicMf"),
    Propagator = cms.string('PropagatorWithMaterialParabolicMf'),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    src           = cms.InputTag("lstOutputConverter"),
    AlgorithmName = cms.string("lst"),
    Fitter        = cms.string("FlexibleKFFittingSmoother")
)

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction_trackingOnly*lstInputSequence*process.lstPixelSeedInputProducer*process.lstPhase2OTHitsInputProducer*process.lstProducer*process.lstOutputConverter*process.lstTracks)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.output_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.endjob_step,process.output_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)



# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
