# Event products + dictionaries + automatic device->host copy.
#   cmsRun integ_products_cfg.py backend=serial_sync
#   cmsRun integ_products_cfg.py backend=cuda_async [maxEvents=20 streams=4]
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('backend', 'serial_sync', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 'alpaka backend: serial_sync or cuda_async')
options.register('streams', 2, VarParsing.multiplicity.singleton, VarParsing.varType.int, 'threads and streams')
options.setDefault('maxEvents', 10)
options.parseArguments()

process = cms.Process('MKFITPRODTEST')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Accelerators_cff')
process.source = cms.Source('EmptySource')
process.maxEvents.input = options.maxEvents
process.options.numberOfThreads = options.streams
process.options.numberOfStreams = options.streams
process.options.accelerators = ['cpu'] if options.backend == 'serial_sync' else ['gpu-nvidia']

process.productsTest = cms.EDProducer('mkfitdev::MkFitAlpakaProductsTest@alpaka',
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(options.backend)))
process.productsCheck = cms.EDAnalyzer('MkFitAlpakaProductsCheck', src = cms.InputTag('productsTest'))
process.p = cms.Path(process.productsTest + process.productsCheck)
