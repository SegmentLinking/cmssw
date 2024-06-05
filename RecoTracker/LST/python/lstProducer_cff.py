import FWCore.ParameterSet.Config as cms

from RecoTracker.LST.lstProducer_cfi import lstProducer

from Configuration.ProcessModifiers.gpu_cff import gpu
(~gpu).toModify(lstProducer.alpaka, backend = 'serial_sync')

from RecoTracker.LST.lstModulesDevESProducer_cfi import lstModulesDevESProducer
(~gpu).toModify(lstModulesDevESProducer.alpaka, backend = 'serial_sync')

# not scheduled task to get the framework to hide the producer
dummyLSTModulesDevESProducerTask = cms.Task(lstModulesDevESProducer)
