import FWCore.ParameterSet.Config as cms

from RecoTracker.LST.lstProducer_cfi import lstProducer

from Configuration.ProcessModifiers.gpu_cff import gpu
(~gpu).toModify(lstProducer.alpaka, backend = 'serial_sync')

from RecoTracker.LST.lstModulesDevESProducer_cfi import lstModulesDevESProducer
(~gpu).toModify(lstModulesDevESProducer.alpaka, backend = 'serial_sync')
