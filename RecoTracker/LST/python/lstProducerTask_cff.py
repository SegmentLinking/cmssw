import FWCore.ParameterSet.Config as cms

from RecoTracker.LST.lstProducer_cfi import lstProducer

from RecoTracker.LST.lstModulesDevESProducer_cfi import lstModulesDevESProducer

from RecoTracker.LSTCore.lstGeometryESProducer_cfi import lstGeometryDevESProducer

lstProducerTask = cms.Task(lstGeometryDevESProducer, lstModulesDevESProducer, lstProducer)
