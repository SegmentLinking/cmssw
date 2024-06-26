import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerTFP.KalmanFilterFormats_cfi import TrackTriggerKalmanFilterFormats_params

TrackTriggerKalmanFilterFormats = cms.ESProducer("trackerTFP::ProducerFormatsKF", TrackTriggerKalmanFilterFormats_params)
