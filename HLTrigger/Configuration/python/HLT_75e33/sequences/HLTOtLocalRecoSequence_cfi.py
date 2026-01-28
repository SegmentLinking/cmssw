import FWCore.ParameterSet.Config as cms

from ..modules.hltMeasurementTrackerEvent_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *

HLTOtLocalRecoSequence = cms.Sequence(hltMeasurementTrackerEvent
                                      +hltSiPhase2RecHits
                                      )

from Configuration.ProcessModifiers.phase2LegacyTracking_cff import phase2LegacyTracking
phase2LegacyTracking.toReplaceWith(HLTOtLocalRecoSequence,
                                   HLTOtLocalRecoSequence.copyAndExclude([hltSiPhase2RecHits])
                                   )
