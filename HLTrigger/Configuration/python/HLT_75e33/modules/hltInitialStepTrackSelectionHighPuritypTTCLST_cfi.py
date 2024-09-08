import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import hltInitialStepTrackSelectionHighPurity as _hltInitialStepTrackSelectionHighPurity
hltInitialStepTrackSelectionHighPuritypTTCLST = _hltInitialStepTrackSelectionHighPurity.clone(
    originalMVAVals = cms.InputTag("hltInitialStepTrackCutClassifierpTTCLST","MVAValues"),
    originalQualVals = cms.InputTag("hltInitialStepTrackCutClassifierpTTCLST","QualityMasks"),
    originalSource = cms.InputTag("hltInitialStepTrackspTTCLST")
)

