import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import hltInitialStepTrackSelectionHighPurity as _hltInitialStepTrackSelectionHighPurity
hltInitialStepTrackSelectionHighPuritypLSTCLST = _hltInitialStepTrackSelectionHighPurity.clone(
    originalMVAVals = cms.InputTag("hltInitialStepTrackCutClassifierpLSTCLST","MVAValues"),
    originalQualVals = cms.InputTag("hltInitialStepTrackCutClassifierpLSTCLST","QualityMasks"),
    originalSource = cms.InputTag("hltInitialStepTrackspLSTCLST")
)

