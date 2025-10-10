import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import hltInitialStepTrackSelectionHighPurity as _hltInitialStepTrackSelectionHighPurity
hltInitialStepTrackSelectionHighPurityT5TCLST = _hltInitialStepTrackSelectionHighPurity.clone(
    originalMVAVals = "hltInitialStepTrackCutClassifierT5TCLST:MVAValues",
    originalQualVals = "hltInitialStepTrackCutClassifierT5TCLST:QualityMasks",
    originalSource = "hltInitialStepTracksT5TCLST"
)
