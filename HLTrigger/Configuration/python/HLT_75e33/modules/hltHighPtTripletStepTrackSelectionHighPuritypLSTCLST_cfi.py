import FWCore.ParameterSet.Config as cms

from ..modules.hltHighPtTripletStepTrackSelectionHighPurity_cfi import hltHighPtTripletStepTrackSelectionHighPurity as _hltHighPtTripletStepTrackSelectionHighPurity
hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST = _hltHighPtTripletStepTrackSelectionHighPurity.clone(
    originalMVAVals = cms.InputTag("hltHighPtTripletStepTrackCutClassifierpLSTCLST","MVAValues"),
    originalQualVals = cms.InputTag("hltHighPtTripletStepTrackCutClassifierpLSTCLST","QualityMasks"),
    originalSource = cms.InputTag("hltHighPtTripletStepTrackspLSTCLST")
)

