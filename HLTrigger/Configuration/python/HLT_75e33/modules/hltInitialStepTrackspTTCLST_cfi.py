import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTracks_cfi import hltInitialStepTracks as _hltInitialStepTracks
hltInitialStepTrackspTTCLST = _hltInitialStepTracks.clone( src = cms.InputTag("initialStepTrackCandidates:pTTCsLST") )

