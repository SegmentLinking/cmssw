import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTracks_cfi import hltInitialStepTracks as _hltInitialStepTracks
hltInitialStepTracksT5TCLST = _hltInitialStepTracks.clone( src = cms.InputTag("hltInitialStepTrackCandidates:t5TCsLST") )

