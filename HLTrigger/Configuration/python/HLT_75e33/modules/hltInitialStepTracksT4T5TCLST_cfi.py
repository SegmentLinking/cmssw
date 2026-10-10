import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTracks_cfi import _hltInitialStepTracksTrackProducer as _hltInitialStepTracks
hltInitialStepTracksT4T5TCLST = _hltInitialStepTracks.clone( src = "hltInitialStepTrackCandidates:t4t5TCsLST" )
