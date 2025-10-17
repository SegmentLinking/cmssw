import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackCutClassifier_cfi import hltInitialStepTrackCutClassifier as _hltInitialStepTrackCutClassifier
hltInitialStepTrackCutClassifierT5TCLST = _hltInitialStepTrackCutClassifier.clone(
    src = "hltInitialStepTracksT5TCLST",
    mva = dict( passThroughForDisplaced = True )
)
