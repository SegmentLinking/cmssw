import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepSeeds_cfi import *
from ..modules.hltInitialStepTrackCandidates_cfi import *
from ..modules.hltInitialStepTrackCutClassifier_cfi import *
from ..modules.hltInitialStepTracks_cfi import *
from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import *

HLTInitialStepSequence = cms.Sequence(hltInitialStepSeeds+
                                      hltInitialStepTrackCandidates+
                                      hltInitialStepTracks+
                                      hltInitialStepTrackCutClassifier+
                                      hltInitialStepTrackSelectionHighPurity)

from ..modules.hltInitialStepSeedTracksLST_cfi import *
from ..sequences.HLTHighPtTripletStepSeedingSequence_cfi import *
from ..modules.hltHighPtTripletStepSeedTracksLST_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *
from ..modules.hltInputLST_cfi import *
from ..modules.hltLST_cfi import *
_HLTInitialStepSequenceLST = cms.Sequence(
     hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits # Probably need to move elsewhere in the final setup
    +hltInputLST
    +hltLST
    +hltInitialStepTrackCandidates
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)

from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST

(trackingLST & ~seedingLST).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLST)

from ..modules.hltInitialStepTrajectorySeedsLST_cfi import *
_HLTInitialStepSequenceLSTSeeding = cms.Sequence(
     hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits # Probably need to move elsewhere in the final setup
    +hltInputLST
    +hltLST
    +hltInitialStepTrajectorySeedsLST
    +hltInitialStepTrackCandidates
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)

(trackingLST & seedingLST).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLSTSeeding)

from ..modules.hltInitialStepTracksT4T5TCLST_cfi import *
_HLTInitialStepSequenceNGTScouting = cms.Sequence(
    hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits
    +hltInputLST
    +hltLST
    +hltInitialStepTrackCandidates
    +hltInitialStepTracksT4T5TCLST
)

from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
ngtScouting.toReplaceWith(HLTInitialStepSequence,_HLTInitialStepSequenceNGTScouting)

from ..sequences.HLTMkFitInputSequence_cfi import *
from ..modules.hltInitialStepMkFitSeeds_cfi import *
from ..modules.hltInitialStepTrackCandidatesMkFit_cfi import *
_HLTInitialStepSequenceMkFitTracking = cms.Sequence(
    hltInitialStepSeeds
    +hltSiPhase2RecHits
    +HLTMkFitInputSequence
    +hltInitialStepMkFitSeeds
    +hltInitialStepTrackCandidatesMkFit
    +hltInitialStepTrackCandidates
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)

from Configuration.ProcessModifiers.hltTrackingMkFitInitialStep_cff import hltTrackingMkFitInitialStep
(~seedingLST & ~trackingLST & hltTrackingMkFitInitialStep).toReplaceWith(HLTInitialStepSequence,_HLTInitialStepSequenceMkFitTracking)

_HLTInitialStepSequenceLSTSeedingMkFitTracking = cms.Sequence(
     hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits # Probably need to move elsewhere in the final setup                                                 
    +hltInputLST
    +hltLST
    +hltInitialStepTrajectorySeedsLST
    +HLTMkFitInputSequence
    +hltInitialStepMkFitSeeds
    +hltInitialStepTrackCandidatesMkFit
    +hltInitialStepTrackCandidates
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)

(trackingLST & seedingLST & hltTrackingMkFitInitialStep).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLSTSeedingMkFitTracking)

from ..modules.hltInitialStepTrackCandidatesMkFitFit_cfi import *

_HLTInitialStepSequenceMkFitFitTracking = cms.Sequence(
    hltInitialStepSeeds
    +hltSiPhase2RecHits
    +HLTMkFitInputSequence
    +hltInitialStepMkFitSeeds
    +hltInitialStepTrackCandidatesMkFit
    +hltInitialStepTrackCandidatesMkFitFit
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)

_HLTInitialStepSequenceLSTSeedingMkFitFitTracking = cms.Sequence(
     hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits # Probably need to move elsewhere in the final setup
    +hltInputLST
    +hltLST
    +hltInitialStepTrajectorySeedsLST
    +HLTMkFitInputSequence
    +hltInitialStepMkFitSeeds
    +hltInitialStepTrackCandidatesMkFit
    +hltInitialStepTrackCandidatesMkFitFit
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)

from Configuration.ProcessModifiers.trackingMkFitFit_cff import trackingMkFitFit
(~seedingLST & ~trackingLST & hltTrackingMkFitInitialStep & trackingMkFitFit).toReplaceWith(HLTInitialStepSequence,_HLTInitialStepSequenceMkFitFitTracking)
(trackingLST & seedingLST & hltTrackingMkFitInitialStep & trackingMkFitFit).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLSTSeedingMkFitFitTracking)
