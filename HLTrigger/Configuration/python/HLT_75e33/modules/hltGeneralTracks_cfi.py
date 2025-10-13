import FWCore.ParameterSet.Config as cms

hltGeneralTracks = cms.EDProducer("TrackListMerger",
    Epsilon = cms.double(-0.001),
    FoundHitBonus = cms.double(5.0),
    LostHitPenalty = cms.double(5.0),
    MaxNormalizedChisq = cms.double(1000.0),
    MinFound = cms.int32(3),
    MinPT = cms.double(0.9),
    ShareFrac = cms.double(0.19),
    TrackProducers = cms.VInputTag("hltInitialStepTrackSelectionHighPurity", "hltHighPtTripletStepTrackSelectionHighPurity"),
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(True),
    copyMVA = cms.bool(False),
    hasSelector = cms.vint32(0, 0),
    indivShareFrac = cms.vdouble(1.0, 1.0),
    makeReKeyedSeeds = cms.untracked.bool(False),
    newQuality = cms.string('confirmed'),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hltInitialStepTrackSelectionHighPurity"), cms.InputTag("hltHighPtTripletStepTrackSelectionHighPurity")),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0, 1)
    )),
    trackAlgoPriorityOrder = cms.string('trackAlgoPriorityOrder'),
    writeOnlyTrkQuals = cms.bool(False)
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST

#(~singleIterPatatrack & trackingLST & ~seedingLST).toModify(hltGeneralTracks, indivShareFrac = [0.1,0.1])
#
#(~singleIterPatatrack & trackingLST & seedingLST).toModify(hltGeneralTracks, indivShareFrac = [0.1,0.1])

(~singleIterPatatrack & trackingLST).toModify(hltGeneralTracks, indivShareFrac = [0.1,0.1])

_hltGeneralTracksSingleIterPatatrack = hltGeneralTracks.clone(
    TrackProducers = ["hltInitialStepTrackSelectionHighPurity"],
    hasSelector = [0],
    indivShareFrac = [1.0],
    selectedTrackQuals = ["hltInitialStepTrackSelectionHighPurity"],
    setsToMerge = [cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0)
    )]
)

(singleIterPatatrack & trackingLST).toModify(_hltGeneralTracksSingleIterPatatrack, indivShareFrac = [0.1])
(singleIterPatatrack).toReplaceWith(hltGeneralTracks, _hltGeneralTracksSingleIterPatatrack)

#(singleIterPatatrack & ~trackingLST & ~seedingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksSingleIterPatatrack)
#
#(singleIterPatatrack & trackingLST & ~seedingLST).toModify(_hltGeneralTracksSingleIterPatatrack, indivShareFrac = [0.1])
#(singleIterPatatrack & trackingLST & ~seedingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksSingleIterPatatrack)
#
#(singleIterPatatrack & trackingLST & seedingLST).toModify(_hltGeneralTracksSingleIterPatatrack, indivShareFrac = [0.1])
#(singleIterPatatrack & trackingLST & seedingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksSingleIterPatatrack)
