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

_hltGeneralTracksLST = hltGeneralTracks.clone(
    TrackProducers = cms.VInputTag("hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTrackSelectionHighPuritypLSTCLST", "hltInitialStepTracksT5TCLST", "hltHighPtTripletStepTrackSelectionHighPurity"),
    hasSelector = cms.vint32(0,0,0,0),
    indivShareFrac = cms.vdouble(0.1,0.1,0.1,0.1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hltInitialStepTrackSelectionHighPuritypTTCLST"), cms.InputTag("hltInitialStepTrackSelectionHighPuritypLSTCLST"), cms.InputTag("hltInitialStepTracksT5TCLST"), cms.InputTag("hltHighPtTripletStepTrackSelectionHighPurity")),
    setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(True),
        tLists = cms.vint32(0,1,2,3)
    ))
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(hltGeneralTracks, _hltGeneralTracksLST)

_hltGeneralTracksLSTSeeding = hltGeneralTracks.clone(
            TrackProducers = cms.VInputTag("hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTracksT5TCLST", "hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST"),
            hasSelector = cms.vint32(0,0,0),
            indivShareFrac = cms.vdouble(0.1,0.1,0.1),
            selectedTrackQuals = cms.VInputTag(cms.InputTag("hltInitialStepTrackSelectionHighPuritypTTCLST"), cms.InputTag("hltInitialStepTracksT5TCLST"), cms.InputTag("hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST")),
            setsToMerge = cms.VPSet(cms.PSet(
               pQual = cms.bool(True),
               tLists = cms.vint32(0,1,2)
            ))
    )

from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
(seedingLST & trackingLST).toReplaceWith(hltGeneralTracks, _hltGeneralTracksLSTSeeding)
