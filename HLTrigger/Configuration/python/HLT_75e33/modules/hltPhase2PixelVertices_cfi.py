import FWCore.ParameterSet.Config as cms

hltPhase2PixelVertices = cms.EDProducer("PixelVertexProducer",
    Finder = cms.string('DivisiveVertexFinder'),
    Method2 = cms.bool(True),
    NTrkMin = cms.int32(2),
    PVcomparer = cms.PSet(
        refToPSet_ = cms.string('pSetPvClusterComparerForIT')
    ),
    PtMin = cms.double(1.0),
    TrackCollection = cms.InputTag("hltPhase2PixelTracksCAExtension"),
    UseError = cms.bool(True),
    Verbosity = cms.int32(0),
    WtAverage = cms.bool(True),
    ZOffset = cms.double(5.0),
    ZSeparation = cms.double(0.005),
    beamSpot = cms.InputTag("hltOnlineBeamSpot")
)

from Configuration.ProcessModifiers.phase2LegacyTracking_cff import phase2LegacyTracking
phase2LegacyTracking.toModify(hltPhase2PixelVertices,
    TrackCollection = "hltPhase2PixelTracks"
)
