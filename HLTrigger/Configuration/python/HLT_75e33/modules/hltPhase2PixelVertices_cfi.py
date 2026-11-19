import FWCore.ParameterSet.Config as cms

hltPhase2PixelVertices = cms.EDProducer("PixelVertexProducer",
    Finder = cms.string('DivisiveVertexFinder'),
    Method2 = cms.bool(True),
    NTrkMin = cms.int32(2),
    PVcomparer = cms.PSet(
        refToPSet_ = cms.string('pSetPvClusterComparerForIT')
    ),
    PtMin = cms.double(1.0),
    TrackCollection = cms.InputTag("hltPhase2PixelTracks"),
    UseError = cms.bool(True),
    Verbosity = cms.int32(0),
    WtAverage = cms.bool(True),
    ZOffset = cms.double(5.0),
    ZSeparation = cms.double(0.005),
    beamSpot = cms.InputTag("hltOnlineBeamSpot")
)

# phase2MkSeeder: the vertices come from the seeder's seeds; rank them with the track pT capped at 10 GeV (instead of
# 30): the badly measured high-pT seeds otherwise pick the wrong signal vertex (TTbar PU200 mistag 8% -> 4%)
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
phase2MkSeeder.toModify(hltPhase2PixelVertices, PVcomparer = cms.PSet(
    track_chi2_max = cms.double(20.0),
    track_prob_min = cms.double(-1.0),
    track_pt_max = cms.double(10.0),
    track_pt_min = cms.double(1.0)
))
