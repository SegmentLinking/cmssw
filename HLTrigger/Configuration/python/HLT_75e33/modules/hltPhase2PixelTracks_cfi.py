import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    trackSrc = cms.InputTag("hltPhase2PixelTrackTorchHighPuritySelector"),
    outerTrackerRecHitSrc = cms.InputTag("hltSiPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("hltPhase2OtRecHitsSoA"),
    useOTExtension = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(False)
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
_hltPhase2PixelTracksLegacy = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('hltPixelTracksCleanerBySharedHits'),
    Filter = cms.InputTag("hltPhase2PixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("hltPhase2PixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("hltPhase2PixelTracksHitSeeds"),
    mightGet = cms.optional.untracked.vstring,
    passLabel = cms.string('hltPhase2PixelTracks')
)
hltPhase2LegacyTracking.toReplaceWith(hltPhase2PixelTracks, _hltPhase2PixelTracksLegacy)

from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from .hltPhase2PixelTracksWithVectorHits_cfi import hltPhase2PixelTracksWithVectorHits as _hltPhase2PixelTracksWithVectorHits
(phase2CAVectorHits | ngtScouting).toReplaceWith(hltPhase2PixelTracks, _hltPhase2PixelTracksWithVectorHits)

# phase2MkSeeder: the "pixel tracks" are the mkFit seeder's seeds that pass the seed classifier, as tracks, made by
# MkSeederPixelTrackProducer from the perigees of the classified seeds (hltMkFitSeederSeedsMkFitDevice, the seeds'
# beam-line states at the beam line); the pixel vertices and the MTV hltPhase2Pixel folder use them, and the pixel CA
# does not run. The tracks' seed references point to hltMkFitSeederSeedsClassified (the same seeds with the forward
# state), whose hits (the same) it copies
_hltPhase2PixelTracksMkSeeder = cms.EDProducer("MkSeederPixelTrackProducer",
    src = cms.InputTag("hltMkFitSeederSeedsClassified"),
    seeds = cms.InputTag("hltMkFitSeederSeedsMkFitDevice"),
    selection = cms.InputTag("hltMkFitSeederSeedsClassified", "mkFitIndices")
)
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
phase2MkSeeder.toReplaceWith(hltPhase2PixelTracks, _hltPhase2PixelTracksMkSeeder)
# with trackingMkFitAlpaka the seeds stay on the device (hltInitialStepPixelSeedsDevice; no TrajectorySeeds): the
# tracks of the same selection with the hits by cluster index, in the order of the device pixel seeds' rows; the pixel
# hits from the device EventOfHits' cluster index map, the OT hits made on demand from their clusters (no legacy OT
# rechits)
_hltPhase2PixelTracksMkSeederDevice = _hltPhase2PixelTracksMkSeeder.clone(
    src = "",
    scores = cms.InputTag("hltMkFitSeederSeedsClassifiedDevice"),
    mkFitPixelHits = cms.InputTag("hltMkFitEventOfHits", "pixelIndexToHit"),
    mkFitStripHits = cms.InputTag(""),
    otClustersOnDemand = cms.InputTag("hltSiPhase2Clusters"),
    Phase2StripCPE = cms.ESInputTag("hltESPPhase2StripCPE", "Phase2StripCPEHLT")
)
from Configuration.ProcessModifiers.trackingMkFitAlpaka_cff import trackingMkFitAlpaka
(phase2MkSeeder & trackingMkFitAlpaka).toReplaceWith(hltPhase2PixelTracks, _hltPhase2PixelTracksMkSeederDevice)
