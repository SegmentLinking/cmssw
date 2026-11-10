import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackTorchHighPuritySelector = cms.EDProducer('PixelTrackTorchHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(9_984),
    minNumberOfHits = cms.int32(0),
    avgHitsPerTrack = cms.int32(8),
    minimumTrackQuality = cms.string('tight'),
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/pixel_track_classifier_FP16.pt'),
    scoreThreshold = cms.double(0.4),
    batchSize = cms.int32(4_992)
)

# Vector hit chain (phase2CAVectorHits): 31-feature gradient-boosted forest (17 fit/covariance, 10 hit/vector hit CA,
# 4 hit-walk features). toReplaceWith keeps the module label, so downstream wiring is untouched.
_hltPhase2PixelTrackForestHighPuritySelector = cms.EDProducer('PixelTrackForestHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(9_984),     # sized for PU200 with headroom, so nothing is truncated
    minNumberOfHits = cms.int32(0),
    avgHitsPerTrack = cms.int32(8),
    minimumTrackQuality = cms.string('tight'),
    # Compact gradient-boosted forest binary, shared per device.
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/prompt_tree31_wp_20260914.bin'),
    scoreThreshold = cms.double(0.232),
    # scoreThresholdLowDxy < 0 disables the dxy-dependent threshold ramp: one flat score cut.
    scoreThresholdLowDxy = cms.double(-1.0),
    dxyRampKnee = cms.double(2.0),
    # The hit features read the pixel rechits and vector hits the CA indexed, and the raw OT rechits.
    useHitFeatures = cms.bool(True),
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    vectorHitsSrc = cms.InputTag('hltPhase2OTVectorHitsSoA'),
    otRecHitsSoASrc = cms.InputTag('hltCAOTRecHitsSoA')
)

from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(phase2CAVectorHits | ngtScouting).toReplaceWith(
        hltPhase2PixelTrackTorchHighPuritySelector,
        _hltPhase2PixelTrackForestHighPuritySelector
)

# Inner-tracker-only CA (phase2ITonlyCA): a model trained on inner-tracker-only CA tracks; short (3-layer) tracks get
# a looser threshold, and the barrel tracks (|eta| < 1) skip the score cut (the later iterations recover more of them
# than the model rejects). With its triplets the CA has more tight tracks per event than the default buffers hold;
# the model runs only on the batches that hold tracks.
from Configuration.ProcessModifiers.phase2ITonlyCA_cff import phase2ITonlyCA
phase2ITonlyCA.toModify(hltPhase2PixelTrackTorchHighPuritySelector,
    maxNumberOfTracks = 256*1024,
    maxPreselectedTracks = 49920,
    model = 'RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/pixel_track_classifier_ITonly_v1_FP16.pt',
    scoreThreshold = 0.2,
    shortTrackMaxLayers = cms.int32(3),
    scoreThresholdShortTracks = cms.double(0.05),
    passThroughMaxAbsEta = cms.double(1.0),
    dynamicBatches = cms.bool(True)
)
