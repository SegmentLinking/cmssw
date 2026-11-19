import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

from ..modules.hltPhase2OtRecHitsSoA_cfi import hltPhase2OtRecHitsSoA
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import hltPhase2PixelFitterByHelixProjections
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import hltPhase2PixelTrackFilterByKinematics
from ..modules.hltPhase2PixelTracks_cfi import hltPhase2PixelTracks
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import hltPhase2PixelTracksAndHighPtStepTrackingRegions
from ..modules.hltPhase2PixelTracksHitDoublets_cfi import hltPhase2PixelTracksHitDoublets
from ..modules.hltPhase2PixelTracksHitSeeds_cfi import hltPhase2PixelTracksHitSeeds
from ..modules.hltPhase2PixelTracksSeedLayers_cfi import hltPhase2PixelTracksSeedLayers
from ..modules.hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA
from ..modules.hltPhase2PixelTrackTorchHighPuritySelector_cfi import hltPhase2PixelTrackTorchHighPuritySelector
from ..sequences.HLTPhase2PixelVertexingSequence_cfi import *
from ..sequences.HLTBeamSpotSequence_cfi import HLTBeamSpotSequence

HLTPhase2PixelTracksAndVerticesSequence = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelFitterByHelixProjections # Currently needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # Currently needed by tracker muons
    +hltPhase2OtRecHitsSoA
    +hltPhase2PixelTracksSoA
    +hltPhase2PixelTrackTorchHighPuritySelector
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)

# Empty sequence as a placeholder to be filled when alpakaValidationHLT is active
HLTPhase2PixelTracksAndVerticesSequenceSerialSync = cms.Sequence()

hltPhase2PixelTracksSoASerialSync = makeSerialClone(hltPhase2PixelTracksSoA)
hltPhase2PixelTrackTorchHighPuritySelectorSerialSync = makeSerialClone(
    hltPhase2PixelTrackTorchHighPuritySelector.clone(
        pixelTrackSrc = cms.InputTag("hltPhase2PixelTracksSoASerialSync")
    )
)
hltPhase2PixelTracksSerialSync = hltPhase2PixelTracks.clone(
    trackSrc = cms.InputTag("hltPhase2PixelTrackTorchHighPuritySelectorSerialSync")
)

# Sequence for CPU vs. GPU validation, to be kept in sync with default sequence
from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
alpakaValidationHLT.toReplaceWith(HLTPhase2PixelTracksAndVerticesSequenceSerialSync,
    cms.Sequence(
        HLTBeamSpotSequence
        +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
        +hltPhase2PixelFitterByHelixProjections # needed by tracker muons
        +hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
        +hltPhase2OtRecHitsSoA
        +hltPhase2PixelTracksSoASerialSync
        +hltPhase2PixelTrackTorchHighPuritySelectorSerialSync
        +hltPhase2PixelTracksSerialSync
        +HLTPhase2PixelVertexingSequenceSerialSync
    )
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
_HLTPhase2PixelTracksAndVerticesSequenceLegacy = cms.Sequence(
    hltPhase2PixelTracksSeedLayers
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelTracksHitDoublets
    +hltPhase2PixelTracksHitSeeds
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)
hltPhase2LegacyTracking.toReplaceWith(HLTPhase2PixelTracksAndVerticesSequence, _HLTPhase2PixelTracksAndVerticesSequenceLegacy)

# Vector-hit-based tracking sequence with OT vector hits.
from ..modules.hltCAOTRecHitsSoA_cfi import hltCAOTRecHitsSoA
from ..modules.hltPhase2OTVectorHitsSoA_cfi import hltPhase2OTVectorHitsSoA
from ..modules.hltSiPixelClusters_cfi import hltSiPixelClusters
from ..modules.hltSiPixelRecHits_cfi import hltSiPixelRecHits

_HLTPhase2PixelTracksAndVerticesSequenceCAVectorHits = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
    +hltPhase2PixelFitterByHelixProjections # Currently needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # Currently needed by tracker muons
    +hltSiPixelClusters                     # legacy pixel clusters for the legacy rechits
    +hltSiPixelRecHits                      # legacy pixel rechits for the legacy converter
    +hltCAOTRecHitsSoA
    +hltPhase2OTVectorHitsSoA
    +hltPhase2PixelTracksSoA                     # vector hit CA via the modifier (label preserved)
    +hltPhase2PixelTrackTorchHighPuritySelector  # forest selector via the modifier (label preserved)
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)

from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(phase2CAVectorHits | ngtScouting).toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequence,
    _HLTPhase2PixelTracksAndVerticesSequenceCAVectorHits
)

# Sequence for CPU vs. GPU validation, to be kept in sync with the vector hit sequence.
# SerialSync clones get the same modifiers so they are vector-hit aware
_HLTPhase2PixelTracksAndVerticesSequenceCAVectorHitsSerialSync = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
    +hltPhase2PixelFitterByHelixProjections # needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
    +hltSiPixelClusters
    +hltSiPixelRecHits
    +hltCAOTRecHitsSoA
    +hltPhase2OTVectorHitsSoA
    +hltPhase2PixelTracksSoASerialSync
    +hltPhase2PixelTrackTorchHighPuritySelectorSerialSync
    +hltPhase2PixelTracksSerialSync
    +HLTPhase2PixelVertexingSequenceSerialSync
)

((phase2CAVectorHits | ngtScouting) & alpakaValidationHLT).toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequenceSerialSync,
    _HLTPhase2PixelTracksAndVerticesSequenceCAVectorHitsSerialSync
)

# phase2MkSeeder: the mkFit seeder (hltMkFitSeederSeedsMkFitDevice_cfi.py) and its seed classifier
# (hltMkFitSeederSeedsClassifiedDevice_cfi.py) replace the pixel CA; the selected seeds as TrajectorySeeds
# (hltMkFitSeederSeedsClassified, for LST) and as tracks (hltPhase2PixelTracks, from the seeder's perigees) feed the
# pixel vertices
from ..sequences.HLTMkFitInputSequence_cfi import *
from ..modules.hltMkFitSeederSeedsMkFitDevice_cfi import hltMkFitSeederSeedsMkFitDevice as _hltMkFitSeederSeedsMkFitDevice
from ..modules.hltMkFitSeederSeedsMkFitDevice_cfi import hltMkFitSeederSeedsMkFitDeviceModifier
from ..modules.hltMkFitSeederSeedsClassifiedDevice_cfi import hltMkFitSeederSeedsClassifiedDevice as _hltMkFitSeederSeedsClassifiedDevice
from ..modules.hltMkFitSeederSeedsClassifiedDevice_cfi import hltMkFitSeederSeedsClassifiedDeviceModifier
from ..modules.hltMkFitSeederSeedsClassified_cfi import hltMkFitSeederSeedsClassified as _hltMkFitSeederSeedsClassified
from ..modules.hltMkFitSeederSeedsClassified_cfi import hltMkFitSeederSeedsClassifiedModifier
_HLTPhase2PixelTracksAndVerticesSequenceMkSeeder = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelFitterByHelixProjections # Currently needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # Currently needed by tracker muons
    +HLTMkFitInputSequence
    +_hltMkFitSeederSeedsMkFitDevice
    +_hltMkFitSeederSeedsClassifiedDevice
    +_hltMkFitSeederSeedsClassified
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
phase2MkSeeder.toReplaceWith(HLTPhase2PixelTracksAndVerticesSequence, _HLTPhase2PixelTracksAndVerticesSequenceMkSeeder)

# with trackingMkFitAlpaka (all-Alpaka initial step): the selected seeds stay on the device as the LST input's pixel
# seeds (hltInitialStepPixelSeedsDevice, customizeHLTforMkFitAlpaka), the pixel tracks without TrajectorySeeds
from ..modules.hltInitialStepPixelSeedsDevice_cfi import hltInitialStepPixelSeedsDevice as _hltInitialStepPixelSeedsDevice
from ..modules.hltInitialStepPixelSeedsDevice_cfi import hltInitialStepPixelSeedsDeviceModifier
_HLTPhase2PixelTracksAndVerticesSequenceMkSeederDevice = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelFitterByHelixProjections # Currently needed by tracker muons
    +hltPhase2PixelTrackFilterByKinematics  # Currently needed by tracker muons
    +HLTMkFitInputSequence
    +_hltMkFitSeederSeedsMkFitDevice
    +_hltMkFitSeederSeedsClassifiedDevice
    +_hltInitialStepPixelSeedsDevice
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)
from Configuration.ProcessModifiers.trackingMkFitAlpaka_cff import trackingMkFitAlpaka
(phase2MkSeeder & trackingMkFitAlpaka).toReplaceWith(HLTPhase2PixelTracksAndVerticesSequence,
                                                     _HLTPhase2PixelTracksAndVerticesSequenceMkSeederDevice)
