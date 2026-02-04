import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone


from ..modules.hltPhase2OtRecHitsSoA_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelRecHitsExtendedSoA_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *
from ..modules.hltTrackerClusterCheck_cfi import *
from ..sequences.HLTBeamSpotSequence_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTMkFitInputSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *

HLTTrackingSequenceCommon_HeterogeneousValidation = cms.Sequence(
    HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltTrackerClusterCheck
    + HLTBeamSpotSequence
    + hltPhase2PixelTracksAndHighPtStepTrackingRegions
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTrackFilterByKinematics
    + hltPhase2OtRecHitsSoA
    + hltPhase2PixelRecHitsExtendedSoA
    + hltSiPhase2RecHits
    + HLTMkFitInputSequence
)


hltBackend = cms.EDProducer( "AlpakaBackendProducer@alpaka"
)

hltStatusOnGPUFilter = cms.EDFilter( "AlpakaBackendFilter",
    producer = cms.InputTag( 'hltBackend','backend' ),
    backends = cms.vstring( 'CudaAsync','ROCmAsync' )
)


from ..modules.hltPhase2PixelTracksSoA_cfi import *
from ..modules.hltPhase2PixelTracksCAExtension_cfi import *
from ..modules.hltPhase2PixelVertices_cfi import *
from ..modules.hltPhase2PixelTracksCutClassifier_cfi import *
from ..modules.hltPhase2PixelTracks_cfi import *

HLTPhase2PixelTracksAndVerticesSequence_HeterogeneousValidation = cms.Sequence(
    hltPhase2PixelTracksSoA
    + hltPhase2PixelTracksCAExtension
    + hltPhase2PixelVertices
    + hltPhase2PixelTracksCutClassifier
    + hltPhase2PixelTracks
)

hltPhase2PixelTracksSoASerialSync = makeSerialClone(hltPhase2PixelTracksSoA)
hltPhase2PixelTracksCAExtensionSerialSync = hltPhase2PixelTracksCAExtension.clone(
    trackSrc = "hltPhase2PixelTracksSoASerialSync"
)
hltPhase2PixelVerticesSerialSync = hltPhase2PixelVertices.clone(
    TrackCollection = "hltPhase2PixelTracksCAExtensionSerialSync"
)
hltPhase2PixelTracksCutClassifierSerialSync = hltPhase2PixelTracksCutClassifier.clone(
    src = "hltPhase2PixelTracksCAExtensionSerialSync",
    vertices = "hltPhase2PixelVerticesSerialSync"
)
hltPhase2PixelTracksSerialSync = hltPhase2PixelTracks.clone(
    originalMVAVals = cms.InputTag("hltPhase2PixelTracksCutClassifierSerialSync","MVAValues"),
    originalQualVals = cms.InputTag("hltPhase2PixelTracksCutClassifierSerialSync","QualityMasks"),
    originalSource = cms.InputTag("hltPhase2PixelTracksCAExtensionSerialSync")
)

HLTPhase2PixelTracksAndVerticesSequence_HeterogeneousValidationSerialSync = cms.Sequence(
    hltPhase2PixelTracksSoASerialSync
    + hltPhase2PixelTracksCAExtensionSerialSync
    + hltPhase2PixelVerticesSerialSync
    + hltPhase2PixelTracksCutClassifierSerialSync
    + hltPhase2PixelTracksSerialSync
)


from ..modules.hltInitialStepSeeds_cfi import *
from ..modules.hltInitialStepSeedTracksLST_cfi import *
from ..modules.hltInputLST_cfi import *
from ..modules.hltLST_cfi import *
from ..modules.hltInitialStepTrajectorySeedsLST_cfi import *
from ..modules.hltInitialStepMkFitSeeds_cfi import *
from ..modules.hltInitialStepTrackCandidatesMkFit_cfi import *
from ..modules.hltInitialStepTrackCandidates_cfi import *
from ..modules.hltInitialStepTracks_cfi import *

HLTInitialStepSequence_HeterogeneousValidation = cms.Sequence(
    hltInitialStepSeeds
    + hltInitialStepSeedTracksLST
    + hltInputLST
    + hltLST
    + hltInitialStepTrajectorySeedsLST
    + hltInitialStepMkFitSeeds
    + hltInitialStepTrackCandidatesMkFit
    + hltInitialStepTrackCandidates
    + hltInitialStepTracks
)

hltInitialStepSeedsSerialSync = hltInitialStepSeeds.clone(
    InputCollection = cms.InputTag("hltPhase2PixelTracksSerialSync")
)
hltInitialStepSeedTracksLSTSerialSync = hltInitialStepSeedTracksLST.clone(
    src = "hltInitialStepSeedsSerialSync"
)
hltInputLSTSerialSync = makeSerialClone(hltInputLST)
hltLSTSerialSync = makeSerialClone(hltLST,
    lstInput = "hltInputLSTSerialSync"
)
hltInitialStepTrajectorySeedsLSTSerialSync = hltInitialStepTrajectorySeedsLST.clone(
    lstOutput = "hltLSTSerialSync",
    lstInput = "hltInputLSTSerialSync",
    lstPixelSeeds = "hltInputLSTSerialSync"
)
hltInitialStepMkFitSeedsSerialSync = hltInitialStepMkFitSeeds.clone(
    seeds = "hltInitialStepTrajectorySeedsLSTSerialSync"
)
hltInitialStepTrackCandidatesMkFitSerialSync = hltInitialStepTrackCandidatesMkFit.clone(
    seeds = "hltInitialStepMkFitSeedsSerialSync"
)
hltInitialStepTrackCandidatesSerialSync = hltInitialStepTrackCandidates.clone(
    mkFitSeeds = "hltInitialStepMkFitSeedsSerialSync",
    seeds = "hltInitialStepTrajectorySeedsLSTSerialSync",
    tracks = "hltInitialStepTrackCandidatesMkFitSerialSync",
)
hltInitialStepTracksSerialSync = hltInitialStepTracks.clone(
    src = "hltInitialStepTrackCandidatesSerialSync",
)

HLTInitialStepSequence_HeterogeneousValidationSerialSync = cms.Sequence(
    hltInitialStepSeedsSerialSync
    +hltInitialStepSeedTracksLSTSerialSync
    +hltInputLSTSerialSync
    +hltLSTSerialSync
    +hltInitialStepTrajectorySeedsLSTSerialSync
    +hltInitialStepMkFitSeedsSerialSync
    +hltInitialStepTrackCandidatesMkFitSerialSync
    +hltInitialStepTrackCandidatesSerialSync
    +hltInitialStepTracksSerialSync
)


DQM_TRKHeterogeneousValidation = cms.Path(
    HLTBeginSequence
    + HLTTrackingSequenceCommon_HeterogeneousValidation
    + hltBackend
    + hltStatusOnGPUFilter
    + HLTPhase2PixelTracksAndVerticesSequence_HeterogeneousValidation
    + HLTInitialStepSequence_HeterogeneousValidation
    + HLTPhase2PixelTracksAndVerticesSequence_HeterogeneousValidationSerialSync
    + HLTInitialStepSequence_HeterogeneousValidationSerialSync
)
