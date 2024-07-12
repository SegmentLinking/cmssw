import FWCore.ParameterSet.Config as cms
# validation & dqm modules 
from HLTrigger.Configuration.phase2TrackingValidation_cff import *

def addTrackingValidation(process):

    process.TrackMon_gentk  = TrackMon_gentk.clone()
    process.TrackSplitMonitor = TrackSplitMonitor.clone()
    process.TrackerCollisionSelectedTrackMonCommongeneralTracks = TrackerCollisionSelectedTrackMonCommongeneralTracks.clone()
    process.dqmInfoTracking = dqmInfoTracking.clone()
    process.pvMonitor = pvMonitor.clone()

    # Sim Track/Tracking particle associator to clusters
    process.tpClusterProducer = tpClusterProducer.clone()
    # Utility to associate the number of layers to TPs
    process.trackingParticleNumberOfLayersProducer = trackingParticleNumberOfLayersProducer.clone()
    # TP to Track association
    # The associator itself
    process.quickTrackAssociatorByHits = quickTrackAssociatorByHits.clone()
    # The association to pixel tracks
    process.trackingParticlePixelTrackAssociation = trackingParticlePixelTrackAssociation.clone()
    # The association to general track
    process.trackingParticleGeneralTrackAssociation = trackingParticleGeneralTrackAssociation.clone()

    #The validators
    process.trackValidatorPixelTrackingOnly = trackValidatorPixelTrackingOnly.clone()
    process.trackValidatorGeneralTrackingOnly = trackValidatorGeneralTrackingOnly.clone()
    
    #Associating the vertices to the sim PVs
    process.VertexAssociatorByPositionAndTracks = VertexAssociatorByPositionAndTracks.clone()
    # ... and for pixel vertices
    process.VertexAssociatorByPositionAndTracksPixel = VertexAssociatorByPositionAndTracksPixel.clone()

    ##The vertex validator
    #process.vertexAnalysis = vertexAnalysis.clone()

    #DQM
    process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
        dataset = cms.untracked.PSet(
            dataTier = cms.untracked.string('DQMIO'),
            filterName = cms.untracked.string('')
        ),
        fileName = cms.untracked.string('file:Phase2HLT_DQM.root'),
        outputCommands = process.DQMEventContent.outputCommands,
        splitLevel = cms.untracked.int32(0)
    )
    process.DQMoutput_step = cms.EndPath(process.DQMoutput)
    process.dqm = cms.Task(process.pvMonitor, process.TrackSplitMonitor,process.dqmInfoTracking,process.TrackerCollisionSelectedTrackMonCommongeneralTracks,process.TrackMon_gentk)
    process.dqm_step = cms.EndPath(process.dqm)

    #Validation
    process.tracksValidation = cms.Sequence(process.trackValidatorGeneralTrackingOnly + process.trackValidatorPixelTrackingOnly)# + process.vertexAnalysis)
    process.tracksValidationTruth = cms.Task(process.VertexAssociatorByPositionAndTracksPixel,process.VertexAssociatorByPositionAndTracks, process.quickTrackAssociatorByHits, process.tpClusterProducer, process.trackingParticleNumberOfLayersProducer, process.trackingParticleGeneralTrackAssociation,process.trackingParticlePixelTrackAssociation)
    process.validation = cms.Sequence(process.tracksValidation,process.tracksValidationTruth)
    process.validation_step = cms.EndPath(process.validation)

    process.schedule.extend([process.validation_step,
        process.dqm_step,
        process.DQMoutput_step])
    
    return process


def customisePhase2HLTForTrackingOnly(process):
    
    #Baseline tracking path
    process.HLTTrackingV61Path = cms.Path(process.HLTTrackingV61Sequence)
 
    #process.localTask = cms.Task(process.RawToDigiTask,process.calolocalrecoTask)
    #process.localSeq = cms.Sequence(process.localTask) #For the moment no MTD,process.mtdRecoTask)
    #process.localPath = cms.Path(process.localSeq)

    #process.vertexRecoTask = cms.Task(process.ak4CaloJetsForTrk,
    #                                  process.initialStepPVTask,
    #                                  process.offlinePrimaryVertices,
    #                                  process.trackRefsForJetsBeforeSorting,
    #                                  process.trackWithVertexRefSelectorBeforeSorting,
    #                                  process.unsortedOfflinePrimaryVertices,
    #                                  process.goodOfflinePrimaryVertices)
    #
    #process.vertexRecoSeq = cms.Sequence(process.vertexRecoTask) ## No MTD : ,process.vertex4DrecoTask)
    #process.vertexRecoPath = cms.Path(process.vertexRecoSeq)    
    #
    ###Local Reco
    #process.localPath = cms.Path(process.RawToDigiTask,process.localrecoTask)

    process.output = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string('Phase2_HLT_LST_noValidation.root'),
        outputCommands = cms.untracked.vstring('keep *_generalTracks_*_*')
    )
    process.output_step = cms.EndPath(process.output)

    process.schedule = cms.Schedule(*[
        #process.localPath,
        process.HLTTrackingV61Path,
        #process.vertexRecoPath,
        #process.output_step,
        ])
        
    return process


def customisePhase2HLTForPatatrack(process):

    from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

    if not hasattr(process, "CUDAService"):
        from HeterogeneousCore.CUDAServices.CUDAService_cfi import CUDAService
        process.add_(CUDAService)

    from RecoLocalTracker.SiPixelRecHits.pixelCPEFastESProducerPhase2_cfi import pixelCPEFastESProducerPhase2
    process.PixelCPEFastESProducerPhase2 = pixelCPEFastESProducerPhase2.clone()
    ### SiPixelClusters on GPU

    process.siPixelClustersLegacy = process.siPixelClusters.clone()

    from RecoLocalTracker.SiPixelClusterizer.siPixelPhase2DigiToClusterCUDA_cfi import siPixelPhase2DigiToClusterCUDA as _siPixelPhase2DigiToClusterCUDA
    process.siPixelClustersCUDA = _siPixelPhase2DigiToClusterCUDA.clone()
    
    from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
    process.siPixelDigisPhase2SoA = _siPixelDigisSoAFromCUDA.clone(
        src = "siPixelClusters"
    )

    from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAPhase2_cfi import siPixelDigisClustersFromSoAPhase2 as _siPixelDigisClustersFromSoAPhase2

    process.siPixelClusters = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelClustersLegacy = cms.VPSet(cms.PSet(
                type = cms.string('SiPixelClusteredmNewDetSetVector')
            ))
            ),
        cuda = _siPixelDigisClustersFromSoAPhase2.clone(
            clusterThreshold_layer1 = 4000,
            clusterThreshold_otherLayers = 4000,
            src = "siPixelDigisPhase2SoA",
            produceDigis = False
            )
    )

    process.siPixelClustersTask = cms.Task(
                            process.siPixelClustersLegacy,
                            process.siPixelClustersCUDA,
                            process.siPixelDigisPhase2SoA,
                            process.siPixelClusters)
    
    ### SiPixel Hits

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDAPhase2_cfi import siPixelRecHitCUDAPhase2 as _siPixelRecHitCUDAPhase2
    process.siPixelRecHitsCUDA = _siPixelRecHitCUDAPhase2.clone(
        src = cms.InputTag('siPixelClustersCUDA'),
        beamSpot = "offlineBeamSpotToCUDA"
    )
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase2_cfi import siPixelRecHitSoAFromLegacyPhase2 as _siPixelRecHitsSoAPhase2
    process.siPixelRecHitsCPU = _siPixelRecHitsSoAPhase2.clone(
        convertToLegacy=True, 
        src = cms.InputTag('siPixelClusters'),
        CPE = cms.string('PixelCPEFastPhase2'))

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDAPhase2_cfi import siPixelRecHitSoAFromCUDAPhase2 as _siPixelRecHitSoAFromCUDAPhase2
    process.siPixelRecHitsSoA = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelRecHitsCPU = cms.VPSet(
                 cms.PSet(type = cms.string("pixelTopologyPhase2TrackingRecHitSoAHost")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
        cuda = _siPixelRecHitSoAFromCUDAPhase2.clone()

    )

    
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAPhase2_cfi import siPixelRecHitFromCUDAPhase2 as _siPixelRecHitFromCUDAPhase2

    _siPixelRecHits = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelRecHitsCPU = cms.VPSet(
                 cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
        cuda = _siPixelRecHitFromCUDAPhase2.clone(
            pixelRecHitSrc = cms.InputTag('siPixelRecHitsCUDA'),
            src = cms.InputTag('siPixelClusters'),
        )
    )

    process.siPixelRecHits = _siPixelRecHits.clone()
    process.siPixelRecHitsTask = cms.Task(
        process.siPixelRecHitsCUDA,
        process.siPixelRecHitsCPU,
        process.siPixelRecHits,
        process.siPixelRecHitsSoA
        )

    ### Pixeltracks

    #from RecoPixelVertexing.PixelTriplets.caHitNtupletCUDAPhase2_cfi import caHitNtupletCUDAPhase2 as _pixelTracksCUDAPhase2
    from RecoTracker.PixelSeeding.caHitNtupletCUDAPhase2_cfi import caHitNtupletCUDAPhase2 as _pixelTracksCUDAPhase2
    process.pixelTracksCUDA = _pixelTracksCUDAPhase2.clone(
        pixelRecHitSrc = "siPixelRecHitsCUDA",
        idealConditions = False,
        onGPU = True,
        includeJumpingForwardDoublets = True,
        minHitsPerNtuplet = 4
    )

    #from RecoPixelVertexing.PixelTrackFitting.pixelTrackSoAFromCUDAPhase2_cfi import pixelTrackSoAFromCUDAPhase2 as _pixelTracksSoAPhase2
    from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAPhase2_cfi import pixelTrackSoAFromCUDAPhase2 as _pixelTracksSoAPhase2
    process.pixelTracksSoA = SwitchProducerCUDA(
        # build pixel ntuplets and pixel tracks in SoA format on the CPU
        cpu = _pixelTracksCUDAPhase2.clone(
            pixelRecHitSrc = "siPixelRecHitsCPU",
            idealConditions = False,
            onGPU = False,
            includeJumpingForwardDoublets = True,
        	minHitsPerNtuplet = 4
        ),
        cuda = _pixelTracksSoAPhase2.clone()
    )

    #from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoAPhase2_cfi import pixelTrackProducerFromSoAPhase2 as _pixelTrackProducerFromSoAPhase2
    from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAPhase2_cfi import pixelTrackProducerFromSoAPhase2 as _pixelTrackProducerFromSoAPhase2
    process.pixelTracks = _pixelTrackProducerFromSoAPhase2.clone(
        pixelRecHitLegacySrc = "siPixelRecHits"
    )

    process.pixelTracksTask = cms.Task(
        process.pixelTracksCUDA,
        process.pixelTracksSoA,
        process.pixelTracks
    )

    process.HLTTrackingV61Task = cms.Task(process.MeasurementTrackerEvent, 
                                          process.generalTracks, 
                                          process.highPtTripletStepClusters, 
                                          process.highPtTripletStepHitDoublets, 
                                          process.highPtTripletStepHitTriplets, 
                                          process.highPtTripletStepSeedLayers, 
                                          process.highPtTripletStepSeeds, 
                                          process.highPtTripletStepTrackCandidates, 
                                          process.highPtTripletStepTrackCutClassifier, 
                                          process.highPtTripletStepTrackSelectionHighPurity, 
                                          process.highPtTripletStepTrackingRegions, 
                                          process.highPtTripletStepTracks, 
                                          process.initialStepSeeds, 
                                          process.initialStepTrackCandidates, 
                                          process.initialStepTrackCutClassifier, 
                                          process.initialStepTrackSelectionHighPurity, 
                                          process.initialStepTracks, 
                                          process.pixelVertices, ## for the moment leaving it as it was
                                          )

    process.trackerClusterCheckTask = cms.Task(process.trackerClusterCheck,
                                               process.siPhase2Clusters, 
                                               process.siPixelClusterShapeCache)
    process.HLTTrackingV61Sequence = cms.Sequence(process.trackerClusterCheckTask,
                                                  process.siPixelClustersTask,
                                                  process.siPixelRecHitsTask,
                                                  process.pixelTracksTask,
                                                  process.HLTTrackingV61Task)
    process.HLTTrackingV61LSTPath = cms.Path(process.HLTTrackingV61Sequence)

    process.output = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string('Phase2_HLT_LST_noValidation.root'),
        outputCommands = cms.untracked.vstring('keep *_generalTracks_*_*')
    )
    process.output_step = cms.EndPath(process.output)

    process.schedule = cms.Schedule(*[
        process.HLTTrackingV61Path,
        process.output_step,
        ])
    
    return process


def customisePhase2HLTForPatatrackLSTCKFOnLegacyTriplets(process):

    from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

    if not hasattr(process, "CUDAService"):
        from HeterogeneousCore.CUDAServices.CUDAService_cfi import CUDAService
        process.add_(CUDAService)

    if not hasattr(process, "AlpakaService"):
        from HeterogeneousCore.AlpakaServices.AlpakaServiceCudaAsync_cfi import AlpakaServiceCudaAsync
        process.add_(AlpakaServiceCudaAsync)
        from HeterogeneousCore.AlpakaServices.AlpakaServiceSerialSync_cfi import AlpakaServiceSerialSync
        process.add_(AlpakaServiceSerialSync)

    from RecoLocalTracker.SiPixelRecHits.pixelCPEFastESProducerPhase2_cfi import pixelCPEFastESProducerPhase2
    process.PixelCPEFastESProducerPhase2 = pixelCPEFastESProducerPhase2.clone()
    ### SiPixelClusters on GPU

    process.siPixelClustersLegacy = process.siPixelClusters.clone()

    from RecoLocalTracker.SiPixelClusterizer.siPixelPhase2DigiToClusterCUDA_cfi import siPixelPhase2DigiToClusterCUDA as _siPixelPhase2DigiToClusterCUDA
    process.siPixelClustersCUDA = _siPixelPhase2DigiToClusterCUDA.clone()
    
    from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
    process.siPixelDigisPhase2SoA = _siPixelDigisSoAFromCUDA.clone(
        src = "siPixelClusters"
    )

    from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAPhase2_cfi import siPixelDigisClustersFromSoAPhase2 as _siPixelDigisClustersFromSoAPhase2

    process.siPixelClusters = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelClustersLegacy = cms.VPSet(cms.PSet(
                type = cms.string('SiPixelClusteredmNewDetSetVector')
            ))
            ),
        cuda = _siPixelDigisClustersFromSoAPhase2.clone(
            clusterThreshold_layer1 = 4000,
            clusterThreshold_otherLayers = 4000,
            src = "siPixelDigisPhase2SoA",
            produceDigis = False
            )
    )

    process.siPixelClustersTask = cms.Task(
                            process.siPixelClustersLegacy,
                            process.siPixelClustersCUDA,
                            process.siPixelDigisPhase2SoA,
                            process.siPixelClusters)
    
    ### SiPixel Hits

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDAPhase2_cfi import siPixelRecHitCUDAPhase2 as _siPixelRecHitCUDAPhase2
    process.siPixelRecHitsCUDA = _siPixelRecHitCUDAPhase2.clone(
        src = cms.InputTag('siPixelClustersCUDA'),
        beamSpot = "offlineBeamSpotToCUDA"
    )
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase2_cfi import siPixelRecHitSoAFromLegacyPhase2 as _siPixelRecHitsSoAPhase2
    process.siPixelRecHitsCPU = _siPixelRecHitsSoAPhase2.clone(
        convertToLegacy=True, 
        src = cms.InputTag('siPixelClusters'),
        CPE = cms.string('PixelCPEFastPhase2'))

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDAPhase2_cfi import siPixelRecHitSoAFromCUDAPhase2 as _siPixelRecHitSoAFromCUDAPhase2
    process.siPixelRecHitsSoA = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelRecHitsCPU = cms.VPSet(
                 cms.PSet(type = cms.string("pixelTopologyPhase2TrackingRecHitSoAHost")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
        cuda = _siPixelRecHitSoAFromCUDAPhase2.clone()

    )

    
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAPhase2_cfi import siPixelRecHitFromCUDAPhase2 as _siPixelRecHitFromCUDAPhase2

    _siPixelRecHits = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelRecHitsCPU = cms.VPSet(
                 cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
        cuda = _siPixelRecHitFromCUDAPhase2.clone(
            pixelRecHitSrc = cms.InputTag('siPixelRecHitsCUDA'),
            src = cms.InputTag('siPixelClusters'),
        )
    )

    process.siPixelRecHits = _siPixelRecHits.clone()
    process.siPixelRecHitsTask = cms.Task(
        process.siPixelRecHitsCUDA,
        process.siPixelRecHitsCPU,
        process.siPixelRecHits,
        process.siPixelRecHitsSoA
        )

    ### Pixeltracks

    from RecoTracker.PixelSeeding.caHitNtupletCUDAPhase2_cfi import caHitNtupletCUDAPhase2 as _pixelTracksCUDAPhase2
    process.pixelTracksCUDA = _pixelTracksCUDAPhase2.clone(
        pixelRecHitSrc = "siPixelRecHitsCUDA",
        idealConditions = False,
        onGPU = True,
        includeJumpingForwardDoublets = True,
        minHitsPerNtuplet = 4
    )

    from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAPhase2_cfi import pixelTrackSoAFromCUDAPhase2 as _pixelTracksSoAPhase2
    process.pixelTracksSoA = SwitchProducerCUDA(
        # build pixel ntuplets and pixel tracks in SoA format on the CPU
        cpu = _pixelTracksCUDAPhase2.clone(
            pixelRecHitSrc = "siPixelRecHitsCPU",
            idealConditions = False,
            onGPU = False,
            includeJumpingForwardDoublets = True,
        	minHitsPerNtuplet = 4
        ),
        cuda = _pixelTracksSoAPhase2.clone()
    )

    from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAPhase2_cfi import pixelTrackProducerFromSoAPhase2 as _pixelTrackProducerFromSoAPhase2
    process.pixelTracks = _pixelTrackProducerFromSoAPhase2.clone(
        pixelRecHitLegacySrc = "siPixelRecHits"
    )

    process.pixelTracksTask = cms.Task(
        process.pixelTracksCUDA,
        process.pixelTracksSoA,
        process.pixelTracks
    )


    process.hltESPTTRHBuilderWithoutRefit = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
        ComponentName = cms.string('WithoutRefit'),
        ComputeCoarseLocalPositionFromDisk = cms.bool(False),
        Matcher = cms.string('Fake'),
        Phase2StripCPE = cms.string(''),
        PixelCPE = cms.string('Fake'),
        StripCPE = cms.string('Fake')
    )

    process.initialStepSeeds = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
        InputCollection = cms.InputTag("pixelTracks"),
        InputVertexCollection = cms.InputTag(""),
        SeedCreatorPSet = cms.PSet(
            refToPSet_ = cms.string('seedFromProtoTracks')
        ),
        TTRHBuilder = cms.string('WithTrackAngle'),
        originHalfLength = cms.double(0.3),
        originRadius = cms.double(0.1),
        useEventsWithNoVertex = cms.bool(True),
        usePV = cms.bool(False),
        includeFourthHit = cms.bool(True),
        useProtoTrackKinematics = cms.bool(False)
    )

    from HLTrigger.Configuration.HLT_75e33.modules.highPtTripletStepClusters_cfi import highPtTripletStepClusters as _highPtTripletStepClusters
    process.highPtTripletStepClusters = _highPtTripletStepClusters.clone(
        #TrackQuality = cms.string(''),
        trajectories = cms.InputTag("pixelTracks")
    )

    from RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi import siPhase2RecHits as _siPhase2RecHits
    process.siPhase2RecHits = _siPhase2RecHits.clone()
    from RecoTracker.LST.lstSeedTracks_cfi import lstInitialStepSeedTracks as _lstInitialStepSeedTracks
    process.lstInitialStepSeedTracks = _lstInitialStepSeedTracks.clone()
    from RecoTracker.LST.lstSeedTracks_cfi import lstHighPtTripletStepSeedTracks as _lstHighPtTripletStepSeedTracks
    process.lstHighPtTripletStepSeedTracks = _lstHighPtTripletStepSeedTracks.clone()
    from RecoTracker.LST.lstPixelSeedInputProducer_cfi import lstPixelSeedInputProducer as _lstPixelSeedInputProducer
    process.lstPixelSeedInputProducer = _lstPixelSeedInputProducer.clone()
    from RecoTracker.LST.lstPhase2OTHitsInputProducer_cfi import lstPhase2OTHitsInputProducer as _lstPhase2OTHitsInputProducer
    process.lstPhase2OTHitsInputProducer = _lstPhase2OTHitsInputProducer.clone()
    #from RecoTracker.LST.alpaka_cuda_asyncLSTProducer_cfi import alpaka_cuda_asyncLSTProducer as _lstProducer
    from RecoTracker.LST.alpaka_serial_syncLSTProducer_cfi import alpaka_serial_syncLSTProducer as _lstProducer
    process.lstProducer = _lstProducer.clone()

    from RecoTracker.LST.lstOutputConverter_cfi import lstOutputConverter as _lstOutputConverter
    process.initialStepTrackCandidates = _lstOutputConverter.clone()

    from HLTrigger.Configuration.HLT_75e33.modules.highPtTripletStepTracks_cfi import highPtTripletStepTracks as _highPtTripletStepTracks
    from HLTrigger.Configuration.HLT_75e33.modules.highPtTripletStepTrackCutClassifier_cfi import highPtTripletStepTrackCutClassifier as _highPtTripletStepTrackCutClassifier
    from HLTrigger.Configuration.HLT_75e33.modules.highPtTripletStepTrackSelectionHighPurity_cfi import highPtTripletStepTrackSelectionHighPurity as _highPtTripletStepTrackSelectionHighPurity

    process.initialStepTrackspTTCLST = _highPtTripletStepTracks.clone( src = cms.InputTag("initialStepTrackCandidates:pTTCsLST") )
    process.initialStepTrackCutClassifierpTTCLST = _highPtTripletStepTrackCutClassifier.clone( src = cms.InputTag("initialStepTrackspTTCLST") )
    process.initialStepTrackSelectionHighPuritypTTCLST = _highPtTripletStepTrackSelectionHighPurity.clone(
        originalMVAVals = cms.InputTag("initialStepTrackCutClassifierpTTCLST","MVAValues"),
        originalQualVals = cms.InputTag("initialStepTrackCutClassifierpTTCLST","QualityMasks"),
        originalSource = cms.InputTag("initialStepTrackspTTCLST")
    )

    process.initialStepTrackspLSTCLST = _highPtTripletStepTracks.clone( src = cms.InputTag("initialStepTrackCandidates:pLSTCsLST") )
    process.initialStepTrackCutClassifierpLSTCLST = _highPtTripletStepTrackCutClassifier.clone( src = cms.InputTag("initialStepTrackspLSTCLST") )
    process.initialStepTrackSelectionHighPuritypLSTCLST = _highPtTripletStepTrackSelectionHighPurity.clone(
        originalMVAVals = cms.InputTag("initialStepTrackCutClassifierpLSTCLST","MVAValues"),
        originalQualVals = cms.InputTag("initialStepTrackCutClassifierpLSTCLST","QualityMasks"),
        originalSource = cms.InputTag("initialStepTrackspLSTCLST")
    )

    process.initialStepTracksT5TCLST = _highPtTripletStepTracks.clone( src = cms.InputTag("initialStepTrackCandidates:t5TCsLST") )


    from HLTrigger.Configuration.HLT_75e33.modules.generalTracks_cfi import generalTracks as _generalTracks
    process.generalTracks = _generalTracks.clone(
            TrackProducers = cms.VInputTag("initialStepTrackSelectionHighPuritypTTCLST","initialStepTrackSelectionHighPuritypLSTCLST","initialStepTracksT5TCLST", "highPtTripletStepTrackSelectionHighPurity"),
            hasSelector = cms.vint32(0,0,0,0),
            indivShareFrac = cms.vdouble(0.1,0.1,0.1,0.1),
            selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepTrackSelectionHighPuritypTTCLST"),cms.InputTag("initialStepTrackSelectionHighPuritypLSTCLST"),cms.InputTag("initialStepTracksT5TCLST"),cms.InputTag("highPtTripletStepTrackSelectionHighPurity")),
            setsToMerge = cms.VPSet(cms.PSet(
               pQual = cms.bool(True),
               tLists = cms.vint32(0,1,2,3)
            ))
    )

    process.HLTTrackingV61Task = cms.Task(process.MeasurementTrackerEvent, 
                                          process.initialStepSeeds, 
                                          process.highPtTripletStepTrackingRegions, 
                                          process.highPtTripletStepClusters, 
                                          process.highPtTripletStepSeedLayers, 
                                          process.highPtTripletStepHitDoublets, 
                                          process.highPtTripletStepHitTriplets, 
                                          process.highPtTripletStepSeeds, 
                                          process.lstInitialStepSeedTracks,
                                          process.lstHighPtTripletStepSeedTracks,
                                          process.lstPixelSeedInputProducer,
                                          process.siPhase2RecHits,
                                          process.lstPhase2OTHitsInputProducer,
                                          process.lstProducer,
                                          process.initialStepTrackCandidates, 
                                          process.initialStepTrackspTTCLST,
                                          process.initialStepTrackspLSTCLST,
                                          process.initialStepTracksT5TCLST,
                                          process.initialStepTrackCutClassifierpTTCLST,
                                          process.initialStepTrackCutClassifierpLSTCLST,
                                          process.initialStepTrackSelectionHighPuritypTTCLST,
                                          process.initialStepTrackSelectionHighPuritypLSTCLST,
                                          process.highPtTripletStepTrackCandidates, 
                                          process.highPtTripletStepTracks,
                                          process.highPtTripletStepTrackCutClassifier,
                                          process.highPtTripletStepTrackSelectionHighPurity,
                                          process.generalTracks, 
                                          process.pixelVertices ## for the moment leaving it as it was
                                          )

    process.trackerClusterCheckTask = cms.Task(process.trackerClusterCheck,
                                               process.siPhase2Clusters, 
                                               process.siPixelClusterShapeCache)
    process.HLTTrackingV61Sequence = cms.Sequence(process.trackerClusterCheckTask,
                                                  process.siPixelClustersTask,
                                                  process.siPixelRecHitsTask,
                                                  process.pixelTracksTask,
                                                  process.HLTTrackingV61Task)
    process.HLTTrackingV61LSTPath = cms.Path(process.HLTTrackingV61Sequence)

    process.output = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string('Phase2_HLT_LST_noValidation.root'),
        outputCommands = cms.untracked.vstring('keep recoTracks_*Track*_*_*')
    )
    process.output_step = cms.EndPath(process.output)

    process.schedule = cms.Schedule(*[
        process.HLTTrackingV61Path,
        process.output_step,
        ])
    
    return process

def customisePhase2HLTForPatatrackLSTCKFOnLSTSeeds(process):

    from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

    if not hasattr(process, "CUDAService"):
        from HeterogeneousCore.CUDAServices.CUDAService_cfi import CUDAService
        process.add_(CUDAService)

    if not hasattr(process, "AlpakaService"):
        from HeterogeneousCore.AlpakaServices.AlpakaServiceCudaAsync_cfi import AlpakaServiceCudaAsync
        process.add_(AlpakaServiceCudaAsync)
        from HeterogeneousCore.AlpakaServices.AlpakaServiceSerialSync_cfi import AlpakaServiceSerialSync
        process.add_(AlpakaServiceSerialSync)

    from RecoLocalTracker.SiPixelRecHits.pixelCPEFastESProducerPhase2_cfi import pixelCPEFastESProducerPhase2
    process.PixelCPEFastESProducerPhase2 = pixelCPEFastESProducerPhase2.clone()
    ### SiPixelClusters on GPU

    process.siPixelClustersLegacy = process.siPixelClusters.clone()

    from RecoLocalTracker.SiPixelClusterizer.siPixelPhase2DigiToClusterCUDA_cfi import siPixelPhase2DigiToClusterCUDA as _siPixelPhase2DigiToClusterCUDA
    process.siPixelClustersCUDA = _siPixelPhase2DigiToClusterCUDA.clone()
    
    from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
    process.siPixelDigisPhase2SoA = _siPixelDigisSoAFromCUDA.clone(
        src = "siPixelClusters"
    )

    from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAPhase2_cfi import siPixelDigisClustersFromSoAPhase2 as _siPixelDigisClustersFromSoAPhase2

    process.siPixelClusters = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelClustersLegacy = cms.VPSet(cms.PSet(
                type = cms.string('SiPixelClusteredmNewDetSetVector')
            ))
            ),
        cuda = _siPixelDigisClustersFromSoAPhase2.clone(
            clusterThreshold_layer1 = 4000,
            clusterThreshold_otherLayers = 4000,
            src = "siPixelDigisPhase2SoA",
            produceDigis = False
            )
    )

    process.siPixelClustersTask = cms.Task(
                            process.siPixelClustersLegacy,
                            process.siPixelClustersCUDA,
                            process.siPixelDigisPhase2SoA,
                            process.siPixelClusters)
    
    ### SiPixel Hits

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDAPhase2_cfi import siPixelRecHitCUDAPhase2 as _siPixelRecHitCUDAPhase2
    process.siPixelRecHitsCUDA = _siPixelRecHitCUDAPhase2.clone(
        src = cms.InputTag('siPixelClustersCUDA'),
        beamSpot = "offlineBeamSpotToCUDA"
    )
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase2_cfi import siPixelRecHitSoAFromLegacyPhase2 as _siPixelRecHitsSoAPhase2
    process.siPixelRecHitsCPU = _siPixelRecHitsSoAPhase2.clone(
        convertToLegacy=True, 
        src = cms.InputTag('siPixelClusters'),
        CPE = cms.string('PixelCPEFastPhase2'))

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDAPhase2_cfi import siPixelRecHitSoAFromCUDAPhase2 as _siPixelRecHitSoAFromCUDAPhase2
    process.siPixelRecHitsSoA = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelRecHitsCPU = cms.VPSet(
                 cms.PSet(type = cms.string("pixelTopologyPhase2TrackingRecHitSoAHost")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
        cuda = _siPixelRecHitSoAFromCUDAPhase2.clone()

    )

    
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAPhase2_cfi import siPixelRecHitFromCUDAPhase2 as _siPixelRecHitFromCUDAPhase2

    _siPixelRecHits = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelRecHitsCPU = cms.VPSet(
                 cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
        cuda = _siPixelRecHitFromCUDAPhase2.clone(
            pixelRecHitSrc = cms.InputTag('siPixelRecHitsCUDA'),
            src = cms.InputTag('siPixelClusters'),
        )
    )

    process.siPixelRecHits = _siPixelRecHits.clone()
    process.siPixelRecHitsTask = cms.Task(
        process.siPixelRecHitsCUDA,
        process.siPixelRecHitsCPU,
        process.siPixelRecHits,
        process.siPixelRecHitsSoA
        )

    ### Pixeltracks

    from RecoTracker.PixelSeeding.caHitNtupletCUDAPhase2_cfi import caHitNtupletCUDAPhase2 as _pixelTracksCUDAPhase2
    process.pixelTracksCUDA = _pixelTracksCUDAPhase2.clone(
        pixelRecHitSrc = "siPixelRecHitsCUDA",
        idealConditions = False,
        onGPU = True,
        includeJumpingForwardDoublets = True,
        minHitsPerNtuplet = 4
    )

    from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAPhase2_cfi import pixelTrackSoAFromCUDAPhase2 as _pixelTracksSoAPhase2
    process.pixelTracksSoA = SwitchProducerCUDA(
        # build pixel ntuplets and pixel tracks in SoA format on the CPU
        cpu = _pixelTracksCUDAPhase2.clone(
            pixelRecHitSrc = "siPixelRecHitsCPU",
            idealConditions = False,
            onGPU = False,
            includeJumpingForwardDoublets = True,
        	minHitsPerNtuplet = 4
        ),
        cuda = _pixelTracksSoAPhase2.clone()
    )

    from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAPhase2_cfi import pixelTrackProducerFromSoAPhase2 as _pixelTrackProducerFromSoAPhase2
    process.pixelTracks = _pixelTrackProducerFromSoAPhase2.clone(
        pixelRecHitLegacySrc = "siPixelRecHits"
    )

    process.pixelTracksTask = cms.Task(
        process.pixelTracksCUDA,
        process.pixelTracksSoA,
        process.pixelTracks
    )


    process.hltESPTTRHBuilderWithoutRefit = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
        ComponentName = cms.string('WithoutRefit'),
        ComputeCoarseLocalPositionFromDisk = cms.bool(False),
        Matcher = cms.string('Fake'),
        Phase2StripCPE = cms.string(''),
        PixelCPE = cms.string('Fake'),
        StripCPE = cms.string('Fake')
    )

    process.initialStepSeeds = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
        InputCollection = cms.InputTag("pixelTracks"),
        InputVertexCollection = cms.InputTag(""),
        SeedCreatorPSet = cms.PSet(
            refToPSet_ = cms.string('seedFromProtoTracks')
        ),
        TTRHBuilder = cms.string('WithTrackAngle'),
        originHalfLength = cms.double(0.3),
        originRadius = cms.double(0.1),
        useEventsWithNoVertex = cms.bool(True),
        usePV = cms.bool(False),
        includeFourthHit = cms.bool(True),
        useProtoTrackKinematics = cms.bool(False)
    )

    from HLTrigger.Configuration.HLT_75e33.modules.highPtTripletStepClusters_cfi import highPtTripletStepClusters as _highPtTripletStepClusters
    process.highPtTripletStepClusters = _highPtTripletStepClusters.clone(
        #TrackQuality = cms.string(''),
        trajectories = cms.InputTag("pixelTracks")
    )

    from RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi import siPhase2RecHits as _siPhase2RecHits
    process.siPhase2RecHits = _siPhase2RecHits.clone()
    from RecoTracker.LST.lstSeedTracks_cfi import lstInitialStepSeedTracks as _lstInitialStepSeedTracks
    process.lstInitialStepSeedTracks = _lstInitialStepSeedTracks.clone()
    from RecoTracker.LST.lstSeedTracks_cfi import lstHighPtTripletStepSeedTracks as _lstHighPtTripletStepSeedTracks
    process.lstHighPtTripletStepSeedTracks = _lstHighPtTripletStepSeedTracks.clone()
    from RecoTracker.LST.lstPixelSeedInputProducer_cfi import lstPixelSeedInputProducer as _lstPixelSeedInputProducer
    process.lstPixelSeedInputProducer = _lstPixelSeedInputProducer.clone()
    from RecoTracker.LST.lstPhase2OTHitsInputProducer_cfi import lstPhase2OTHitsInputProducer as _lstPhase2OTHitsInputProducer
    process.lstPhase2OTHitsInputProducer = _lstPhase2OTHitsInputProducer.clone()
    #from RecoTracker.LST.alpaka_cuda_asyncLSTProducer_cfi import alpaka_cuda_asyncLSTProducer as _lstProducer
    from RecoTracker.LST.alpaka_serial_syncLSTProducer_cfi import alpaka_serial_syncLSTProducer as _lstProducer
    process.lstProducer = _lstProducer.clone()

    from RecoTracker.LST.lstOutputConverter_cfi import lstOutputConverter as _lstOutputConverter
    process.initialStepTrackCandidates = _lstOutputConverter.clone()


    from HLTrigger.Configuration.HLT_75e33.modules.initialStepTrackCandidates_cfi import initialStepTrackCandidates as _pLSTrackCandidates
    from HLTrigger.Configuration.HLT_75e33.modules.highPtTripletStepTracks_cfi import highPtTripletStepTracks as _highPtTripletStepTracks
    from HLTrigger.Configuration.HLT_75e33.modules.highPtTripletStepTrackCutClassifier_cfi import highPtTripletStepTrackCutClassifier as _highPtTripletStepTrackCutClassifier
    from HLTrigger.Configuration.HLT_75e33.modules.highPtTripletStepTrackSelectionHighPurity_cfi import highPtTripletStepTrackSelectionHighPurity as _highPtTripletStepTrackSelectionHighPurity

    process.initialStepTrackspTCLST = _highPtTripletStepTracks.clone( src = cms.InputTag("initialStepTrackCandidates:pTCsLST") )
    process.initialStepTrackCutClassifierpTCLST = _highPtTripletStepTrackCutClassifier.clone( src = cms.InputTag("initialStepTrackspTCLST") )
    process.initialStepTrackSelectionHighPuritypTCLST = _highPtTripletStepTrackSelectionHighPurity.clone(
        originalMVAVals = cms.InputTag("initialStepTrackCutClassifierpTCLST","MVAValues"),
        originalQualVals = cms.InputTag("initialStepTrackCutClassifierpTCLST","QualityMasks"),
        originalSource = cms.InputTag("initialStepTrackspTCLST")
    )

    process.initialStepTrackspTTCLST = _highPtTripletStepTracks.clone( src = cms.InputTag("initialStepTrackCandidates:pTTCsLST") )
    process.initialStepTrackCutClassifierpTTCLST = _highPtTripletStepTrackCutClassifier.clone( src = cms.InputTag("initialStepTrackspTTCLST") )
    process.initialStepTrackSelectionHighPuritypTTCLST = _highPtTripletStepTrackSelectionHighPurity.clone(
        originalMVAVals = cms.InputTag("initialStepTrackCutClassifierpTTCLST","MVAValues"),
        originalQualVals = cms.InputTag("initialStepTrackCutClassifierpTTCLST","QualityMasks"),
        originalSource = cms.InputTag("initialStepTrackspTTCLST")
    )

    process.initialStepTrackCandidatespLSTCLST = _pLSTrackCandidates.clone( src = cms.InputTag("initialStepTrackCandidates:pLSTSsLST") )
    process.initialStepTrackspLSTCLST = _highPtTripletStepTracks.clone( src = cms.InputTag("initialStepTrackCandidatespLSTCLST") )
    process.initialStepTrackCutClassifierpLSTCLST = _highPtTripletStepTrackCutClassifier.clone( src = cms.InputTag("initialStepTrackspLSTCLST") )
    process.initialStepTrackSelectionHighPuritypLSTCLST = _highPtTripletStepTrackSelectionHighPurity.clone(
        originalMVAVals = cms.InputTag("initialStepTrackCutClassifierpLSTCLST","MVAValues"),
        originalQualVals = cms.InputTag("initialStepTrackCutClassifierpLSTCLST","QualityMasks"),
        originalSource = cms.InputTag("initialStepTrackspLSTCLST")
    )

    process.initialStepTracksT5TCLST = _highPtTripletStepTracks.clone( src = cms.InputTag("initialStepTrackCandidates:t5TCsLST") )


    from HLTrigger.Configuration.HLT_75e33.modules.generalTracks_cfi import generalTracks as _generalTracks
    process.generalTracks = _generalTracks.clone(
            TrackProducers = cms.VInputTag("initialStepTrackSelectionHighPuritypTTCLST","initialStepTrackSelectionHighPuritypLSTCLST","initialStepTracksT5TCLST"),
            hasSelector = cms.vint32(0,0,0),
            indivShareFrac = cms.vdouble(0.1,0.1,0.1),
            selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepTrackSelectionHighPuritypTTCLST"),cms.InputTag("initialStepTrackSelectionHighPuritypLSTCLST"),cms.InputTag("initialStepTracksT5TCLST")),
            setsToMerge = cms.VPSet(cms.PSet(
               pQual = cms.bool(True),
               tLists = cms.vint32(0,1,2)
            ))
    )

    process.HLTTrackingV61Task = cms.Task(process.MeasurementTrackerEvent, 
                                          process.initialStepSeeds, 
                                          process.highPtTripletStepTrackingRegions, 
                                          process.highPtTripletStepClusters, 
                                          process.highPtTripletStepSeedLayers, 
                                          process.highPtTripletStepHitDoublets, 
                                          process.highPtTripletStepHitTriplets, 
                                          process.highPtTripletStepSeeds, 
                                          process.lstInitialStepSeedTracks,
                                          process.lstHighPtTripletStepSeedTracks,
                                          process.lstPixelSeedInputProducer,
                                          process.siPhase2RecHits,
                                          process.lstPhase2OTHitsInputProducer,
                                          process.lstProducer,
                                          process.initialStepTrackCandidates, 
                                          process.initialStepTrackCandidatespLSTCLST,
                                          process.initialStepTrackspTTCLST,
                                          process.initialStepTrackspLSTCLST,
                                          process.initialStepTracksT5TCLST,
                                          process.initialStepTrackCutClassifierpTTCLST,
                                          process.initialStepTrackCutClassifierpLSTCLST,
                                          process.initialStepTrackSelectionHighPuritypTTCLST,
                                          process.initialStepTrackSelectionHighPuritypLSTCLST,
                                          process.generalTracks, 
                                          process.pixelVertices ## for the moment leaving it as it was
                                          )

    process.trackerClusterCheckTask = cms.Task(process.trackerClusterCheck,
                                               process.siPhase2Clusters, 
                                               process.siPixelClusterShapeCache)
    process.HLTTrackingV61Sequence = cms.Sequence(process.trackerClusterCheckTask,
                                                  process.siPixelClustersTask,
                                                  process.siPixelRecHitsTask,
                                                  process.pixelTracksTask,
                                                  process.HLTTrackingV61Task)
    process.HLTTrackingV61LSTPath = cms.Path(process.HLTTrackingV61Sequence)

    process.output = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string('Phase2_HLT_LST_noValidation.root'),
        outputCommands = cms.untracked.vstring('keep recoTracks_*Track*_*_*')
    )
    process.output_step = cms.EndPath(process.output)

    process.schedule = cms.Schedule(*[
        process.HLTTrackingV61Path,
        process.output_step,
        ])
    
    return process

def customizePhase2HLTMkFitInitialStepTracks(process):
    
    process.siPhase2RecHits = cms.EDProducer("Phase2TrackerRecHits",
        Phase2StripCPE = cms.ESInputTag("phase2StripCPEESProducer","Phase2StripCPE"),
        src = cms.InputTag("siPhase2Clusters")
    )

    process.mkFitSiPixelHits = cms.EDProducer("MkFitSiPixelHitConverter",
        hits = cms.InputTag("siPixelRecHits"),
        mightGet = cms.optional.untracked.vstring,
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.mkFitSiStripHits = cms.EDProducer("MkFitSiStripHitConverter",
        mightGet = cms.optional.untracked.vstring,
        minGoodStripCharge = cms.PSet(
            refToPSet_ = cms.string('SiStripClusterChargeCutLoose')
        ),
        rphiHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        stereoHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.mkFitSiPhase2Hits = cms.EDProducer("MkFitPhase2HitConverter",
        mightGet = cms.optional.untracked.vstring,
        siPhase2Hits = cms.InputTag("siPhase2RecHits"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.mkFitEventOfHits = cms.EDProducer("MkFitEventOfHitsProducer",
        beamSpot = cms.InputTag("offlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelHits = cms.InputTag("mkFitSiPixelHits"),
        stripHits = cms.InputTag("mkFitSiPhase2Hits"),
        usePixelQualityDB = cms.bool(True),
        useStripStripQualityDB = cms.bool(False)
    )

    process.mkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
        appendToDataLabel = cms.string('')
    )

    process.initialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('initialStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-initialStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0)
    )

    process.initialStepTrackCandidatesMkFitSeeds = cms.EDProducer("MkFitSeedConverter",
        maxNSeeds = cms.uint32(500000),
        mightGet = cms.optional.untracked.vstring,
        seeds = cms.InputTag("initialStepSeeds"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.initialStepTrackCandidatesMkFit = cms.EDProducer("MkFitProducer",
        backwardFitInCMSSW = cms.bool(False),
        buildingRoutine = cms.string('cloneEngine'),
        clustersToSkip = cms.InputTag(""),
        config = cms.ESInputTag("","initialStepTrackCandidatesMkFitConfig"),
        eventOfHits = cms.InputTag("mkFitEventOfHits"),
        limitConcurrency = cms.untracked.bool(False),
        mightGet = cms.optional.untracked.vstring,
        minGoodStripCharge = cms.PSet(
            refToPSet_ = cms.string('SiStripClusterChargeCutLoose')
        ),
        mkFitSilent = cms.untracked.bool(True),
        pixelHits = cms.InputTag("mkFitSiPixelHits"),
        removeDuplicates = cms.bool(True),
        seedCleaning = cms.bool(True),
        seeds = cms.InputTag("initialStepTrackCandidatesMkFitSeeds"),
        stripHits = cms.InputTag("mkFitSiPhase2Hits")
    )

    process.initialStepTrackCandidates = cms.EDProducer("MkFitOutputConverter",
        batchSize = cms.int32(16),
        candMVASel = cms.bool(False),
        candWP = cms.double(0),
        doErrorRescale = cms.bool(True),
        mightGet = cms.optional.untracked.vstring,
        mkFitEventOfHits = cms.InputTag("mkFitEventOfHits"),
        mkFitPixelHits = cms.InputTag("mkFitSiPixelHits"),
        mkFitSeeds = cms.InputTag("initialStepTrackCandidatesMkFitSeeds"),
        mkFitStripHits = cms.InputTag("mkFitSiPhase2Hits"),
        propagatorAlong = cms.ESInputTag("","PropagatorWithMaterial"),
        propagatorOpposite = cms.ESInputTag("","PropagatorWithMaterialOpposite"),
        qualityMaxInvPt = cms.double(100),
        qualityMaxPosErr = cms.double(100),
        qualityMaxR = cms.double(120),
        qualityMaxZ = cms.double(280),
        qualityMinTheta = cms.double(0.01),
        qualitySignPt = cms.bool(True),
        seeds = cms.InputTag("initialStepSeeds"),
        tfDnnLabel = cms.string('trackSelectionTf'),
        tracks = cms.InputTag("initialStepTrackCandidatesMkFit"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.initialStepSequence = cms.Sequence(process.initialStepSeeds+process.mkFitSiPixelHits+process.mkFitSiPhase2Hits+process.mkFitEventOfHits+process.initialStepTrackCandidatesMkFitSeeds+process.initialStepTrackCandidatesMkFit+process.initialStepTrackCandidates+process.initialStepTracks+process.initialStepTrackCutClassifier+process.initialStepTrackSelectionHighPurity)

    process.itLocalRecoSequence = cms.Sequence(process.siPhase2Clusters+process.siPhase2RecHits+process.siPixelClusters+process.siPixelClusterShapeCache+process.siPixelRecHits)

    return process
