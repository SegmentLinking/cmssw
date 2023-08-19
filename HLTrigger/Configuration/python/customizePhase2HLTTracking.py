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

    #The vertex validator
    process.vertexAnalysis = vertexAnalysis.clone()

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
 
    process.localTask = cms.Task(process.RawToDigiTask, process.calolocalrecoTask)
    process.localSeq = cms.Sequence(process.localTask) #For the moment no MTD,process.mtdRecoTask)
    process.localPath = cms.Path(process.localSeq)

    process.vertexRecoTask = cms.Task(process.ak4CaloJetsForTrk,
                                      process.initialStepPVTask,
                                      process.offlinePrimaryVertices,
                                      process.trackRefsForJetsBeforeSorting,
                                      process.trackWithVertexRefSelectorBeforeSorting,
                                      process.unsortedOfflinePrimaryVertices,
                                      process.goodOfflinePrimaryVertices)

    process.vertexRecoSeq = cms.Sequence(process.vertexRecoTask) ## No MTD : ,process.vertex4DrecoTask)
    process.vertexRecoPath = cms.Path(process.vertexRecoSeq)    

    ##Local Reco
    process.localPath = cms.Path(process.RawToDigiTask,process.localrecoTask)

    process.schedule = cms.Schedule(*[
        process.localPath,
        process.HLTTrackingV61Path,
        process.vertexRecoPath,
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
    
    return process


def customisePhase2HLTForTrackingOnlyLST(process):

    if not hasattr(process, "AlpakaService"):
        process.load('Configuration.StandardSequences.Accelerators_cff')
        from HeterogeneousCore.AlpakaServices.AlpakaServiceCudaAsync_cfi import AlpakaServiceCudaAsync
        process.add_(AlpakaServiceCudaAsync)
        from HeterogeneousCore.AlpakaServices.AlpakaServiceSerialSync_cfi import AlpakaServiceSerialSync
        process.add_(AlpakaServiceSerialSync)

    process.localTask = cms.Task(process.RawToDigiTask, process.calolocalrecoTask)
    process.localSeq = cms.Sequence(process.localTask) #For the moment no MTD,process.mtdRecoTask)
    process.localPath = cms.Path(process.localSeq)

    process.vertexRecoTask = cms.Task(process.ak4CaloJetsForTrk, process.initialStepPVTask, process.offlinePrimaryVertices, process.trackRefsForJetsBeforeSorting, process.trackWithVertexRefSelectorBeforeSorting, process.unsortedOfflinePrimaryVertices,process.goodOfflinePrimaryVertices)

    process.vertexRecoSeq = cms.Sequence(process.vertexRecoTask) ## No MTD : ,process.vertex4DrecoTask)
    process.vertexRecoPath = cms.Path(process.vertexRecoSeq)    

    ##Local Reco
    process.localPath = cms.Path(process.RawToDigiTask,process.localrecoTask)

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
    from RecoTracker.LST.alpaka_cuda_asyncLSTProducer_cfi import alpaka_cuda_asyncLSTProducer as _lstProducer
    process.lstProducer = _lstProducer.clone()
    from RecoTracker.LST.lstOutputConverter_cfi import lstOutputConverter as _lstOutputConverter
    process.highPtTripletStepTrackCandidates = _lstOutputConverter.clone()

    process.highPtTripletStepTrackCutClassifier = cms.EDProducer("TrackCutClassifier",
        beamspot = cms.InputTag("hltOnlineBeamSpot"),
        ignoreVertices = cms.bool(False),
        mva = cms.PSet(
            dr_par = cms.PSet(
                d0err = cms.vdouble(0.003, 0.003, 0.003),
                d0err_par = cms.vdouble(0.002, 0.002, 0.002),#d0err_par = cms.vdouble(0.002, 0.002, 0.001),
                dr_exp = cms.vint32(4, 4, 4),
                dr_par1 = cms.vdouble(0.7, 0.6, 0.7),#dr_par1 = cms.vdouble(0.7, 0.6, 0.6),
                dr_par2 = cms.vdouble(0.6, 0.5, 0.6)#dr_par2 = cms.vdouble(0.6, 0.5, 0.45)
            ),
            dz_par = cms.PSet(
                dz_exp = cms.vint32(4, 4, 4),
                dz_par1 = cms.vdouble(0.8, 0.7, 0.8),#dz_par1 = cms.vdouble(0.8, 0.7, 0.7),
                dz_par2 = cms.vdouble(0.6, 0.6, 0.6)#dz_par2 = cms.vdouble(0.6, 0.6, 0.55)
            ),
            maxChi2 = cms.vdouble(9999.0, 9999.0, 9999.0),
            maxChi2n = cms.vdouble(2.0, 1.0, 1.0),#maxChi2n = cms.vdouble(2.0, 1.0, 0.8),
            maxDr = cms.vdouble(-1.5, 0.03, 100.0),#maxDr = cms.vdouble(0.5, 0.03, 3.40282346639e+38),
            maxDz = cms.vdouble(0.5, 0.2, 100.0),#maxDz = cms.vdouble(0.5, 0.2, 3.40282346639e+38),
            maxDzWrtBS = cms.vdouble(3.40282346639e+38, 24.0, 100.0),#maxDzWrtBS = cms.vdouble(3.40282346639e+38, 24.0, 15.0),
            maxLostLayers = cms.vint32(3, 3, 3),#maxLostLayers = cms.vint32(3, 3, 2),
            min3DLayers = cms.vint32(3, 3, 3),#min3DLayers = cms.vint32(3, 3, 0),#min3DLayers = cms.vint32(3, 3, 4),
            minLayers = cms.vint32(3, 3, 3),#minLayers = cms.vint32(3, 3, 4),
            minNVtxTrk = cms.int32(3),
            minNdof = cms.vdouble(1e-05, 1e-05, 1e-05),
            minPixelHits = cms.vint32(0, 0, 0)#minPixelHits = cms.vint32(0, 0, 3)
        ),
        qualityCuts = cms.vdouble(-0.7, 0.1, 0.7),
        src = cms.InputTag("highPtTripletStepTracks"),
        vertices = cms.InputTag("pixelVertices")
    )

    from HLTrigger.Configuration.HLT_75e33.modules.generalTracks_cfi import generalTracks as _generalTracks
    process.generalTracks = _generalTracks.clone(
            TrackProducers = cms.VInputTag("highPtTripletStepTrackSelectionHighPurity"),
            #TrackProducers = cms.VInputTag("highPtTripletStepTracks"),
            hasSelector = cms.vint32(0),
            indivShareFrac = cms.vdouble(0.1),
            selectedTrackQuals = cms.VInputTag(cms.InputTag("highPtTripletStepTrackSelectionHighPurity")),
            #selectedTrackQuals = cms.VInputTag(cms.InputTag("highPtTripletStepTracks")),
            setsToMerge = cms.VPSet(cms.PSet(
               pQual = cms.bool(True),
               tLists = cms.vint32(0)
            ))
    )

    process.HLTTrackingV61LSTTask = cms.Task(
        process.HLTBeamSpotTask,
        process.MeasurementTrackerEvent,
        process.generalTracks,
        process.highPtTripletStepClusters,
        process.highPtTripletStepHitDoublets,
        process.highPtTripletStepHitTriplets,
        process.highPtTripletStepSeedLayers,
        process.highPtTripletStepSeeds,
        process.siPhase2RecHits,
        process.lstInitialStepSeedTracks,
        process.lstHighPtTripletStepSeedTracks,
        process.lstPixelSeedInputProducer,
        process.lstPhase2OTHitsInputProducer,
        process.lstProducer,
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
        process.pixelFitterByHelixProjections,
        process.pixelTrackFilterByKinematics,
        process.pixelTracks,
        process.pixelTracksHitDoublets,
        process.pixelTracksHitSeeds,
        process.pixelTracksSeedLayers,
        process.pixelTracksTrackingRegions,
        process.pixelVertices,
        process.siPhase2Clusters,
        process.siPixelClusterShapeCache,
        process.siPixelClusters,
        process.siPixelRecHits,
        process.trackerClusterCheck
    )

    process.HLTTrackingV61LSTSeq = cms.Sequence(process.HLTTrackingV61LSTTask)
    process.HLTTrackingV61LSTPath = cms.Path(process.HLTTrackingV61LSTSeq)

    process.schedule = cms.Schedule(*[
        process.localPath,
        process.HLTTrackingV61LSTPath,
        process.vertexRecoPath,
        ])
        
    return process


def customisePhase2HLTForPatatrackLST(process):

    from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

    if not hasattr(process, "CUDAService"):
        from HeterogeneousCore.CUDAServices.CUDAService_cfi import CUDAService
        process.add_(CUDAService)

    if not hasattr(process, "AlpakaService"):
        #process.load('Configuration.StandardSequences.Accelerators_cff')
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
    from RecoTracker.LST.alpaka_cuda_asyncLSTProducer_cfi import alpaka_cuda_asyncLSTProducer as _lstProducer
    process.lstProducer = _lstProducer.clone()
    from RecoTracker.LST.lstOutputConverter_cfi import lstOutputConverter as _lstOutputConverter
    process.highPtTripletStepTrackCandidates = _lstOutputConverter.clone()

    process.highPtTripletStepTrackCutClassifier = cms.EDProducer("TrackCutClassifier",
        beamspot = cms.InputTag("hltOnlineBeamSpot"),
        ignoreVertices = cms.bool(False),
        mva = cms.PSet(
            dr_par = cms.PSet(
                d0err = cms.vdouble(0.003, 0.003, 0.003),
                d0err_par = cms.vdouble(0.002, 0.002, 0.002),#d0err_par = cms.vdouble(0.002, 0.002, 0.001),
                dr_exp = cms.vint32(4, 4, 4),
                dr_par1 = cms.vdouble(0.7, 0.6, 0.7),#dr_par1 = cms.vdouble(0.7, 0.6, 0.6),
                dr_par2 = cms.vdouble(0.6, 0.5, 0.6)#dr_par2 = cms.vdouble(0.6, 0.5, 0.45)
            ),
            dz_par = cms.PSet(
                dz_exp = cms.vint32(4, 4, 4),
                dz_par1 = cms.vdouble(0.8, 0.7, 0.8),#dz_par1 = cms.vdouble(0.8, 0.7, 0.7),
                dz_par2 = cms.vdouble(0.6, 0.6, 0.6)#dz_par2 = cms.vdouble(0.6, 0.6, 0.55)
            ),
            maxChi2 = cms.vdouble(9999.0, 9999.0, 9999.0),
            maxChi2n = cms.vdouble(2.0, 1.0, 1.0),#maxChi2n = cms.vdouble(2.0, 1.0, 0.8),
            maxDr = cms.vdouble(0.5, 0.03, 100.0),#maxDr = cms.vdouble(0.5, 0.03, 3.40282346639e+38),
            maxDz = cms.vdouble(0.5, 0.2, 100.0),#maxDz = cms.vdouble(0.5, 0.2, 3.40282346639e+38),
            maxDzWrtBS = cms.vdouble(3.40282346639e+38, 24.0, 100.0),#maxDzWrtBS = cms.vdouble(3.40282346639e+38, 24.0, 15.0),
            maxLostLayers = cms.vint32(3, 3, 3),#maxLostLayers = cms.vint32(3, 3, 2),
            min3DLayers = cms.vint32(3, 3, 3),#min3DLayers = cms.vint32(3, 3, 0),#min3DLayers = cms.vint32(3, 3, 4),
            minLayers = cms.vint32(3, 3, 3),#minLayers = cms.vint32(3, 3, 4),
            minNVtxTrk = cms.int32(3),
            minNdof = cms.vdouble(1e-05, 1e-05, 1e-05),
            minPixelHits = cms.vint32(0, 0, 0)#minPixelHits = cms.vint32(0, 0, 3)
        ),
        qualityCuts = cms.vdouble(-0.7, 0.1, 0.7),
        src = cms.InputTag("highPtTripletStepTracks"),
        vertices = cms.InputTag("pixelVertices")
    )

    from HLTrigger.Configuration.HLT_75e33.modules.generalTracks_cfi import generalTracks as _generalTracks
    process.generalTracks = _generalTracks.clone(
            TrackProducers = cms.VInputTag("highPtTripletStepTrackSelectionHighPurity"),
            #TrackProducers = cms.VInputTag("highPtTripletStepTracks"),
            hasSelector = cms.vint32(0),
            indivShareFrac = cms.vdouble(0.1),
            selectedTrackQuals = cms.VInputTag(cms.InputTag("highPtTripletStepTrackSelectionHighPurity")),
            #selectedTrackQuals = cms.VInputTag(cms.InputTag("highPtTripletStepTracks")),
            setsToMerge = cms.VPSet(cms.PSet(
               pQual = cms.bool(True),
               tLists = cms.vint32(0)
            ))
    )


    process.HLTTrackingV61Task = cms.Task(process.MeasurementTrackerEvent, 
                                          process.generalTracks, 
                                          process.highPtTripletStepClusters, 
                                          process.highPtTripletStepHitDoublets, 
                                          process.highPtTripletStepHitTriplets, 
                                          process.highPtTripletStepSeedLayers, 
                                          process.highPtTripletStepSeeds, 
                                          process.siPhase2RecHits,
                                          process.lstInitialStepSeedTracks,
                                          process.lstHighPtTripletStepSeedTracks,
                                          process.lstPixelSeedInputProducer,
                                          process.lstPhase2OTHitsInputProducer,
                                          process.lstProducer,
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
    
    return process
