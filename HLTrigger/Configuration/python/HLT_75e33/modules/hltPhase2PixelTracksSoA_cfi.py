import FWCore.ParameterSet.Config as cms

# list of layers to exclude from the CA (empty list doesn't exclude; [28, 29, 30] excludes the OT)
layersToExclude = []

# layers
layers = [
    #     0,        1,     2,       3,      4,        5,        6,       7
    # index, isBarrel, caDCA, caTheta, startR, maxDCurv,floorDCurv, fishCut
    [     0,     True,  0.15,   0.002,   99.0,     99.0,      99.0, 0.99999],
    [     1,     True,  0.25,   0.002,   99.0,     99.0,      99.0, 0.99999],
    [     2,     True,  0.20,   0.002,   99.0,     99.0,      99.0, 0.99999],
    [     3,     True,  0.20,   0.002,   99.0, 4.90e-02,  8.54e-04, 0.99999],
    [     4,    False,  0.25,   0.003,   99.0, 6.68e-02,  9.13e-04, 0.99999],
    [     5,    False,  0.25,   0.003,   99.0, 8.73e-02,  1.67e-03, 0.99999],
    [     6,    False,  0.25,   0.003,   99.0, 1.02e-01,  3.65e-03, 0.99999],
    [     7,    False,  0.25,   0.003,   99.0, 1.06e-01,  4.16e-03, 0.99999],
    [     8,    False,  0.25,   0.003,   99.0, 5.24e-02,  4.87e-03, 0.999999],
    [     9,    False,  0.25,   0.003,   99.0, 1.18e-01,  4.04e-03, 0.999999],
    [    10,    False,  0.25,   0.003,   99.0, 1.12e-01,  4.01e-03, 0.999999],
    [    11,    False,  0.25,   0.003,   99.0, 8.79e-02,  4.15e-03, 0.999999],
    [    12,    False,  0.40,   0.003,   99.0, 8.79e-02,  4.15e-03, 0.999999],
    [    13,    False,  0.40,   0.003,   99.0, 1.05e-01,  4.33e-03, 0.999999],
    [    14,    False,  0.25,   0.003,   99.0, 1.05e-01,  4.33e-03, 0.999999],
    [    15,    False,  0.25,   0.003,   99.0, 1.05e-01,  4.33e-03, 0.999999],
    [    16,    False,  0.25,   0.003,   99.0, 6.68e-02,  9.13e-04, 0.99999],
    [    17,    False,  0.25,   0.003,   99.0, 8.73e-02,  1.67e-03, 0.99999],
    [    18,    False,  0.25,   0.003,   99.0, 1.02e-01,  3.65e-03, 0.99999],
    [    19,    False,  0.25,   0.003,   99.0, 1.06e-01,  4.16e-03, 0.99999],
    [    20,    False,  0.25,   0.003,   99.0, 5.24e-02,  4.87e-03, 0.999999],
    [    21,    False,  0.25,   0.003,   99.0, 1.18e-01,  4.04e-03, 0.999999],
    [    22,    False,  0.25,   0.003,   99.0, 1.12e-01,  4.01e-03, 0.999999],
    [    23,    False,  0.25,   0.003,   99.0, 8.79e-02,  4.15e-03, 0.999999],
    [    24,    False,  0.40,   0.003,   99.0, 8.79e-02,  4.15e-03, 0.999999],
    [    25,    False,  0.40,   0.003,   99.0, 1.05e-01,  4.33e-03, 0.999999],
    [    26,    False,  0.25,   0.003,   99.0, 1.05e-01,  4.33e-03, 0.999999],
    [    27,    False,  0.25,   0.003,   99.0, 1.05e-01,  4.33e-03, 0.999999],
    [    28,     True,  0.50,   0.005,   99.0, 8.16e-02,  8.32e-04, 0.99999],
    [    29,     True,  0.50,   0.005,   99.0, 4.82e-02,  4.27e-04, 0.99999],
    [    30,     True,  0.50,   0.005,   99.0, 3.79e-02,  2.76e-04, 0.99999],
]

# layerPairs for doublet building including pair-specific cut values
layerPairs = [
    #  0,  1,     2,     3,      4,      5,      6,       7,       8,     9,     10,     11,     12
    #  i,  o, start,  skip, phiCut,  minIn,  maxIn,  minOut,  maxOut, maxDR,  minDZ,  maxDZ, ptCuts
    [  0,  1,  True, False,    350,  -17.0,   17.0,  -10000,   10000,   5.0,  -16.0,   16.0,  0.7],
    [  0,  2,  True,  True,    600,  -14.0,   14.0,  -10000,   10000,  10.0,  -16.0,   16.0,  0.8],
    [  0,  4,  True, False,    450,    3.0,  10000,       0,    12.0,   8.5,    0.0,   25.0,  0.6],
    [  0,  5,  True,  True,    522,    7.0,  10000,       0,   10000,   5.0,    0.0,   25.0,  0.85],
  # [  0,  6, False,  True,    522,   11.0,  10000,       0,   10000,   5.0, -10000,  10000,  0.85],
    [  0, 16,  True, False,    450, -10000,   -3.0,       0,    12.0,   8.5,  -25.0,    0.0,  0.6],
    [  0, 17,  True,  True,    522, -10000,   -7.0,       0,   10000,   5.0,  -25.0,    0.0,  0.85],
  # [  0, 18, False,  True,    522, -10000,  -10.0,       0,   10000,   5.0, -10000,  10000,  0.85],
    [  1,  2,  True, False,    400,  -17.0,   17.0,  -10000,   10000,   7.0,  -13.0,   13.0,  0.85],
    [  1,  3, False,  True,    650,  -15.0,   15.0,  -10000,   10000,  10.0,  -17.0,   17.0,  0.85],
    [  1,  4,  True, False,    500,    6.0,  10000,     6.5,   10000,   8.5,    0.0,   19.0,  0.85],
    [  1,  5, False,  True,    730,    9.0,  10000,     6.5,   10000,  10.0,    0.0,   21.0,  0.85],
  # [  1,  6, False,  True,    730,   13.0,  10000,     6.5,   10000,   8.0, -10000,  10000,  0.85],
    [  1, 16,  True, False,    500, -10000,   -6.0,     6.5,   10000,   8.5,  -19.0,    0.0,  0.85],
    [  1, 17, False,  True,    730, -10000,   -9.0,     6.5,   10000,  10.0,  -21.0,    0.0,  0.85],
  # [  1, 18, False,  True,    730, -10000,  -13.0,     6.5,   10000,   8.0, -10000,  10000,  0.85],
  # [  1, 28, False, False,   1300,    7.0,  10000,    30.0,    40.0, 10000,   19.0,   32.0,   1.0],
  # [  1, 28, False, False,   1300, -10000,   -7.0,   -40.0,   -30.0, 10000,  -32.0,  -19.0,   1.0],
    [  2,  3, False, False,    350,  -18.0,   18.0,  -10000,   10000,   7.0,   -9.0,    9.0,  0.85],
    [  2,  4, False, False,    400,   11.0,  10000,    11.7,   10000,   7.0,    0.0,   13.0,  0.85],
    [  2, 16, False, False,    400, -10000,  -11.0,    11.7,   10000,   7.0,  -13.0,    0.0,  0.85],
    [  2, 28, False, False,   1200,    -10,     10,   -30.0,    30.0, 10000,  -15.0,   15.0,   2.0],
    [  2, 28, False, False,   1200,    -20,    -10,   -50.0,   -25.0, 10000,  -35.0,  -10.0,  0.85],
    [  2, 28, False, False,   1200,     10,     20,    25.0,    50.0, 10000,   10.0,   35.0,  0.85],
  # [  2, 28, False, False,   1200,    -20,     20,   -50.0,    50.0, 10000,  -35.0,   35.0,  0.85],
    [  3, 28, False, False,   1000,    -20,     20,   -45.0,    45.0, 10000,  -22.0,   22.0,  0.85],
  # [  3, 29, False,  True,   1500,    -40,     40,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  4,  5,  True, False,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [  4,  6, False,  True,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [  4, 28, False, False,   1000,   11.6,  10000,    30.0,    57.5,  16.0,    5.0,   32.5,  0.85],
  # [  4, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  5,  6,  True, False,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [  5,  7, False,  True,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [  5, 28, False, False,   1000,   11.6,  10000,    40.0,    80.0,  16.0,  -10.0,   50.0,  0.85],
  # [  5, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  6,  7,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [  6,  8, False,  True,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [  6, 28, False, False,   1000,   11.6,  10000,    55.0,    95.0,  16.0,    5.0,   50.0,  0.85],
  # [  6, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  7,  8,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [  7,  9, False,  True,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85],
    [  7, 28, False, False,   1000,   11.8,  10000,    70.0,   110.0,  16.0,   15.0,   70.0,  0.85],
  # [  7, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  8,  9,  True, False,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85],
    [  8, 10, False,  True,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85],
    [  8, 28, False, False,    850,      0,  10000,    80.0,   10000,  14.0,   25.0,   70.0,  0.85],
  # [  8, 29, False,  True,   1000,      0,  10000,       0,   10000, 10000, -10000,  10000,  0.85],
    [  9, 10,  True, False,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,  0.85],
    [  9, 11, False,  True,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,  0.85],
  # [  9, 28, False, False,   1000,      0,  10000,       0,   10000, 10000, -10000,  10000,  0.85],
    [ 10, 11,  True, False,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85],
    [ 10, 12, False,  True,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,  0.85],
    [ 11, 12, False, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,  0.85],
    [ 11, 13, False, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,  0.85],
    [ 11, 14, False, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,  0.85],
    [ 11, 15, False, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,  0.85],
    [ 12, 13, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85],
    [ 13, 14, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85],
    [ 14, 15, False, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,  0.85],
    [ 16, 17,  True, False,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [ 16, 18, False,  True,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [ 16, 28, False, False,   1000,   11.6,  10000,   -57.5,   -30.0,  16.0,  -32.5,   -5.0,  0.85],
  # [ 16, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 17, 18,  True, False,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [ 17, 19, False,  True,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [ 17, 28, False, False,   1000,   11.6,  10000,   -80.0,   -40.0,  16.0,  -50.0,  -10.0,  0.85],
  # [ 17, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 18, 19,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [ 18, 20, False,  True,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [ 18, 28, False, False,   1000,   11.6,  10000,   -95.0,   -55.0,  16.0,  -50.0,   -5.0,  0.85],
  # [ 18, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 19, 20,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [ 19, 21, False,  True,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85],
    [ 19, 28, False, False,   1000,   11.8,  10000,  -110.0,   -70.0,  16.0,  -70.0,  -15.0,  0.85],
  # [ 19, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 20, 21,  True, False,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85],
    [ 20, 22, False,  True,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85],
    [ 20, 28, False, False,   1000,      0,  10000,  -10000,   -80.0,  14.0,  -70.0,  -25.0,  0.85],
  # [ 20, 29, False,  True,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 21, 22,  True, False,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,  0.85],
    [ 21, 23, False,  True,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,  0.85],
  # [ 21, 28, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 22, 23,  True, False,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85],
    [ 22, 24, False,  True,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,  0.85],
    [ 23, 24, False, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,  0.85],
    [ 23, 25, False, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,  0.85],
    [ 23, 26, False, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,  0.85],
    [ 23, 27, False, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,  0.85],
    [ 24, 25, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85],
    [ 25, 26, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85],
    [ 26, 27, False, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,  0.85],
    [ 28, 29, False, False,   1100,  -1200,   1200,  -10000,   10000, 10000,  -50.0,   50.0,  0.85],
  # [ 28, 30, False,  True,   2000,    -40,     40,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 29, 30, False, False,   1250,  -1200,   1200,  -10000,   10000, 10000,  -40.0,   40.0,  0.85],
]

# find the layerPairs that contain a layer that is excluded
excludeLayerPair = [any([(lp[0] == l) or (lp[1] == l) for l in layersToExclude]) for lp in layerPairs]

# exclude those layerPairs
layerPairsFinal = []
for i, lp in enumerate(layerPairs):
    if not excludeLayerPair[i]:
        layerPairsFinal.append(lp)

# get startingPairs for Ntuplet building
startingPairs = []
for i, lp in enumerate(layerPairsFinal):
    if lp[2]:
        startingPairs.append(i)

hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2OT@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    trackerRecHitsSoA = cms.InputTag('hltPhase2OtRecHitsSoA'),
    ptmin = cms.double(0.9),
    hardCurvCut = cms.double(0.02), # corresponds to 800 MeV in 3.8T.
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    onlySameLayersFishbone = cms.bool(True),
    fillStatistics = cms.bool(False),
    minHitsPerNtuplet = cms.uint32(4),
    maxNumberOfDoublets = cms.string(str(6e6)), #12*512*1024
    maxNumberOfTuples = cms.string(str(2*60*1024)),
    cellZ0Cut = cms.double(13.),
    minYsizeB1 = cms.int32(15),
    minYsizeB2 = cms.int32(14),
    maxDYsize12 = cms.int32(15),
    maxDYsize = cms.int32(20),
    maxDYPred = cms.int32(24),
    avgHitsPerTrack = cms.double(8.0),
    avgCellsPerHit = cms.double(20),
    avgCellsPerCell = cms.double(0.5),
    avgTracksPerCell = cms.double(0.09),
    minHitsForSharingCut = cms.uint32(1),
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    doTripletCleaner = cms.bool(True),
    doFastDuplicateRemover = cms.bool(True),
    doEarlyDuplicateRemover = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2TripletsOrQuadruplets = cms.double(1.0),
        maxChi2Quintuplets = cms.double(3.0),
        maxChi2 = cms.double(5.0),
        minPt   = cms.double(0.9),
        maxTip  = cms.double(0.3),
        maxZip  = cms.double(12),
    ),
    geometry = cms.PSet(
        # This cut also uses the hardCurvCut parameters inside the
        # Kernel_connect "function". This is used to cut connections that have
        # either a too low p_t or that do not intersect the BS+tolerance
        # region. Internally, this cut is compared against the circle.dca0() in
        # natural units divided by circle.curvature(), where circle is the
        # circle passing through the 3 points of the triplet under
        # investigation. Therefore the cut represent the compatibility of the
        # circle in the transverse plane and the units are meant to be cm.
        caDCACuts   = cms.vdouble([l[2] for l in layers]),
        # caThetaCut is used in the areAlignedRZ function to check if two
        # sibling cell are compatible in the R-Z plane. In that same function,
        # we also use ptmin variable. The caThetaCut is assigned to the SoA of
        # the layers, and is percolated into this compatibility function via
        # the SoA itself.
        caThetaCuts = cms.vdouble([l[3] for l in layers]),
        startingPairs = cms.vuint32(startingPairs),
        startMaxInnerR = cms.vdouble([l[4] for l in layers]),
        maxDCurv    = cms.vdouble([l[5] for l in layers]),
        floorDCurv       = cms.vdouble([l[6] for l in layers]),
        fishboneCuts   = cms.vdouble([l[7] for l in layers]),
        pairGraph = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairsFinal], [])),
        skipsLayers = cms.vuint32( [int(lp[ 3]) for lp in layerPairsFinal]),
        phiCuts     = cms.vint32( [lp[ 4] for lp in layerPairsFinal]),
        minInner    = cms.vdouble([lp[ 5] for lp in layerPairsFinal]),
        maxInner    = cms.vdouble([lp[ 6] for lp in layerPairsFinal]),
        minOuter    = cms.vdouble([lp[ 7] for lp in layerPairsFinal]),
        maxOuter    = cms.vdouble([lp[ 8] for lp in layerPairsFinal]),
        maxDR       = cms.vdouble([lp[ 9] for lp in layerPairsFinal]),
        minDZ       = cms.vdouble([lp[10] for lp in layerPairsFinal]),
        maxDZ       = cms.vdouble([lp[11] for lp in layerPairsFinal]),
        ptCuts      = cms.vdouble([lp[12] for lp in layerPairsFinal]),
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from .hltPhase2PixelTracksSoAWithVectorHits_cfi import hltPhase2PixelTracksSoAWithVectorHits as _hltPhase2PixelTracksSoAWithVectorHits
# the parameters before any modifier replaces the module (the inner-tracker-only CA below starts from them)
_hltPhase2PixelTracksSoABase = hltPhase2PixelTracksSoA.clone()
(phase2CAVectorHits | ngtScouting).toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoAWithVectorHits)

# Inner-tracker-only CA (phase2ITonlyCA): the cuts above on the pixel layers alone, with the outer-tracker barrel
# layers 28-30 and every pair reaching them dropped from the graph
_nITLayers = 28
_layersIT = [l for l in layers if l[0] < _nITLayers]
_layerPairsIT = [lp for lp in layerPairs if lp[0] < _nITLayers and lp[1] < _nITLayers]

_hltPhase2PixelTracksSoAITonly = cms.EDProducer('CAHitNtupletAlpakaPhase2@alpaka',
    **{k: v for k, v in _hltPhase2PixelTracksSoABase.parameters_().items() if k not in ('trackerRecHitsSoA', 'geometry')}
)
_hltPhase2PixelTracksSoAITonly.geometry = cms.PSet(
    # caDCACuts are 1.5 times and the doublet pT cuts 0.8 times those of the default CA (pixel-only tracks have a
    # shorter lever arm), caThetaCuts 0.0025 on BPix1-3; looser maxDR / maxOuter for BPix1/2 -> TFPX1 and a lower
    # pT cut for BPix2 -> BPix3, from a study of the cuts failed by real tracks on true hits
    caDCACuts      = cms.vdouble([1.5 * l[2] for l in _layersIT]),
    caThetaCuts    = cms.vdouble([0.0025 if l[0] < 3 else l[3] for l in _layersIT]),
    startingPairs  = cms.vuint32([i for i, lp in enumerate(_layerPairsIT) if lp[2]]),
    startMaxInnerR = cms.vdouble([l[4] for l in _layersIT]),
    maxDCurv       = cms.vdouble([l[5] for l in _layersIT]),
    floorDCurv     = cms.vdouble([l[6] for l in _layersIT]),
    fishboneCuts   = cms.vdouble([l[7] for l in _layersIT]),
    pairGraph   = cms.vuint32(sum([[lp[0], lp[1]] for lp in _layerPairsIT], [])),
    skipsLayers = cms.vuint32( [int(lp[ 3]) for lp in _layerPairsIT]),
    phiCuts     = cms.vint32(  [lp[ 4] for lp in _layerPairsIT]),
    minInner    = cms.vdouble([lp[ 5] for lp in _layerPairsIT]),
    maxInner    = cms.vdouble([lp[ 6] for lp in _layerPairsIT]),
    minOuter    = cms.vdouble([lp[ 7] for lp in _layerPairsIT]),
    maxOuter    = cms.vdouble([12.6 if (lp[0], lp[1]) in ((0, 4), (0, 16)) else lp[8] for lp in _layerPairsIT]),
    maxDR       = cms.vdouble([10.0 if (lp[0], lp[1]) in ((0, 4), (0, 16), (1, 4), (1, 16)) else lp[9]
                               for lp in _layerPairsIT]),
    minDZ       = cms.vdouble([lp[10] for lp in _layerPairsIT]),
    maxDZ       = cms.vdouble([lp[11] for lp in _layerPairsIT]),
    ptCuts      = cms.vdouble([0.8 * (0.75 if (lp[0], lp[1]) == (1, 2) else lp[12]) for lp in _layerPairsIT]),
)
_hltPhase2PixelTracksSoAITonly.cellZ0Cut = 16.0
# Without the outer-tracker layers a track crossing the barrel-endcap transition (and a barrel track through a gap
# between ladders) leaves only three pixel layers: triplets are accepted up to |eta| = 2.0
_hltPhase2PixelTracksSoAITonly.minHitsPerNtuplet = 3
_hltPhase2PixelTracksSoAITonly.maxEtaForTriplets = cms.double(2.0)
# tracks per PU200 event: mean 41k, max 84k in 4000 ttbar / QCD events (the overflow sentinel reports a truncation)
_hltPhase2PixelTracksSoAITonly.maxNumberOfTuples = str(128*1024)
# pixel-only quadruplets have no outer-tracker hit to lower their chi2: the same cut for every length
_hltPhase2PixelTracksSoAITonly.trackQualityCuts.maxChi2TripletsOrQuadruplets = 5.0
_hltPhase2PixelTracksSoAITonly.trackQualityCuts.maxChi2Quintuplets = 5.0
_hltPhase2PixelTracksSoAITonly.doEarlyDuplicateRemover = False
# a triplet is demoted only if it shares a hit with a longer tight track (not by the lowest-|tip| track, often a fake)
_hltPhase2PixelTracksSoAITonly.tripletCleanerMode = cms.uint32(2)
# pixel-only triplet DNN gate (CATripletDNNWeights_pixelOnly.h)
_hltPhase2PixelTracksSoAITonly.useTripletDNN = cms.bool(True)
_hltPhase2PixelTracksSoAITonly.tripletDNNThreshold = cms.double(0.05)
# hit-expectation tests of the 3- and 4-hit tracks (CATripletHoleCheck.h), with a |tip| cut on the triplets with a
# BPix1 hit
_hltPhase2PixelTracksSoAITonly.tripletHoleCheck = cms.PSet(
    enabled = cms.bool(True),
    clusterYLo = cms.double(-1.5),
    clusterYHi = cms.double(2.2),
    maxTip = cms.vdouble(0.1, 0.1, -1., 0.1, 0.1, -1.)
)
# z vertices from the >= 4-hit tracks, restricting the less pure 3- and 4-hit track categories to them, and a rescue
# of the prompt 3-hit tracks near a vertex that fail the hit-expectation tests (CAZVertexHisto.h)
_hltPhase2PixelTracksSoAITonly.zVertexFilter = cms.PSet(
    enabled = cms.bool(True),
    ptPower = cms.double(2.),
    ptCap = cms.double(10.),
    nMax = cms.int32(3),
    dxyKnee = cms.double(0.05),
    # 3-hit tracks: 3 hit-pattern classes x 2 |tip| classes x 3 |eta| regions; then 4-hit tracks: 2 x 3
    dzTable = cms.vdouble([-1., -1., -1., 0.3, 0.3, 0.3] * 3 + [-1., -1., -1., 0.4, 0.4, -1.]),
    tipTable = cms.vdouble([0.3] * 18 + [-1.] * 6),
    rescueDz = cms.double(0.05),
    rescueMaxTip = cms.double(0.03)
)
# containers sized from the counts of each event, and the output track collection sized to the event
_hltPhase2PixelTracksSoAITonly.useExactAllocations = cms.bool(True)
_hltPhase2PixelTracksSoAITonly.compactOutput = cms.bool(True)

from Configuration.ProcessModifiers.phase2ITonlyCA_cff import phase2ITonlyCA
phase2ITonlyCA.toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoAITonly)
