# Phase-2 HLT: the mkFit chain of the LST initial step on the device with Alpaka (one code for GPU and CPU backends).
# Apply with --customise RecoTracker/MkFitAlpaka/customizeHLTforMkFitAlpaka.customizeHLTforMkFitAlpaka on top of the
# menu with --procModifiers trackingMkFitFit. Module labels of the menu are kept, so downstream InputTags resolve
# unchanged. What runs where:
#   hltMkFitAlpakaOTRecHits         device OT rechits (also the CA OT hit SoA); the legacy OT rechit producer leaves the
#                                   menu, its readers make the few OT hits they need on demand
#   hltMkFitEventOfHits             device hit input + EventOfHits binning (pixel rechit SoA + OT rechit SoA)
#   hltInputLSTDevice               device LST input; pLS parameters from the device KF of the pixel-track hits
#   hltInitialStepMkFitSeeds        device hand-off of the LST track candidates to the mkFit seeds
#   hltInitialStepTrackCandidatesMkFitDevice     device building (seed fit, clone engine, backward fit, cleaning)
#   hltInitialStepTrackCandidatesMkFitFitDevice  device mkFit final fit (with the pixel CPE)
#   hltInitialStepOutConvStates / hltInitialStepNavDevice  beam-line PCA states and missing-hit navigation on device
#   hltInitialStepTracks            host conversion to reco::Track
# Physics differences to the reference mkFit are listed in doc/DEVIATIONS.txt.
import FWCore.ParameterSet.Config as cms

ES_LABEL = 'hltMkFitAlpakaES'
CPE_ES_LABEL = 'hltMkFitAlpakaFitCpe'
MODULE_TABLE_LABEL = 'MkFitAlpakaHitModuleTable'
EOH = 'hltMkFitEventOfHits'
BUILD = 'hltInitialStepTrackCandidatesMkFit'
BUILD_DEVICE = 'hltInitialStepTrackCandidatesMkFitDevice'
FIT = 'hltInitialStepTrackCandidatesMkFitFit'
FIT_DEVICE = 'hltInitialStepTrackCandidatesMkFitFitDevice'
PIX_HITS = 'hltMkFitSiPixelHits'
OT_HITS = 'hltMkFitSiPhase2Hits'
OT_DEVICE = 'hltMkFitAlpakaOTRecHits'
OT_CLUSTERS = 'hltSiPhase2Clusters'
OT_RECHITS = 'hltSiPhase2RecHits'
CA_OT = 'hltPhase2OtRecHitsSoA'
PIXEL_TRACKS_CA = 'hltPhase2PixelTracksSoA'
LST_INPUT_DEVICE = 'hltInputLSTDevice'
LST_SEEDS = 'hltInitialStepTrajectorySeedsLST'
MKFIT_SEEDS = 'hltInitialStepMkFitSeeds'
OUTCONV_STATES = 'hltInitialStepOutConvStates'
NAV_DEVICE = 'hltInitialStepNavDevice'

# Final-fit outliers near the outer end (DEVIATION DEV-10): the 3 hits nearest the outer end tested with the forward
# pass and one outlier removed per round, rounds until a round removes no hit (the KF fit-smoother's rule).
NEAR_END_OUTLIERS = True

# friendly class names of the device products deleted early (per backend; names not in the job are ignored)
_EOH_TYPES = ('alpakaDevCudaRt128falsemkfitdevEventOfHitsBlocksLayoutvoidPortableDeviceCollectionedmDeviceProduct',
              'alpakaDevHipRt128falsemkfitdevEventOfHitsBlocksLayoutvoidPortableDeviceCollectionedmDeviceProduct',
              'mkfitdevEventOfHitsPooledHostCollection')
_OT_TYPES = ('alpakaDevCudaRt128falsemkfitdevOTRecHitSoALayoutvoidPortableDeviceCollectionedmDeviceProduct',
             'alpakaDevHipRt128falsemkfitdevOTRecHitSoALayoutvoidPortableDeviceCollectionedmDeviceProduct',
             '128falsemkfitdevOTRecHitSoALayoutPortableHostCollection')
_LSTIN_TYPES = ('alpakaDevCudaRt128falselstLSTInputSoALayoutvoidPortableDeviceCollectionedmDeviceProduct',
                'alpakaDevHipRt128falselstLSTInputSoALayoutvoidPortableDeviceCollectionedmDeviceProduct',
                '128falselstLSTInputSoALayoutPortableHostCollection')
_TRK_TYPES = ('alpakaDevCudaRt128falsemkfitdevTrackLayoutvoidPortableDeviceCollectionedmDeviceProduct',
              'alpakaDevHipRt128falsemkfitdevTrackLayoutvoidPortableDeviceCollectionedmDeviceProduct',
              '128falsemkfitdevTrackLayoutPortableHostCollection')

# PixelCPEGeneric settings the device CPE implements (interface/fit/CpeGeneric.h; the phase-2 values of
# PixelCPEGeneric_cfi.py). name: (supported value, C++ default when the PSet does not set it)
_CPE_GENERIC_SUPPORTED = {
    'UseErrorsFromTemplates': (True, True), 'TruncatePixelCharge': (False, True),
    'IrradiationBiasCorrection': (False, False), 'DoCosmics': (False, False), 'inflate_errors': (False, False),
    'eff_charge_cut_lowX': (0., 0.), 'eff_charge_cut_lowY': (0., 0.),
    'eff_charge_cut_highX': (1., 1.), 'eff_charge_cut_highY': (1., 1.),
    'size_cutX': (3., 3.), 'size_cutY': (3., 3.),
    'EdgeClusterErrorX': (50., 50.), 'EdgeClusterErrorY': (85., 85.),
    'useLAWidthFromDB': (True, True), 'LoadTemplatesFromDB': (True, True),
}
# the module constants (Lorentz shift / width) come from the PixelCPEFastParams product: both producers must agree
_CPE_LORENTZ_DEFAULTS = {'useLAFromDB': True, 'lAOffset': 0., 'lAWidthBPix': 0., 'lAWidthFPix': 0.,
                         'doLorentzFromAlignment': False, 'useLAWidthFromDB': True}
# parameters of the menu MkFitFitProducer the device fit honours; any other non-default value refuses the swap
_FIT_HONOURED = {'tracks', 'eventOfHits', 'mkFitPixelHits', 'config', 'pixelCPE', 'candCutSel', 'candMinPtCut',
                 'candMinNHitsCut', 'candMinPtRelaxedCut', 'candMinAbsEtaForRelaxedCut', 'mkFitSilent',
                 'limitConcurrency', 'mightGet'}
# plugins that read the hit labels only as MkFitClusterIndexToHit
_INDEX_ONLY_READERS = ('MkFitOutputConverter', 'MkFitOutputTrackConverter', 'MkFitAlpakaBuildProducer@alpaka',
                       'MkFitAlpakaFitDeviceProducer@alpaka', 'MkFitAlpakaPhase2ClusterIndexToHit')


def _modules(process):
    for d in (process.producers_(), process.filters_(), process.analyzers_()):
        for label, m in d.items():
            yield label, m


def _visitTags(pset, path, visit):
    for n in pset.parameterNames_():
        p = getattr(pset, n)
        if isinstance(p, cms.InputTag):
            visit(pset, n, None, p.getModuleLabel(), path + n)
        elif isinstance(p, cms.VInputTag):
            for i, t in enumerate(p):
                visit(pset, n, i, t.getModuleLabel() if isinstance(t, cms.InputTag) else str(t).split(':')[0], path + n)
        elif isinstance(p, cms.PSet):
            _visitTags(p, path + n + '.', visit)
        elif isinstance(p, cms.VPSet):
            for i, q in enumerate(p):
                _visitTags(q, '%s%s[%d].' % (path, n, i), visit)


def _consumers(process, label, skip=()):
    """Parameters ('module.parameter') of the modules that read module label 'label'."""
    found = []
    for l, m in _modules(process):
        if l not in skip:
            _visitTags(m, l + '.', lambda pset, n, i, tl, where: found.append(where) if tl == label else None)
    return found


def _retag(process, label, tag, skip=()):
    """Point every InputTag at module label 'label' to 'tag'."""
    def visit(pset, n, i, tl, where):
        if tl != label:
            return
        new = cms.InputTag(tag.getModuleLabel(), tag.getProductInstanceLabel())
        if i is None:
            setattr(pset, n, new)
        else:
            getattr(pset, n)[i] = new
    for l, m in _modules(process):
        if l not in skip:
            _visitTags(m, l + '.', visit)


def _inSequenceOrTask(process, label):
    for s in list(process.sequences_().values()) + list(process.paths_().values()) + list(process.endpaths_().values()):
        if label in s.moduleNames():
            return True
    return any(label in t.moduleNames() for t in process.tasks_().values())


def _removeFromDirectSequences(process, module, replacement=None):
    """Remove (or replace) a module only in the sequences that hold it directly: an edit through an outer sequence
    copies the nested sequences, and later edits of the nested ones would then miss the copies."""
    for s in process.sequences_().values():
        c = s._seq
        kids = list(c._collection) if hasattr(c, '_collection') else ([c] if c is not None else [])
        if any(k is module for k in kids):
            if replacement is not None:
                s.replace(module, replacement)
            else:
                s.remove(module)


def _canDeleteEarly(process, label, types):
    have = set(process.options.canDeleteEarly)
    process.options.canDeleteEarly.extend(
        [b for b in ('%s_%s__%s' % (t, label, process.name_()) for t in types) if b not in have])


# ---- checks: the menu modules are inside the configuration the device code reproduces ----
def _checkMkFitBuildConfig(mkf):
    want = {'buildingRoutine': 'cloneEngine', 'backwardFitInCMSSW': False, 'seedCleaning': True,
            'removeDuplicates': True, 'config': cms.ESInputTag('', 'hltInitialStepTrackCandidatesMkFitConfig')}
    bad = []
    for k, v in want.items():
        have = getattr(mkf, k).value() if hasattr(mkf, k) else None
        if have != (v.value() if hasattr(v, 'value') else v):
            bad.append('%s=%s (supported: %s)' % (k, have, v))
    if mkf.clustersToSkip.getModuleLabel() != '':
        bad.append('clustersToSkip=%s (supported: empty)' % mkf.clustersToSkip.value())
    if len(mkf.minGoodStripCharge.parameterNames_()) != 0:
        bad.append('minGoodStripCharge set (supported: empty PSet)')
    if bad:
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s is outside the supported configuration: %s'
                           % (mkf.label_(), '; '.join(bad)))


def _checkMkFitFitConfig(menu):
    for name in menu.parameterNames_():
        if name in _FIT_HONOURED:
            continue
        val = getattr(menu, name).value()
        if name == 'storeHitStates':
            if not val:
                continue
            raise RuntimeError('customizeHLTforMkFitAlpaka: %s has storeHitStates = True (procModifier mtd_at_hlt): '
                               'the device mkFit fit does not provide per-hit states' % FIT)
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s parameter %s = %r is not supported by the device fit'
                           % (FIT, name, val))


def _esByComponentName(process, name, typeSubstr):
    found = [e for e in process.es_producers_().values()
             if typeSubstr in e.type_() and hasattr(e, 'ComponentName') and e.ComponentName.value() == name]
    if len(found) != 1:
        raise RuntimeError('customizeHLTforMkFitAlpaka: expected one %s ES producer with ComponentName %s, found %d'
                           % (typeSubstr, name, len(found)))
    return found[0]


def _checkCpeConfig(process, fitModule):
    """The menu fit's CPE is the PixelCPEGeneric the device code implements, and the PixelCPEFastParams producer (the
    module constants of the device tables) has the same Lorentz settings. The C++ side cross-checks the tables per IOV."""
    gen = _esByComponentName(process, fitModule.pixelCPE.value(), 'PixelCPEGeneric')
    fast = _esByComponentName(process, 'PixelCPEFastParamsPhase2', 'PixelCPEFastParams')

    def val(es, k, default):
        return getattr(es, k).value() if hasattr(es, k) else default

    bad = ['%s=%s (supported %s)' % (k, val(gen, k, d), v) for k, (v, d) in _CPE_GENERIC_SUPPORTED.items()
           if val(gen, k, d) != v]
    bad += ['Lorentz %s: PixelCPEGeneric %s vs PixelCPEFastParamsPhase2 %s' % (k, val(gen, k, d), val(fast, k, d))
            for k, d in _CPE_LORENTZ_DEFAULTS.items() if val(gen, k, d) != val(fast, k, d)]
    if bad:
        raise RuntimeError('customizeHLTforMkFitAlpaka: unsupported CPE configuration: %s' % '; '.join(bad))
    return fast


def _checkPixelRecHitsFromSoA(process, eoh):
    """The device hit transform takes row k of a module from SoA row moduleStart + originalId(k): right only if the
    legacy pixel rechits are the legacy copy of the same rechit SoA."""
    rh = getattr(process, eoh.pixelRecHits.getModuleLabel(), None)
    soa = eoh.pixelSoA.getModuleLabel()
    if rh is None or not rh.type_().startswith('SiPixelRecHitFromSoAAlpaka') or \
            rh.pixelRecHitSrc.getModuleLabel().split('@')[0] != soa:
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s must be the legacy copy of %s (found %s)'
                           % (eoh.pixelRecHits.getModuleLabel(), soa, rh.type_() if rh is not None else None))


# ---- the steps, in the order they are applied ----
def _swapBuild(process):
    """Device EventOfHits and device building; the menu building module leaves the menu and a status check of the
    device building takes its place."""
    menuEOH = getattr(process, EOH)
    menuBuild = getattr(process, BUILD)
    if menuEOH.type_() != 'MkFitEventOfHitsProducer' or menuBuild.type_() != 'MkFitProducer':
        raise RuntimeError('customizeHLTforMkFitAlpaka: expected the mkFit-fit menu modules %s/%s, found %s/%s'
                           % (EOH, BUILD, menuEOH.type_(), menuBuild.type_()))
    _checkMkFitBuildConfig(menuBuild)
    if not hasattr(process, 'hltMkFitAlpakaESProducer'):
        process.hltMkFitAlpakaESProducer = cms.ESProducer(
            'MkFitAlpakaESProducer@alpaka', ComponentName=cms.string(ES_LABEL),
            iterationConfig=cms.ESInputTag('', 'hltInitialStepTrackCandidatesMkFitConfig'))

    setattr(process, EOH, cms.EDProducer('MkFitAlpakaEventOfHitsProducer@alpaka',
                                         usePixelQualityDB=cms.bool(menuEOH.usePixelQualityDB.value()),
                                         esData=cms.ESInputTag('', ES_LABEL)))
    setattr(process, BUILD_DEVICE, cms.EDProducer('MkFitAlpakaBuildProducer@alpaka',
                                                  seeds=cms.InputTag(menuBuild.seeds.value()),
                                                  pixelHits=cms.InputTag(menuBuild.pixelHits.value()),
                                                  eventOfHits=cms.InputTag(EOH),
                                                  esData=cms.ESInputTag('', ES_LABEL),
                                                  removeDuplicates=cms.bool(menuBuild.removeDuplicates.value()),
                                                  # the C++ configuration check sees the menu values too
                                                  clustersToSkip=cms.InputTag(menuBuild.clustersToSkip.getModuleLabel(),
                                                                              menuBuild.clustersToSkip.getProductInstanceLabel(),
                                                                              menuBuild.clustersToSkip.getProcessName()),
                                                  buildingRoutine=cms.string(menuBuild.buildingRoutine.value()),
                                                  seedCleaning=cms.bool(menuBuild.seedCleaning.value()),
                                                  backwardFitInCMSSW=cms.bool(menuBuild.backwardFitInCMSSW.value()),
                                                  # backward-search gate: the beam spot of the menu EventOfHits
                                                  beamSpot=cms.InputTag(menuEOH.beamSpot.value()),
                                                  lstSeedFit=cms.bool(False)))
    chk = cms.EDAnalyzer('MkFitAlpakaStatusCheck', src=cms.InputTag(BUILD_DEVICE), requireClean=cms.bool(False),
                         summaryFile=cms.string(''))
    setattr(process, BUILD + 'Status', chk)
    _removeFromDirectSequences(process, menuBuild, chk)
    delattr(process, BUILD)
    process.hltMkFitAlpakaTask = cms.Task(getattr(process, BUILD_DEVICE))
    process.HLTInitialStepSequence.associate(process.hltMkFitAlpakaTask)


def _deviceFit(process):
    """The device mkFit final fit on the building output, with the PixelCPEGeneric track-angle CPE."""
    if not hasattr(process, FIT):
        raise RuntimeError('customizeHLTforMkFitAlpaka: no %s in the menu (procModifier trackingMkFitFit needed)' % FIT)
    menu = getattr(process, FIT)
    if menu.type_() != 'MkFitFitProducer':
        raise RuntimeError('customizeHLTforMkFitAlpaka: expected MkFitFitProducer as %s, found %s'
                           % (FIT, menu.type_()))
    _checkMkFitFitConfig(menu)
    fast = _checkCpeConfig(process, menu)
    _swapBuild(process)
    if not hasattr(process, 'hltMkFitAlpakaFitCpeESProducer'):
        process.hltMkFitAlpakaFitCpeESProducer = cms.ESProducer(
            'MkFitAlpakaFitCpeESProducer@alpaka', ComponentName=cms.string(CPE_ES_LABEL),
            cpeFastParams=cms.string(fast.ComponentName.value()))
    dev = cms.EDProducer('MkFitAlpakaFitDeviceProducer@alpaka', tracks=cms.InputTag(BUILD_DEVICE),
                         eventOfHits=cms.InputTag(EOH), pixelHits=cms.InputTag(menu.mkFitPixelHits.getModuleLabel()),
                         esData=cms.ESInputTag('', ES_LABEL), candCutSel=menu.candCutSel,
                         candMinPtCut=menu.candMinPtCut, candMinNHitsCut=menu.candMinNHitsCut,
                         candMinPtRelaxedCut=menu.candMinPtRelaxedCut,
                         candMinAbsEtaForRelaxedCut=menu.candMinAbsEtaForRelaxedCut, cpe=cms.bool(True),
                         cpeTables=cms.string(CPE_ES_LABEL), pixelCPE=cms.string(menu.pixelCPE.value()),
                         cpeCheck=cms.string('throw'))
    setattr(process, FIT_DEVICE, dev)
    process.hltMkFitAlpakaTask.add(dev)
    delattr(process, FIT)


def _deviceHits(process):
    """Device hit input: the EventOfHits builds its hits from the pixel rechit SoA and the OT rechits; the
    RecoTracker/MkFit OT hit converter is replaced by a size-only map under the same label (the pixel map comes from
    the EventOfHits, _pixelIndexFromEventOfHits)."""
    eoh = getattr(process, EOH)
    pix, ot = getattr(process, PIX_HITS), getattr(process, OT_HITS)
    if pix.type_() != 'MkFitSiPixelHitConverter' or ot.type_() != 'MkFitPhase2HitConverter':
        raise RuntimeError('customizeHLTforMkFitAlpaka: expected the RecoTracker/MkFit hit converters, found %s/%s'
                           % (pix.type_(), ot.type_()))
    eoh.pixelSoA = cms.InputTag('hltPhase2SiPixelRecHitsSoA')
    eoh.pixelRecHits = cms.InputTag(pix.hits.getModuleLabel())
    eoh.pixelClusters = cms.InputTag(pix.clusters.getModuleLabel())
    eoh.otClusters = cms.InputTag(ot.clusters.getModuleLabel())
    _checkPixelRecHitsFromSoA(process, eoh)
    setattr(process, OT_HITS, cms.EDProducer('MkFitAlpakaPhase2ClusterIndexToHit',
                                             sizeFromClusters=cms.InputTag(ot.clusters.getModuleLabel())))
    bad = ['%s (%s)' % (c, getattr(process, c.split('.')[0]).type_())
           for lab in (PIX_HITS, OT_HITS) for c in _consumers(process, lab)
           if getattr(process, c.split('.')[0]).type_() not in _INDEX_ONLY_READERS]
    if bad:
        raise RuntimeError('customizeHLTforMkFitAlpaka: modules that need the host MkFitHitWrapper read %s/%s: %s'
                           % (PIX_HITS, OT_HITS, bad))


def _eventOfHitsEarlyDelete(process):
    """The device EventOfHits is deleted right after its last reader, the device fit. The fit runs on the EventOfHits'
    queue or waits for it, so the caching allocator cannot hand the block to another queue while it is read."""
    others = [c for c in _consumers(process, EOH) if _inSequenceOrTask(process, c.split('.')[0])
              and not (c.startswith(FIT_DEVICE + '.') or c.startswith(BUILD_DEVICE + '.'))]
    if others:
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s has other readers %s' % (EOH, others))
    getattr(process, FIT_DEVICE).eventOfHitsQueue = cms.InputTag(EOH, 'queue')
    _canDeleteEarly(process, EOH, _EOH_TYPES)


def _otSoAEarlyDelete(process):
    """The device OT rechit SoA is deleted right after its last reader. Its device readers, the EventOfHits and the LST
    input, order its allocation queue after their reads, so the caching allocator cannot hand the block to another
    queue while it is read. The other parameters read the producer's other products (CA hits, hitModuleStart, keys)."""
    guarded = {EOH: ('otSoA', 'otSoAQueue'), LST_INPUT_DEVICE: ('otSoA', 'otSoAQueue')}
    otherProducts = ('trackerRecHitsSoA', 'outerTrackerRecHitSoAConverterSrc', 'otRecHitsSoA', 'otCAKeyStart')
    others = []
    for c in _consumers(process, OT_DEVICE):
        label, param = c.split('.', 1)
        if _inSequenceOrTask(process, label) and param not in guarded.get(label, ()) + otherProducts:
            others.append(c)
    if others:
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s has other readers %s' % (OT_DEVICE, others))
    _canDeleteEarly(process, OT_DEVICE, _OT_TYPES)


def _lstInputEarlyDelete(process):
    """The LST input collection is deleted right after its only reader, the LST producer, which reuses the input's queue:
    the caching allocator gives the block to another queue only after the LST work queued on it has finished."""
    readers = []

    def visit(pset, n, i, tl, where):
        tag = getattr(pset, n) if i is None else getattr(pset, n)[i]
        main = not isinstance(tag, cms.InputTag) or tag.getProductInstanceLabel() == ''
        if tl == LST_INPUT_DEVICE and main and _inSequenceOrTask(process, where.split('.')[0]):
            readers.append(where)

    for label, m in _modules(process):
        _visitTags(m, label + '.', visit)
    if readers != ['hltLST.lstInput'] or process.hltLST.type_() != 'LSTProducer@alpaka':
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s has readers %s' % (LST_INPUT_DEVICE, readers))
    _canDeleteEarly(process, LST_INPUT_DEVICE, _LSTIN_TYPES)


def _outputConversion(process):
    """hltInitialStepTracks: one host conversion of the device fit's TrackSoA to reco::Track
    (MkFitOutputTrackConverter leaves the menu)."""
    menuConv = process.hltInitialStepTracks
    if menuConv.type_() != 'MkFitOutputTrackConverter':
        raise RuntimeError('customizeHLTforMkFitAlpaka: hltInitialStepTracks is %s, expected MkFitOutputTrackConverter'
                           % menuConv.type_())
    if menuConv.TrajectoryInEvent.value():
        raise RuntimeError('customizeHLTforMkFitAlpaka: TrajectoryInEvent (mtd_at_hlt) needs per-hit states')
    if menuConv.src.getModuleLabel() != FIT:
        raise RuntimeError('customizeHLTforMkFitAlpaka: hltInitialStepTracks.src is %s' % menuConv.src.value())
    v = getattr(menuConv, 'ttrhBuilder', None)
    if v is not None:
        # the device conversion uses the device fit's pixel CPE and the CPE of the legacy OT rechits
        builder = _esByComponentName(process, v.getDataLabel(), 'TkTransientTrackingRecHitBuilder')
        if (builder.PixelCPE.value() != getattr(process, FIT_DEVICE).pixelCPE.value()
                or builder.Phase2StripCPE.value() != process.hltSiPhase2RecHits.Phase2StripCPE.getDataLabel()):
            raise RuntimeError('customizeHLTforMkFitAlpaka: hltInitialStepTracks.ttrhBuilder = %s is not supported'
                               % v.value())
    process.hltInitialStepTracks = cms.EDProducer(
        'MkFitAlpakaOutputTrackConverter',
        tracks=cms.InputTag(FIT_DEVICE), status=cms.InputTag(FIT_DEVICE),
        mkFitPixelHits=menuConv.mkFitPixelHits, mkFitStripHits=menuConv.mkFitStripHits, seeds=menuConv.seeds,
        propagatorAlong=menuConv.propagatorAlong, propagatorOpposite=menuConv.propagatorOpposite,
        qualityMaxInvPt=menuConv.qualityMaxInvPt, qualityMinTheta=menuConv.qualityMinTheta,
        qualityMaxR=menuConv.qualityMaxR, qualityMaxZ=menuConv.qualityMaxZ,
        qualityMaxPosErr=menuConv.qualityMaxPosErr, qualitySignPt=menuConv.qualitySignPt,
        NavigationSchool=menuConv.NavigationSchool, measurementTrackerEvent=menuConv.measurementTrackerEvent,
        beamSpot=cms.InputTag('offlineBeamSpot'),  # hard-coded in MkFitOutputTrackConverter
        TrajectoryInEvent=cms.bool(False))


def _otRecHitsDevice(process):
    """Device OT rechits: they feed the CA OT layers, the EventOfHits and the LST input; the legacy OT rechit producer
    leaves the menu and its other readers make their few OT hits on demand from the clusters."""
    ca = getattr(process, CA_OT)
    cpe = process.hltSiPhase2RecHits.Phase2StripCPE
    setattr(process, OT_DEVICE, cms.EDProducer('MkFitAlpakaOTRecHitsProducer@alpaka',
                                               src=cms.InputTag(OT_CLUSTERS), Phase2StripCPE=cpe,
                                               produceCAHits=cms.bool(True), beamSpot=ca.beamSpot,
                                               pixelRecHitSoASource=ca.pixelRecHitSoASource))
    process.hltMkFitAlpakaTask.add(getattr(process, OT_DEVICE))
    for name, m in process.producers_().items():
        for p in ('trackerRecHitsSoA', 'outerTrackerRecHitSoAConverterSrc'):
            if hasattr(m, p) and getattr(m, p).getModuleLabel() == CA_OT:
                setattr(m, p, cms.InputTag(OT_DEVICE))
    for s in process.sequences_().values():
        s.remove(ca)
    for t in process.tasks_().values():
        t.remove(ca)
    delattr(process, CA_OT)
    eoh = getattr(process, EOH)
    eoh.otSoA = cms.InputTag(OT_DEVICE)
    # the EventOfHits orders the producer's queue after its read (early deletion)
    eoh.otSoAQueue = cms.InputTag(OT_DEVICE, 'queue')
    blank = cms.InputTag('')
    for name, m in process.producers_().items():
        t = m.type_()
        if t.endswith('LSTInputProducer') or t.startswith('LSTInputProducer'):
            m.phase2OTRecHits = blank  # replaced by hltInputLSTDevice
        elif t == 'MkFitAlpakaOutputTrackConverter':
            m.otClustersOnDemand = cms.InputTag(OT_CLUSTERS)
            m.Phase2StripCPE = cpe
        elif t == 'PixelTrackProducerFromSoAAlpaka' and m.useOTExtension.value():
            m.otClustersOnDemand = cms.InputTag(OT_CLUSTERS)
            m.Phase2StripCPE = cpe
            m.outerTrackerRecHitSrc = blank
    legacy = getattr(process, OT_RECHITS)
    readers = sorted(set(c.split('.')[0] for c in _consumers(process, OT_RECHITS, skip=(OT_RECHITS,))))
    validation = [l for l in readers if l in process.analyzers_() or 'Validat' in getattr(process, l).type_()]
    production = [l for l in readers if l not in validation]
    if production:
        raise RuntimeError('customizeHLTforMkFitAlpaka: readers of %s left: %s' % (OT_RECHITS, production))
    for s in process.sequences_().values():
        s.remove(legacy)
    for t in process.tasks_().values():
        t.remove(legacy)
    if validation:  # DQM validation of the legacy rechits: the producer stays available unscheduled
        process.hltMkFitAlpakaTask.add(legacy)


def _buildEarlyDelete(process):
    """The build TrackSoA is deleted after the device fit (covered by the fit's EventOfHits queue guard)."""
    readers = [c for c in _consumers(process, BUILD) if _inSequenceOrTask(process, c.split('.')[0])]
    if readers:
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s has readers %s' % (BUILD, readers))
    left = [c for c in _consumers(process, BUILD_DEVICE)
            if _inSequenceOrTask(process, c.split('.')[0]) and not c.startswith(FIT_DEVICE + '.')
            and not c.startswith(BUILD + 'Status.')]
    if left:
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s has other readers %s' % (BUILD_DEVICE, left))
    _canDeleteEarly(process, BUILD_DEVICE, _TRK_TYPES)


def _lightSeeds(process):
    """The mkFit seeds carry hits and a placeholder state; the build module fits the T5/T4/pT3/pT5 seed states on the
    device (one forward pass, origin prior 3) and drops failed fits (DEVIATIONS: light seeds)."""
    b = getattr(process, BUILD_DEVICE)
    b.lstSeedFit = cms.bool(True)
    b.lstSeedFitPasses = cms.int32(1)
    b.lstSeedFitErrScale = cms.double(1.0)
    b.lstSeedFitOriginPrior = cms.int32(3)
    if getattr(process, LST_SEEDS).produceTrackCandidates.value():
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s produces track candidates' % LST_SEEDS)
    b.lstSeedFitDropFailed = cms.bool(True)


def _fitDeviations(process):
    """Final-fit outlier handling and the output algorithm label (DEVIATIONS: edge outliers, outlier rounds, negative
    chi2, propagation to the first hit, algorithm)."""
    f = getattr(process, FIT_DEVICE)
    f.edgeOutliers = cms.bool(True)
    f.outlierRounds = cms.int32(3)
    conv = process.hltInitialStepTracks
    conv.dropNegativeChi2 = cms.bool(True)
    f.firstHitProp = cms.int32(1)
    conv.algorithm = cms.string('initialStep')


def _nearEndOutliers(process):
    """NEAR_END_OUTLIERS: the outlier test of the 3 hits nearest the outer end, one outlier per round, rounds until none
    is left (as the KF fit-smoother)."""
    f = getattr(process, FIT_DEVICE)
    f.nearEdgeHitsOuter = cms.int32(3)
    f.outliersPerRound = cms.int32(1)
    f.outlierRounds = cms.int32(0)


def _lstInputDevice(process):
    """hltInputLSTDevice replaces hltInputLST and the host pixel seeds: the pLS parameters from the device KF of the
    pixel-track hits (the host seed creator's structure, then the beam-line PCA), the OT rows from the device OT
    rechits; the build module takes the pixel seed states from the device."""
    seeds = process.hltInitialStepSeeds
    if (seeds.type_() != 'SeedGeneratorFromProtoTracksEDProducer' or seeds.useProtoTrackKinematics.value()
            or seeds.removeOTRechits.value() or seeds.produceComplement.value() or seeds.usePV.value()
            or seeds.InputVertexCollection.value() not in ('', '""') or not seeds.useEventsWithNoVertex.value()
            or seeds.InputCollection.value() != 'hltPhase2PixelTracks'):
        raise RuntimeError('customizeHLTforMkFitAlpaka: hltInitialStepSeeds configuration not supported')
    ca = getattr(process, PIXEL_TRACKS_CA)
    if not ca.type_().startswith('CAHitNtupletAlpaka'):
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s is not the pixel-track CA' % PIXEL_TRACKS_CA)
    eoh = getattr(process, EOH)
    dev = cms.EDProducer('MkFitAlpakaLstInputProducer@alpaka',
                         ptCut=process.hltInputLST.ptCut,
                         otSoA=cms.InputTag(OT_DEVICE),
                         otSoAQueue=cms.InputTag(OT_DEVICE, 'queue'),
                         pixelRecHits=cms.InputTag('hltSiPixelRecHits'),
                         otRecHitsSoA=cms.InputTag(OT_DEVICE),
                         pixelTracksSoA=process.hltPhase2PixelTracks.trackSrc,
                         pixelRecHitSrc=ca.pixelRecHitSrc,
                         trackerRecHitsSoA=ca.trackerRecHitsSoA,
                         beamSpot=process.hltInputLST.beamSpot,
                         eventOfHits=cms.InputTag(EOH),
                         pixelSrcRows=cms.InputTag(EOH, 'pixelSrcRows'),
                         esData=cms.ESInputTag('', ES_LABEL),
                         otCAKeyStart=cms.InputTag(OT_DEVICE, 'caKeyStart'),
                         alpaka=cms.untracked.PSet(backend=cms.untracked.string('')))
    setattr(process, LST_INPUT_DEVICE, dev)
    eoh.producePixelSrcRows = cms.bool(True)
    # one OT module table for the LST input and the EventOfHits
    if not hasattr(process, 'mkFitAlpakaHitModuleTable'):
        process.mkFitAlpakaHitModuleTable = cms.ESProducer('MkFitAlpakaEventOfHitsModuleTableESProducer@alpaka',
                                                           ComponentName=cms.string(MODULE_TABLE_LABEL))
    eoh.moduleTable = cms.string(MODULE_TABLE_LABEL)
    seq = process.HLTInitialStepSequence
    seq.insert(seq.index(process.hltInputLST), dev)
    seq.remove(process.hltInputLST)
    process.hltLST.lstInput = LST_INPUT_DEVICE
    b = getattr(process, BUILD_DEVICE)
    b.pixelSeedStates = cms.InputTag(LST_INPUT_DEVICE, 'pixelSeedStates')
    seq.remove(seeds)
    # the device EventOfHits (it does not depend on LST) runs before the LST input, which reads its hits
    inp = process.HLTMkFitInputSequence
    if seq.index(inp) > seq.index(dev):
        seq.remove(inp)
        seq.insert(seq.index(dev), inp)


def _seedHandoff(process):
    """The mkFit seeds are made on the device from the LST track candidates and the pixel seed hit lists (they replace
    LSTOutputConverter + MkFitSeedConverter, no host copy of the candidates); no TrajectorySeed collection is made."""
    lstSeeds = getattr(process, LST_SEEDS)
    mkSeeds = getattr(process, MKFIT_SEEDS)
    if lstSeeds.type_() != 'LSTOutputConverter' or mkSeeds.type_() != 'MkFitSeedConverter':
        raise RuntimeError('customizeHLTforMkFitAlpaka: expected LSTOutputConverter / MkFitSeedConverter, found %s / %s'
                           % (lstSeeds.type_(), mkSeeds.type_()))
    if mkSeeds.seeds.getModuleLabel() != LST_SEEDS:
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s reads %s' % (MKFIT_SEEDS, mkSeeds.seeds.getModuleLabel()))
    build = getattr(process, BUILD_DEVICE)
    if not (build.lstSeedFit.value() and build.lstSeedFitOriginPrior.value() == 3 and build.lstSeedFitDropFailed.value()):
        raise RuntimeError('customizeHLTforMkFitAlpaka: the seed hand-off needs the device LST seed fit')
    handoff = cms.EDProducer('MkFitAlpakaSeedHandoffK1@alpaka',
                             lstOutput=lstSeeds.lstOutput,
                             pixelSeedStates=build.pixelSeedStates,
                             eventOfHits=build.eventOfHits,
                             pixelTracks=cms.InputTag('hltPhase2PixelTracks'),
                             otClusters=cms.InputTag(OT_CLUSTERS),
                             Phase2StripCPE=getattr(process, OT_RECHITS).Phase2StripCPE,
                             includeFourthHit=cms.bool(process.hltInitialStepSeeds.includeFourthHit.value()),
                             dropOTHitsPurePLS=lstSeeds.dropOTHitsPurePLS,
                             maxITHitsToDropOTHitsPurePLS=lstSeeds.maxITHitsToDropOTHitsPurePLS,
                             # the placeholder state is not converted for the seeds the device seed fit refits
                             deviceFitStates=cms.bool(True),
                             SeedCreatorPSet=lstSeeds.SeedCreatorPSet.clone())
    setattr(process, MKFIT_SEEDS, handoff)
    build.deviceSeeds = cms.InputTag(MKFIT_SEEDS)
    build.pixelTrackOfSeed = cms.InputTag(MKFIT_SEEDS, 'pixelTrackOfSeed')
    trk = process.hltInitialStepTracks
    if trk.seeds.getModuleLabel() == LST_SEEDS:
        trk.seeds = ''
        trk.mkFitSeeds = cms.InputTag('')
        trk.seedCount = cms.InputTag(MKFIT_SEEDS, 'pixelTrackOfSeed')
    # the seed collection leaves the menu: a remaining reader fails with ProductNotFound
    delattr(process, LST_SEEDS)


def _navigationDevice(process):
    """The converter's missing-hit navigation searched on the device after the fit, on flat layer tables built once
    per IOV; calls the portable search cannot decide go to DetLayer::compatibleDets on the host."""
    conv = process.hltInitialStepTracks
    if not hasattr(process, 'hltMkFitAlpakaNavFlatTables'):
        process.hltMkFitAlpakaNavFlatTables = cms.ESProducer('MkFitAlpakaNavFlatTablesESProducer',
                                                             appendToDataLabel=cms.string(''))
    conv.navPortable = cms.bool(True)
    setattr(process, NAV_DEVICE, cms.EDProducer('MkFitAlpakaNavDeviceProducer@alpaka',
                                                tracks=cms.InputTag(FIT_DEVICE),
                                                NavigationSchool=conv.NavigationSchool))
    process.hltMkFitAlpakaTask.add(getattr(process, NAV_DEVICE))
    conv.navDevice = cms.InputTag(NAV_DEVICE)


def _pixelIndexFromEventOfHits(process):
    """The pixel MkFitClusterIndexToHit comes from the EventOfHits (filled in its pass over the legacy pixel rechits);
    the menu pixel hit converter leaves the menu."""
    eoh = getattr(process, EOH)
    pix = getattr(process, PIX_HITS)
    if eoh.pixelRecHits.getModuleLabel() != pix.hits.getModuleLabel():
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s reads %s, %s reads %s'
                           % (EOH, eoh.pixelRecHits.getModuleLabel(), PIX_HITS, pix.hits.getModuleLabel()))
    eoh.producePixelIndexToHit = cms.bool(True)
    _retag(process, PIX_HITS, cms.InputTag(EOH, 'pixelIndexToHit'), skip=(PIX_HITS,))
    left = _consumers(process, PIX_HITS, skip=(PIX_HITS,))
    if left:
        raise RuntimeError('customizeHLTforMkFitAlpaka: readers of %s left: %s' % (PIX_HITS, left))
    delattr(process, PIX_HITS)


def _pcaDevice(process):
    """The converter's state at the beam-line PCA from the device (closed-form field volume); rows the device cannot
    do fall back to TSCBLBuilderNoMaterial on the host."""
    conv = process.hltInitialStepTracks
    setattr(process, OUTCONV_STATES, cms.EDProducer('MkFitAlpakaOutConvStateProducer@alpaka',
                                                    tracks=cms.InputTag(FIT_DEVICE), beamSpot=conv.beamSpot))
    process.hltMkFitAlpakaTask.add(getattr(process, OUTCONV_STATES))
    conv.pcaStates = cms.InputTag(OUTCONV_STATES)


def customizeHLTforMkFitAlpaka(process, nearEndOutliers=None):
    """The mkFit chain of the LST initial step on the device (needs --procModifiers trackingMkFitFit)."""
    if not hasattr(process, BUILD):
        return process
    _deviceFit(process)
    _deviceHits(process)
    _outputConversion(process)
    _eventOfHitsEarlyDelete(process)
    _otRecHitsDevice(process)
    _buildEarlyDelete(process)
    _lightSeeds(process)
    _fitDeviations(process)
    if NEAR_END_OUTLIERS if nearEndOutliers is None else nearEndOutliers:
        _nearEndOutliers(process)
    _lstInputDevice(process)
    _seedHandoff(process)
    _navigationDevice(process)
    _pixelIndexFromEventOfHits(process)
    _pcaDevice(process)
    _otSoAEarlyDelete(process)
    _lstInputEarlyDelete(process)
    return process


def customizeHLTforMkFitAlpakaNearEndOutliers(process):
    """customizeHLTforMkFitAlpaka with the near-end outlier test of the final fit (NEAR_END_OUTLIERS)."""
    return customizeHLTforMkFitAlpaka(process, nearEndOutliers=True)
