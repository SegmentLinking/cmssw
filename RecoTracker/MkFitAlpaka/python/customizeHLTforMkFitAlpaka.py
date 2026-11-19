# Phase-2 HLT: the mkFit chain of the LST initial step on the device with Alpaka (one code for GPU and CPU backends).
# Switched on with --procModifiers trackingMkFitFit,trackingMkFitAlpaka (HLT_75e33 menus), or applied with --customise
# RecoTracker/MkFitAlpaka/customizeHLTforMkFitAlpaka.customizeHLTforMkFitAlpaka on top of the menu with --procModifiers
# trackingMkFitFit. Module labels of the menu are kept, so downstream InputTags resolve
# unchanged. What runs where:
#   hltMkFitAlpakaOTRecHits         device OT rechits (also the CA OT hit SoA); the legacy OT rechit producer leaves the
#                                   menu, its readers make the few OT hits they need on demand
#   hltMkFitEventOfHits             device hit input + EventOfHits binning (pixel rechit SoA + OT rechit SoA; the pixel
#                                   rows in SoA order, no legacy pixel input)
#   hltMkFitSiPixelHits             host map of the EventOfHits' pixel rows to the legacy pixel rechits (unscheduled, for
#                                   the host readers)
#   hltInputLSTDevice               device LST input; pLS parameters from the device KF of the pixel-track hits
#   hltMkFitPixelTrackCount         host counts of the device pixel tracks (the LST input numbers them on the device as
#                                   hltPhase2PixelTracks does; no legacy pixel input on the device path)
#   hltInitialStepMkFitSeeds        device hand-off of the LST track candidates to the mkFit seeds
#   hltInitialStepTrackCandidatesMkFitDevice     device building (seed fit, clone engine, backward fit, cleaning)
#   hltInitialStepTrackCandidatesMkFitFitDevice  device mkFit final fit (with the pixel CPE)
#   hltInitialStepOutConvStates / hltInitialStepNavDevice  beam-line PCA states and missing-hit navigation on device
#   hltInitialStepTracks            host conversion to reco::Track
# Pixel seeds of the LST input that are not the seeds of the pixel-track CA (hltInitialStepSeeds), for example the mkFit
# seeder's (phase2MkSeeder): with a producer of their generic device rows labelled hltInitialStepPixelSeedsDevice
# (MkFitAlpakaLstInputProducer pixelSeeds; row k = the seed of pixel track k of hltPhase2PixelTracks) they take the
# device path as the CA's (hltInputLSTDevice, the device seed hand-off); without it they keep the host LST input and
# seed path (hltInputLST, LSTOutputConverter, MkFitSeedConverter). The host mkFit hit converters and the legacy OT
# rechits stay when modules other than the chain read them.
# Physics differences to the reference mkFit are listed in doc/DEVIATIONS.txt.
import FWCore.ParameterSet.Config as cms

ES_LABEL = 'hltMkFitAlpakaES'
GEOMETRY_ES_LABEL = 'hltMkFitAlpakaGeometry'
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
PIXEL_TRACK_COUNT = 'hltMkFitPixelTrackCount'
LST_SEEDS = 'hltInitialStepTrajectorySeedsLST'
MKFIT_SEEDS = 'hltInitialStepMkFitSeeds'
OUTCONV_STATES = 'hltInitialStepOutConvStates'
DEVICE_PIXEL_SEEDS = 'hltInitialStepPixelSeedsDevice'
NAV_DEVICE = 'hltInitialStepNavDevice'

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
                       'MkFitAlpakaFitDeviceProducer@alpaka', 'MkFitAlpakaPhase2ClusterIndexToHit',
                       'MkFitAlpakaOutputTrackConverter')


# the EventOfHits' host products (not deleted early)
_EOH_HOST_PRODUCTS = ('pixelIndexToHit', 'pixelSrcRows', 'queue')
# device readers of the EventOfHits that finish their device work before they end (stream::SynchronizingEDProducer:
# the framework waits for their queue between acquire and produce), so the early deletion after the fit is safe for them
_EOH_SYNCHRONIZING_READERS = ('MkSeederProducer@alpaka',)


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
    module constants of the device tables) has the same Lorentz settings. The C++ side cross-checks the tables per
    IOV."""
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


def _caPixelSeeds(process):
    """True if the LST input reads the seeds of the pixel-track CA (hltInitialStepSeeds), which hltInputLSTDevice
    reproduces on the device; other pixel seeds (e.g. the mkFit seeder's, phase2MkSeeder) keep the host LST input and
    the host seed path (LSTOutputConverter + MkFitSeedConverter)."""
    seeds = [t.getModuleLabel() if isinstance(t, cms.InputTag) else str(t).split(':')[0]
             for t in process.hltInputLST.pixelSeeds]
    return seeds == ['hltInitialStepSeeds'] and _inSequenceOrTask(process, 'hltInitialStepSeeds')


def _devicePixelSeeds(process):
    """True for non-CA pixel seeds of the LST input whose producer also gives their generic device rows
    (MkFitAlpakaLstInputProducer pixelSeeds) under the label hltInitialStepPixelSeedsDevice: the LST input and the seed
    hand-off then run on the device as for the CA seeds."""
    return not _caPixelSeeds(process) and hasattr(process, DEVICE_PIXEL_SEEDS)


def _wrapperReaders(process, label):
    """Modules that read the host MkFitHitWrapper of 'label' (not only its MkFitClusterIndexToHit)."""
    return ['%s (%s)' % (c, getattr(process, c.split('.')[0]).type_()) for c in _consumers(process, label, skip=(label,))
            if getattr(process, c.split('.')[0]).type_() not in _INDEX_ONLY_READERS
            and _inSequenceOrTask(process, c.split('.')[0])]


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
    if not hasattr(process, 'hltMkFitAlpakaGeometryESProducer'):
        # the geometry-only device ES product (no iteration configuration): the iteration's product is built on it
        process.hltMkFitAlpakaGeometryESProducer = cms.ESProducer(
            'MkFitAlpakaGeometryESProducer@alpaka', ComponentName=cms.string(GEOMETRY_ES_LABEL))
    if not hasattr(process, 'hltMkFitAlpakaESProducer'):
        # DEVIATION DEV-11: track building with the field constants fitted to the field map
        process.hltMkFitAlpakaESProducer = cms.ESProducer(
            'MkFitAlpakaESProducer@alpaka', ComponentName=cms.string(ES_LABEL),
            iterationConfig=cms.ESInputTag('', 'hltInitialStepTrackCandidatesMkFitConfig'),
            geometry=cms.ESInputTag('', GEOMETRY_ES_LABEL),
            buildingFitField=cms.bool(True))

    setattr(process, EOH, cms.EDProducer('MkFitAlpakaEventOfHitsProducer@alpaka',
                                         usePixelQualityDB=cms.bool(menuEOH.usePixelQualityDB.value()),
                                         esData=cms.ESInputTag('', ES_LABEL)))
    skip = menuBuild.clustersToSkip
    setattr(process, BUILD_DEVICE, cms.EDProducer('MkFitAlpakaBuildProducer@alpaka',
                                                  seeds=cms.InputTag(menuBuild.seeds.value()),
                                                  nPixelHits=cms.InputTag(EOH, 'nPixel'),
                                                  eventOfHits=cms.InputTag(EOH),
                                                  esData=cms.ESInputTag('', ES_LABEL),
                                                  removeDuplicates=cms.bool(menuBuild.removeDuplicates.value()),
                                                  # the C++ configuration check sees the menu values too
                                                  clustersToSkip=cms.InputTag(skip.getModuleLabel(),
                                                                              skip.getProductInstanceLabel(),
                                                                              skip.getProcessName()),
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
                         eventOfHits=cms.InputTag(EOH), nPixelHits=cms.InputTag(EOH, 'nPixel'),
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
    RecoTracker/MkFit OT hit converter is replaced by a size-only map under the same label. Without host readers of
    the converters' hits, the pixel rows are in pixel rechit SoA order (no legacy pixel input on the device path; the
    pixel map comes from a host module, _pixelIndexFromEventOfHits)."""
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
    # modules that need the host MkFitHitWrapper (e.g. the mkFit seeder, phase2MkSeeder) keep the host converters
    # (same hit order as the device EventOfHits; the index maps then come from them)
    if _wrapperReaders(process, PIX_HITS) or _wrapperReaders(process, OT_HITS):
        return
    setattr(process, OT_HITS, cms.EDProducer('MkFitAlpakaPhase2ClusterIndexToHit',
                                             sizeFromClusters=cms.InputTag(ot.clusters.getModuleLabel())))
    eoh.pixelRowsFromSoA = cms.bool(True)
    # the readers of the rows' pixel clusters take them from the same SoA: the device fit's CPE, and other device readers
    # of the EventOfHits with a 'pixelSoA' parameter
    getattr(process, FIT_DEVICE).pixelSoA = eoh.pixelSoA
    for label in sorted(set(c.split('.')[0] for c in _consumers(process, EOH, skip=(EOH,)))):
        if hasattr(getattr(process, label), 'pixelSoA'):
            getattr(process, label).pixelSoA = eoh.pixelSoA


def _eventOfHitsEarlyDelete(process):
    """The device EventOfHits is deleted right after its last reader, the device fit. The fit runs on the EventOfHits'
    queue or orders that queue after its work (no host wait), so the caching allocator cannot hand the block to another
    queue while it is read. Other readers: of its host products (the pixel cluster index maps, the queue handle), or
    synchronizing device readers (_EOH_SYNCHRONIZING_READERS: their device work is done when they end, before the
    deletion)."""
    def allowed(c):
        label, param = c.split('.', 1)
        if label in (FIT_DEVICE, BUILD_DEVICE) or getattr(process, label).type_() in _EOH_SYNCHRONIZING_READERS:
            return True
        tags = []
        _visitTags(getattr(process, label), label + '.',
                   lambda pset, n, i, tl, where: tags.append(getattr(pset, n) if i is None else getattr(pset, n)[i])
                   if where == c and tl == EOH else None)
        return all(isinstance(t, cms.InputTag) and t.getProductInstanceLabel() in _EOH_HOST_PRODUCTS for t in tags)
    others = [c for c in _consumers(process, EOH) if _inSequenceOrTask(process, c.split('.')[0]) and not allowed(c)]
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
    """The LST input collection is deleted right after its only reader, the LST producer, which reuses the input's
    queue: the caching allocator gives the block to another queue only after the LST work queued on it has finished."""
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
    if menuConv.trajectoryInEvent.value():
        raise RuntimeError('customizeHLTforMkFitAlpaka: trajectoryInEvent (mtd_at_hlt) needs per-hit states')
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
        algorithmName=menuConv.algorithmName, beamSpot=menuConv.beamSpot, trajectoryInEvent=cms.bool(False))


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
        if (t.endswith('LSTInputProducer') or t.startswith('LSTInputProducer')) and (
                _caPixelSeeds(process) or _devicePixelSeeds(process)):
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
    # modules outside the schedule do not run (the vector hit CA's OT rechit SoA without phase2CAVectorHits)
    production = [l for l in readers if l not in validation and _inSequenceOrTask(process, l)]
    if production:
        # host readers (the host LST input and mkFit OT hit converter of non-CA pixel seeds): the producer stays
        if _caPixelSeeds(process):
            raise RuntimeError('customizeHLTforMkFitAlpaka: readers of %s left: %s' % (OT_RECHITS, production))
        return
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


def _nearEndOutliers(process):
    """DEVIATION DEV-10: the final fit's outlier test of the 3 hits nearest the outer end (each with >= 3 other layers
    before it), window-only hits only in a round without another flagged hit, one outlier per round (as the KF
    fit-smoother)."""
    f = getattr(process, FIT_DEVICE)
    f.nearEdgeHitsOuter = cms.int32(3)
    f.outliersPerRound = cms.int32(1)


def _hitModuleTable(process):
    """The OT module table of the device EventOfHits (shared with the LST input)."""
    if not hasattr(process, 'mkFitAlpakaHitModuleTable'):
        process.mkFitAlpakaHitModuleTable = cms.ESProducer('MkFitAlpakaEventOfHitsModuleTableESProducer@alpaka',
                                                           ComponentName=cms.string(MODULE_TABLE_LABEL))
    getattr(process, EOH).moduleTable = cms.string(MODULE_TABLE_LABEL)


def _lstInputDevice(process):
    """hltInputLSTDevice replaces hltInputLST and the host pixel seeds: the pLS parameters from the device KF of the
    pixel-track hits (the host seed creator's structure, then the beam-line PCA), the OT rows from the device OT
    rechits; the build module takes the pixel seed states from the device."""
    seeds = process.hltInitialStepSeeds
    # removeOTRechits keeps the pixel hits and the P-sensor hits of the TOB PS modules, i.e. every hit of pixel tracks that
    # do not expand vector hits: with includeFourthHit the seeds are then made from all the hits, as without it
    allHits = not seeds.removeOTRechits.value() or (seeds.includeFourthHit.value() and not getattr(
        process.hltPhase2PixelTracks, 'expandVectorHits', cms.bool(False)).value())
    if (seeds.type_() != 'SeedGeneratorFromProtoTracksEDProducer' or seeds.useProtoTrackKinematics.value()
            or not allHits or seeds.produceComplement.value() or seeds.usePV.value()
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
    if getattr(eoh, 'pixelRowsFromSoA', cms.bool(False)).value():
        # the EventOfHits' pixel rows are the SoA rows: pixelSrcRows is the exact identity, no legacy pixel input
        dev.pixelRecHits = cms.InputTag('')
    # the pixel tracks' index (seedIdx = the hltPhase2PixelTracks index) made on the device when the converter's
    # selection is the one emulated (quality >= tight as the LST input's, every track converted otherwise)
    pt = process.hltPhase2PixelTracks
    if (pt.type_() == 'PixelTrackProducerFromSoAAlpaka' and pt.trackSrc.value() == dev.pixelTracksSoA.value()
            and getattr(pt, 'minQuality', cms.string('loose')).value() == 'tight'
            and getattr(pt, 'minNumberOfHits', cms.int32(0)).value() <= 3
            and not getattr(pt, 'requireQuadsFromConsecutiveLayers', cms.bool(False)).value()):
        setattr(process, PIXEL_TRACK_COUNT, cms.EDProducer('MkFitAlpakaPixelTrackCountProducer@alpaka',
                                                           pixelTracksSoA=dev.pixelTracksSoA,
                                                           minQuality=cms.string('tight'),
                                                           alpaka=cms.untracked.PSet(backend=cms.untracked.string(''))))
        process.hltMkFitAlpakaTask.add(getattr(process, PIXEL_TRACK_COUNT))
        dev.pixelTracks = cms.InputTag('')
        dev.pixelTrackCount = cms.InputTag(PIXEL_TRACK_COUNT)
    # one OT module table for the LST input and the EventOfHits
    _hitModuleTable(process)
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


def _lstInputDeviceFromSeeds(process):
    """hltInputLSTDevice from the generic device pixel seeds (hltInitialStepPixelSeedsDevice) in place of hltInputLST and
    the host pixel seeds: the pLS from the rows' states (with the rows' optional 'plsStates': the state each pLS starts
    from, hltInputLST's stateAtLastPixelHit), the OT rows from the device OT rechits; the build module takes the pixel
    seed states from the device."""
    inp = process.hltInputLST
    src = getattr(process, DEVICE_PIXEL_SEEDS)

    def opt(m, name, default):
        return getattr(m, name).value() if hasattr(m, name) else default

    atLastPixel = opt(inp, 'stateAtLastPixelHit', False)
    if (atLastPixel != opt(src, 'stateAtLastPixelHit', False)
            or (atLastPixel and opt(inp, 'stateAtLastPixelHitMaxPt', 0.) != opt(src, 'stateAtLastPixelHitMaxPt', 0.))):
        raise RuntimeError('customizeHLTforMkFitAlpaka: the stateAtLastPixelHit settings of hltInputLST and %s differ'
                           % DEVICE_PIXEL_SEEDS)
    if not _inSequenceOrTask(process, DEVICE_PIXEL_SEEDS):
        raise RuntimeError('customizeHLTforMkFitAlpaka: %s is not scheduled' % DEVICE_PIXEL_SEEDS)
    dev = cms.EDProducer('MkFitAlpakaLstInputProducer@alpaka',
                         ptCut=inp.ptCut,
                         otSoA=cms.InputTag(OT_DEVICE),
                         otSoAQueue=cms.InputTag(OT_DEVICE, 'queue'),
                         beamSpot=inp.beamSpot,
                         esData=cms.ESInputTag('', ES_LABEL),
                         pixelSeeds=cms.InputTag(DEVICE_PIXEL_SEEDS),
                         pixelSeedsPLSStates=cms.InputTag(DEVICE_PIXEL_SEEDS, 'plsStates') if atLastPixel
                         else cms.InputTag(''),
                         alpaka=cms.untracked.PSet(backend=cms.untracked.string('')))
    setattr(process, LST_INPUT_DEVICE, dev)
    _hitModuleTable(process)
    seq = process.HLTInitialStepSequence
    seq.insert(seq.index(inp), dev)
    seq.remove(inp)
    process.hltLST.lstInput = LST_INPUT_DEVICE
    b = getattr(process, BUILD_DEVICE)
    b.pixelSeedStates = cms.InputTag(LST_INPUT_DEVICE, 'pixelSeedStates')
    # DEVIATION: the device seeds are built without the iteration's seed cleaner (MkFitCore's, e.g. phase1:default of
    # the mkFit seeder's lstStep configuration, runs only on host seeds; none is ported to the device)
    b.seedCleaning = cms.bool(False)


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
    if not (build.lstSeedFit.value() and build.lstSeedFitOriginPrior.value() == 3
            and build.lstSeedFitDropFailed.value()):
        raise RuntimeError('customizeHLTforMkFitAlpaka: the seed hand-off needs the device LST seed fit')
    # the pixel seeds' hits: the CA seeds' creator option, all the hits of generic device pixel seeds
    fourthHit = process.hltInitialStepSeeds.includeFourthHit.value() if _caPixelSeeds(process) else True
    # the dropOTHitsPurePLS creator refits on the device (the LST seed fit's creator emulation, its field model) when
    # the creator is the one emulated; else on the host from the legacy pixel tracks
    creator = lstSeeds.SeedCreatorPSet

    def cval(name, default):
        return getattr(creator, name).value() if hasattr(creator, name) else default
    deviceRefit = (cval('ComponentName', '') == 'SeedFromConsecutiveHitsCreator'
                   and cval('propagator', '') == 'PropagatorWithMaterial' and cval('MinOneOverPtError', 1.) == 1.
                   and cval('OriginTransverseErrorMultiplier', 1.) == 1.
                   and not cval('forceKinematicWithRegionDirection', False) and cval('magneticField', '') == '')
    handoff = cms.EDProducer('MkFitAlpakaSeedHandoffK1@alpaka',
                             lstOutput=lstSeeds.lstOutput,
                             pixelSeedStates=build.pixelSeedStates,
                             eventOfHits=build.eventOfHits,
                             pixelTracks=cms.InputTag('' if deviceRefit else 'hltPhase2PixelTracks'),
                             esData=cms.ESInputTag('', ES_LABEL),
                             refitPasses=build.lstSeedFitPasses,
                             refitBackwardTol=getattr(build, 'lstSeedFitBackwardTol', cms.double(0.01)),
                             otClusters=cms.InputTag(OT_CLUSTERS),
                             Phase2StripCPE=getattr(process, OT_RECHITS).Phase2StripCPE,
                             includeFourthHit=cms.bool(fourthHit),
                             dropOTHitsPurePLS=lstSeeds.dropOTHitsPurePLS,
                             maxITHitsToDropOTHitsPurePLS=lstSeeds.maxITHitsToDropOTHitsPurePLS,
                             # the placeholder state is not converted for the seeds the device seed fit refits
                             deviceFitStates=cms.bool(True),
                             SeedCreatorPSet=lstSeeds.SeedCreatorPSet.clone())
    setattr(process, MKFIT_SEEDS, handoff)
    lstIn = getattr(process, LST_INPUT_DEVICE, None)
    if not deviceRefit and lstIn is not None and hasattr(lstIn, 'pixelTrackCount'):
        # the host refits find their pixel track by the hltPhase2PixelTracks index: the LST input uses its map (the
        # device index can order pixel tracks of equal SoA pT differently)
        lstIn.pixelTracks = cms.InputTag('hltPhase2PixelTracks')
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
    """The pixel MkFitClusterIndexToHit of the EventOfHits' pixel rows. Rows in SoA order: from
    MkFitAlpakaPixelClusterIndexToHit under the converter's label, unscheduled (only host readers wait for the legacy
    rechits), readers of the EventOfHits' map moved to it, the seed hand-off's host requests matched through it.
    Legacy row order: from the EventOfHits (filled in its pass over the legacy pixel rechits); the menu pixel hit
    converter leaves the menu."""
    eoh = getattr(process, EOH)
    pix = getattr(process, PIX_HITS)
    if _wrapperReaders(process, PIX_HITS):  # the host converter stays (_deviceHits)
        return
    if getattr(eoh, 'pixelRowsFromSoA', cms.bool(False)).value():
        if eoh.pixelRecHits.getModuleLabel() != pix.hits.getModuleLabel():  # the legacy copy of eoh.pixelSoA
            raise RuntimeError('customizeHLTforMkFitAlpaka: %s reads %s, %s reads %s'
                               % (EOH, eoh.pixelRecHits.getModuleLabel(), PIX_HITS, pix.hits.getModuleLabel()))
        _removeFromDirectSequences(process, pix)
        for t in process.tasks_().values():
            t.remove(pix)
        setattr(process, PIX_HITS, cms.EDProducer('MkFitAlpakaPixelClusterIndexToHit', pixelSoA=eoh.pixelSoA,
                                                  pixelRecHits=pix.hits))
        process.hltMkFitAlpakaTask.add(getattr(process, PIX_HITS))

        def visit(pset, n, i, tl, where):
            t = getattr(pset, n) if i is None else getattr(pset, n)[i]
            if tl == EOH and isinstance(t, cms.InputTag) and t.getProductInstanceLabel() == 'pixelIndexToHit':
                if i is None:
                    setattr(pset, n, cms.InputTag(PIX_HITS))
                else:
                    getattr(pset, n)[i] = cms.InputTag(PIX_HITS)
        for l, m in _modules(process):
            _visitTags(m, l + '.', visit)
        if (hasattr(process, MKFIT_SEEDS) and getattr(process, MKFIT_SEEDS).type_() == 'MkFitAlpakaSeedHandoffK1@alpaka'
                and getattr(process, MKFIT_SEEDS).pixelTracks.getModuleLabel() != ''):  # the host refits' hit match
            getattr(process, MKFIT_SEEDS).pixelHits = cms.InputTag(PIX_HITS)
        return
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


def customizeHLTforMkFitAlpaka(process):
    """The mkFit chain of the LST initial step on the device (needs --procModifiers trackingMkFitFit)."""
    if not hasattr(process, BUILD) or hasattr(process, BUILD_DEVICE):  # not the mkFit-fit menu, or applied already
        return process
    _deviceFit(process)
    _deviceHits(process)
    _outputConversion(process)
    _eventOfHitsEarlyDelete(process)
    _otRecHitsDevice(process)
    _buildEarlyDelete(process)
    caSeeds = _caPixelSeeds(process)
    deviceSeeds = caSeeds or _devicePixelSeeds(process)
    if deviceSeeds:
        _lightSeeds(process)
    else:
        # host seeds: the iteration's seed cleaner (MkFitCore, e.g. phase1:default) runs on them in the build module
        getattr(process, BUILD_DEVICE).seedCleanerConfig = cms.ESInputTag('', 'hltInitialStepTrackCandidatesMkFitConfig')
    _nearEndOutliers(process)
    if caSeeds:
        _lstInputDevice(process)
        _seedHandoff(process)
    elif deviceSeeds:
        _lstInputDeviceFromSeeds(process)
        _seedHandoff(process)
    else:
        _hitModuleTable(process)
    _navigationDevice(process)
    _pixelIndexFromEventOfHits(process)
    _pcaDevice(process)
    _otSoAEarlyDelete(process)
    if deviceSeeds:
        _lstInputEarlyDelete(process)
    return process


def customizeHLTforMkFitAlpakaSeedValidation(process):
    """Validation only (not for production menus): with the device seed hand-off, which makes no TrajectorySeeds, a
    TrajectorySeed collection of the device mkFit seeds (the build module's seeds with the states building starts from)
    under the label of the LST seeds it replaces (hltInitialStepTrajectorySeedsLST), so that the seed-level validation
    (TrackFromSeedProducer hltInitialStepTrajectorySeedsLSTTracks, MTV) runs unchanged. The build module exports its
    seed rows (produceFittedSeeds; one extra device-to-host copy per event). No change without the hand-off (the host
    seed path keeps its TrajectorySeeds)."""
    if hasattr(process, LST_SEEDS) or not hasattr(process, MKFIT_SEEDS) \
            or getattr(process, MKFIT_SEEDS).type_() != 'MkFitAlpakaSeedHandoffK1@alpaka':
        return process
    handoff = getattr(process, MKFIT_SEEDS)
    build = getattr(process, BUILD_DEVICE)
    build.produceFittedSeeds = cms.bool(True)
    eoh = getattr(process, EOH)
    pixelHits = (cms.InputTag(EOH, 'pixelIndexToHit') if getattr(eoh, 'producePixelIndexToHit', cms.bool(False)).value()
                 else cms.InputTag(PIX_HITS))
    setattr(process, LST_SEEDS, cms.EDProducer('MkFitAlpakaSeedsToTrajectorySeeds',
                                               seeds=cms.InputTag(BUILD_DEVICE, 'fittedSeeds'),
                                               mkFitPixelHits=pixelHits,
                                               otClusters=handoff.otClusters,
                                               Phase2StripCPE=handoff.Phase2StripCPE))
    process.hltMkFitAlpakaTask.add(getattr(process, LST_SEEDS))
    return process
