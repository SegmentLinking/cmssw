# The host-hit fallback of the mkFit seeder in the all-Alpaka menu (phase2MkSeeder with trackingMkFitAlpaka, where the
# seeder reads its hits from the device EventOfHits): the seeder reads the host mkFit hit converters' MkFitHitWrappers
# again (RecoTracker/MkFit's MkFitSiPixelHitConverter and MkFitPhase2HitConverter on the legacy pixel and OT rechits,
# under their own labels: the all-Alpaka customise has replaced the menu's). Everything else stays as it is.
# Use as a cmsDriver customisation:
#   --customise RecoTracker/MkSeederAlpaka/customizeMkSeederHostHits.customizeMkSeederHostHits
import FWCore.ParameterSet.Config as cms


def customizeMkSeederHostHits(process):
    seeder = getattr(process, 'hltMkFitSeederSeedsMkFitDevice', None)
    if seeder is None or not hasattr(seeder, 'eventOfHits') or seeder.eventOfHits.value() == '':
        return process
    from HLTrigger.Configuration.HLT_75e33.modules.hltMkFitSiPixelHits_cfi import hltMkFitSiPixelHits as _pix
    from HLTrigger.Configuration.HLT_75e33.modules.hltMkFitSiPhase2Hits_cfi import hltMkFitSiPhase2Hits as _ot
    process.hltMkSeederSiPixelHits = _pix.clone()
    process.hltMkSeederSiPhase2Hits = _ot.clone()
    seeder.eventOfHits = cms.InputTag('')
    if hasattr(seeder, 'pixelSoA'):
        seeder.pixelSoA = cms.InputTag('')
    seeder.pixelHits = cms.InputTag('hltMkSeederSiPixelHits')
    seeder.stripHits = cms.InputTag('hltMkSeederSiPhase2Hits')
    # the converters and the legacy OT rechits they read run when the seeder asks for them
    process.hltMkSeederHostHitsTask = cms.Task(process.hltMkSeederSiPixelHits, process.hltMkSeederSiPhase2Hits,
                                               process.hltSiPhase2RecHits)
    process.HLTPhase2PixelTracksAndVerticesSequence.associate(process.hltMkSeederHostHitsTask)
    # the seeds' pixel hit indices are then the legacy cluster keys: the device EventOfHits back to the legacy pixel row
    # order (its pixel map from its host pass, the SoA-row map module leaves the menu)
    from RecoTracker.MkFitAlpaka.customizeHLTforMkFitAlpaka import EOH, PIX_HITS, _retag, _consumers
    eoh = getattr(process, EOH, None)
    if eoh is not None and getattr(eoh, 'pixelRowsFromSoA', cms.bool(False)).value():
        eoh.pixelRowsFromSoA = False
        eoh.producePixelIndexToHit = cms.bool(True)
        for label in sorted(set(c.split('.')[0] for c in _consumers(process, EOH, skip=(EOH,)))):
            m = getattr(process, label)
            if hasattr(m, 'pixelSoA'):
                m.pixelSoA = cms.InputTag('')
        pix = getattr(process, PIX_HITS)
        for t in process.tasks_().values():
            t.remove(pix)
        _retag(process, PIX_HITS, cms.InputTag(EOH, 'pixelIndexToHit'), skip=(PIX_HITS,))
        delattr(process, PIX_HITS)
    return process
