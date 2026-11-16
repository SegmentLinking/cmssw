import FWCore.ParameterSet.Config as cms

hltLST = cms.EDProducer('LSTProducer@alpaka',
    lstInput = cms.InputTag('hltInputLST'),
    verbose = cms.bool(False),
    ptCut = cms.double(0.8),
    nopLSDupClean = cms.bool(True),
    tcpLSTriplets = cms.bool(True),
    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)


from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toModify(hltLST, nopLSDupClean = False,
                             tcpLSTriplets = False)

# Pixel segments (pLS) with an outer-tracker hit are cleaned against a T5 only if the T5 contains that hit: the dR and
# embedding tests alone match neighbouring but different tracks, for example in jets. Used with the mkFit seeder's pixel
# seeds (phase2MkSeeder) and in the all-Alpaka LST + mkFit initial step (trackingMkFitAlpaka); the default menu keeps
# the plain test.
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
from Configuration.ProcessModifiers.trackingMkFitAlpaka_cff import trackingMkFitAlpaka
(phase2MkSeeder | trackingMkFitAlpaka).toModify(hltLST, plsT5SharedHit = cms.bool(True))
