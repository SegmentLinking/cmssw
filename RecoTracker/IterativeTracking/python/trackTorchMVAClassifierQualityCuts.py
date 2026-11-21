import FWCore.ParameterSet.Config as cms
qualityCutDictionaryPrompt = cms.PSet(
   InitialStep         =        cms.vdouble(0.26,   0.45,  0.54),
   HighPtTripletStep   =        cms.vdouble(0.318, 0.384, 0.384), #99.5%, 99%, 99% efficiency, pT < 5 GeV
   LowPtQuadStep       =        cms.vdouble(0.26,   0.45,  0.54),
   LowPtTripletStep    =        cms.vdouble(0.26,   0.45,  0.54),
   DetachedQuadStep    =        cms.vdouble(0.26,   0.45,  0.54),
   PixelPairStep       =        cms.vdouble(0.26,   0.45,  0.54)
)
qualityCutDictionaryDisplaced = cms.PSet(
   InitialStep         =        cms.vdouble(0.11,   0.23,  0.28),
   HighPtTripletStep   =        cms.vdouble(0.138, 0.175, 0.175), #99.5%, 99%, 99% efficiency, pT < 5 GeV
   LowPtQuadStep       =        cms.vdouble(0.11,   0.23,  0.28),
   LowPtTripletStep    =        cms.vdouble(0.11,   0.23,  0.28),
   DetachedQuadStep    =        cms.vdouble(0.11,   0.23,  0.28),
   PixelPairStep       =        cms.vdouble(0.11,   0.23,  0.28)
)
# Cuts for tracks with pT >= ptThreshold (5 GeV); steps not listed use the same cuts at all pT
qualityCutDictionaryPromptHighPt = cms.PSet(
   HighPtTripletStep   =        cms.vdouble(0.170, 0.237, 0.237), #99.5%, 99%, 99% efficiency, pT >= 5 GeV
)
qualityCutDictionaryDisplacedHighPt = cms.PSet(
   HighPtTripletStep   =        cms.vdouble(0.108, 0.126, 0.126), #99.5%, 99%, 99% efficiency, pT >= 5 GeV
)
