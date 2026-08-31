import FWCore.ParameterSet.Config as cms
qualityCutDictionaryPrompt = cms.PSet(
   InitialStep         =        cms.vdouble(0.26,   0.45,  0.54),
   HighPtTripletStep   =        cms.vdouble(0.26,   0.34,  0.34), #99.5%, 99%, 99% efficiency
   LowPtQuadStep       =        cms.vdouble(0.26,   0.45,  0.54),
   LowPtTripletStep    =        cms.vdouble(0.26,   0.45,  0.54),
   DetachedQuadStep    =        cms.vdouble(0.26,   0.45,  0.54),
   PixelPairStep       =        cms.vdouble(0.26,   0.45,  0.54)
)
qualityCutDictionaryDisplaced = cms.PSet(
   InitialStep         =        cms.vdouble(0.11,   0.23,  0.28),
   HighPtTripletStep   =        cms.vdouble(0.11,   0.14,  0.14), #99.5%, 99%, 99% efficiency
   LowPtQuadStep       =        cms.vdouble(0.11,   0.23,  0.28),
   LowPtTripletStep    =        cms.vdouble(0.11,   0.23,  0.28),
   DetachedQuadStep    =        cms.vdouble(0.11,   0.23,  0.28),
   PixelPairStep       =        cms.vdouble(0.11,   0.23,  0.28)
)
