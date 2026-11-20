import FWCore.ParameterSet.Config as cms

# This modifier runs the Phase-2 HLT pixel CA on the inner-tracker layers only (CAHitNtupletAlpakaPhase2), instead
# of the default CA extended to the outer-tracker barrel layers (CAHitNtupletAlpakaPhase2OT)
phase2ITonlyCA = cms.Modifier()
