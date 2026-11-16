import FWCore.ParameterSet.Config as cms

# This modifier replaces the pixel-track seeds of the Phase-2 HLT initialStep (LST input) with the seeds of the
# mkFit seeder (MkSeederProducer, RecoTracker/MkSeederAlpaka; alpaka: it runs on a GPU or on the CPU backend),
# built from the inner-tracker hits and the outer-tracker layers 1-3, classified by the seed classifier on alpaka
phase2MkSeeder = cms.Modifier()
