# Device vs host check of the generic CPE with track angles (pixelCPEforDeviceTrackAngles.h), with the Phase-2 HLT
# settings of PixelCPEGeneric and PixelCPEFastParams
import argparse
import sys

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9

parser = argparse.ArgumentParser(prog=sys.argv[0], description="Device vs host check of the track-angle pixel CPE")
parser.add_argument("--accelerators", type=str, default="cpu", help="cpu, gpu-nvidia, gpu-amd")
parser.add_argument("--moduleStride", type=int, default=50, help="check every N-th pixel module")
args = parser.parse_args()

process = cms.Process("TEST", Phase2C22I13M9)
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("Configuration.Geometry.GeometryExtendedRun4D121Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic_T35", "")

from HLTrigger.Configuration.HLT_75e33.eventsetup.hltESPPixelCPEGeneric_cfi import hltESPPixelCPEGeneric
from HLTrigger.Configuration.HLT_75e33.eventsetup.hltESPPixelCPEFastParams_cfi import hltESPPixelCPEFastParamsPhase2
process.hltESPPixelCPEGeneric = hltESPPixelCPEGeneric
process.hltESPPixelCPEFastParamsPhase2 = hltESPPixelCPEFastParamsPhase2
# configured with the parameters of the PixelCPEGeneric it reproduces
process.pixelGenErrorTablesPhase2 = cms.ESProducer("PixelGenErrorTablesESProducerAlpakaPhase2@alpaka",
    cms.PSet(**hltESPPixelCPEGeneric.parameters_()))
process.pixelGenErrorTablesPhase2.ComponentName = "PixelGenErrorTablesPhase2"
process.pixelGenErrorTablesPhase2.PixelCPEFastParams = cms.string("PixelCPEFastParamsPhase2")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1
process.options.accelerators = [args.accelerators]

process.testPixelCPETrackAngles = cms.EDProducer("TestPixelCPETrackAngles@alpaka",
    PixelCPE = cms.string("hltESPPixelCPEGeneric"),
    moduleStride = cms.int32(args.moduleStride)
)
process.path = cms.Path(process.testPixelCPETrackAngles)
