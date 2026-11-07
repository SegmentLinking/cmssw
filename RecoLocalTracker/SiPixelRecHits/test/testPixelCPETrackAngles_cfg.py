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
process.pixelGenErrorTablesPhase2 = cms.ESProducer("PixelGenErrorTablesESProducerAlpakaPhase2@alpaka",
    PixelCPEFastParams = cms.string("PixelCPEFastParamsPhase2"),
    eff_charge_cut_lowX = hltESPPixelCPEGeneric.eff_charge_cut_lowX,
    eff_charge_cut_lowY = hltESPPixelCPEGeneric.eff_charge_cut_lowY,
    eff_charge_cut_highX = hltESPPixelCPEGeneric.eff_charge_cut_highX,
    eff_charge_cut_highY = hltESPPixelCPEGeneric.eff_charge_cut_highY,
    size_cutX = hltESPPixelCPEGeneric.size_cutX,
    size_cutY = hltESPPixelCPEGeneric.size_cutY,
    EdgeClusterErrorX = hltESPPixelCPEGeneric.EdgeClusterErrorX,
    EdgeClusterErrorY = hltESPPixelCPEGeneric.EdgeClusterErrorY
)

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1
process.options.accelerators = [args.accelerators]

process.testPixelCPETrackAngles = cms.EDProducer("TestPixelCPETrackAngles@alpaka",
    PixelCPE = cms.string("hltESPPixelCPEGeneric"),
    moduleStride = cms.int32(args.moduleStride)
)
process.path = cms.Path(process.testPixelCPETrackAngles)
