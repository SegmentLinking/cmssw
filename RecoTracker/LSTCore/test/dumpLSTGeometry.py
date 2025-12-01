import FWCore.ParameterSet.Config as cms

trackingLSTCommon = cms.Modifier()
trackingLST = cms.ModifierChain(trackingLSTCommon)

###################################################################
# Set default phase-2 settings
###################################################################
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)

# No era in Fireworks/Geom reco dumper
process = cms.Process('DUMP', _PH2_ERA, trackingLST)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, '')

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.LSTGeometryESProducer = dict(limit=-1)

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1


process.add_(cms.ESProducer("LSTGeometryESProducer"))

defaultOutputDirectory="data"

process.dump = cms.EDAnalyzer("DumpLSTGeometry",
                              outputDirectory = cms.untracked.string(defaultOutputDirectory)
                              )

print("Requesting LST geometry dump into directory:", defaultOutputDirectory, "\n");
process.p = cms.Path(process.dump)