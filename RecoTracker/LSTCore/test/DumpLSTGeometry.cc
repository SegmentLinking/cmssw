#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/LSTCore/interface/LSTGeometry/LSTGeometry.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/IO.h"

class DumpLSTGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DumpLSTGeometry(const edm::ParameterSet& config);

private:
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  edm::ESGetToken<lstgeometry::LSTGeometry, TrackerRecoGeometryRecord> lstGeoToken_;

  std::string outputDirectory_;
  bool binaryOutput_;
};

DumpLSTGeometry::DumpLSTGeometry(const edm::ParameterSet& config)
    : lstGeoToken_{esConsumes()},
      outputDirectory_(config.getUntrackedParameter<std::string>("outputDirectory", "data/")),
      binaryOutput_(config.getUntrackedParameter<bool>("output_as_binary", true)) {}

void DumpLSTGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& lstg = iSetup.getData(lstGeoToken_);

  lstgeometry::writeCentroids(lstg.centroids, outputDirectory_ + "sensor_centroids", binaryOutput_);
  lstgeometry::writeSlopes(
      lstg.barrel_slopes, lstg.sensor_info, outputDirectory_ + "tilted_barrel_orientation", binaryOutput_);
  lstgeometry::writeSlopes(
      lstg.endcap_slopes, lstg.sensor_info, outputDirectory_ + "endcap_orientation", binaryOutput_);
  lstgeometry::writePixelMaps(lstg.pixel_map, outputDirectory_ + "pixelmap/pLS_map", binaryOutput_);
  lstgeometry::writeModuleConnections(
      lstg.merged_line_connections, outputDirectory_ + "module_connection_tracing_merged", binaryOutput_);

  edm::LogInfo("DumpLSTGeometry") << "Centroids size: " << lstg.centroids.size() << std::endl;
}

DEFINE_FWK_MODULE(DumpLSTGeometry);