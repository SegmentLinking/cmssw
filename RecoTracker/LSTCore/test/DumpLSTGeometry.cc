#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// temporary
#include <string>
#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(std::string);
// end temporary

class DumpLSTGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DumpLSTGeometry(const edm::ParameterSet& config);

private:
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  
  edm::ESGetToken<std::string, TrackerRecoGeometryRecord> lstGeoToken_;

  std::string outputDirectory_;
};

DumpLSTGeometry::DumpLSTGeometry(const edm::ParameterSet& config)
    : lstGeoToken_{esConsumes()}, outputDirectory_(config.getUntrackedParameter<std::string>("outputDirectory", "data")) {}

void DumpLSTGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const auto& lstg = iSetup.getData(lstGeoToken_);
    
    edm::LogInfo("DumpLSTGeometry") << lstg << std::endl;
}

DEFINE_FWK_MODULE(DumpLSTGeometry);