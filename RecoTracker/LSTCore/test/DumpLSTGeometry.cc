#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

class DumpLSTGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DumpLSTGeometry(const edm::ParameterSet& config);

private:
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  std::string outputDirectory_;
};

DumpLSTGeometry::DumpLSTGeometry(const edm::ParameterSet& config)
    : outputDirectory_(config.getUntrackedParameter<std::string>("outputDirectory", "data")) {}

void DumpLSTGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

DEFINE_FWK_MODULE(DumpLSTGeometry);