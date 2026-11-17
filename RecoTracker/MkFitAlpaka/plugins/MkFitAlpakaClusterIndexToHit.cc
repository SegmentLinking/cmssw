// Size-only replacement of the RecoTracker/MkFit Phase-2 OT hit converter (MkFitPhase2HitConverter) under its menu
// label: the readers of its MkFitClusterIndexToHit in the device menu need only the size (the device build and fit
// modules: the strip hit row base; the output conversion makes the OT hits on demand from the clusters). The map has
// one nullptr entry per OT cluster, the size mkfit::convertHits gives with one rechit per cluster; no rechits are read.
#include <utility>

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"

class MkFitAlpakaPhase2ClusterIndexToHit : public edm::global::EDProducer<> {
public:
  explicit MkFitAlpakaPhase2ClusterIndexToHit(edm::ParameterSet const& iConfig)
      : clustersToken_{consumes(iConfig.getParameter<edm::InputTag>("sizeFromClusters"))}, putToken_{produces()} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add("sizeFromClusters", edm::InputTag{"hltSiPhase2Clusters"})
        ->setComment("the OT clusters: one (nullptr) map entry per cluster");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
    MkFitClusterIndexToHit out;
    out.hits().resize(iEvent.get(clustersToken_).dataSize(), nullptr);
    iEvent.emplace(putToken_, std::move(out));
  }

private:
  const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clustersToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> putToken_;
};

DEFINE_FWK_MODULE(MkFitAlpakaPhase2ClusterIndexToHit);
