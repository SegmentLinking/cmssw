// Index-only replacement of the RecoTracker/MkFit Phase-2 OT hit converter (MkFitPhase2HitConverter) under its menu
// label: the device EventOfHits makes the mkFit OT hits, so its readers need only the map from the OT cluster key to
// the legacy rechit (the output conversion) and its size (the strip hit row base). Same map as mkfit::convertHits,
// without the hit conversion.
#include <algorithm>
#include <utility>

#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
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
      : hitsToken_{consumes(iConfig.getParameter<edm::InputTag>("hits"))}, putToken_{produces()} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add("hits", edm::InputTag{"siPhase2RecHits"})->setComment("the legacy Phase-2 OT rechits");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
    const auto& hits = iEvent.get(hitsToken_);
    MkFitClusterIndexToHit out;
    auto& map = out.hits();
    if (!hits.empty()) {
      // convertHits size: max(last hit's cluster key + 1, dataSize())
      map.resize(std::max<std::size_t>(hits.data().back().firstClusterRef().index() + 1, hits.dataSize()), nullptr);
      for (const auto& ds : hits)
        for (const auto& hit : ds) {
          const auto key = hit.firstClusterRef().index();
          if (key >= map.size())
            map.resize(key + 1, nullptr);
          map[key] = &hit;
        }
    }
    iEvent.emplace(putToken_, std::move(out));
  }

private:
  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> hitsToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> putToken_;
};

DEFINE_FWK_MODULE(MkFitAlpakaPhase2ClusterIndexToHit);
