// MkFitClusterIndexToHit replacements of the RecoTracker/MkFit hit converters under their menu labels:
// - MkFitAlpakaPhase2ClusterIndexToHit (OT, MkFitPhase2HitConverter): size only. The readers of its map in the device
//   menu need only the size (the output conversion makes the OT hits on demand from the clusters). The map has one
//   nullptr entry per OT cluster, the size mkfit::convertHits gives with one rechit per cluster; no rechits are read.
// - MkFitAlpakaPixelClusterIndexToHit (pixel, MkFitSiPixelHitConverter): the map of the device EventOfHits' pixel rows
//   in pixel rechit SoA order (MkFitAlpakaEventOfHitsProducer pixelRowsFromSoA) to the legacy pixel rechits, for the
//   host readers (output conversion, seed hand-off host requests, validation). Entry r = the rechit of SoA row r =
//   moduleStart[det] + originalId of its cluster; the row is accepted only if the SoA local position is bitwise the
//   rechit's (a copy of it), else (and for clusters without originalId) it is found by position in the module. A host
//   module off the device path: the device EventOfHits does not wait for the legacy clusters / rechits.
#include <algorithm>
#include <atomic>
#include <utility>

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
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

class MkFitAlpakaPixelClusterIndexToHit : public edm::global::EDProducer<> {
public:
  explicit MkFitAlpakaPixelClusterIndexToHit(edm::ParameterSet const& iConfig)
      : soaToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelSoA"))},
        hitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelRecHits"))},
        putToken_{produces()} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add("pixelSoA", edm::InputTag{"hltPhase2SiPixelRecHitsSoA"})
        ->setComment("the pixel rechit SoA (host copy): the device EventOfHits' pixel rows");
    desc.add("pixelRecHits", edm::InputTag{"hltSiPixelRecHits"})->setComment("the legacy pixel rechits of the SoA");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
    auto const& soa = iEvent.get(soaToken_);
    auto const& pixRH = iEvent.get(hitsToken_);
    const auto hv = soa.const_view().trackingHits();
    const auto mv = soa.const_view().hitModules();
    const uint32_t nSoa = soa.nHits(), nMod = soa.nModules();
    MkFitClusterIndexToHit out;
    auto& map = out.hits();
    map.resize(nSoa, nullptr);
    uint32_t keyMismatch = 0, positionSearch = 0, nHits = 0;
    for (const auto& ds : pixRH) {
      if (ds.empty())
        continue;
      const uint32_t gind = ds.begin()->det()->index();
      if (gind >= nMod)
        throw cms::Exception("MkFitAlpakaPixelClusterIndexToHit")
            << "det index " << gind << " beyond the " << nMod << " modules of the pixel rechit SoA";
      const uint32_t b = mv[gind].moduleStart(), e = std::min(mv[gind + 1].moduleStart(), nSoa);
      for (const auto& h : ds) {
        const float lx = h.localPosition().x(), ly = h.localPosition().y();
        const auto id = h.cluster()->originalId();
        uint32_t j = nSoa;
        if (id != SiPixelCluster::invalidClusterId) {
          j = b + id;
          if (!(j < e && hv[j].xLocal() == lx && hv[j].yLocal() == ly)) {
            ++keyMismatch;
            j = nSoa;
          }
        } else {
          ++positionSearch;
        }
        if (j == nSoa) {
          for (uint32_t r = b; r < e; ++r)
            if (hv[r].xLocal() == lx && hv[r].yLocal() == ly && map[r] == nullptr) {
              j = r;
              break;
            }
        }
        if (j >= nSoa || hv[j].detectorIndex() != gind || map[j] != nullptr)
          throw cms::Exception("MkFitAlpakaPixelClusterIndexToHit")
              << "pixel rechit of cluster key " << h.cluster().key() << " (det index " << gind
              << ") has no free row in the pixel rechit SoA";
        map[j] = &h;
        ++nHits;
      }
    }
    if ((keyMismatch > 0 || positionSearch > 0 || nHits != nSoa) && !warned_.exchange(true))
      edm::LogWarning("MkFitAlpakaPixelClusterIndexToHit")
          << "event " << iEvent.id().event() << ": " << nHits << " legacy pixel rechits for " << nSoa
          << " SoA rows; rows found by position: " << keyMismatch << " originalId mismatches, " << positionSearch
          << " clusters without originalId (first occurrence only)";
    iEvent.emplace(putToken_, std::move(out));
  }

private:
  const edm::EDGetTokenT<reco::TrackingRecHitHost> soaToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> hitsToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> putToken_;
  mutable std::atomic<bool> warned_{false};
};

DEFINE_FWK_MODULE(MkFitAlpakaPixelClusterIndexToHit);
