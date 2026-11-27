// Host check of the MkFitAlpaka event products written by MkFitAlpakaProductsTest (any backend; the framework
// copies device products to the host). Recomputes every filled value; throws on the first event with a mismatch.

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/MkFitAlpaka/interface/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/TrackProduct.h"
#include "RecoTracker/MkFitAlpaka/test/plugins/MkFitProductsTestValues.h"

class MkFitAlpakaProductsCheck : public edm::global::EDAnalyzer<> {
public:
  explicit MkFitAlpakaProductsCheck(edm::ParameterSet const& ps)
      : eohToken_{consumes(ps.getParameter<edm::InputTag>("src"))},
        eohPooledToken_{consumes(ps.getParameter<edm::InputTag>("src"))},
        trkToken_{consumes(ps.getParameter<edm::InputTag>("src"))} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("productsTest"));
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID, edm::Event const& ev, edm::EventSetup const&) const override {
    using namespace mkfitdev::productstest;
    const uint32_t eventNumber = ev.id().event();
    // the host copy (GPU backends) or, on CPU backends, the pooled product itself
    auto const hEoh = ev.getHandle(eohToken_);
    auto const hPooled = ev.getHandle(eohPooledToken_);
    if (!hEoh.isValid() && !hPooled.isValid())
      throw cms::Exception("ProductNotFound") << "MkFitAlpakaProductsCheck: no EventOfHits product";
    auto const& trk = ev.get(trkToken_);
    long bad = 0, checked = 0;
    auto chk = [&](bool ok) {
      ++checked;
      bad += !ok;
    };
    auto eohView = hEoh.isValid() ? hEoh->const_view() : hPooled->const_view();
    auto hits = eohView.hits();
    auto layers = eohView.layers();
    auto binned = eohView.binnedHits();
    auto bins = eohView.bins();
    chk(hits.metadata().size() == kHits && layers.metadata().size() == kLayers && binned.metadata().size() == kHits &&
        bins.metadata().size() == kBins);
    chk(hits.nPixel() == eventNumber && hits.nStrip() == kHits && layers.nBinsTotal() == kBins);
    for (int i = 0; i < kHits; ++i) {
      chk(hits[i].x() == val(eventNumber, i, 0) && hits[i].y() == val(eventNumber, i, 1) &&
          hits[i].z() == val(eventNumber, i, 2));
      chk(hits[i].e00() == val(eventNumber, i, 3) && hits[i].e10() == val(eventNumber, i, 4) &&
          hits[i].e11() == val(eventNumber, i, 5) && hits[i].e20() == val(eventNumber, i, 6) &&
          hits[i].e21() == val(eventNumber, i, 7) && hits[i].e22() == val(eventNumber, i, 8));
      chk(hits[i].layer() == i % kLayers);
      chk(binned[i].rank() == uint32_t(kHits - 1 - i) && binned[i].phi() == val(eventNumber, i, 9));
    }
    for (int l = 0; l < kLayers; ++l)
      chk(layers[l].binBegin() == uint32_t(l * (kBins / kLayers)) && layers[l].nHits() == eventNumber + l);
    for (int bin = 0; bin < kBins; ++bin)
      chk(bins[bin].content() == eventNumber * 7 + bin && bins[bin].dead() == (bin % 3 == 0));

    auto tracks = trk.const_view();
    chk(tracks.metadata().size() == kTracks && tracks.nTracks() == kTracks);
    for (int r = 0; r < kTracks; ++r) {
      for (int k = 0; k < 6; ++k)
        chk(tracks[r].params().v[k] == val(eventNumber, r, 20 + k));
      for (int k = 0; k < 21; ++k)
        chk(tracks[r].errors().v[k] == val(eventNumber, r, 40 + k));
      chk(tracks[r].charge() == ((r % 2) ? 1 : -1) && tracks[r].chi2() == val(eventNumber, r, 70) &&
          tracks[r].label() == r);
    }

    edm::LogPrint("MkFitAlpakaProductsCheck")
        << "PRODUCTS_CHECK event " << eventNumber << " checked " << checked << " mismatches " << bad;
    if (bad)
      throw cms::Exception("MkFitAlpakaProductsCheck") << bad << " mismatches in event " << eventNumber;
  }

private:
  const edm::EDGetTokenT<mkfitdev::EventOfHitsHostCollection> eohToken_;
  const edm::EDGetTokenT<mkfitdev::EventOfHitsPooledHostCollection> eohPooledToken_;
  const edm::EDGetTokenT<mkfitdev::TrackSoAHostCollection> trkToken_;
};

DEFINE_FWK_MODULE(MkFitAlpakaProductsCheck);
