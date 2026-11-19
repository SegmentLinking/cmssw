// Host check of the MkFitAlpaka event products written by MkFitAlpakaProductsTest (any backend; the framework
// copies device products to the host). Recomputes every filled value; throws on the first event with a mismatch.

#include <cstring>

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
#include "RecoTracker/MkFitAlpaka/interface/CandStoreProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/TrackProduct.h"

namespace {
  // same as mkfitdev::productstest::val / constants (plugins/alpaka/MkFitProductsTestKernels.h, device-side header)
  inline float val(uint32_t e, int r, int k) { return float((e % 64) * 1024 + (r % 1024)) + 0.125f * k; }
  constexpr int kHits = 300, kLayers = 5, kBins = 512, kTracks = 200, kSeeds = 40, kHotsPerSeed = 16;
}  // namespace

class MkFitAlpakaProductsCheck : public edm::global::EDAnalyzer<> {
public:
  explicit MkFitAlpakaProductsCheck(edm::ParameterSet const& ps)
      : eohToken_{consumes(ps.getParameter<edm::InputTag>("src"))},
        eohPooledToken_{consumes(ps.getParameter<edm::InputTag>("src"))},
        trkToken_{consumes(ps.getParameter<edm::InputTag>("src"))},
        candToken_{consumes(ps.getParameter<edm::InputTag>("src"))} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("productsTest"));
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID, edm::Event const& ev, edm::EventSetup const&) const override {
    const uint32_t e = ev.id().event();
    // the host copy (GPU backends) or, on CPU backends, the pooled product itself
    auto const hEoh = ev.getHandle(eohToken_);
    auto const hPooled = ev.getHandle(eohPooledToken_);
    if (!hEoh.isValid() && !hPooled.isValid())
      throw cms::Exception("ProductNotFound") << "MkFitAlpakaProductsCheck: no EventOfHits product";
    auto const& trk = ev.get(trkToken_);
    auto const& cand = ev.get(candToken_);
    long bad = 0, checked = 0;
    auto chk = [&](bool ok) {
      ++checked;
      bad += !ok;
    };
    auto v = hEoh.isValid() ? hEoh->const_view() : hPooled->const_view();
    auto hits = v.hits();
    auto layers = v.layers();
    auto binned = v.binnedHits();
    auto bins = v.bins();
    chk(hits.metadata().size() == kHits && layers.metadata().size() == kLayers && binned.metadata().size() == kHits &&
        bins.metadata().size() == kBins);
    chk(hits.nPixel() == e && hits.nStrip() == kHits && layers.nBinsTotal() == kBins);
    for (int i = 0; i < kHits; ++i) {
      chk(hits[i].x() == val(e, i, 0) && hits[i].y() == val(e, i, 1) && hits[i].z() == val(e, i, 2));
      chk(hits[i].e00() == val(e, i, 3) && hits[i].e10() == val(e, i, 4) && hits[i].e11() == val(e, i, 5) &&
          hits[i].e20() == val(e, i, 6) && hits[i].e21() == val(e, i, 7) && hits[i].e22() == val(e, i, 8));
      chk(hits[i].layer() == i % kLayers);
      chk(binned[i].rank() == uint32_t(kHits - 1 - i) && binned[i].phi() == val(e, i, 9));
    }
    for (int l = 0; l < kLayers; ++l)
      chk(layers[l].binBegin() == uint32_t(l * (kBins / kLayers)) && layers[l].nHits() == e + l);
    for (int b = 0; b < kBins; ++b)
      chk(bins[b].content() == e * 7 + b && bins[b].dead() == (b % 3 == 0));

    auto t = trk.const_view();
    chk(t.metadata().size() == kTracks && t.nTracks() == kTracks);
    for (int r = 0; r < kTracks; ++r) {
      // params[0..2] = hit r position and errors[0..5] = hit r errors, through pack::loadHit on the device
      for (int k = 0; k < 6; ++k)
        chk(t[r].params().v[k] == (k < 3 ? val(e, r, k) : val(e, r, 20 + k)));
      for (int k = 0; k < 21; ++k)
        chk(t[r].errors().v[k] == (k < 6 ? val(e, r, 3 + k) : val(e, r, 40 + k)));
      chk(t[r].charge() == ((r % 2) ? 1 : -1) && t[r].chi2() == val(e, r, 70) && t[r].label() == r);
    }

    auto c = cand.const_view();
    chk(c.seeds().metadata().size() == kSeeds && c.seeds().hotsPerSeed() == kHotsPerSeed);
    chk(c.slots().metadata().size() == kSeeds * ::mkfitdev::kSlotsPerSeed &&
        c.hots().metadata().size() == kSeeds * kHotsPerSeed);
    for (int s = 0; s < kSeeds; ++s) {
      chk(c.seeds()[s].nCands() == 1 && c.seeds()[s].seedOriginIdx() == int32_t(e + s));
      // slot 0 of seed s = track s, through pack::loadTrack + pack::storeCandState on the device
      auto const& st = c.slots()[::mkfitdev::candSlotRow(s, 0, 0)].state();
      chk(std::memcmp(st.par, t[s].params().v, sizeof(st.par)) == 0);
      chk(std::memcmp(st.err, t[s].errors().v, sizeof(st.err)) == 0);
      chk(st.charge == t[s].charge());
    }
    for (int h = 0; h < kSeeds * kHotsPerSeed; ++h)
      chk(c.hots()[h].node().index == h);

    edm::LogPrint("MkFitAlpakaProductsCheck")
        << "PRODUCTS_CHECK event " << e << " checked " << checked << " mismatches " << bad;
    if (bad)
      throw cms::Exception("MkFitAlpakaProductsCheck") << bad << " mismatches in event " << e;
  }

private:
  const edm::EDGetTokenT<mkfitdev::EventOfHitsHostCollection> eohToken_;
  const edm::EDGetTokenT<mkfitdev::EventOfHitsPooledHostCollection> eohPooledToken_;
  const edm::EDGetTokenT<mkfitdev::TrackSoAHostCollection> trkToken_;
  const edm::EDGetTokenT<mkfitdev::CandStoreHostCollection> candToken_;
};

DEFINE_FWK_MODULE(MkFitAlpakaProductsCheck);
