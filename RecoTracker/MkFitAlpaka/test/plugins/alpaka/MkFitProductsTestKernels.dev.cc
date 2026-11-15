#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/test/plugins/alpaka/MkFitProductsTestKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using namespace ::mkfitdev::productstest;

  class KernelFillProducts {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::mkfitdev::EventOfHitsBlocks::View eoh,
                                  ::mkfitdev::TrackSoAView trk,
                                  uint32_t eventNumber) const {
      auto hits = eoh.hits();
      auto layers = eoh.layers();
      auto binned = eoh.binnedHits();
      auto bins = eoh.bins();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, kHits)) {
        hits[i].x() = val(eventNumber, i, 0);
        hits[i].y() = val(eventNumber, i, 1);
        hits[i].z() = val(eventNumber, i, 2);
        hits[i].e00() = val(eventNumber, i, 3);
        hits[i].e10() = val(eventNumber, i, 4);
        hits[i].e11() = val(eventNumber, i, 5);
        hits[i].e20() = val(eventNumber, i, 6);
        hits[i].e21() = val(eventNumber, i, 7);
        hits[i].e22() = val(eventNumber, i, 8);
        hits[i].layer() = i % kLayers;
        binned[i].rank() = kHits - 1 - i;
        binned[i].phi() = val(eventNumber, i, 9);
        if (i == 0) {
          hits.nPixel() = eventNumber;
          hits.nStrip() = kHits;
          layers.nBinsTotal() = kBins;
          trk.nTracks() = kTracks;
        }
      }
      for (int32_t l : cms::alpakatools::uniform_elements(acc, kLayers)) {
        layers[l].binBegin() = l * (kBins / kLayers);
        layers[l].nHits() = eventNumber + l;
      }
      for (int32_t bin : cms::alpakatools::uniform_elements(acc, kBins)) {
        bins[bin].content() = eventNumber * 7 + bin;
        bins[bin].dead() = bin % 3 == 0;
      }
      for (int32_t r : cms::alpakatools::uniform_elements(acc, kTracks)) {
        for (int k = 0; k < 6; ++k)
          trk[r].params().v[k] = val(eventNumber, r, 20 + k);
        for (int k = 0; k < 21; ++k)
          trk[r].errors().v[k] = val(eventNumber, r, 40 + k);
        trk[r].charge() = (r % 2) ? 1 : -1;
        trk[r].chi2() = val(eventNumber, r, 70);
        trk[r].label() = r;
      }
    }
  };

  void fillProductsTest(Queue& queue, EventOfHitsDeviceCollection& eoh, TrackSoADeviceCollection& trk, uint32_t event) {
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(4, 128);
    alpaka::exec<Acc1D>(queue, workDiv, KernelFillProducts{}, eoh.view(), trk.view(), event);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev
