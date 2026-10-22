#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexBackend.h"
#include "RecoTracker/MkFitAlpaka/test/plugins/alpaka/MkFitProductsTestKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/Packers.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using namespace ::mkfitdev::productstest;

  class KernelFillProducts {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::mkfitdev::EventOfHitsBlocks::View eoh,
                                  ::mkfitdev::TrackSoAView trk,
                                  ::mkfitdev::CandStoreBlocks::View cand,
                                  uint32_t e) const {
      auto hits = eoh.hits();
      auto layers = eoh.layers();
      auto binned = eoh.binnedHits();
      auto bins = eoh.bins();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, kHits)) {
        hits[i].x() = val(e, i, 0);
        hits[i].y() = val(e, i, 1);
        hits[i].z() = val(e, i, 2);
        hits[i].e00() = val(e, i, 3);
        hits[i].e10() = val(e, i, 4);
        hits[i].e11() = val(e, i, 5);
        hits[i].e20() = val(e, i, 6);
        hits[i].e21() = val(e, i, 7);
        hits[i].e22() = val(e, i, 8);
        hits[i].layer() = i % kLayers;
        binned[i].rank() = kHits - 1 - i;
        binned[i].phi() = val(e, i, 9);
        if (i == 0) {
          hits.nPixel() = e;
          hits.nStrip() = kHits;
          layers.nBinsTotal() = kBins;
          trk.nTracks() = kTracks;
          cand.seeds().hotsPerSeed() = kHotsPerSeed;
        }
      }
      for (int32_t l : cms::alpakatools::uniform_elements(acc, kLayers)) {
        layers[l].binBegin() = l * (kBins / kLayers);
        layers[l].nHits() = e + l;
      }
      for (int32_t b : cms::alpakatools::uniform_elements(acc, kBins)) {
        bins[b].content() = e * 7 + b;
        bins[b].dead() = b % 3 == 0;
      }
      for (int32_t s : cms::alpakatools::uniform_elements(acc, kSeeds)) {
        cand.seeds()[s].nCands() = 1;
        cand.seeds()[s].seedOriginIdx() = e + s;
      }
      for (int32_t h : cms::alpakatools::uniform_elements(acc, kSeeds * kHotsPerSeed)) {
        cand.hots()[h].node().index = h;
      }
    }
  };

  // packers on device: per Matriplex of kNN slots, thread = Matriplex
  class KernelPackProducts {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::mkfitdev::HitSoAConstView hits,
                                  ::mkfitdev::TrackSoAView trk,
                                  ::mkfitdev::CandStoreBlocks::View cand,
                                  uint32_t e) const {
      constexpr int N = kNN;
      for (int32_t g : cms::alpakatools::uniform_elements(acc, (kTracks + N - 1) / N)) {
        MPlexHS<N> msErr;
        MPlexHV<N> msPar;
        for (int n = 0; n < N && g * N + n < kTracks; ++n)
          ::mkfitdev::pack::loadHit<N>(hits, g * N + n, n, msErr, msPar);
        for (int n = 0; n < N && g * N + n < kTracks; ++n) {
          const int r = g * N + n;
          for (int k = 0; k < 6; ++k)
            trk[r].params().v[k] = k < 3 ? msPar.constAt(n, k, 0) : val(e, r, 20 + k);
          float eh[6];
          msErr.copyOut(n, eh);
          for (int k = 0; k < 21; ++k)
            trk[r].errors().v[k] = k < 6 ? eh[k] : val(e, r, 40 + k);
          trk[r].charge() = (r % 2) ? 1 : -1;
          trk[r].chi2() = val(e, r, 70);
          trk[r].label() = r;
        }
      }
    }
  };

  // tracks -> candidate slot 0 of seed s (s < kSeeds); separate kernel: reads rows written by other threads above
  class KernelPackCands {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::mkfitdev::TrackSoAConstView trk,
                                  ::mkfitdev::CandStoreBlocks::View cand) const {
      constexpr int N = kNN;
      for (int32_t g : cms::alpakatools::uniform_elements(acc, (kSeeds + N - 1) / N)) {
        MPlexLS<N> err;
        MPlexLV<N> par;
        MPlexQI<N> chg;
        MPlexQF<N> chi2;
        for (int n = 0; n < N && g * N + n < kSeeds; ++n)
          ::mkfitdev::pack::loadTrack<N>(trk, g * N + n, n, err, par, chg, chi2);
        for (int n = 0; n < N && g * N + n < kSeeds; ++n) {
          const int s = g * N + n;
          ::mkfitdev::pack::storeCandState<N>(err, par, chg, n, cand.slots()[::mkfitdev::candSlotRow(s, 0, 0)].state());
        }
      }
    }
  };

  void fillProductsTest(Queue& queue,
                        EventOfHitsDeviceCollection& eoh,
                        TrackSoADeviceCollection& trk,
                        CandStoreDeviceCollection& cand,
                        uint32_t event) {
    const auto wd = cms::alpakatools::make_workdiv<Acc1D>(4, 128);
    alpaka::exec<Acc1D>(queue, wd, KernelFillProducts{}, eoh.view(), trk.view(), cand.view(), event);
    // kernels on one queue run in order: fill, hits -> tracks, tracks -> candidates
    alpaka::exec<Acc1D>(queue, wd, KernelPackProducts{}, eoh.const_view().hits(), trk.view(), cand.view(), event);
    alpaka::exec<Acc1D>(queue, wd, KernelPackCands{}, trk.const_view(), cand.view());
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev
