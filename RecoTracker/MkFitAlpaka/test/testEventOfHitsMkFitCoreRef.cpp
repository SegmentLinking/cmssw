// MkFitCore reference for the EventOfHits test: synthetic hits on a pixel barrel, a pixel endcap, a strip barrel and a
// strip endcap layer, registered into mkfit::LayerOfHits as MkFitEventOfHitsProducer does (cmssw_LoadHits_Begin,
// registerHit in wrapper order, endRegistrationOfHits). Writes the inputs and what LayerOfHits holds (internal order,
// hit infos with the v2p2 extents, bin table, largest extents, range queries) for testMkFitAlpakaEventOfHits<Backend>.
// The hits include what the extents treat apart: r = 0, a transverse error that is not positive definite (negative phi
// or r variance, clamped), duplicate keys, q outside the axis, phi at the wrap; every layer gets enough hits for the
// radix sort (stable), whose order the device reproduces.
// Usage: testMkFitAlpakaEventOfHitsMkFitCoreRef <out.bin> [seed=4242]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "eventOfHitsTestIO.h"

using namespace mkfitdev::eohtest;

namespace {

  constexpr int kHitsPerLayer = 600;
  constexpr int kUnregisteredPerWrapper = 25;
  constexpr int kQueriesPerLayer = 200;

  struct LayerSpec {
    mkfit::LayerInfo::LayerType_e type;
    bool pixel;
    float rin, rout, zmin, zmax, qBin;
  };
  // layer order as Phase-2: pixel first, barrel before endcap
  const LayerSpec kLayers[] = {{mkfit::LayerInfo::Barrel, true, 2.8f, 3.2f, -20.f, 20.f, 1.f},
                               {mkfit::LayerInfo::EndCapPos, true, 3.f, 16.f, 30.f, 31.f, 0.5f},
                               {mkfit::LayerInfo::Barrel, false, 60.f, 62.f, -110.f, 110.f, 4.f},
                               {mkfit::LayerInfo::EndCapPos, false, 25.f, 110.f, 130.f, 131.f, 5.6f}};
  constexpr int kNLayers = sizeof(kLayers) / sizeof(kLayers[0]);

  // A hit of the layer: position on the layer's surface (q partly outside the axis range), error from a random
  // lower-triangular factor, a few with an x-y correlation beyond the variances or at r = 0.
  mkfit::Hit makeHit(LayerSpec const& spec, std::mt19937& rng, int k) {
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    const float phi =
        k % 97 == 0 ? (k % 2 ? mkfit::Const::PI : -mkfit::Const::PI) : mkfit::Const::PI * (2 * uni(rng) - 1);
    const bool barrel = spec.type == mkfit::LayerInfo::Barrel;
    const float qLo = barrel ? spec.zmin : spec.rin, qHi = barrel ? spec.zmax : spec.rout;
    const float q = qLo - 0.05f * (qHi - qLo) + 1.1f * (qHi - qLo) * uni(rng);
    float r = barrel ? spec.rin + (spec.rout - spec.rin) * uni(rng) : q;
    const float z = barrel ? q : spec.zmin + (spec.zmax - spec.zmin) * uni(rng);
    if (k % 211 == 5)
      r = 0.f;
    const float scale = spec.pixel ? 2e-3f : 3e-2f;
    float a[3][3] = {};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j <= i; ++j)
        a[i][j] = scale * (i == j ? 0.2f + uni(rng) : uni(rng) - 0.5f);
    mkfit::SMatrixSym33 err;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j <= i; ++j) {
        float sum = 0.f;
        for (int m = 0; m <= j; ++m)
          sum += a[i][m] * a[j][m];
        err(i, j) = sum;
      }
    if (k % 13 == 7)
      err(1, 0) = (k % 2 ? 3.f : -3.f) * std::sqrt(err(0, 0) * err(1, 1));
    return mkfit::Hit(mkfit::SVector3(r * std::cos(phi), r * std::sin(phi), z), err);
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <out.bin> [seed]\n", argv[0]);
    return 2;
  }
  std::mt19937 rng(argc > 2 ? std::atoi(argv[2]) : 4242);

  mkfit::TrackerInfo ti;
  ti.reserve_layers(kNLayers, 0, 0);
  for (int l = 0; l < kNLayers; ++l) {
    mkfit::LayerInfo& li = ti.new_barrel_layer();
    li.set_layer_type(kLayers[l].type);
    li.set_is_pixel(kLayers[l].pixel);
    li.set_limits(kLayers[l].rin, kLayers[l].rout, kLayers[l].zmin, kLayers[l].zmax);
    li.set_q_bin(kLayers[l].qBin);
  }

  // the two hit wrappers; each hit goes to a random layer of its kind (rows interleaved as in an event), some to none
  mkfit::HitVec wrapperHits[2];
  std::vector<int> wrapperLayer[2];
  std::vector<int> nPerLayer(kNLayers, 0);
  std::uniform_int_distribution<int> pick(0, kNLayers - 1);
  int k = 0;
  for (int done = 0; done < kNLayers * kHitsPerLayer;) {
    const int l = pick(rng);
    if (nPerLayer[l] == kHitsPerLayer)
      continue;
    const int w = kLayers[l].pixel ? 0 : 1;
    // a duplicate of the previous hit of the wrapper (same key) now and then
    if (k % 31 == 30 && !wrapperHits[w].empty() && wrapperLayer[w].back() == l)
      wrapperHits[w].push_back(wrapperHits[w].back());
    else
      wrapperHits[w].push_back(makeHit(kLayers[l], rng, k));
    wrapperLayer[w].push_back(l);
    ++nPerLayer[l];
    ++done;
    ++k;
  }
  for (int w = 0; w < 2; ++w)
    for (int u = 0; u < kUnregisteredPerWrapper; ++u) {
      std::uniform_int_distribution<size_t> at(0, wrapperHits[w].size());
      const size_t pos = at(rng);
      wrapperHits[w].insert(wrapperHits[w].begin() + pos, makeHit(kLayers[w == 0 ? 0 : 2], rng, k++));
      wrapperLayer[w].insert(wrapperLayer[w].begin() + pos, -1);
    }

  // MkFitEventOfHitsProducer: cmssw_LoadHits_Begin, registerHit per wrapper in index order, cmssw_LoadHits_End
  mkfit::EventOfHits eoh(ti);
  eoh.reset();
  for (int l = 0; l < kNLayers; ++l)
    eoh[l].beginRegistrationOfHits(wrapperHits[eoh[l].is_pixel() ? 0 : 1]);
  for (int w = 0; w < 2; ++w)
    for (size_t i = 0; i < wrapperHits[w].size(); ++i)
      if (wrapperLayer[w][i] >= 0)
        eoh[wrapperLayer[w][i]].registerHit(i);
  for (int l = 0; l < kNLayers; ++l)
    eoh[l].endRegistrationOfHits(false);

  Reference event;
  const mkfit::LayerOfHits::axis_phi_t aphi(-mkfit::Const::PI, mkfit::Const::PI);
  event.header = Header{kNLayers,
                        int32_t(wrapperHits[0].size()),
                        int32_t(wrapperHits[1].size()),
                        kNLayers * kQueriesPerLayer,
                        int32_t(aphi.size_of_N()),
                        aphi.m_R_min,
                        aphi.m_M_fac,
                        aphi.m_N_fac};
  for (int w = 0; w < 2; ++w)
    for (size_t i = 0; i < wrapperHits[w].size(); ++i) {
      const mkfit::Hit& h = wrapperHits[w][i];
      const float* e = h.errArray();
      event.hits.push_back(HitRec{h.x(), h.y(), h.z(), e[0], e[1], e[2], e[3], e[4], e[5], wrapperLayer[w][i]});
    }
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  for (int l = 0; l < kNLayers; ++l) {
    const mkfit::LayerOfHits& loh = eoh[l];
    const mkfit::LayerOfHits::Initializator init(ti.layer(l));
    const mkfit::LayerOfHits::axis_eta_t aq(init.m_qmin, init.m_qmax, init.m_nq);
    event.layers.push_back(LayerRec{aq.m_R_min,
                                    aq.m_R_max,
                                    aq.m_M_fac,
                                    aq.m_N_fac,
                                    aq.m_M_lbhp,
                                    aq.m_N_lbhp,
                                    aq.m_last_M_bin,
                                    aq.m_last_N_bin,
                                    loh.qNBins(),
                                    loh.is_barrel(),
                                    loh.is_pixel(),
                                    loh.nHits(),
                                    loh.max_hit_q_half_length(),
                                    loh.max_hit_phi_half_extent()});
    std::vector<BinnedRec> binned;
    for (unsigned int i = 0; i < loh.nHits(); ++i) {
      const auto& hi = loh.hit_info(i);
      binned.push_back(BinnedRec{
          loh.getOriginalHitIndex(i), hi.phi, hi.q, hi.q_half_length, hi.qbar, hi.phi_half_extent, hi.qbar_half_extent});
    }
    event.binned.push_back(binned);
    std::vector<BinRec> bins;
    for (unsigned int qi = 0; qi < loh.qNBins(); ++qi)
      for (unsigned int pi = 0; pi < aphi.size_of_N(); ++pi) {
        const auto content = loh.phiQBinContent(pi, qi);
        bins.push_back(BinRec{content.first, content.count});
      }
    event.bins.push_back(bins);
    // ranges inside and beyond the axes, phi ranges across the wrap
    for (int iq = 0; iq < kQueriesPerLayer; ++iq) {
      const float phiLo = 1.2f * mkfit::Const::PI * (2 * uni(rng) - 1), phiHi = phiLo + 0.5f * uni(rng);
      const float qSpan = aq.m_R_max - aq.m_R_min;
      const float qLo = aq.m_R_min + qSpan * (1.2f * uni(rng) - 0.1f), qHi = qLo + 0.2f * qSpan * uni(rng);
      const auto phiRange = loh.phiRangeBins(phiLo, phiHi);
      const auto qRange = loh.qRangeBins(qLo, qHi);
      event.queries.push_back(
          QueryRec{l, phiLo, phiHi, qLo, qHi, phiRange.begin, phiRange.end, qRange.begin, qRange.end});
    }
  }

  std::FILE* file = std::fopen(argv[1], "wb");
  if (!file) {
    std::fprintf(stderr, "cannot write %s\n", argv[1]);
    return 2;
  }
  writeReference(file, event);
  std::fclose(file);
  std::printf("testMkFitAlpakaEventOfHitsMkFitCoreRef: %d layers, %d pixel + %d strip hits, %d queries -> %s\n",
              event.header.nLayers,
              event.header.nPixel,
              event.header.nStrip,
              event.header.nQueries,
              argv[1]);
  return 0;
}
