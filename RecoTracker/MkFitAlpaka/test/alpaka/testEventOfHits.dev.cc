// Builds the device EventOfHits of MkFitAlpaka (hits::buildEventOfHits) from the hits written by
// testMkFitAlpakaEventOfHitsMkFitCoreRef and compares it with MkFitCore's LayerOfHits: internal order, hit infos with
// the v2p2 extents, bin table, largest extents per layer and the LayerOfHitsAccess range queries.
// Usage: testMkFitAlpakaEventOfHits<Backend> <mkfitcore_ref.bin>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/LayerOfHitsAccess.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/hits/EventOfHitsKernels.h"
#include "RecoTracker/MkFitAlpaka/test/eventOfHitsTestIO.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ::mkfitdev::eohtest;
namespace mkd = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev;

namespace {

  // The floats may differ from MkFitCore's in the last bits on every backend: MkFitCore is built with -Ofast, which
  // lets the compiler reorder float operations that this package keeps in source order, and GPU backends round on
  // their own. A hit within that rounding of a phi bin edge (the wrap at +-PI included) can then land in the next bin,
  // which changes the internal order and the bin table around it. The floats are therefore compared per original hit
  // index, relative to the value or, for the extents, to the layer's largest extent; order and bins with a tolerance.
  constexpr bool kCpu = std::is_same_v<Device, alpaka::DevCpu>;
  constexpr float kRelTol = kCpu ? 1e-6f : 1e-5f;  // relative float tolerance
  constexpr double kRowFraction = 1e-2;            // fraction of hits and of bins allowed in another place

  // the float fields compared, for the largest relative difference of each
  enum Field {
    kPhi,
    kQ,
    kQHalfLength,
    kQbar,
    kPhiHalfExtent,
    kQbarHalfExtent,
    kMaxQHalfLength,
    kMaxPhiHalfExtent,
    kNFields
  };
  const char* const kFieldNames[kNFields] = {"phi",
                                             "q",
                                             "q half-length",
                                             "qbar",
                                             "phi half-extent",
                                             "qbar half-extent",
                                             "layer max q half-length",
                                             "layer max phi half-extent"};

  // |dev - ref| / max(|ref|, scale): 0 for equal values (also both NaN), infinity if only one is NaN or a zero is
  // matched by a non-zero without a scale
  float relDiff(float dev, float ref, float scale) {
    if (std::isnan(dev) || std::isnan(ref))
      return std::isnan(dev) && std::isnan(ref) ? 0.f : std::numeric_limits<float>::infinity();
    if (dev == ref)
      return 0.f;
    const float norm = std::abs(ref) < scale ? scale : std::abs(ref);
    return norm == 0.f ? std::numeric_limits<float>::infinity() : std::abs(dev - ref) / norm;
  }

  struct Counts {
    long rows = 0, orderDiffs = 0, floatDiffs = 0, bins = 0, binDiffs = 0, layerDiffs = 0, queries = 0, queryDiffs = 0;
    float maxRel[kNFields] = {};
    // true if within the tolerance; keeps the largest difference of the field
    bool close(Field field, float dev, float ref, float scale = 0.f) {
      const float rel = relDiff(dev, ref, scale);
      maxRel[field] = maxRel[field] < rel ? rel : maxRel[field];
      return rel <= kRelTol;
    }
  };

  void compare(Reference const& ref, ::mkfitdev::EventOfHitsBlocks::ConstView const& eoh, Counts& c) {
    const auto layers = eoh.layers();
    const auto binned = eoh.binnedHits();
    const auto bins = eoh.bins();
    const uint32_t nPhiBins = ref.header.nPhiBins;
    for (int l = 0; l < ref.header.nLayers; ++l) {
      LayerRec const& lr = ref.layers[l];
      const ::mkfitdev::LayerOfHitsAccess loh{layers, binned, bins, l};
      const bool maxQ = c.close(kMaxQHalfLength, loh.max_hit_q_half_length(), lr.maxQHalfLength);
      const bool maxPhi = c.close(kMaxPhiHalfExtent, loh.max_hit_phi_half_extent(), lr.maxPhiHalfExtent);
      if (loh.nHits() != lr.nHits || loh.qNBins() != lr.nQBins || !maxQ || !maxPhi) {
        ++c.layerDiffs;
        std::printf(
            "layer %d: nHits %u / %u, qNBins %u / %u, max q half-length %.9g / %.9g, max phi half-extent "
            "%.9g / %.9g (device / MkFitCore)\n",
            l,
            loh.nHits(),
            lr.nHits,
            loh.qNBins(),
            lr.nQBins,
            loh.max_hit_q_half_length(),
            lr.maxQHalfLength,
            loh.max_hit_phi_half_extent(),
            lr.maxPhiHalfExtent);
        continue;
      }
      // MkFitCore's row of each original hit index, and the largest qbar extent as its scale
      std::vector<int> refRow(
          lr.nHits == 0 ? 0
                        : 1 + std::max_element(ref.binned[l].begin(),
                                               ref.binned[l].end(),
                                               [](BinnedRec const& a, BinnedRec const& b) { return a.rank < b.rank; })
                                  ->rank,
          -1);
      float qbarScale = 0.f;
      for (uint32_t i = 0; i < lr.nHits; ++i) {
        refRow[ref.binned[l][i].rank] = i;
        qbarScale = qbarScale < ref.binned[l][i].qbarHalfExtent ? ref.binned[l][i].qbarHalfExtent : qbarScale;
      }
      for (uint32_t i = 0; i < lr.nHits; ++i) {
        ++c.rows;
        const uint32_t rank = loh.getOriginalHitIndex(i);
        if (rank != ref.binned[l][i].rank) {
          ++c.orderDiffs;
          if (c.orderDiffs <= 5)
            std::printf("layer %d row %u: hit %u / %u, phi %.9g / %.9g (device / MkFitCore)\n",
                        l,
                        i,
                        rank,
                        ref.binned[l][i].rank,
                        loh.hit_phi(i),
                        ref.binned[l][i].phi);
        }
        if (rank >= refRow.size() || refRow[rank] < 0) {
          ++c.floatDiffs;
          continue;
        }
        BinnedRec const& br = ref.binned[l][refRow[rank]];
        bool ok = c.close(kPhi, loh.hit_phi(i), br.phi);
        ok = c.close(kQ, loh.hit_q(i), br.q) && ok;
        ok = c.close(kQHalfLength, loh.hit_q_half_length(i), br.qHalfLength, lr.maxQHalfLength) && ok;
        ok = c.close(kQbar, loh.hit_qbar(i), br.qbar) && ok;
        ok = c.close(kPhiHalfExtent, loh.hit_phi_half_extent(i), br.phiHalfExtent, lr.maxPhiHalfExtent) && ok;
        ok = c.close(kQbarHalfExtent, loh.hit_qbar_half_extent(i), br.qbarHalfExtent, qbarScale) && ok;
        if (!ok) {
          ++c.floatDiffs;
          if (c.floatDiffs <= 5)
            std::printf(
                "layer %d hit %u: phi %.9g / %.9g, q %.9g / %.9g, q half-length %.9g / %.9g, qbar %.9g / "
                "%.9g, phi half-extent %.9g / %.9g, qbar half-extent %.9g / %.9g (device / MkFitCore)\n",
                l,
                i,
                loh.hit_phi(i),
                br.phi,
                loh.hit_q(i),
                br.q,
                loh.hit_q_half_length(i),
                br.qHalfLength,
                loh.hit_qbar(i),
                br.qbar,
                loh.hit_phi_half_extent(i),
                br.phiHalfExtent,
                loh.hit_qbar_half_extent(i),
                br.qbarHalfExtent);
        }
      }
      for (uint32_t qi = 0; qi < lr.nQBins; ++qi)
        for (uint32_t pi = 0; pi < nPhiBins; ++pi) {
          BinRec const& b = ref.bins[l][qi * nPhiBins + pi];
          const uint32_t content = loh.binContent(pi, qi);
          ++c.bins;
          if (::mkfitdev::binFirst(content) != b.first || ::mkfitdev::binCount(content) != b.count)
            ++c.binDiffs;
        }
    }
    // the queries need only the layer constants, which the device does not compute: exact on every backend
    for (QueryRec const& q : ref.queries) {
      const ::mkfitdev::LayerOfHitsAccess loh{layers, binned, bins, q.layer};
      const auto phiRange = loh.phiRangeBins(q.phiLo, q.phiHi);
      const auto qRange = loh.qRangeBins(q.qLo, q.qHi);
      ++c.queries;
      if (phiRange.begin != q.phiBegin || phiRange.end != q.phiEnd || qRange.begin != q.qBegin ||
          qRange.end != q.qEnd) {
        ++c.queryDiffs;
        if (c.queryDiffs <= 5)
          std::printf("layer %d: phi [%.9g, %.9g] -> [%u, %u) / [%u, %u), q [%.9g, %.9g] -> [%u, %u) / [%u, %u)\n",
                      q.layer,
                      q.phiLo,
                      q.phiHi,
                      phiRange.begin,
                      phiRange.end,
                      q.phiBegin,
                      q.phiEnd,
                      q.qLo,
                      q.qHi,
                      qRange.begin,
                      qRange.end,
                      q.qBegin,
                      q.qEnd);
      }
    }
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <mkfitcore_ref.bin>\n", argv[0]);
    return 2;
  }
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::printf("no %s device: skipped\n", EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE));
    return 0;
  }
  Reference ref;
  std::FILE* file = std::fopen(argv[1], "rb");
  if (!file || !readReference(file, ref)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 2;
  }
  std::fclose(file);
  if (ref.header.nPhiBins != int32_t(::mkfitdev::kNPhiBins)) {
    std::printf("FAIL: MkFitCore has %d phi bins, the device %u\n", ref.header.nPhiBins, ::mkfitdev::kNPhiBins);
    return 3;
  }
  Queue queue(devices[0]);

  // inputs as MkFitAlpakaEventOfHitsProducer leaves them: the hits and the static part of the layer table
  const uint32_t nHits = ref.hits.size(), nLayers = ref.header.nLayers;
  PortableHostCollection<::mkfitdev::HitSoA> hitsHost(queue, nHits);
  auto hv = hitsHost.view();
  for (uint32_t i = 0; i < nHits; ++i) {
    HitRec const& h = ref.hits[i];
    auto row = hv[i];
    row.x() = h.x;
    row.y() = h.y;
    row.z() = h.z;
    row.e00() = h.e00;
    row.e10() = h.e10;
    row.e11() = h.e11;
    row.e20() = h.e20;
    row.e21() = h.e21;
    row.e22() = h.e22;
    row.packed() = 0;
    row.layer() = h.layer;
  }
  hv.nPixel() = ref.header.nPixel;
  hv.nStrip() = ref.header.nStrip;
  PortableHostCollection<::mkfitdev::LayerSoA> layersHost(queue, nLayers);
  auto lv = layersHost.view();
  uint32_t binBegin = 0;
  for (uint32_t l = 0; l < nLayers; ++l) {
    LayerRec const& lr = ref.layers[l];
    auto row = lv[l];
    row.qRMin() = lr.qRMin;
    row.qRMax() = lr.qRMax;
    row.qMFac() = lr.qMFac;
    row.qNFac() = lr.qNFac;
    row.qMLbhp() = lr.qMLbhp;
    row.qNLbhp() = lr.qNLbhp;
    row.qLastMBin() = lr.qLastMBin;
    row.qLastNBin() = lr.qLastNBin;
    row.nQBins() = lr.nQBins;
    row.isBarrel() = lr.isBarrel;
    row.isPixel() = lr.isPixel;
    row.binBegin() = binBegin;
    row.hitBase() = lr.isPixel ? 0 : ref.header.nPixel;
    row.hitBegin() = 0;
    row.nHits() = 0;
    // outputs of the build: a value it must overwrite
    row.maxQHalfLength() = -1.f;
    row.maxPhiHalfExtent() = -1.f;
    binBegin += lr.nQBins * ::mkfitdev::kNPhiBins;
  }
  lv.phiRMin() = ref.header.phiRMin;
  lv.phiMFac() = ref.header.phiMFac;
  lv.phiNFac() = ref.header.phiNFac;
  lv.nBinsTotal() = binBegin;
  lv.nOverflowFirst() = 0;
  lv.nOverflowCount() = 0;

  const ::mkfitdev::EventOfHitsSizes sizes{{int32_t(nHits), int32_t(nLayers), int32_t(nHits), int32_t(binBegin)}};
  mkd::EventOfHitsDeviceCollection product(queue, sizes);
  auto pv = product.view();
  const auto device = alpaka::getDev(queue);
  alpaka::memcpy(queue,
                 cms::alpakatools::make_device_view(device,
                                                    reinterpret_cast<std::byte*>(pv.hits().metadata().addressOf_x()),
                                                    ::mkfitdev::HitSoA::computeDataSize(nHits)),
                 hitsHost.buffer());
  alpaka::memcpy(
      queue,
      cms::alpakatools::make_device_view(device,
                                         reinterpret_cast<std::byte*>(pv.layers().metadata().addressOf_qRMin()),
                                         ::mkfitdev::LayerSoA::computeDataSize(nLayers)),
      layersHost.buffer());
  if (!mkd::hits::buildEventOfHits(
          queue, mkd::EventOfHitsViews{pv.hits(), pv.layers(), pv.binnedHits(), pv.bins()}, {})) {
    std::printf("FAIL: the device refused to build the event\n");
    return 3;
  }

  Counts c;
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  alpaka::wait(queue);
  compare(ref, product.const_view(), c);
#else
  auto hostCopy = product.copyToHost(queue);
  alpaka::wait(queue);
  compare(ref, hostCopy.const_view(), c);
#endif

  const bool pass = c.layerDiffs == 0 && c.queryDiffs == 0 && c.floatDiffs == 0 &&
                    c.orderDiffs <= kRowFraction * c.rows && c.binDiffs <= kRowFraction * c.bins;
  for (int field = 0; field < kNFields; ++field)
    std::printf("  largest relative difference, %-26s %.3g\n", kFieldNames[field], c.maxRel[field]);
  std::printf(
      "%s %s: layers differing %ld; hits %ld, in another order %ld, floats beyond %g %ld; bins %ld, differing "
      "%ld; range queries %ld, differing %ld\n",
      pass ? "PASS" : "FAIL",
      EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE),
      c.layerDiffs,
      c.rows,
      c.orderDiffs,
      kRelTol,
      c.floatDiffs,
      c.bins,
      c.binDiffs,
      c.queries,
      c.queryDiffs);
  return pass ? 0 : 3;
}
