#include <cmath>
#include <cstring>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpakaMath/interface/math/vdtMath.h"

#include "MkSeederDeviceHits.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  namespace {
    using HitsOut = mkseeder::MutableTableView<::mkseeder::MkSeederHitsSoA>;
    using OTHitsOut = mkseeder::MutableTableView<::mkseeder::MkSeederOTHitsSoA>;
    using LayersOut = mkseeder::MutableTableView<::mkseeder::MkSeederLayersSoA>;
    using PSStripOut = mkseeder::MutableTableView<::mkseeder::MkSeederPSStripHitsSoA>;
    using Layer = ::mkseeder::MkSeederDeviceHitsLayer;
    using DetSet = ::mkseeder::MkSeederDeviceHitsDetSet;

    // the extents: per layer qbar low, qbar high, q low, q high as order-preserving unsigned keys of the floats
    constexpr uint32_t kNoMin = 0xffffffffu, kNoMax = 0u;
    ALPAKA_FN_HOST_ACC inline uint32_t orderedKey(float f) {
      const uint32_t u = __builtin_bit_cast(uint32_t, f);
      return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    ALPAKA_FN_HOST_ACC inline float fromOrderedKey(uint32_t k) {
      return __builtin_bit_cast(float, (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
    }

    // the sort key's fine bins (MkSeederTableBuilder.cc: kFineBits 16, kQFineShift 8)
    constexpr unsigned int kFineBits = 16, kQFineShift = 8;

    struct KernelInitExtents {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    uint32_t *extents,
                                    uint32_t nLayers,
                                    uint32_t *mismatches,
                                    HitsOut out,
                                    uint32_t nTot) const {
        if (cms::alpakatools::once_per_grid(acc))
          *mismatches = 0, out.nHits() = nTot;
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, 4 * nLayers))
          extents[i] = (i & 1) ? kNoMax : kNoMin;
      }
    };

    // a thread per HitSoA row: its detset (binary search), its cluster-order row, its values (as the host fill from
    // mkfit::Hit) and its beam-relative coordinates and sort key (computeHitBinning; vdt's atan2, the device's cos and
    // sin: the last bit of phi, x and y may differ from the host's)
    struct KernelFillHits {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    ::mkfitdev::HitSoA::ConstView eoh,
                                    uint32_t nRows,
                                    uint32_t nPixelRows,
                                    const Layer *layers,
                                    uint32_t nLayers,
                                    const DetSet *detSets,
                                    uint32_t nDetSets,
                                    ::mkseeder::MkSeederDeviceHitsBeamSpot bs,
                                    HitsOut out,
                                    OTHitsOut outOT,
                                    PSStripOut outPSStrip,
                                    uint32_t *extents,
                                    uint32_t *mismatches) const {
        for (uint32_t r : cms::alpakatools::uniform_elements(acc, nRows)) {
          const int devLayer = eoh.layer(r);
          // the last detset with firstRow <= r
          uint32_t lo = 0, hi = nDetSets;
          while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (detSets[mid].firstRow <= r)
              lo = mid + 1;
            else
              hi = mid;
          }
          const bool inSet = lo > 0 && r < detSets[lo - 1].firstRow + detSets[lo - 1].n;
          if (!inSet) {
            if (devLayer >= 0 && uint32_t(devLayer) < nLayers && layers[devLayer].binned)
              alpaka::atomicAdd(acc, mismatches, 1u, alpaka::hierarchy::Blocks{});
            continue;
          }
          const DetSet ds = detSets[lo - 1];
          if (devLayer != ds.layer) {
            alpaka::atomicAdd(acc, mismatches, 1u, alpaka::hierarchy::Blocks{});
            continue;
          }
          const uint32_t k = r - ds.firstRow, d = ds.dest + k;
          const float gx = eoh.x(r), gy = eoh.y(r), gz = eoh.z(r);
          const uint32_t packed = eoh.packed(r);
          if (ds.psStrip >= 0) {
            // an S hit of OT1-3 for the PS mark (the host fill's psStripHits)
            if (::mkfitdev::hitpack::detIDinLayer(packed) != ds.psModule) {
              alpaka::atomicAdd(acc, mismatches, 1u, alpaka::hierarchy::Blocks{});
              continue;
            }
            const uint32_t q = uint32_t(ds.psStrip) + k;
            outPSStrip.x(q) = gx, outPSStrip.y(q) = gy, outPSStrip.z(q) = gz;
            outPSStrip.size(q) = uint16_t(::mkfitdev::hitpack::spanRows(packed));
          }
          const Layer L = layers[devLayer];
          if (!L.binned)
            continue;
          out.x(d) = gx, out.y(d) = gy, out.z(d) = gz;
          out.clusterRows(d) = int32_t(::mkfitdev::hitpack::spanRows(packed));
          out.clusterCols(d) = int32_t(::mkfitdev::hitpack::spanCols(packed));
          out.detIdInLayer(d) = ::mkfitdev::hitpack::detIDinLayer(packed);
          out.clusterIndex(d) = ::mkfitdev::originalIndex(r, nPixelRows);
          // mkfit::Hit::errArray(): xx, xy, yy, xz, yz, zz
          const float e00 = eoh.e00(r), e10 = eoh.e10(r), e11 = eoh.e11(r), e20 = eoh.e20(r), e21 = eoh.e21(r),
                      e22 = eoh.e22(r);
          auto cov = out.covariance(d);
          cov(0) = e00, cov(1) = e10, cov(2) = e11, cov(3) = e20, cov(4) = e21, cov(5) = e22;
          if (ds.otHit >= 0) {
            // the extension's order: xx, yy, xy, xz, yz, zz
            auto oc = outOT.covariance(uint32_t(ds.otHit) + k);
            oc(0) = e00, oc(1) = e11, oc(2) = e10, oc(3) = e20, oc(4) = e21, oc(5) = e22;
          }
          // computeHitBinning()
          const float dz = gz - bs.z;
          const float x = gx - (bs.x + bs.dxdz * dz), y = gy - (bs.y + bs.dydz * dz);
          const float phi = ::mkfitdev::vdt::fast_atan2f(y, x);
          const float rr = alpaka::math::sqrt(acc, x * x + y * y);
          using Bin = unsigned short;
          const unsigned int b1 =
              Bin(Bin(alpaka::math::floor(acc, (phi - L.phiAxisMin) * L.phiFineScale)) & Bin((1 << kFineBits) - 1));
          const float qv = L.isDisc ? rr : gz;
          const unsigned int b2 = qv <= L.qAxisMin ? Bin(0)
                                  : (qv >= L.qFineLastBinHighEdge)
                                      ? Bin(L.qLastFineBin)
                                      : Bin(alpaka::math::floor(acc, (qv - L.qAxisMin) * L.qFineScale));
          const uint32_t key = ((b2 << kFineBits) | b1) & ~(((1u << kQFineShift) - 1) << kFineBits);
          out.binIndex(d) = uint32_t(devLayer) << 24 | (key >> 24) << 16 | (key & 0xffffu);
          out.phi(d) = phi, out.r(d) = rr;
          out.invR(d) = 1.0f / rr;
          out.xBeam(d) = rr * alpaka::math::cos(acc, phi);
          out.yBeam(d) = rr * alpaka::math::sin(acc, phi);
          // the layer's extent: qbar (r of a barrel layer, z of a disc), q the other
          const float u = L.isDisc ? gz : rr, v = L.isDisc ? rr : gz;
          uint32_t *e = extents + 4 * devLayer;
          alpaka::atomicMin(acc, e + 0, orderedKey(u), alpaka::hierarchy::Blocks{});
          alpaka::atomicMax(acc, e + 1, orderedKey(u), alpaka::hierarchy::Blocks{});
          alpaka::atomicMin(acc, e + 2, orderedKey(v), alpaka::hierarchy::Blocks{});
          alpaka::atomicMax(acc, e + 3, orderedKey(v), alpaka::hierarchy::Blocks{});
        }
      }
    };

    // the binned layers' extents: the nominal ones widened to hold every hit (as the host's in double)
    struct KernelSetExtents {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    const Layer *layers,
                                    uint32_t nLayers,
                                    const uint32_t *extents,
                                    LayersOut eventLayers) const {
        for (uint32_t l : cms::alpakatools::uniform_elements(acc, nLayers)) {
          const Layer L = layers[l];
          if (!L.binned)
            continue;
          const uint32_t *e = extents + 4 * l;
          double ext[4] = {L.qbarLowNominal, L.qbarHighNominal, L.qLowNominal, L.qHighNominal};
          if (e[0] != kNoMin) {
            ext[0] = alpaka::math::min(acc, ext[0], double(fromOrderedKey(e[0])));
            ext[1] = alpaka::math::max(acc, ext[1], double(fromOrderedKey(e[1])));
            ext[2] = alpaka::math::min(acc, ext[2], double(fromOrderedKey(e[2])));
            ext[3] = alpaka::math::max(acc, ext[3], double(fromOrderedKey(e[3])));
          }
          eventLayers.qbarLo(l) = ext[0], eventLayers.qbarHi(l) = ext[1], eventLayers.qLo(l) = ext[2],
          eventLayers.qHi(l) = ext[3];
        }
      }
    };
  }  // namespace

  void fillHitsFromEventOfHits(Queue &queue,
                               ::mkfitdev::HitSoA::ConstView eoh,
                               uint32_t nPixelRows,
                               std::vector<::mkseeder::MkSeederDeviceHitsLayer> const &layers,
                               std::vector<::mkseeder::MkSeederDeviceHitsDetSet> const &detSets,
                               ::mkseeder::MkSeederDeviceHitsBeamSpot const &beamSpot,
                               mkseeder::MutableTableView<::mkseeder::MkSeederHitsSoA> out,
                               mkseeder::MutableTableView<::mkseeder::MkSeederOTHitsSoA> outOT,
                               mkseeder::MutableTableView<::mkseeder::MkSeederPSStripHitsSoA> outPSStrip,
                               uint32_t nTot,
                               mkseeder::MutableTableView<::mkseeder::MkSeederLayersSoA> eventLayers,
                               uint32_t *mismatches) {
    using namespace cms::alpakatools;
    const uint32_t nLayers = layers.size(), nDetSets = detSets.size();
    const uint32_t nRows = eoh.metadata().size();
    // the tables to the device (one host block: the layers, then the detsets)
    const size_t layerBytes = nLayers * sizeof(Layer), detSetBytes = nDetSets * sizeof(DetSet);
    const size_t layerWords = (layerBytes + sizeof(double) - 1) / sizeof(double);
    const size_t words = layerWords + (detSetBytes + sizeof(double) - 1) / sizeof(double) + 1;
    auto tablesHost = make_host_buffer<double[]>(queue, words);
    std::memcpy(tablesHost.data(), layers.data(), layerBytes);
    if (nDetSets > 0)
      std::memcpy(tablesHost.data() + layerWords, detSets.data(), detSetBytes);
    auto tables = make_device_buffer<double[]>(queue, words);
    alpaka::memcpy(queue, tables, tablesHost);
    const auto *dLayers = reinterpret_cast<const Layer *>(tables.data());
    const auto *dDetSets = reinterpret_cast<const DetSet *>(tables.data() + layerWords);
    auto extents = make_device_buffer<uint32_t[]>(queue, 4 * nLayers);
    constexpr uint32_t kThreads = 128;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(4 * nLayers, kThreads), kThreads),
                        KernelInitExtents{},
                        extents.data(),
                        nLayers,
                        mismatches,
                        out,
                        nTot);
    if (nRows > 0)
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(divide_up_by(nRows, kThreads), kThreads),
                          KernelFillHits{},
                          eoh,
                          nRows,
                          nPixelRows,
                          dLayers,
                          nLayers,
                          dDetSets,
                          nDetSets,
                          beamSpot,
                          out,
                          outOT,
                          outPSStrip,
                          extents.data(),
                          mismatches);
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nLayers, kThreads), kThreads),
                        KernelSetExtents{},
                        dLayers,
                        nLayers,
                        extents.data(),
                        eventLayers);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder
