#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederBinKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederBinKernels_h

// The hits' binning on the device (MkSeederProducer on a GPU). The host writes every hit's values in the layers'
// cluster order (MkSeederUnbinnedHitsSoA), with the key mkFit layer id << 24 | q N-bin << 16 | phi M-bin per hit in
// binIndex. A stable LSD radix sort of the keys (4 digits) gives the bin order (a stable sort by the key within each
// layer, the layers one after the other in increasing id, as mkFit's SeedLayerOfHits::fill() orders them), and one
// scatter writes the bin-order hits (MkSeederEventHitsSoA) with each hit's N-bin, the inverse permutation
// (binOrderPosition) and the clusters' hit positions.

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederDeviceTypes.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::bins {

  class KernelFillBinKeys {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, HitsView hits, uint32_t n, uint64_t *hi, uint64_t *lo, uint32_t *idx) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, n)) {
        hi[j] = 0;
        lo[j] = hits.binIndex(j);
        idx[j] = j;
      }
    }
  };

  // A pixel hit as a merged-cluster candidate (MkSeederConfig mrg_*): its cluster's charge per path length and its
  // sizes against those of one particle crossing the module along the line from the beam spot (the beam spot's z;
  // the expected size along the local axis a is thickness |d_a / d_z| / pitch_a in the module frame). With
  // mrg_err > 0 a tagged hit's covariance (mkfit::Hit::errArray(), global xx, xy, yy, xz, yz, zz) gets
  // (mrg_err pitch_a max(size excess, 1) / sqrt(12))^2 along each local axis a = x, y.
  ALPAKA_FN_HOST_ACC inline uint8_t mergedCandidate(
      const HitTagInputs &T, uint32_t m, float gx, float gy, float gz, float charge, int sizeX, int sizeY, float *cov) {
    const auto row = T.modules[m];
    const float t = row.thickness(), px = row.pitchX(), py = row.pitchY();
    if (!row.isPixel() || !(t > 0.f) || !(px > 0.f) || !(py > 0.f))
      return 0;
    const float d[3] = {gx - T.beamX, gy - T.beamY, gz - T.beamZ};
    float l[3];
    for (int i = 0; i < 3; ++i)
      l[i] = row.rotation()(i, 0) * d[0] + row.rotation()(i, 1) * d[1] + row.rotation()(i, 2) * d[2];
    const float dn = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (!(std::abs(l[2]) > 1e-6f * dn))
      return 0;
    const float esx = t * std::abs(l[0] / l[2]) / px, esy = t * std::abs(l[1] / l[2]) / py;
    const float path = t * dn / std::abs(l[2]);
    const float dsx = sizeX - esx, dsy = sizeY - esy;
    const uint8_t tag = (charge >= T.mergedChargeCut * T.mergedChargeMpv * path) | (dsx >= T.mergedSizeXCut) << 1 |
                        (dsy >= T.mergedSizeYCut) << 2;
    if (tag && T.mergedErrorScale > 0.f) {
      constexpr float kInvSqrt12 = 0.28867513f;
      const float sx = T.mergedErrorScale * px * std::max(dsx, 1.f) * kInvSqrt12,
                  sy = T.mergedErrorScale * py * std::max(dsy, 1.f) * kInvSqrt12;
      const float sx2 = sx * sx, sy2 = sy * sy;
      auto add = [&](int k, int i, int j) {
        cov[k] += sx2 * row.rotation()(0, i) * row.rotation()(0, j) + sy2 * row.rotation()(1, i) * row.rotation()(1, j);
      };
      add(0, 0, 0), add(1, 0, 1), add(2, 1, 1), add(3, 0, 2), add(4, 1, 2), add(5, 2, 2);
    }
    return tag;
  }

  // one thread per bin-order position p: the hit j = idx[p] of the cluster order; with the hits' tags (HitTagInputs)
  class KernelScatterBinnedHits {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  HitsView in,
                                  OTHitsView inOTHits,
                                  MutableTableView<::mkseeder::MkSeederHitsSoA> out,
                                  MutableTableView<::mkseeder::MkSeederClusterOrderSoA> clusterOrder,
                                  MutableTableView<::mkseeder::MkSeederOTHitsSoA> outOTHits,
                                  EventLayersView layers,
                                  LayerGeometryView layerGeometry,
                                  MutableTableView<::mkseeder::MkSeederPixelClustersSoA> pixelClusters,
                                  MutableTableView<::mkseeder::MkSeederOTClustersSoA> otClusters,
                                  const uint64_t *lo,
                                  const uint32_t *idx,
                                  uint32_t n,
                                  HitTagInputs tags) const {
      if (cms::alpakatools::once_per_grid(acc))
        out.nHits() = n;
      for (uint32_t p : cms::alpakatools::uniform_elements(acc, n)) {
        const uint32_t j = idx[p];
        const uint32_t key = uint32_t(lo[p]);
        const int layer = key >> 24;
        out.phi(p) = in.phi(j), out.z(p) = in.z(j), out.r(p) = in.r(j), out.invR(p) = in.invR(j);
        out.xBeam(p) = in.xBeam(j), out.yBeam(p) = in.yBeam(j), out.clusterIndex(p) = in.clusterIndex(j);
        // the bin: q N-bin * n_phi_bins + phi N-bin (the phi M-bin >> 5: 16 - 11 bits)
        out.binIndex(p) = ((key >> 16) & 0xffu) * uint32_t(::mkseeder::kNPhiBins) + ((key & 0xffffu) >> (16 - 11));
        clusterOrder.binOrderPosition(j) = p;
        if (layerGeometry[layer].isPixel())
          pixelClusters.hitPosition(in.clusterIndex(j)) = j;
        else
          otClusters.hitPosition(in.clusterIndex(j)) = j;
        out.x(p) = in.x(j), out.y(p) = in.y(j), out.clusterRows(p) = in.clusterRows(j);
        out.clusterCols(p) = in.clusterCols(j), out.detIdInLayer(p) = in.detIdInLayer(j);
        float cov[6];
        for (int c = 0; c < 6; ++c)
          cov[c] = in.covariance(j)(c);
        uint8_t tag = 0;
        if (tags.mergedTag > 0 && layerGeometry[layer].isPixel() && in.clusterIndex(j) < tags.nPixelClusters) {
          const uint32_t o = in.clusterIndex(j);
          tag = mergedCandidate(tags,
                                layerGeometry[layer].firstModule() + in.detIdInLayer(j),
                                in.x(j),
                                in.y(j),
                                in.z(j),
                                pixelClusters.charge(o),
                                pixelClusters.sizeX(o),
                                pixelClusters.sizeY(o),
                                cov);
        }
        for (int c = 0; c < 6; ++c)
          out.covariance(p)(c) = cov[c];
        out.mergedTag(p) = tag;
        if (const int firstOTHit = layers[layer].firstOTHit(); firstOTHit >= 0) {
          const uint32_t firstHit = layers[layer].firstHit();
          const uint32_t cp = uint32_t(firstOTHit) + (p - firstHit), cj = uint32_t(firstOTHit) + (j - firstHit);
          for (int c = 0; c < 6; ++c)
            outOTHits.covariance(cp)(c) = inOTHits.covariance(cj)(c);
          // the PS-module confirmation of the P hit by the S hits of its partner module (zeros without ps_mark)
          ::mkseeder::PSMarkOut mark;
          const auto mod = tags.modules[layerGeometry[layer].firstModule() + in.detIdInLayer(j)];
          if (tags.psMark > 0 && mod.psPartner() >= 0) {
            float f[9];
            for (int a = 0; a < 3; ++a)
              for (int d = 0; d < 3; ++d)
                f[3 * a + d] = mod.psFrame()(a, d);
            const uint32_t s0 = tags.psFirstStart[(layer - 4) / 2] + uint32_t(mod.psPartner());
            const auto sh = tags.psStripHits;
            mark = ::mkseeder::markPSHit(f,
                                         tags.psParameters,
                                         tags.beamX,
                                         tags.beamY,
                                         tags.beamZ,
                                         tags.bz,
                                         in.x(j),
                                         in.y(j),
                                         in.z(j),
                                         in.clusterRows(j),
                                         tags.psStripStarts.start(s0),
                                         tags.psStripStarts.start(s0 + 1),
                                         [&sh](uint32_t q, float &x, float &y, float &z, int &size) {
                                           x = sh.x(q), y = sh.y(q), z = sh.z(q), size = sh.size(q);
                                         });
          }
          outOTHits.psMatches(cp) = mark.nMatches, outOTHits.psQOverPt(cp) = mark.qOverPt;
          outOTHits.psCounts(cp) = mark.counts, outOTHits.psQOverPt2(cp) = mark.qOverPt2;
          outOTHits.psCountsLow(cp) = mark.countsLow;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::bins

#endif
