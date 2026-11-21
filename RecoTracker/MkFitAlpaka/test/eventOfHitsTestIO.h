#ifndef RecoTracker_MkFitAlpaka_test_eventOfHitsTestIO_h
#define RecoTracker_MkFitAlpaka_test_eventOfHitsTestIO_h

// Record format shared by the MkFitCore EventOfHits reference writer (testEventOfHitsMkFitCoreRef.cpp) and the port
// comparator (alpaka/testEventOfHits.dev.cc): the layers (axis constants as MkFitCore's axis classes compute them), the
// input hits in HitSoA row order, and per layer what mkfit::LayerOfHits holds after the registration: the hits in
// internal order with their infos, the bin table, the largest hit extents and the answers of the range queries.

#include <cstdint>
#include <cstdio>
#include <vector>

namespace mkfitdev::eohtest {

  struct Header {
    int32_t nLayers;
    int32_t nPixel;  // HitSoA rows [0, nPixel) are the pixel wrapper's hits, the rest the strip wrapper's
    int32_t nStrip;
    int32_t nQueries;
    int32_t nPhiBins;  // axis_pow2_u1::size_of_N()
    float phiRMin;     // axis_pow2_u1(-PI, PI): m_R_min, m_M_fac, m_N_fac
    float phiMFac;
    float phiNFac;
  };

  // LayerOfHits::m_ax_eta of the layer and its MkFitCore outputs
  struct LayerRec {
    float qRMin, qRMax, qMFac, qNFac, qMLbhp, qNLbhp;
    uint32_t qLastMBin, qLastNBin, nQBins;
    int32_t isBarrel, isPixel;
    uint32_t nHits;
    float maxQHalfLength, maxPhiHalfExtent;
  };

  // one HitSoA row (Hit position and packed SMatrixSym33 error); layer -1 = not registered
  struct HitRec {
    float x, y, z, e00, e10, e11, e20, e21, e22;
    int32_t layer;
  };

  // LayerOfHits internal index i: getOriginalHitIndex(i) and hit_info(i)
  struct BinnedRec {
    uint32_t rank;
    float phi, q, qHalfLength, qbar, phiHalfExtent, qbarHalfExtent;
  };

  // phiQBinContent(pi, qi)
  struct BinRec {
    uint32_t first, count;
  };

  // phiRangeBins(phiLo, phiHi) and qRangeBins(qLo, qHi) of a layer
  struct QueryRec {
    int32_t layer;
    float phiLo, phiHi, qLo, qHi;
    uint32_t phiBegin, phiEnd, qBegin, qEnd;
  };

  struct Reference {
    Header header;
    std::vector<LayerRec> layers;
    std::vector<HitRec> hits;
    std::vector<std::vector<BinnedRec>> binned;  // per layer, nHits rows
    std::vector<std::vector<BinRec>> bins;       // per layer, row qi * nPhiBins + pi
    std::vector<QueryRec> queries;
  };

  inline void writeReference(std::FILE* file, Reference const& event) {
    std::fwrite(&event.header, sizeof(Header), 1, file);
    std::fwrite(event.layers.data(), sizeof(LayerRec), event.layers.size(), file);
    std::fwrite(event.hits.data(), sizeof(HitRec), event.hits.size(), file);
    for (size_t l = 0; l < event.layers.size(); ++l) {
      std::fwrite(event.binned[l].data(), sizeof(BinnedRec), event.binned[l].size(), file);
      std::fwrite(event.bins[l].data(), sizeof(BinRec), event.bins[l].size(), file);
    }
    std::fwrite(event.queries.data(), sizeof(QueryRec), event.queries.size(), file);
  }

  // false on a truncated or malformed file
  inline bool readReference(std::FILE* file, Reference& event) {
    Header& h = event.header;
    if (std::fread(&h, sizeof(Header), 1, file) != 1 || h.nLayers < 0 || h.nPixel < 0 || h.nStrip < 0 ||
        h.nQueries < 0 || h.nPhiBins <= 0)
      return false;
    event.layers.resize(h.nLayers);
    event.hits.resize(h.nPixel + h.nStrip);
    event.queries.resize(h.nQueries);
    if (std::fread(event.layers.data(), sizeof(LayerRec), h.nLayers, file) != size_t(h.nLayers) ||
        std::fread(event.hits.data(), sizeof(HitRec), event.hits.size(), file) != event.hits.size())
      return false;
    event.binned.resize(h.nLayers);
    event.bins.resize(h.nLayers);
    for (int l = 0; l < h.nLayers; ++l) {
      event.binned[l].resize(event.layers[l].nHits);
      event.bins[l].resize(event.layers[l].nQBins * uint32_t(h.nPhiBins));
      if (std::fread(event.binned[l].data(), sizeof(BinnedRec), event.binned[l].size(), file) !=
              event.binned[l].size() ||
          std::fread(event.bins[l].data(), sizeof(BinRec), event.bins[l].size(), file) != event.bins[l].size())
        return false;
    }
    return std::fread(event.queries.data(), sizeof(QueryRec), h.nQueries, file) == size_t(h.nQueries);
  }

}  // namespace mkfitdev::eohtest

#endif
