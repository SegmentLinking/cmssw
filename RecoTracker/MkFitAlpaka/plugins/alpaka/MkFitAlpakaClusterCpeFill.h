#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaClusterCpeFill_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaClusterCpeFill_h

// Host fills of the per-pixel-hit inputs of the device PixelCPEGeneric (interface/fit/CpeGeneric.h), shared by the
// device final fit (MkFitAlpakaFitDeviceProducer) and the device pixel-seed creator (MkFitAlpakaLstInputProducer).
// The mkFit pixel hit row is the legacy cluster key (convertHits); rows without a cluster get module -1 (no CPE).
// No SiPixelGenError / boost::multi_array include here (MkFitAlpakaFitCpeTables.h has them): it breaks the boost concept
// checks in a CUDA-backend translation unit that includes Eigen first (MkFitAlpakaLstInputProducer.cc).

#include <cstdint>
#include <vector>

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeESData.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeGeneric.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/FitHandoff.h"

namespace mkfitdev::cpe {

  // PixelCPEGenericBase::collect_edge_charges without truncation (menu TruncatePixelCharge = False) + charge.
  inline ClusterCpe clusterCpe(SiPixelCluster const& cl, int module) {
    ClusterCpe c{};
    c.module = module;
    c.e.minRow = cl.minPixelRow();
    c.e.maxRow = cl.maxPixelRow();
    c.e.minCol = cl.minPixelCol();
    c.e.maxCol = cl.maxPixelCol();
    for (int k = 0; k < cl.size(); ++k) {
      const auto px = cl.pixel(k);
      const int x = px.x, y = px.y, q = px.adc;
      if (x == c.e.minRow)
        c.e.qfX += q;
      if (x == c.e.maxRow)
        c.e.qlX += q;
      if (y == c.e.minCol)
        c.e.qfY += q;
      if (y == c.e.maxCol)
        c.e.qlY += q;
    }
    c.charge = cl.charge();
    return c;
  }

  // clusterCpe() for the rows with need[row] != 0 (every row when need is empty); other rows untouched
  inline void fillClusterCpe(edmNew::DetSetVector<SiPixelCluster> const& dsv,
                             CpeTablesHost const& tables,
                             uint32_t nPix,
                             ClusterCpe* out,
                             std::vector<uint8_t> const& need = {}) {
    const bool all = need.empty();
    auto const& data = dsv.data();
    const SiPixelCluster* base = data.data();
    for (auto const& ds : dsv) {
      int mod = -2;  // looked up on the first filled cluster of the module
      for (auto const& cl : ds) {
        const uint32_t key = &cl - base;
        if (key >= nPix || !(all || need[key]))
          continue;
        if (mod == -2) {
          auto it = tables.rawToModule.find(ds.detId());
          mod = it == tables.rawToModule.end() ? -1 : it->second;
        }
        out[key] = clusterCpe(cl, mod);
      }
    }
    for (uint32_t i = data.size(); i < nPix; ++i)
      if (all || need[i]) {
        out[i] = ClusterCpe{};
        out[i].module = -1;
      }
  }

  // device mode: per mkFit pixel row the SoA module index and the SoA cluster id (SiPixelCluster::originalId);
  // false if a cluster has no originalId (persisted clusters): the caller falls back to the host computation
  inline bool fillClusterRefs(edmNew::DetSetVector<SiPixelCluster> const& dsv,
                              CpeTablesHost const& tables,
                              uint32_t nPix,
                              ::mkfitdev::handoff::ClusterRef* out) {
    auto const& data = dsv.data();
    const SiPixelCluster* base = data.data();
    for (auto const& ds : dsv) {
      auto it = tables.rawToModule.find(ds.detId());
      const int mod = it == tables.rawToModule.end() ? -1 : it->second;
      for (auto const& cl : ds) {
        const uint32_t key = &cl - base;
        if (cl.originalId() == SiPixelCluster::invalidClusterId)
          return false;
        if (key < nPix)
          out[key] = ::mkfitdev::handoff::ClusterRef{mod, int32_t(cl.originalId())};
      }
    }
    for (uint32_t i = data.size(); i < nPix; ++i)
      out[i] = ::mkfitdev::handoff::ClusterRef{-1, 0};
    return true;
  }

}  // namespace mkfitdev::cpe

#endif
