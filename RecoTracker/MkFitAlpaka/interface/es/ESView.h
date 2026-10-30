#ifndef RecoTracker_MkFitAlpaka_interface_es_ESView_h
#define RecoTracker_MkFitAlpaka_interface_es_ESView_h

// Trivially copyable bundle of const views into the ES data of one memory space: pass it by value to kernels.
// Obtain it with mkfitdev::ESData<TDev>::view().

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"
#include "RecoTracker/MkFitAlpaka/interface/es/MaterialView.h"

namespace mkfitdev {

  // Hash slot of a detid in the DetIdMap table: murmur3 32-bit finalizer, top bits.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t detIdHashSlot(uint32_t detid, uint32_t hashShift) {
    uint32_t hash = detid;
    hash ^= hash >> 16;
    hash *= 0x85ebca6bu;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35u;
    hash ^= hash >> 16;
    return hash >> hashShift;
  }

  // mkfit::WithinSensitiveRegion_e / WSR_Result (same values)
  enum WithinSensitiveRegion : int { WSR_Undef = -1, WSR_Inside = 0, WSR_Edge, WSR_Outside, WSR_Failed };
  struct WSRResult {
    int wsr;
    bool in_gap;
  };

  struct ESView {
    LayerInfoSoA::ConstView layers;    // row = layer id
    ModuleInfoSoA::ConstView modules;  // row = layers[l].module_begin() + short id
    DetIdMapSoA::ConstView detIdMap;   // use findModule()
    MaterialView material;             // TrackerInfo material map
    ESConfig const* config;            // iteration/steering/propagation config

    // detid -> global module row, or es::kNoModule. Replaces LayerInfo::short_id (m_detid2sid) plus the
    // detid -> layer lookup: modules[row].layer() and modules[row].sid() give both.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int findModule(uint32_t detid) const {
      if (detid == 0u)
        return es::kNoModule;
      const uint32_t mask = static_cast<uint32_t>(detIdMap.metadata().size()) - 1u;
      uint32_t slot = detIdHashSlot(detid, detIdMap.hash_shift());
      const int maxProbe = detIdMap.max_probe();
      for (int probe = 0; probe <= maxProbe; ++probe) {
        const uint32_t key = detIdMap[slot].key();
        if (key == detid)
          return detIdMap[slot].module();
        if (key == 0u)
          return es::kNoModule;
        slot = (slot + 1u) & mask;
      }
      return es::kNoModule;
    }

    // LayerInfo::short_id(detid) for a known layer; -1 if the detid is not a module of that layer.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int shortId(int layer, uint32_t detid) const {
      const int row = findModule(detid);
      return (row != es::kNoModule && modules[row].layer() == layer) ? modules[row].sid() : -1;
    }

    // Global module row of (layer, short id), as LayerInfo::module_info(sid).
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int moduleRow(int layer, int sid) const {
      return layers[layer].module_begin() + sid;
    }

    // ---- LayerInfo geometry predicates, same operations and order as TrackerInfo.h
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isBarrel(int layer) const {
      return layers[layer].layer_type() == static_cast<int>(LayerType::Barrel);
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE WSRResult isWithinZSensitiveRegion(int layer, float z, float dz) const {
      const float zmin = layers[layer].zmin(), zmax = layers[layer].zmax();
      if (z > zmax + dz || z < zmin - dz)
        return WSRResult{WSR_Outside, false};
      if (z < zmax - dz && z > zmin + dz)
        return WSRResult{WSR_Inside, false};
      return WSRResult{WSR_Edge, false};
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE WSRResult isWithinRSensitiveRegion(int layer, float r, float dr) const {
      const float rin = layers[layer].rin(), rout = layers[layer].rout();
      if (r > rout + dr || r < rin - dr)
        return WSRResult{WSR_Outside, false};
      if (r < rout - dr && r > rin + dr) {
        if (layers[layer].has_r_range_hole()) {
          const float hmin = layers[layer].hole_r_min(), hmax = layers[layer].hole_r_max();
          if (r < hmax - dr && r > hmin + dr)
            return WSRResult{WSR_Outside, true};
          if (r < hmax + dr && r > hmin - dr)
            return WSRResult{WSR_Edge, true};
        }
        return WSRResult{WSR_Inside, false};
      }
      return WSRResult{WSR_Edge, false};
    }
  };

}  // namespace mkfitdev

#endif
