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
    uint32_t h = detid;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h >> hashShift;
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
    ModuleShapeSoA::ConstView shapes;  // row = layers[l].shape_begin() + shapeid
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
      for (int p = 0; p <= maxProbe; ++p) {
        const uint32_t k = detIdMap[slot].key();
        if (k == detid)
          return detIdMap[slot].module();
        if (k == 0u)
          return es::kNoModule;
        slot = (slot + 1u) & mask;
      }
      return es::kNoModule;
    }

    // LayerInfo::short_id(detid) for a known layer; -1 if the detid is not a module of that layer.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int shortId(int layer, uint32_t detid) const {
      const int m = findModule(detid);
      return (m != es::kNoModule && modules[m].layer() == layer) ? modules[m].sid() : -1;
    }

    // Global module row of (layer, short id), as LayerInfo::module_info(sid).
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int moduleRow(int layer, int sid) const {
      return layers[layer].module_begin() + sid;
    }

    // ---- LayerInfo geometry predicates, same operations and order as TrackerInfo.h
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isBarrel(int l) const {
      return layers[l].layer_type() == static_cast<int>(LayerType::Barrel);
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isWithinZLimits(int l, float z) const {
      return z > layers[l].zmin() && z < layers[l].zmax();
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isWithinRLimits(int l, float r) const {
      return r > layers[l].rin() && r < layers[l].rout();
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isWithinQLimits(int l, float q) const {
      return isBarrel(l) ? isWithinZLimits(l, q) : isWithinRLimits(l, q);
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isInRHole(int l, float r) const {
      return layers[l].has_r_range_hole() ? (r > layers[l].hole_r_min() && r < layers[l].hole_r_max()) : false;
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE WSRResult isWithinZSensitiveRegion(int l, float z, float dz) const {
      const float zmin = layers[l].zmin(), zmax = layers[l].zmax();
      if (z > zmax + dz || z < zmin - dz)
        return WSRResult{WSR_Outside, false};
      if (z < zmax - dz && z > zmin + dz)
        return WSRResult{WSR_Inside, false};
      return WSRResult{WSR_Edge, false};
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE WSRResult isWithinRSensitiveRegion(int l, float r, float dr) const {
      const float rin = layers[l].rin(), rout = layers[l].rout();
      if (r > rout + dr || r < rin - dr)
        return WSRResult{WSR_Outside, false};
      if (r < rout - dr && r > rin + dr) {
        if (layers[l].has_r_range_hole()) {
          const float hmin = layers[l].hole_r_min(), hmax = layers[l].hole_r_max();
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
