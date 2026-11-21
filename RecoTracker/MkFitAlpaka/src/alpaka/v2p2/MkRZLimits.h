#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkRZLimits_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkRZLimits_h

// Portable copy of RecoTracker/MkFitCore/src/MkRZLimits.h / .cc. The LayerInfo of the entry is copied from the ES
// layer table instead of pointed to (layer_info_1()). No member initializers: it lives in block shared memory.
// Only single-layer plan entries exist in CMSSW (IterationConfig serialises LayerControl::m_layer only), so the
// two-layer setup() is not ported and m_is_double is always false.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  struct MkRZLimits {
    // layer_info_1(): the members of LayerInfo the finder reads
    int m_layer_id;
    int m_layer_type;  // ::mkfitdev::LayerType
    bool m_layer_is_pixel;
    bool m_layer_has_r_range_hole;
    float m_layer_hole_r_min, m_layer_hole_r_max;

    float m_rin, m_rout, m_zmin, m_zmax;
    bool m_is_barrel;
    bool m_is_outward;
    bool m_is_double;

    // setup(const LayerInfo& li, bool is_outward) for li = geo[layer]
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void setup(::mkfitdev::LayerInfoSoA::ConstView geo,
                                                   int layer,
                                                   bool is_outward) {
      m_layer_id = layer;
      m_layer_type = geo[layer].layer_type();
      m_layer_is_pixel = geo[layer].is_pixel();
      m_layer_has_r_range_hole = geo[layer].has_r_range_hole();
      m_layer_hole_r_min = geo[layer].hole_r_min();
      m_layer_hole_r_max = geo[layer].hole_r_max();

      m_rin = geo[layer].rin();
      m_rout = geo[layer].rout();
      m_zmin = geo[layer].zmin();
      m_zmax = geo[layer].zmax();
      m_is_barrel = m_layer_type == static_cast<int>(::mkfitdev::LayerType::Barrel);
      m_is_outward = is_outward;
      m_is_double = false;
    }

    // reset(): the values of a default-constructed MkFitCore MkRZLimits, no layer
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void reset() {
      m_layer_id = -1;
      m_layer_type = static_cast<int>(::mkfitdev::LayerType::Undef);
      m_layer_is_pixel = false;
      m_layer_has_r_range_hole = false;
      m_layer_hole_r_min = m_layer_hole_r_max = 0;
      m_rin = m_rout = m_zmin = m_zmax = 0;
      m_is_barrel = true;
      m_is_outward = true;
      m_is_double = false;
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool is_outward() const { return m_is_outward; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool is_inward() const { return !m_is_outward; }

    // False if the track is already past the layer in z and moving away from it.
    // Exact for a helix, where z is monotonic along the track; does not test r
    // (the WSR verdict does) and has no margin for the track's z error.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool rz_quadrant_check(float z, float pz) const {
      if (is_inward()) {
        pz = -pz;
      }
      if (pz < 0.0f) {
        if (z < m_zmin)
          return false;
      } else {
        if (z > m_zmax)
          return false;
      }
      return true;
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
