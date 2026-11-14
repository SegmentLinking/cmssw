#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkRZLimits_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_MkRZLimits_h

// Portable copy of RecoTracker/MkFitCore/src/MkRZLimits.h / .cc. The LayerInfo of the entry is copied from the ES
// layer table instead of pointed to (layer_info_1()). No member initializers: it lives in block shared memory.
// Only single-layer plan entries exist in CMSSW (IterationConfig serialises LayerControl::m_layer only), so the
// two-layer setup() is not ported and m_is_double is always false.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
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
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void setup(::mkfitdev::LayerInfoSoA::ConstView geo, int layer, bool is_outward);

    // reset()
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void reset();

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool is_outward() const { return m_is_outward; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool is_inward() const { return !m_is_outward; }

    // rz_quadrant_check(z, pz)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool rz_quadrant_check(float z, float pz) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
