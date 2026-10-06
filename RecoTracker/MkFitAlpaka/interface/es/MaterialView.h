#ifndef RecoTracker_MkFitAlpaka_interface_es_MaterialView_h
#define RecoTracker_MkFitAlpaka_interface_es_MaterialView_h

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"

namespace mkfitdev {

  // Flat, device-usable view of mkfit::TrackerInfo's (z, r) material map. Element (binZ, binR) sits at
  // binZ * nBinsR + binR, as rectvec stores it; bins are int(z * facZ) and int(r * facR), as TrackerInfo::mat_bin_z/r.
  // Out-of-range bins mean no material (TrackerInfo::material_checked). Filled by the ES producer
  // (mkfitdev::ESData::view().material, pointing into the MaterialSoA columns of the same memory space).
  struct MaterialView {
    const float* bbxi;
    const float* radl;
    int nBinsZ;
    int nBinsR;
    float facZ;
    float facR;
    // mkfit::Config::mag_* (runtime; MkFitGeometryESProducer bFieldParams): the
    // parametrised field of every propagation with use_param_b_field. Carried here because every
    // PropagationFlags carries this view (it replaces MkFitCore's TrackerInfo back-pointer).
    Config::BFieldParams bField;

    // TrackerInfo::mat_bin_z / mat_bin_r / check_bins
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int binZ(float z) const { return z * facZ; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int binR(float r) const { return r * facR; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool checkBins(int bz, int br) const {
      return bz >= 0 && bz < nBinsZ && br >= 0 && br < nBinsR;
    }

    // TrackerInfo::material_checked(z, r): {bbxi, radl}, or {0, 0} outside the map
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void materialChecked(float z, float r, float& outBbxi, float& outRadl) const {
      const int zbin = binZ(z), rbin = binR(r);
      if (checkBins(zbin, rbin)) {
        outBbxi = bbxi[zbin * nBinsR + rbin];
        outRadl = radl[zbin * nBinsR + rbin];
      } else {
        outBbxi = 0.f;
        outRadl = 0.f;
      }
    }
  };

}  // namespace mkfitdev

#endif
