#ifndef RecoTracker_LSTCore_src_alpaka_Hit_h
#define RecoTracker_LSTCore_src_alpaka_Hit_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"
#include "RecoTracker/LSTCore/interface/alpaka/HitsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float eta(TAcc const& acc, float x, float y, float z) {
    float r3 = alpaka::math::sqrt(acc, x * x + y * y + z * z);
    float rt = alpaka::math::sqrt(acc, x * x + y * y);
    float eta = ((z > 0) - (z < 0)) * alpaka::math::acosh(acc, r3 / rt);
    return eta;
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi_mpi_pi(TAcc const& acc, float x) {
    if (alpaka::math::abs(acc, x) <= float(M_PI))
      return x;

    constexpr float o2pi = 1.f / (2.f * float(M_PI));
    float n = alpaka::math::round(acc, x * o2pi);
    return x - n * float(2.f * float(M_PI));
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float phi(TAcc const& acc, float x, float y) {
    return phi_mpi_pi(acc, float(M_PI) + alpaka::math::atan2(acc, -y, -x));
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhi(TAcc const& acc, float x1, float y1, float x2, float y2) {
    float phi1 = phi(acc, x1, y1);
    float phi2 = phi(acc, x2, y2);
    return phi_mpi_pi(acc, (phi2 - phi1));
  }

  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float deltaPhiChange(TAcc const& acc, float x1, float y1, float x2, float y2) {
    return deltaPhi(acc, x1, y1, x2 - x1, y2 - y1);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float calculate_dPhi(float phi1, float phi2) {
    // Calculate dPhi
    float dPhi = phi1 - phi2;

    // Normalize dPhi to be between -pi and pi
    if (dPhi > float(M_PI)) {
      dPhi -= 2 * float(M_PI);
    } else if (dPhi < -float(M_PI)) {
      dPhi += 2 * float(M_PI);
    }

    return dPhi;
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int binary_search(const unsigned int* data,  // Array that we are searching over
                                                        unsigned int search_val,  // Value we want to find in data array
                                                        unsigned int ndata)       // Number of elements in data array
  {
    unsigned int low = 0;
    unsigned int high = ndata - 1;

    while (low <= high) {
      unsigned int mid = (low + high) / 2;
      unsigned int test_val = data[mid];
      if (test_val == search_val)
        return mid;
      else if (test_val > search_val)
        high = mid - 1;
      else
        low = mid + 1;
    }
    // Couldn't find search value in array.
    return -1;
  }

  struct ModuleRangesKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  HitsOccupancy hitsOccupancy,
                                  int nLowerModules) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int lowerIndex = globalThreadIdx[2]; lowerIndex < nLowerModules; lowerIndex += gridThreadExtent[2]) {
        uint16_t upperIndex = modulesInGPU.partnerModuleIndices[lowerIndex];
        if (hitsOccupancy.hitRanges()[lowerIndex][0] != -1 && hitsOccupancy.hitRanges()[upperIndex][0] != -1) {
          hitsOccupancy.hitRangesLower()[lowerIndex] = hitsOccupancy.hitRanges()[lowerIndex][0];
          hitsOccupancy.hitRangesUpper()[lowerIndex] = hitsOccupancy.hitRanges()[upperIndex][0];
          hitsOccupancy.hitRangesnLower()[lowerIndex] =
              hitsOccupancy.hitRanges()[lowerIndex][1] - hitsOccupancy.hitRanges()[lowerIndex][0] + 1;
          hitsOccupancy.hitRangesnUpper()[lowerIndex] =
              hitsOccupancy.hitRanges()[upperIndex][1] - hitsOccupancy.hitRanges()[upperIndex][0] + 1;
        }
      }
    }
  };

  struct HitLoopKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t Endcap,          // Integer corresponding to endcap in module subdets
                                  uint16_t TwoS,            // Integer corresponding to TwoS in moduleType
                                  unsigned int nModules,    // Number of modules
                                  unsigned int nEndCapMap,  // Number of elements in endcap map
                                  EndcapGeometryDevConst endcapGeometry,
                                  Modules modulesInGPU,
                                  Hits hits,
                                  HitsOccupancy hitsOccupancy,
                                  unsigned int nHits) const  // Total number of hits in event
    {
      auto geoMapDetId = endcapGeometry.geoMapDetId();  // DetId's from endcap map
      auto geoMapPhi = endcapGeometry.geoMapPhi();      // Phi values from endcap map
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
      for (unsigned int ihit = globalThreadIdx[2]; ihit < nHits; ihit += gridThreadExtent[2]) {
        float ihit_x = hits.xs()[ihit];
        float ihit_y = hits.ys()[ihit];
        float ihit_z = hits.zs()[ihit];
        int iDetId = hits.detid()[ihit];

        hits.rts()[ihit] = alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y);
        hits.phis()[ihit] = phi(acc, ihit_x, ihit_y);
        hits.etas()[ihit] =
            ((ihit_z > 0) - (ihit_z < 0)) *
            alpaka::math::acosh(
                acc, alpaka::math::sqrt(acc, ihit_x * ihit_x + ihit_y * ihit_y + ihit_z * ihit_z) / hits.rts()[ihit]);
        int found_index = binary_search(modulesInGPU.mapdetId, iDetId, nModules);
        uint16_t lastModuleIndex = modulesInGPU.mapIdx[found_index];

        hits.moduleIndices()[ihit] = lastModuleIndex;

        if (modulesInGPU.subdets[lastModuleIndex] == Endcap && modulesInGPU.moduleType[lastModuleIndex] == TwoS) {
          found_index = binary_search(geoMapDetId, iDetId, nEndCapMap);
          float phi = geoMapPhi[found_index];
          float cos_phi = alpaka::math::cos(acc, phi);
          hits.highEdgeXs()[ihit] = ihit_x + 2.5f * cos_phi;
          hits.lowEdgeXs()[ihit] = ihit_x - 2.5f * cos_phi;
          float sin_phi = alpaka::math::sin(acc, phi);
          hits.highEdgeYs()[ihit] = ihit_y + 2.5f * sin_phi;
          hits.lowEdgeYs()[ihit] = ihit_y - 2.5f * sin_phi;
        }
        // Need to set initial value if index hasn't been seen before.
        int old = alpaka::atomicCas(acc,
                                    &(hitsOccupancy.hitRanges()[lastModuleIndex][0]),
                                    -1,
                                    static_cast<int>(ihit),
                                    alpaka::hierarchy::Threads{});
        // For subsequent visits, stores the min value.
        if (old != -1)
          alpaka::atomicMin(acc,
                            &hitsOccupancy.hitRanges()[lastModuleIndex][0],
                            static_cast<int>(ihit),
                            alpaka::hierarchy::Threads{});

        alpaka::atomicMax(
            acc, &hitsOccupancy.hitRanges()[lastModuleIndex][1], static_cast<int>(ihit), alpaka::hierarchy::Threads{});
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
