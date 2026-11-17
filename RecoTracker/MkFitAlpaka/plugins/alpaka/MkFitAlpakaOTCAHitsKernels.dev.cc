// See MkFitAlpakaOTCAHitsKernels.h.
#include <cmath>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTCpe.h"

#include "MkFitAlpakaOTCAHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits {

  using namespace cms::alpakatools;
  using ::mkfitdev::Contract;

  namespace {
    // gx * gx + gy * gy in double with the contraction variant M (as sumOfProducts in float)
    template <Contract M>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE double sumSq(double a, double b) {
      if constexpr (M == Contract::kFuseSecond)
        return std::fma(b, b, a * a);
      else if constexpr (M == Contract::kFuseFirst)
        return std::fma(a, a, b * b);
      else {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __dadd_rn(__dmul_rn(a, a), __dmul_rn(b, b));
#else
        volatile double aa = a * a;
        volatile double bb = b * b;
        return aa + bb;
#endif
      }
    }

    template <Contract M>
    struct KernelOTCAHits {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::OTRecHitSoA::ConstView ot,
                                    ::mkfitdev::OTCpeModule const* table,
                                    int32_t const* pOffset,
                                    uint32_t const* hitStart,
                                    uint32_t const* keyStart,
                                    CAHitsParams p,
                                    ::reco::TrackingRecHitView hits) const {
        for (uint32_t k : uniform_elements(acc, p.nOT)) {
          const int32_t mi = ot[k].module() - p.firstIndex;
          const int32_t off = pOffset[mi];
          if (off < 0)
            continue;
          const uint32_t idx = hitStart[off] + (k - keyStart[off]);
          auto h = hits[idx];
          const float lx = ot[k].lx(), ly = ot[k].ly();
          h.xLocal() = lx;
          h.yLocal() = ly;
          h.xerrLocal() = ot[k].exx();
          h.yerrLocal() = ot[k].eyy();
          float g[3];
          ::mkfitdev::otcpe::toGlobal<M>(table[mi], lx, ly, g);
          // Phase2OTRecHitsSoAConverter: double gx = globalPosition.x() - bs.x0() (stored as float)
          const double gx = double(g[0]) - p.bsx;
          const double gy = double(g[1]) - p.bsy;
          const double gz = double(g[2]) - p.bsz;
          h.xGlobal() = gx;
          h.yGlobal() = gy;
          h.zGlobal() = gz;
          h.rGlobal() = std::sqrt(sumSq<M>(gx, gy));
          h.iphi() = unsafe_atan2s<7>(gy, gx);
          h.chargeAndStatus().charge = 0;
          h.chargeAndStatus().status = {false, false, false, false, 0};
          h.clusterSizeX() = -1;
          h.clusterSizeY() = -1;
          h.detectorIndex() = p.modulesInPixel + off;
        }
      }
    };
  }  // namespace

  void runOTCAHits(Queue& queue,
                   ::mkfitdev::OTRecHitSoA::ConstView ot,
                   ::mkfitdev::OTCpeModule const* table,
                   int32_t const* pOffset,
                   uint32_t const* hitStart,
                   uint32_t const* keyStart,
                   CAHitsParams const& p,
                   ::reco::TrackingRecHitView hits) {
    if (p.nOT == 0)
      return;
    constexpr uint32_t kBlock = 128;
    const auto wd = make_workdiv<Acc1D>(divide_up_by(p.nOT, kBlock), kBlock);
    // the contraction of Phase2OTRecHitsSoAConverter's GeomDet::toGlobal (first product fused)
    alpaka::exec<Acc1D>(
        queue, wd, KernelOTCAHits<Contract::kFuseFirst>{}, ot, table, pOffset, hitStart, keyStart, p, hits);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::othits
