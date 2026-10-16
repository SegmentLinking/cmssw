#ifndef RecoTracker_MkFitAlpaka_src_alpaka_inputs_DeviceHitsKernel_h
#define RecoTracker_MkFitAlpaka_src_alpaka_inputs_DeviceHitsKernel_h

// One thread per HitSoA row. The transform variant is Contract::kFuseFirst: it is the one that reproduces the
// RecoTracker/MkFit host converters bitwise (0 differences in position and error on pixel and OT
// hits); fmaf is correctly rounded on every backend, so the result does not depend on the device compiler's
// contraction.
#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/DeviceHitsBuild.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {

  struct KernelFillHitsFromDeviceInputs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, DeviceHitInputs in, ::mkfitdev::HitSoA::View h) const {
      using ::mkfitdev::Contract;
      const uint32_t n = in.nPixel + in.nStrip;
      if (cms::alpakatools::once_per_grid(acc)) {
        h.nPixel() = in.nPixel;
        h.nStrip() = in.nStrip;
      }
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const bool pixel = i < in.nPixel;
        float lx = 0.f, ly = 0.f, exx = 0.f, exy = 0.f, eyy = 0.f;
        int32_t mod = -1;
        if (pixel) {
          const uint32_t j = in.srcRow[i];
          if (j != ::mkfitdev::kNoHit) {
            mod = in.detectorIndex[j];
            lx = in.xLocal[j];
            ly = in.yLocal[j];
            exx = in.xerrLocal[j];
            eyy = in.yerrLocal[j];
          }
        } else if (in.otModule != nullptr) {  // device OT rechit SoA (exy = 0 in Phase2StripCPE)
          const uint32_t k = i - in.nPixel;
          mod = in.otModule[k];
          lx = in.otLx[k];
          ly = in.otLy[k];
          exx = in.otExx[k];
          eyy = in.otEyy[k];
        } else {
          mod = in.module[i];
          lx = in.lx[i];
          ly = in.ly[i];
          exx = in.exx[i];
          exy = in.exy[i];
          eyy = in.eyy[i];
        }
        auto r = h[i];
        if (mod < 0) {  // cluster index without a hit: default mkfit::Hit, layer -1
          r.x() = 0.f;
          r.y() = 0.f;
          r.z() = 0.f;
          r.e00() = 0.f;
          r.e10() = 0.f;
          r.e11() = 0.f;
          r.e20() = 0.f;
          r.e21() = 0.f;
          r.e22() = 0.f;
          r.packed() = 0;
          r.layer() = -1;
          continue;
        }
        const ::mkfitdev::HitModuleDev m = in.modules[mod];
        float g[3], e[6];
        ::mkfitdev::localToGlobal<Contract::kFuseFirst>(m, lx, ly, g);
        ::mkfitdev::localToGlobalError<Contract::kFuseFirst>(m, exx, exy, eyy, e);
        r.x() = g[0];
        r.y() = g[1];
        r.z() = g[2];
        r.e00() = e[0];
        r.e10() = e[1];
        r.e11() = e[2];
        r.e20() = e[3];
        r.e21() = e[4];
        r.e22() = e[5];
        const uint32_t sp = (!pixel && in.otModule != nullptr) ? uint32_t(in.otSize[i - in.nPixel]) : in.spans[i];
        r.packed() = pixel ? ::mkfitdev::packPixel(m.detIdInLayer, int(sp & 0xffffu), int(sp >> 16))
                           : ::mkfitdev::packStrip(m.detIdInLayer, int(sp));
        r.layer() = m.layer;
      }
    }
  };

  inline void fillHitsFromDeviceInputs(Queue& queue, DeviceHitInputs const& in, ::mkfitdev::HitSoA::View hits) {
    const uint32_t n = in.nPixel + in.nStrip;
    const uint32_t threads = 128;
    const uint32_t blocks = n == 0 ? 1 : (n + threads - 1) / threads;
    alpaka::exec<Acc1D>(
        queue, cms::alpakatools::make_workdiv<Acc1D>(blocks, threads), KernelFillHitsFromDeviceInputs{}, in, hits);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits

#endif
