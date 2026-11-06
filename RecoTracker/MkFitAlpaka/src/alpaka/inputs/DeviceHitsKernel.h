#ifndef RecoTracker_MkFitAlpaka_src_alpaka_inputs_DeviceHitsKernel_h
#define RecoTracker_MkFitAlpaka_src_alpaka_inputs_DeviceHitsKernel_h

// Fills the HitSoA rows of the device hit input, one thread per row. The transform variant is Contract::kFuseFirst,
// the one that reproduces the RecoTracker/MkFit host converters bitwise in position and error on pixel and OT hits;
// fmaf is correctly rounded on every backend, so the result does not depend on the device compiler's contraction.
#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/DeviceHitsBuild.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {

  struct KernelFillHitsFromDeviceInputs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, DeviceHitInputs in, ::mkfitdev::HitSoA::View hits) const {
      using ::mkfitdev::Contract;
      const uint32_t nHits = in.nPixel + in.nStrip;
      if (cms::alpakatools::once_per_grid(acc)) {
        hits.nPixel() = in.nPixel;
        hits.nStrip() = in.nStrip;
      }
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nHits)) {
        const bool pixel = i < in.nPixel;
        float lx = 0.f, ly = 0.f, exx = 0.f, exy = 0.f, eyy = 0.f;
        int32_t mod = -1;
        if (pixel) {
          const uint32_t soaRow = in.srcRow[i];
          if (soaRow != ::mkfitdev::kNoHit) {
            mod = in.detectorIndex[soaRow];
            lx = in.xLocal[soaRow];
            ly = in.yLocal[soaRow];
            exx = in.xerrLocal[soaRow];
            eyy = in.yerrLocal[soaRow];
          }
        } else if (in.otModule != nullptr) {  // device OT rechit SoA (exy = 0 in Phase2StripCPE)
          const uint32_t otRow = i - in.nPixel;
          mod = in.otModule[otRow];
          lx = in.otLx[otRow];
          ly = in.otLy[otRow];
          exx = in.otExx[otRow];
          eyy = in.otEyy[otRow];
        } else {
          mod = in.module[i];
          lx = in.lx[i];
          ly = in.ly[i];
          exx = in.exx[i];
          exy = in.exy[i];
          eyy = in.eyy[i];
        }
        auto row = hits[i];
        if (mod < 0) {  // cluster index without a hit: default mkfit::Hit, layer -1
          row.x() = 0.f;
          row.y() = 0.f;
          row.z() = 0.f;
          row.e00() = 0.f;
          row.e10() = 0.f;
          row.e11() = 0.f;
          row.e20() = 0.f;
          row.e21() = 0.f;
          row.e22() = 0.f;
          row.packed() = 0;
          row.layer() = -1;
          continue;
        }
        const ::mkfitdev::HitModuleDev moduleInfo = in.modules[mod];
        float global[3], err[6];
        ::mkfitdev::localToGlobal<Contract::kFuseFirst>(moduleInfo, lx, ly, global);
        ::mkfitdev::localToGlobalError<Contract::kFuseFirst>(moduleInfo, exx, exy, eyy, err);
        row.x() = global[0];
        row.y() = global[1];
        row.z() = global[2];
        row.e00() = err[0];
        row.e10() = err[1];
        row.e11() = err[2];
        row.e20() = err[3];
        row.e21() = err[4];
        row.e22() = err[5];
        // pixel: sizeX | sizeY << 16; strip: cluster size
        const uint32_t sizes = (!pixel && in.otModule != nullptr) ? uint32_t(in.otSize[i - in.nPixel]) : in.spans[i];
        row.packed() = pixel ? ::mkfitdev::packPixel(moduleInfo.detIdInLayer,
                                                     int(sizes & ::mkfitdev::kSpanXMask),
                                                     int(sizes >> ::mkfitdev::kSpanYShift))
                             : ::mkfitdev::packStrip(moduleInfo.detIdInLayer, int(sizes));
        row.layer() = moduleInfo.layer;
      }
    }
  };

  inline void fillHitsFromDeviceInputs(Queue& queue, DeviceHitInputs const& in, ::mkfitdev::HitSoA::View hits) {
    constexpr uint32_t kThreadsPerBlock = 128;
    const uint32_t nHits = in.nPixel + in.nStrip;
    const uint32_t blocks = nHits == 0 ? 1 : cms::alpakatools::divide_up_by(nHits, kThreadsPerBlock);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreadsPerBlock),
                        KernelFillHitsFromDeviceInputs{},
                        in,
                        hits);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits

#endif
