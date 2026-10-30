#ifndef RecoTracker_MkFitAlpaka_interface_hits_alpaka_DeviceHitsBuild_h
#define RecoTracker_MkFitAlpaka_interface_hits_alpaka_DeviceHitsBuild_h

// Device hit input: HitSoA rows made on the device from the pixel rechit SoA
// columns, the raw OT rechit rows and the per-module table, bitwise equal to the RecoTracker/MkFit hit converters.
#include <cstdint>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitInput.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitRawSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {
  using DeviceHitRawDeviceCollection = PortableCollection<::mkfitdev::DeviceHitRawSoA>;

  struct DeviceHitInputs {
    // pixel rechit SoA columns (device memory)
    const float* xLocal;
    const float* yLocal;
    const float* xerrLocal;
    const float* yerrLocal;
    const uint16_t* detectorIndex;
    const ::mkfitdev::HitModuleDev* modules;  // indexed by GeomDet::index()
    // raw rows (DeviceHitRawSoA columns)
    const uint32_t* srcRow;
    const int32_t* module;
    const uint32_t* spans;
    const float* lx;
    const float* ly;
    const float* exx;
    const float* exy;
    const float* eyy;
    uint32_t nPixel;
    uint32_t nStrip;
    // OT rows from the device OT rechit SoA (row i >= nPixel reads index i - nPixel);
    // nullptr = the raw rows above
    const int32_t* otModule = nullptr;
    const uint16_t* otSize = nullptr;
    const float* otLx = nullptr;
    const float* otLy = nullptr;
    const float* otExx = nullptr;
    const float* otEyy = nullptr;
  };

  // fills hits rows [0, nPixel + nStrip) and the nPixel / nStrip scalars
  void runFillHitsFromDeviceInputs(Queue& queue, DeviceHitInputs const& in, ::mkfitdev::HitSoA::View hits);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits

#endif
