#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "PixelCPETrackAnglesTestKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelCPEforDeviceTest {

  using namespace pixelCPEforDevice::test;

  class TrackAnglesKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::Phase2> const* params,
                                  PixelGenErrorTablesSoAConstView tables,
                                  TrackAnglesCase const* cases,
                                  TrackAnglesResult* results,
                                  int32_t size) const {
      for (int32_t i : cms::alpakatools::uniform_elements(acc, size)) {
        auto const& testCase = cases[i];
        results[i].valid = pixelCPEforDevice::hitParametersTrackAngles(params->commonParams(),
                                                                       params->detParams(testCase.module),
                                                                       tables,
                                                                       testCase.module,
                                                                       testCase.cluster,
                                                                       testCase.cotAlpha,
                                                                       testCase.cotBeta,
                                                                       results[i].hit);
      }
    }
  };

  void runTrackAngles(Queue& queue,
                      pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::Phase2> const* params,
                      PixelGenErrorTablesSoAConstView tables,
                      TrackAnglesCase const* cases,
                      TrackAnglesResult* results,
                      int32_t size) {
    constexpr int32_t threads = 256;
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(size, threads), threads);
    alpaka::exec<Acc1D>(queue, workDiv, TrackAnglesKernel{}, params, tables, cases, results, size);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::pixelCPEforDeviceTest
