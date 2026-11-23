#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

#include "testPhase2StripCPEPosition.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testPhase2StripCPEPosition {

  namespace {
    struct PositionKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    phase2StripCPE::ModuleParams params,
                                    SOAFrame<float> frame,
                                    ClusterInput const* clusters,
                                    GlobalPosition* positions,
                                    uint32_t size) const {
        for (uint32_t index : cms::alpakatools::uniform_elements(acc, size)) {
          auto const& cluster = clusters[index];
          auto const local = phase2StripCPE::localPosition(params, cluster.firstStrip, cluster.column, cluster.size);
          frame.toGlobal(local.x, local.y, positions[index].x, positions[index].y, positions[index].z);
        }
      }
    };
  }  // namespace

  void globalPositions(Queue& queue,
                       phase2StripCPE::ModuleParams const& params,
                       SOAFrame<float> const& frame,
                       ClusterInput const* clusters,
                       GlobalPosition* positions,
                       uint32_t size) {
    auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(size, 256), 256);
    alpaka::exec<Acc1D>(queue, workDiv, PositionKernel{}, params, frame, clusters, positions, size);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testPhase2StripCPEPosition
