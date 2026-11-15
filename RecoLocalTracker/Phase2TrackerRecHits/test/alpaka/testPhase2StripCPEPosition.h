#ifndef RecoLocalTracker_Phase2TrackerRecHits_test_alpaka_testPhase2StripCPEPosition_h
#define RecoLocalTracker_Phase2TrackerRecHits_test_alpaka_testPhase2StripCPEPosition_h

#include <cstdint>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testPhase2StripCPEPosition {

  struct ClusterInput {
    uint32_t firstStrip;
    uint32_t column;
    uint16_t size;
  };

  struct GlobalPosition {
    float x;
    float y;
    float z;
  };

  // phase2StripCPE::localPosition of each cluster on the module surface frame, computed on the device
  void globalPositions(Queue& queue,
                       phase2StripCPE::ModuleParams const& params,
                       SOAFrame<float> const& frame,
                       ClusterInput const* clusters,
                       GlobalPosition* positions,
                       uint32_t size);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testPhase2StripCPEPosition

#endif  // RecoLocalTracker_Phase2TrackerRecHits_test_alpaka_testPhase2StripCPEPosition_h
