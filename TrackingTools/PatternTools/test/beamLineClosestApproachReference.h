#ifndef TrackingTools_PatternTools_test_beamLineClosestApproachReference_h
#define TrackingTools_PatternTools_test_beamLineClosestApproachReference_h

#include <cstdint>
#include <vector>

#include "TrackingTools/PatternTools/interface/beamLineClosestApproach.h"

// Inputs and host results of testBeamLineClosestApproach. Only plain types here: the host classes are used in
// beamLineClosestApproachReference.cc alone, compiled by the host compiler.
namespace beamLineClosestApproachTest {

  using beamLineClosestApproach::BeamLine;
  using beamLineClosestApproach::FreeState;
  using curvilinearJacobian::Vector3;

  struct Inputs {
    std::vector<FreeState> tracks;
    std::vector<Vector3> fields;  // the host field at each track position, tesla
    std::vector<BeamLine> beamLines;
  };

  struct HostResult {
    bool valid;
    FreeState state;  // TSCBLBuilderNoMaterial
    // the AnalyticalCurvilinearJacobian of that state and its inputs (end point, momentum and path length)
    Vector3 endPosition, endMomentum;
    double pathLength;
    double jacobian[5][5];
  };

  Inputs makeInputs(int nTracks, uint32_t seed);
  std::vector<HostResult> hostResults(Inputs const& inputs);

}  // namespace beamLineClosestApproachTest

#endif  // TrackingTools_PatternTools_test_beamLineClosestApproachReference_h
