#ifndef TrackingTools_PatternTools_test_TSCBLBuilderNoMaterialPortableReference_h
#define TrackingTools_PatternTools_test_TSCBLBuilderNoMaterialPortableReference_h

#include <cstdint>
#include <vector>

#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterialPortable.h"

// Inputs and host results of testTSCBLBuilderNoMaterialPortable. Only plain types here: the host classes are used in
// TSCBLBuilderNoMaterialPortableReference.cc alone, compiled by the host compiler.
namespace tscblBuilderNoMaterialPortableTest {

  using BeamLine = TSCBLBuilderNoMaterialPortable::BeamLine;
  using FreeState = TSCBLBuilderNoMaterialPortable::FreeState;
  using Matrix5 = TSCBLBuilderNoMaterialPortable::Matrix5;
  using Vector3 = TSCBLBuilderNoMaterialPortable::Vector3;

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
    Matrix5 jacobian;
  };

  Inputs makeInputs(int nTracks, uint32_t seed);
  std::vector<HostResult> hostResults(Inputs const& inputs);

}  // namespace tscblBuilderNoMaterialPortableTest

#endif  // TrackingTools_PatternTools_test_TSCBLBuilderNoMaterialPortableReference_h
