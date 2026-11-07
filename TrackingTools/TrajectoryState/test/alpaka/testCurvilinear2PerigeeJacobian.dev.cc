// Compares curvilinear2PerigeeJacobian::compute, run on each device of the backend, with
// PerigeeConversions::jacobianCurvilinear2Perigee on the host, for random states in a non-uniform field.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/curvilinear2PerigeeJacobian.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  // solenoid-like test field: Bz falls with |z|, a radial component grows with |z|; zero beyond |z| = 500 cm
  class TestField final : public MagneticField {
  public:
    TestField() { setNominalValue(); }
    GlobalVector inTesla(const GlobalPoint& point) const override {
      const float pointZ = point.z();
      if (std::abs(pointZ) > 500.f)
        return GlobalVector(0.f, 0.f, 0.f);
      return GlobalVector(2.e-6f * point.x() * pointZ, 2.e-6f * point.y() * pointZ, 3.8f - 6.e-6f * pointZ * pointZ);
    }
  };

  struct State {
    float momentum[3];
    float field[3];  // 1/GeV
    float signedInverseMomentum;
    float transverseCurvature;
  };

  struct JacobianKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  State const* __restrict__ states,
                                  curvilinear2PerigeeJacobian::Matrix5* __restrict__ jacobians,
                                  int32_t nStates) const {
      using curvilinear2PerigeeJacobian::Vector3;
      for (int32_t index : cms::alpakatools::uniform_elements(acc, nStates)) {
        State const& state = states[index];
        jacobians[index] = curvilinear2PerigeeJacobian::Matrix5{};
        curvilinear2PerigeeJacobian::compute(
            Vector3(state.momentum[0], state.momentum[1], state.momentum[2]),
            Vector3(state.field[0], state.field[1], state.field[2]),
            state.signedInverseMomentum,
            state.transverseCurvature,
            [](double angle, double& sinAngle, double& cosAngle) {
              sinAngle = std::sin(angle);
              cosAngle = std::cos(angle);
            },
            jacobians[index]);
      }
    }
  };

}  // namespace

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // states anywhere in the tracker, pT 0.1-1000 GeV, any direction; every 20th state outside the field
  constexpr int32_t nStates = 200000;
  const TestField field;
  std::mt19937 engine(20261010);
  std::uniform_real_distribution<float> uniform(0.f, 1.f);
  std::vector<FreeTrajectoryState> host;
  host.reserve(nStates);
  auto statesHost = cms::alpakatools::make_host_buffer<State[], Platform>(nStates);
  for (int32_t index = 0; index < nStates; ++index) {
    const float radius = 120.f * uniform(engine), positionPhi = 6.2831853f * uniform(engine);
    const float positionZ = index % 20 == 19 ? 600.f : 560.f * (uniform(engine) - 0.5f);
    const float pt = 0.1f * std::pow(1.e4f, uniform(engine)), eta = 8.f * (uniform(engine) - 0.5f);
    const float phi = 6.2831853f * (uniform(engine) - 0.5f);
    const GlobalPoint position(radius * std::cos(positionPhi), radius * std::sin(positionPhi), positionZ);
    const GlobalVector momentum(pt * std::cos(phi), pt * std::sin(phi), pt * std::sinh(eta));
    host.emplace_back(GlobalTrajectoryParameters(position, momentum, uniform(engine) < 0.5f ? -1 : 1, &field));
    FreeTrajectoryState const& state = host.back();
    const GlobalVector fieldInInverseGeV = state.parameters().magneticFieldInInverseGeV();
    statesHost[index] = State{{momentum.x(), momentum.y(), momentum.z()},
                              {fieldInInverseGeV.x(), fieldInInverseGeV.y(), fieldInInverseGeV.z()},
                              static_cast<float>(state.signedInverseMomentum()),
                              static_cast<float>(state.transverseCurvature())};
  }

  bool failed = false;
  for (auto const& device : devices) {
    Queue queue(device);
    auto states = cms::alpakatools::make_device_buffer<State[]>(queue, nStates);
    auto jacobians = cms::alpakatools::make_device_buffer<curvilinear2PerigeeJacobian::Matrix5[]>(queue, nStates);
    auto jacobiansHost = cms::alpakatools::make_host_buffer<curvilinear2PerigeeJacobian::Matrix5[]>(queue, nStates);
    alpaka::memcpy(queue, states, statesHost);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nStates, 128), 128);
    alpaka::exec<Acc1D>(queue, workDiv, JacobianKernel{}, states.data(), jacobians.data(), nStates);
    alpaka::memcpy(queue, jacobiansHost, jacobians);
    alpaka::wait(queue);

    // differences relative to max(1, |host element|): the float momentum direction carries ~1e-7 relative rounding,
    // which the curvature row amplifies by sec^2(lambda) = cosh^2(eta) and its derivative (up to ~30x at |eta| = 4)
    double maxDifference = 0;
    int worst = -1;
    for (int32_t index = 0; index < nStates; ++index) {
      const AlgebraicMatrix55 reference = PerigeeConversions::jacobianCurvilinear2Perigee(host[index]);
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col < 5; ++col) {
          const double difference = std::abs(jacobiansHost[index].element[row][col] - reference(row, col)) /
                                    std::max(1., std::abs(reference(row, col)));
          if (!(difference <= maxDifference)) {  // NaN counts as a difference
            maxDifference = difference;
            worst = index;
          }
        }
    }
    constexpr double tolerance = 1.e-4;
    const bool deviceFailed = !(maxDifference < tolerance);
    std::cout << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " " << alpaka::getName(device) << ", " << nStates
              << " states: max |Jacobian difference| / max(1, |element|) " << maxDifference << " (state " << worst
              << ")" << (deviceFailed ? " FAILED" : " passed") << " (tolerance " << tolerance << ")\n";
    failed = failed || deviceFailed;
  }
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
