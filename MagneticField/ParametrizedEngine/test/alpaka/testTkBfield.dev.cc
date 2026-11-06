// Compares the OAE parametrized field evaluated in a kernel (magfieldparam::inTeslaOAE on a copy of TkBfield::bcycl())
// with the host OAEParametrizedMagneticField on random points inside and around its validity volume.
#include <algorithm>
#include <cmath>
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
#include "MagneticField/ParametrizedEngine/interface/TkBfield.h"
#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct Point {
  float x, y, z;
};

struct Field {
  float bx, by, bz;
  bool defined;
};

struct OAEFieldKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                magfieldparam::BCycl<float> bcycl,
                                Point const* __restrict__ points,
                                Field* __restrict__ fields,
                                uint32_t size) const {
    for (auto index : cms::alpakatools::uniform_elements(acc, size)) {
      Point const& point = points[index];
      float field[3];
      magfieldparam::inTeslaOAE(bcycl, point.x, point.y, point.z, field);
      fields[index] = Field{
          field[0], field[1], field[2], magfieldparam::isDefinedOAE(point.x * point.x + point.y * point.y, point.z)};
    }
  }
};

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // the host field the Phase-2 menus use inside the tracker volume (slave of the volume-based field)
  OAEParametrizedMagneticField const hostField("3_8T");
  magfieldparam::TkBfield const tkBfield("3_8T");

  // uniform in r, phi, z over a volume slightly larger than the validity volume (r < 115 cm, |z| < 280 cm)
  constexpr uint32_t size = 1 << 20;
  std::mt19937 generator(12345);
  std::uniform_real_distribution<float> radius(0.f, 120.f), phi(-M_PI, M_PI), zeta(-290.f, 290.f);
  auto points_host = cms::alpakatools::make_host_buffer<Point[], Platform>(size);
  for (uint32_t i = 0; i < size; ++i) {
    float const rho = radius(generator), angle = phi(generator);
    points_host[i] = Point{rho * std::cos(angle), rho * std::sin(angle), zeta(generator)};
  }
  auto fields_host = cms::alpakatools::make_host_buffer<Field[], Platform>(size);

  bool failed = false;
  for (auto const& device : devices) {
    Queue queue(device);
    auto points_dev = cms::alpakatools::make_device_buffer<Point[]>(queue, size);
    auto fields_dev = cms::alpakatools::make_device_buffer<Field[]>(queue, size);
    alpaka::memcpy(queue, points_dev, points_host);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(size, 256), 256);
    alpaka::exec<Acc1D>(queue, workDiv, OAEFieldKernel{}, tkBfield.bcycl(), points_dev.data(), fields_dev.data(), size);
    alpaka::memcpy(queue, fields_host, fields_dev);
    alpaka::wait(queue);

    uint32_t nDefined = 0, nDefinedMismatch = 0, nIdentical = 0;
    float maxAbsDiff = 0.f, maxRelDiff = 0.f;
    for (uint32_t i = 0; i < size; ++i) {
      GlobalPoint const point(points_host[i].x, points_host[i].y, points_host[i].z);
      Field const& kernel = fields_host[i];
      bool const defined = hostField.isDefined(point);
      nDefinedMismatch += defined != kernel.defined;
      if (!defined)
        continue;
      ++nDefined;
      GlobalVector const host = hostField.inTeslaUnchecked(point);
      float const diffs[3] = {kernel.bx - host.x(), kernel.by - host.y(), kernel.bz - host.z()};
      nIdentical += diffs[0] == 0.f && diffs[1] == 0.f && diffs[2] == 0.f;
      for (float diff : diffs)
        maxAbsDiff = std::max(maxAbsDiff, std::abs(diff));
      maxRelDiff = std::max(maxRelDiff, GlobalVector(diffs[0], diffs[1], diffs[2]).mag() / host.mag());
    }
    std::cout << alpaka::getName(device) << ": " << nDefined << " points inside the validity volume, " << nIdentical
              << " bitwise identical to the host; max |dB| " << maxAbsDiff << " T, max |dB|/|B| " << maxRelDiff
              << "; validity mismatches " << nDefinedMismatch << std::endl;
    // float rounding only (FMA contraction, sqrt and exp implementations)
    if (nDefinedMismatch != 0 || maxRelDiff > 1e-5f || nDefined == 0)
      failed = true;
  }
  if (failed) {
    std::cerr << "FAILED" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
