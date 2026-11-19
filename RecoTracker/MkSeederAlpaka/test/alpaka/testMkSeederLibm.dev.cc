// Unit test of the seeder's portable libm (plugins/alpaka/MkSeederLibm.h): on the host, the ports of glibc's atan2f
// and asinf against the library; on the backend's device, libm::atan2 / asin / tan against the host's libm (the CPU
// backends run the ports, the GPU backends the device library and the port of tanf).

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/MkSeederAlpaka/plugins/alpaka/MkSeederLibm.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  namespace libm = ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::libm;

  // the distance of two floats in units in the last place (ordered integer representation)
  int64_t ulpDistance(float a, float b) {
    auto ordered = [](float x) {
      const int32_t i = std::bit_cast<int32_t>(x);
      return int64_t(i < 0 ? int32_t(0x80000000) - i : i);
    };
    return std::abs(ordered(a) - ordered(b));
  }

  struct KernelLibm {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const float* y,
                                  const float* x,
                                  const float* u,
                                  const float* t,
                                  float* atan2Out,
                                  float* asinOut,
                                  float* tanOut,
                                  uint32_t n) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        atan2Out[i] = libm::atan2(acc, y[i], x[i]);
        asinOut[i] = libm::asin(acc, u[i]);
        tanOut[i] = libm::tan(acc, t[i]);
      }
    }
  };

}  // namespace

TEST_CASE("Seeder portable libm for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  if (cms::alpakatools::devices<Platform>().empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend");

  constexpr uint32_t n = 20000;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> unit(-1.f, 1.f), angle(-100.f, 100.f), wide(-1.e3f, 1.e3f);
  std::vector<float> y(n), x(n), u(n), t(n);
  for (uint32_t i = 0; i < n; ++i) {
    // atan2: all quadrants, mixed magnitudes; asin on [-1, 1]; tan up to 2^7 pi / 2 (a seed's polar angles are small)
    const float scale = std::pow(10.f, 6.f * unit(rng));
    y[i] = wide(rng) * scale * 1e-3f, x[i] = wide(rng) * scale * 1e-3f;
    u[i] = unit(rng);
    t[i] = angle(rng);
  }
  // special arguments of the quadrant logic and of the end points
  y[0] = 0.f, x[0] = 1.f, y[1] = 0.f, x[1] = -1.f, y[2] = 1.f, x[2] = 0.f, y[3] = -1.f, x[3] = 0.f;
  y[4] = 1.f, x[4] = 1.f, y[5] = -1.f, x[5] = -1.f, y[6] = 1e-20f, x[6] = 1.f, y[7] = 1.f, x[7] = 1e-20f;
  u[0] = 1.f, u[1] = -1.f, u[2] = 0.f, u[3] = 0.5f, u[4] = 0.99f, u[5] = 1e-9f;
  t[0] = 0.f, t[1] = 0.785398f, t[2] = -0.785398f, t[3] = 1.5707963f;

  SECTION("the ports of glibc's atan2f and asinf on the host, within one ulp of the library") {
    int64_t worstAtan2 = 0, worstAsin = 0;
    for (uint32_t i = 0; i < n; ++i) {
      worstAtan2 = std::max(worstAtan2, ulpDistance(libm::glibc::atan2f(y[i], x[i]), std::atan2(y[i], x[i])));
      worstAsin = std::max(worstAsin, ulpDistance(libm::glibc::asinf(u[i]), std::asin(u[i])));
    }
    std::cout << "glibc ports on the host: max ulp atan2f " << worstAtan2 << ", asinf " << worstAsin << "\n";
    REQUIRE(worstAtan2 <= 1);
    REQUIRE(worstAsin <= 1);
  }

  SECTION("libm::atan2, asin and tan on the device against the host's libm") {
    for (auto const& device : cms::alpakatools::devices<Platform>()) {
      std::cout << "libm on " << alpaka::getName(device) << "\n";
      Queue queue(device);
      auto yH = cms::alpakatools::make_host_buffer<float[]>(n), xH = cms::alpakatools::make_host_buffer<float[]>(n),
           uH = cms::alpakatools::make_host_buffer<float[]>(n), tH = cms::alpakatools::make_host_buffer<float[]>(n),
           aH = cms::alpakatools::make_host_buffer<float[]>(n), sH = cms::alpakatools::make_host_buffer<float[]>(n),
           tanH = cms::alpakatools::make_host_buffer<float[]>(n);
      for (uint32_t i = 0; i < n; ++i)
        yH[i] = y[i], xH[i] = x[i], uH[i] = u[i], tH[i] = t[i];
      auto yD = cms::alpakatools::make_device_buffer<float[]>(queue, n),
           xD = cms::alpakatools::make_device_buffer<float[]>(queue, n),
           uD = cms::alpakatools::make_device_buffer<float[]>(queue, n),
           tD = cms::alpakatools::make_device_buffer<float[]>(queue, n),
           aD = cms::alpakatools::make_device_buffer<float[]>(queue, n),
           sD = cms::alpakatools::make_device_buffer<float[]>(queue, n),
           tanD = cms::alpakatools::make_device_buffer<float[]>(queue, n);
      alpaka::memcpy(queue, yD, yH);
      alpaka::memcpy(queue, xD, xH);
      alpaka::memcpy(queue, uD, uH);
      alpaka::memcpy(queue, tD, tH);
      const auto wd = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(n, 64u), 64u);
      alpaka::exec<Acc1D>(
          queue, wd, KernelLibm{}, yD.data(), xD.data(), uD.data(), tD.data(), aD.data(), sD.data(), tanD.data(), n);
      alpaka::memcpy(queue, aH, aD);
      alpaka::memcpy(queue, sH, sD);
      alpaka::memcpy(queue, tanH, tanD);
      alpaka::wait(queue);
      // the CPU backends run the ports of the host's library (the same values); the device libraries of the GPU
      // backends are accurate to a few ulp
      const int64_t tolerance = std::is_same_v<Device, alpaka::DevCpu> ? 1 : 4;
      int64_t worstAtan2 = 0, worstAsin = 0, worstTan = 0;
      for (uint32_t i = 0; i < n; ++i) {
        worstAtan2 = std::max(worstAtan2, ulpDistance(aH[i], std::atan2(y[i], x[i])));
        worstAsin = std::max(worstAsin, ulpDistance(sH[i], std::asin(u[i])));
        worstTan = std::max(worstTan, ulpDistance(tanH[i], std::tan(t[i])));
      }
      std::cout << "max ulp atan2 " << worstAtan2 << ", asin " << worstAsin << ", tan " << worstTan << "\n";
      REQUIRE(worstAtan2 <= tolerance);
      REQUIRE(worstAsin <= tolerance);
      REQUIRE(worstTan <= tolerance);
    }
  }
}
