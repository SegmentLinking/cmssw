// The header-only package on every backend: TkBfield and the vdt functions evaluated in a kernel of this binary and on
// the host, compared. The binary uses only RecoTracker/MkFitAlpakaMath (no MkFitAlpaka device code is linked in), so a
// failing launch of its own kernel would also show up here. Exit 0 = pass (or no device of the backend).
#include <cmath>
#include <cstdio>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpakaMath/interface/math/TkBfield.h"
#include "RecoTracker/MkFitAlpakaMath/interface/math/vdtMath.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  constexpr int kOut = 7;  // bx, by, bz, sin, cos, atan2, log

  ALPAKA_FN_HOST_ACC inline void evaluate(float x, float y, float z, float* o) {
    ::mkfitdev::field::tkBfield(x, y, z, o[0], o[1], o[2]);
    const float phi = 0.0137f * (x - y) + 0.1f * z;
    ::mkfitdev::vdt::fast_sincosf(phi, o[3], o[4]);
    o[5] = ::mkfitdev::vdt::fast_atan2f(y, x);
    o[6] = ::mkfitdev::vdt::fast_logf(1.f + x * x + y * y);
  }

  struct KernelEvaluate {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, float const* in, float* out, uint32_t n) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n))
        evaluate(in[3 * i], in[3 * i + 1], in[3 * i + 2], out + kOut * i);
    }
  };
}  // namespace

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::printf("no device for %s: skipped\n", EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE));
    return 0;
  }
  // points inside the tracker volume of the closed-form field (r < 115 cm, |z| < 280 cm)
  std::vector<float> pts;
  for (float r = 0.5f; r < 110.f; r += 7.3f)
    for (float ph = -3.1f; ph < 3.1f; ph += 0.37f)
      for (float z = -270.f; z < 270.f; z += 31.f) {
        pts.push_back(r * std::cos(ph));
        pts.push_back(r * std::sin(ph));
        pts.push_back(z);
      }
  const uint32_t n = pts.size() / 3;
  int nFail = 0;
  for (auto const& device : devices) {
    Queue queue(device);
    auto hIn = cms::alpakatools::make_host_buffer<float[]>(queue, 3 * n);
    auto hOut = cms::alpakatools::make_host_buffer<float[]>(queue, kOut * n);
    std::copy(pts.begin(), pts.end(), hIn.data());
    auto dIn = cms::alpakatools::make_device_buffer<float[]>(queue, 3 * n);
    auto dOut = cms::alpakatools::make_device_buffer<float[]>(queue, kOut * n);
    alpaka::memcpy(queue, dIn, hIn);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(n, 128u), 128u),
                        KernelEvaluate{},
                        dIn.data(),
                        dOut.data(),
                        n);
    alpaka::memcpy(queue, hOut, dOut);
    alpaka::wait(queue);
    float maxDiff = 0.f;
    for (uint32_t i = 0; i < n; ++i) {
      float ref[kOut];
      evaluate(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2], ref);
      for (int k = 0; k < kOut; ++k) {
        const float d = hOut.data()[kOut * i + k], e = std::abs(d - ref[k]) / std::max(1.f, std::abs(ref[k]));
        maxDiff = std::max(maxDiff, e);
        // the device may contract to FMA: a few ulp
        if (!std::isfinite(d) || e > 1e-5f) {
          if (++nFail < 10)
            std::printf("FAIL: point %u output %d: device %.9g host %.9g\n", i, k, d, ref[k]);
        }
      }
    }
    std::printf("%s: %u points, max relative difference device - host %.3g\n",
                EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE),
                n,
                maxDiff);
  }
  std::printf("testMkFitAlpakaMath: %s (%d failures)\n", nFail ? "FAIL" : "PASS", nFail);
  return nFail ? 1 : 0;
}
