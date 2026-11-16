// Unit test of the seeder's helix arithmetic on the backend it is built for: the propagation of a state onto a plane
// (MkSeederFitKernels.h) against the analytic helix, the conversion of a curvilinear state (MkSeederPcaKernels.h) and the
// closest approach to the beam line against the geometry of the circle.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/MkSeederAlpaka/plugins/alpaka/MkSeederFitKernels.h"
#include "RecoTracker/MkSeederAlpaka/plugins/alpaka/MkSeederPcaKernels.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  namespace seeder = ALPAKA_ACCELERATOR_NAMESPACE::mkseeder;
  constexpr float kField = ::mkfitdev::Config::Bfield;  // T (the propagation's without the parametrization)
  constexpr float kPtPerRadius = ::mkfitdev::Const::sol_over_100 * kField;  // GeV / cm

  // the seed fits' propagation (MkFitAlpaka's propagateHelixToPlaneMPlex, uniform field, no material) of one state
  // onto the plane (point, normal), through the fits' lane states: the output state and its covariance
  struct KernelPropagate {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const float* inPar,
                                  const float* inErr,
                                  int chg,
                                  const float* pnt,
                                  const float* nrm,
                                  float* outPar,
                                  float* outErr) const {
      if (!cms::alpakatools::once_per_grid(acc))
        return;
      namespace mkf = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev;
      seeder::fit::Lanes<1> in, out;
      in.load(0, inPar, inErr, chg);
      mkf::MPlexHV<1> p, n;
      for (int d = 0; d < 3; ++d)
        p(0, d, 0) = pnt[d], n(0, d, 0) = nrm[d];
      mkf::MPlexQI<1> fail;
      mkf::propagateHelixToPlaneMPlex<1>(
          in.err, in.par, in.chg, p, n, out.err, out.par, fail, 1, seeder::fit::PropagationFlags{});
      out.chg = in.chg;
      int q;
      out.store(0, outPar, outErr, q);
    }
  };

  // curvilinearToFreeState() and closestApproachToBeamline() of one state: res = p (3), cov(0,0), cov(1,1), the state
  // at the beam line: x (3), p (3), ok flags
  struct KernelFreeState {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const float* par,
                                  const float* err,
                                  int chg,
                                  float bz,
                                  const double* beamSpot,
                                  float* res) const {
      if (!cms::alpakatools::once_per_grid(acc))
        return;
      seeder::pca::FreeTrajectoryState f{}, b{};
      const bool ok0 = seeder::pca::curvilinearToFreeState(par, err, chg, f);
      res[0] = f.p.x, res[1] = f.p.y, res[2] = f.p.z, res[3] = f.C[0][0], res[4] = f.C[1][1];
      f.b = {0.f, 0.f, bz};
      const bool ok1 = ok0 && seeder::pca::closestApproachToBeamline(f, beamSpot, b);
      res[5] = b.x.x, res[6] = b.x.y, res[7] = b.x.z, res[8] = b.p.x, res[9] = b.p.y, res[10] = b.p.z;
      res[11] = ok0, res[12] = ok1;
    }
  };

  // runs a one-thread kernel on the device and returns the output buffers
  template <typename TKernel, typename... TArgs>
  void run(Queue& queue, TKernel kernel, TArgs&&... args) {
    alpaka::exec<Acc1D>(queue, cms::alpakatools::make_workdiv<Acc1D>(1u, 1u), kernel, std::forward<TArgs>(args)...);
  }

  template <typename T>
  auto toDevice(Queue& queue, T const* data, size_t n) {
    auto host = cms::alpakatools::make_host_buffer<T[]>(n);
    std::copy(data, data + n, host.data());
    auto device = cms::alpakatools::make_device_buffer<T[]>(queue, n);
    alpaka::memcpy(queue, device, host);
    return device;
  }

  template <typename T, typename TBuffer>
  std::vector<T> toHost(Queue& queue, TBuffer& device, size_t n) {
    auto host = cms::alpakatools::make_host_buffer<T[]>(n);
    alpaka::memcpy(queue, host, device);
    alpaka::wait(queue);
    return std::vector<T>(host.data(), host.data() + n);
  }

  // the state at the origin, direction phi = 0 in the transverse plane and polar angle theta, 1/pT = 1 / pt
  struct Helix {
    float pt, theta;
    int chg;
    float radius() const { return pt / kPtPerRadius; }
  };

  // runs the propagation of the helix onto the plane and returns {outPar (6), outErr (21)}
  std::vector<float> propagateOnDevice(Queue& queue, Helix const& h, const float* pnt, const float* nrm) {
    const float inPar[6] = {0.f, 0.f, 0.f, 1.f / h.pt, 0.f, h.theta};
    float inErr[21] = {};
    const float sigma[6] = {1e-2f, 1e-2f, 1e-2f, 1e-3f, 1e-3f, 1e-3f};
    for (int i = 0; i < 6; ++i)
      inErr[seeder::fit::sidx(i, i)] = sigma[i] * sigma[i];
    auto inParD = toDevice(queue, inPar, 6), inErrD = toDevice(queue, inErr, 21), pntD = toDevice(queue, pnt, 3),
         nrmD = toDevice(queue, nrm, 3);
    auto outParD = cms::alpakatools::make_device_buffer<float[]>(queue, 6);
    auto outErrD = cms::alpakatools::make_device_buffer<float[]>(queue, 21);
    run(queue,
        KernelPropagate{},
        inParD.data(),
        inErrD.data(),
        h.chg,
        pntD.data(),
        nrmD.data(),
        outParD.data(),
        outErrD.data());
    auto out = toHost<float>(queue, outParD, 6);
    auto err = toHost<float>(queue, outErrD, 21);
    out.insert(out.end(), err.begin(), err.end());
    return out;
  }

}  // namespace

TEST_CASE("Seeder helix arithmetic for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  if (cms::alpakatools::devices<Platform>().empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend");

  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    std::cout << "helix tests on " << alpaka::getName(device) << "\n";
    Queue queue(device);

    SECTION("the propagation onto a plane follows the analytic helix") {
      // a positive charge turns clockwise (y decreases), a negative one counter-clockwise
      for (int chg : {-1, 1}) {
        for (float theta : {1.5707963f, 1.2f, 0.9f}) {
          const Helix h{0.8f, theta, chg};
          const float R = h.radius();
          const float sinT = std::sin(theta), cosT = std::cos(theta);
          INFO("chg " << chg << " theta " << theta << " R " << R);

          // a plane x = xp (normal along x): the transverse angle at the plane is asin(x / R)
          {
            const float xp = 25.f;
            const float pnt[3] = {xp, 0.f, 0.f}, nrm[3] = {1.f, 0.f, 0.f};
            const auto r = propagateOnDevice(queue, h, pnt, nrm);
            const float delta = std::asin(xp / R);
            REQUIRE(r[0] == Catch::Approx(xp).margin(2e-3));
            REQUIRE(r[1] == Catch::Approx(-chg * R * (1.f - std::cos(delta))).margin(2e-3));
            REQUIRE(r[2] ==
                    Catch::Approx(R * delta * cosT / sinT).margin(5e-3));  // z = s cos(theta), s = R delta / sin
            REQUIRE(r[3] == Catch::Approx(1.f / h.pt).epsilon(1e-4));
            REQUIRE(r[4] == Catch::Approx(-chg * delta).margin(2e-4));
            REQUIRE(r[5] == Catch::Approx(theta).margin(1e-5));
            // the covariance: the 1/pT variance is conserved, the others stay finite and the matrix has a positive
            // diagonal
            REQUIRE(r[6 + seeder::fit::sidx(3, 3)] == Catch::Approx(1e-6f).epsilon(2e-2));
            for (int i = 0; i < 6; ++i)
              REQUIRE(r[6 + seeder::fit::sidx(i, i)] > 0.f);
          }
          // a plane z = zp (normal along z, not for the transverse plane's theta = pi/2)
          if (cosT > 0.1f) {
            const float zp = 15.f;  // the arc stays short: the path is refined once (as in the fit)
            const float pnt[3] = {0.f, 0.f, zp}, nrm[3] = {0.f, 0.f, 1.f};
            const auto r = propagateOnDevice(queue, h, pnt, nrm);
            const float s = zp / cosT, delta = s * sinT / R;
            REQUIRE(r[0] == Catch::Approx(R * std::sin(delta)).margin(2e-3));
            REQUIRE(r[1] == Catch::Approx(-chg * R * (1.f - std::cos(delta))).margin(2e-3));
            REQUIRE(r[2] == Catch::Approx(zp).margin(1e-3));
            REQUIRE(r[4] == Catch::Approx(-chg * delta).margin(2e-4));
          }
        }
      }
    }

    SECTION("the curvilinear state is converted to a free state and brought to the beam line") {
      const float R = 50.f, pz = 0.3f;
      const float pt = R * kPtPerRadius;
      const float theta = std::atan2(pt, pz);
      const float cx = 60.f, cy = 20.f;  // the circle's centre; the origin is outside the circle
      const float centreDistance = std::hypot(cx, cy);
      const float a0 = std::atan2(-cy, -cx);  // the direction from the centre to the closest point
      const double beamSpot[5] = {0., 0., 0., 0., 0.};
      for (int chg : {-1, 1}) {
        // the position 40 degrees from the closest point on the circle, the tangent along the motion
        const float a = a0 + 40.f * float(M_PI) / 180.f;
        const float x = cx + R * std::cos(a), y = cy + R * std::sin(a);
        const float tx = chg > 0 ? std::sin(a) : -std::sin(a), ty = chg > 0 ? -std::cos(a) : std::cos(a);
        const float phi = std::atan2(ty, tx);
        const float par[6] = {x, y, 5.f, 1.f / pt, phi, theta};
        float err[21] = {};
        const float sigma[6] = {1e-2f, 1e-2f, 1e-2f, 1e-3f, 2e-3f, 3e-3f};
        for (int i = 0; i < 6; ++i)
          err[seeder::fit::sidx(i, i)] = sigma[i] * sigma[i];
        auto parD = toDevice(queue, par, 6), errD = toDevice(queue, err, 21);
        auto bsD = toDevice(queue, beamSpot, 5);
        auto resD = cms::alpakatools::make_device_buffer<float[]>(queue, 13);
        run(queue, KernelFreeState{}, parD.data(), errD.data(), chg, kField, bsD.data(), resD.data());
        const auto res = toHost<float>(queue, resD, 13);
        INFO("chg " << chg);
        REQUIRE(res[11] == 1.f);
        // the momentum and the first covariance entries (q/p and lambda of the curvilinear frame)
        REQUIRE(res[0] == Catch::Approx(pt * std::cos(phi)).margin(1e-5));
        REQUIRE(res[1] == Catch::Approx(pt * std::sin(phi)).margin(1e-5));
        REQUIRE(res[2] == Catch::Approx(pz).epsilon(1e-4));
        const double sinT = std::sin(theta), cosT = std::cos(theta);
        const double covQopOverPt = double(sigma[3]) * sigma[3];
        const double expectedCov00 = sinT * sinT * covQopOverPt + cosT * cosT / (double(pt) * pt) * sigma[5] * sigma[5];
        REQUIRE(res[3] == Catch::Approx(expectedCov00).epsilon(1e-4));
        REQUIRE(res[4] == Catch::Approx(double(sigma[5]) * sigma[5]).epsilon(1e-4));  // lambda = pi/2 - theta
        // the closest approach: the point of the circle nearest to the beam line, the momentum perpendicular to it
        REQUIRE(res[12] == 1.f);
        const float f = 1.f - R / centreDistance;
        REQUIRE(res[5] == Catch::Approx(cx * f).margin(2e-3));
        REQUIRE(res[6] == Catch::Approx(cy * f).margin(2e-3));
        const float dca = centreDistance - R;
        REQUIRE(std::hypot(res[5], res[6]) == Catch::Approx(dca).margin(2e-3));
        REQUIRE(res[5] * res[8] + res[6] * res[9] == Catch::Approx(0.f).margin(2e-3 * pt * dca));
        REQUIRE(std::hypot(res[8], res[9]) == Catch::Approx(pt).epsilon(1e-4));
        REQUIRE(res[10] == Catch::Approx(pz).epsilon(1e-4));
      }
    }
  }
}
