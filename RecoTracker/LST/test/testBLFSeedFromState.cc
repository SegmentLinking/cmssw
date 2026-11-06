// Unit test of lst::blfStateOnPlane, lst::blfSeedFromState and lst::decodeStateHit, on the host, in a uniform field:
//  - stored states stepped onto barrel-like planes (tilted in azimuth or in z) and endcap-like planes: the position on
//    the plane and the path length against the helix of the same free state in double precision
//  - every rejection, a forced step failure with and without attaching the state, the seed's hits, detId and
//    direction, and the LogicErrors of a bad hit list or a propagator that is not any-direction
//  - the attached state against the free state attached to the surface without propagation, written out here: the
//    persistent state and the global state bit for bit, the double curvilinear error within 2 ulp
//  - the stateHit decoding

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "MagneticField/UniformEngine/interface/UniformMagneticField.h"
#include "RecoTracker/LST/interface/BLFCurvilinearJacobian.h"
#include "RecoTracker/LST/interface/BLFSeedFromState.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoTracker/LST/test/BLFTestStates.h"

namespace {

  using lst::BLFSeedStatus;

  constexpr int kNStates = 20000;
  constexpr uint32_t kSeed = 20261021;

  // AnalyticalPropagator returns the position on the plane in float: half an ulp of a coordinate up to 300 cm is
  // 1.8e-5 cm, 3.1e-5 cm in 3D. Its helix-plane crossing works in double from the float state and the float plane, as
  // the reference below does, so the rest of the deviation is a few roundings of the crossing. The tolerances are
  // three times the 3D output rounding.
  constexpr double kPositionTolerance = 1.e-4;  // cm, 3D distance from the reference crossing
  // the path length is a double, not rounded on output, but the barrel-plane crossing takes the distance to the plane
  // from the float Plane::localZ, hence the scale of the position tolerance
  constexpr double kPathTolerance = 1.e-4;  // cm, signed path length
  // A plane tilted in z takes HelixArbitraryPlaneCrossing (TrackingTools/GeomPropagators/interface/
  // OptimalHelixPlaneCrossing.h:19-33), which iterates only until the point is within max(1e-4 cm, 5e-7 |plane
  // position|) of the plane (HelixArbitraryPlaneCrossing.cc:72-73, 230-231): along the helix that is that distance
  // over |cos| of the angle between the direction and the normal. Its tolerance adds this bound to the one above.
  constexpr double kArbitraryPlaneMaxDistance = 1.e-4;  // cm
  constexpr double kArbitraryPlanePrecision = 5.e-7;    // relative to |plane position|

  // plane offsets from the state point: barrel-like along the plane normal (radial, turned in azimuth or tilted in
  // z), from 2 um (the barrel bulk) to 1.5 cm; endcap-like along z, cm-scale as the endcap steps
  constexpr double kBarrelOffsets[] = {2.e-4, 0.05, -0.3, 1.5};
  constexpr double kBarrelTilt = 0.15;  // rad, azimuth of the barrel-like normal relative to the radial direction
  // rad, angle of the normal out of the transverse plane, (cos a cos phi, cos a sin phi, sin a) as the tilted barrel
  // PS modules, a of the sign of z
  constexpr double kBarrelTiltsInZ[] = {0.3, 0.8};
  constexpr double kEndcapOffsets[] = {0.3, -2., 6.};
  // the reference crossing is unique near the state for these cuts: the transverse direction (the 3D direction for a
  // normal tilted in z) at least 0.3 along the barrel-like normal, |cot| > 0.1 for the endcap-like planes, and under
  // 0.5 rad of turn (barrel) or 1 rad (endcap), below the propagator's limit on the turn of a step (1.6 rad)
  constexpr double kMinNormalProjection = 0.3, kMinEndcapCot = 0.1, kMaxBarrelTurn = 0.5, kMaxEndcapTurn = 1.;

  // a step failure: a barrel-like plane at this distance from the origin, beyond the reach (2 R + 2 cm) of every
  // generated helix with pT below kMaxPtUnreachable
  constexpr double kFarPlaneDistance = 1.e4;  // cm
  constexpr double kMaxPtUnreachable = 20.;   // GeV, 2 R = 3.5e3 cm

  class TestDet : public GeomDet {
  public:
    TestDet(Plane::PlanePointer const& plane, uint32_t rawId) : GeomDet(plane) { setDetId(DetId(rawId)); }
  };

  // a rec hit that only carries its detector and a tag in its local x
  class TestHit : public TrackingRecHit {
  public:
    TestHit(GeomDet const& det, float tag) : TrackingRecHit(det), tag_(tag) {}
    TestHit(DetId id, float tag) : TrackingRecHit(id), tag_(tag) {}
    TestHit* clone() const override { return new TestHit(*this); }
    AlgebraicVector parameters() const override { return AlgebraicVector(); }
    AlgebraicSymMatrix parametersError() const override { return AlgebraicSymMatrix(); }
    AlgebraicMatrix projectionMatrix() const override { return AlgebraicMatrix(); }
    int dimension() const override { return 2; }
    std::vector<const TrackingRecHit*> recHits() const override { return {}; }
    std::vector<TrackingRecHit*> recHits() override { return {}; }
    LocalPoint localPosition() const override { return LocalPoint(tag_, 0.f, 0.f); }
    LocalError localPositionError() const override { return LocalError(); }

  private:
    float tag_;
  };

  // a plane through point with unit normal n; local x and y complete a right-handed frame
  Plane::PlanePointer makePlane(double const point[3], double const n[3]) {
    // local x: horizontal for a barrel-like plane, global x for an endcap-like one
    double x[3] = {-n[1], n[0], 0.};
    double norm = std::hypot(x[0], x[1]);
    if (norm < 1e-9) {
      x[0] = 1.;
      x[1] = 0.;
      norm = 1.;
    }
    for (double& c : x)
      c /= norm;
    const double y[3] = {n[1] * x[2] - n[2] * x[1], n[2] * x[0] - n[0] * x[2], n[0] * x[1] - n[1] * x[0]};
    return Plane::build(GlobalPoint(point[0], point[1], point[2]),
                        Surface::RotationType(x[0], x[1], x[2], y[0], y[1], y[2], n[0], n[1], n[2]));
  }

  // The free state attached to the surface without propagation, written out here as the reference of the attached
  // state: the float momentum bField / |kappa| (cos phi, sin phi, cotTheta) and lst::blfToCurvilinearCovariance.
  FreeTrajectoryState unpropagatedFreeState(lst::BLFStateRow const& row,
                                            float bField,
                                            MagneticField const& magneticField) {
    double covariance[5][5];
    ::lst::blfToCurvilinearCovariance(row.kappa, row.cotTheta, bField, row.covCircle, row.covLine, covariance);
    AlgebraicSymMatrix55 curvilinear;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        curvilinear(i, j) = covariance[i][j];
    const float pt = bField / std::abs(row.kappa);
    const FreeTrajectoryState freeState(
        GlobalTrajectoryParameters(GlobalPoint(row.x, row.y, row.z),
                                   GlobalVector(pt * std::cos(row.phi), pt * std::sin(row.phi), pt * row.cotTheta),
                                   row.charge,
                                   &magneticField),
        CurvilinearTrajectoryError(curvilinear));
    return freeState;
  }

  PTrajectoryStateOnDet unpropagatedSeedState(lst::BLFStateRow const& row,
                                              float bField,
                                              TrackingRecHit const& lastHit,
                                              MagneticField const& magneticField) {
    return trajectoryStateTransform::persistentState(
        TrajectoryStateOnSurface(unpropagatedFreeState(row, bField, magneticField), lastHit.det()->surface()),
        lastHit.geographicalId().rawId());
  }

  bool sameBits(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }
  bool sameBits(double a, double b) { return std::bit_cast<uint64_t>(a) == std::bit_cast<uint64_t>(b); }

  bool samePersistentState(PTrajectoryStateOnDet const& a, PTrajectoryStateOnDet const& b) {
    auto const& pa = a.parameters();
    auto const& pb = b.parameters();
    bool same = a.detId() == b.detId() && a.surfaceSide() == b.surfaceSide() && sameBits(a.pt(), b.pt()) &&
                sameBits(pa.qbp(), pb.qbp()) && sameBits(pa.dxdz(), pb.dxdz()) && sameBits(pa.dydz(), pb.dydz()) &&
                sameBits(pa.position().x(), pb.position().x()) && sameBits(pa.position().y(), pb.position().y()) &&
                sameBits(pa.pzSign(), pb.pzSign()) && pa.charge() == pb.charge() && a.hasError() == b.hasError();
    for (int i = 0; i < 15; ++i)
      same = same && sameBits(a.error(i), b.error(i));
    return same;
  }

  // the global state and its curvilinear error, bit for bit
  bool sameFreeState(FreeTrajectoryState const& a, FreeTrajectoryState const& b) {
    bool same = a.charge() == b.charge() && a.hasError() == b.hasError();
    same = same && sameBits(a.position().x(), b.position().x()) && sameBits(a.position().y(), b.position().y()) &&
           sameBits(a.position().z(), b.position().z()) && sameBits(a.momentum().x(), b.momentum().x()) &&
           sameBits(a.momentum().y(), b.momentum().y()) && sameBits(a.momentum().z(), b.momentum().z());
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        same = same && sameBits(a.curvilinearError().matrix()(i, j), b.curvilinearError().matrix()(i, j));
    return same;
  }

  // the global position, momentum and charge, bit for bit
  bool sameGlobalState(FreeTrajectoryState const& a, FreeTrajectoryState const& b) {
    return a.charge() == b.charge() && sameBits(a.position().x(), b.position().x()) &&
           sameBits(a.position().y(), b.position().y()) && sameBits(a.position().z(), b.position().z()) &&
           sameBits(a.momentum().x(), b.momentum().x()) && sameBits(a.momentum().y(), b.momentum().y()) &&
           sameBits(a.momentum().z(), b.momentum().z());
  }

  // The curvilinear error element by element within 2 ulp of double relative to sqrt(C_ii C_jj) of the reference b
  // (nothing else is relaxed: the library and this test compile the same expression in two translation units, and
  // the compiler may fuse multiply-adds differently, so the double intermediate can differ by an ulp while the
  // stored state does not). The largest difference, in units of that bound per ulp, goes to maxUlps.
  constexpr double kCurvilinearUlps = 2.;
  bool curvilinearWithinUlps(FreeTrajectoryState const& a, FreeTrajectoryState const& b, double& maxUlps) {
    constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
    auto const& ma = a.curvilinearError().matrix();
    auto const& mb = b.curvilinearError().matrix();
    bool within = a.hasError() == b.hasError();
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j) {
        const double scale = kEpsilon * std::sqrt(mb(i, i) * mb(j, j));
        const double difference = std::abs(ma(i, j) - mb(i, j));
        maxUlps = std::max(maxUlps, difference / scale);
        within = within && difference <= kCurvilinearUlps * scale;
      }
    return within;
  }

  // Where two free states differ, for the report: states with a different position or momentum bit, per upper-triangle
  // curvilinear element the states with a different bit, and the largest difference relative to sqrt(C_ii C_jj).
  struct FreeStateDifferences {
    int position = 0, momentum = 0, curvilinear[5][5] = {};
    double maxRelative = 0.;
    void add(FreeTrajectoryState const& a, FreeTrajectoryState const& b) {
      if (!(sameBits(a.position().x(), b.position().x()) && sameBits(a.position().y(), b.position().y()) &&
            sameBits(a.position().z(), b.position().z())))
        ++position;
      if (!(sameBits(a.momentum().x(), b.momentum().x()) && sameBits(a.momentum().y(), b.momentum().y()) &&
            sameBits(a.momentum().z(), b.momentum().z())))
        ++momentum;
      auto const& ma = a.curvilinearError().matrix();
      auto const& mb = b.curvilinearError().matrix();
      for (int i = 0; i < 5; ++i)
        for (int j = i; j < 5; ++j)
          if (!sameBits(ma(i, j), mb(i, j))) {
            ++curvilinear[i][j];
            maxRelative = std::max(maxRelative, std::abs(ma(i, j) - mb(i, j)) / std::sqrt(ma(i, i) * ma(j, j)));
          }
    }
  };

  // The helix of a free state in double: start point, transverse direction phi0, cot(theta) and CMSSW's transverse
  // curvature w = dphi / ds_T, all from the float values the propagator starts from.
  struct Helix {
    double x0[3], phi0, cot, w;

    explicit Helix(FreeTrajectoryState const& fts)
        : x0{fts.position().x(), fts.position().y(), fts.position().z()},
          phi0(std::atan2(double(fts.momentum().y()), double(fts.momentum().x()))),
          cot(fts.momentum().z() / std::hypot(double(fts.momentum().x()), double(fts.momentum().y()))),
          w(fts.transverseCurvature()) {}

    // the position after the transverse arc length s; the chord along the mean azimuth avoids the cancellation of
    // sin(phi) - sin(phi0)
    void at(double s, double r[3]) const {
      const double halfTurn = 0.5 * w * s;
      const double chord = halfTurn != 0. ? s * std::sin(halfTurn) / halfTurn : s;
      r[0] = x0[0] + chord * std::cos(phi0 + halfTurn);
      r[1] = x0[1] + chord * std::sin(phi0 + halfTurn);
      r[2] = x0[2] + cot * s;
    }

    // the transverse arc length to the plane (point p, normal n) by Newton from s0; false if it does not converge
    bool crossing(double const p[3], double const n[3], double s0, double& s) const {
      s = s0;
      for (int it = 0; it < 50; ++it) {
        double r[3];
        at(s, r);
        const double f = (r[0] - p[0]) * n[0] + (r[1] - p[1]) * n[1] + (r[2] - p[2]) * n[2];
        const double phi = phi0 + w * s;
        const double df = std::cos(phi) * n[0] + std::sin(phi) * n[1] + cot * n[2];
        const double ds = -f / df;
        s += ds;
        if (std::abs(ds) < 1e-13)
          return true;
      }
      return false;
    }
  };

  struct MaxDeviation {
    double value = 0.;
    void update(double deviation) {
      if (!(deviation <= value))  // a NaN is kept
        value = deviation;
    }
  };

  // the first generated state with a pT in the given range, |cot(theta)| < 2 and a transverse direction within 37
  // degrees of the radial direction, so that barrel-like planes near the state point are reached
  blfTestStates::TestState pickState(std::vector<blfTestStates::TestState> const& states, double ptMin, double ptMax) {
    for (auto const& state : states) {
      auto const& row = state.row;
      const double radial = std::cos(row.phi - std::atan2(row.y, row.x));
      if (state.pt >= ptMin && state.pt < ptMax && std::abs(state.cot) < 2. && radial > 0.8)
        return state;
    }
    throw edm::Exception(edm::errors::LogicError) << "no generated state in the requested range";
  }

  // a barrel-like plane at distance offset from the state point along its normal, the normal tilted by tilt from the
  // radial direction of the point
  Plane::PlanePointer barrelPlane(lst::BLFStateRow const& row, double offset, double tilt) {
    const double azimuth = std::atan2(double(row.y), double(row.x)) + tilt;
    const double n[3] = {std::cos(azimuth), std::sin(azimuth), 0.};
    const double point[3] = {row.x + offset * n[0], row.y + offset * n[1], double(row.z)};
    return makePlane(point, n);
  }

  // a barrel-like plane at distance offset from the state point along its normal, the normal radial in azimuth and
  // tilted in z by tiltInZ, with the sign of z of the point
  Plane::PlanePointer barrelPlaneTiltedInZ(lst::BLFStateRow const& row, double offset, double tiltInZ) {
    const double azimuth = std::atan2(double(row.y), double(row.x));
    const double a = std::copysign(tiltInZ, double(row.z));
    const double n[3] = {std::cos(a) * std::cos(azimuth), std::cos(a) * std::sin(azimuth), std::sin(a)};
    const double point[3] = {row.x + offset * n[0], row.y + offset * n[1], row.z + offset * n[2]};
    return makePlane(point, n);
  }

  enum class PlaneKind { kBarrel, kBarrelTiltedInZ, kEndcap };

}  // namespace

TEST_CASE("BLF seed from state: helix step onto planes", "[BLFSeedFromState]") {
  const UniformMagneticField field(blfTestStates::kFieldTesla);
  const float bField = static_cast<float>(1. / field.inverseBzAtOriginInGeV());
  const AnalyticalPropagator propagator(&field, anyDirection);
  const auto states = blfTestStates::makeStates(kNStates, kSeed);

  MaxDeviation barrelPosition, barrelPath, endcapPosition, endcapPath;
  int nBarrel = 0, nEndcap = 0, nNotConverged = 0;
  double maxBarrelStep = 0., maxEndcapStep = 0.;
  // planes tilted in z: the deviations, their ratios to the per-step tolerance, and the largest tolerance
  MaxDeviation tiltedPosition, tiltedPath, tiltedPositionRatio, tiltedPathRatio;
  int nTilted = 0;
  double maxTiltedStep = 0., maxTiltedPositionTolerance = 0.;
  for (auto const& state : states) {
    auto const& row = state.row;
    const FreeTrajectoryState fts = unpropagatedFreeState(row, bField, field);
    const Helix helix(fts);
    const double directionT[2] = {std::cos(helix.phi0), std::sin(helix.phi0)};
    const double cosLambda = 1. / std::sqrt(1. + helix.cot * helix.cot);

    auto check = [&](Plane const& plane, double s0, PlaneKind kind) {
      const double p[3] = {plane.position().x(), plane.position().y(), plane.position().z()};
      const double n[3] = {plane.normalVector().x(), plane.normalVector().y(), plane.normalVector().z()};
      double s;
      if (!helix.crossing(p, n, s0, s)) {
        ++nNotConverged;
        return;
      }
      double reference[3];
      helix.at(s, reference);
      const double referencePath = s / cosLambda;
      const auto result = lst::blfStateOnPlane(row, 14, bField, plane, field, propagator, false);
      CAPTURE(state.pt, state.cot, row.x, row.y, row.z, int(kind), s);
      REQUIRE(result.status == BLFSeedStatus::kOk);
      REQUIRE(result.state.isValid());
      CHECK(&result.state.surface() == &plane);
      const GlobalPoint position = result.state.globalPosition();
      const double distance =
          std::hypot(position.x() - reference[0], position.y() - reference[1], position.z() - reference[2]);
      const double pathDeviation = std::abs(result.pathLength - referencePath);
      if (kind == PlaneKind::kBarrelTiltedInZ) {
        // the stopping distance of HelixArbitraryPlaneCrossing along the helix, with the direction at the crossing
        const double phi = helix.phi0 + helix.w * s;
        const double cosine = (std::cos(phi) * n[0] + std::sin(phi) * n[1] + helix.cot * n[2]) * cosLambda;
        const double stop = std::max(kArbitraryPlaneMaxDistance,
                                     kArbitraryPlanePrecision * std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])) /
                            std::abs(cosine);
        const double positionTolerance = kPositionTolerance + stop, pathTolerance = kPathTolerance + stop;
        CAPTURE(cosine, positionTolerance);
        CHECK(distance < positionTolerance);
        CHECK(pathDeviation < pathTolerance);
        ++nTilted;
        tiltedPosition.update(distance);
        tiltedPath.update(pathDeviation);
        tiltedPositionRatio.update(distance / positionTolerance);
        tiltedPathRatio.update(pathDeviation / pathTolerance);
        maxTiltedStep = std::max(maxTiltedStep, std::abs(referencePath));
        maxTiltedPositionTolerance = std::max(maxTiltedPositionTolerance, positionTolerance);
        return;
      }
      CHECK(distance < kPositionTolerance);
      CHECK(pathDeviation < kPathTolerance);
      if (kind == PlaneKind::kBarrel) {
        ++nBarrel;
        barrelPosition.update(distance);
        barrelPath.update(pathDeviation);
        maxBarrelStep = std::max(maxBarrelStep, std::abs(referencePath));
      } else {
        ++nEndcap;
        endcapPosition.update(distance);
        endcapPath.update(pathDeviation);
        maxEndcapStep = std::max(maxEndcapStep, std::abs(referencePath));
      }
    };

    for (double offset : kBarrelOffsets) {
      const auto plane = barrelPlane(row, offset, kBarrelTilt);
      const double projection = directionT[0] * plane->normalVector().x() + directionT[1] * plane->normalVector().y();
      if (std::abs(projection) < kMinNormalProjection)
        continue;
      const double s0 = offset / projection;
      if (std::abs(helix.w * s0) > kMaxBarrelTurn)
        continue;
      check(*plane, s0, PlaneKind::kBarrel);
    }
    for (double tiltInZ : kBarrelTiltsInZ)
      for (double offset : kBarrelOffsets) {
        const auto plane = barrelPlaneTiltedInZ(row, offset, tiltInZ);
        // the rate of approach to the plane per unit transverse arc length; times cosLambda, the cosine of the 3D
        // direction with the normal
        const double projection = directionT[0] * plane->normalVector().x() +
                                  directionT[1] * plane->normalVector().y() + helix.cot * plane->normalVector().z();
        if (std::abs(projection * cosLambda) < kMinNormalProjection)
          continue;
        const double s0 = offset / projection;
        if (std::abs(helix.w * s0) > kMaxBarrelTurn)
          continue;
        check(*plane, s0, PlaneKind::kBarrelTiltedInZ);
      }
    if (std::abs(helix.cot) > kMinEndcapCot) {
      for (double offset : kEndcapOffsets) {
        const double s0 = offset / helix.cot;
        if (std::abs(helix.w * s0) > kMaxEndcapTurn)
          continue;
        const double n[3] = {0., 0., 1.};
        const double point[3] = {0., 0., row.z + offset};
        check(*makePlane(point, n), s0, PlaneKind::kEndcap);
      }
    }
  }
  CHECK(nNotConverged == 0);
  CHECK(nBarrel > 10000);
  CHECK(nEndcap > 10000);
  CHECK(nTilted > 10000);
  std::cout << "BLF seed from state, helix step: " << kNStates << " generated states; " << nBarrel
            << " barrel-like steps (|path| up to " << maxBarrelStep << " cm): max |position - reference| "
            << barrelPosition.value << " cm, max |path - reference| " << barrelPath.value << " cm; " << nEndcap
            << " endcap-like steps (|path| up to " << maxEndcapStep << " cm): " << endcapPosition.value << " cm, "
            << endcapPath.value << " cm (tolerances " << kPositionTolerance << ", " << kPathTolerance
            << " cm); reference not converged " << nNotConverged << std::endl;
  std::cout << "BLF seed from state, helix step: " << nTilted << " barrel-like steps tilted in z (|path| up to "
            << maxTiltedStep << " cm): max |position - reference| " << tiltedPosition.value
            << " cm, max |path - reference| " << tiltedPath.value << " cm; max ratio to the per-step tolerance "
            << tiltedPositionRatio.value << " (position), " << tiltedPathRatio.value
            << " (path); largest per-step position tolerance " << maxTiltedPositionTolerance << " cm" << std::endl;
}

TEST_CASE("BLF seed from state: rejections", "[BLFSeedFromState]") {
  const UniformMagneticField field(blfTestStates::kFieldTesla);
  const float bField = static_cast<float>(1. / field.inverseBzAtOriginInGeV());
  const AnalyticalPropagator propagator(&field, anyDirection);
  const auto states = blfTestStates::makeStates(1000, kSeed);
  const lst::BLFStateRow good = pickState(states, 1., 5.).row;
  const auto plane = barrelPlane(good, 0.01, kBarrelTilt);
  const TestDet det(plane, 0x14000010u);
  const TestHit hit(det, 1.f);
  const TrackingRecHit* hits[] = {&hit};

  int nCases = 0;
  auto expect = [&](lst::BLFStateRow const& row, uint8_t nFit, float b, BLFSeedStatus expected, const char* what) {
    ++nCases;
    CAPTURE(what);
    for (bool attach : {false, true}) {
      const auto onPlane = lst::blfStateOnPlane(row, nFit, b, *plane, field, propagator, attach);
      CHECK(onPlane.status == expected);
      CHECK(onPlane.pathLength == 0.);
      CHECK(!onPlane.state.isValid());
      const auto seed = lst::blfSeedFromState(row, nFit, b, hits, field, propagator, attach);
      CHECK(seed.status == expected);
      CHECK(seed.pathLength == 0.);
      CHECK(!seed.seed.has_value());
    }
  };

  // the valid row is accepted
  {
    const auto seed = lst::blfSeedFromState(good, 14, bField, hits, field, propagator, false);
    REQUIRE(seed.status == BLFSeedStatus::kOk);
    REQUIRE(seed.seed.has_value());
  }

  expect(good, 0, bField, BLFSeedStatus::kNotFitted, "nFit = 0");
  // the unfit row as the device writes it, nFit = 0 and a NaN kappa: nFit is tested before lst::checkState
  {
    lst::BLFStateRow row = good;
    row.kappa = std::numeric_limits<float>::quiet_NaN();
    expect(row, 0, bField, BLFSeedStatus::kNotFitted, "unfit row");
  }

  // each element non-finite: NaN, +inf, -inf
  const float nonFinite[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
  for (int element = 0; element < 15; ++element)
    for (float value : nonFinite) {
      lst::BLFStateRow row = good;
      float* fields[15] = {&row.x,
                           &row.y,
                           &row.z,
                           &row.phi,
                           &row.cotTheta,
                           &row.kappa,
                           &row.covCircle[0],
                           &row.covCircle[1],
                           &row.covCircle[2],
                           &row.covCircle[3],
                           &row.covCircle[4],
                           &row.covCircle[5],
                           &row.covLine[0],
                           &row.covLine[1],
                           &row.covLine[2]};
      *fields[element] = value;
      CAPTURE(element, value);
      expect(row, 14, bField, BLFSeedStatus::kNonFinite, "non-finite element");
    }

  // charge mismatch: opposite charge, kappa = +-0 with either charge, charge 0
  {
    lst::BLFStateRow row = good;
    row.charge = -row.charge;
    expect(row, 14, bField, BLFSeedStatus::kChargeMismatch, "sign(kappa) != charge");
    for (float zero : {0.f, -0.f})
      for (int8_t charge : {int8_t(-1), int8_t(1)}) {
        row = good;
        row.kappa = zero;
        row.charge = charge;
        expect(row, 14, bField, BLFSeedStatus::kChargeMismatch, "kappa = 0");
      }
    row = good;
    row.charge = 0;
    expect(row, 14, bField, BLFSeedStatus::kChargeMismatch, "charge 0");
  }

  // each native variance zero, negative zero or negative
  for (int variance = 0; variance < 5; ++variance)
    for (float value : {0.f, -0.f, -1.e-6f}) {
      lst::BLFStateRow row = good;
      float* variances[5] = {&row.covCircle[0], &row.covCircle[2], &row.covCircle[5], &row.covLine[0], &row.covLine[2]};
      *variances[variance] = value;
      CAPTURE(variance, value);
      expect(row, 14, bField, BLFSeedStatus::kNonPositiveVariance, "native variance <= 0");
    }

  // the curvilinear guard: a row passing lst::checkState gives positive finite curvilinear variances for every finite
  // positive field value (the double products of float inputs neither overflow nor underflow), so only a field value
  // that is zero, negative zero, infinite or NaN reaches it
  for (float b : {0.f,
                  -0.f,
                  std::numeric_limits<float>::infinity(),
                  -std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::quiet_NaN()}) {
    CAPTURE(b);
    expect(good, 14, b, BLFSeedStatus::kCurvilinearErrorNotPositive, "curvilinear guard");
  }

  // a bad hit list is a LogicError
  {
    const std::vector<TrackingRecHit const*> empty;
    CHECK_THROWS_AS(lst::blfSeedFromState(good, 14, bField, empty, field, propagator, true), edm::Exception);
    const TrackingRecHit* withNull[] = {nullptr, &hit};
    CHECK_THROWS_AS(lst::blfSeedFromState(good, 14, bField, withNull, field, propagator, true), edm::Exception);
    const TestHit noDet(DetId(0x14000011u), 2.f);
    const TrackingRecHit* lastWithoutDet[] = {&hit, &noDet};
    CHECK_THROWS_AS(lst::blfSeedFromState(good, 14, bField, lastWithoutDet, field, propagator, true), edm::Exception);
  }

  // a propagator that is not any-direction is a LogicError
  for (PropagationDirection direction : {alongMomentum, oppositeToMomentum}) {
    CAPTURE(int(direction));
    const AnalyticalPropagator oneWay(&field, direction);
    CHECK_THROWS_AS(lst::blfStateOnPlane(good, 14, bField, *plane, field, oneWay, false), edm::Exception);
    CHECK_THROWS_AS(lst::blfSeedFromState(good, 14, bField, hits, field, oneWay, true), edm::Exception);
  }
  std::cout << "BLF seed from state, rejections: " << nCases << " rejected rows, each with and without attach"
            << std::endl;
}

TEST_CASE("BLF seed from state: the seed", "[BLFSeedFromState]") {
  const UniformMagneticField field(blfTestStates::kFieldTesla);
  const float bField = static_cast<float>(1. / field.inverseBzAtOriginInGeV());
  const AnalyticalPropagator propagator(&field, anyDirection);
  const auto states = blfTestStates::makeStates(1000, kSeed);
  const lst::BLFStateRow row = pickState(states, 1., 5.).row;

  // three hits inside out on three detectors, the state's hit last on a plane 0.3 cm outside the state point
  const auto innerPlane = barrelPlane(row, -10., 0.);
  const auto middlePlane = barrelPlane(row, -5., 0.);
  const auto lastPlane = barrelPlane(row, 0.3, kBarrelTilt);
  const TestDet inner(innerPlane, 0x14000001u), middle(middlePlane, 0x14000002u), last(lastPlane, 0x14000003u);
  const TestHit hit0(inner, 0.5f), hit1(middle, 1.5f), hit2(last, 2.5f);
  const std::vector<TrackingRecHit const*> hits = {&hit0, &hit1, &hit2};

  const auto onPlane = lst::blfStateOnPlane(row, 14, bField, *lastPlane, field, propagator, false);
  REQUIRE(onPlane.status == BLFSeedStatus::kOk);
  const auto result = lst::blfSeedFromState(row, 14, bField, hits, field, propagator, false);
  REQUIRE(result.status == BLFSeedStatus::kOk);
  REQUIRE(result.seed.has_value());
  CHECK(sameBits(result.pathLength, onPlane.pathLength));
  CHECK(result.pathLength > 0.);
  TrajectorySeed const& seed = *result.seed;
  CHECK(seed.direction() == alongMomentum);
  CHECK(seed.startingState().detId() == 0x14000003u);
  CHECK(
      samePersistentState(seed.startingState(), trajectoryStateTransform::persistentState(onPlane.state, 0x14000003u)));
  REQUIRE(seed.nHits() == 3);
  int i = 0;
  for (auto const& seedHit : seed.recHits()) {
    CAPTURE(i);
    CHECK(&seedHit != hits[i]);  // a clone
    CHECK(seedHit.geographicalId() == hits[i]->geographicalId());
    CHECK(seedHit.det() == hits[i]->det());
    CHECK(seedHit.localPosition().x() == hits[i]->localPosition().x());
    ++i;
  }

  // a step that fails: without attach no seed; with attach the seed of the free state on the last hit's surface
  const auto farPlane = barrelPlane(row, kFarPlaneDistance, 0.);
  const TestDet far(farPlane, 0x14000004u);
  const TestHit farHit(far, 3.5f);
  const std::vector<TrackingRecHit const*> farHits = {&hit0, &hit1, &farHit};
  const auto failed = lst::blfSeedFromState(row, 14, bField, farHits, field, propagator, false);
  CHECK(failed.status == BLFSeedStatus::kStepFailed);
  CHECK(failed.pathLength == 0.);
  CHECK(!failed.seed.has_value());
  const auto attached = lst::blfSeedFromState(row, 14, bField, farHits, field, propagator, true);
  CHECK(attached.status == BLFSeedStatus::kAttachedAfterFailedStep);
  CHECK(attached.pathLength == 0.);
  REQUIRE(attached.seed.has_value());
  CHECK(attached.seed->direction() == alongMomentum);
  CHECK(attached.seed->startingState().detId() == 0x14000004u);
  REQUIRE(attached.seed->nHits() == 3);
  CHECK(std::prev(attached.seed->recHits().end())->geographicalId() == farHit.geographicalId());
  CHECK(samePersistentState(attached.seed->startingState(), unpropagatedSeedState(row, bField, farHit, field)));

  // an endcap-like plane that a helix with cot(theta) = 0 never reaches
  {
    lst::BLFStateRow flat = row;
    flat.cotTheta = 0.f;
    const double n[3] = {0., 0., 1.};
    const double point[3] = {0., 0., row.z + 5.};
    const auto endcap = makePlane(point, n);
    const auto stepFailed = lst::blfStateOnPlane(flat, 14, bField, *endcap, field, propagator, false);
    CHECK(stepFailed.status == BLFSeedStatus::kStepFailed);
    const auto attachedFlat = lst::blfStateOnPlane(flat, 14, bField, *endcap, field, propagator, true);
    CHECK(attachedFlat.status == BLFSeedStatus::kAttachedAfterFailedStep);
    REQUIRE(attachedFlat.state.isValid());
    const auto unpropagatedFlat = unpropagatedFreeState(flat, bField, field);
    double flatUlps = 0.;
    CHECK(sameGlobalState(*attachedFlat.state.freeState(), unpropagatedFlat));
    CHECK(curvilinearWithinUlps(*attachedFlat.state.freeState(), unpropagatedFlat, flatUlps));
  }
}

TEST_CASE("BLF seed from state: attached state equals the unpropagated free state", "[BLFSeedFromState]") {
  const UniformMagneticField field(blfTestStates::kFieldTesla);
  const float bField = static_cast<float>(1. / field.inverseBzAtOriginInGeV());
  const AnalyticalPropagator propagator(&field, anyDirection);
  const auto states = blfTestStates::makeStates(kNStates, kSeed);

  int nCompared = 0, nDifferent = 0, nPersistentDifferent = 0;
  double maxUlps = 0.;
  FreeStateDifferences differences;
  for (auto const& state : states) {
    if (state.pt >= kMaxPtUnreachable)
      continue;
    auto const& row = state.row;
    const auto farPlane = barrelPlane(row, kFarPlaneDistance, 0.);
    const TestDet far(farPlane, 0x14000005u);
    const TestHit farHit(far, 1.f);
    const TrackingRecHit* hits[] = {&farHit};
    CAPTURE(state.pt, state.cot, row.x, row.y, row.z);
    const auto failed = lst::blfSeedFromState(row, 14, bField, hits, field, propagator, false);
    CHECK(failed.status == BLFSeedStatus::kStepFailed);
    const auto attached = lst::blfSeedFromState(row, 14, bField, hits, field, propagator, true);
    REQUIRE(attached.status == BLFSeedStatus::kAttachedAfterFailedStep);
    REQUIRE(attached.seed.has_value());
    const auto onPlane = lst::blfStateOnPlane(row, 14, bField, *farPlane, field, propagator, true);
    REQUIRE(onPlane.state.isValid());
    ++nCompared;
    const auto unpropagated = unpropagatedFreeState(row, bField, field);
    const bool samePersistent =
        samePersistentState(attached.seed->startingState(), unpropagatedSeedState(row, bField, farHit, field));
    const bool sameGlobal = sameGlobalState(*onPlane.state.freeState(), unpropagated);
    const bool curvilinearClose = curvilinearWithinUlps(*onPlane.state.freeState(), unpropagated, maxUlps);
    if (!(samePersistent && sameFreeState(*onPlane.state.freeState(), unpropagated)))
      ++nDifferent;
    if (!samePersistent)
      ++nPersistentDifferent;
    differences.add(*onPlane.state.freeState(), unpropagated);
    CHECK(samePersistent);
    CHECK(sameGlobal);
    CHECK(curvilinearClose);
  }
  CHECK(nCompared > 10000);
  std::cout << "BLF seed from state, attached after a failed step: " << nCompared
            << " states compared with the free state attached without propagation, " << nDifferent
            << " differ in any bit (" << nPersistentDifferent
            << " in the persistent state); free state: " << differences.position << " in position, "
            << differences.momentum << " in momentum, per curvilinear element (upper triangle, row by row):";
  for (int i = 0; i < 5; ++i)
    for (int j = i; j < 5; ++j)
      std::cout << ' ' << differences.curvilinear[i][j];
  std::cout << ", max |difference| / sqrt(C_ii C_jj) " << differences.maxRelative << ", i.e. " << maxUlps
            << " ulp (bound " << kCurvilinearUlps << ")" << std::endl;
}

TEST_CASE("BLF seed from state: stateHit decoding", "[BLFSeedFromState]") {
  using lst::StateHitKind;
  auto decoded = [](uint32_t stateHit, uint32_t nOT, uint32_t nPixel, StateHitKind kind, uint32_t index) {
    const auto d = lst::decodeStateHit(stateHit, nOT, nPixel);
    CAPTURE(stateHit, nOT, nPixel);
    CHECK(d.kind == kind);
    if (kind == StateHitKind::kOT || kind == StateHitKind::kPixel)
      CHECK(d.index == index);
  };
  // the sentinel first, whatever the ranges; it coincides with the encoding of pixel position 0x7FFFFFFF
  decoded(lst::kInvalidU32Idx, 0xFFFFFFFFu, 0xFFFFFFFFu, StateHitKind::kInvalid, 0);
  static_assert(lst::encodeStateHitPixel(0x7FFFFFFFu) == lst::kInvalidU32Idx);
  // OT
  decoded(lst::encodeStateHitOT(0), 1, 0, StateHitKind::kOT, 0);
  decoded(lst::encodeStateHitOT(7), 10, 0, StateHitKind::kOT, 7);
  decoded(lst::encodeStateHitOT(0x7FFFFFFFu), 0x80000000u, 0, StateHitKind::kOT, 0x7FFFFFFFu);
  decoded(lst::encodeStateHitOT(7), 7, 100, StateHitKind::kOutOfRange, 0);
  decoded(lst::encodeStateHitOT(0), 0, 100, StateHitKind::kOutOfRange, 0);
  // pixel
  decoded(lst::encodeStateHitPixel(0), 0, 1, StateHitKind::kPixel, 0);
  decoded(lst::encodeStateHitPixel(3), 0, 4, StateHitKind::kPixel, 3);
  decoded(lst::encodeStateHitPixel(0x7FFFFFFEu), 0, 0x7FFFFFFFu, StateHitKind::kPixel, 0x7FFFFFFEu);
  decoded(lst::encodeStateHitPixel(3), 100, 3, StateHitKind::kOutOfRange, 0);
  decoded(lst::encodeStateHitPixel(0), 100, 0, StateHitKind::kOutOfRange, 0);  // a row without a pixel track
  static_assert(lst::decodeStateHit(lst::encodeStateHitPixel(2), 0, 3).kind == StateHitKind::kPixel);
}
