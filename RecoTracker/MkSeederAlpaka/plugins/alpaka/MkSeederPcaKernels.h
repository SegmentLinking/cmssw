#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPcaKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPcaKernels_h

// The final seeds' states taken to the beam line on the device, one thread per seed, for the seed classifier's state
// features (MkSeederSeedFeaturesSoA.h) and the pixel tracks (the perigee block of MkSeederSeedsSoA.h), and the
// classifier's other features. The functions port the chain that CMSSW applies to a mkFit seed (conversion to a
// state on the last hit's module plane, the trajectory state on the beam line, the perigee); the steps are described
// at the functions. The conversion, the plane crossing, the TSCBL, the curvilinear Jacobian and the field are the
// portable ports of RecoTracker/MkFitAlpaka (mkfitdev::pca, mkfitdev::navdev, mkfitdev::field). Float is used where CMSSW computes in float (GlobalPoint / GlobalVector), double where it
// computes in double; the device's libm and fused multiply-adds round differently (physics-equivalent, not
// bit-identical: a looper's TSCBL, whose distance to the beam line can have several minima, can converge to another
// one). Every final seed gets its features and perigee. A state outside the parametrization's volume uses the
// parametrization there too (the seeds' last hits are inside it).

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/MkFitAlpakaMath/interface/math/HelixCrossings.h"
#include "RecoTracker/MkFitAlpakaMath/interface/math/PcaToBeamLine.h"
#include "RecoTracker/MkFitAlpakaMath/interface/math/TkBfield.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederPSMark.h"

#include "MkSeederDeviceTypes.h"
#include "MkSeederScratchSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::pca {

  namespace mpca = ::mkfitdev::pca;

  // ---- float 3-vectors as GlobalPoint / GlobalVector (Basic3DVector<float>): RecoTracker/MkFitAlpaka's (mag, perp,
  // unit), with the cross and dot products
  using Vector3 = mpca::F3;
  ALPAKA_FN_HOST_ACC inline Vector3 cross(const Vector3 &a, const Vector3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
  }
  ALPAKA_FN_HOST_ACC inline float dot(const Vector3 &a, const Vector3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

  // packed lower triangle of a symmetric matrix
  ALPAKA_FN_HOST_ACC inline int sym(int i, int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }

  // the field at gp: the tracker volume's parametrized field (OAEParametrizedMagneticField, TkBfield 3.8 T; valid for
  // r < 115 cm, |z| < 280 cm, the seeds' last hits' volume)
  ALPAKA_FN_HOST_ACC inline void field(const Vector3 &gp, Vector3 &B) {
    ::mkfitdev::field::tkBfield(gp.x, gp.y, gp.z, B.x, B.y, B.z);
  }

  // a FreeTrajectoryState: position, momentum (float), charge, the curvilinear covariance (double) of
  // RecoTracker/MkFitAlpaka's PCA input, and the field at the position (GlobalTrajectoryParameters' cache)
  struct FreeTrajectoryState : mpca::TrackIn {
    Vector3 b;
  };

  // ---- mkfit::TrackState (CCS: x, y, z, 1/pt, phi, theta; err: packed 6x6) -> FreeTrajectoryState, as
  // seedTsosOnLastHit (mkfit::TrackState::convertFromCCSToGlbCurvilinear: mkfitdev::pca::ccsToCurvilinear); false:
  // the covariance's diagonal has a negative entry (CurvilinearTrajectoryError::posDef())
  ALPAKA_FN_HOST_ACC inline bool curvilinearToFreeState(const float *par,
                                                        const float *err,
                                                        int chg,
                                                        FreeTrajectoryState &f) {
    mpca::ccsToCurvilinear(par, err, chg, f);
    bool pd = true;
    for (int i = 0; i < 5; ++i)
      pd = pd && f.C[i][i] >= 0;
    field(f.x, f.b);
    return pd;
  }

  // ---- TSCBLBuilderNoMaterial (mkfitdev::pca::pcaToBeamLine: TwoTrackMinimumDistanceHelixLine, the field at the
  // start, AnalyticalCurvilinearJacobian) to the beam line (bs: x0, y0, z0, dxdz, dydz); false: invalid
  ALPAKA_FN_HOST_ACC inline bool closestApproachToBeamline(const FreeTrajectoryState &in,
                                                           const double *bs,
                                                           FreeTrajectoryState &out) {
    const mpca::BeamIn beamLine{{float(bs[0]), float(bs[1]), float(bs[2])}, {float(bs[3]), float(bs[4]), 1.f}};
    mpca::PcaOut o;
    if (mpca::pcaToBeamLine(in, in.b, beamLine, o) != mpca::kPcaOk)
      return false;
    out.x = o.x, out.p = o.p, out.charge = in.charge;
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        out.C[i][j] = o.C[i][j];
    field(out.x, out.b);
    return true;
  }

  // ---- PerigeeConversions::jacobianCurvilinear2Perigee
  template <typename TAcc>
  ALPAKA_FN_ACC inline void curvilinearToPerigeeJacobian(const TAcc &acc,
                                                         const FreeTrajectoryState &f,
                                                         double (&jac)[5][5]) {
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        jac[i][j] = 0.;
    const Vector3 p = f.p;
    const Vector3 Zv = {0.f, 0.f, 1.f};
    const Vector3 T = mpca::unit(p);
    const Vector3 U = mpca::unit(cross(Zv, T));
    const Vector3 V = cross(T, U);
    const Vector3 I = mpca::unit(Vector3{-p.x, -p.y, 0.f});
    const Vector3 Jv = {-I.y, I.x, 0.f};
    const Vector3 &K = Zv;
    const Vector3 B = {2.99792458e-3f * f.b.x, 2.99792458e-3f * f.b.y, 2.99792458e-3f * f.b.z};
    const Vector3 H = mpca::unit(B);
    const Vector3 HxT = cross(H, T);
    const Vector3 N = mpca::unit(HxT);
    const double alpha = mpca::mag(HxT);
    const double qbp = float(f.charge) / mpca::mag(p);
    const double Q = -mpca::mag(B) * qbp;
    const double alphaQ = alpha * Q;
    const double lambda = 0.5 * M_PI - alpaka::math::atan2(acc, mpca::perp(p), p.z);
    const double sinlambda = alpaka::math::sin(acc, lambda), coslambda = alpaka::math::cos(acc, lambda);
    const double seclambda = 1. / coslambda;
    const double ITI = 1. / dot(T, I);
    const double NU = dot(N, U), NV = dot(N, V), UI = dot(U, I), VI = dot(V, I);
    const double UJ = dot(U, Jv), VJ = dot(V, Jv), UK = dot(U, K), VK = dot(V, K);
    const float tcurv = -2.99792458e-3f * (float(f.charge) / mpca::perp(p)) * f.b.z;
    if (alpaka::math::abs(acc, tcurv) < 1.e-10) {
      jac[0][0] = seclambda;
      jac[0][1] = sinlambda * seclambda * seclambda * alpaka::math::abs(acc, qbp);
    } else {
      const double Bz = B.z;
      jac[0][0] = -Bz * seclambda;
      jac[0][1] = -Bz * sinlambda * seclambda * seclambda * qbp;
      jac[1][3] = alphaQ * NV * UI * ITI;
      jac[1][4] = alphaQ * NV * VI * ITI;
      jac[0][3] = -jac[0][1] * jac[1][3];
      jac[0][4] = -jac[0][1] * jac[1][4];
      jac[2][3] = -alphaQ * seclambda * NU * UI * ITI;
      jac[2][4] = -alphaQ * seclambda * NU * VI * ITI;
    }
    jac[1][1] = -1.;
    jac[2][2] = 1.;
    jac[3][3] = VK * ITI;
    jac[3][4] = -UK * ITI;
    jac[4][3] = -VJ * ITI;
    jac[4][4] = UJ * ITI;
  }

  // ---- AnalyticalPropagator onto a plane (point m[0..2], normal m[3..5]): the helix in the field's z component at
  // the start (HelixArbitraryPlaneCrossing: mkfitdev::navdev::arbitraryPlaneCrossing), the crossing along the momentum,
  // else the one opposite to it (the propagators of seedTsosOnLastHit), within the propagator's largest turning angle
  // (1.6); the field's z component at the crossing within 50% of the start's (the propagator's curvature check); the
  // curvilinear covariance with AnalyticalCurvilinearJacobian. false: no crossing (the seed is dropped).
  // oppositeOnly: only the crossing opposite to the momentum (an AnalyticalPropagator oppositeToMomentum)
  ALPAKA_FN_HOST_ACC inline bool propagateToPlane(const FreeTrajectoryState &in,
                                                  const float *m,
                                                  FreeTrajectoryState &out,
                                                  bool oppositeOnly = false) {
    namespace navdev = ::mkfitdev::navdev;
    constexpr double kMaxTurn = 1.6;
    const float pt = mpca::perp(in.p);
    if (!(pt > 0.f))
      return false;
    const float rho = -2.99792458e-3f * (float(in.charge) / pt) * in.b.z;  // transverseCurvature()
    const double sinTheta = pt / mpca::mag(in.p);
    const Vector3 pnt{m[0], m[1], m[2]}, nrm{m[3], m[4], m[5]};
    navdev::Crossing c;
    for (const bool along : {true, false}) {
      if (along && oppositeOnly)
        continue;
      c = navdev::arbitraryPlaneCrossing(in.x, in.p, rho, along, pnt, nrm);
      if (c.status == navdev::kXingOk && std::abs(rho * c.s * sinTheta) <= kMaxTurn)
        break;
      c.status = navdev::kXingNone;
    }
    if (c.status != navdev::kXingOk)
      return false;
    out.x = {c.x, c.y, c.z};
    out.p = {pt * c.dx, pt * c.dy, in.p.z};
    out.charge = in.charge;
    field(out.x, out.b);
    if (std::abs(double(out.b.z) - double(in.b.z)) > 0.5 * std::abs(double(in.b.z)))
      return false;
    double J[5][5];
    mpca::curvilinearJacobian(in.x, in.p, in.charge, out.x, out.p, c.s, in.b, J);
    mpca::similarity5(J, in.C, out.C);
    return true;
  }

  // ---- the state of a seed on its last hit's module (mkseeder FitInputs frame: pos, zdir, xdir), as an FreeTrajectoryState
  // (seedTsosOnLastHit: |local z| < 1e-4 cm: the state itself, else taken onto the plane); false: the seed is
  // dropped (not posDef, no crossing)
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool seedFreeState(
      const TAcc &acc, const float *par, const float *err, int chg, const float *m, FreeTrajectoryState &f) {
    FreeTrajectoryState f0;
    if (!curvilinearToFreeState(par, err, chg, f0))
      return false;
    const float lz = m[3] * (f0.x.x - m[0]) + m[4] * (f0.x.y - m[1]) + m[5] * (f0.x.z - m[2]);
    if (alpaka::math::abs(acc, lz) < 1e-4f) {
      f = f0;
      return true;
    }
    return propagateToPlane(f0, m, f);
  }

  // ---- FreeTrajectoryState -> mkfit::TrackState (CCS: x, y, z, 1/pt, phi, theta; err: packed 6x6), as
  // mkfit::TrackState::convertFromGlbCurvilinearToCCS() of the curvilinear covariance (the 6x6 with a zero last row)
  ALPAKA_FN_HOST_ACC inline void freeStateToCCS(const FreeTrajectoryState &f, float *par, float *err, int &chg) {
    const float px = f.p.x, py = f.p.y, pz = f.p.z;
    const float pt2 = px * px + py * py;
    const float pt = std::sqrt(pt2);
    par[0] = f.x.x, par[1] = f.x.y, par[2] = f.x.z;
    par[3] = 1.f / pt;
    par[4] = std::atan2(py, px);
    par[5] = std::atan2(pt, pz);
    chg = f.charge;
    // jacobianCurvilinearToCCS (zero elsewhere)
    const float invpt = 1.f / pt, invpt2 = 1.f / pt2, invp = 1.f / std::sqrt(pt2 + pz * pz);
    const float sinPhi = py * invpt, cosPhi = px * invpt, sinLam = pz * invp, cosLam = pt * invp;
    float J[6][5] = {};
    J[0][3] = -sinPhi;
    J[0][4] = -sinLam * cosPhi;
    J[1][3] = cosPhi;
    J[1][4] = -sinLam * sinPhi;
    J[2][4] = cosLam;
    J[3][0] = float(f.charge) / cosLam;
    J[3][1] = pz * invpt2;
    J[4][2] = 1.f;
    J[5][1] = -1.f;
    double JC[6][5];
    for (int i = 0; i < 6; ++i)
      for (int k = 0; k < 5; ++k) {
        double a = 0.;
        for (int l = 0; l < 5; ++l)
          a += double(J[i][l]) * f.C[l][k];
        JC[i][k] = a;
      }
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j) {
        double a = 0.;
        for (int k = 0; k < 5; ++k)
          a += JC[i][k] * double(J[j][k]);
        err[sym(i, j)] = float(a);
      }
  }

  // the classifier's state features of seed i (MkFitSeedFeatureCalc.cc computeState()) of the state f, as a
  // reco::Track of it (reco::Track(0, 0, vertex, momentum, charge, curvilinear covariance): the covariance stored as
  // float)
  template <typename TAcc>
  ALPAKA_FN_ACC inline void stateFeatures(
      const TAcc &acc, const FreeTrajectoryState &f, const double *bs, SeedFeaturesView features, uint32_t i) {
    using namespace ::mkseeder::seedfeature;
    auto inputs = features.inputs(i);
    float c[15];
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j <= i; ++j)
        c[sym(i, j)] = float(f.C[i][j]);
    const double vx = f.x.x, vy = f.x.y, vz = f.x.z;
    const double px = f.p.x, py = f.p.y, pz = f.p.z;
    const double pt2 = px * px + py * py, p2 = pt2 + pz * pz;
    const double pt = alpaka::math::sqrt(acc, pt2);
    const int q = f.charge;
    inputs(kPt) = pt;
    {
      // ROOT's Eta_FromRhoZ
      const double zs = pz / pt;
      inputs(kAbsEta) =
          alpaka::math::abs(acc,
                            float(pt > 0 ? alpaka::math::log(acc, zs + alpaka::math::sqrt(acc, zs * zs + 1.0))
                                         : (pz >= 0 ? 1e30 : -1e30)));
    }
    {
      const double ptimespt = alpaka::math::sqrt(acc, p2 * pt2);
      const double oneovercharge = 1 / double(q);
      const double ptErr2 = pt2 * p2 * oneovercharge * oneovercharge * double(c[0]) +
                            2 * ptimespt * oneovercharge * pz * double(c[1]) + pz * pz * double(c[2]);
      const double ptErr = alpaka::math::sqrt(acc, ptErr2);
      inputs(kPtErrorRelative) = ptErr / pt;
    }
    const double theta = alpaka::math::atan2(acc, pt, pz);
    const double cot = 1. / alpaka::math::tan(acc, theta);
    const double lambdaErr = alpaka::math::sqrt(acc, double(c[2]));
    inputs(kCotThetaError) = (1. + cot * cot) * lambdaErr;
    inputs(kEtaError) = lambdaErr * alpaka::math::sqrt(acc, p2 / pt2);
    inputs(kPhiError) = alpaka::math::sqrt(acc, double(c[5]));
    inputs(kDxy) = (-(vx - bs[0]) * py + (vy - bs[1]) * px) / pt;
    inputs(kDxyError) = alpaka::math::sqrt(acc, double(c[9]));
    inputs(kDz) = (vz - bs[2]) - ((vx - bs[0]) * px + (vy - bs[1]) * py) * pz * (1 / pt2);
    inputs(kDzError) = alpaka::math::sqrt(acc, double(c[14])) * alpaka::math::sqrt(acc, p2 / pt2);
  }

  // The state of each seed on its last hit's module (seedFreeState), then to the beam line (the TSCBL of
  // closestApproachToBeamline: the beam line as the line, the field from the OAE 1103l parametrization at 3.8 T, i.e.
  // TkBfield's VolumeBasedMagneticField parametrization in the tracker volume, AnalyticalCurvilinearJacobian, the
  // similarity), then the perigee (PerigeeConversions::ftsToPerigeeError). A seed whose state does not convert is
  // dropped; a TSCBL that does not converge gives the state at its own position (hasStateAtBeamline = 0, the pixel
  // track of a failed fit).
  class KernelComputeSeedPerigees {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, PcaInputs v) const {
      using namespace ::mkseeder::seedfeature;
      using ::mkseeder::kSeedFeatureNone;
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        auto inputs = v.features.inputs(i);
        for (int k = kPt; k <= kDzError; ++k)
          inputs(k) = kSeedFeatureNone;
        v.features.isDropped(i) = 1, v.features.hasStateAtBeamline(i) = 0;
        v.perigee.status(i) = ::mkseeder::kPerigeeNone;
        // the mkFit seeder's own features: the seed fit's chi2 and the cleaning's quality
        inputs(kSeedChi2) = v.states.chi2(i);
        inputs(kSeedFakeScore) = v.quality.fakeScore(i);
        inputs(kSeedCleanScore) = v.quality.score(i);
        inputs(kSeedNAmbiguous) = v.quality.nAmbiguous(i);
        const auto ps = v.seeds[i];
        const int nh = ps.nHits();
        if (!ps.isKept() || nh < 1)
          continue;
        float m[9];
        modulePlane(v.modules,
                    v.layerGeometry[ps.hitLayers()(nh - 1)].firstModule() + v.hits.detIdInLayer(ps.hitRows()(nh - 1)),
                    m);
        float par[6], err[21], chi2;
        int chg;
        loadState(v.states, i, par, err, chg, chi2);
        // the classifier's features from the forward state; isDropped: the state does not convert (its curvilinear
        // covariance is not posDef, or no crossing with its plane: the seed would be dropped; it
        // takes no part in the sharing features)
        FreeTrajectoryState f0, f1;
        if (seedFreeState(acc, par, err, chg, m, f0)) {
          // at the beam line, or (hasStateAtBeamline 0) at its own position
          const bool atBeamline = closestApproachToBeamline(f0, v.beamSpot, f1);
          stateFeatures(acc, atBeamline ? f1 : f0, v.beamSpot, v.features, i);
          v.features.isDropped(i) = 0, v.features.hasStateAtBeamline(i) = atBeamline;
        }
        // the pixel track from the beam-line state (the forward one where the beam-line fit failed: the copy keeps it)
        FreeTrajectoryState b0, b1;
        if (v.states.hasBeamlineState(i)) {
          const auto st = v.states[i];
          for (int d = 0; d < 6; ++d)
            par[d] = st.beamlineParameters()(d);
          for (int d = 0; d < 21; ++d)
            err[d] = st.beamlineCovariance()(d);
          chg = st.beamlineCharge();
        }
        if (!seedFreeState(acc, par, err, chg, m, b0))
          continue;
        if (!closestApproachToBeamline(b0, v.beamSpot, b1)) {
          v.perigee.status(i) = ::mkseeder::kPerigeeFailedFit;
          continue;
        }
        double jac[5][5], pc[5][5];
        curvilinearToPerigeeJacobian(acc, b1, jac);
        mpca::similarity5(jac, b1.C, pc);
        auto perigee = v.perigee[i];
        perigee.charge() = b0.charge;
        perigee.vertex()(0) = b1.x.x, perigee.vertex()(1) = b1.x.y, perigee.vertex()(2) = b1.x.z;
        perigee.momentum()(0) = b1.p.x, perigee.momentum()(1) = b1.p.y, perigee.momentum()(2) = b1.p.z;
        for (int i = 0; i < 5; ++i)
          for (int j = 0; j <= i; ++j)
            perigee.covariance()(sym(i, j)) = float(pc[i][j]);
        perigee.status() = ::mkseeder::kPerigeeValid;
      }
    }
  };

  // The state of each final seed whose last hit is an outer-tracker hit at its last pixel hit (MkSeederSeedPixelStates:
  // LSTInputProducer's stateAtLastPixelHit): the state on its last hit's module (seedFreeState, as the seed's
  // TrajectorySeed state), then the AnalyticalPropagator oppositeToMomentum onto the last pixel hit's module, without
  // material; the number of pixel hits of every seed.
  class KernelComputeSeedPixelStates {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, PcaInputs v, SeedPixelStatesView out) const {
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        auto o = out[i];
        o.status() = ::mkseeder::kPixelStateNone;
        o.charge() = 0;
        const auto ps = v.seeds[i];
        const int nh = ps.nHits();
        int nPix = 0, lastPix = -1;
        for (int k = 0; k < nh; ++k)
          if (v.layerGeometry[ps.hitLayers()(k)].isPixel())
            ++nPix, lastPix = k;
        o.nPixelHits() = uint8_t(nPix);
        if (!ps.isKept() || nh < 2 || lastPix < 0 || lastPix == nh - 1)
          continue;
        float m[9], mPix[9];
        modulePlane(v.modules,
                    v.layerGeometry[ps.hitLayers()(nh - 1)].firstModule() + v.hits.detIdInLayer(ps.hitRows()(nh - 1)),
                    m);
        modulePlane(v.modules,
                    v.layerGeometry[ps.hitLayers()(lastPix)].firstModule() + v.hits.detIdInLayer(ps.hitRows()(lastPix)),
                    mPix);
        float par[6], err[21], chi2;
        int chg;
        loadState(v.states, i, par, err, chg, chi2);
        FreeTrajectoryState f0, f1;
        if (!seedFreeState(acc, par, err, chg, m, f0) || !propagateToPlane(f0, mPix, f1, true))
          continue;
        float cpar[6], cerr[21];
        int cchg;
        freeStateToCCS(f1, cpar, cerr, cchg);
        for (int d = 0; d < 6; ++d)
          o.parameters()(d) = cpar[d];
        for (int d = 0; d < 21; ++d)
          o.covariance()(d) = cerr[d];
        o.charge() = int8_t(cchg);
        o.status() = ::mkseeder::kPixelStateValid;
      }
    }
  };

  // ==== the classifier's other features (MkFitSeedFeatureCalc.cc compute(): hits, circle, clusters, sharing), from
  // the bin-order hits (MkSeederHitsSoA: the mkfit::Hit positions, the TrackingRecHits' global positions bit for bit),
  // the pixel clusters' charges and sizes, the pixel modules' frames, pitches and thicknesses, the outer-tracker
  // clusters' sizes (a cluster index outside the tables, which the hits' conversion rules out, adds no cluster
  // feature). No per-thread arrays (the kernel's stack sets the context's local memory: a 2 kB stack per thread cost
  // ~150 MiB of device memory): the hits are read where used, the sharers counted in place.
  constexpr float kNone = ::mkseeder::kSeedFeatureNone;

  // the classifier's other features with the sharing's lists: per hit the seeds with >= shareMinHits hits on it
  struct ClassifierFeatureState : ClassifierFeatureInputs {
    ::mkseeder::MkSeederItemLists *hitSeeds;
  };

  ALPAKA_FN_HOST_ACC inline bool eligible(const SeedHitsConstView &seeds,
                                          uint32_t i,
                                          SeedFeaturesView features,
                                          LayerGeometryView layerGeometry,
                                          int shareMinHits,
                                          int shareMinPixelHits) {
    if (!(seeds.isKept(i) && !features.isDropped(i) && (shareMinHits <= 0 || seeds.nHits(i) >= shareMinHits)))
      return false;
    if (shareMinPixelHits <= 0)
      return true;
    int npix = 0;
    for (int k = 0; k < seeds.nHits(i); ++k)
      npix += layerGeometry[seeds.hitLayers(i)(k)].isPixel();
    return npix >= shareMinPixelHits;
  }

  // the seeds on each hit (CSR over the hits): counts, then (after the scan) the fill
  class KernelShareCount {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ClassifierFeatureState v, int fill) const {
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, v.cap)) {
        if (i >= n)
          continue;
        if (!eligible(v.seeds, i, v.features, v.layerGeometry, v.shareMinHits, v.shareMinPixelHits))
          continue;
        const auto ps = v.seeds[i];
        for (int k = 0; k < ps.nHits(); ++k) {
          const uint32_t h = ps.hitRows()(k);
          if (!fill)
            v.hitSeeds->count(acc, h);
          else
            v.hitSeeds->fill(acc, h, i);
        }
      }
    }
  };

  class KernelComputeClassifierFeatures {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ClassifierFeatureState v) const {
      using namespace ::mkseeder::seedfeature;
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        auto O = v.features.inputs(i);
        for (int k = kNHits; k <= kMaxSharedHits; ++k)
          O(k) = kNone;
        for (int k = kPSNOTP; k < kNFeatures; ++k)
          O(k) = kNone;
        v.features.nMergedCandidates(i) = 0;
        const auto ps = v.seeds[i];
        if (!ps.isKept())
          continue;
        // the hits
        const int nh = ps.nHits();
        auto g = [&](int k) -> Vector3 {
          const uint32_t h = ps.hitRows()(k);
          return {v.hits.x(h), v.hits.y(h), v.hits.z(h)};
        };
        auto hr = [&](int k) {
          const Vector3 p = g(k);
          return alpaka::math::sqrt(acc, p.x * p.x + p.y * p.y);
        };
        int npxb = 0, npxf = 0, notb = 0, note = 0;
        unsigned long long laymask = 0;
        for (int k = 0; k < nh; ++k) {
          const int lay = ps.hitLayers()(k);
          // the subdetector from the mkFit layer (pixel / outer tracker, barrel / disc); the layers as the mkFit ids
          // (one per layer code: only their number enters)
          const auto lg = v.layerGeometry[lay];
          const bool pix = lg.isPixel(), barrel = lg.isBarrel();
          npxb += pix && barrel, npxf += pix && !barrel, notb += !pix && barrel, note += !pix && !barrel;
          if (lay >= 0 && lay < 64)
            laymask |= 1ull << lay;
        }
        O(kNHits) = nh;
        O(kNPixelBarrelHits) = npxb, O(kNPixelDiscHits) = npxf, O(kNOuterBarrelHits) = notb, O(kNOuterDiscHits) = note;
        O(kNPixelHits) = npxb + npxf, O(kNOuterHits) = notb + note;
        O(kNLayers) = alpaka::popcount(acc, static_cast<std::uint64_t>(laymask));
        // the circle through the first, the middle (by radius) and the last hit; the line in (s, z)
        int circ_ok = 0;
        float circ_pt = kNone, circ_res_max = kNone, circ_res_rms = kNone, rz_res_max = kNone, rz_res_rms = kNone;
        double cx = 0, cy = 0, R = 0;
        bool straight = false;
        double ux = 0, uy = 0;
        double sL = 0, zcot = 0;
        int turn = 0;
        if (nh >= 1)
          O(kFirstHitR) = hr(0), O(kFirstHitZ) = g(0).z, O(kLastHitR) = hr(nh - 1), O(kLastHitZ) = g(nh - 1).z;
        if (nh >= 3) {
          const Vector3 g0 = g(0), gL = g(nh - 1);
          const double rm = 0.5 * (hr(0) + hr(nh - 1));
          int im = 1;
          float rim = hr(1);
          for (int k = 1; k < nh - 1; ++k) {
            const float rk = hr(k);
            if (alpaka::math::abs(acc, rk - rm) < alpaka::math::abs(acc, rim - rm))
              im = k, rim = rk;
          }
          const Vector3 gm = g(im);
          const double x0 = g0.x, y0 = g0.y;
          const double ax = gm.x - x0, ay = gm.y - y0;
          const double bx = gL.x - x0, by = gL.y - y0;
          const double d = 2 * (ax * by - ay * bx);
          const double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by;
          const double chord = alpaka::math::sqrt(acc, b2);
          if (chord > 0) {
            circ_ok = 1;
            straight = alpaka::math::abs(acc, d) < 2e-7 * chord * alpaka::math::sqrt(acc, a2);
            turn = d > 0 ? 1 : -1;
            if (!straight) {
              const double ox = (by * a2 - ay * b2) / d, oy = (ax * b2 - bx * a2) / d;
              cx = x0 + ox, cy = y0 + oy, R = std::hypot(ox, oy);
              circ_pt = 0.299792458e-2 * alpaka::math::abs(acc, v.fieldAtOrigin) * R;
            } else {
              ux = bx / chord, uy = by / chord;
              circ_pt = 1e4;
            }
            auto arc = [&](const Vector3 &p) {
              if (straight)
                return (p.x - x0) * ux + (p.y - y0) * uy;
              const double px = x0 - cx, py = y0 - cy, qx = p.x - cx, qy = p.y - cy;
              return R * alpaka::math::abs(acc, alpaka::math::atan2(acc, px * qy - py * qx, px * qx + py * qy));
            };
            sL = arc(gL);
            const double z0 = g0.z, dzL = gL.z - z0;
            zcot = sL > 0 ? dzL / sL : 0;
            const double perpf = sL > 0 ? sL / std::hypot(sL, dzL) : 1;
            double cmax = 0, csum = 0, zmax = 0, zsum = 0;
            for (int k = 0; k < nh; ++k) {
              const Vector3 p = g(k);
              const double cres = straight ? (p.x - x0) * uy - (p.y - y0) * ux : std::hypot(p.x - cx, p.y - cy) - R;
              const double zres = (p.z - (z0 + zcot * arc(p))) * perpf;
              cmax = std::max(cmax, alpaka::math::abs(acc, cres)), csum += cres * cres;
              zmax = std::max(zmax, alpaka::math::abs(acc, zres)), zsum += zres * zres;
            }
            const int nfree = nh - 3;
            circ_res_max = cmax, rz_res_max = zmax;
            circ_res_rms = nfree > 0 ? alpaka::math::sqrt(acc, csum / nfree) : 0;
            rz_res_rms = nh > 2 ? alpaka::math::sqrt(acc, zsum / (nh - 2)) : 0;
          }
        }
        O(kCircleResidualMax) = circ_res_max, O(kCircleResidualRms) = circ_res_rms;
        O(kRzResidualMax) = rz_res_max, O(kRzResidualRms) = rz_res_rms;
        // pt / the circle's pt, in float, as in the seed features
        {
          const float pt = O(kPt);
          O(kPtOverCirclePt) = pt > 0 && circ_pt > 0 ? pt / circ_pt : kNone;
        }
        // the clusters: sizes, and those expected from the circle's direction at the hit
        constexpr float kFMax = 3.402823466e+38f;
        float dsx_min = kFMax, dsy_min = kFMax, q_min = kFMax, dsx_max = -kFMax, dsy_max = -kFMax, q_max = -kFMax;
        int ot_size_max = 0, npixc = 0;
        for (int k = 0; k < nh; ++k) {
          const uint32_t h = ps.hitRows()(k);
          const int lay = ps.hitLayers()(k);
          int sx = -1, sy = -1;
          float qc = 0, esx = kNone, esy = kNone;
          if (v.layerGeometry[lay].isPixel()) {
            const uint32_t c = v.hits.clusterIndex(h);
            if (c >= v.nPixelClusters)
              continue;
            sx = v.pixelClusters.sizeX(c), sy = v.pixelClusters.sizeY(c), qc = v.pixelClusters.charge(c);
          } else {
            const uint32_t c = v.hits.clusterIndex(h);
            if (c >= v.nOTClusters)
              continue;
            sx = v.otClusters.size(c);
            ot_size_max = std::max(ot_size_max, sx);
          }
          if (circ_ok && sy >= 0) {
            double tx, ty;
            if (straight)
              tx = ux, ty = uy;
            else {
              const Vector3 p = g(k);
              const double rx = p.x - cx, ry = p.y - cy, rr = std::hypot(rx, ry);
              tx = -turn * ry / rr, ty = turn * rx / rr;
            }
            const Vector3 dir = {float(tx), float(ty), float(zcot)};
            float m[12];
            pixelModuleFrame(v.modules, v.layerGeometry[lay].firstModule() + v.hits.detIdInLayer(h), m);
            const float lx = m[0] * dir.x + m[1] * dir.y + m[2] * dir.z;
            const float ly = m[3] * dir.x + m[4] * dir.y + m[5] * dir.z;
            const float lz = m[6] * dir.x + m[7] * dir.y + m[8] * dir.z;
            if (alpaka::math::abs(acc, lz) > 1e-6) {
              esx = m[11] * alpaka::math::abs(acc, lx / lz) / m[9];
              // clf_lorentz: the Lorentz drift's extra width (pixels) on the barrel layers 2-4
              if (lay >= 1 && lay <= 3)
                esx += v.lorentzSizeX;
              esy = m[11] * alpaka::math::abs(acc, ly / lz) / m[10];
            }
          }
          if (sy >= 0) {
            ++npixc;
            q_min = std::min(q_min, qc), q_max = std::max(q_max, qc);
            if (esx != kNone) {
              dsx_min = std::min(dsx_min, sx - esx), dsx_max = std::max(dsx_max, sx - esx);
              dsy_min = std::min(dsy_min, sy - esy), dsy_max = std::max(dsy_max, sy - esy);
            }
          }
        }
        if (npixc == 0)
          q_min = q_max = kNone;
        if (dsx_min == kFMax)
          dsx_min = dsx_max = dsy_min = dsy_max = kNone;
        O(kPixelSizeXExcessMin) = dsx_min, O(kPixelSizeXExcessMax) = dsx_max;
        O(kPixelSizeYExcessMin) = dsy_min, O(kPixelSizeYExcessMax) = dsy_max;
        O(kPixelChargeMin) = q_min, O(kPixelChargeMax) = q_max, O(kOuterClusterSizeMax) = ot_size_max;
        // the PS-module confirmation features (MkSeederSeedFeaturesSoA.h seedpsfeature)
        {
          using namespace ::mkseeder::seedpsfeature;
          namespace psc = ::mkseeder::psCounts;
          auto Q = v.features.psInputs(i);
          for (int k = 0; k < kNPSFeatures; ++k)
            Q(k) = kNone;
          if (v.psMark) {
            // the seed's q/pT from its perigee; without one (failed closest approach) the q/pT-dependent features
            // stay kNone
            const bool hasQ = v.perigee.status(i) == ::mkseeder::kPerigeeValid;
            const auto pm = v.perigee.momentum(i);
            const float ppt = hasQ ? alpaka::math::sqrt(acc, pm(0) * pm(0) + pm(1) * pm(1)) : 0.f;
            const float sq = ppt > 0 ? v.perigee.charge(i) / ppt : 0.f;
            const int rs = psc::range(sq);
            int nP = 0, nMk = 0, nC1 = 0, nC2 = 0, nOut = 0, cTot = 0, nMb = 0, nMa = 0;
            float pmax = kNone, pmin = kNone;
            for (int k = 0; k < nh; ++k) {
              const int lay = ps.hitLayers()(k);
              if (lay != 4 && lay != 6 && lay != 8)
                continue;
              const int c0 = v.layers.firstOTHit(lay);
              if (c0 < 0)
                continue;
              const uint32_t h = ps.hitRows()(k);
              const uint32_t c = uint32_t(c0) + (h - v.layers.firstHit(lay));
              ++nP;
              const uint16_t w = v.otHits.psCounts(c);
              const int cr = psc::count(w, rs), ct = psc::total(w);
              nC1 += cr >= 1, nC2 += cr >= 2, nOut += cr == 0 && ct > 0, cTot = std::max(cTot, ct);
              nMb += (w & psc::kBestMerged) != 0, nMa += (w & psc::kAnyMerged) != 0;
              if (v.otHits.psMatches(c) == 0)
                continue;
              ++nMk;
              const float dx = v.hits.x(h) - float(v.beamSpotX), dy = v.hits.y(h) - float(v.beamSpotY);
              const float r = alpaka::math::sqrt(acc, dx * dx + dy * dy);
              const float pull = alpaka::math::abs(acc, v.otHits.psQOverPt(c) - sq) /
                                 (v.psCurvatureResolution * 23.f / std::max(r, 1.f));
              pmax = std::max(pmax, pull), pmin = pmin < 0 ? pull : std::min(pmin, pull);
            }
            Q(kNOTP) = nP, Q(kNMarked) = nMk, Q(kMaxTotal) = cTot, Q(kNBestMerged) = nMb, Q(kNAnyMerged) = nMa;
            if (hasQ) {
              Q(kPullMax) = pmax, Q(kPullMin) = pmin;
              Q(kNRangeMatch) = nC1, Q(kNRangeMatch2) = nC2, Q(kNOutOfRange) = nOut;
            }
          }
          // the model's PS inputs
          O(::mkseeder::seedfeature::kPSNOTP) = Q(kNOTP), O(::mkseeder::seedfeature::kPSNMarked) = Q(kNMarked);
          O(::mkseeder::seedfeature::kPSPullMax) = Q(kPullMax), O(::mkseeder::seedfeature::kPSPullMin) = Q(kPullMin);
          O(::mkseeder::seedfeature::kPSNRangeMatch) = Q(kNRangeMatch);
          O(::mkseeder::seedfeature::kPSNOutOfRange) = Q(kNOutOfRange);
        }
        {
          int nmb = 0, nmd = 0;
          for (int k = 0; k < nh; ++k) {
            const auto lg = v.layerGeometry[ps.hitLayers()(k)];
            const bool m = lg.isPixel() && v.hits.mergedTag(ps.hitRows()(k)) != 0;
            nmb += m && lg.isBarrel(), nmd += m && !lg.isBarrel();
          }
          v.features.nMergedCandidates(i) = nmb + nmd;
          O(::mkseeder::seedfeature::kNMergedBarrel) = nmb, O(::mkseeder::seedfeature::kNMergedDisc) = nmd;
        }
        // the sharing: the other seeds (with >= shareMinHits hits) on this seed's hits; per other seed j the number of
        // hits it shares, counted at j's first shared hit (in this seed's hit order)
        int nshare1 = 0, nshare2 = 0, share_max = 0;
        for (int k = 0; k < nh; ++k) {
          const uint32_t h = ps.hitRows()(k);
          for (const uint32_t *e = v.hitSeeds->begin(h), *e1 = v.hitSeeds->end(h); e < e1; ++e) {
            const uint32_t j = *e;
            if (j == i)
              continue;
            const auto pj = v.seeds[j];
            const int nj = pj.nHits();
            int c = 0;
            bool first = true;
            for (int a = 0; a < nh; ++a) {
              const uint32_t ha = ps.hitRows()(a);
              bool on = false;
              for (int b = 0; b < nj && !on; ++b)
                on = pj.hitRows()(b) == ha;
              if (on) {
                ++c;
                first = first && a >= k;
              }
            }
            if (!first)
              continue;
            ++nshare1;
            nshare2 += c >= 2;
            share_max = std::max(share_max, c);
          }
        }
        O(kNSharingSeeds) = nshare1, O(kNSharingSeeds2Hits) = nshare2, O(kMaxSharedHits) = share_max;
      }  // for (i : seeds)
    }  // KernelComputeClassifierFeatures::operator()
  };  // class KernelComputeClassifierFeatures

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::pca

#endif
