#ifndef RecoTracker_MkFitAlpaka_interface_math_HelixCrossings_h
#define RecoTracker_MkFitAlpaka_interface_math_HelixCrossings_h

// the helix crossings the TkDetLayers searches use to pick the sub-layer / ring /
// rod, portable (ALPAKA_FN_HOST_ACC), transliterated from TrackingTools/GeomPropagators with the
// same float / double types:
//   HelixForwardPlaneCrossing          (pathLength to a z plane + position(s))               forwardPlaneCrossing
//   HelixBarrelCylinderCrossing        (onlyPos: the chosen solution's position and path)    barrelCylinderCrossing
//   HelixBarrelPlaneCrossingByCircle   (pathLength to a plane parallel to z + position(s))   barrelPlaneCrossing
// The straight-line branches (|rho| R < 1e-7, pT above ~1e5 GeV) are not ported: status kStraight, the caller routes the
// track to the host. float asin is evaluated in double and rounded once (the host's asinf is correctly rounded in
// practice; the device's is not). Checked per call against the CMSSW classes.

#include <cmath>
#include <limits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/PcaToBeamLine.h"

namespace mkfitdev::navdev {

  // the navigation device code is inlined into its kernels on the device (no ABI calls, hence no
  // callee-save spills); the host compilation keeps plain inline, so the host search is unchanged
#ifndef MKFITDEV_NAV_INLINE
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define MKFITDEV_NAV_INLINE __forceinline__
#define MKFITDEV_NAV_LAMBDA_INLINE __attribute__((always_inline))
#else
#define MKFITDEV_NAV_INLINE inline
#define MKFITDEV_NAV_LAMBDA_INLINE
#endif
#endif

  enum CrossingStatus : int { kXingOk = 0, kXingNone = 1, kXingStraight = 2 };
  struct Crossing {
    int status = kXingNone;
    double s = 0;                  // signed path length
    float x = 0, y = 0, z = 0;     // position at s (GlobalPoint)
    float dx = 0, dy = 0, dz = 0;  // direction at s (HelixPlaneCrossing::direction, not normalised)
  };

  // Basic3DVector<float> helpers (same operation order as the extended vector code)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float perp2F(float x, float y) { return x * x + y * y; }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float magF(::mkfitdev::pca::F3 a) {
    return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool notFinite(double x) {
    return !(std::abs(x) <= std::numeric_limits<double>::max());
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float asinF(float x) { return float(std::asin(double(x))); }

  // RealQuadEquation (TrackingTools/GeomPropagators/src/RealQuadEquation.h)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool realQuad(double A, double B, double C, double& first, double& second) {
    const double D = B * B - 4 * A * C;
    if (D < 0)
      return false;
    const double q = -0.5 * (B + std::copysign(std::sqrt(D), B));
    first = q / A;
    second = C / q;
    return true;
  }

  // HelixForwardPlaneCrossing(point, direction, curvature (float), propDir).pathLength(plane at zPlane) and
  // position(s); rho is the TSOS transverse curvature (float)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE Crossing
  forwardPlaneCrossing(::mkfitdev::pca::F3 pos, ::mkfitdev::pca::F3 dir, float curvature, bool along, float zPlane) {
    Crossing c;
    const double x0 = pos.x, y0 = pos.y, z0 = pos.z, rho = curvature;
    const double px = dir.x, py = dir.y, pz = dir.z;
    const double pt2 = px * px + py * py;
    const double p2 = pt2 + pz * pz;
    const double pI = 1. / std::sqrt(p2);
    const double ptI = 1. / std::sqrt(pt2);
    const double cosPhi0 = px * ptI, sinPhi0 = py * ptI;
    const double cosTheta = pz * pI, sinTheta = pt2 * ptI * pI;
    if (std::abs(cosTheta) < std::numeric_limits<float>::min())
      return c;
    const double dS = (double(zPlane) - z0) / cosTheta;
    if ((along && dS < 0.) || (!along && dS > 0.) || notFinite(dS))
      return c;
    c.status = kXingOk;
    c.s = dS;
    const double dPhi = dS * rho * sinTheta;
    double sdPhi, cdPhi;
    ::mkfitdev::vdt::fast_sincos(dPhi, sdPhi, cdPhi);
    if (std::abs(dPhi) > 1.e-4) {
      c.dx = float(cosPhi0 * cdPhi - sinPhi0 * sdPhi);
      c.dy = float(sinPhi0 * cdPhi + cosPhi0 * sdPhi);
      c.dz = float(cosTheta / sinTheta);
      const double o = 1. / rho;
      c.x = float(x0 + (-sinPhi0 * (1. - cdPhi) + cosPhi0 * sdPhi) * o);
      c.y = float(y0 + (cosPhi0 * (1. - cdPhi) + sinPhi0 * sdPhi) * o);
      c.z = float(z0 + dS * cosTheta);
    } else {
      const double dph = dS * rho * sinTheta;
      c.dx = float(cosPhi0 - (sinPhi0 + 0.5 * cosPhi0 * dph) * dph);
      c.dy = float(sinPhi0 + (cosPhi0 - 0.5 * sinPhi0 * dph) * dph);
      c.dz = float(cosTheta / sinTheta);
      const double st = dS * sinTheta;
      c.x = float(x0 + (cosPhi0 - st * 0.5 * rho * sinPhi0) * st);
      c.y = float(y0 + (sinPhi0 + st * 0.5 * rho * cosPhi0) * st);
      c.z = float(z0 + st * cosTheta / sinTheta);
    }
    return c;
  }

  // HelixBarrelCylinderCrossing(startingPos, startingDir, rho, propDir, cylinder of radius R centred on the z axis,
  // onlyPos): the chosen solution (chooseSolution) and its position
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE Crossing
  barrelCylinderCrossing(::mkfitdev::pca::F3 pos, ::mkfitdev::pca::F3 dir, double rho, bool along, float radius) {
    Crossing c;
    const double R = radius;
    const float posPerp = std::sqrt(perp2F(pos.x, pos.y));
    if (std::abs(rho) * R < 1.e-7 && std::abs(rho) * posPerp < 1.e-7) {
      c.status = kXingStraight;
      return c;
    }
    const double R2cyl = R * R;
    const double pt = std::sqrt(perp2F(dir.x, dir.y));
    const double cx = pos.x - dir.y / (pt * rho), cy = pos.y + dir.x / (pt * rho);
    const double p2 = perp2F(pos.x, pos.y);
    bool solveForX;
    double B, C, E, F;
    if (std::abs(cx) > std::abs(cy)) {
      solveForX = false;
      E = (R2cyl - p2) / (2. * cx);
      F = cy / cx;
      B = 2. * (pos.y - F * pos.x - E * F);
      C = 2. * E * pos.x + E * E + p2 - R2cyl;
    } else {
      solveForX = true;
      E = (R2cyl - p2) / (2. * cy);
      F = cx / cy;
      B = 2. * (pos.x - F * pos.y - E * F);
      C = 2. * E * pos.y + E * E + p2 - R2cyl;
    }
    double q1, q2;
    if (!realQuad(1 + F * F, B, C, q1, q2))
      return c;
    double d1x, d1y, d2x, d2y;
    if (solveForX) {
      d1x = q1;
      d1y = E - F * q1;
      d2x = q2;
      d2y = E - F * q2;
    } else {
      d1x = E - F * q1;
      d1y = q1;
      d2x = E - F * q2;
      d2y = q2;
    }
    // chooseSolution (propDir along / opposite)
    const double momProj1 = dir.x * d1x + dir.y * d1y;
    const double momProj2 = dir.x * d2x + dir.y * d2y;
    const int propSign = along ? 1 : -1;
    double dx, dy;
    if (momProj1 * momProj2 < 0) {
      const bool one = momProj1 * propSign > 0;
      dx = one ? d1x : d2x;
      dy = one ? d1y : d2y;
    } else if (momProj1 * propSign > 0) {
      const bool one = (d1x * d1x + d1y * d1y) < (d2x * d2x + d2y * d2y);
      dx = one ? d1x : d2x;
      dy = one ? d1y : d2y;
    } else
      return c;
    const float ipabs = 1.f / magF(dir);
    const float sinTheta = float(pt) * ipabs;
    const float cosTheta = dir.z * ipabs;
    const double dMag = std::sqrt(dx * dx + dy * dy);
    float tmp = 0.5f * float(dMag * rho);
    if (std::abs(tmp) > 1.f)
      tmp = std::copysign(1.f, tmp);
    c.status = kXingOk;
    c.s = propSign * 2.f * asinF(tmp) / (float(rho) * sinTheta);
    c.x = float(pos.x + dx);
    c.y = float(pos.y + dy);
    c.z = float(pos.z + c.s * cosTheta);
    return c;
  }

  // HelixBarrelPlaneCrossingByCircle(pos, dir, rho, propDir).pathLength(plane) and position(pathLength) for a plane
  // with position pp and normal n (float, as Plane stores them)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE Crossing barrelPlaneCrossing(::mkfitdev::pca::F3 pos,
                                                                      ::mkfitdev::pca::F3 dir,
                                                                      double rho,
                                                                      bool along,
                                                                      ::mkfitdev::pca::F3 pp,
                                                                      ::mkfitdev::pca::F3 n) {
    Crossing c;
    // init()
    const double pabsI = 1. / magF(dir);
    const double pt = std::sqrt(perp2F(dir.x, dir.y));
    const double cosTheta = dir.z * pabsI;
    const double sinTheta = pt * pabsI;
    if (std::abs(rho) < 1.e-7 && std::abs(rho) * std::sqrt(perp2F(pos.x, pos.y)) < 1.e-7) {
      c.status = kXingStraight;
      return c;
    }
    const double o = 1. / (pt * rho);
    const double xc = pos.x - dir.y * o;
    const double yc = pos.y + dir.x * o;
    // pathLength(plane): distToPlane = -plane.localZ(pos) (float dot)
    const float lz = n.x * (pos.x - pp.x) + n.y * (pos.y - pp.y) + n.z * (pos.z - pp.z);
    const double distToPlane = -lz;
    const double nx = n.x, ny = n.y;
    const double distCx = pos.x - xc;
    const double distCy = pos.y - yc;
    double nfac, dfac, A, B, C;
    bool solveForX;
    if (std::abs(nx) > std::abs(ny)) {
      solveForX = false;
      nfac = ny / nx;
      dfac = distToPlane / nx;
      B = distCy - nfac * distCx;
      C = (2. * distCx + dfac) * dfac;
    } else {
      solveForX = true;
      nfac = nx / ny;
      dfac = distToPlane / ny;
      B = distCx - nfac * distCy;
      C = (2. * distCy + dfac) * dfac;
    }
    B -= nfac * dfac;
    B *= 2;
    A = 1. + nfac * nfac;
    double q1, q2;
    if (!realQuad(A, B, C, q1, q2))
      return c;
    double dx1, dx2, dy1, dy2;
    if (solveForX) {
      dx1 = q1;
      dx2 = q2;
      dy1 = dfac - nfac * dx1;
      dy2 = dfac - nfac * dx2;
    } else {
      dy1 = q1;
      dy2 = q2;
      dx1 = dfac - nfac * dy1;
      dx2 = dfac - nfac * dy2;
    }
    // chooseSolution (propDir along / opposite; mathSSE::samesign = equal sign bits)
    const double momProj1 = dir.x * dx1 + dir.y * dy1;
    const double momProj2 = dir.x * dx2 + dir.y * dy2;
    const double propSign = along ? 1 : -1;
    double dx, dy;
    if (std::signbit(momProj1) != std::signbit(momProj2)) {
      const bool one = std::signbit(momProj1) == std::signbit(propSign);
      dx = one ? dx1 : dx2;
      dy = one ? dy1 : dy2;
    } else if (std::signbit(momProj1) == std::signbit(propSign)) {
      const bool one = (dx1 * dx1 + dy1 * dy1) < (dx2 * dx2 + dy2 * dy2);
      dx = one ? dx1 : dx2;
      dy = one ? dy1 : dy2;
    } else
      return c;
    const double dMag = std::sqrt(dx * dx + dy * dy);
    double sinAlpha = 0.5 * dMag * rho;
    if (std::abs(sinAlpha) > 1.)
      sinAlpha = std::copysign(1., sinAlpha);
    c.status = kXingOk;
    c.s = propSign * 2. / (rho * sinTheta) * std::asin(sinAlpha);
    // position(theS)
    c.x = float(pos.x + dx);
    c.y = float(pos.y + dy);
    c.z = float(pos.z + c.s * cosTheta);
    // direction(theS)
    double tmp = 0.5 * dMag * rho;
    if (c.s < 0)
      tmp = -tmp;
    double sinPhi = 1. - (tmp * tmp);
    if (sinPhi < 0)
      sinPhi = 0.;
    sinPhi = 2. * tmp * std::sqrt(sinPhi);
    const double cosPhi = 1. - 2. * (tmp * tmp);
    c.dx = float(dir.x * cosPhi - dir.y * sinPhi);
    c.dy = float(dir.x * sinPhi + dir.y * cosPhi);
    c.dz = dir.z;
    return c;
  }

  // ---- HelixArbitraryPlaneCrossing (+ HelixArbitraryPlaneCrossing2Order): tilted planes ----
  struct Helix2Order {  // HelixArbitraryPlaneCrossing2Order
    double x0, y0, z0, cosPhi0, sinPhi0, cosTheta, sinThetaI, rho;
  };
  // Plane::localZ(GlobalPoint(x, y, z)) (float)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE float planeLocalZ(
      ::mkfitdev::pca::F3 pp, ::mkfitdev::pca::F3 n, double x, double y, double z) {
    return n.x * (float(x) - pp.x) + n.y * (float(y) - pp.y) + n.z * (float(z) - pp.z);
  }
  // pathLength; dir: +1 along, -1 opposite, 0 any
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE bool helix2OrderPath(
      Helix2Order const& h, ::mkfitdev::pca::F3 pp, ::mkfitdev::pca::F3 n, int dir, double& path) {
    const double nPx = n.x, nPy = n.y, nPz = n.z;
    const double cP = planeLocalZ(pp, n, h.x0, h.y0, h.z0);
    const double ceq1 = h.rho * (nPx * h.sinPhi0 - nPy * h.cosPhi0);
    const double ceq2 = nPx * h.cosPhi0 + nPy * h.sinPhi0 + nPz * h.cosTheta * h.sinThetaI;
    const double ceq3 = cP;
    double dS1, dS2;
    constexpr double fltMin = std::numeric_limits<float>::min();
    if (std::abs(ceq1) > fltMin) {
      const double deq1 = ceq2 * ceq2;
      const double deq2 = ceq1 * ceq3;
      if (std::abs(deq1) < fltMin || std::abs(deq2 / deq1) > 1.e-6) {
        const double deq = deq1 + 2 * deq2;
        if (deq < 0.)
          return false;
        const double ceq = ceq2 + std::copysign(std::sqrt(deq), ceq2);
        dS1 = (ceq / ceq1) * h.sinThetaI;
        dS2 = -2. * (ceq3 / ceq) * h.sinThetaI;
      } else {
        const double ceq = (ceq2 / ceq1) * h.sinThetaI;
        double deq = deq2 / deq1;
        deq *= (1 - 0.5 * deq);
        dS1 = -ceq * deq;
        dS2 = ceq * (2 + deq);
      }
    } else {
      dS1 = dS2 = -(ceq3 / ceq2) * h.sinThetaI;
    }
    // solutionByDirection
    bool valid = false;
    path = 0;
    if (dir == 0) {
      valid = true;
      path = std::abs(dS1) < std::abs(dS2) ? dS1 : dS2;
    } else {
      const double propSign = dir;
      double s1 = propSign * dS1, s2 = propSign * dS2;
      if (s1 > s2) {
        const double t = s1;
        s1 = s2;
        s2 = t;
      }
      if ((s1 < 0) & (s2 >= 0)) {
        valid = true;
        path = propSign * s2;
      } else if (s1 >= 0) {
        valid = true;
        path = propSign * s1;
      }
    }
    if (notFinite(path))
      valid = false;
    return valid;
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void helix2OrderPos(Helix2Order const& h, double s, double* x) {
    const double st = s / h.sinThetaI;
    x[0] = h.x0 + (h.cosPhi0 - (st * 0.5 * h.rho) * h.sinPhi0) * st;
    x[1] = h.y0 + (h.sinPhi0 + (st * 0.5 * h.rho) * h.cosPhi0) * st;
    x[2] = h.z0 + st * h.cosTheta * h.sinThetaI;
  }
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE void helix2OrderDir(Helix2Order const& h, double s, double* d) {
    const double dph = s * h.rho / h.sinThetaI;
    d[0] = h.cosPhi0 - (h.sinPhi0 + 0.5 * dph * h.cosPhi0) * dph;
    d[1] = h.sinPhi0 + (h.cosPhi0 - 0.5 * dph * h.sinPhi0) * dph;
    d[2] = h.cosTheta * h.sinThetaI;
  }
  // HelixArbitraryPlaneCrossing(point, direction, curvature, propDir).pathLength(plane), position(s), direction(s)
  ALPAKA_FN_HOST_ACC MKFITDEV_NAV_INLINE Crossing arbitraryPlaneCrossing(::mkfitdev::pca::F3 pos,
                                                                         ::mkfitdev::pca::F3 dir,
                                                                         float curvature,
                                                                         bool along,
                                                                         ::mkfitdev::pca::F3 pp,
                                                                         ::mkfitdev::pca::F3 n) {
    Crossing c;
    const double px = dir.x, py = dir.y, pz = dir.z;
    const double pt2 = px * px + py * py;
    const double p2 = pt2 + pz * pz;
    const double pI = 1. / std::sqrt(p2);
    const double ptI = 1. / std::sqrt(pt2);
    const double x0 = pos.x, y0 = pos.y, z0 = pos.z, rho = curvature;
    const double cosPhi0 = px * ptI, sinPhi0 = py * ptI, cosTheta = pz * pI, sinTheta = pt2 * ptI * pI;
    const Helix2Order q0{x0, y0, z0, cosPhi0, sinPhi0, cosTheta, p2 * pI * ptI, rho};  // the start quadratic
    auto posD = [&](double s, double* x) MKFITDEV_NAV_LAMBDA_INLINE {
      const double dPhi = s * rho * sinTheta;
      if (std::abs(dPhi) > 1.e-4) {
        double sd, cd;
        ::mkfitdev::vdt::fast_sincos(dPhi, sd, cd);
        const double o = 1. / rho;
        x[0] = x0 + (-sinPhi0 * (1. - cd) + cosPhi0 * sd) * o;
        x[1] = y0 + (cosPhi0 * (1. - cd) + sinPhi0 * sd) * o;
        x[2] = z0 + s * cosTheta;
      } else
        helix2OrderPos(q0, s, x);
    };
    auto dirD = [&](double s, double* d) MKFITDEV_NAV_LAMBDA_INLINE {
      const double dPhi = s * rho * sinTheta;
      if (std::abs(dPhi) > 1.e-4) {
        double sd, cd;
        ::mkfitdev::vdt::fast_sincos(dPhi, sd, cd);
        d[0] = cosPhi0 * cd - sinPhi0 * sd;
        d[1] = sinPhi0 * cd + cosPhi0 * sd;
        d[2] = cosTheta / sinTheta;
      } else
        helix2OrderDir(q0, s, d);
    };
    // pathLength
    constexpr int maxIterations = 20;
    const float ppMag = std::sqrt(pp.x * pp.x + pp.y * pp.y + pp.z * pp.z);
    const float maxNumDz = 5.e-7f * ppMag;
    const float safeMaxDist = (1.e-4f > maxNumDz ? 1.e-4f : maxNumDz);
    double dSTotal = 0;
    if (!(std::abs(planeLocalZ(pp, n, x0, y0, z0)) < safeMaxDist)) {
      const int propSign = along ? 1 : -1;
      if (!helix2OrderPath(q0, pp, n, propSign, dSTotal))
        return c;
      double xnew[3];
      posD(dSTotal, xnew);
      if ((dSTotal >= 0 ? 1 : -1) != propSign)
        return c;
      int iteration = maxIterations;
      while (std::abs(planeLocalZ(pp, n, xnew[0], xnew[1], xnew[2])) > safeMaxDist) {
        if (--iteration == 0)
          return c;
        double pnew[3];
        dirD(dSTotal, pnew);
        const Helix2Order q{xnew[0], xnew[1], xnew[2], pnew[0], pnew[1], cosTheta, 1. / sinTheta, rho};
        double dS2;
        if (!helix2OrderPath(q, pp, n, 0, dS2))
          return c;
        dSTotal += dS2;
        if ((dSTotal >= 0 ? 1 : -1) != propSign)
          return c;
        posD(dSTotal, xnew);
      }
    }
    c.status = kXingOk;
    c.s = dSTotal;
    double x[3], d[3];
    posD(dSTotal, x);
    dirD(dSTotal, d);
    c.x = float(x[0]);
    c.y = float(x[1]);
    c.z = float(x[2]);
    c.dx = float(d[0]);
    c.dy = float(d[1]);
    c.dz = float(d[2]);
    return c;
  }

}  // namespace mkfitdev::navdev

#endif
