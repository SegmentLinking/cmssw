#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPostMath_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPostMath_h

// The per-seed math of the triplet MLP and of the steps after the fit (host and device code):
//  - FillCircle: the circle through three hits, the transverse arc length s from the first, the point at s, the
//    crossing of a radius (the pixel fill);
//  - extensionLocalResidual: an OT hit's residual in its module frame from the hit covariance (the outer-tracker
//    extension with extensionLocal);
//  - computeTripletFeatures / evaluateTripletMlp: the triplet MLP's features and network, the weights by pointer.
// Ported from mkFit's SeederPostMath.h.

#include <algorithm>
#include <cmath>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// libm::atan2: the same values on every backend (MkSeederLibm.h)
#include "MkSeederLibm.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  constexpr float kPi = 3.14159265358979323846f;  // mkfit::Const::PI
  constexpr float kTwoPi = 6.28318530717958647692f, kInvTwoPi = 0.15915494309189533577f;
  template <typename TAcc>
  ALPAKA_FN_ACC inline float wrapPhiRound(const TAcc &acc, float d) {
    return d - kTwoPi * alpaka::math::round(acc, d * kInvTwoPi);
  }

  // The circle in (x, y) through the hits a, b, c (b between them); s is the transverse arc length from a, positive
  // towards b. A (near) straight line: lin, with the unit direction a -> c.
  struct FillCircle {
    float ax = 0, ay = 0, a2 = 0;
    float xc = 0, yc = 0, R = 1e9f, sgn = 1, ang0 = 0, ux = 0, uy = 0;
    bool lin = false, ok = false;

    template <typename TAcc>
    ALPAKA_FN_ACC void make(const TAcc &acc, float ax_, float ay_, float bx, float by, float cx, float cy) {
      ax = ax_, ay = ay_;
      const float d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
      a2 = ax * ax + ay * ay;
      const float b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
      const bool line = alpaka::math::abs(acc, d) < 1e-6f;
      xc = 0, yc = 0, R = 1e9f, sgn = 1, ang0 = 0, ux = 0, uy = 0;
      if (!line) {
        xc = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
        yc = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
        R = alpaka::math::sqrt(acc, (ax - xc) * (ax - xc) + (ay - yc) * (ay - yc));
        ang0 = libm::atan2(acc, ay - yc, ax - xc);
        sgn = wrapPhiRound(acc, libm::atan2(acc, by - yc, bx - xc) - ang0) >= 0 ? 1.f : -1.f;
      }
      lin = line || R > 1e5f;
      ok = true;
      if (lin) {
        const float lu = alpaka::math::sqrt(acc, (cx - ax) * (cx - ax) + (cy - ay) * (cy - ay));
        if (lu < 1e-6f) {
          ok = false;
          return;
        }
        ux = (cx - ax) / lu, uy = (cy - ay) / lu;
      }
    }
    template <typename TAcc>
    ALPAKA_FN_ACC float arcLengthOf(const TAcc &acc, float x, float y) const {
      return lin ? (x - ax) * ux + (y - ay) * uy : sgn * wrapPhiRound(acc, libm::atan2(acc, y - yc, x - xc) - ang0) * R;
    }
    template <typename TAcc>
    ALPAKA_FN_ACC void positionAtArcLength(const TAcc &acc, float s, float &x, float &y) const {
      if (lin)
        x = ax + s * ux, y = ay + s * uy;
      else {
        const float an = ang0 + sgn * s / R;
        x = xc + R * alpaka::math::cos(acc, an), y = yc + R * alpaka::math::sin(acc, an);
      }
    }
    // the point at radius r with s_lo < s < s_max (the smallest such s); false if none
    template <typename TAcc>
    ALPAKA_FN_ACC bool positionAtR(
        const TAcc &acc, float r, float s_lo, float s_max, float &x, float &y, float &s) const {
      float cand[2][2];
      if (lin) {
        const float b = ax * ux + ay * uy, q = a2 - r * r, disc = b * b - q;
        if (disc < 0)
          return false;
        const float sq = alpaka::math::sqrt(acc, disc);
        cand[0][0] = ax + (-b - sq) * ux, cand[0][1] = ay + (-b - sq) * uy;
        cand[1][0] = ax + (-b + sq) * ux, cand[1][1] = ay + (-b + sq) * uy;
      } else {
        const float dc = alpaka::math::sqrt(acc, xc * xc + yc * yc);
        if (dc < 1e-6f)
          return false;
        const float aa = (r * r - R * R + dc * dc) / (2 * dc), h2 = r * r - aa * aa;
        if (h2 < 0)
          return false;
        const float h = alpaka::math::sqrt(acc, h2), ex = xc / dc, ey = yc / dc;
        cand[0][0] = aa * ex - h * ey, cand[0][1] = aa * ey + h * ex;
        cand[1][0] = aa * ex + h * ey, cand[1][1] = aa * ey - h * ex;
      }
      bool found = false;
      for (int k = 0; k < 2; ++k) {
        const float sk = arcLengthOf(acc, cand[k][0], cand[k][1]);
        if (sk > s_lo && sk < s_max && (!found || sk < s))
          x = cand[k][0], y = cand[k][1], s = sk, found = true;
      }
      return found;
    }
  };

  // An OT hit at (hx, hy, hz), its covariance E = {xx, yy, xy, xz, yz, zz}: the module frame (ux along phi, uy the long
  // axis in r-z), the helix H (any type with positionAtR(r, x, y, q, s3)) taken linear in r between hr -+ 1 cm onto the module
  // plane; dx, dy the residual. False if the helix does not reach it.
  template <typename TAcc, typename Helix>
  ALPAKA_FN_ACC inline bool extensionLocalResidual(
      const TAcc &acc, const Helix &H, float hx, float hy, float hz, const float *E, float &dx, float &dy) {
    const float hr = alpaka::math::sqrt(acc, hx * hx + hy * hy), cr = hx / hr, sr = hy / hr;
    const float vr = cr * cr * E[0] + sr * sr * E[1] + 2 * cr * sr * E[2];
    const float crz = cr * E[3] + sr * E[4];
    const float al = 0.5f * libm::atan2(acc, 2 * crz, vr - E[5]);
    const float ux[3] = {-sr, cr, 0.f},
                uy[3] = {alpaka::math::cos(acc, al) * cr, alpaka::math::cos(acc, al) * sr, alpaka::math::sin(acc, al)};
    const float nn[3] = {ux[1] * uy[2] - ux[2] * uy[1], ux[2] * uy[0] - ux[0] * uy[2], ux[0] * uy[1] - ux[1] * uy[0]};
    float ax, ay, aq, as, bx, by, bq, bs;
    if (!H.positionAtR(acc, hr - 1.f, ax, ay, aq, as) || !H.positionAtR(acc, hr + 1.f, bx, by, bq, bs))
      return false;
    const float fa = (ax - hx) * nn[0] + (ay - hy) * nn[1] + (aq - hz) * nn[2];
    const float fb = (bx - hx) * nn[0] + (by - hy) * nn[1] + (bq - hz) * nn[2];
    if (fa == fb)
      return false;
    const float t = fa / (fa - fb);
    const float d[3] = {ax + t * (bx - ax) - hx, ay + t * (by - ay) - hy, aq + t * (bq - aq) - hz};
    dx = d[0] * ux[0] + d[1] * ux[1], dy = d[0] * uy[0] + d[1] * uy[1] + d[2] * uy[2];
    return true;
  }

  // The triplet MLP: features of the three hits (global x, y, z in crossing order, cluster rows /
  // columns as mkFit keeps them), the beam spot; f[19]. False for a degenerate triplet.
  constexpr int kTripletMlpNFeatures = 19, kTripletMlpNHidden1 = 32, kTripletMlpNHidden2 = 16;
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool computeTripletFeatures(const TAcc &acc,
                                                   const float x[3],
                                                   const float y[3],
                                                   const float z[3],
                                                   const int rows[3],
                                                   const int cols[3],
                                                   float bx,
                                                   float by,
                                                   float bz,
                                                   float *f,
                                                   const float *Pin = nullptr) {
    // Pin: the hits' atan2(y - by, x - bx) precomputed by the caller (the same values), else computed here
    float X[3], Y[3], R[3], P[3], Z[3];
    for (int i = 0; i < 3; ++i) {
      X[i] = x[i] - bx, Y[i] = y[i] - by, Z[i] = z[i] - bz;
      R[i] = alpaka::math::sqrt(acc, X[i] * X[i] + Y[i] * Y[i]), P[i] = Pin ? Pin[i] : libm::atan2(acc, Y[i], X[i]);
    }
    const float ax = X[0], ay = Y[0], qx = X[1], qy = Y[1], cx = X[2], cy = Y[2];
    const float d = 2.f * (ax * (qy - cy) + qx * (cy - ay) + cx * (ay - qy));
    float kap = 0.f, dca = 0.f;
    if (alpaka::math::abs(acc, d) > 1e-6f) {
      const float a2 = ax * ax + ay * ay, q2 = qx * qx + qy * qy, c2 = cx * cx + cy * cy;
      const float ox = (a2 * (qy - cy) + q2 * (cy - ay) + c2 * (ay - qy)) / d;
      const float oy = (a2 * (cx - qx) + q2 * (ax - cx) + c2 * (qx - ax)) / d;
      const float rr = alpaka::math::sqrt(acc, (ax - ox) * (ax - ox) + (ay - oy) * (ay - oy));
      kap = 1.f / rr;
      dca = alpaka::math::abs(acc, alpaka::math::sqrt(acc, ox * ox + oy * oy) - rr);
    }
    if (R[2] - R[0] == 0.f && Z[2] - Z[0] == 0.f)
      return false;
    const bool barrelish = alpaka::math::abs(acc, R[2] - R[0]) > 1e-3f;
    const float cot = barrelish ? (Z[2] - Z[0]) / (R[2] - R[0]) : 1e3f;
    const float z0 = Z[0] - cot * R[0];
    const float rz = Z[1] - (Z[0] + cot * (R[1] - R[0]));
    const float dp12 = alpaka::math::abs(acc, wrapPhiRound(acc, P[1] - P[0])),
                dp23 = alpaka::math::abs(acc, wrapPhiRound(acc, P[2] - P[1]));
    int n = 0;
    for (int i = 0; i < 3; ++i)
      f[n++] = R[i];
    for (int i = 0; i < 3; ++i)
      f[n++] = Z[i];
    f[n++] = kap;
    f[n++] = alpaka::math::log(acc, dca + 1e-4f);
    f[n++] = std::max(-30.f, std::min(30.f, z0));
    f[n++] = std::max(-5.f, std::min(5.f, rz));
    f[n++] = std::max(-20.f, std::min(20.f, cot));
    f[n++] = dp12;
    f[n++] = dp23;
    for (int i = 0; i < 3; ++i)
      f[n++] = (float)std::min(32, rows[i]);
    for (int i = 0; i < 3; ++i)
      f[n++] = (float)std::min(32, cols[i]);
    return n == kTripletMlpNFeatures;
  }
  // the z mirror fold of the features f of computeTripletFeatures (MkSeederTripletMlpWeights::zFold): the signed ones (the
  // hits' z, z0, the r-z residual, cot theta) times fold * sign(cot theta)
  ALPAKA_FN_HOST_ACC inline void foldTripletFeatures(float *f, float fold) {
    if (fold == 0.f)
      return;
    const float s = f[10] >= 0.f ? fold : -fold;
    f[3] *= s, f[4] *= s, f[5] *= s, f[8] *= s, f[9] *= s, f[10] *= s;
  }
  // the network: standardization -> relu (32) -> relu (16) -> sigmoid; w = mean, scale, W1, B1, W2, B2, W3, B3
  // (MkSeederTripletMlpWeights)
  template <typename TAcc>
  ALPAKA_FN_ACC inline float evaluateTripletMlp(const TAcc &acc,
                                                const float *feat,
                                                const float *mean,
                                                const float *scale,
                                                const float *W1,
                                                const float *B1,
                                                const float *W2,
                                                const float *B2,
                                                const float *W3,
                                                const float *B3) {
    float xn[kTripletMlpNFeatures];
    for (int i = 0; i < kTripletMlpNFeatures; ++i)
      xn[i] = (feat[i] - mean[i]) / scale[i];
    float h1[kTripletMlpNHidden1];
    for (int j = 0; j < kTripletMlpNHidden1; ++j) {
      float s = B1[j];
      for (int i = 0; i < kTripletMlpNFeatures; ++i)
        s += xn[i] * W1[i * kTripletMlpNHidden1 + j];
      h1[j] = s > 0.f ? s : 0.f;
    }
    float h2[kTripletMlpNHidden2];
    for (int k = 0; k < kTripletMlpNHidden2; ++k) {
      float s = B2[k];
      for (int j = 0; j < kTripletMlpNHidden1; ++j)
        s += h1[j] * W2[j * kTripletMlpNHidden2 + k];
      h2[k] = s > 0.f ? s : 0.f;
    }
    float o = B3[0];
    for (int k = 0; k < kTripletMlpNHidden2; ++k)
      o += h2[k] * W3[k];
    return 1.f / (1.f + alpaka::math::exp(acc, -o));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
