#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletHoleCheck_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletHoleCheck_h

// Hit-expectation tests of the short tracks of the inner-tracker-only CA (pixelTopology::Phase2,
// tripletHoleCheck), run after Kernel_classifyTracks on the 3- and 4-hit tracks of quality >= strict.
//
// The CA keeps a triplet where a track crosses only three active pixel layers: through a gap between
// modules, or where the layout has only three layers along the line. The tests check that expectation
// with the helix through the hits (for a 4-hit track through its first, second and last hit):
//  - holes: every module the helix crosses on a layer without a hit of the track, and
//  - overlaps: every other module it crosses on a layer with a hit,
//    at least `margin` inside the module edges, must have a rechit within |dx| < dx, |dy| < dy (local
//    coordinates) of the crossing. A real track leaves a hit there (one the CA did not attach, or the second
//    hit of an overlap); a combinatorial triplet usually does not.
//  - cluster length: for each hit, the cluster length along local y (SoA size / 8 - 1, in pixels) minus the
//    length of a straight crossing along the track direction (thickness x |tan| / pitchY) must be within
//    [clusterYLo, clusterYHi]: hits of different particles combined into a triplet rarely match its
//    direction. Edge and capped clusters are not tested.
//  - |tip| (3-hit tracks only): below maxTip[3 * region + class], region 0 for |eta| < 0.9 and 1 above, class 0
//    for BPix1-BPix2-next layer, 1 from BPix1 with a skipped layer, 2 without a BPix1 hit (negative = no cut):
//    the fake fraction of triplets grows with |tip| at a rate that depends on the hit pattern and region.
// A track failing a test is demoted to dup; a failing 3-hit track is flagged in `suspect` (with its quality)
// for the vertex rescue of CAZVertexHisto.h.
//
// Coordinates: the rechit SoA global positions are beam-spot subtracted while the module frames are in the
// global frame; the offset is recovered from the track's first hit (its local position through its module
// frame minus its global position), so no beam spot is needed here.
//
// The helix is parametrised relative to the hits (no circle centre and no 1/curvature): the signed curvature
// from the chords of the first, second and last hit, the arc length of each hit from the first one, the
// tangent at each hit from the chord to its neighbour rotated by half the turning angle, and dz/ds over the
// full lever arm. Every prediction is propagated from the nearest hit, so that everything stays finite and
// precise in single precision down to zero curvature (TeV tracks).

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHoleCheck {

  // CA layer of module m (-1 if none), from the layer start table cl
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int layerOfModule(::reco::CALayersSoAConstView const& cl, uint32_t m) {
    const int nLayers = cl.metadata().size() - 1;
    for (int l = 0; l < nLayers; ++l)
      if (m >= cl[l].layerStarts() && m < cl[l + 1].layerStarts())
        return l;
    return -1;
  }

  // hit-pattern class of a 3-hit track from its CA layers: 0 = BPix1, BPix2 and the next layer (BPix3 or
  // TFPX1 on either side), 1 = from BPix1 with a skipped layer, 2 = no BPix1 hit
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int tripletClass(int l0, int l1, int l2) {
    if (l0 != 0)
      return 2;
    return (l1 == 1 && (l2 == 2 || l2 == 4 || l2 == 16)) ? 0 : 1;
  }

  template <typename TrackerTraits>
  class Kernel_tripletHoleCheck {
  public:
    using HitsView = caStructures::HitsViewT<TrackerTraits>;
    using ModulesView = caStructures::ModulesViewT<TrackerTraits>;
    using HitContainer = caStructures::SequentialContainer;

    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ::reco::TrackSoAView tracks_view,
                                  HitContainer const* __restrict__ foundNtuplets,
                                  HitsView hh,
                                  ModulesView mm,
                                  ::reco::CAModulesConstView cm,
                                  ::reco::CALayersSoAConstView cl,
                                  caStructures::TripletHoleCheckParams p,
                                  uint8_t* __restrict__ suspect) const {
      using Quality = ::pixelTrack::Quality;
      const int nLayers = cl.metadata().size() - 1;
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        const auto nh = foundNtuplets->size(it);
        if (nh == 0)
          break;  // guard
        if ((nh != 3 && nh != 4) || tracks_view[it].quality() < Quality::strict)
          continue;
        const int n = nh;

        // the hits: module, layer, global position in the detector frame
        float x[4], y[4], z[4];
        int mod[4], lay[4];
        auto hp = foundNtuplets->begin(it);
        for (int k = 0; k < n; ++k) {
          const auto h = hp[k];
          mod[k] = hh[h].detectorIndex();
          lay[k] = layerOfModule(cl, mod[k]);
          x[k] = hh[h].xGlobal();
          y[k] = hh[h].yGlobal();
          z[k] = hh[h].zGlobal();
        }
        {  // beam-spot offset: true global = SoA global + offset (from the first hit's local position)
          const auto h = hp[0];
          float gx, gy, gz;
          cm.detFrame(mod[0]).toGlobal(hh[h].xLocal(), hh[h].yLocal(), 0.f, gx, gy, gz);
          const float ox = gx - x[0], oy = gy - y[0], oz = gz - z[0];
          for (int k = 0; k < n; ++k) {
            x[k] += ox;
            y[k] += oy;
            z[k] += oz;
          }
        }
        const int il = n - 1;

        // signed curvature from the chords of hits 0, 1 and the last one (Menger), > 0 counter-clockwise
        const float ax = x[1] - x[0], ay = y[1] - y[0], bx = x[il] - x[1], by = y[il] - y[1];
        const float cxh = x[il] - x[0], cyh = y[il] - y[0];
        const float la = alpaka::math::sqrt(acc, ax * ax + ay * ay), lb = alpaka::math::sqrt(acc, bx * bx + by * by),
                    lc = alpaka::math::sqrt(acc, cxh * cxh + cyh * cyh);
        if (!(la * lb * lc > 0.f))
          continue;
        const float kappa = 2.f * (ax * by - ay * bx) / (la * lb * lc);
        // arc length of a chord of length L: L asin(kappa L / 2) / (kappa L / 2)
        auto arcOfChord = [&acc, kappa](float L, bool& ok) {
          const float u = 0.5f * kappa * L;
          if (alpaka::math::abs(acc, u) >= 1.f) {
            ok = false;
            return L;
          }
          return alpaka::math::abs(acc, u) < 1e-3f ? L * (1.f + u * u / 6.f) : L * alpaka::math::asin(acc, u) / u;
        };
        bool okArc = true;
        float sh[4];  // arc length of each hit from hit 0
        sh[0] = 0.f;
        for (int k = 1; k < n; ++k) {
          const float dx = x[k] - x[0], dy = y[k] - y[0];
          sh[k] = arcOfChord(alpaka::math::sqrt(acc, dx * dx + dy * dy), okArc);
        }
        bool ordered = okArc;
        for (int k = 1; k < n; ++k)
          ordered = ordered && sh[k] > sh[k - 1];
        if (!ordered)
          continue;
        // unit tangent at each hit
        float tx[4], ty[4];
        for (int k = 0; k < n; ++k) {
          const int a0 = k < il ? k : k - 1, a1 = a0 + 1;
          const float dx = x[a1] - x[a0], dy = y[a1] - y[a0];
          const float L = alpaka::math::sqrt(acc, dx * dx + dy * dy);
          const float half = 0.5f * kappa * (sh[a1] - sh[a0]) * (k < il ? -1.f : 1.f);
          const float c = alpaka::math::cos(acc, half), sn = alpaka::math::sin(acc, half);
          tx[k] = (dx * c - dy * sn) / L;
          ty[k] = (dx * sn + dy * c) / L;
        }
        // dz/ds over the full lever arm (the slope of one short interval, e.g. between two nearby disks, carries the
        // hit resolution over a tiny lever arm and predicts z worse)
        const float cotTheta = (z[il] - z[0]) / sh[il];
        // the reference hit of a prediction, copied out of the per-hit arrays once per layer (no runtime index into
        // them in the propagation, which would put the arrays on the stack)
        struct Ref {
          float x, y, z, tx, ty, s;
        };
        auto refOf = [&](int k) {
          Ref r{x[0], y[0], z[0], tx[0], ty[0], sh[0]};
          for (int j = 1; j < 4; ++j)
            if (j == k)
              r = Ref{x[j], y[j], z[j], tx[j], ty[j], sh[j]};
          return r;
        };
        // propagate from the reference hit by the arc ds: position and unit transverse direction
        auto prop = [&acc, kappa, cotTheta](
                        Ref const& k, float ds, float& px, float& py, float& pz, float& dxo, float& dyo) {
          // turning angle u = kappa ds: series below 0.05 (error ~u^6/720), which covers the short steps to the
          // module planes, sin/cos only for the long low-pT propagations between layers
          const float u = kappa * ds;
          float S, C, cu, su;
          if (alpaka::math::abs(acc, u) < 0.05f) {
            const float u2 = u * u;
            S = ds * (1.f - u2 * (1.f / 6.f - u2 / 120.f));
            C = ds * u * (0.5f - u2 * (1.f / 24.f - u2 / 720.f));
            cu = 1.f - u2 * (0.5f - u2 / 24.f);
            su = u * (1.f - u2 * (1.f / 6.f - u2 / 120.f));
          } else {
            su = alpaka::math::sin(acc, u);
            cu = alpaka::math::cos(acc, u);
            S = su / kappa;
            C = (1.f - cu) / kappa;
          }
          const float nx = -k.ty, ny = k.tx;  // left normal
          px = k.x + k.tx * S + nx * C;
          py = k.y + k.ty * S + ny * C;
          pz = k.z + cotTheta * ds;
          dxo = k.tx * cu + nx * su;
          dyo = k.ty * cu + ny * su;
        };
        // the outgoing half turn, from the point of closest approach to the beam line
        const float sLo = -alpaka::math::max(acc, 0.f, x[0] * tx[0] + y[0] * ty[0]);
        const float sHi =
            alpaka::math::abs(acc, kappa) > 1e-6f ? sLo + 3.14159265f / alpaka::math::abs(acc, kappa) : 1e30f;

        bool miss = false;
        // |tip| per hit-pattern class and region (3-hit tracks)
        if (n == 3) {
          const int cls = tripletClass(lay[0], lay[1], lay[2]);
          const int region = alpaka::math::abs(acc, cotTheta) < 1.0265f ? 0 : 1;  // |eta| 0.9
          const float maxTip = p.maxTip[3 * region + cls];
          if (maxTip >= 0.f && alpaka::math::abs(acc, ::reco::tip(tracks_view, it)) > maxTip)
            miss = true;
        }
        // cluster length along local y vs the track direction at the hit
        for (int k = 0; k < n && !miss; ++k) {
          const auto h = hp[k];
          const int ys = hh[h].clusterSizeY();
          const float th = cm[mod[k]].thickness(), py = cm[mod[k]].pitchY();
          if (ys <= 0 || ys >= int(TrackerTraits::maxSizeCluster) || !(th > 0.f) || !(py > 0.f))
            continue;
          auto const& r = cm.detFrame(mod[k]).rotation();
          float lx, ly, lz;
          r.multiply(tx[k], ty[k], cotTheta, lx, ly, lz);
          if (alpaka::math::abs(acc, lz) < 1e-4f)
            continue;
          const float expected = th * alpaka::math::abs(acc, ly / lz) / py;
          const float d = float(ys) / 8.f - 1.f - expected;
          if (d < p.clusterYLo || d > p.clusterYHi)
            miss = true;
        }
        // holes and overlaps
        for (int l = 0; l < nLayers && !miss; ++l) {
          const bool hasHit = (l == lay[0] || l == lay[1] || l == lay[2] || (n == 4 && l == lay[3]));
          const uint32_t m0 = cl[l].layerStarts(), m1 = cl[l + 1].layerStarts();
          if (m1 <= m0)
            continue;
          auto const& f0 = cm.detFrame(m0);
          // crossing with the layer at the radius (barrel) or z (disk) of its first module, from the nearest hit
          int kr = 0;
          Ref ref{};
          float ds = 0.f;
          bool found = false;
          if (cl[l].isBarrel()) {
            const float rl = alpaka::math::sqrt(acc, f0.x() * f0.x() + f0.y() * f0.y());
            float best = 1e30f;
            for (int k = 0; k < n; ++k) {
              const float dr = alpaka::math::abs(acc, alpaka::math::sqrt(acc, x[k] * x[k] + y[k] * y[k]) - rl);
              if (dr < best) {
                best = dr;
                kr = k;
              }
            }
            // straight-line guess, then one Newton step on |p(ds)|^2 = rl^2 along the helix: the tangent-line guess
            // is off by O(kappa ds^2), the step leaves O(kappa^2 ds^3)
            ref = refOf(kr);
            const float B = ref.x * ref.tx + ref.y * ref.ty, Cq = ref.x * ref.x + ref.y * ref.y - rl * rl;
            const float disc = B * B - Cq;
            if (disc < 0.f)
              continue;
            const float sq = alpaka::math::sqrt(acc, disc), r1 = -B + sq, r2 = -B - sq;
            ds = alpaka::math::abs(acc, r1) < alpaka::math::abs(acc, r2) ? r1 : r2;
            float px, py, pz, dxo, dyo;
            prop(ref, ds, px, py, pz, dxo, dyo);
            const float fp = 2.f * (px * dxo + py * dyo);
            if (alpaka::math::abs(acc, fp) >= 1e-6f) {
              ds -= (px * px + py * py - rl * rl) / fp;
              found = true;
            }
          } else {
            float best = 1e30f;
            for (int k = 0; k < n; ++k)
              if (alpaka::math::abs(acc, z[k] - f0.z()) < best) {
                best = alpaka::math::abs(acc, z[k] - f0.z());
                kr = k;
              }
            if (alpaka::math::abs(acc, cotTheta) < 1e-4f)
              continue;
            ref = refOf(kr);
            ds = (f0.z() - ref.z) / cotTheta;
            found = true;
          }
          if (!found || ref.s + ds < sLo || ref.s + ds > sHi)
            continue;
          float px, py, pz, dxo, dyo;
          prop(ref, ds, px, py, pz, dxo, dyo);
          for (uint32_t m = m0; m < m1 && !miss; ++m) {
            if (hasHit && (int(m) == mod[0] || int(m) == mod[1] || int(m) == mod[2] || (n == 4 && int(m) == mod[3])))
              continue;  // the module of the track's own hit
            auto const& f = cm.detFrame(m);
            const float hw = cm[m].halfWidth(), hl = cm[m].halfLength();
            const float ex = px - f.x(), ey = py - f.y(), ez = pz - f.z();
            const float reach = hw + hl + 2.f;  // modules at alternating radii / disk faces
            if (ex * ex + ey * ey + ez * ez > reach * reach)
              continue;
            // crossing with the module plane: two Newton steps along the helix from the layer crossing
            auto const& r = f.rotation();
            float dsm = ds, qx = px, qy = py, qz = pz, ux = dxo, uy = dyo;
            bool ok = true;
            const float norm = 1.f / alpaka::math::sqrt(acc, 1.f + cotTheta * cotTheta);
            for (int iter = 0; iter < 2; ++iter) {
              const float t3x = ux * norm, t3y = uy * norm, t3z = cotTheta * norm;
              const float nd = r.zx() * t3x + r.zy() * t3y + r.zz() * t3z;
              if (alpaka::math::abs(acc, nd) < 1e-4f) {
                ok = false;
                break;
              }
              const float t = (r.zx() * (f.x() - qx) + r.zy() * (f.y() - qy) + r.zz() * (f.z() - qz)) / nd;
              dsm += t * norm;  // transverse path
              prop(ref, dsm, qx, qy, qz, ux, uy);
            }
            if (!ok || ref.s + dsm < sLo || ref.s + dsm > sHi)
              continue;
            float lx, ly, lz;
            f.toLocal(qx, qy, qz, lx, ly, lz);
            if (alpaka::math::abs(acc, lx) > hw - p.margin || alpaka::math::abs(acc, ly) > hl - p.margin)
              continue;  // not crossed, or too close to the edge to expect a hit
            // a compatible rechit on the crossed module?
            bool hit = false;
            const uint32_t h0 = caStructures::moduleStartOf(mm, m), h1 = caStructures::moduleStartOf(mm, m + 1);
            for (uint32_t h = h0; h < h1 && !hit; ++h)
              hit = alpaka::math::abs(acc, hh[h].xLocal() - lx) < p.dx &&
                    alpaka::math::abs(acc, hh[h].yLocal() - ly) < p.dy;
            if (!hit)
              miss = true;
          }
        }
        if (miss) {
          // vertex rescue: remember the quality of a failing 3-hit track (CAZVertexHisto.h may restore it)
          if (suspect && n == 3)
            suspect[it] = uint8_t(tracks_view[it].quality()) + 1;
          tracks_view[it].quality() = Quality::dup;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caHoleCheck

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CATripletHoleCheck_h
