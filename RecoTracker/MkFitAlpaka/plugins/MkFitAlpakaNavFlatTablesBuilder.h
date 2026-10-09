#ifndef RecoTracker_MkFitAlpaka_plugins_MkFitAlpakaNavFlatTablesBuilder_h
#define RecoTracker_MkFitAlpaka_plugins_MkFitAlpakaNavFlatTablesBuilder_h

// Host builder of the flat layer tables of the portable compatibleDets search (interface/math/NavSearch.h), used once
// per TrackerRecoGeometryRecord IOV by MkFitAlpakaNavFlatTablesESProducer. Each layer is laid out the way its
// TkDetLayers builder makes it (rings, rods, sub-disks and their surfaces); a layer of any other layout is not
// supported and its calls are searched on the host.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/RodPlaneBuilderFromDet.h"
#include "RecoTracker/MkFitAlpaka/interface/navdev/NavFlatTables.h"

namespace mkfitdev::navdev {

  // One Phase2EndcapRing as Phase2EndcapRingBuilder (useBrothers) lays it out: basicComponents() = front lower
  // sensors, back lower sensors, front brothers, back brothers; front / back split at the mean z; sub-layer planes at
  // ForwardRingDiskBuilderFromDet's z (middle of the corner z range); PeriodicBinFinderInPhi per sub-layer.
  struct RingLayout {
    bool ok = false;
    bool single = false;              // Phase2EndcapSingleRing (pixel double disks): sub[0] only, no brothers
    const BoundDisk* disk = nullptr;  // the ring surface (tkDetUtil crossing + ring order)
    ReferenceCountingPointer<BoundDisk> ownDisk;  // a ring rebuilt from its dets (ForwardRingDiskBuilderFromDet)
    std::array<std::vector<const GeomDet*>, 2> sub, bro;
    std::array<Plane::PlanePointer, 2> plane;
    std::array<float, 2> phiOffset{}, phiStep{}, invPhiStep{};
    float ringR = 0, thetaMin = 0, thetaMax = 0;
  };

  // A Phase2OTBarrelRod (inner / outer sub-rods split at the mean r, brothers = upper sensors, sub-rod planes from
  // RodPlaneBuilderFromDet) or a PixelRod (dets in z).
  struct RodLayout {
    bool ok = false, stacked = false;
    std::array<std::vector<const GeomDet*>, 2> sub, bro;
    std::array<Plane::PlanePointer, 2> plane;
    std::vector<const GeomDet*> dets;  // PixelRod
  };

  // A TBPLayer (inner / outer rods split at the mean r, cylinders from CylinderBuilderFromDet) and, for a
  // Phase2OTtiltedBarrelLayer, its tilted rings per z side in construction order.
  struct BarrelLayout {
    bool ok = false;
    std::array<std::vector<const GeometricSearchDet*>, 2> rods;
    std::array<ReferenceCountingPointer<BoundCylinder>, 2> cyl;
    std::array<std::vector<RingLayout>, 2> rings;  // [negative z | positive z]
  };

  // Pixel endcap double disk: sub-disks of single rings.
  struct SubDiskLayout {
    float z = 0;
    std::vector<RingLayout> rings;
  };
  struct PixEndcapLayout {
    bool ok = false, doubleDisk = false;
    std::vector<SubDiskLayout> subs;
  };

  class NavFlatTablesBuilder {
  public:
    NavFlatTablesBuilder(TrackerTopology const& tTopo, NavFlatTables& out) : tTopo_(tTopo), out_(out) {}

    // Lays out one layer in the flat tables; false = not supported (searched on the host).
    bool addLayer(const DetLayer* layer) {
      auto [lit, isNew] = out_.layers.try_emplace(layer, std::array<int, 3>{{0, -1, 0}});
      if (!isNew)
        return lit->second[1] >= 0;
      const auto sub = layer->subDetector();
      if (GeomDetEnumerators::isBarrel(sub))
        addBarrel(layer, sub, lit->second);
      else
        addEndcap(layer, sub, lit->second);
      return lit->second[1] >= 0;
    }

  private:
    // the portable det plane of a CMSSW plane with rectangular bounds
    static void fillDetPlane(Plane const& plane, RectangularPlaneBounds const& rb, DetPlane& pl) {
      const auto gx = plane.toGlobal(LocalVector(1, 0, 0)), gy = plane.toGlobal(LocalVector(0, 1, 0)),
                 gz = plane.toGlobal(LocalVector(0, 0, 1));
      for (int i = 0; i < 3; ++i) {
        pl.pos[i] = i == 0 ? plane.position().x() : (i == 1 ? plane.position().y() : plane.position().z());
        pl.ax[i] = i == 0 ? gx.x() : (i == 1 ? gx.y() : gx.z());
        pl.ay[i] = i == 0 ? gy.x() : (i == 1 ? gy.y() : gy.z());
        pl.az[i] = i == 0 ? gz.x() : (i == 1 ? gz.y() : gz.z());
      }
      pl.halfWidth = rb.width() / 2;
      pl.halfLength = rb.length() / 2;
      pl.halfThickness = rb.thickness() / 2;
    }

    static void planeAxes(Plane const& plane, DetPlane& pl) {
      const auto gx = plane.toGlobal(LocalVector(1, 0, 0)), gy = plane.toGlobal(LocalVector(0, 1, 0)),
                 gz = plane.toGlobal(LocalVector(0, 0, 1));
      const auto& pp = plane.position();
      const float v[4][3] = {
          {pp.x(), pp.y(), pp.z()}, {gx.x(), gx.y(), gx.z()}, {gy.x(), gy.y(), gy.z()}, {gz.x(), gz.y(), gz.z()}};
      for (int i = 0; i < 3; ++i) {
        pl.pos[i] = v[0][i];
        pl.ax[i] = v[1][i];
        pl.ay[i] = v[2][i];
        pl.az[i] = v[3][i];
      }
    }

    // flat row of a det (added at first use)
    int detRow(const GeomDet* g) {
      auto [di, isNew] = detIndex_.try_emplace(g, int(out_.dets.size()));
      if (isNew) {
        auto const& pl = g->surface();
        NavDet d{};
        if (auto const* rb = dynamic_cast<const RectangularPlaneBounds*>(&pl.bounds()))
          fillDetPlane(pl, *rb, d.plane);
        else {
          ++out_.nonRect;
          d.plane.halfThickness = -1;  // never compatible
        }
        d.phi = float(pl.phi());
        d.phiSpanLo = pl.phiSpan().first;
        d.phiSpanHi = pl.phiSpan().second;
        d.posZ = g->position().z();
        d.posPerp = g->position().perp();
        d.normalZ = pl.normalVector().z();
        const auto o = pl.toLocal(GlobalPoint(0., 0., 0.));
        d.originLx = o.x();
        d.originLy = o.y();
        out_.dets.push_back(d);
        out_.detPtr.push_back(g);
      }
      return di->second;
    }

    NavRing ringRow(RingLayout const& R) {
      NavRing nr{};
      nr.diskZ = R.disk != nullptr ? R.disk->position().z() : 0.f;  // tilted barrel rings: no disk (not tkDetUtil)
      nr.ringR = R.ringR;
      nr.thetaMin = R.thetaMin;
      nr.thetaMax = R.thetaMax;
      nr.single = R.single;
      for (int s = 0; s < (R.single ? 1 : 2); ++s) {
        nr.subZ[s] = R.plane[s]->position().z();
        nr.phiOffset[s] = R.phiOffset[s];
        nr.invPhiStep[s] = R.invPhiStep[s];
        nr.n[s] = R.sub[s].size();
        nr.sub[s] = out_.idx.size();
        for (auto const* g : R.sub[s])
          out_.idx.push_back(detRow(g));
      }
      for (int s = 0; s < 2; ++s) {
        nr.bro[s] = -1;
        if (!R.single && !R.bro[s].empty()) {
          nr.bro[s] = out_.idx.size();
          for (auto const* g : R.bro[s])
            out_.idx.push_back(detRow(g));
        }
      }
      return nr;
    }

    void addBarrel(const DetLayer* layer, GeomDetEnumerators::SubDetector sub, std::array<int, 3>& entry) {
      BarrelLayout const& B = barrelOf(layer);
      const bool stacked = !GeomDetEnumerators::isInnerTracker(sub);
      bool good = B.ok;
      NavBarrel nb{};
      nb.stacked = stacked;
      std::vector<NavRod> rows;
      for (int s = 0; good && s < 2; ++s) {
        nb.cylR[s] = B.cyl[s]->radius();
        constexpr float kTwoPi = 2 * float(3.141592653589793238);
        nb.phiStep[s] = kTwoPi / float(B.rods[s].size());
        nb.invPhiStep[s] = 1.f / nb.phiStep[s];
        nb.phiOffset[s] = float(B.rods[s].front()->position().phi()) - 0.5f * nb.phiStep[s];
        nb.nRod[s] = B.rods[s].size();
        nb.rod0[s] = out_.rods.size() + rows.size();
        for (auto const* rod : B.rods[s]) {
          RodLayout const& R = rodOf(rod, stacked);
          if (!R.ok) {
            good = false;
            break;
          }
          NavRod nr{};
          nr.stacked = stacked;
          nr.phi = float(rod->surface().phi());
          nr.phiSpanLo = rod->surface().phiSpan().first;
          nr.phiSpanHi = rod->surface().phiSpan().second;
          if (stacked) {
            for (int t = 0; t < 2; ++t) {
              const int n = R.sub[t].size();
              nr.n[t] = n;
              nr.sub[t] = out_.idx.size();
              for (auto const* g : R.sub[t])
                out_.idx.push_back(detRow(g));
              nr.bro[t] = out_.idx.size();
              for (auto const* g : R.bro[t])
                out_.idx.push_back(detRow(g));
              // GenericBinFinderInZ
              nr.zs[t] = out_.zs.size();
              for (auto const* g : R.sub[t])
                out_.zs.push_back(g->position().z());
              for (int i = 0; i + 1 < n; ++i)
                out_.zs.push_back((R.sub[t][i]->position().z() + R.sub[t][i + 1]->position().z()) / 2.);
              const float* borders = out_.zs.data() + nr.zs[t] + n;
              nr.zOffset[t] = borders[0];
              nr.zStep[t] = (borders[n - 2] - borders[0]) / (n - 2);
              planeAxes(*R.plane[t], nr.plane[t]);
            }
          } else {
            const int n = R.dets.size();
            nr.n[0] = n;
            nr.sub[0] = out_.idx.size();
            for (auto const* g : R.dets)
              out_.idx.push_back(detRow(g));
            nr.bro[0] = nr.bro[1] = -1;
            // PeriodicBinFinderInZ
            const float zFirst = R.dets.front()->surface().position().z();
            nr.zStep[0] = (R.dets.back()->surface().position().z() - zFirst) / (n - 1);
            nr.zOffset[0] = zFirst - 0.5 * nr.zStep[0];
            planeAxes(static_cast<const Plane&>(rod->surface()), nr.plane[0]);
          }
          rows.push_back(nr);
        }
      }
      for (int z = 0; good && z < 2; ++z) {
        nb.ring0[z] = out_.rings.size();
        nb.nRing[z] = B.rings[z].size();
        for (auto const& R : B.rings[z])
          out_.rings.push_back(ringRow(R));
      }
      if (good) {
        out_.rods.insert(out_.rods.end(), rows.begin(), rows.end());
        entry = {{2, int(out_.barrels.size()), 1}};
        out_.barrels.push_back(nb);
      }
    }

    void addEndcap(const DetLayer* layer, GeomDetEnumerators::SubDetector sub, std::array<int, 3>& entry) {
      PixEndcapLayout const* P = GeomDetEnumerators::isInnerTracker(sub) ? &pixEndcapOf(layer) : nullptr;
      if (P != nullptr && P->doubleDisk) {
        if (P->ok) {
          std::vector<NavSubDisk> subs;
          for (auto const& d : P->subs) {
            NavSubDisk sd{d.z, int(out_.rings.size()), int(d.rings.size())};
            for (auto const& R : d.rings)
              out_.rings.push_back(ringRow(R));
            subs.push_back(sd);
          }
          entry = {{1, int(out_.subDisks.size()), int(subs.size())}};
          out_.subDisks.insert(out_.subDisks.end(), subs.begin(), subs.end());
        }
        return;
      }
      std::vector<RingLayout const*> rs;
      bool good = true;
      for (auto const* comp : layer->components()) {
        RingLayout const& R = ringOf(comp);
        good = good && R.ok && !R.single && R.disk != nullptr;
        rs.push_back(&R);
      }
      if (good) {
        std::vector<NavRing> rows;
        rows.reserve(rs.size());
        for (auto const* R : rs)
          rows.push_back(ringRow(*R));
        entry = {{0, int(out_.rings.size()), int(rows.size())}};
        out_.rings.insert(out_.rings.end(), rows.begin(), rows.end());
      }
    }

    // a ring from its basicComponents (front lowers, back lowers, front brothers, back brothers); false = not that
    // layout
    bool buildRing(std::vector<const GeomDet*> const& bc, RingLayout& e) const {
      if (bc.size() >= 2 && !tTopo_.isLower(bc.front()->geographicalId()) &&
          !tTopo_.isUpper(bc.front()->geographicalId())) {
        // no brothers (pixel Phase2EndcapRing): front / back split at the mean z
        double mz = 0;
        for (auto const* g : bc)
          mz += g->position().z();
        mz /= bc.size();
        for (auto const* g : bc)
          e.sub[std::abs(g->position().z()) < std::abs(mz) ? 0 : 1].push_back(g);
        if (e.sub[0].empty() || e.sub[1].empty())
          return false;
      } else {
        const size_t half = bc.size() / 2;
        if (bc.size() < 4 || bc.size() % 2 != 0)
          return false;
        double mz = 0, mzb = 0;
        for (size_t i = 0; i < half; ++i) {
          mz += bc[i]->position().z();
          mzb += bc[half + i]->position().z();
        }
        mz /= half;
        mzb /= half;
        for (size_t i = 0; i < half; ++i) {
          e.sub[std::abs(bc[i]->position().z()) < std::abs(mz) ? 0 : 1].push_back(bc[i]);
          e.bro[std::abs(bc[half + i]->position().z()) < std::abs(mzb) ? 0 : 1].push_back(bc[half + i]);
        }
        bool good = !e.sub[0].empty() && !e.sub[1].empty();
        for (int s = 0; s < 2 && good; ++s) {
          good = e.sub[s].size() == e.bro[s].size();
          for (size_t i = 0; good && i < e.sub[s].size(); ++i)
            good = tTopo_.stack(e.sub[s][i]->geographicalId()) == tTopo_.stack(e.bro[s][i]->geographicalId());
        }
        if (!good)
          return false;
      }
      for (int s = 0; s < 2; ++s) {
        // ForwardRingDiskBuilderFromDet::computeBounds: z range of all corners (start from the first det's centre)
        float zmin = e.sub[s].front()->surface().position().z(), zmax = zmin;
        for (auto const* g : e.sub[s])
          for (auto const& c : BoundingBox().corners(g->specificSurface())) {
            zmin = std::min(zmin, c.z());
            zmax = std::max(zmax, c.z());
          }
        e.plane[s] = Plane::build(Surface::PositionType(0., 0., (zmax + zmin) / 2.), Surface::RotationType());
        constexpr float kTwoPi = 2 * float(3.141592653589793238);
        e.phiStep[s] = kTwoPi / float(e.sub[s].size());
        e.invPhiStep[s] = 1.f / e.phiStep[s];
        e.phiOffset[s] = float(e.sub[s].front()->surface().position().phi()) - 0.5f * e.phiStep[s];
      }
      e.ok = true;
      return true;
    }

    RingLayout const& ringOf(const GeometricSearchDet* r) {
      auto [it, isNew] = rings_.try_emplace(r);
      RingLayout& e = it->second;
      if (!isNew)
        return e;
      if (!buildRing(r->basicComponents(), e)) {
        e.ok = false;
        return e;
      }
      // tkDetUtil::fillRingParametersFromDisk
      auto const& rd = static_cast<const BoundDisk&>(r->surface());
      e.disk = &rd;
      const float ringMinZ = std::abs(rd.position().z()) - rd.bounds().thickness() / 2.;
      const float ringMaxZ = std::abs(rd.position().z()) + rd.bounds().thickness() / 2.;
      e.thetaMin = rd.innerRadius() / ringMaxZ;
      e.thetaMax = rd.outerRadius() / ringMinZ;
      e.ringR = (rd.innerRadius() + rd.outerRadius()) / 2.;
      e.ok = true;
      return e;
    }

    // Phase2EndcapSingleRing: the ring disk rebuilt from its dets
    static bool singleRing(std::vector<const GeomDet*> const& dets, RingLayout& e) {
      if (dets.size() < 2)
        return false;
      e.single = true;
      e.sub[0] = dets;
      e.ownDisk = ReferenceCountingPointer<BoundDisk>(ForwardRingDiskBuilderFromDet()(dets));
      e.disk = e.ownDisk.get();
      e.plane[0] = Plane::build(Surface::PositionType(0., 0., e.disk->position().z()), Surface::RotationType());
      constexpr float kTwoPi = 2 * float(3.141592653589793238);
      e.phiStep[0] = kTwoPi / float(dets.size());
      e.invPhiStep[0] = 1.f / e.phiStep[0];
      e.phiOffset[0] = float(dets.front()->surface().position().phi()) - 0.5f * e.phiStep[0];
      const float ringMinZ = std::abs(e.disk->position().z()) - e.disk->bounds().thickness() / 2.;
      const float ringMaxZ = std::abs(e.disk->position().z()) + e.disk->bounds().thickness() / 2.;
      e.thetaMin = e.disk->innerRadius() / ringMaxZ;
      e.thetaMax = e.disk->outerRadius() / ringMinZ;
      e.ringR = (e.disk->innerRadius() + e.disk->outerRadius()) / 2.;
      e.ok = true;
      return true;
    }

    // Phase2EndcapLayerDoubleDisk (no components()): sub-disks by panel, single rings by blade
    PixEndcapLayout const& pixEndcapOf(const DetLayer* layer) {
      auto [it, isNew] = pixEndcaps_.try_emplace(layer);
      PixEndcapLayout& e = it->second;
      if (!isNew)
        return e;
      bool hasComps = true;
      try {
        (void)layer->components();
      } catch (cms::Exception const&) {
        hasComps = false;
      }
      if (hasComps)
        return e;  // a Phase2EndcapLayer: a ring layer
      e.doubleDisk = true;
      std::vector<std::vector<std::vector<const GeomDet*>>> subs;  // [subdisk][ring][det]
      int lastPanel = -1, lastBlade = -1;
      for (auto const* g : layer->basicComponents()) {
        const DetId id = g->geographicalId();
        const int panel = tTopo_.pxfPanel(id), blade = tTopo_.pxfBlade(id);
        if (panel != lastPanel) {
          subs.emplace_back();
          lastBlade = -1;
        }
        if (blade != lastBlade)
          subs.back().emplace_back();
        subs.back().back().push_back(g);
        lastPanel = panel;
        lastBlade = blade;
      }
      if (subs.size() < 2)
        return e;
      for (auto const& sd : subs) {
        SubDiskLayout d;
        float zmin = std::numeric_limits<float>::max(), zmax = -zmin;
        for (auto const& rd : sd) {
          RingLayout r;
          if (!singleRing(rd, r))
            return e;
          // tkDetUtil::computeDisk: z range of the ring disks
          zmin = std::min(zmin, r.disk->position().z() - r.disk->bounds().thickness() / 2);
          zmax = std::max(zmax, r.disk->position().z() + r.disk->bounds().thickness() / 2);
          d.rings.push_back(std::move(r));
        }
        d.z = (zmax + zmin) / 2;
        e.subs.push_back(std::move(d));
      }
      e.ok = true;
      return e;
    }

    RodLayout const& rodOf(const GeometricSearchDet* r, bool stacked) {
      auto [it, isNew] = rods_.try_emplace(r);
      RodLayout& e = it->second;
      if (!isNew)
        return e;
      auto const& bc = r->basicComponents();
      e.stacked = stacked;
      if (!stacked) {  // PixelRod: theDets, sorted in z
        if (bc.size() < 2)
          return e;
        e.dets.assign(bc.begin(), bc.end());
        e.ok = true;
        return e;
      }
      const size_t half = bc.size() / 2;
      if (bc.size() < 4 || bc.size() % 2 != 0)
        return e;
      double mr = 0, mrb = 0;
      for (size_t i = 0; i < half; ++i) {
        mr += bc[i]->position().perp();
        mrb += bc[half + i]->position().perp();
      }
      mr /= half;
      mrb /= half;
      for (size_t i = 0; i < half; ++i) {
        e.sub[bc[i]->position().perp() < mr ? 0 : 1].push_back(bc[i]);
        e.bro[bc[half + i]->position().perp() < mrb ? 0 : 1].push_back(bc[half + i]);
      }
      for (int s = 0; s < 2; ++s) {
        if (e.sub[s].size() < 3 || e.sub[s].size() != e.bro[s].size())
          return e;
        for (size_t i = 0; i < e.sub[s].size(); ++i)
          if (tTopo_.stack(e.sub[s][i]->geographicalId()) != tTopo_.stack(e.bro[s][i]->geographicalId()))
            return e;
        e.plane[s] = Plane::PlanePointer(RodPlaneBuilderFromDet()(e.sub[s]));
      }
      e.ok = true;
      return e;
    }

    BarrelLayout const& barrelOf(const DetLayer* layer) {
      auto [it, isNew] = barrels_.try_emplace(layer);
      BarrelLayout& e = it->second;
      if (!isNew)
        return e;
      auto const& comps = layer->components();  // TBLayer::theComps = inner rods, outer rods
      if (comps.size() < 4)
        return e;
      double mr = 0;
      for (auto const* c : comps)
        mr += c->position().perp();
      mr /= comps.size();
      for (auto const* c : comps)
        e.rods[c->position().perp() < mr ? 0 : 1].push_back(c);
      if (e.rods[0].empty() || e.rods[1].empty() || comps[0] != e.rods[0].front() || comps.back() != e.rods[1].back())
        return e;
      for (int s = 0; s < 2; ++s) {
        std::vector<const GeomDet*> tmp;
        for (auto const* rod : e.rods[s])
          tmp.insert(tmp.end(), rod->basicComponents().begin(), rod->basicComponents().end());
        e.cyl[s] = ReferenceCountingPointer<BoundCylinder>(CylinderBuilderFromDet()(tmp.begin(), tmp.end()));
      }
      // tilted rings: the tilted (TOB side 1/2) basic components in order, a ring = a run of lowers then uppers
      std::vector<const GeomDet*> cur;
      bool prevUpper = false;
      auto flush = [&]() {
        if (cur.empty())
          return true;
        RingLayout r;
        if (!buildRing(cur, r))
          return false;
        e.rings[cur.front()->position().z() < 0 ? 0 : 1].push_back(std::move(r));
        cur.clear();
        return true;
      };
      for (auto const* g : layer->basicComponents()) {
        const DetId id = g->geographicalId();
        if (id.subdetId() != StripSubdetector::TOB || tTopo_.tobSide(id) >= 3)
          continue;
        const bool upper = tTopo_.isUpper(id);
        if (!upper && prevUpper && !flush())
          return e;
        cur.push_back(g);
        prevUpper = upper;
      }
      if (!flush())
        return e;
      e.ok = true;
      return e;
    }

    TrackerTopology const& tTopo_;
    NavFlatTables& out_;
    std::unordered_map<const GeomDet*, int> detIndex_;
    std::unordered_map<const GeometricSearchDet*, RingLayout> rings_;
    std::unordered_map<const GeometricSearchDet*, RodLayout> rods_;
    std::unordered_map<const DetLayer*, BarrelLayout> barrels_;
    std::unordered_map<const DetLayer*, PixEndcapLayout> pixEndcaps_;
  };

}  // namespace mkfitdev::navdev

#endif
