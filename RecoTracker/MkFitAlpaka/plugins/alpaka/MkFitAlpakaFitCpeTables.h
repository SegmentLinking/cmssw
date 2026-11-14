#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaFitCpeTables_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaFitCpeTables_h

// Host builder of the device PixelCPEGeneric tables of the mkFit final fit (interface/fit/CpeGeneric.h):
// per CPE module (PixelCPEFastParamsPhase2 detParams order) the position constants, the GenError template index
// (SiPixelGenErrorDBObject::getGenErrorID, PixelCPEBase.cc:169) and the local B field (PixelCPEBase.cc:182-184),
// plus the flattened GenError stores (SiPixelGenError::pushfile from the DB object, PixelCPEGeneric.cc:73).
// Also the per-hit cluster quantities from a SiPixelCluster.
// built once per IOV by the ES producer MkFitAlpakaFitCpeESProducer; crossCheckCpe below compares
// the result with the PixelCPEGeneric object of the menu.

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeESData.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeGeneric.h"
#include "RecoTracker/MkFitAlpaka/plugins/alpaka/MkFitAlpakaClusterCpeFill.h"

namespace mkfitdev::cpe {

  inline void flattenGenError(SiPixelGenErrorStore const& s, CpeTablesHost& t) {
    auto const& h = s.head;
    if (h.Dtype < 0 || h.Dtype > 5 || h.NTy < 2 || h.NTyx < 1 || h.NTxx < 2)
      throw cms::Exception("MkFitAlpakaFitCpe") << "GenError ID " << h.ID << " not supported (Dtype " << h.Dtype
                                                << ", NTy/NTyx/NTxx " << h.NTy << "/" << h.NTyx << "/" << h.NTxx << ")";
    GenErrTemplate g{};
    g.id = h.ID;
    g.NTy = h.NTy;
    g.NTyx = h.NTyx;
    g.NTxx = h.NTxx;
    g.Dtype = h.Dtype;
    g.qscale = h.qscale;
    g.fbin0 = h.fbin[0];
    g.fbin1 = h.fbin[1];
    g.fbin2 = h.fbin[2];
    g.cotalpha0 = s.enty[0].cotalpha;
    auto& p = t.pool;
    g.offCotbY = p.size();
    p.insert(p.end(), s.cotbetaY, s.cotbetaY + h.NTy);
    g.offCotbX = p.size();
    p.insert(p.end(), s.cotbetaX, s.cotbetaX + h.NTyx);
    g.offCotaX = p.size();
    p.insert(p.end(), s.cotalphaX, s.cotalphaX + h.NTxx);
    g.offEnty = p.size();
    for (int i = 0; i < h.NTy; ++i) {
      auto const& e = s.enty[i];
      const float v[kEntY] = {e.qavg, e.syone, e.sytwo, e.yrmsgen[0], e.yrmsgen[1], e.yrmsgen[2], e.yrmsgen[3]};
      p.insert(p.end(), v, v + kEntY);
    }
    g.offEntx = p.size();
    for (int iy = 0; iy < h.NTyx; ++iy)
      for (int ix = 0; ix < h.NTxx; ++ix)
        p.insert(p.end(), s.entx[iy][ix].xrmsgen, s.entx[iy][ix].xrmsgen + 4);
    g.offSx = p.size();
    for (int ix = 0; ix < h.NTxx; ++ix) {
      p.push_back(s.entx[0][ix].sxone);
      p.push_back(s.entx[0][ix].sxtwo);
    }
    t.templ.push_back(g);
  }

  inline void buildCpeTables(PixelCPEFastParamsHost<pixelTopology::Phase2> const& params,
                             SiPixelGenErrorDBObject const& genErrDB,
                             TrackerGeometry const& geom,
                             MagneticField const& mf,
                             CpeTablesHost& t) {
    t = CpeTablesHost();
    std::vector<SiPixelGenErrorStore> store;
    if (!SiPixelGenError::pushfile(genErrDB, store))
      throw cms::Exception("MkFitAlpakaFitCpe") << "SiPixelGenError::pushfile failed";
    std::unordered_map<int, int> idToTempl;
    for (auto const& s : store) {
      idToTempl[s.head.ID] = t.templ.size();
      flattenGenError(s, t);
    }
    auto const& p = *params.buffer().data();
    const int nMod = pixelTopology::Phase2::numberOfModules;
    t.modules.resize(nMod);
    t.rawId.resize(nMod);
    for (int i = 0; i < nMod; ++i) {
      auto const& dp = p.detParams(i);
      ModuleCpe& m = t.modules[i];
      m.pitchX = dp.thePitchX;
      m.pitchY = dp.thePitchY;
      m.thickness = dp.isBarrel ? p.commonParams().theThicknessB : p.commonParams().theThicknessE;
      m.shiftX = dp.shiftX;
      m.shiftY = dp.shiftY;
      m.chargeWidthX = dp.chargeWidthX;
      m.chargeWidthY = dp.chargeWidthY;
      m.nRows = dp.nRows;
      m.nCols = dp.nCols;
      m.templ = -1;
      m.bx = m.bz = 0.f;
      m.bigPerRocX = m.bigPerRocY = 0;
      t.rawToModule[dp.rawId] = i;
      t.rawId[i] = dp.rawId;
      const auto* det = geom.idToDetUnit(DetId(dp.rawId));
      if (!det)
        continue;
      auto it = idToTempl.find(genErrDB.getGenErrorID(dp.rawId));
      if (it != idToTempl.end())
        m.templ = it->second;
      const LocalVector b = det->surface().toLocal(mf.inTesla(det->surface().position()));
      m.bz = b.z();
      m.bx = b.x();
      // big pixels: RectangularPixelPhase2Topology puts 2 * BIG_PIX_PER_ROC big pixels around the centre
      auto const* topo = dynamic_cast<PixelTopology const*>(&det->topology());
      if (topo) {
        int nbx = 0, nby = 0;
        for (int r = 0; r < topo->nrows(); ++r)
          nbx += topo->isItBigPixelInX(r) ? 1 : 0;
        for (int c = 0; c < topo->ncolumns(); ++c)
          nby += topo->isItBigPixelInY(c) ? 1 : 0;
        m.bigPerRocX = nbx / 2;
        m.bigPerRocY = nby / 2;
        // the position (localPositionTrackAngles) uses the linear pitch formula: no big pixels
        if (nbx != 0 || nby != 0)
          throw cms::Exception("MkFitAlpakaFitCpe") << "module " << dp.rawId << " has big pixels (" << nbx << " rows, "
                                                    << nby << " columns): the device CPE position supports none";
      }
    }
  }

  // ---- cross-check of the device CPE against the CPE object of the menu ('PixelCPEGeneric') ----
  // Tolerances: position 1e-6 cm (0.01 um), errors 1e-5 relative, exy exactly 0 (host and device). The earlier per-hit
  // check of the same formulas found x <= 0.0024 um, y <= 0.0048 um, errors <= 6e-7 relative (float rounding).
  struct CpeCheckStats {
    long n = 0, nBad = 0, nPortFail = 0;
    double maxDx = 0, maxDy = 0, maxRelXX = 0, maxRelYY = 0, maxXY = 0;
    std::string first;
    void merge(CpeCheckStats const& o) {
      n += o.n;
      nBad += o.nBad;
      nPortFail += o.nPortFail;
      maxDx = std::max(maxDx, o.maxDx);
      maxDy = std::max(maxDy, o.maxDy);
      maxRelXX = std::max(maxRelXX, o.maxRelXX);
      maxRelYY = std::max(maxRelYY, o.maxRelYY);
      maxXY = std::max(maxXY, o.maxXY);
      if (first.empty())
        first = o.first;
    }
    std::string summary() const {
      std::ostringstream o;
      o << n << " calls, " << nBad << " beyond tolerance (" << nPortFail << " without device CPE); max |dx| "
        << 1e4 * maxDx << " um, |dy| " << 1e4 * maxDy << " um, rel exx " << maxRelXX << ", rel eyy " << maxRelYY
        << ", |exy| " << maxXY;
      if (!first.empty())
        o << "; first: " << first;
      return o.str();
    }
  };

  // getParameters(cluster, det, ltp) (as MkFitFitProducer.cc:144-158 calls it) vs cpeTrackAngles
  inline bool compareCpe(PixelClusterParameterEstimator const& hostCpe,
                         GeomDetUnit const& det,
                         SiPixelCluster const& cl,
                         CpeTables const& tv,
                         int module,
                         float cotA,
                         float cotB,
                         float x,
                         float y,
                         CpeCheckStats& st) {
    const LocalTrajectoryParameters ltp(0.1f, cotA, cotB, x, y, 1.f);
    auto const sp = hostCpe.getParameters(cl, det, ltp);
    const float s[5] = {std::get<0>(sp).x(),
                        std::get<0>(sp).y(),
                        float(std::get<1>(sp).xx()),
                        float(std::get<1>(sp).xy()),
                        float(std::get<1>(sp).yy())};
    float o[5];
    ++st.n;
    bool bad = false;
    if (!cpeTrackAngles(tv, clusterCpe(cl, module), cotA, cotB, o)) {
      ++st.nPortFail;
      bad = true;
    } else {
      auto rel = [](double a, double b) { return b == 0 ? std::abs(a) : std::abs(a - b) / std::abs(b); };
      const double dx = std::abs(double(o[0]) - s[0]), dy = std::abs(double(o[1]) - s[1]);
      const double rxx = rel(o[2], s[2]), ryy = rel(o[4], s[4]), dxy = std::max(std::abs(o[3]), std::abs(s[3]));
      st.maxDx = std::max(st.maxDx, dx);
      st.maxDy = std::max(st.maxDy, dy);
      st.maxRelXX = std::max(st.maxRelXX, rxx);
      st.maxRelYY = std::max(st.maxRelYY, ryy);
      st.maxXY = std::max(st.maxXY, dxy);
      bad = dx > 1e-6 || dy > 1e-6 || rxx > 1e-5 || ryy > 1e-5 || dxy != 0;
    }
    if (bad) {
      ++st.nBad;
      if (st.first.empty()) {
        std::ostringstream m;
        m << "det " << det.geographicalId().rawId() << " cluster rows " << cl.minPixelRow() << "-" << cl.maxPixelRow()
          << " cols " << cl.minPixelCol() << "-" << cl.maxPixelCol() << " charge " << cl.charge() << " cotA " << cotA
          << " cotB " << cotB << ": host (" << s[0] << ", " << s[1] << ", " << s[2] << ", " << s[3] << ", " << s[4]
          << ") device (" << o[0] << ", " << o[1] << ", " << o[2] << ", " << o[3] << ", " << o[4] << ")";
        st.first = m.str();
      }
    }
    return !bad;
  }

  // Once per IOV: synthetic clusters on every moduleStride-th module and on the first module of every GenError
  // template, at the centre (sizes 1x1 .. 3x4, one pixel above 30k electrons to see a charge truncation) and on the
  // four edges (edge errors), times 9 track-angle pairs. Covers the module constants of the other ES product
  // (PixelCPEFastParamsPhase2), the hard-coded menu settings (CpeGeneric.h), topology / deformations, Lorentz angle.
  inline CpeCheckStats crossCheckCpeSynthetic(PixelClusterParameterEstimator const& hostCpe,
                                              TrackerGeometry const& geom,
                                              CpeTablesHost const& t,
                                              int moduleStride) {
    CpeCheckStats st;
    const CpeTables tv = t.view();
    std::vector<char> templSeen(t.templ.size(), 0);
    struct Px {
      int r, c, adc;
    };
    const float cotAs[3] = {-0.2f, 0.05f, 0.3f}, cotBs[3] = {-3.0f, 0.0f, 1.2f};
    for (int i = 0; i < int(t.modules.size()); ++i) {
      auto const& m = t.modules[i];
      const bool firstOfTempl = m.templ >= 0 && !templSeen[m.templ];
      if (i % moduleStride != 0 && !firstOfTempl)
        continue;
      if (m.templ >= 0)
        templSeen[m.templ] = 1;
      auto const* det = geom.idToDetUnit(DetId(t.rawId[i]));
      if (!det)
        continue;
      const int nr = int(m.nRows), nc = int(m.nCols), r0 = nr / 2 + 3, c0 = nc / 2 + 5;
      const std::vector<std::vector<Px>> clusters = {
          {{r0, c0, 21000}},
          {{r0, c0, 12000}, {r0 + 1, c0, 9000}},
          {{r0, c0, 8000}, {r0, c0 + 1, 15000}, {r0, c0 + 2, 7000}},
          {{r0, c0, 6000},
           {r0 + 1, c0, 9000},
           {r0 + 2, c0 + 1, 7000},
           {r0 + 1, c0 + 2, 11000},
           {r0, c0 + 3, 5000},
           {r0 + 2, c0 + 3, 8000}},
          {{r0, c0, 35000}, {r0 + 1, c0 + 1, 4000}},
          {{0, c0, 10000}, {1, c0, 9000}},                 // edge in x (first row)
          {{nr - 1, c0, 14000}},                           // edge in x (last row)
          {{r0, 0, 9000}, {r0, 1, 12000}},                 // edge in y (first column)
          {{r0, nc - 2, 7000}, {r0 + 1, nc - 1, 13000}}};  // edge in y (last column)
      for (auto const& px : clusters) {
        SiPixelCluster cl(SiPixelCluster::PixelPos(px[0].r, px[0].c), px[0].adc);
        double sr = px[0].r, sc = px[0].c;
        for (std::size_t k = 1; k < px.size(); ++k) {
          cl.add(SiPixelCluster::PixelPos(px[k].r, px[k].c), px[k].adc);
          sr += px[k].r;
          sc += px[k].c;
        }
        const float x = float((sr / px.size() + 0.5 - 0.5 * nr) * m.pitchX);
        const float y = float((sc / px.size() + 0.5 - 0.5 * nc) * m.pitchY);
        for (float ca : cotAs)
          for (float cb : cotBs)
            compareCpe(hostCpe, *det, cl, tv, i, ca, cb, x, y, st);
      }
    }
    return st;
  }

}  // namespace mkfitdev::cpe

#endif
