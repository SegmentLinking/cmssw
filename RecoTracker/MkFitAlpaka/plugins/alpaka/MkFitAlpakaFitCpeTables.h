#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaFitCpeTables_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaFitCpeTables_h

// Host builder of the device PixelCPEGeneric tables of the mkFit final fit (interface/fit/CpeGeneric.h):
// per CPE module (PixelCPEFastParamsPhase2 detParams order) the position constants, the GenError template index
// (SiPixelGenErrorDBObject::getGenErrorID, PixelCPEBase.cc:169) and the local B field (PixelCPEBase.cc:182-184),
// plus the flattened GenError stores (SiPixelGenError::pushfile from the DB object, PixelCPEGeneric.cc:73).
// Built once per IOV by the ES producer MkFitAlpakaFitCpeESProducer.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeESData.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeGeneric.h"

namespace mkfitdev::cpe {

  inline void flattenGenError(SiPixelGenErrorStore const& store, CpeTablesHost& tables) {
    auto const& head = store.head;
    if (head.Dtype < 0 || head.Dtype > 5 || head.NTy < 2 || head.NTyx < 1 || head.NTxx < 2)
      throw cms::Exception("MkFitAlpakaFitCpe")
          << "GenError ID " << head.ID << " not supported (Dtype " << head.Dtype << ", NTy/NTyx/NTxx " << head.NTy
          << "/" << head.NTyx << "/" << head.NTxx << ")";
    GenErrTemplate flat{};
    flat.id = head.ID;
    flat.NTy = head.NTy;
    flat.NTyx = head.NTyx;
    flat.NTxx = head.NTxx;
    flat.Dtype = head.Dtype;
    flat.qscale = head.qscale;
    flat.fbin0 = head.fbin[0];
    flat.fbin1 = head.fbin[1];
    flat.fbin2 = head.fbin[2];
    flat.cotalpha0 = store.enty[0].cotalpha;
    auto& pool = tables.pool;
    flat.offCotbY = pool.size();
    pool.insert(pool.end(), store.cotbetaY, store.cotbetaY + head.NTy);
    flat.offCotbX = pool.size();
    pool.insert(pool.end(), store.cotbetaX, store.cotbetaX + head.NTyx);
    flat.offCotaX = pool.size();
    pool.insert(pool.end(), store.cotalphaX, store.cotalphaX + head.NTxx);
    flat.offEnty = pool.size();
    for (int i = 0; i < head.NTy; ++i) {
      auto const& entry = store.enty[i];
      const float values[kEntY] = {
          entry.qavg, entry.syone, entry.sytwo, entry.yrmsgen[0], entry.yrmsgen[1], entry.yrmsgen[2], entry.yrmsgen[3]};
      pool.insert(pool.end(), values, values + kEntY);
    }
    flat.offEntx = pool.size();
    for (int iy = 0; iy < head.NTyx; ++iy)
      for (int ix = 0; ix < head.NTxx; ++ix)
        pool.insert(pool.end(), store.entx[iy][ix].xrmsgen, store.entx[iy][ix].xrmsgen + 4);
    flat.offSx = pool.size();
    for (int ix = 0; ix < head.NTxx; ++ix) {
      pool.push_back(store.entx[0][ix].sxone);
      pool.push_back(store.entx[0][ix].sxtwo);
    }
    tables.templ.push_back(flat);
  }

  inline void buildCpeTables(PixelCPEFastParamsHost<pixelTopology::Phase2> const& params,
                             SiPixelGenErrorDBObject const& genErrDB,
                             TrackerGeometry const& geom,
                             MagneticField const& mf,
                             CpeTablesHost& tables) {
    tables = CpeTablesHost();
    std::vector<SiPixelGenErrorStore> store;
    if (!SiPixelGenError::pushfile(genErrDB, store))
      throw cms::Exception("MkFitAlpakaFitCpe") << "SiPixelGenError::pushfile failed";
    std::unordered_map<int, int> idToTempl;
    for (auto const& genError : store) {
      idToTempl[genError.head.ID] = tables.templ.size();
      flattenGenError(genError, tables);
    }
    auto const& fastParams = *params.buffer().data();
    const int nMod = pixelTopology::Phase2::numberOfModules;
    tables.modules.resize(nMod);
    for (int i = 0; i < nMod; ++i) {
      auto const& dp = fastParams.detParams(i);
      ModuleCpe& moduleCpe = tables.modules[i];
      moduleCpe.pitchX = dp.thePitchX;
      moduleCpe.pitchY = dp.thePitchY;
      moduleCpe.thickness =
          dp.isBarrel ? fastParams.commonParams().theThicknessB : fastParams.commonParams().theThicknessE;
      moduleCpe.shiftX = dp.shiftX;
      moduleCpe.shiftY = dp.shiftY;
      moduleCpe.chargeWidthX = dp.chargeWidthX;
      moduleCpe.chargeWidthY = dp.chargeWidthY;
      moduleCpe.nRows = dp.nRows;
      moduleCpe.nCols = dp.nCols;
      moduleCpe.templ = -1;
      moduleCpe.bx = moduleCpe.bz = 0.f;
      moduleCpe.bigPerRocX = moduleCpe.bigPerRocY = 0;
      tables.rawToModule[dp.rawId] = i;
      const auto* det = geom.idToDetUnit(DetId(dp.rawId));
      if (!det)
        continue;
      auto it = idToTempl.find(genErrDB.getGenErrorID(dp.rawId));
      if (it != idToTempl.end())
        moduleCpe.templ = it->second;
      const LocalVector localField = det->surface().toLocal(mf.inTesla(det->surface().position()));
      moduleCpe.bz = localField.z();
      moduleCpe.bx = localField.x();
      // big pixels: RectangularPixelPhase2Topology puts 2 * BIG_PIX_PER_ROC big pixels around the centre
      auto const* topo = dynamic_cast<PixelTopology const*>(&det->topology());
      if (topo) {
        int nbx = 0, nby = 0;
        for (int row = 0; row < topo->nrows(); ++row)
          nbx += topo->isItBigPixelInX(row) ? 1 : 0;
        for (int col = 0; col < topo->ncolumns(); ++col)
          nby += topo->isItBigPixelInY(col) ? 1 : 0;
        moduleCpe.bigPerRocX = nbx / 2;
        moduleCpe.bigPerRocY = nby / 2;
        // the position (localPositionTrackAngles) uses the linear pitch formula: no big pixels
        if (nbx != 0 || nby != 0)
          throw cms::Exception("MkFitAlpakaFitCpe") << "module " << dp.rawId << " has big pixels (" << nbx << " rows, "
                                                    << nby << " columns): the device CPE position supports none";
      }
    }
  }

}  // namespace mkfitdev::cpe

#endif
