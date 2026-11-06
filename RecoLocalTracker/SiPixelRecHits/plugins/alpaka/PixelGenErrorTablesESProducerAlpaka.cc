#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelGenErrorTablesHost.h"
// automatic copy of the host product to the device
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelGenErrorTablesDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The GenError tables and per-module values of the generic CPE with track angles
  // (pixelCPEforDeviceTrackAngles.h), in the module order of the PixelCPEFastParams product.
  template <typename TrackerTraits>
  class PixelGenErrorTablesESProducerAlpaka : public ESProducer {
  public:
    PixelGenErrorTablesESProducerAlpaka(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig),
          effChargeCutLowX_(iConfig.getParameter<double>("eff_charge_cut_lowX")),
          effChargeCutLowY_(iConfig.getParameter<double>("eff_charge_cut_lowY")),
          effChargeCutHighX_(iConfig.getParameter<double>("eff_charge_cut_highX")),
          effChargeCutHighY_(iConfig.getParameter<double>("eff_charge_cut_highY")),
          sizeCutX_(iConfig.getParameter<double>("size_cutX")),
          sizeCutY_(iConfig.getParameter<double>("size_cutY")),
          edgeClusterErrorX_(iConfig.getParameter<double>("EdgeClusterErrorX")),
          edgeClusterErrorY_(iConfig.getParameter<double>("EdgeClusterErrorY")) {
      auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
      fastParamsToken_ = cc.consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("PixelCPEFastParams")));
      genErrorToken_ = cc.consumes();
      geometryToken_ = cc.consumes();
      magneticFieldToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("MagneticFieldRecord"));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName", std::string("PixelGenErrorTables") + TrackerTraits::nameModifier);
      desc.add<std::string>("PixelCPEFastParams", std::string("PixelCPEFastParams") + TrackerTraits::nameModifier);
      desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag());
      // as PixelCPEGeneric
      desc.add<double>("eff_charge_cut_lowX", 0.0);
      desc.add<double>("eff_charge_cut_lowY", 0.0);
      desc.add<double>("eff_charge_cut_highX", 1.0);
      desc.add<double>("eff_charge_cut_highY", 1.0);
      desc.add<double>("size_cutX", 3.0);
      desc.add<double>("size_cutY", 3.0);
      desc.add<double>("EdgeClusterErrorX", 50.0);
      desc.add<double>("EdgeClusterErrorY", 85.0);
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<PixelGenErrorTablesHost> produce(PixelCPEFastParamsRecord const& iRecord) {
      auto const& fastParams = *iRecord.get(fastParamsToken_).data();
      auto const& genErrorDB = iRecord.get(genErrorToken_);
      auto const& geometry = iRecord.get(geometryToken_);
      auto const& magneticField = iRecord.get(magneticFieldToken_);

      std::vector<SiPixelGenErrorStore> stores;
      if (!SiPixelGenError::pushfile(genErrorDB, stores))
        throw cms::Exception("PixelGenErrorTables") << "SiPixelGenError::pushfile failed";
      const int nStores = stores.size();
      int poolSize = 0;
      std::unordered_map<int, int> storeOfId;
      for (int i = 0; i < nStores; ++i) {
        auto const& head = stores[i].head;
        if (head.Dtype < 0 || head.Dtype > 5 || head.NTy < 2 || head.NTyx < 2 || head.NTxx < 2)
          throw cms::Exception("PixelGenErrorTables")
              << "GenError ID " << head.ID << " not supported (Dtype " << head.Dtype << ", NTy/NTyx/NTxx " << head.NTy
              << "/" << head.NTyx << "/" << head.NTxx << ")";
        storeOfId[head.ID] = i;
        poolSize += head.NTy * (1 + pixelGenErrorTables::kYEntrySize) +
                    head.NTxx * (2 + head.NTyx * pixelGenErrorTables::kXEntrySize) + head.NTyx;
      }

      constexpr int nModules = TrackerTraits::numberOfModules;
      auto product = std::make_unique<PixelGenErrorTablesHost>(cms::alpakatools::host(), nModules, nStores, poolSize);
      auto modules = product->view().modules();
      auto headers = product->view().stores();
      float* pool = product->view().pool().value().data();

      modules.effChargeCutLowX() = effChargeCutLowX_;
      modules.effChargeCutLowY() = effChargeCutLowY_;
      modules.effChargeCutHighX() = effChargeCutHighX_;
      modules.effChargeCutHighY() = effChargeCutHighY_;
      modules.sizeCutX() = sizeCutX_;
      modules.sizeCutY() = sizeCutY_;
      modules.edgeClusterErrorX() = edgeClusterErrorX_;
      modules.edgeClusterErrorY() = edgeClusterErrorY_;

      int offset = 0;
      auto append = [&pool, &offset](float value) { pool[offset++] = value; };
      for (int i = 0; i < nStores; ++i) {
        auto const& store = stores[i];
        auto const& head = store.head;
        auto header = headers[i];
        header.detectorType() = head.Dtype;
        header.nCotBetaY() = head.NTy;
        header.nCotBetaX() = head.NTyx;
        header.nCotAlphaX() = head.NTxx;
        header.qscale() = head.qscale;
        header.fbin0() = head.fbin[0];
        header.fbin1() = head.fbin[1];
        header.fbin2() = head.fbin[2];
        header.cotAlpha0() = store.enty[0].cotalpha;
        header.cotBetaYOffset() = offset;
        for (int iy = 0; iy < head.NTy; ++iy)
          append(store.cotbetaY[iy]);
        header.cotBetaXOffset() = offset;
        for (int iy = 0; iy < head.NTyx; ++iy)
          append(store.cotbetaX[iy]);
        header.cotAlphaXOffset() = offset;
        for (int ix = 0; ix < head.NTxx; ++ix)
          append(store.cotalphaX[ix]);
        header.yEntryOffset() = offset;
        for (int iy = 0; iy < head.NTy; ++iy) {
          auto const& entry = store.enty[iy];
          append(entry.qavg);
          append(entry.syone);
          for (float rms : entry.yrmsgen)
            append(rms);
        }
        header.xEntryOffset() = offset;
        for (int iy = 0; iy < head.NTyx; ++iy)
          for (int ix = 0; ix < head.NTxx; ++ix)
            for (float rms : store.entx[iy][ix].xrmsgen)
              append(rms);
        header.singleXOffset() = offset;
        for (int ix = 0; ix < head.NTxx; ++ix)
          append(store.entx[0][ix].sxone);
      }
      if (offset != poolSize)
        throw cms::Exception("LogicError") << "PixelGenErrorTables: pool size " << offset << ", expected " << poolSize;

      for (int i = 0; i < nModules; ++i) {
        auto const rawId = fastParams.detParams(i).rawId;
        auto module = modules[i];
        module.genErrorStore() = -1;
        module.localBx() = 0.f;
        module.localBz() = 0.f;
        auto const* det = geometry.idToDetUnit(DetId(rawId));
        if (det == nullptr)
          continue;
        auto const* topology = dynamic_cast<PixelTopology const*>(&det->topology());
        if (topology == nullptr)
          throw cms::Exception("PixelGenErrorTables") << "module " << rawId << " has no pixel topology";
        // the position assumes pixels of equal pitch
        for (int row = 0; row < topology->nrows(); ++row)
          if (topology->isItBigPixelInX(row))
            throw cms::Exception("PixelGenErrorTables") << "module " << rawId << " has big pixels, not supported";
        for (int col = 0; col < topology->ncolumns(); ++col)
          if (topology->isItBigPixelInY(col))
            throw cms::Exception("PixelGenErrorTables") << "module " << rawId << " has big pixels, not supported";
        auto store = storeOfId.find(genErrorDB.getGenErrorID(rawId));
        if (store != storeOfId.end())
          module.genErrorStore() = store->second;
        // PixelCPEBase::fillDetParams
        const LocalVector field = det->surface().toLocal(magneticField.inTesla(det->surface().position()));
        module.localBx() = field.x();
        module.localBz() = field.z();
      }
      return product;
    }

  private:
    edm::ESGetToken<PixelCPEFastParamsHost<TrackerTraits>, PixelCPEFastParamsRecord> fastParamsToken_;
    edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> genErrorToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
    const float effChargeCutLowX_;
    const float effChargeCutLowY_;
    const float effChargeCutHighX_;
    const float effChargeCutHighY_;
    const float sizeCutX_;
    const float sizeCutY_;
    const float edgeClusterErrorX_;
    const float edgeClusterErrorY_;
  };

  using PixelGenErrorTablesESProducerAlpakaPhase2 = PixelGenErrorTablesESProducerAlpaka<pixelTopology::Phase2>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PixelGenErrorTablesESProducerAlpakaPhase2);
