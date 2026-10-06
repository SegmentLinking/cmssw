// The flat layer tables of the portable compatibleDets search for the whole tracker, built once per
// TrackerRecoGeometryRecord IOV (MkFitAlpakaNavFlatTablesBuilder.h).
#include <memory>
#include <set>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "MkFitAlpakaNavFlatTablesBuilder.h"
#include "RecoTracker/MkFitAlpaka/interface/navdev/NavFlatTables.h"

class MkFitAlpakaNavFlatTablesESProducer : public edm::ESProducer {
public:
  explicit MkFitAlpakaNavFlatTablesESProducer(edm::ParameterSet const& iConfig) {
    auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("appendToDataLabel"));
    mkFitGeomToken_ = cc.consumes();
    trackerToken_ = cc.consumes();
    tTopoToken_ = cc.consumesFrom<TrackerTopology, TrackerTopologyRcd>();
  }
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("appendToDataLabel", "");
    descriptions.addWithDefaultLabel(desc);
  }
  std::unique_ptr<mkfitdev::navdev::NavFlatTables> produce(TrackerRecoGeometryRecord const& iRecord) {
    auto const& geom = iRecord.get(mkFitGeomToken_);
    auto const& tTopo = iRecord.get(tTopoToken_);
    auto const& tracker = iRecord.get(trackerToken_);
    auto out = std::make_unique<mkfitdev::navdev::NavFlatTables>();
    mkfitdev::navdev::NavFlatTablesBuilder builder(tTopo, *out);
    for (const DetLayer* layer : geom.detLayers()) {
      if (layer == nullptr || layer->basicComponents().empty())
        continue;
      ++out->nLayers;
      out->nUnsupported += !builder.addLayer(layer);
    }
    out->allLayers = tracker.allLayers();
    for (size_t i = 0; i < out->allLayers.size(); ++i) {
      const DetLayer* l = out->allLayers[i];
      out->layerIndex[l] = i;
      auto const it = out->layers.find(l);
      out->layerFlat.push_back(it == out->layers.end() ? std::array<int, 3>{{0, -1, 0}} : it->second);
    }
    // the iteration order of SimpleNavigableLayer::compatibleLayers' std::set<const DetLayer*>
    const std::set<const DetLayer*> byAddress(out->allLayers.begin(), out->allLayers.end());
    if (byAddress.size() != out->allLayers.size())
      throw cms::Exception("LogicError") << "MkFitAlpakaNavFlatTablesESProducer: a DetLayer is listed twice";
    for (const DetLayer* l : byAddress)
      out->pointerOrder.push_back(out->layerIndex.at(l));
    for (const DetLayer* l : geom.detLayers()) {
      auto const it = l == nullptr ? out->layerIndex.end() : out->layerIndex.find(l);
      out->mkFitToLayer.push_back(it == out->layerIndex.end() ? -1 : it->second);
    }
    edm::LogInfo("MkFitAlpakaNavFlatTables")
        << "flat navigation tables: " << out->nLayers << " layers (" << out->nUnsupported << " on the host), "
        << out->dets.size() << " dets (" << out->nonRect << " non-rectangular), " << out->rings.size() << " rings, "
        << out->rods.size() << " rods, " << out->barrels.size() << " barrels, " << out->subDisks.size() << " sub-disks";
    return out;
  }

private:
  edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> trackerToken_;
};

DEFINE_FWK_EVENTSETUP_MODULE(MkFitAlpakaNavFlatTablesESProducer);
