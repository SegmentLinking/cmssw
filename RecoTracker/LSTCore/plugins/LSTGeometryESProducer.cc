#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"

// LST includes



class LSTGeometryESProducer : public edm::ESProducer {
public:
  LSTGeometryESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<int> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> trackerToken_;

  const TrackerTopology *trackerTopo_ = nullptr;
  const TrackerGeometry *trackerGeom_ = nullptr;
};

LSTGeometryESProducer::LSTGeometryESProducer(const edm::ParameterSet &iConfig) {
  auto cc = setWhatProduced(this);
  geomToken_ = cc.consumes();
  ttopoToken_ = cc.consumes();
  trackerToken_ = cc.consumes();
}

void LSTGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}



DEFINE_FWK_EVENTSETUP_MODULE(LSTGeometryESProducer);
