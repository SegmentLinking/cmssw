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

#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"

// LST includes
#include "RecoTracker/LSTCore/interface/LSTGeometry/SensorInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/ModuleInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Module.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/LSTGeometryMethods.h"

#include <cmath>
#include <vector>
#include <unordered_map>

class LSTGeometryESProducer : public edm::ESProducer {
public:
  LSTGeometryESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<lstgeometry::LSTGeometry> produce(const TrackerRecoGeometryRecord &iRecord);

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

std::unique_ptr<lstgeometry::LSTGeometry> LSTGeometryESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
  trackerGeom_ = &iRecord.get(geomToken_);
  trackerTopo_ = &iRecord.get(ttopoToken_);

  std::vector<lstgeometry::ModuleInfo> modules;
  std::unordered_map<unsigned int, lstgeometry::SensorInfo> sensors;

  std::vector<double> avg_r_cm(6, 0.0);
  std::vector<double> avg_z_cm(5, 0.0);
  std::vector<unsigned int> avg_r_counter(6, 0);
  std::vector<unsigned int> avg_z_counter(5, 0);

  for (auto &det : trackerGeom_->dets()) {
    const DetId detId = det->geographicalId();

    const auto &surface = det->surface();
    const Bounds *b = &(surface).bounds();
    const auto &position = surface.position();

    double rho_cm = position.perp();
    double z_cm = position.z();
    const double phi_rad = position.phi();

    if (det->isLeaf()) {
      // Leafs are the sensors
      lstgeometry::SensorInfo sensor;
      sensor.detId = detId();
      sensor.sensorCenterRho_cm = rho_cm;
      sensor.sensorCenterZ_cm = z_cm;
      sensor.phi_rad = phi_rad;
      sensors[detId()] = std::move(sensor);
      continue;
    }

    const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b);
    if (!b2) {
      throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
    }

    double tiltAngle_rad = std::asin(det->rotation().zz());

    double meanWidth_cm = b2->width();
    double length_cm = b2->length();
    double dx = b2->width() * 0.5;
    double dy = b2->length() * 0.5;

    double sensorSpacing_cm = det->components()[0]->toLocal(det->components()[1]->position()).mag();

    double vtxOneX_cm_tmp = rho_cm - dy * std::sin(tiltAngle_rad);
    double vtxOneY_cm_tmp = dx;
    double vtxTwoX_cm_tmp = rho_cm + dy * std::sin(tiltAngle_rad);
    double vtxTwoY_cm_tmp = dx;
    double vtxThreeX_cm_tmp = rho_cm - dy * std::sin(tiltAngle_rad);
    double vtxThreeY_cm_tmp = -dx;
    double vtxFourX_cm_tmp = rho_cm + dy * std::sin(tiltAngle_rad);
    double vtxFourY_cm_tmp = -dx;

    double vtxOneX_cm = vtxOneX_cm_tmp * cos(phi_rad) + vtxOneY_cm_tmp * sin(phi_rad);
    double vtxOneY_cm = vtxOneX_cm_tmp * sin(phi_rad) - vtxOneY_cm_tmp * cos(phi_rad);
    double vtxTwoX_cm = vtxTwoX_cm_tmp * cos(phi_rad) + vtxTwoY_cm_tmp * sin(phi_rad);
    double vtxTwoY_cm = vtxTwoX_cm_tmp * sin(phi_rad) - vtxTwoY_cm_tmp * cos(phi_rad);
    double vtxThreeX_cm = vtxThreeX_cm_tmp * cos(phi_rad) + vtxThreeY_cm_tmp * sin(phi_rad);
    double vtxThreeY_cm = vtxThreeX_cm_tmp * sin(phi_rad) - vtxThreeY_cm_tmp * cos(phi_rad);
    double vtxFourX_cm = vtxFourX_cm_tmp * cos(phi_rad) + vtxFourY_cm_tmp * sin(phi_rad);
    double vtxFourY_cm = vtxFourX_cm_tmp * sin(phi_rad) - vtxFourY_cm_tmp * cos(phi_rad);

    unsigned int detid = detId();

    unsigned short layer = lstgeometry::Module::parseLayer(detid);
    if (lstgeometry::Module::parseSubdet(detid) == lstgeometry::Module::SubDet::Barrel) {
      avg_r_cm[layer - 1] += rho_cm;
      avg_r_counter[layer - 1] += 1;
    } else {
      avg_z_cm[layer - 1] += std::fabs(z_cm);
      avg_z_counter[layer - 1] += 1;
    }

    lstgeometry::ModuleInfo module;
    module.detId = detid;
    module.sensorCenterRho_cm = rho_cm;
    module.sensorCenterZ_cm = z_cm;
    module.tiltAngle_rad = tiltAngle_rad;
    module.skewAngle_rad = 0.0;
    module.yawAngle_rad = 0.0;
    module.phi_rad = phi_rad;
    module.vtxOneX_cm = vtxOneX_cm;
    module.vtxOneY_cm = vtxOneY_cm;
    module.vtxTwoX_cm = vtxTwoX_cm;
    module.vtxTwoY_cm = vtxTwoY_cm;
    module.vtxThreeX_cm = vtxThreeX_cm;
    module.vtxThreeY_cm = vtxThreeY_cm;
    module.vtxFourX_cm = vtxFourX_cm;
    module.vtxFourY_cm = vtxFourY_cm;
    module.meanWidth_cm = meanWidth_cm;
    module.length_cm = length_cm;
    module.sensorSpacing_cm = sensorSpacing_cm;
    modules.push_back(module);
  }

  for (size_t i = 0; i < avg_r_cm.size(); ++i) {
    avg_r_cm[i] /= avg_r_counter[i];
  }
  for (size_t i = 0; i < avg_z_cm.size(); ++i) {
    avg_z_cm[i] /= avg_z_counter[i];
  }

  auto lstGeometry = makeLSTGeometry(modules, sensors, avg_r_cm, avg_z_cm);

  return lstGeometry;
}

DEFINE_FWK_EVENTSETUP_MODULE(LSTGeometryESProducer);