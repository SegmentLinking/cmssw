#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"

#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/ModuleInfo.h"
#include "RecoTracker/LSTGeometry/interface/Geometry.h"

#include <cmath>
#include <vector>
#include <unordered_map>

class LSTGeometryESProducer : public edm::ESProducer {
public:
  LSTGeometryESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<lstgeometry::Geometry> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  double ptCut_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;

  const TrackerGeometry *trackerGeom_ = nullptr;
  const TrackerTopology *trackerTopo_ = nullptr;
};

LSTGeometryESProducer::LSTGeometryESProducer(const edm::ParameterSet &iConfig)
    : ptCut_(iConfig.getParameter<double>("ptCut")) {
  auto cc = setWhatProduced(this, "LSTGeometry");
  geomToken_ = cc.consumes();
  ttopoToken_ = cc.consumes();
}

void LSTGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("ptCut", 0.8);
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<lstgeometry::Geometry> LSTGeometryESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
  trackerGeom_ = &iRecord.get(geomToken_);
  trackerTopo_ = &iRecord.get(ttopoToken_);

  std::unordered_map<unsigned int, lstgeometry::ModuleInfo> modules;
  std::unordered_map<unsigned int, lstgeometry::Sensor> sensors;

  std::vector<double> avg_r_cm(6, 0.0);
  std::vector<double> avg_z_cm(5, 0.0);
  std::vector<unsigned int> avg_r_counter(6, 0);
  std::vector<unsigned int> avg_z_counter(5, 0);

  for (auto &det : trackerGeom_->dets()) {
    const DetId detId = det->geographicalId();
    const auto moduleType = trackerGeom_->getDetectorType(detId);

    // TODO: Is there a more straightforward way to only loop through these?
    if (moduleType != TrackerGeometry::ModuleType::Ph2PSP && moduleType != TrackerGeometry::ModuleType::Ph2PSS &&
        moduleType != TrackerGeometry::ModuleType::Ph2SS) {
      continue;
    }

    const unsigned int detid = detId();

    const auto &surface = det->surface();
    const Bounds *b = &(surface).bounds();
    const auto &position = surface.position();

    const double rho_cm = position.perp();
    const double z_cm = lstgeometry::roundCoordinate(position.z());
    const double phi_rad = lstgeometry::roundAngle(position.phi());

    if (det->isLeaf()) {
      // Leafs are the sensors
      const unsigned int moduleDetId = detid & ~0b11;  // TODO: Is there a CMSSW method for this?
      sensors[detid] = lstgeometry::Sensor(moduleDetId, rho_cm, z_cm, phi_rad, moduleType);
      continue;
    }

    const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b);
    if (!b2) {
      throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
    }

    const auto subdet = static_cast<GeomDetEnumerators::SubDetector>(detId.subdetId());
    const auto side = trackerTopo_->barrelTiltTypeP2(detId);
    const auto location = (GeomDetEnumerators::isBarrel(subdet) ? GeomDetEnumerators::Location::barrel
                                                                : GeomDetEnumerators::Location::endcap);
    const unsigned int layer = trackerTopo_->layer(detId);
    const unsigned int ring = trackerTopo_->endcapRingP2(detId);
    const bool isLower = trackerTopo_->isLower(detId);

    double tiltAngle_rad = lstgeometry::roundAngle(std::asin(det->rotation().zz()));

    double meanWidth_cm = b2->width();
    double length_cm = b2->length();

    double sensorSpacing_cm = det->components()[0]->toLocal(det->components()[1]->position()).mag();

    // Fix angles of some modules
    if (std::fabs(std::fabs(tiltAngle_rad) - std::numbers::pi_v<double> / 2) < 1e-3) {
      tiltAngle_rad = std::numbers::pi_v<double> / 2;
    } else if (std::fabs(tiltAngle_rad) > 1e-3) {
      tiltAngle_rad = std::copysign(tiltAngle_rad, z_cm);
    }

    if (location == GeomDetEnumerators::Location::barrel) {
      avg_r_cm[layer - 1] += rho_cm;
      avg_r_counter[layer - 1] += 1;
    } else {
      avg_z_cm[layer - 1] += std::fabs(z_cm);
      avg_z_counter[layer - 1] += 1;
    }

    lstgeometry::ModuleInfo module{moduleType,
                                   subdet,
                                   location,
                                   side,
                                   layer,
                                   ring,
                                   isLower,
                                   rho_cm,
                                   z_cm,
                                   tiltAngle_rad,
                                   0.0,
                                   0.0,
                                   phi_rad,
                                   meanWidth_cm,
                                   length_cm,
                                   sensorSpacing_cm,
                                   lstgeometry::MatrixD8x3::Zero()};
    modules[detid] = std::move(module);
  }

  for (size_t i = 0; i < avg_r_cm.size(); ++i) {
    avg_r_cm[i] /= avg_r_counter[i];
  }
  for (size_t i = 0; i < avg_z_cm.size(); ++i) {
    avg_z_cm[i] /= avg_z_counter[i];
  }

  auto lstGeometry = std::make_unique<lstgeometry::Geometry>(modules, sensors, avg_r_cm, avg_z_cm, ptCut_);

  return lstGeometry;
}

DEFINE_FWK_EVENTSETUP_MODULE(LSTGeometryESProducer);
