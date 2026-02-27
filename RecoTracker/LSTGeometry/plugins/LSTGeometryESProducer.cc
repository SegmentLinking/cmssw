#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"

#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"
#include "RecoTracker/LSTGeometry/interface/Geometry.h"

#include <cmath>
#include <vector>
#include <unordered_map>
#include <iostream>  /////////////////////// remove

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

  lstgeometry::Modules modules;
  lstgeometry::Sensors sensors;

  std::vector<float> avg_r_cm(6, 0.0);
  std::vector<float> avg_z_cm(5, 0.0);
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

    const float rho_cm = position.perp();
    const float z_cm = lstgeometry::roundCoordinate(position.z());
    const float phi_rad = lstgeometry::roundAngle(position.phi());

    if (det->isLeaf()) {
      // Leafs are the sensors
      const unsigned int moduleDetId = detid & ~0b11;  // TODO: Is there a CMSSW method for this?
      sensors[detid] = lstgeometry::Sensor(moduleDetId, rho_cm, z_cm, phi_rad, moduleType);

      ///////////// tmp
      // const RectangularPlaneBounds *sensor_bounds = dynamic_cast<const RectangularPlaneBounds *>(b);
      // float wid = sensor_bounds->width();
      // float len = sensor_bounds->length();
      // auto c1 = GloballyPositioned<float>::LocalPoint(wid / 2, len / 2, 0);
      // auto c2 = GloballyPositioned<float>::LocalPoint(-wid / 2, len / 2, 0);
      // auto c3 = GloballyPositioned<float>::LocalPoint(-wid / 2, -len / 2, 0);
      // auto c4 = GloballyPositioned<float>::LocalPoint(wid / 2, -len / 2, 0);
      // auto c1g = surface.toGlobal(c1);
      // auto c2g = surface.toGlobal(c2);
      // auto c3g = surface.toGlobal(c3);
      // auto c4g = surface.toGlobal(c4);
      // if (detid == 440165400 + 1) {
      //   std::cout << "Corners for detid " << detid << ": " << std::endl;
      //   std::cout << c1g << std::endl;
      //   std::cout << c2g << std::endl;
      //   std::cout << c3g << std::endl;
      //   std::cout << c4g << std::endl;
      // }
      ///

      continue;
    }

    const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b);
    if (!b2) {
      throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
    }

    const auto subdet = trackerGeom_->geomDetSubDetector(detId.subdetId());
    const auto side = trackerTopo_->barrelTiltTypeP2(detId);
    // GeomDetEnumerators::isBarrel doesn't give the right answer
    const auto location =
        (subdet == lstgeometry::SubDetector::TEC ? lstgeometry::Location::barrel : lstgeometry::Location::endcap);
    const unsigned int layer = trackerTopo_->layer(detId);
    const unsigned int ring = trackerTopo_->endcapRingP2(detId);
    const bool isLower = trackerTopo_->isLower(detId);

    // std::cout << "Processing detId " << detid << " with subdet " << subdet << ", " << static_cast<unsigned int>(subdet)
    //           << " layer " << layer << " ring " << ring << " side " << side << ", isbarrel "
    //           << GeomDetEnumerators::isBarrel(subdet) << std::endl;  ////////////////////// remove

    float tiltAngle_rad = lstgeometry::roundAngle(std::asin(det->rotation().zz()));

    float meanWidth_cm = b2->width();
    float length_cm = b2->length();

    float sensorSpacing_cm = det->components()[0]->toLocal(det->components()[1]->position()).mag();

    // Fix angles of some modules
    if (std::fabs(std::fabs(tiltAngle_rad) - std::numbers::pi_v<float> / 2) < 1e-3) {
      tiltAngle_rad = std::numbers::pi_v<float> / 2;
    } else if (std::fabs(tiltAngle_rad) > 1e-3) {
      tiltAngle_rad = std::copysign(tiltAngle_rad, z_cm);
    }

    if (location == lstgeometry::Location::barrel) {
      avg_r_cm[layer - 1] += rho_cm;
      avg_r_counter[layer - 1] += 1;
    } else {
      avg_z_cm[layer - 1] += std::fabs(z_cm);
      avg_z_counter[layer - 1] += 1;
    }

    lstgeometry::Module module{moduleType,
                               subdet,
                               location,
                               side,
                               layer,
                               ring,
                               isLower,
                               rho_cm,
                               z_cm,
                               phi_rad,
                               tiltAngle_rad,
                               0.0,
                               0.0,
                               meanWidth_cm,
                               length_cm,
                               sensorSpacing_cm,
                               lstgeometry::MatrixF8x3::Zero()};
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
