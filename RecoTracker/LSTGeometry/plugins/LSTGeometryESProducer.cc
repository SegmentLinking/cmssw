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
#include <sstream>

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
  std::ostringstream ptCutOSS;
  ptCutOSS << std::setprecision(1) << ptCut_;
  std::string ptCutStr = ptCutOSS.str();

  auto cc = setWhatProduced(this, "LSTGeometry_" + ptCutStr);
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
  auto sensors = std::make_shared<lstgeometry::Sensors>();

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
    const auto &position = surface.position();

    const float rho_cm = position.perp();
    const float z_cm = lstgeometry::roundCoordinate(position.z());
    const float phi_rad = lstgeometry::roundAngle(position.phi());

    const auto subdet = trackerGeom_->geomDetSubDetector(detId.subdetId());
    const auto location =
        GeomDetEnumerators::isBarrel(subdet) ? lstgeometry::Location::barrel : lstgeometry::Location::endcap;
    const auto side = static_cast<lstgeometry::Side>(
        location == lstgeometry::Location::barrel ? static_cast<unsigned int>(trackerTopo_->barrelTiltTypeP2(detId))
                                                  : trackerTopo_->side(detId));
    const unsigned int moduleId = trackerTopo_->module(detId);
    const unsigned int layer = trackerTopo_->layer(detId);
    const unsigned int ring = trackerTopo_->endcapRingP2(detId);

    if (det->isLeaf()) {
      // Leafs are the sensors
      const unsigned int moduleDetId = detid & ~0b11;  // I don't think there is there a CMSSW method for this
      // Can't use TrackerTopology::isLower since it doesn't consider if the module is inverted
      const bool isLow = isLower(moduleId, location, side, layer, detid);
      (*sensors)[detid] = lstgeometry::Sensor(moduleDetId, rho_cm, z_cm, phi_rad, isLow, moduleType, surface);

      continue;
    }

    if (location == lstgeometry::Location::barrel) {
      avg_r_cm[layer - 1] += rho_cm;
      avg_r_counter[layer - 1] += 1;
    } else {
      avg_z_cm[layer - 1] += std::fabs(z_cm);
      avg_z_counter[layer - 1] += 1;
    }

    lstgeometry::Module module{moduleType, subdet, location, side, moduleId, layer, ring, rho_cm, z_cm, phi_rad};
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
