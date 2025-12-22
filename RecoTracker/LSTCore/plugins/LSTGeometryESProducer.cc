#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
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
  std::string ptCut_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;

  const TrackerGeometry *trackerGeom_ = nullptr;
};

LSTGeometryESProducer::LSTGeometryESProducer(const edm::ParameterSet &iConfig)
    : ptCut_(iConfig.getParameter<std::string>("ptCut")) {
  auto cc = setWhatProduced(this, ptCut_);
  geomToken_ = cc.consumes();
}

void LSTGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ptCut", "0.8");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<lstgeometry::LSTGeometry> LSTGeometryESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
  trackerGeom_ = &iRecord.get(geomToken_);

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

    const double rho_cm = position.perp();
    const double z_cm = lstgeometry::roundCoordinate(position.z());
    const double phi_rad = lstgeometry::roundAngle(position.phi());

    if (det->isLeaf()) {
      // Leafs are the sensors
      lstgeometry::SensorInfo sensor{detId(), rho_cm, z_cm, phi_rad};
      sensors[detId()] = std::move(sensor);
      continue;
    }

    const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b);
    if (!b2) {
      throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
    }

    double tiltAngle_rad = lstgeometry::roundAngle(std::asin(det->rotation().zz()));

    double meanWidth_cm = b2->width();
    double length_cm = b2->length();

    double sensorSpacing_cm = det->components()[0]->toLocal(det->components()[1]->position()).mag();

    unsigned int detid = detId();

    unsigned short layer = lstgeometry::Module::parseLayer(detid);

    // This part is a little weird, but this is how to match the csv files.
    // I think it might be better to not do this since other parts of the code
    // assume the center of the module is at (rho_cm, z_cm).
    //
    // z_cm += sensorSpacing_cm / 2.0 * std::sin(tiltAngle_rad);
    // bool isFlipped = surface.normalVector().basicVector().dot(position.basicVector()) < 0;
    // rho_cm += (isFlipped ? -1 : 1) * signsensorSpacing_cm / 2.0 * std::cos(tiltAngle_rad);

    // Fix angles of some modules
    if (std::fabs(std::fabs(tiltAngle_rad) - std::numbers::pi_v<double> / 2) < 1e-3) {
      tiltAngle_rad = std::numbers::pi_v<double> / 2;
    } else if (std::fabs(tiltAngle_rad) > 1e-3) {
      tiltAngle_rad = std::copysign(tiltAngle_rad, z_cm);
    }

    if (lstgeometry::Module::parseSubdet(detid) == lstgeometry::Module::SubDet::Barrel) {
      avg_r_cm[layer - 1] += rho_cm;
      avg_r_counter[layer - 1] += 1;
    } else {
      avg_z_cm[layer - 1] += std::fabs(z_cm);
      avg_z_counter[layer - 1] += 1;
    }

    lstgeometry::ModuleInfo module{detid,
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
    modules.push_back(module);
  }

  for (size_t i = 0; i < avg_r_cm.size(); ++i) {
    avg_r_cm[i] /= avg_r_counter[i];
  }
  for (size_t i = 0; i < avg_z_cm.size(); ++i) {
    avg_z_cm[i] /= avg_z_counter[i];
  }

  double ptCut = std::stod(ptCut_);
  auto lstGeometry = makeLSTGeometry(modules, sensors, avg_r_cm, avg_z_cm, ptCut);

  return lstGeometry;
}

DEFINE_FWK_EVENTSETUP_MODULE(LSTGeometryESProducer);
