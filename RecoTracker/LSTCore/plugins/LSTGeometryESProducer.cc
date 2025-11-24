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

#include "RecoTracker/LSTCore/interface/LSTGeometry/SensorInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/ModuleInfo.h"

#include <cmath>
#include <vector>

// temporary
#include "FWCore/Utilities/interface/typelookup.h"
#include <string>

TYPELOOKUP_DATA_REG(std::string);

// LST includes

class LSTGeometryESProducer : public edm::ESProducer {
public:
  LSTGeometryESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<std::string> produce(const TrackerRecoGeometryRecord &iRecord);

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

std::unique_ptr<std::string> LSTGeometryESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
  trackerGeom_ = &iRecord.get(geomToken_);
  trackerTopo_ = &iRecord.get(ttopoToken_);
  
  std::vector<lstgeometry::SensorInfo> sensors;
  std::vector<lstgeometry::ModuleInfo> modules;
  
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
            sensors.push_back(sensor);
            continue;
      }
      
      double tiltAngle_rad = std::asin(det->rotation().zz());
      
      double meanWidth_cm = 0.0;
      double length_cm = 0.0;
      
      if (det->components().size() != 2) {
          std::cout << "Not equal to 2! " << det->components().size() << "\n";
          continue;
      }
      
      double sensorSpacing_cm = det->components()[0]->toLocal(det->components()[1]->position()).mag() * 10;
      
      // if (detId() != 438043652)
      //     continue;
    
      // std::cout << "DetId " << detId() << "\n";
      // std::cout << "rho_cm " << rho_cm << "\n";
      // std::cout << "z_cm " << z_cm << "\n";
      // std::cout << "det " << detId.det() << "\n";
      // std::cout << "subdet " << detId.subdetId() << "\n";
      // std::cout << "layer " << trackerTopo_->layer(detId) << "\n";
      // std::cout << "print " << trackerTopo_->print(detId) << "\n";
      
      double vtxOneX_cm, vtxOneY_cm, vtxTwoX_cm, vtxTwoY_cm, vtxThreeX_cm, vtxThreeY_cm, vtxFourX_cm, vtxFourY_cm;
      double vtxOneX_cm_tmp, vtxOneY_cm_tmp, vtxTwoX_cm_tmp, vtxTwoY_cm_tmp, vtxThreeX_cm_tmp, vtxThreeY_cm_tmp, vtxFourX_cm_tmp, vtxFourY_cm_tmp;
      
      if (GeomDetEnumerators::isBarrel(det->subDetector())) {
          if (const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b)) {
              // Rectangular
              meanWidth_cm = b2->width();
              length_cm = b2->length();
              float dx = b2->width() * 0.5;   // half width
              float dy = b2->length() * 0.5;  // half length
              // std::cout << "Rectangle parameters " << b2->width() << " " << b2->length() << "\n";
              // std::cout << std::sin(phi_rad) << "\n";
              // vtxOneX_cm = rho_cm * cos(phi_rad)  -dx * sin(phi_rad);
              // vtxOneY_cm = rho_cm * sin(phi_rad) -dy * cos(phi_rad);
              // vtxTwoX_cm = rho_cm * cos(phi_rad) -dx * sin(phi_rad);
              // vtxTwoY_cm = rho_cm * sin(phi_rad) + dy * cos(phi_rad);
              // vtxThreeX_cm = rho_cm * cos(phi_rad) + dx * sin(phi_rad);
              // vtxThreeY_cm = rho_cm * sin(phi_rad) + dy * cos(phi_rad);
              // vtxFourX_cm = rho_cm * cos(phi_rad) + dx * sin(phi_rad);
              // vtxFourY_cm = rho_cm * sin(phi_rad) -dy * cos(phi_rad);
              vtxOneX_cm_tmp = rho_cm - dy * std::sin(tiltAngle_rad);
              vtxOneY_cm_tmp = dx;
              vtxTwoX_cm_tmp = rho_cm + dy * std::sin(tiltAngle_rad);
              vtxTwoY_cm_tmp = dx;
              vtxThreeX_cm_tmp = rho_cm - dy * std::sin(tiltAngle_rad);
              vtxThreeY_cm_tmp = -dx;
              vtxFourX_cm_tmp = rho_cm + dy * std::sin(tiltAngle_rad);
              vtxFourY_cm_tmp = -dx;
            } else {
              throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
            }
          // rho_cm -= sensorSpacing_cm/2.0;
      } else {
          if (const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b)) {
              // Rectangular
              meanWidth_cm = b2->width();
              length_cm = b2->length();
              float dx = b2->width() * 0.5;   // half width
              float dy = b2->length() * 0.5;  // half length
              // std::cout << "Rectangle parameters " << b2->width() << " " << b2->length() << "\n";
              vtxOneX_cm_tmp = rho_cm + dy;
              vtxOneY_cm_tmp = dx;
              vtxTwoX_cm_tmp = rho_cm - dy;
              vtxTwoY_cm_tmp = dx;
              vtxThreeX_cm_tmp = rho_cm - dy;
              vtxThreeY_cm_tmp = -dx;
              vtxFourX_cm_tmp = rho_cm + dy;
              vtxFourY_cm_tmp = -dx;
              std::cout << detId() << " " << vtxOneX_cm_tmp << " " << vtxOneY_cm_tmp << " " << vtxTwoX_cm_tmp << " " << vtxTwoY_cm_tmp << " " << vtxThreeX_cm_tmp << " " << vtxThreeY_cm_tmp << " " << vtxFourX_cm_tmp << " " << vtxFourY_cm_tmp << std::endl;
            } else {
              throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
            }
          
          // z_cm -= sensorSpacing_cm/2.0; // Not sure why
      }
      
      vtxOneX_cm = vtxOneX_cm_tmp * cos(phi_rad) + vtxOneY_cm_tmp * sin(phi_rad);
      vtxOneY_cm = vtxOneX_cm_tmp * sin(phi_rad) - vtxOneY_cm_tmp * cos(phi_rad);
      vtxTwoX_cm = vtxTwoX_cm_tmp * cos(phi_rad) + vtxTwoY_cm_tmp * sin(phi_rad);
      vtxTwoY_cm = vtxTwoX_cm_tmp * sin(phi_rad) - vtxTwoY_cm_tmp * cos(phi_rad);
      vtxThreeX_cm = vtxThreeX_cm_tmp * cos(phi_rad) + vtxThreeY_cm_tmp * sin(phi_rad);
      vtxThreeY_cm = vtxThreeX_cm_tmp * sin(phi_rad) - vtxThreeY_cm_tmp * cos(phi_rad);
      vtxFourX_cm = vtxFourX_cm_tmp * cos(phi_rad) + vtxFourY_cm_tmp * sin(phi_rad);
      vtxFourY_cm = vtxFourX_cm_tmp * sin(phi_rad) - vtxFourY_cm_tmp * cos(phi_rad);
      
       std::cout << detId() << " " << rho_cm << " " << z_cm << " " << phi_rad << " " << vtxOneX_cm << " " << vtxOneY_cm << " " << vtxTwoX_cm << " " << vtxTwoY_cm << " " << vtxThreeX_cm << " " << vtxThreeY_cm << " " << vtxFourX_cm << " " << vtxFourY_cm << " " << meanWidth_cm << " " << length_cm << " " << sensorSpacing_cm << " " << tiltAngle_rad << std::endl;
  }

  // placeholder
  return std::make_unique<std::string>("LSTGeometryESProducer");
}

DEFINE_FWK_EVENTSETUP_MODULE(LSTGeometryESProducer);
