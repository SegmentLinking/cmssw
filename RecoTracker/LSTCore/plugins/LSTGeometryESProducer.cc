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
      
      double rho_mm = position.perp() * 10;
      double z_mm = position.z() * 10;
      const double phi_rad = position.phi();
      
      if (det->isLeaf()) {
          // Leafs are the sensors 
            lstgeometry::SensorInfo sensor;
            sensor.detId = detId();
            sensor.sensorCenterRho_mm = rho_mm;
            sensor.sensorCenterZ_mm = z_mm;
            sensor.phi_rad = phi_rad;
            sensors.push_back(sensor);
            continue;
      }
      
      double tiltAngle_rad = std::asin(det->rotation().zz());
      
      double meanWidth_mm = 0.0;
      double length_mm = 0.0;
      
      if (det->components().size() != 2) {
          std::cout << "Not equal to 2! " << det->components().size() << "\n";
          continue;
      }
      
      double sensorSpacing_mm = det->components()[0]->toLocal(det->components()[1]->position()).mag() * 10;
      
      // if (detId() != 438043652)
      //     continue;
    
      // std::cout << "DetId " << detId() << "\n";
      // std::cout << "rho_mm " << rho_mm << "\n";
      // std::cout << "z_mm " << z_mm << "\n";
      // std::cout << "det " << detId.det() << "\n";
      // std::cout << "subdet " << detId.subdetId() << "\n";
      // std::cout << "layer " << trackerTopo_->layer(detId) << "\n";
      // std::cout << "print " << trackerTopo_->print(detId) << "\n";
      
      double vtxOneX_mm, vtxOneY_mm, vtxTwoX_mm, vtxTwoY_mm, vtxThreeX_mm, vtxThreeY_mm, vtxFourX_mm, vtxFourY_mm;
      double vtxOneX_mm_tmp, vtxOneY_mm_tmp, vtxTwoX_mm_tmp, vtxTwoY_mm_tmp, vtxThreeX_mm_tmp, vtxThreeY_mm_tmp, vtxFourX_mm_tmp, vtxFourY_mm_tmp;
      
      if (GeomDetEnumerators::isBarrel(det->subDetector())) {
          if (const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b)) {
              // Rectangular
              meanWidth_mm = b2->width() * 10;
              length_mm = b2->length() * 10;
              float dx = b2->width()* 10 * 0.5;   // half width
              float dy = b2->length()* 10 * 0.5;  // half length
              // std::cout << "Rectangle parameters " << b2->width() << " " << b2->length() << "\n";
              // std::cout << std::sin(phi_rad) << "\n";
              // vtxOneX_mm = rho_mm * cos(phi_rad)  -dx * sin(phi_rad);
              // vtxOneY_mm = rho_mm * sin(phi_rad) -dy * cos(phi_rad);
              // vtxTwoX_mm = rho_mm * cos(phi_rad) -dx * sin(phi_rad);
              // vtxTwoY_mm = rho_mm * sin(phi_rad) + dy * cos(phi_rad);
              // vtxThreeX_mm = rho_mm * cos(phi_rad) + dx * sin(phi_rad);
              // vtxThreeY_mm = rho_mm * sin(phi_rad) + dy * cos(phi_rad);
              // vtxFourX_mm = rho_mm * cos(phi_rad) + dx * sin(phi_rad);
              // vtxFourY_mm = rho_mm * sin(phi_rad) -dy * cos(phi_rad);
              vtxOneX_mm_tmp = rho_mm - dy * std::sin(tiltAngle_rad);
              vtxOneY_mm_tmp = dx;
              vtxTwoX_mm_tmp = rho_mm + dy * std::sin(tiltAngle_rad);
              vtxTwoY_mm_tmp = dx;
              vtxThreeX_mm_tmp = rho_mm - dy * std::sin(tiltAngle_rad);
              vtxThreeY_mm_tmp = -dx;
              vtxFourX_mm_tmp = rho_mm + dy * std::sin(tiltAngle_rad);
              vtxFourY_mm_tmp = -dx;
            } else {
              throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
            }
          // rho_mm -= sensorSpacing_mm/2.0;
      } else {
          if (const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b)) {
              // Rectangular
              meanWidth_mm = b2->width() * 10;
              length_mm = b2->length() * 10;
              float dx = b2->width()* 10 * 0.5;   // half width
              float dy = b2->length()* 10 * 0.5;  // half length
              // std::cout << "Rectangle parameters " << b2->width() << " " << b2->length() << "\n";
              vtxOneX_mm_tmp = rho_mm + dy;
              vtxOneY_mm_tmp = dx;
              vtxTwoX_mm_tmp = rho_mm - dy;
              vtxTwoY_mm_tmp = dx;
              vtxThreeX_mm_tmp = rho_mm - dy;
              vtxThreeY_mm_tmp = -dx;
              vtxFourX_mm_tmp = rho_mm + dy;
              vtxFourY_mm_tmp = -dx;
              std::cout << detId() << " " << vtxOneX_mm_tmp << " " << vtxOneY_mm_tmp << " " << vtxTwoX_mm_tmp << " " << vtxTwoY_mm_tmp << " " << vtxThreeX_mm_tmp << " " << vtxThreeY_mm_tmp << " " << vtxFourX_mm_tmp << " " << vtxFourY_mm_tmp << std::endl;
            } else {
              throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
            }
          
          // z_mm -= sensorSpacing_mm/2.0; // Not sure why
      }
      
      vtxOneX_mm = vtxOneX_mm_tmp * cos(phi_rad) + vtxOneY_mm_tmp * sin(phi_rad);
      vtxOneY_mm = vtxOneX_mm_tmp * sin(phi_rad) - vtxOneY_mm_tmp * cos(phi_rad);
      vtxTwoX_mm = vtxTwoX_mm_tmp * cos(phi_rad) + vtxTwoY_mm_tmp * sin(phi_rad);
      vtxTwoY_mm = vtxTwoX_mm_tmp * sin(phi_rad) - vtxTwoY_mm_tmp * cos(phi_rad);
      vtxThreeX_mm = vtxThreeX_mm_tmp * cos(phi_rad) + vtxThreeY_mm_tmp * sin(phi_rad);
      vtxThreeY_mm = vtxThreeX_mm_tmp * sin(phi_rad) - vtxThreeY_mm_tmp * cos(phi_rad);
      vtxFourX_mm = vtxFourX_mm_tmp * cos(phi_rad) + vtxFourY_mm_tmp * sin(phi_rad);
      vtxFourY_mm = vtxFourX_mm_tmp * sin(phi_rad) - vtxFourY_mm_tmp * cos(phi_rad);
      
       std::cout << detId() << " " << rho_mm << " " << z_mm << " " << phi_rad << " " << vtxOneX_mm << " " << vtxOneY_mm << " " << vtxTwoX_mm << " " << vtxTwoY_mm << " " << vtxThreeX_mm << " " << vtxThreeY_mm << " " << vtxFourX_mm << " " << vtxFourY_mm << " " << meanWidth_mm << " " << length_mm << " " << sensorSpacing_mm << " " << tiltAngle_rad << std::endl;
  }

  // placeholder
  return std::make_unique<std::string>("LSTGeometryESProducer");
}

DEFINE_FWK_EVENTSETUP_MODULE(LSTGeometryESProducer);
