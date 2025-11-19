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
  
  // Maybe limit it to a subset of dets
  for (auto &det : trackerGeom_->dets()) {
      const DetId detId = det->geographicalId();
      
      const auto &surface = det->surface();
      const Bounds *b = &(surface).bounds();
      const auto &position = surface.position();
      
      const double rho_mm = position.perp() * 10;
      const double z_mm = position.z() * 10;
      const double phi_rad = position.phi();
      
      // if (detId() != 441201726)
      //     continue;
    
      // std::cout << "DetId " << detId() << "\n";
      // std::cout << "det " << detId.det() << "\n";
      // std::cout << "subdet " << detId.subdetId() << "\n";
      // std::cout << "layer " << trackerTopo_->layer(detId) << "\n";
      // std::cout << "print " << trackerTopo_->print(detId) << "\n";
      
      double vtxOneX_mm, vtxOneY_mm, vtxTwoX_mm, vtxTwoY_mm, vtxThreeX_mm, vtxThreeY_mm, vtxFourX_mm, vtxFourY_mm;
      
      if (const TrapezoidalPlaneBounds *b2 = dynamic_cast<const TrapezoidalPlaneBounds *>(b)) {
          // See sec. "TrapezoidalPlaneBounds parameters" in doc/reco-geom-notes.txt
          std::array<const float, 4> const &par = b2->parameters();
          vtxOneX_mm = rho_mm * cos(phi_rad) -par[0]* 10 * sin(phi_rad);
          vtxOneY_mm = rho_mm * sin(phi_rad) -par[3]* 10 * cos(phi_rad);
          vtxTwoX_mm = rho_mm * cos(phi_rad) -par[1]* 10 * sin(phi_rad);
          vtxTwoY_mm = rho_mm * sin(phi_rad) + par[3]* 10 * cos(phi_rad);
          vtxThreeX_mm = rho_mm * cos(phi_rad) + par[1]* 10 * sin(phi_rad);
          vtxThreeY_mm = rho_mm * sin(phi_rad) + par[3]* 10 * cos(phi_rad);
          vtxFourX_mm = rho_mm * cos(phi_rad) + par[0]* 10 * sin(phi_rad);
          vtxFourY_mm = rho_mm * sin(phi_rad) + -par[3]* 10 * cos(phi_rad);
          //dz = par[2];
          //ms.round_assign(par[0], par[1], par[3], par[2]);
        } else if (const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b)) {
          // Rectangular
          float dx = b2->width()* 10 * 0.5;   // half width
          float dy = b2->length()* 10 * 0.5;  // half length
          vtxOneX_mm = rho_mm * cos(phi_rad)  -dx * sin(phi_rad);
          vtxOneY_mm = rho_mm * sin(phi_rad) -dy * cos(phi_rad);
          vtxTwoX_mm = rho_mm * cos(phi_rad) -dx * sin(phi_rad);
          vtxTwoY_mm = rho_mm * sin(phi_rad) + dy * cos(phi_rad);
          vtxThreeX_mm = rho_mm * cos(phi_rad) + dx * sin(phi_rad);
          vtxThreeY_mm = rho_mm * sin(phi_rad) + dy * cos(phi_rad);
          vtxFourX_mm = rho_mm * cos(phi_rad) + dx * sin(phi_rad);
          vtxFourY_mm = rho_mm * sin(phi_rad) -dy * cos(phi_rad);
          //dz = b2->thickness() * 0.5;  // half thickness
          //ms.round_assign(dx, 0.0f, dy, dz);
        } else {
          throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
        }
       // std::cout << detId() << " " << rho_mm << " " << z_mm << " " << phi_rad << " " << vtxOneX_mm << " " << vtxOneY_mm << " " << vtxTwoX_mm << " " << vtxTwoY_mm << " " << vtxThreeX_mm << " " << vtxThreeY_mm << " " << vtxFourX_mm << " " << vtxFourY_mm << std::endl;
  }

  // placeholder
  return std::make_unique<std::string>("LSTGeometryESProducer");
}

DEFINE_FWK_EVENTSETUP_MODULE(LSTGeometryESProducer);
