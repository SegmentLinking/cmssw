#include "RecoTracker/LSTGeometry/interface/Geometry.h"
#include "RecoTracker/LSTGeometry/interface/DetectorGeometry.h"
#include "RecoTracker/LSTGeometry/interface/Slope.h"

using namespace lstgeometry;

Geometry::Geometry(std::shared_ptr<Sensors> sensors,
                   std::array<float, kBarrelLayers> const &average_r_barrel,
                   std::array<float, kEndcapLayers> const &average_z_endcap,
                   float pt_cut)
    : sensors(sensors) {
  auto slopes = computeSlopes(*sensors);
  barrel_slopes = std::move(std::get<0>(slopes));
  endcap_slopes = std::move(std::get<1>(slopes));

  auto det_geom = DetectorGeometry(sensors, average_r_barrel, average_z_endcap);
  det_geom.buildByLayer(*sensors);

  pixel_map = buildPixelMap(*sensors, det_geom, pt_cut);

  module_map = buildModuleMap(*sensors, det_geom, pt_cut);
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(Geometry);
