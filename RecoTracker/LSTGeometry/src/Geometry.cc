#include "RecoTracker/LSTGeometry/interface/Geometry.h"
#include "RecoTracker/LSTGeometry/interface/DetectorGeometry.h"
#include "RecoTracker/LSTGeometry/interface/Slope.h"

using namespace lstgeometry;

Geometry::Geometry(Modules &modules,
                   std::shared_ptr<Sensors> sensors,
                   std::vector<float> const &average_r,
                   std::vector<float> const &average_z,
                   float pt_cut)
    : sensors(sensors) {
  auto slopes = computeSlopes(modules, *sensors);
  barrel_slopes = std::move(std::get<0>(slopes));
  endcap_slopes = std::move(std::get<1>(slopes));

  auto det_geom = DetectorGeometry(sensors, average_r, average_z);
  det_geom.buildByLayer(modules, *sensors);

  pixel_map = buildPixelMap(modules, *sensors, det_geom, pt_cut);

  module_map = buildModuleMap(modules, *sensors, det_geom, pt_cut);
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(Geometry);
