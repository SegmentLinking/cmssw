#include "RecoTracker/LSTGeometry/interface/Geometry.h"
#include "RecoTracker/LSTGeometry/interface/CornerMethods.h"
#include "RecoTracker/LSTGeometry/interface/DetectorGeometry.h"
#include "RecoTracker/LSTGeometry/interface/OrientationMethods.h"

using namespace lstgeometry;

Geometry::Geometry(Modules &modules,
                   Sensors &sensors_input,
                   std::vector<float> const &average_r,
                   std::vector<float> const &average_z,
                   double pt_cut) {
  for (auto &[_, mod] : modules)
    transformSensorCorners(mod);

  assignCornersToSensors(modules, sensors_input);

  auto slopes = processCorners(modules, sensors_input);
  barrel_slopes = std::move(std::get<0>(slopes));
  endcap_slopes = std::move(std::get<1>(slopes));

  auto det_geom = DetectorGeometry(sensors_input, average_r, average_z);
  det_geom.buildByLayer(modules, sensors_input);

  pixel_map = buildPixelMap(modules, det_geom, pt_cut);

  module_map = buildModuleMap(modules, sensors_input, det_geom, pt_cut);

  sensors = sensors_input;
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(Geometry);
