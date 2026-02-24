#include "RecoTracker/LSTGeometry/interface/Geometry.h"
#include "RecoTracker/LSTGeometry/interface/CornerMethods.h"
#include "RecoTracker/LSTGeometry/interface/DetectorGeometry.h"
#include "RecoTracker/LSTGeometry/interface/OrientationMethods.h"
#include "RecoTracker/LSTGeometry/interface/PixelMapMethods.h"
#include "RecoTracker/LSTGeometry/interface/ModuleMapMethods.h"

using namespace lstgeometry;

Geometry::Geometry(std::unordered_map<unsigned int, ModuleInfo> &modules_info,
                   std::unordered_map<unsigned int, Sensor> &sensors_input,
                   std::vector<double> const &average_r,
                   std::vector<double> const &average_z,
                   double ptCut) {
  for (auto &[_, mod] : modules_info)
    transformSensorCorners(mod);

  auto assigned_corners = assignCornersToSensors(modules_info, sensors_input);

  auto slopes = processCorners(assigned_corners);
  barrel_slopes = std::move(std::get<0>(slopes));
  endcap_slopes = std::move(std::get<1>(slopes));

  auto det_geom = DetectorGeometry(assigned_corners, average_r, average_z);
  det_geom.buildByLayer(modules_info, sensors_input);

  pixel_map = computePixelMap(det_geom, ptCut);

  auto detids_etaphi_layer_ref = det_geom.getDetIds([&modules_info, &sensors_input](const auto &x) {
    auto mod = modules_info.at(sensors_input.at(x.first).moduleDetId);
    // exclude the outermost modules that do not have connections to other modules
    return ((mod.subdet == 5 && mod.isLower && mod.layer != 6) ||
            (mod.subdet == 4 && mod.isLower && mod.layer != 5 && !(mod.ring == 15 && mod.layer == 1) &&
             !(mod.ring == 15 && mod.layer == 2) && !(mod.ring == 12 && mod.layer == 3) &&
             !(mod.ring == 12 && mod.layer == 4)));
  });

  std::unordered_map<unsigned int, std::vector<unsigned int>> straight_line_connections;
  std::unordered_map<unsigned int, std::vector<unsigned int>> curved_line_connections;

  for (auto ref_detid : detids_etaphi_layer_ref) {
    straight_line_connections[ref_detid] = getStraightLineConnections(ref_detid, sensors_input, det_geom);
    curved_line_connections[ref_detid] = getCurvedLineConnections(ref_detid, sensors_input, det_geom, ptCut);
  }
  merged_line_connections = mergeLineConnections({&straight_line_connections, &curved_line_connections});

  sensors = sensors_input;
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(Geometry);
