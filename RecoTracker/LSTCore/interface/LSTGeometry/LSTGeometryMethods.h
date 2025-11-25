#ifndef RecoTracker_LSTCore_interface_LSTGeometry_LSTGeometryMethods_h
#define RecoTracker_LSTCore_interface_LSTGeometry_LSTGeometryMethods_h

#include "RecoTracker/LSTCore/interface/LSTGeometry/LSTGeometry.h"

namespace lstgeometry {

  std::unique_ptr<LSTGeometry> makeLSTGeometry(std::vector<ModuleInfo> &modules_info,
                                               std::unordered_map<unsigned int, SensorInfo> &sensors_info,
                                               std::vector<double> &average_r,
                                               std::vector<double> &average_z) {
    for (auto &mod : modules_info)
      transformSensorCorners(mod);

    auto assigned_corners = assignCornersToSensors(modules_info, sensors_info);

    auto centroids = computeCentroids(sensors_info);

    auto [barrel_slopes, endcap_slopes] = processCorners(assigned_corners);

    auto det_geom = DetectorGeometry(assigned_corners, average_r, average_z);
    det_geom.buildByLayer();

    auto pixel_map = computePixelMap(centroids, det_geom);

    auto detids_etaphi_layer_ref = det_geom.getDetIds([](const auto &x) {
      auto mod = Module(x.first);
      return ((mod.subdet() == 5 && mod.isLower() == 1 && mod.layer() != 6) ||
              (mod.subdet() == 4 && mod.isLower() == 1 && mod.layer() != 5 && !(mod.ring() == 15 && mod.layer() == 1) &&
               !(mod.ring() == 15 && mod.layer() == 2) && !(mod.ring() == 12 && mod.layer() == 3) &&
               !(mod.ring() == 12 && mod.layer() == 4)));
    });

    std::unordered_map<unsigned int, std::vector<unsigned int>> straight_line_connections;
    std::unordered_map<unsigned int, std::vector<unsigned int>> curved_line_connections;

    for (auto ref_detid : detids_etaphi_layer_ref) {
      straight_line_connections[ref_detid] = getStraightLineConnections(ref_detid, centroids, det_geom);
      curved_line_connections[ref_detid] = getCurvedLineConnections(ref_detid, centroids, det_geom);
    }
    auto merged_line_connections = mergeLineConnections({&straight_line_connections, &curved_line_connections});

    auto lstGeometry = std::make_unique<LSTGeometry>(std::move(centroids),
                                                     std::move(barrel_slopes),
                                                     std::move(endcap_slopes),
                                                     std::move(pixel_map),
                                                     std::move(merged_line_connections));

    return lstGeometry;
  }

}  // namespace lstgeometry

#endif
