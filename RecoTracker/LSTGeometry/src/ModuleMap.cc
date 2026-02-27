#include <iterator>
#include <vector>
#include <tuple>
#include <initializer_list>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "RecoTracker/LSTGeometry/interface/ModuleMap.h"
#include "RecoTracker/LSTGeometry/interface/Helix.h"

namespace lstgeometry {

  using Point = boost::geometry::model::d2::point_xy<float>;
  using Polygon = boost::geometry::model::polygon<Point>;

  Polygon getEtaPhiPolygon(MatrixFNx3 const& mod_boundaries, float refphi, float zshift = 0) {
    int npoints = mod_boundaries.rows();
    MatrixFNx2 mod_boundaries_etaphi(npoints, 2);
    for (int i = 0; i < npoints; ++i) {
      auto ref_etaphi = getEtaPhi(mod_boundaries(i, 1), mod_boundaries(i, 2), mod_boundaries(i, 0) + zshift, refphi);
      mod_boundaries_etaphi(i, 0) = ref_etaphi.first;
      mod_boundaries_etaphi(i, 1) = ref_etaphi.second;
    }

    Polygon poly;
    // <= because we need to close the polygon with the first point
    for (int i = 0; i <= npoints; ++i) {
      boost::geometry::append(poly,
                              Point(mod_boundaries_etaphi(i % npoints, 0), mod_boundaries_etaphi(i % npoints, 1)));
    }
    boost::geometry::correct(poly);
    return poly;
  }

  bool moduleOverlapsInEtaPhi(MatrixF4x3 const& ref_mod_boundaries,
                              MatrixF4x3 const& tar_mod_boundaries,
                              float refphi = 0,
                              float zshift = 0) {
    RowVectorF3 ref_center = ref_mod_boundaries.colwise().sum();
    ref_center /= 4.;
    RowVectorF3 tar_center = tar_mod_boundaries.colwise().sum();
    tar_center /= 4.;

    float ref_center_phi = std::atan2(ref_center(2), ref_center(1));
    float tar_center_phi = std::atan2(tar_center(2), tar_center(1));

    if (std::fabs(phi_mpi_pi(ref_center_phi - tar_center_phi)) > std::numbers::pi_v<float> / 2.)
      return false;

    MatrixF4x2 ref_mod_boundaries_etaphi;
    MatrixF4x2 tar_mod_boundaries_etaphi;

    for (int i = 0; i < 4; ++i) {
      auto ref_etaphi =
          getEtaPhi(ref_mod_boundaries(i, 1), ref_mod_boundaries(i, 2), ref_mod_boundaries(i, 0) + zshift, refphi);
      auto tar_etaphi =
          getEtaPhi(tar_mod_boundaries(i, 1), tar_mod_boundaries(i, 2), tar_mod_boundaries(i, 0) + zshift, refphi);
      ref_mod_boundaries_etaphi(i, 0) = ref_etaphi.first;
      ref_mod_boundaries_etaphi(i, 1) = ref_etaphi.second;
      tar_mod_boundaries_etaphi(i, 0) = tar_etaphi.first;
      tar_mod_boundaries_etaphi(i, 1) = tar_etaphi.second;
    }

    // Quick cut
    RowVectorF2 diff = ref_mod_boundaries_etaphi.row(0) - tar_mod_boundaries_etaphi.row(0);
    if (std::fabs(diff(0)) > 0.5)
      return false;
    if (std::fabs(phi_mpi_pi(diff(1))) > 1.)
      return false;

    Polygon ref_poly, tar_poly;

    // <= 4 because we need to close the polygon with the first point
    for (int i = 0; i <= 4; ++i) {
      boost::geometry::append(ref_poly,
                              Point(ref_mod_boundaries_etaphi(i % 4, 0), ref_mod_boundaries_etaphi(i % 4, 1)));
      boost::geometry::append(tar_poly,
                              Point(tar_mod_boundaries_etaphi(i % 4, 0), tar_mod_boundaries_etaphi(i % 4, 1)));
    }
    boost::geometry::correct(ref_poly);
    boost::geometry::correct(tar_poly);

    return boost::geometry::intersects(ref_poly, tar_poly);
  }

  std::vector<unsigned int> getStraightLineConnections(unsigned int ref_detid,
                                                       Modules const& modules,
                                                       Sensors const& sensors,
                                                       DetectorGeometry const& det_geom) {
    auto& sensor = sensors.at(ref_detid);

    double refphi = std::atan2(sensor.centerY, sensor.centerX);

    auto refmodule = modules.at(sensors.at(ref_detid).moduleDetId);

    unsigned short ref_layer = refmodule.layer;
    auto ref_location = refmodule.location;

    auto etaphi = getEtaPhi(sensor.centerX, sensor.centerY, sensor.centerZ);
    auto etaphibins = DetectorGeometry::getEtaPhiBins(etaphi.first, etaphi.second);

    auto const& tar_detids_to_be_considered =
        ref_location == Location::barrel
            ? det_geom.getBarrelLayerDetIds(ref_layer + 1, etaphibins.first, etaphibins.second)
            : det_geom.getEndcapLayerDetIds(ref_layer + 1, etaphibins.first, etaphibins.second);

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    for (unsigned int tar_detid : tar_detids_to_be_considered) {
      if (moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, 0) ||
          moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, 10) ||
          moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, -10))
        list_of_detids_etaphi_layer_tar.push_back(tar_detid);
    }

    // Consider barrel to endcap connections if the intersection area is > 0
    // We construct the reference polygon as a vector of polygons because the boost::geometry::difference
    // function can return multiple polygons if the difference results in disjoint pieces
    if (ref_location == Location::barrel) {
      std::unordered_set<unsigned int> barrel_endcap_connected_tar_detids;

      for (float zshift : {0, 10, -10}) {
        std::vector<Polygon> ref_polygon;
        ref_polygon.push_back(getEtaPhiPolygon(det_geom.getCorners(ref_detid), refphi, zshift));

        // Check whether there is still significant non-zero area
        for (unsigned int tar_detid : list_of_detids_etaphi_layer_tar) {
          if (!ref_polygon.size())
            break;
          Polygon tar_polygon = getEtaPhiPolygon(det_geom.getCorners(tar_detid), refphi, zshift);

          std::vector<Polygon> difference;
          for (auto& ref_polygon_piece : ref_polygon) {
            std::vector<Polygon> tmp_difference;
            boost::geometry::difference(ref_polygon_piece, tar_polygon, tmp_difference);
            difference.insert(difference.end(), tmp_difference.begin(), tmp_difference.end());
          }

          ref_polygon = std::move(difference);
        }

        double area = 0.;
        for (auto& ref_polygon_piece : ref_polygon)
          area += boost::geometry::area(ref_polygon_piece);

        if (area <= 1e-6)
          continue;

        auto const& new_tar_detids_to_be_considered =
            det_geom.getEndcapLayerDetIds(1, etaphibins.first, etaphibins.second);

        for (unsigned int tar_detid : new_tar_detids_to_be_considered) {
          auto& sensor_target = sensors.at(tar_detid);
          double tarphi = std::atan2(sensor_target.centerY, sensor_target.centerX);

          if (std::fabs(phi_mpi_pi(tarphi - refphi)) > std::numbers::pi_v<double> / 2.)
            continue;

          Polygon tar_polygon = getEtaPhiPolygon(det_geom.getCorners(tar_detid), refphi, zshift);

          bool intersects = false;
          for (auto& ref_polygon_piece : ref_polygon) {
            if (boost::geometry::intersects(ref_polygon_piece, tar_polygon)) {
              intersects = true;
              break;
            }
          }

          if (intersects)
            barrel_endcap_connected_tar_detids.insert(tar_detid);
        }
      }
      list_of_detids_etaphi_layer_tar.insert(list_of_detids_etaphi_layer_tar.end(),
                                             barrel_endcap_connected_tar_detids.begin(),
                                             barrel_endcap_connected_tar_detids.end());
    }

    return list_of_detids_etaphi_layer_tar;
  }

  MatrixF4x3 boundsAfterCurved(unsigned int ref_detid,
                               Modules const& modules,
                               Sensors const& sensors,
                               DetectorGeometry const& det_geom,
                               double ptCut,
                               bool doR = true) {
    auto bounds = det_geom.getCorners(ref_detid);
    auto& sensor = sensors.at(ref_detid);
    int charge = 1;
    double theta =
        std::atan2(std::sqrt(sensor.centerX * sensor.centerX + sensor.centerY * sensor.centerY), sensor.centerZ);
    double refphi = std::atan2(sensor.centerY, sensor.centerX);
    auto refmodule = modules.at(sensors.at(ref_detid).moduleDetId);
    unsigned short ref_layer = refmodule.layer;
    auto ref_location = refmodule.location;
    MatrixF4x3 next_layer_bound_points;

    for (int i = 0; i < bounds.rows(); i++) {
      auto helix_p10 = Helix(ptCut, 0, 0, 10, bounds(i, 1), bounds(i, 2), bounds(i, 0), -charge);
      auto helix_m10 = Helix(ptCut, 0, 0, -10, bounds(i, 1), bounds(i, 2), bounds(i, 0), -charge);
      auto helix_p10_pos = Helix(ptCut, 0, 0, 10, bounds(i, 1), bounds(i, 2), bounds(i, 0), charge);
      auto helix_m10_pos = Helix(ptCut, 0, 0, -10, bounds(i, 1), bounds(i, 2), bounds(i, 0), charge);
      double bound_theta =
          std::atan2(std::sqrt(bounds(i, 1) * bounds(i, 1) + bounds(i, 2) * bounds(i, 2)), bounds(i, 0));
      double bound_phi = std::atan2(bounds(i, 2), bounds(i, 1));
      double phi_diff = phi_mpi_pi(bound_phi - refphi);

      std::tuple<double, double, double, double> next_point;
      if (ref_location == Location::barrel) {
        if (doR) {
          double tar_layer_radius = det_geom.getBarrelLayerAverageRadius(ref_layer + 1);
          if (bound_theta > theta) {
            auto& h = phi_diff > 0 ? helix_p10 : helix_p10_pos;
            next_point = h.pointFromRadius(tar_layer_radius);
          } else {
            auto& h = phi_diff > 0 ? helix_m10 : helix_m10_pos;
            next_point = h.pointFromRadius(tar_layer_radius);
          }
        } else {
          double tar_layer_z = det_geom.getEndcapLayerAverageAbsZ(1);
          if (bound_theta > theta) {
            if (phi_diff > 0) {
              next_point = helix_p10.pointFromZ(std::copysign(tar_layer_z, helix_p10.lambda));
            } else {
              next_point = helix_p10_pos.pointFromZ(std::copysign(tar_layer_z, helix_p10_pos.lambda));
            }
          } else {
            if (phi_diff > 0) {
              next_point = helix_m10.pointFromZ(std::copysign(tar_layer_z, helix_m10.lambda));
            } else {
              next_point = helix_m10_pos.pointFromZ(std::copysign(tar_layer_z, helix_m10_pos.lambda));
            }
          }
        }
      } else {
        double tar_layer_z = det_geom.getEndcapLayerAverageAbsZ(ref_layer + 1);
        if (bound_theta > theta) {
          if (phi_diff > 0) {
            next_point = helix_p10.pointFromZ(std::copysign(tar_layer_z, helix_p10.lambda));
          } else {
            next_point = helix_p10_pos.pointFromZ(std::copysign(tar_layer_z, helix_p10_pos.lambda));
          }
        } else {
          if (phi_diff > 0) {
            next_point = helix_m10.pointFromZ(std::copysign(tar_layer_z, helix_m10.lambda));
          } else {
            next_point = helix_m10_pos.pointFromZ(std::copysign(tar_layer_z, helix_m10_pos.lambda));
          }
        }
      }
      next_layer_bound_points(i, 0) = std::get<2>(next_point);
      next_layer_bound_points(i, 1) = std::get<0>(next_point);
      next_layer_bound_points(i, 2) = std::get<1>(next_point);
    }

    return next_layer_bound_points;
  }

  std::vector<unsigned int> getCurvedLineConnections(unsigned int ref_detid,
                                                     Modules const& modules,
                                                     Sensors const& sensors,
                                                     DetectorGeometry const& det_geom,
                                                     double ptCut) {
    auto& sensor = sensors.at(ref_detid);

    double refphi = std::atan2(sensor.centerY, sensor.centerX);

    auto refmodule = modules.at(sensors.at(ref_detid).moduleDetId);

    unsigned short ref_layer = refmodule.layer;
    auto ref_location = refmodule.location;

    auto etaphi = getEtaPhi(sensor.centerX, sensor.centerY, sensor.centerZ);
    auto etaphibins = DetectorGeometry::getEtaPhiBins(etaphi.first, etaphi.second);

    auto const& tar_detids_to_be_considered =
        ref_location == Location::barrel
            ? det_geom.getBarrelLayerDetIds(ref_layer + 1, etaphibins.first, etaphibins.second)
            : det_geom.getEndcapLayerDetIds(ref_layer + 1, etaphibins.first, etaphibins.second);

    auto next_layer_bound_points = boundsAfterCurved(ref_detid, modules, sensors, det_geom, ptCut);

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    for (unsigned int tar_detid : tar_detids_to_be_considered) {
      if (moduleOverlapsInEtaPhi(next_layer_bound_points, det_geom.getCorners(tar_detid), refphi, 0))
        list_of_detids_etaphi_layer_tar.push_back(tar_detid);
    }

    // Consider barrel to endcap connections if the intersection area is > 0
    // We construct the reference polygon as a vector of polygons because the boost::geometry::difference
    // function can return multiple polygons if the difference results in disjoint pieces
    if (ref_location == Location::barrel) {
      std::unordered_set<unsigned int> barrel_endcap_connected_tar_detids;

      int zshift = 0;

      std::vector<Polygon> ref_polygon;
      ref_polygon.push_back(getEtaPhiPolygon(next_layer_bound_points, refphi, zshift));

      // Check whether there is still significant non-zero area
      for (unsigned int tar_detid : list_of_detids_etaphi_layer_tar) {
        if (!ref_polygon.size())
          break;
        Polygon tar_polygon = getEtaPhiPolygon(det_geom.getCorners(tar_detid), refphi, zshift);

        std::vector<Polygon> difference;
        for (auto& ref_polygon_piece : ref_polygon) {
          std::vector<Polygon> tmp_difference;
          boost::geometry::difference(ref_polygon_piece, tar_polygon, tmp_difference);
          difference.insert(difference.end(), tmp_difference.begin(), tmp_difference.end());
        }

        ref_polygon = std::move(difference);
      }

      double area = 0.;
      for (auto& ref_polygon_piece : ref_polygon)
        area += boost::geometry::area(ref_polygon_piece);

      if (area > 1e-6) {
        auto const& new_tar_detids_to_be_considered =
            det_geom.getEndcapLayerDetIds(1, etaphibins.first, etaphibins.second);

        for (unsigned int tar_detid : new_tar_detids_to_be_considered) {
          auto& sensor_target = sensors.at(tar_detid);
          double tarphi = std::atan2(sensor_target.centerY, sensor_target.centerX);

          if (std::fabs(phi_mpi_pi(tarphi - refphi)) > std::numbers::pi_v<double> / 2.)
            continue;

          Polygon tar_polygon = getEtaPhiPolygon(det_geom.getCorners(tar_detid), refphi, zshift);

          bool intersects = false;
          for (auto& ref_polygon_piece : ref_polygon) {
            if (boost::geometry::intersects(ref_polygon_piece, tar_polygon)) {
              intersects = true;
              break;
            }
          }

          if (intersects)
            barrel_endcap_connected_tar_detids.insert(tar_detid);
        }
      }

      list_of_detids_etaphi_layer_tar.insert(list_of_detids_etaphi_layer_tar.end(),
                                             barrel_endcap_connected_tar_detids.begin(),
                                             barrel_endcap_connected_tar_detids.end());
    }

    return list_of_detids_etaphi_layer_tar;
  }

  ModuleMap mergeLineConnections(
      std::initializer_list<const std::unordered_map<unsigned int, std::vector<unsigned int>>*> connections_list) {
    ModuleMap merged;

    for (auto* connections : connections_list) {
      for (const auto& [detid, list] : *connections) {
        auto& target = merged[detid];
        target.insert(list.begin(), list.end());
      }
    }

    return merged;
  }

  ModuleMap buildModuleMap(Modules const& modules,
                           Sensors const& sensors,
                           DetectorGeometry const& det_geo,
                           float pt_cut) {
    auto detids_etaphi_layer_ref = det_geo.getDetIds([&modules, &sensors](const auto& x) {
      auto& s = sensors.at(x.first);
      auto& m = modules.at(s.moduleDetId);
      // exclude the outermost modules that do not have connections to other modules
      return ((m.location == Location::barrel && s.isLower && m.layer != 6) ||
              (m.location == Location::endcap && s.isLower && m.layer != 5 && !(m.ring == 15 && m.layer == 1) &&
               !(m.ring == 15 && m.layer == 2) && !(m.ring == 12 && m.layer == 3) && !(m.ring == 12 && m.layer == 4)));
    });

    std::unordered_map<unsigned int, std::vector<unsigned int>> straight_line_connections;
    std::unordered_map<unsigned int, std::vector<unsigned int>> curved_line_connections;

    for (auto ref_detid : detids_etaphi_layer_ref) {
      straight_line_connections[ref_detid] = getStraightLineConnections(ref_detid, modules, sensors, det_geo);
      curved_line_connections[ref_detid] = getCurvedLineConnections(ref_detid, modules, sensors, det_geo, pt_cut);
    }
    return mergeLineConnections({&straight_line_connections, &curved_line_connections});
  }

}  // namespace lstgeometry
