#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <vector>
#include <tuple>
#include <initializer_list>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/LSTGeometry/interface/Helix.h"
#include "RecoTracker/LSTGeometry/interface/ModuleMap.h"

namespace lstgeometry {

  using Point = boost::geometry::model::d2::point_xy<float>;
  using Polygon = boost::geometry::model::polygon<Point>;

  struct EtaPhiBounds {
    float minEta = std::numeric_limits<float>::max();
    float maxEta = std::numeric_limits<float>::lowest();
    float minPhi = std::numeric_limits<float>::max();
    float maxPhi = std::numeric_limits<float>::lowest();
  };

  struct EtaPhiPolygon {
    MatrixF4x2 corners;
    EtaPhiBounds bounds;
    Polygon polygon;
  };

  struct ModuleMapTiming {
    double straightOverlapMs = 0.;
    double straightSubtractMs = 0.;
    double straightEndcapMs = 0.;
    double curvedOverlapMs = 0.;
    double curvedSubtractMs = 0.;
    double curvedEndcapMs = 0.;
  };

  MatrixF4x2 getEtaPhiCorners(MatrixF4x3 const& corners, float refphi, float zshift = 0) {
    MatrixF4x2 corners_etaphi;
    for (int i = 0; i < 4; ++i) {
      auto ref_etaphi = getEtaPhi(corners(i, 1), corners(i, 2), corners(i, 0) + zshift, refphi);
      corners_etaphi(i, 0) = ref_etaphi.first;
      corners_etaphi(i, 1) = ref_etaphi.second;
    }
    return corners_etaphi;
  }

  EtaPhiBounds getEtaPhiBounds(MatrixF4x2 const& corners_etaphi) {
    EtaPhiBounds bounds;
    for (int i = 0; i < 4; ++i) {
      bounds.minEta = std::min(bounds.minEta, corners_etaphi(i, 0));
      bounds.maxEta = std::max(bounds.maxEta, corners_etaphi(i, 0));
      bounds.minPhi = std::min(bounds.minPhi, corners_etaphi(i, 1));
      bounds.maxPhi = std::max(bounds.maxPhi, corners_etaphi(i, 1));
    }
    return bounds;
  }

  bool etaPhiBoundsOverlap(EtaPhiBounds const& lhs, EtaPhiBounds const& rhs) {
    return lhs.minEta <= rhs.maxEta && lhs.maxEta >= rhs.minEta && lhs.minPhi <= rhs.maxPhi &&
           lhs.maxPhi >= rhs.minPhi;
  }

  Polygon getEtaPhiPolygon(MatrixF4x2 const& corners_etaphi) {
    Polygon poly;
    poly.outer().reserve(5);
    // <= because we need to close the polygon with the first point
    for (int i = 0; i <= 4; ++i) {
      boost::geometry::append(poly, Point(corners_etaphi(i % 4, 0), corners_etaphi(i % 4, 1)));
    }
    boost::geometry::correct(poly);
    return poly;
  }

  Polygon getEtaPhiPolygon(MatrixF4x3 const& corners, float refphi, float zshift = 0) {
    return getEtaPhiPolygon(getEtaPhiCorners(corners, refphi, zshift));
  }

  EtaPhiPolygon getEtaPhiPolygonData(MatrixF4x3 const& corners, float refphi, float zshift = 0) {
    EtaPhiPolygon data;
    data.corners = getEtaPhiCorners(corners, refphi, zshift);
    data.bounds = getEtaPhiBounds(data.corners);
    data.polygon = getEtaPhiPolygon(data.corners);
    return data;
  }

  float getCenterPhi(MatrixF4x3 const& corners) {
    RowVectorF3 center = corners.colwise().sum();
    center /= 4.;
    return std::atan2(center(2), center(1));
  }

  bool moduleOverlapsInEtaPhi(MatrixF4x3 const& ref_mod_boundaries,
                              MatrixF4x3 const& tar_mod_boundaries,
                              float refphi = 0,
                              float zshift = 0,
                              float ref_center_phi = std::numeric_limits<float>::max(),
                              float tar_center_phi = std::numeric_limits<float>::max()) {
    if (ref_center_phi < -std::numbers::pi_v<float> || ref_center_phi > std::numbers::pi_v<float>) {
      RowVectorF3 ref_center = ref_mod_boundaries.colwise().sum();
      ref_center /= 4.;
      ref_center_phi = std::atan2(ref_center(2), ref_center(1));
    }
    if (tar_center_phi < -std::numbers::pi_v<float> || tar_center_phi > std::numbers::pi_v<float>) {
      RowVectorF3 tar_center = tar_mod_boundaries.colwise().sum();
      tar_center /= 4.;
      tar_center_phi = std::atan2(tar_center(2), tar_center(1));
    }

    if (std::fabs(phi_mpi_pi(ref_center_phi - tar_center_phi)) > std::numbers::pi_v<float> / 2.)
      return false;

    MatrixF4x2 ref_mod_boundaries_etaphi = getEtaPhiCorners(ref_mod_boundaries, refphi, zshift);
    MatrixF4x2 tar_mod_boundaries_etaphi = getEtaPhiCorners(tar_mod_boundaries, refphi, zshift);

    // Quick cut
    RowVectorF2 diff = ref_mod_boundaries_etaphi.row(0) - tar_mod_boundaries_etaphi.row(0);
    if (std::fabs(diff(0)) > 0.5)
      return false;
    if (std::fabs(phi_mpi_pi(diff(1))) > 1.)
      return false;

    if (!etaPhiBoundsOverlap(getEtaPhiBounds(ref_mod_boundaries_etaphi), getEtaPhiBounds(tar_mod_boundaries_etaphi)))
      return false;

    return boost::geometry::intersects(getEtaPhiPolygon(ref_mod_boundaries_etaphi),
                                       getEtaPhiPolygon(tar_mod_boundaries_etaphi));
  }

  bool moduleOverlapsInEtaPhi(EtaPhiPolygon const& ref_mod_boundaries_etaphi,
                              MatrixF4x3 const& tar_mod_boundaries,
                              float refphi,
                              float zshift,
                              float ref_center_phi,
                              float tar_center_phi) {
    if (tar_center_phi < -std::numbers::pi_v<float> || tar_center_phi > std::numbers::pi_v<float>) {
      tar_center_phi = getCenterPhi(tar_mod_boundaries);
    }

    if (std::fabs(phi_mpi_pi(ref_center_phi - tar_center_phi)) > std::numbers::pi_v<float> / 2.)
      return false;

    MatrixF4x2 tar_mod_boundaries_etaphi = getEtaPhiCorners(tar_mod_boundaries, refphi, zshift);

    // Quick cut
    RowVectorF2 diff = ref_mod_boundaries_etaphi.corners.row(0) - tar_mod_boundaries_etaphi.row(0);
    if (std::fabs(diff(0)) > 0.5)
      return false;
    if (std::fabs(phi_mpi_pi(diff(1))) > 1.)
      return false;

    if (!etaPhiBoundsOverlap(ref_mod_boundaries_etaphi.bounds, getEtaPhiBounds(tar_mod_boundaries_etaphi)))
      return false;

    return boost::geometry::intersects(ref_mod_boundaries_etaphi.polygon, getEtaPhiPolygon(tar_mod_boundaries_etaphi));
  }

  void appendDifferencePieces(Polygon const& ref_polygon_piece,
                              Polygon const& tar_polygon,
                              std::vector<Polygon>& difference) {
    if (!boost::geometry::intersects(ref_polygon_piece, tar_polygon)) {
      difference.push_back(ref_polygon_piece);
      return;
    }

    std::vector<Polygon> tmp_difference;
    boost::geometry::difference(ref_polygon_piece, tar_polygon, tmp_difference);
    difference.insert(difference.end(), tmp_difference.begin(), tmp_difference.end());
  }

  std::vector<unsigned int> getStraightLineConnections(unsigned int ref_detid,
                                                       Sensors const& sensors,
                                                       BinnedDetIds const& binned_detids,
                                                       ModuleMapTiming* timing = nullptr) {
    auto const& ref_sensor = sensors.at(ref_detid);

    float refphi = std::atan2(ref_sensor.centerY, ref_sensor.centerX);
    unsigned short ref_layer = ref_sensor.extra->layer;
    auto ref_location = ref_sensor.extra->location;

    auto thetaphibins = getThetaPhiBins(ref_sensor.extra->centerTheta, ref_sensor.centerPhi);

    auto const& tar_detids_to_be_considered =
        binned_detids.at(std::make_tuple(ref_location, ref_layer + 1, thetaphibins.first, thetaphibins.second));

    std::array<EtaPhiPolygon, 3> ref_polygons = {getEtaPhiPolygonData(ref_sensor.extra->corners, refphi, 0),
                                                 getEtaPhiPolygonData(ref_sensor.extra->corners, refphi, 10),
                                                 getEtaPhiPolygonData(ref_sensor.extra->corners, refphi, -10)};
    constexpr std::array<float, 3> zshifts = {0.f, 10.f, -10.f};

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    list_of_detids_etaphi_layer_tar.reserve(tar_detids_to_be_considered.size());
    const auto overlapStart = std::chrono::steady_clock::now();
    for (unsigned int tar_detid : tar_detids_to_be_considered) {
      auto const& tar_sensor = sensors.at(tar_detid);
      for (unsigned int i = 0; i < zshifts.size(); ++i) {
        if (moduleOverlapsInEtaPhi(ref_polygons[i],
                                   tar_sensor.extra->corners,
                                   refphi,
                                   zshifts[i],
                                   ref_sensor.centerPhi,
                                   tar_sensor.centerPhi)) {
          list_of_detids_etaphi_layer_tar.push_back(tar_detid);
          break;
        }
      }
    }
    const auto overlapDone = std::chrono::steady_clock::now();
    if (timing)
      timing->straightOverlapMs += std::chrono::duration<double, std::milli>(overlapDone - overlapStart).count();

    // Consider barrel to endcap connections if the intersection area is > 0
    // We construct the reference polygon as a vector of polygons because the boost::geometry::difference
    // function can return multiple polygons if the difference results in disjoint pieces
    if (ref_location == Location::barrel) {
      std::vector<unsigned int> barrel_endcap_connected_tar_detids;

      for (unsigned int i = 0; i < zshifts.size(); ++i) {
        float zshift = zshifts[i];
        std::vector<Polygon> ref_polygon;
        ref_polygon.push_back(ref_polygons[i].polygon);

        // Check whether there is still significant non-zero area
        const auto subtractStart = std::chrono::steady_clock::now();
        for (unsigned int tar_detid : list_of_detids_etaphi_layer_tar) {
          if (ref_polygon.empty())
            break;
          Polygon tar_polygon = getEtaPhiPolygon(sensors.at(tar_detid).extra->corners, refphi, zshift);

          std::vector<Polygon> difference;
          for (auto const& ref_polygon_piece : ref_polygon) {
            appendDifferencePieces(ref_polygon_piece, tar_polygon, difference);
          }

          ref_polygon = std::move(difference);
        }

        float area = 0.;
        for (auto const& ref_polygon_piece : ref_polygon)
          area += boost::geometry::area(ref_polygon_piece);
        const auto subtractDone = std::chrono::steady_clock::now();
        if (timing)
          timing->straightSubtractMs +=
              std::chrono::duration<double, std::milli>(subtractDone - subtractStart).count();

        if (area <= 5e-3)
          continue;

        auto const& new_tar_detids_to_be_considered =
            binned_detids.at(std::make_tuple(Location::endcap, 1, thetaphibins.first, thetaphibins.second));

        const auto endcapStart = std::chrono::steady_clock::now();
        for (unsigned int tar_detid : new_tar_detids_to_be_considered) {
          auto const& sensor_target = sensors.at(tar_detid);
          float tarphi = std::atan2(sensor_target.centerY, sensor_target.centerX);

          if (std::fabs(phi_mpi_pi(tarphi - refphi)) > std::numbers::pi_v<float> / 2.)
            continue;

          Polygon tar_polygon = getEtaPhiPolygon(sensor_target.extra->corners, refphi, zshift);

          bool intersects = false;
          for (auto const& ref_polygon_piece : ref_polygon) {
            if (boost::geometry::intersects(ref_polygon_piece, tar_polygon)) {
              intersects = true;
              break;
            }
          }

          if (intersects)
            barrel_endcap_connected_tar_detids.push_back(tar_detid);
        }
        const auto endcapDone = std::chrono::steady_clock::now();
        if (timing)
          timing->straightEndcapMs += std::chrono::duration<double, std::milli>(endcapDone - endcapStart).count();
      }
      list_of_detids_etaphi_layer_tar.insert(list_of_detids_etaphi_layer_tar.end(),
                                             barrel_endcap_connected_tar_detids.begin(),
                                             barrel_endcap_connected_tar_detids.end());
    }

    return list_of_detids_etaphi_layer_tar;
  }

  MatrixF4x3 boundsAfterCurved(unsigned int ref_detid,
                               Sensors const& sensors,
                               std::array<float, kBarrelLayers> const& average_r_barrel,
                               std::array<float, kEndcapLayers> const& average_z_endcap,
                               float ptCut,
                               bool doR = true) {
    auto const& ref_sensor = sensors.at(ref_detid);
    auto const& bounds = ref_sensor.extra->corners;
    int charge = 1;
    float z_r = ref_sensor.centerZ /
                std::sqrt(ref_sensor.centerX * ref_sensor.centerX + ref_sensor.centerY * ref_sensor.centerY);
    float refphi = std::atan2(ref_sensor.centerY, ref_sensor.centerX);
    unsigned short ref_layer = ref_sensor.extra->layer;
    auto ref_location = ref_sensor.extra->location;
    MatrixF4x3 next_layer_bound_points;

    for (int i = 0; i < bounds.rows(); i++) {
      auto helix_p10 = Helix(ptCut, 0, 0, 10, bounds(i, 1), bounds(i, 2), bounds(i, 0), -charge);
      auto helix_m10 = Helix(ptCut, 0, 0, -10, bounds(i, 1), bounds(i, 2), bounds(i, 0), -charge);
      auto helix_p10_pos = Helix(ptCut, 0, 0, 10, bounds(i, 1), bounds(i, 2), bounds(i, 0), charge);
      auto helix_m10_pos = Helix(ptCut, 0, 0, -10, bounds(i, 1), bounds(i, 2), bounds(i, 0), charge);
      float bound_z_r = bounds(i, 0) / std::sqrt(bounds(i, 1) * bounds(i, 1) + bounds(i, 2) * bounds(i, 2));
      float bound_phi = std::atan2(bounds(i, 2), bounds(i, 1));
      float phi_diff = phi_mpi_pi(bound_phi - refphi);

      std::tuple<float, float, float, float> next_point;
      if (ref_location == Location::barrel) {
        if (doR) {
          float tar_layer_radius = average_r_barrel[ref_layer];
          if (bound_z_r < z_r) {
            auto const& h = phi_diff > 0 ? helix_p10 : helix_p10_pos;
            next_point = h.pointFromRadius(tar_layer_radius);
          } else {
            auto const& h = phi_diff > 0 ? helix_m10 : helix_m10_pos;
            next_point = h.pointFromRadius(tar_layer_radius);
          }
        } else {
          float tar_layer_z = average_z_endcap[0];
          if (bound_z_r < z_r) {
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
        float tar_layer_z = average_z_endcap[ref_layer];
        if (bound_z_r < z_r) {
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
                                                     Sensors const& sensors,
                                                     BinnedDetIds const& binned_detids,
                                                     std::array<float, kBarrelLayers> const& average_r_barrel,
                                                     std::array<float, kEndcapLayers> const& average_z_endcap,
                                                     float ptCut,
                                                     ModuleMapTiming* timing = nullptr) {
    auto const& ref_sensor = sensors.at(ref_detid);

    float refphi = std::atan2(ref_sensor.centerY, ref_sensor.centerX);

    unsigned short ref_layer = ref_sensor.extra->layer;
    auto ref_location = ref_sensor.extra->location;

    auto thetaphibins = getThetaPhiBins(ref_sensor.extra->centerTheta, ref_sensor.centerPhi);

    auto const& tar_detids_to_be_considered =
        binned_detids.at({ref_location, ref_layer + 1, thetaphibins.first, thetaphibins.second});

    auto next_layer_bound_points = boundsAfterCurved(ref_detid, sensors, average_r_barrel, average_z_endcap, ptCut);
    EtaPhiPolygon next_layer_polygon = getEtaPhiPolygonData(next_layer_bound_points, refphi);
    float next_layer_center_phi = getCenterPhi(next_layer_bound_points);

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    list_of_detids_etaphi_layer_tar.reserve(tar_detids_to_be_considered.size());
    const auto overlapStart = std::chrono::steady_clock::now();
    for (unsigned int tar_detid : tar_detids_to_be_considered) {
      auto const& tar_sensor = sensors.at(tar_detid);
      if (moduleOverlapsInEtaPhi(next_layer_polygon,
                                 tar_sensor.extra->corners,
                                 refphi,
                                 0,
                                 next_layer_center_phi,
                                 tar_sensor.centerPhi))
        list_of_detids_etaphi_layer_tar.push_back(tar_detid);
    }
    const auto overlapDone = std::chrono::steady_clock::now();
    if (timing)
      timing->curvedOverlapMs += std::chrono::duration<double, std::milli>(overlapDone - overlapStart).count();

    // Consider barrel to endcap connections if the intersection area is > 0
    // We construct the reference polygon as a vector of polygons because the boost::geometry::difference
    // function can return multiple polygons if the difference results in disjoint pieces
    if (ref_location == Location::barrel) {
      std::vector<unsigned int> barrel_endcap_connected_tar_detids;

      int zshift = 0;

      std::vector<Polygon> ref_polygon;
      ref_polygon.push_back(next_layer_polygon.polygon);

      // Check whether there is still significant non-zero area
      const auto subtractStart = std::chrono::steady_clock::now();
      for (unsigned int tar_detid : list_of_detids_etaphi_layer_tar) {
        if (ref_polygon.empty())
          break;
        Polygon tar_polygon = getEtaPhiPolygon(sensors.at(tar_detid).extra->corners, refphi, zshift);

        std::vector<Polygon> difference;
        for (auto const& ref_polygon_piece : ref_polygon) {
          appendDifferencePieces(ref_polygon_piece, tar_polygon, difference);
        }

        ref_polygon = std::move(difference);
      }

      float area = 0.;
      for (auto const& ref_polygon_piece : ref_polygon)
        area += boost::geometry::area(ref_polygon_piece);
      const auto subtractDone = std::chrono::steady_clock::now();
      if (timing)
        timing->curvedSubtractMs += std::chrono::duration<double, std::milli>(subtractDone - subtractStart).count();

      if (area > 5e-3) {
        auto const& new_tar_detids_to_be_considered =
            binned_detids.at({Location::endcap, 1, thetaphibins.first, thetaphibins.second});

        const auto endcapStart = std::chrono::steady_clock::now();
        for (unsigned int tar_detid : new_tar_detids_to_be_considered) {
          auto const& sensor_target = sensors.at(tar_detid);
          float tarphi = std::atan2(sensor_target.centerY, sensor_target.centerX);

          if (std::fabs(phi_mpi_pi(tarphi - refphi)) > std::numbers::pi_v<float> / 2.)
            continue;

          Polygon tar_polygon = getEtaPhiPolygon(sensor_target.extra->corners, refphi, zshift);

          bool intersects = false;
          for (auto const& ref_polygon_piece : ref_polygon) {
            if (boost::geometry::intersects(ref_polygon_piece, tar_polygon)) {
              intersects = true;
              break;
            }
          }

          if (intersects)
            barrel_endcap_connected_tar_detids.push_back(tar_detid);
        }
        const auto endcapDone = std::chrono::steady_clock::now();
        if (timing)
          timing->curvedEndcapMs += std::chrono::duration<double, std::milli>(endcapDone - endcapStart).count();
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
      for (auto const& [detid, list] : *connections) {
        auto& target = merged[detid];
        target.insert(target.end(), list.begin(), list.end());
      }
    }

    for (auto& [detid, list] : merged) {
      std::sort(list.begin(), list.end());
      list.erase(std::unique(list.begin(), list.end()), list.end());
    }

    return merged;
  }

  ModuleMap buildModuleMap(Sensors const& sensors,
                           BinnedDetIds const& binned_detids,
                           std::array<float, kBarrelLayers> const& average_r_barrel,
                           std::array<float, kEndcapLayers> const& average_z_endcap,
                           float pt_cut) {
    std::unordered_map<unsigned int, std::vector<unsigned int>> straight_line_connections;
    std::unordered_map<unsigned int, std::vector<unsigned int>> curved_line_connections;
    straight_line_connections.reserve(sensors.size());
    curved_line_connections.reserve(sensors.size());

    const auto start = std::chrono::steady_clock::now();
    double straightMs = 0.;
    double curvedMs = 0.;
    unsigned int nRefSensors = 0;
    ModuleMapTiming timing;
    for (auto const& [ref_detid, s] : sensors) {
      // exclude the outermost modules that do not have connections to other modules
      if (!((s.extra->location == Location::barrel && s.extra->lower && s.extra->layer != 6) ||
            (s.extra->location == Location::endcap && s.extra->lower && s.extra->layer != 5 &&
             !(s.extra->ring == 15 && s.extra->layer == 1) && !(s.extra->ring == 15 && s.extra->layer == 2) &&
             !(s.extra->ring == 12 && s.extra->layer == 3) && !(s.extra->ring == 12 && s.extra->layer == 4))))
        continue;
      ++nRefSensors;
      const auto straightStart = std::chrono::steady_clock::now();
      straight_line_connections[ref_detid] = getStraightLineConnections(ref_detid, sensors, binned_detids, &timing);
      const auto straightDone = std::chrono::steady_clock::now();
      curved_line_connections[ref_detid] = getCurvedLineConnections(
          ref_detid, sensors, binned_detids, average_r_barrel, average_z_endcap, pt_cut, &timing);
      const auto curvedDone = std::chrono::steady_clock::now();
      straightMs += std::chrono::duration<double, std::milli>(straightDone - straightStart).count();
      curvedMs += std::chrono::duration<double, std::milli>(curvedDone - straightDone).count();
    }
    const auto connectionsDone = std::chrono::steady_clock::now();
    auto moduleMap = mergeLineConnections({&straight_line_connections, &curved_line_connections});
    const auto mergeDone = std::chrono::steady_clock::now();

    edm::LogInfo("LSTGeometryESProducer")
        << "Temporary timing: buildModuleMap refs " << nRefSensors << ", straight " << straightMs << " ms, curved "
        << curvedMs << " ms, loop total "
        << std::chrono::duration<double, std::milli>(connectionsDone - start).count() << " ms, merge "
        << std::chrono::duration<double, std::milli>(mergeDone - connectionsDone).count() << " ms";
    edm::LogInfo("LSTGeometryESProducer")
        << "Temporary timing: buildModuleMap detail straight overlap " << timing.straightOverlapMs << " ms, subtract "
        << timing.straightSubtractMs << " ms, endcap scan " << timing.straightEndcapMs << " ms; curved overlap "
        << timing.curvedOverlapMs << " ms, subtract " << timing.curvedSubtractMs << " ms, endcap scan "
        << timing.curvedEndcapMs << " ms";

    return moduleMap;
  }

}  // namespace lstgeometry
