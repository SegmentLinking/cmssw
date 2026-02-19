#ifndef RecoTracker_LSTGeometry_interface_ModuleMapMethods_h
#define RecoTracker_LSTGeometry_interface_ModuleMapMethods_h

#include <cmath>
#include <cassert>
#include <iterator>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <initializer_list>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "RecoTracker/LSTGeometry/interface/Math.h"
#include "RecoTracker/LSTGeometry/interface/SensorCentroid.h"
#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/DetectorGeometry.h"

namespace lstgeometry {

  std::vector<unsigned int> getStraightLineConnections(
      unsigned int ref_detid,
      std::unordered_map<unsigned int, SensorCentroid> const& sensor_centroids,
      DetectorGeometry const& det_geom) {
    auto centroid = sensor_centroids.at(ref_detid);

    double refphi = std::atan2(centroid.y, centroid.x);

    Module refmodule(ref_detid);

    unsigned short ref_layer = refmodule.layer();
    unsigned short ref_subdet = refmodule.subdet();

    auto etaphi = getEtaPhi(centroid.x, centroid.y, centroid.z);
    auto etaphibins = getEtaPhiBins(etaphi.first, etaphi.second);

    auto const& tar_detids_to_be_considered =
        ref_subdet == 5 ? det_geom.getBarrelLayerDetIds(ref_layer + 1, etaphibins.first, etaphibins.second)
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
    if (ref_subdet == 5) {
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
          auto centroid_target = sensor_centroids.at(tar_detid);
          double tarphi = std::atan2(centroid_target.y, centroid_target.x);

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

  MatrixD4x3 boundsAfterCurved(unsigned int ref_detid,
                               std::unordered_map<unsigned int, SensorCentroid> const& sensor_centroids,
                               DetectorGeometry const& det_geom,
                               double ptCut,
                               bool doR = true) {
    auto bounds = det_geom.getCorners(ref_detid);
    auto centroid = sensor_centroids.at(ref_detid);
    int charge = 1;
    double theta = std::atan2(std::sqrt(centroid.x * centroid.x + centroid.y * centroid.y), centroid.z);
    double refphi = std::atan2(centroid.y, centroid.x);
    Module refmodule(ref_detid);
    unsigned short ref_layer = refmodule.layer();
    unsigned short ref_subdet = refmodule.subdet();
    MatrixD4x3 next_layer_bound_points;

    for (int i = 0; i < bounds.rows(); i++) {
      Helix helix_p10 = constructHelixFromPoints(ptCut, 0, 0, 10, bounds(i, 1), bounds(i, 2), bounds(i, 0), -charge);
      Helix helix_m10 = constructHelixFromPoints(ptCut, 0, 0, -10, bounds(i, 1), bounds(i, 2), bounds(i, 0), -charge);
      Helix helix_p10_pos = constructHelixFromPoints(ptCut, 0, 0, 10, bounds(i, 1), bounds(i, 2), bounds(i, 0), charge);
      Helix helix_m10_pos =
          constructHelixFromPoints(ptCut, 0, 0, -10, bounds(i, 1), bounds(i, 2), bounds(i, 0), charge);
      double bound_theta =
          std::atan2(std::sqrt(bounds(i, 1) * bounds(i, 1) + bounds(i, 2) * bounds(i, 2)), bounds(i, 0));
      double bound_phi = std::atan2(bounds(i, 2), bounds(i, 1));
      double phi_diff = phi_mpi_pi(bound_phi - refphi);

      std::tuple<double, double, double, double> next_point;
      if (ref_subdet == 5) {
        if (doR) {
          double tar_layer_radius = det_geom.getBarrelLayerAverageRadius(ref_layer + 1);
          if (bound_theta > theta) {
            next_point = getHelixPointFromRadius(phi_diff > 0 ? helix_p10 : helix_p10_pos, tar_layer_radius);
          } else {
            next_point = getHelixPointFromRadius(phi_diff > 0 ? helix_m10 : helix_m10_pos, tar_layer_radius);
          }
        } else {
          double tar_layer_z = det_geom.getEndcapLayerAverageAbsZ(1);
          if (bound_theta > theta) {
            if (phi_diff > 0) {
              next_point = getHelixPointFromZ(helix_p10, std::copysign(tar_layer_z, helix_p10.lam));
            } else {
              next_point = getHelixPointFromZ(helix_p10_pos, std::copysign(tar_layer_z, helix_p10_pos.lam));
            }
          } else {
            if (phi_diff > 0) {
              next_point = getHelixPointFromZ(helix_m10, std::copysign(tar_layer_z, helix_p10.lam));
            } else {
              next_point = getHelixPointFromZ(helix_m10_pos, std::copysign(tar_layer_z, helix_p10_pos.lam));
            }
          }
        }
      } else {
        double tar_layer_z = det_geom.getEndcapLayerAverageAbsZ(ref_layer + 1);
        if (bound_theta > theta) {
          if (phi_diff > 0) {
            next_point = getHelixPointFromZ(helix_p10, std::copysign(tar_layer_z, helix_p10.lam));
          } else {
            next_point = getHelixPointFromZ(helix_p10_pos, std::copysign(tar_layer_z, helix_p10_pos.lam));
          }
        } else {
          if (phi_diff > 0) {
            next_point = getHelixPointFromZ(helix_m10, std::copysign(tar_layer_z, helix_m10.lam));
          } else {
            next_point = getHelixPointFromZ(helix_m10_pos, std::copysign(tar_layer_z, helix_m10_pos.lam));
          }
        }
      }
      next_layer_bound_points(i, 0) = std::get<2>(next_point);
      next_layer_bound_points(i, 1) = std::get<0>(next_point);
      next_layer_bound_points(i, 2) = std::get<1>(next_point);
    }

    return next_layer_bound_points;
  }

  std::vector<unsigned int> getCurvedLineConnections(
      unsigned int ref_detid,
      std::unordered_map<unsigned int, SensorCentroid> const& sensor_centroids,
      DetectorGeometry const& det_geom,
      double ptCut) {
    auto centroid = sensor_centroids.at(ref_detid);

    double refphi = std::atan2(centroid.y, centroid.x);

    Module refmodule(ref_detid);

    unsigned short ref_layer = refmodule.layer();
    unsigned short ref_subdet = refmodule.subdet();

    auto etaphi = getEtaPhi(centroid.x, centroid.y, centroid.z);
    auto etaphibins = getEtaPhiBins(etaphi.first, etaphi.second);

    auto const& tar_detids_to_be_considered =
        ref_subdet == 5 ? det_geom.getBarrelLayerDetIds(ref_layer + 1, etaphibins.first, etaphibins.second)
                        : det_geom.getEndcapLayerDetIds(ref_layer + 1, etaphibins.first, etaphibins.second);

    auto next_layer_bound_points = boundsAfterCurved(ref_detid, sensor_centroids, det_geom, ptCut);

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    for (unsigned int tar_detid : tar_detids_to_be_considered) {
      if (moduleOverlapsInEtaPhi(next_layer_bound_points, det_geom.getCorners(tar_detid), refphi, 0))
        list_of_detids_etaphi_layer_tar.push_back(tar_detid);
    }

    // Consider barrel to endcap connections if the intersection area is > 0
    // We construct the reference polygon as a vector of polygons because the boost::geometry::difference
    // function can return multiple polygons if the difference results in disjoint pieces
    if (ref_subdet == 5) {
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
          auto centroid_target = sensor_centroids.at(tar_detid);
          double tarphi = std::atan2(centroid_target.y, centroid_target.x);

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

  std::unordered_map<unsigned int, std::unordered_set<unsigned int>> mergeLineConnections(
      std::initializer_list<const std::unordered_map<unsigned int, std::vector<unsigned int>>*> connections_list) {
    std::unordered_map<unsigned int, std::unordered_set<unsigned int>> merged;

    for (auto* connections : connections_list) {
      for (const auto& [detid, list] : *connections) {
        auto& target = merged[detid];
        target.insert(list.begin(), list.end());
      }
    }

    return merged;
  }

}  // namespace lstgeometry

#endif
