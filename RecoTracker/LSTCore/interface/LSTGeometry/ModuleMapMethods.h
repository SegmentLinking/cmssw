#ifndef RecoTracker_LSTCore_interface_LSTGeometry_ModuleMapMethods_h
#define RecoTracker_LSTCore_interface_LSTGeometry_ModuleMapMethods_h

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

#include "LSTMath.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/LSTMath.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Centroid.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Module.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/DetectorGeometry.h"

namespace lstgeometry {

  std::vector<unsigned int> getStraightLineConnections(unsigned int ref_detid,
                                                       std::unordered_map<unsigned int, Centroid> const& centroids,
                                                       DetectorGeometry const& det_geom) {
    auto centroid = centroids.at(ref_detid);

    double refphi = std::atan2(centroid.y, centroid.x);

    Module refmodule(ref_detid);

    unsigned short ref_layer = refmodule.layer();
    unsigned short ref_subdet = refmodule.subdet();

    auto const& tar_detids_to_be_considered =
        ref_subdet == 5 ? det_geom.getBarrelLayerDetIds(ref_layer + 1) : det_geom.getEndcapLayerDetIds(ref_layer + 1);

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    for (unsigned int tar_detid : tar_detids_to_be_considered) {
      if (moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, 0) ||
          moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, 10) ||
          moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, -10))
        list_of_detids_etaphi_layer_tar.push_back(tar_detid);
    }

    // Consider barrel to endcap connections if the intersection area is > 0
    if (ref_subdet == 5) {
      std::unordered_set<unsigned int> barrel_endcap_connected_tar_detids;

      for (int zshift : {0, 10, -10}) {
        std::vector<Polygon> ref_polygon;
        ref_polygon.push_back(getEtaPhiPolygon(det_geom.getCorners(ref_detid), refphi, zshift));

        // Check whether there is still significant non-zero area
        for (unsigned int tar_detid : list_of_detids_etaphi_layer_tar) {
          if (!ref_polygon.size())
            break;
          Polygon tar_polygon = getEtaPhiPolygon(det_geom.getCorners(tar_detid), refphi, zshift);

          std::vector<Polygon> difference;
          for (auto &ref_polygon_piece : ref_polygon) {
              std::vector<Polygon> tmp_difference;
              boost::geometry::difference(ref_polygon_piece, tar_polygon, tmp_difference);
              difference.insert(difference.end(), tmp_difference.begin(), tmp_difference.end());
          }

          ref_polygon = std::move(difference);
        }

        double area = 0.;
        for (auto &ref_polygon_piece : ref_polygon)
            area += boost::geometry::area(ref_polygon_piece);
        
        if (area <= 0.0001)
          continue;

        auto const& new_tar_detids_to_be_considered = det_geom.getEndcapLayerDetIds(1);

        for (unsigned int tar_detid : new_tar_detids_to_be_considered) {
          auto centroid_target = centroids.at(tar_detid);
          double tarphi = std::atan2(centroid_target.y, centroid_target.x);

          if (std::fabs(phi_mpi_pi(tarphi - refphi)) > std::numbers::pi_v<double> / 2.)
            continue;

          Polygon tar_polygon = getEtaPhiPolygon(det_geom.getCorners(tar_detid), refphi, zshift);

          bool intersects = false;
          for (auto &ref_polygon_piece : ref_polygon){
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

  std::vector<std::tuple<double, double, double>> boundsAfterCurved(
      unsigned int ref_detid,
      std::unordered_map<unsigned int, Centroid> const& centroids,
      DetectorGeometry const& det_geom,
      bool doR = true) {
    auto bounds = det_geom.getCorners(ref_detid);
    auto centroid = centroids.at(ref_detid);
    int charge = 1;
    double theta =
        std::atan2(std::sqrt(centroid.x * centroid.x + centroid.y * centroid.y), centroid.z);  // TODO: Is this right?
    double refphi = std::atan2(centroid.y, centroid.x);
    Module refmodule(ref_detid);
    unsigned short ref_layer = refmodule.layer();
    unsigned short ref_subdet = refmodule.subdet();
    std::vector<std::tuple<double, double, double>> next_layer_bound_points;

    for (int i = 0; i < bounds.rows(); i++) {
      Helix helix_p10 =
          constructHelixFromPoints(kPtThreshold, 0, 0, 10, bounds(i, 1), bounds(i, 2), bounds(i, 0), -charge);
      Helix helix_m10 =
          constructHelixFromPoints(kPtThreshold, 0, 0, -10, bounds(i, 1), bounds(i, 2), bounds(i, 0), -charge);
      Helix helix_p10_pos =
          constructHelixFromPoints(kPtThreshold, 0, 0, 10, bounds(i, 1), bounds(i, 2), bounds(i, 0), charge);
      Helix helix_m10_pos =
          constructHelixFromPoints(kPtThreshold, 0, 0, -10, bounds(i, 1), bounds(i, 2), bounds(i, 0), charge);
      double bound_theta =
          std::atan2(std::sqrt(bounds(i, 1) * bounds(i, 1) + bounds(i, 2) * bounds(i, 2)), bounds(i, 0));
      double bound_phi = std::atan2(bounds(i, 2), bounds(i, 1));
      double phi_diff = phi_mpi_pi(bound_phi - refphi);

      // TODO: Check if the copysign arguments are correct
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
              next_point = getHelixPointFromZ(helix_p10_pos, std::copysign(tar_layer_z, helix_p10.lam));
            }
          } else {
            if (phi_diff > 0) {
              next_point = getHelixPointFromZ(helix_m10, std::copysign(tar_layer_z, helix_p10.lam));
            } else {
              next_point = getHelixPointFromZ(helix_m10_pos, std::copysign(tar_layer_z, helix_p10.lam));
            }
          }
        }
      } else {
        double tar_layer_z = det_geom.getEndcapLayerAverageAbsZ(ref_layer + 1);
        if (bound_theta > theta) {
          if (phi_diff > 0) {
            next_point = getHelixPointFromZ(helix_p10, std::copysign(tar_layer_z, helix_p10.lam));
          } else {
            next_point = getHelixPointFromZ(helix_p10_pos, std::copysign(tar_layer_z, helix_p10.lam));
          }
        } else {
          if (phi_diff > 0) {
            next_point = getHelixPointFromZ(helix_m10, std::copysign(tar_layer_z, helix_m10.lam));
          } else {
            next_point = getHelixPointFromZ(helix_m10_pos, std::copysign(tar_layer_z, helix_m10.lam));
          }
        }
      }
      next_layer_bound_points.push_back({std::get<2>(next_point), std::get<0>(next_point), std::get<1>(next_point)});
    }

    return next_layer_bound_points;
  }

  std::vector<unsigned int> getCurvedLineConnections(unsigned int ref_detid,
                                                     std::unordered_map<unsigned int, Centroid> const& centroids,
                                                     DetectorGeometry const& det_geom) {
    auto centroid = centroids.at(ref_detid);

    double refphi = std::atan2(centroid.y, centroid.x);

    Module refmodule(ref_detid);

    unsigned short ref_layer = refmodule.layer();
    unsigned short ref_subdet = refmodule.subdet();

    auto const& tar_detids_to_be_considered =
        ref_subdet == 5 ? det_geom.getBarrelLayerDetIds(ref_layer + 1) : det_geom.getEndcapLayerDetIds(ref_layer + 1);

    std::vector<std::tuple<double, double, double>> next_layer_bound_points =
        boundsAfterCurved(ref_detid, centroids, det_geom);
    MatrixDNx3 next_layer_bound_points_matrix(next_layer_bound_points.size(), 3);
    for (size_t i = 0; i < next_layer_bound_points.size(); i++) {
      next_layer_bound_points_matrix(i, 0) = std::get<0>(next_layer_bound_points[i]);
      next_layer_bound_points_matrix(i, 1) = std::get<1>(next_layer_bound_points[i]);
      next_layer_bound_points_matrix(i, 2) = std::get<2>(next_layer_bound_points[i]);
    }

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    for (unsigned int tar_detid : tar_detids_to_be_considered) {
      if (moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, 0))
        list_of_detids_etaphi_layer_tar.push_back(tar_detid);
    }

    // Consider barrel to endcap connections if the intersection area is > 0
    if (ref_subdet == 5) {
      std::unordered_set<unsigned int> barrel_endcap_connected_tar_detids;

      int zshift = 0;

      std::vector<Polygon> ref_polygon;
      ref_polygon.push_back(getEtaPhiPolygon(next_layer_bound_points_matrix, refphi, zshift));

      // Check whether there is still significant non-zero area
      for (unsigned int tar_detid : list_of_detids_etaphi_layer_tar) {
          if (!ref_polygon.size())
          break;
        Polygon tar_polygon = getEtaPhiPolygon(det_geom.getCorners(tar_detid), refphi, zshift);

        std::vector<Polygon> difference;
        for (auto &ref_polygon_piece : ref_polygon) {
            std::vector<Polygon> tmp_difference;
            boost::geometry::difference(ref_polygon_piece, tar_polygon, tmp_difference);
            difference.insert(difference.end(), tmp_difference.begin(), tmp_difference.end());
        }

        ref_polygon = std::move(difference);
      }
      
      double area = 0.;
      for (auto &ref_polygon_piece : ref_polygon)
          area += boost::geometry::area(ref_polygon_piece);

      if (area > 0.0001) {
        auto const& new_tar_detids_to_be_considered = det_geom.getEndcapLayerDetIds(1);

        for (unsigned int tar_detid : new_tar_detids_to_be_considered) {
          auto centroid_target = centroids.at(tar_detid);
          double tarphi = std::atan2(centroid_target.y, centroid_target.x);

          if (std::fabs(phi_mpi_pi(tarphi - refphi)) > std::numbers::pi_v<double> / 2.)
            continue;

          Polygon tar_polygon = getEtaPhiPolygon(det_geom.getCorners(tar_detid), refphi, zshift);

          bool intersects = false;
          for (auto &ref_polygon_piece : ref_polygon){
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
  
  std::unordered_map<unsigned int, std::unordered_set<unsigned int>>
  mergeLineConnections(std::initializer_list<const std::unordered_map<unsigned int, std::vector<unsigned int>>*> connections_list) {
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