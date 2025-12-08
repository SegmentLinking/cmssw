#ifndef RecoTracker_LSTCore_interface_LSTGeometry_LSTMath_h
#define RecoTracker_LSTCore_interface_LSTGeometry_LSTMath_h

#include <cmath>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/math/tools/minima.hpp>

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Helix.h"

namespace lstgeometry {

  using Point = boost::geometry::model::d2::point_xy<double>;
  using Polygon = boost::geometry::model::polygon<Point>;

  // Clarification : phi was derived assuming a negatively charged particle would start
  // at the first quadrant. However the way signs are set up in the get_track_point function
  // implies the particle actually starts out in the fourth quadrant, and phi is measured from
  // the y axis as opposed to x axis in the expression provided in this function. Hence I tucked
  // in an extra pi/2 to account for these effects
  Helix constructHelixFromPoints(
      double pt, double vx, double vy, double vz, double mx, double my, double mz, int charge) {
    double radius = pt / (k2Rinv1GeVf * kB);

    double t = 2. * std::asin(std::sqrt((vx - mx) * (vx - mx) + (vy - my) * (vy - my)) / (2. * radius));
    double phi = std::numbers::pi_v<double> / 2. + std::atan((vy - my) / (vx - mx)) +
                 ((vy - my) / (vx - mx) < 0) * (std::numbers::pi_v<double>)+charge * t / 2. +
                 (my - vy < 0) * (std::numbers::pi_v<double> / 2.) - (my - vy > 0) * (std::numbers::pi_v<double> / 2.);

    double cx = vx + charge * radius * std::sin(phi);
    double cy = vy - charge * radius * std::cos(phi);
    double cz = vz;
    double lam = std::atan((mz - vz) / (radius * t));

    return Helix(cx, cy, cz, radius, phi, lam, charge);
  }

  std::tuple<double, double, double, double> getHelixPointFromRadius(Helix const& helix, double target_r) {
    auto objective_function = [&helix, target_r](double t) {
      double x = helix.center_x - helix.charge * helix.radius * std::sin(helix.phi - helix.charge * t);
      double y = helix.center_y + helix.charge * helix.radius * std::cos(helix.phi - helix.charge * t);
      return std::fabs(std::sqrt(x * x + y * y) - target_r);
    };
    int bits = std::numeric_limits<double>::digits;
    auto result = boost::math::tools::brent_find_minima(objective_function, 0.0, std::numbers::pi_v<double>, bits);
    double t = result.first;

    double x = helix.center_x - helix.charge * helix.radius * std::sin(helix.phi - helix.charge * t);
    double y = helix.center_y + helix.charge * helix.radius * std::cos(helix.phi - helix.charge * t);
    double z = helix.center_z + helix.radius * std::tan(helix.lam) * t;
    double r = std::sqrt(x * x + y * y);

    return std::make_tuple(x, y, z, r);
  }

  std::tuple<double, double, double, double> getHelixPointFromZ(Helix const& helix, double target_z) {
    auto objective_function = [&helix, target_z](double t) {
      double z = helix.center_z + helix.radius * std::tan(helix.lam) * t;
      return std::fabs(z - target_z);
    };
    int bits = std::numeric_limits<double>::digits;
    auto result = boost::math::tools::brent_find_minima(objective_function, 0.0, std::numbers::pi_v<double>, bits);
    double t = result.first;

    double x = helix.center_x - helix.charge * helix.radius * std::sin(helix.phi - helix.charge * t);
    double y = helix.center_y + helix.charge * helix.radius * std::cos(helix.phi - helix.charge * t);
    double z = helix.center_z + helix.radius * std::tan(helix.lam) * t;
    double r = std::sqrt(x * x + y * y);

    return std::make_tuple(x, y, z, r);
  }

  std::pair<double, double> getEtaPhi(double x, double y, double z, double refphi = 0) {
    double phi = phi_mpi_pi(std::atan2(y, x) - refphi);
    double eta = std::copysign(-std::log(std::tan(std::atan(std::sqrt(x * x + y * y) / std::abs(z)) / 2.)), z);
    return std::make_pair(eta, phi);
  }

  Polygon getEtaPhiPolygon(MatrixDNx3 const& mod_boundaries, double refphi, double zshift = 0) {
    int npoints = mod_boundaries.rows();
    MatrixDNx2 mod_boundaries_etaphi(npoints, 2);
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

  bool moduleOverlapsInEtaPhi(MatrixD4x3 const& ref_mod_boundaries,
                              MatrixD4x3 const& tar_mod_boundaries,
                              double refphi = 0,
                              double zshift = 0) {
    RowVectorD3 ref_center = ref_mod_boundaries.colwise().sum();
    ref_center /= 4.;
    RowVectorD3 tar_center = tar_mod_boundaries.colwise().sum();
    tar_center /= 4.;

    double ref_center_phi = std::atan2(ref_center(2), ref_center(1));
    double tar_center_phi = std::atan2(tar_center(2), tar_center(1));

    if (std::fabs(phi_mpi_pi(ref_center_phi - tar_center_phi)) > std::numbers::pi_v<double> / 2.)
      return false;

    MatrixD4x2 ref_mod_boundaries_etaphi;
    MatrixD4x2 tar_mod_boundaries_etaphi;

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
    RowVectorD2 diff = ref_mod_boundaries_etaphi.row(0) - tar_mod_boundaries_etaphi.row(0);
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

}  // namespace lstgeometry
#endif
