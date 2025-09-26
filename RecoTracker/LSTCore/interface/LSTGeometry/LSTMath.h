#ifndef RecoTracker_LSTCore_interface_LSTGeometry_LSTMath_h
#define RecoTracker_LSTCore_interface_LSTGeometry_LSTMath_h

#include <cmath>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"

namespace lstgeometry {

  std::pair<double, double> getEtaPhi(double x, double y, double z, double refphi = 0) {
    if (refphi != 0) {
      double xnew = x * std::cos(-refphi) - y * std::sin(-refphi);
      double ynew = x * std::sin(-refphi) + y * std::cos(-refphi);
      x = xnew;
      y = ynew;
    }
    double phi = std::atan2(y, x);
    double eta = std::copysign(-std::log(std::tan(std::atan(std::sqrt(x * x + y * y) / std::abs(z)) / 2.)), z);
    return std::make_pair(eta, phi);
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
      auto ref_etaphi = getEtaPhi(ref_mod_boundaries(i, 1), ref_mod_boundaries(i, 2), ref_mod_boundaries(i, 0), refphi);
      auto tar_etaphi = getEtaPhi(tar_mod_boundaries(i, 1), tar_mod_boundaries(i, 2), tar_mod_boundaries(i, 0), refphi);
      ref_mod_boundaries_etaphi(i, 0) = ref_etaphi.first;
      ref_mod_boundaries_etaphi(i, 1) = ref_etaphi.second;
      tar_mod_boundaries_etaphi(i, 0) = tar_etaphi.first;
      tar_mod_boundaries_etaphi(i, 1) = tar_etaphi.second;
    }

    // Quick cut
    RowVectorD2 diff = ref_mod_boundaries_etaphi.row(0) - ref_mod_boundaries_etaphi.row(0);
    if (std::fabs(diff(0)) > 0.5)
      return false;
    if (std::fabs(phi_mpi_pi(diff(1))) > 1.)
      return false;

    // TODO: It might be easy enough to implement this without Boost polygon
    using Point = boost::geometry::model::d2::point_xy<double>;
    using Polygon = boost::geometry::model::polygon<Point>;

    Polygon ref_poly, tar_poly;

    // <= 4 because we need to close the polygon with the first point
    for (int i = 0; i <= 4; ++i) {
      boost::geometry::append(ref_poly,
                              Point(ref_mod_boundaries_etaphi(i % 4, 0), ref_mod_boundaries_etaphi(i % 4, 1)));
      boost::geometry::append(tar_poly,
                              Point(tar_mod_boundaries_etaphi(i % 4, 0), tar_mod_boundaries_etaphi(i % 4, 1)));
    }

    return boost::geometry::intersects(ref_poly, tar_poly);
  }

}  // namespace lstgeometry
#endif