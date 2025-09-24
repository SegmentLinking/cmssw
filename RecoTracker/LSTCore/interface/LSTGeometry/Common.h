#ifndef RecoTracker_LSTCore_interface_LSTGeometry_Common_h
#define RecoTracker_LSTCore_interface_LSTGeometry_Common_h

#include <Eigen/Dense>
#include <limits>

namespace lst {

  using MatrixD3x3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
  using MatrixD4x3 = Eigen::Matrix<double, 4, 3, Eigen::RowMajor>;
  using MatrixD8x3 = Eigen::Matrix<double, 8, 3, Eigen::RowMajor>;
  using ColVectorD3 = Eigen::Matrix<double, 3, 1>;
  using RowVectorD3 = Eigen::Matrix<double, 1, 3>;

  constexpr double kPtThreshold = 0.8;

  // This is defined as a constant in case the legacy value (123456789) needs to be used
  double kDefaultSlope = std::numeric_limits<double>::infinity();

  double degToRad(double degrees) { return degrees * (std::numbers::pi_v<double> / 180); }

}  // namespace lst

#endif