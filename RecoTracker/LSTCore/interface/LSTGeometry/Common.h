#ifndef RecoTracker_LSTCore_interface_LSTGeometry_Common_h
#define RecoTracker_LSTCore_interface_LSTGeometry_Common_h

#include <Eigen/Dense>

namespace lst {

  using MatrixD3x3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
  using MatrixD4x3 = Eigen::Matrix<double, 4, 3, Eigen::RowMajor>;
  using MatrixD8x3 = Eigen::Matrix<double, 8, 3, Eigen::RowMajor>;
  using ColVectorD3 = Eigen::Matrix<double, 3, 1>;
  using RowVectorD3 = Eigen::Matrix<double, 1, 3>;

}  // namespace lst

#endif