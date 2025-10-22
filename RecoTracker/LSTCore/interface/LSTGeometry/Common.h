#ifndef RecoTracker_LSTCore_interface_LSTGeometry_Common_h
#define RecoTracker_LSTCore_interface_LSTGeometry_Common_h

#include <Eigen/Dense>
#include <limits>

namespace lstgeometry {

  using MatrixD3x3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
  using MatrixD4x2 = Eigen::Matrix<double, 4, 2, Eigen::RowMajor>;
  using MatrixD4x3 = Eigen::Matrix<double, 4, 3, Eigen::RowMajor>;
  using MatrixD8x3 = Eigen::Matrix<double, 8, 3, Eigen::RowMajor>;
  using MatrixDNx2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
  using MatrixDNx3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using RowVectorD2 = Eigen::Matrix<double, 1, 2>;
  using ColVectorD3 = Eigen::Matrix<double, 3, 1>;
  using RowVectorD3 = Eigen::Matrix<double, 1, 3>;

  // TODO: These should be moved to ../Common.h
  constexpr double kPtThreshold = 0.8;
  constexpr double k2Rinv1GeVf = 0.00299792458;
  constexpr double kB = 3.8112;
  
  // For pixel maps
  constexpr unsigned int kNEta = 25;
  constexpr unsigned int kNPhi = 72;
  constexpr unsigned int kNZ = 25;
  constexpr std::array<double, 3> kPtBounds = {{kPtThreshold, 2.0, 10'000.0}};

  // This is defined as a constant in case the legacy value (123456789) needs to be used
  double kDefaultSlope = std::numeric_limits<double>::infinity();

  double degToRad(double degrees) { return degrees * (std::numbers::pi_v<double> / 180); }

  double phi_mpi_pi(double phi) {
    while (phi > std::numbers::pi_v<double>)
      phi -= 2 * std::numbers::pi_v<double>;
    while (phi < -std::numbers::pi_v<double>)
      phi += 2 * std::numbers::pi_v<double>;
    return phi;
  }

}  // namespace lstgeometry

#endif