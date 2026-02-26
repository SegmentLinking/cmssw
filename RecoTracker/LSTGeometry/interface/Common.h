#ifndef RecoTracker_LSTGeometry_interface_Common_h
#define RecoTracker_LSTGeometry_interface_Common_h

#include <limits>
#include <array>
#include <Eigen/Dense>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"

namespace lstgeometry {

  using MatrixF3x3 = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
  using MatrixF4x2 = Eigen::Matrix<float, 4, 2, Eigen::RowMajor>;
  using MatrixF4x3 = Eigen::Matrix<float, 4, 3, Eigen::RowMajor>;
  using MatrixF8x3 = Eigen::Matrix<float, 8, 3, Eigen::RowMajor>;
  using MatrixFNx2 = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
  using MatrixFNx3 = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using RowVectorF2 = Eigen::Matrix<float, 1, 2>;
  using ColVectorF3 = Eigen::Matrix<float, 3, 1>;
  using RowVectorF3 = Eigen::Matrix<float, 1, 3>;

  using ModuleType = TrackerGeometry::ModuleType;
  using SubDetector = GeomDetEnumerators::SubDetector;
  using Location = GeomDetEnumerators::Location;
  using BarrelModuleTilt = Phase2Tracker::BarrelModuleTilt;

  // TODO: These should be moved to ../Common.h
  constexpr float k2Rinv1GeVf = 0.00299792458;
  constexpr float kB = 3.8112;

  // For pixel maps
  constexpr unsigned int kNEta = 25;
  constexpr unsigned int kNPhi = 72;
  constexpr unsigned int kNZ = 25;
  constexpr std::array<float, 2> kPtBounds = {{2.0, 10'000.0}};

  // This is defined as a constant in case the legacy value (123456789) needs to be used
  constexpr float kDefaultSlope = std::numeric_limits<float>::infinity();

  float degToRad(float degrees);
  float phi_mpi_pi(float phi);
  float roundAngle(float angle, float tol = 1e-3);
  float roundCoordinate(float coord, float tol = 1e-3);
  std::pair<float, float> getEtaPhi(float x, float y, float z, float refphi = 0);

}  // namespace lstgeometry

#endif
