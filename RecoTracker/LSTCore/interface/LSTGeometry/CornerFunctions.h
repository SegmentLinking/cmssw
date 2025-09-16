#ifndef RecoTracker_LSTCore_interface_LSTGeometry_h
#define RecoTracker_LSTCore_interface_LSTGeometry_h

#include <numbers>
#include <cmath>

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/ModuleInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/SensorInfo.h"

namespace lst {

  double degToRad(double degrees) { return degrees * (std::numbers::pi_v<double> / 180); }

  MatrixD3x3 rodriguesRotationMatrix(Eigen::Matrix<double, 3, 1> axis, double theta) {
    axis.normalize();

    MatrixD3x3 k{{0, -axis(2), axis(1)}, {axis(2), 0, -axis(0)}, {-axis(1), axis(0), 0}};

    MatrixD3x3 rotationMatrix = MatrixD3x3::Identity() + sin(theta) * k + (1 - cos(theta)) * (k * k);

    return rotationMatrix;
  }

  MatrixD3x3 tangentialRotationMatrix(double phi, double theta) {
    ColVectorD3 axis;
    axis << -sin(phi), cos(phi), 0;

    return rodriguesRotationMatrix(axis, theta);
  }

  MatrixD3x3 rotationMatrix(double tilt_deg, double skew_deg, double yaw_deg, double phi_deg) {
    if (skew_deg != 0 || yaw_deg != 0)
      throw std::invalid_argument("Skew and yaw angles are not currently supported.");
    double tilt_rad = degToRad(tilt_deg);
    double phi_rad = degToRad(phi_deg);

    // Rotation around Z-axis that makes the sensor "face towards" the beamline (i.e. towards z-axis)
    // So for example if phi=0 then R is the identity (i.e. already facing), or if phi=90deg
    // then R becomes (x,y,z)->(-y,x,z) so the sensor is rotated 90 degrees to face the beamline
    MatrixD3x3 initialR{{cos(phi_rad), -sin(phi_rad), 0}, {sin(phi_rad), cos(phi_rad), 0}, {0, 0, 1}};

    // The tilt angle given in the CSV files is with respect to a module that is facing
    // the beamline, meaning after R_initial is applied. From there we tilt the module according
    // to the rotation below. Note that because this tilt angle is not with respect to the X,Y,Z
    // axes and is instead around an arbitrary axis (defined from the rotation above) we have to apply
    // the Rodrigues' rotation formula
    MatrixD3x3 rTilt = tangentialRotationMatrix(phi_rad, -tilt_rad);

    MatrixD3x3 finalR = rTilt * initialR;

    return finalR;
  }

  void transformSensorCorners(ModuleInfo &moduleInfo) {
    auto module_z = moduleInfo.sensorCenterZ_mm;
    auto module_rho = moduleInfo.sensorCenterRho_mm;
    auto module_phi = moduleInfo.phi_deg;
    auto sensor_spacing = moduleInfo.sensorSpacing_mm;
    auto sensor_width = moduleInfo.meanWidth_mm;
    auto sensor_length = moduleInfo.length_mm;

    auto phi_rad = degToRad(module_phi);
    auto module_x = module_rho * cos(phi_rad);
    auto module_y = module_rho * sin(phi_rad);

    auto half_width = sensor_width / 2;
    auto half_length = sensor_length / 2;
    auto half_spacing = sensor_spacing / 2;

    // Make the module sizes consistent with hit-based method.
    // FIXME: Using the real (smaller) sizes specified by CSV file increases
    // fake rate significantly and lowers efficiency between abs(eta) 1 to 2.
    auto width_extension = 50.0 - half_width;
    auto length_extension = (half_length > 40 ? 50.0 : 25.0) - half_length;

    half_width += width_extension;
    half_length += length_extension;

    MatrixD8x3 corners{{-half_spacing, -half_width, -half_length},
                       {-half_spacing, -half_width, half_length},
                       {-half_spacing, half_width, half_length},
                       {-half_spacing, half_width, -half_length},
                       {half_spacing, -half_width, -half_length},
                       {half_spacing, -half_width, half_length},
                       {half_spacing, half_width, half_length},
                       {half_spacing, half_width, -half_length}};

    MatrixD3x3 rotation_matrix =
        rotationMatrix(moduleInfo.tiltAngle_deg, moduleInfo.skewAngle_deg, moduleInfo.yawAngle_deg, moduleInfo.phi_deg);
    MatrixD8x3 rotated_corners = (rotation_matrix * corners.transpose()).transpose();

    rotated_corners.rowwise() += RowVectorD3{module_x, module_y, module_z};

    rotated_corners /= 10;

    // Coordinate reorder before saving (x,y,z)->(z,x,y)
    moduleInfo.transformedCorners.col(0) = rotated_corners.col(2);
    moduleInfo.transformedCorners.col(1) = rotated_corners.col(0);
    moduleInfo.transformedCorners.col(2) = rotated_corners.col(1);
  }

}  // namespace lst

#endif