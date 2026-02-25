#ifndef RecoTracker_LSTGeometry_interface_CornerMethods_h
#define RecoTracker_LSTGeometry_interface_CornerMethods_h

#include <numbers>
#include <cmath>
#include <unordered_map>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  //Calculates the Rodrigues' rotation matrix for rotating a vector around an arbitrary axis.
  MatrixF3x3 rodriguesRotationMatrix(ColVectorF3 axis, float theta) {
    axis.normalize();

    MatrixF3x3 k{{0, -axis(2), axis(1)}, {axis(2), 0, -axis(0)}, {-axis(1), axis(0), 0}};

    MatrixF3x3 rotationMatrix = MatrixF3x3::Identity() + std::sin(theta) * k + (1 - std::cos(theta)) * (k * k);

    return rotationMatrix;
  }

  // Generates a rotation matrix for rotating around the tangential direction in cylindrical coordinates.
  MatrixF3x3 tangentialRotationMatrix(float phi, float theta) {
    ColVectorF3 axis;
    axis << -std::sin(phi), std::cos(phi), 0;

    return rodriguesRotationMatrix(axis, theta);
  }

  // Computes the final rotation matrix based on tilt and phi angles.
  // Note:
  // Only the tilt angles are non-zero for the current geometry. If the other
  // angles get used, implement their rotations using the tangentialRotationMatrix
  // function above as an example.
  MatrixF3x3 rotationMatrix(float tilt, float skew, float yaw, float phi) {
    if (skew != 0 || yaw != 0)
      throw std::invalid_argument("Skew and yaw angles are not currently supported.");

    // Rotation around Z-axis that makes the sensor "face towards" the beamline (i.e. towards z-axis)
    // So for example if phi=0 then R is the identity (i.e. already facing), or if phi=90deg
    // then R becomes (x,y,z)->(-y,x,z) so the sensor is rotated 90 degrees to face the beamline
    MatrixF3x3 initialR{{std::cos(phi), -std::sin(phi), 0}, {std::sin(phi), std::cos(phi), 0}, {0, 0, 1}};

    // The tilt angle given in the CSV files is with respect to a module that is facing
    // the beamline, meaning after R_initial is applied. From there we tilt the module according
    // to the rotation below. Note that because this tilt angle is not with respect to the X,Y,Z
    // axes and is instead around an arbitrary axis (defined from the rotation above) we have to apply
    // the Rodrigues' rotation formula
    MatrixF3x3 rTilt = tangentialRotationMatrix(phi, -tilt);

    MatrixF3x3 finalR = rTilt * initialR;

    return finalR;
  }

  // Calculates the transformed corners of each sensor
  void transformSensorCorners(Module& module) {
    auto module_z = module.centerZ;
    auto module_rho = module.centerRho;
    auto module_phi = module.centerPhi;
    auto sensor_spacing = module.spacing;
    auto sensor_width = module.meanWidth;
    auto sensor_length = module.length;

    auto module_x = module_rho * std::cos(module_phi);
    auto module_y = module_rho * std::sin(module_phi);

    auto half_width = sensor_width / 2;
    auto half_length = sensor_length / 2;
    auto half_spacing = sensor_spacing / 2;

    // Make the module sizes consistent with hit-based method.
    // FIXME: Using the real (smaller) sizes specified by CSV file increases
    // fake rate significantly and lowers efficiency between abs(eta) 1 to 2.
    auto width_extension = 5.0 - half_width;
    auto length_extension = (half_length > 4 ? 5.0 : 2.5) - half_length;

    half_width += width_extension;
    half_length += length_extension;

    MatrixF8x3 corners{{-half_spacing, -half_width, -half_length},
                       {-half_spacing, -half_width, half_length},
                       {-half_spacing, half_width, half_length},
                       {-half_spacing, half_width, -half_length},
                       {half_spacing, -half_width, -half_length},
                       {half_spacing, -half_width, half_length},
                       {half_spacing, half_width, half_length},
                       {half_spacing, half_width, -half_length}};

    MatrixF3x3 rotation_matrix = rotationMatrix(module.tiltAngle, module.skewAngle, module.yawAngle, module.centerPhi);
    MatrixF8x3 rotated_corners = (rotation_matrix * corners.transpose()).transpose();

    rotated_corners.rowwise() += RowVectorF3{module_x, module_y, module_z};

    // Coordinate reorder before saving (x,y,z)->(z,x,y)
    module.transformedCorners.col(0) = rotated_corners.col(2);
    module.transformedCorners.col(1) = rotated_corners.col(0);
    module.transformedCorners.col(2) = rotated_corners.col(1);
  }

  // Assigns each set of four corners to the correct sensor DetID based on the closest centroid.
  void assignCornersToSensors(Modules const& modules, Sensors& sensors) {
    for (auto const& [detId, module] : modules) {
      unsigned int module_det_id = detId;
      unsigned int sensor_det_id_1 = module_det_id + 1;
      unsigned int sensor_det_id_2 = module_det_id + 2;

      auto& transformed_corners = module.transformedCorners;
      RowVectorF3 centroid_sensor_1 = transformed_corners.topRows(4).colwise().mean();
      RowVectorF3 centroid_sensor_2 = transformed_corners.bottomRows(4).colwise().mean();

      float sensor1_center_z = sensors.at(sensor_det_id_1).centerZ;
      float sensor1_center_x = sensors.at(sensor_det_id_1).centerRho * cos(sensors.at(sensor_det_id_1).centerPhi);
      float sensor1_center_y = sensors.at(sensor_det_id_1).centerRho * sin(sensors.at(sensor_det_id_1).centerPhi);
      float sensor2_center_z = sensors.at(sensor_det_id_2).centerZ;
      float sensor2_center_x = sensors.at(sensor_det_id_2).centerRho * cos(sensors.at(sensor_det_id_2).centerPhi);
      float sensor2_center_y = sensors.at(sensor_det_id_2).centerRho * sin(sensors.at(sensor_det_id_2).centerPhi);

      RowVectorF3 sensor_centroid_1{sensor1_center_z, sensor1_center_x, sensor1_center_y};
      RowVectorF3 sensor_centroid_2{sensor2_center_z, sensor2_center_x, sensor2_center_y};

      float distance_to_sensor_1 = (centroid_sensor_1 - sensor_centroid_1).norm();
      float distance_to_sensor_2 = (centroid_sensor_2 - sensor_centroid_2).norm();

      if (distance_to_sensor_1 < distance_to_sensor_2) {
        sensors[sensor_det_id_1].corners = transformed_corners.topRows(4);
        sensors[sensor_det_id_2].corners = transformed_corners.bottomRows(4);
      } else {
        sensors[sensor_det_id_2].corners = transformed_corners.topRows(4);
        sensors[sensor_det_id_1].corners = transformed_corners.bottomRows(4);
      }
    }
  }

}  // namespace lstgeometry

#endif
