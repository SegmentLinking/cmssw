#ifndef RecoTracker_LSTGeometry_interface_Math_h
#define RecoTracker_LSTGeometry_interface_Math_h

#include <cmath>

#include <boost/math/tools/minima.hpp>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Helix.h"

namespace lstgeometry {

  // Clarification : phi was derived assuming a negatively charged particle would start
  // at the first quadrant. However the way signs are set up in the get_track_point function
  // implies the particle actually starts out in the fourth quadrant, and phi is measured from
  // the y axis as opposed to x axis in the expression provided in this function. Hence I tucked
  // in an extra pi/2 to account for these effects
  Helix::Helix(float pt, float vx, float vy, float vz, float mx, float my, float mz, int charge) : charge(charge) {
    radius = pt / (k2Rinv1GeVf * kB);

    float t = 2. * std::asin(std::sqrt((vx - mx) * (vx - mx) + (vy - my) * (vy - my)) / (2. * radius));
    phi = std::numbers::pi_v<float> / 2. + std::atan((vy - my) / (vx - mx)) +
          ((vy - my) / (vx - mx) < 0) * (std::numbers::pi_v<float>)+charge * t / 2. +
          (my - vy < 0) * (std::numbers::pi_v<float> / 2.) - (my - vy > 0) * (std::numbers::pi_v<float> / 2.);

    center_x = vx + charge * radius * std::sin(phi);
    center_y = vy - charge * radius * std::cos(phi);
    center_z = vz;
    lambda = std::atan((mz - vz) / (radius * t));
  }

  std::tuple<float, float, float, float> Helix::pointFromRadius(float target_r) {
    auto objective_function = [this, target_r](float t) {
      float x = this->center_x - this->charge * this->radius * std::sin(this->phi - this->charge * t);
      float y = this->center_y + this->charge * this->radius * std::cos(this->phi - this->charge * t);
      return std::fabs(std::sqrt(x * x + y * y) - target_r);
    };
    int bits = std::numeric_limits<float>::digits;
    auto result = boost::math::tools::brent_find_minima(objective_function, 0.0f, std::numbers::pi_v<float>, bits);
    float t = result.first;

    float x = center_x - charge * radius * std::sin(phi - charge * t);
    float y = center_y + charge * radius * std::cos(phi - charge * t);
    float z = center_z + radius * std::tan(lambda) * t;
    float r = std::sqrt(x * x + y * y);

    return std::make_tuple(x, y, z, r);
  }

  std::tuple<float, float, float, float> Helix::pointFromZ(float target_z) {
    auto objective_function = [this, target_z](float t) {
      float z = this->center_z + this->radius * std::tan(this->lambda) * t;
      return std::fabs(z - target_z);
    };
    int bits = std::numeric_limits<float>::digits;
    auto result = boost::math::tools::brent_find_minima(objective_function, 0.0f, std::numbers::pi_v<float>, bits);
    float t = result.first;

    float x = center_x - charge * radius * std::sin(phi - charge * t);
    float y = center_y + charge * radius * std::cos(phi - charge * t);
    float z = center_z + radius * std::tan(lambda) * t;
    float r = std::sqrt(x * x + y * y);

    return std::make_tuple(x, y, z, r);
  }

}  // namespace lstgeometry
#endif
