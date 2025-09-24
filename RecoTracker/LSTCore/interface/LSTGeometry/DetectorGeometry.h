#ifndef RecoTracker_LSTCore_interface_LSTGeometry_DetectorGeometry_h
#define RecoTracker_LSTCore_interface_LSTGeometry_DetectorGeometry_h

#include <vector>

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"

namespace lst {

  class DetectorGeometry {
  private:
    std::unordered_map<unsigned int, MatrixD4x3> corners;
    std::vector<double> avg_radii;
    std::vector<double> avg_z;

  public:
    DetectorGeometry(std::unordered_map<unsigned int, MatrixD4x3> corners_in,
                     std::vector<double> avg_radii_in,
                     std::vector<double> avg_z_in)
        : corners(corners_in), avg_radii(avg_radii_in), avg_z(avg_z_in) {}

    std::vector<unsigned int> getDetIds() const {
      std::vector<unsigned int> detIds;
      for (const auto& entry : corners) {
        detIds.push_back(entry.first);
      }
      return detIds;
    }

    double getMinR(unsigned int detId) const {
      auto const& corners = this->corners.at(detId);
      double minR = std::numeric_limits<double>::max();
      for (int i = 0; i < corners.rows(); i++) {
        double x = corners(i, 1);
        double y = corners(i, 2);
        minR = std::min(minR, std::sqrt(x * x + y * y));
      }
      return minR;
    }

    double getMaxR(unsigned int detId) const {
      auto const& corners = this->corners.at(detId);
      double maxR = std::numeric_limits<double>::min();
      for (int i = 0; i < corners.rows(); i++) {
        double x = corners(i, 1);
        double y = corners(i, 2);
        maxR = std::max(maxR, std::sqrt(x * x + y * y));
      }
      return maxR;
    }
  };
}  // namespace lst

#endif