#ifndef RecoTracker_LSTCore_interface_LSTGeometry_DetectorGeometry_h
#define RecoTracker_LSTCore_interface_LSTGeometry_DetectorGeometry_h

#include <algorithm>
#include <vector>
#include <functional>

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Module.h"

namespace lstgeometry {

  class DetectorGeometry {
  private:
    std::unordered_map<unsigned int, MatrixD4x3> corners_;
    std::vector<double> avg_radii_;
    std::vector<double> avg_z_;
    std::vector<std::vector<unsigned int>> barrel_lower_det_ids_;
    std::vector<std::vector<unsigned int>> endcap_lower_det_ids_;

  public:
    DetectorGeometry(std::unordered_map<unsigned int, MatrixD4x3> corners,
                     std::vector<double> avg_radii,
                     std::vector<double> avg_z)
        : corners_(corners), avg_radii_(avg_radii), avg_z_(avg_z) {}

    MatrixD4x3 const& getCorners(unsigned int detId) const { return corners_.at(detId); }

    std::vector<unsigned int> getDetIds(std::function<bool(const std::pair<const unsigned int, MatrixD4x3>&)> filter =
                                            [](const auto&) { return true; }) const {
      std::vector<unsigned int> detIds;
      for (const auto& entry : corners_) {
        if (filter(entry)) {
          detIds.push_back(entry.first);
        }
      }
      return detIds;
    }

    void buildByLayer() {
      // Clear just in case they were already built
      barrel_lower_det_ids_.clear();
      endcap_lower_det_ids_.clear();

      for (unsigned int layer = 1; layer < 7; layer++) {
        barrel_lower_det_ids_.push_back(getDetIds([&layer](const auto& x) {
          Module m(x.first);
          return m.subdet() == 5 && m.layer() == layer && m.isLower() == 1;
        }));
      }
      for (unsigned int layer = 1; layer < 6; layer++) {
        endcap_lower_det_ids_.push_back(getDetIds([&layer](const auto& x) {
          Module m(x.first);
          return m.subdet() == 4 && m.layer() == layer && m.isLower() == 1;
        }));
      }
    }

    std::vector<unsigned int> const& getBarrelLayerDetIds(unsigned int layer) const {
      return barrel_lower_det_ids_[layer - 1];
    }

    std::vector<unsigned int> const& getEndcapLayerDetIds(unsigned int layer) const {
      return endcap_lower_det_ids_[layer - 1];
    }

    double getBarrelLayerAverageRadius(unsigned int layer) const { return avg_radii_[layer - 1]; }

    double getEndcapLayerAverageAbsZ(unsigned int layer) const { return avg_z_[layer - 1]; }

    double getMinR(unsigned int detId) const {
      auto const& corners = corners_.at(detId);
      double minR = std::numeric_limits<double>::max();
      for (int i = 0; i < corners.rows(); i++) {
        double x = corners(i, 1);
        double y = corners(i, 2);
        minR = std::min(minR, std::sqrt(x * x + y * y));
      }
      return minR;
    }

    double getMaxR(unsigned int detId) const {
      auto const& corners = corners_.at(detId);
      double maxR = std::numeric_limits<double>::min();
      for (int i = 0; i < corners.rows(); i++) {
        double x = corners(i, 1);
        double y = corners(i, 2);
        maxR = std::max(maxR, std::sqrt(x * x + y * y));
      }
      return maxR;
    }

    double getMinz(unsigned int detId) const {
      auto const& corners = corners_.at(detId);
      double minZ = std::numeric_limits<double>::max();
      for (int i = 0; i < corners.rows(); i++) {
        double z = corners(i, 0);
        minZ = std::min(minZ, z);
      }
      return minZ;
    }

    double getMaxz(unsigned int detId) const {
      auto const& corners = corners_.at(detId);
      double maxZ = std::numeric_limits<double>::min();
      for (int i = 0; i < corners.rows(); i++) {
        double z = corners(i, 0);
        maxZ = std::max(maxZ, z);
      }
      return maxZ;
    }

    double getMinPhi(unsigned int detId) const {
      auto const& corners = corners_.at(detId);
      double minPhi = std::numeric_limits<double>::max();
      double minPosPhi = std::numeric_limits<double>::max();
      double minNegPhi = std::numeric_limits<double>::max();
      unsigned int nPos = 0;
      unsigned int nOverPi2 = 0;
      for (int i = 0; i < corners.rows(); i++) {
        double phi = phi_mpi_pi(std::numbers::pi_v<double> + std::atan2(-corners(i, 2), -corners(i, 1)));
        minPhi = std::min(minPhi, phi);
        if (phi > 0) {
          minPosPhi = std::min(minPosPhi, phi);
          nPos++;
        } else {
          minNegPhi = std::min(minNegPhi, phi);
        }
        if (std::fabs(phi) > std::numbers::pi_v<double> / 2.) {
          nOverPi2++;
        }
      }
      if (nPos == 4 || nPos == 0)
        return minPhi;
      if (nOverPi2 == 4)
        return minPosPhi;
      return minPhi;
    }

    double getMaxPhi(unsigned int detId) const {
      auto const& corners = corners_.at(detId);
      double maxPhi = std::numeric_limits<double>::min();
      double maxPosPhi = std::numeric_limits<double>::min();
      double maxNegPhi = std::numeric_limits<double>::min();
      unsigned int nPos = 0;
      unsigned int nOverPi2 = 0;
      for (int i = 0; i < corners.rows(); i++) {
        double phi = phi_mpi_pi(std::numbers::pi_v<double> + std::atan2(-corners(i, 2), -corners(i, 1)));
        maxPhi = std::max(maxPhi, phi);
        if (phi > 0) {
          maxPosPhi = std::max(maxPosPhi, phi);
          nPos++;
        } else {
          maxNegPhi = std::max(maxNegPhi, phi);
        }
        if (std::fabs(phi) > std::numbers::pi_v<double> / 2.) {
          nOverPi2++;
        }
      }
      if (nPos == 4 || nPos == 0)
        return maxPhi;
      if (nOverPi2 == 4)
        return maxNegPhi;
      return maxPhi;
    }

    std::pair<double, double> getCompatibleEtaRange(unsigned int detId, double zmin_bound, double zmax_bound) const {
      double minr = getMinR(detId);
      double maxr = getMaxR(detId);
      double minz = getMinz(detId);
      double maxz = getMaxz(detId);
      double mineta = -std::log(std::tan(std::atan2(minz > 0 ? maxr : minr, minz - zmin_bound) / 2.));
      double maxeta = -std::log(std::tan(std::atan2(maxz > 0 ? minr : maxr, maxz - zmax_bound) / 2.));

      if (maxeta < mineta)
        std::swap(maxeta, mineta);
      return std::make_pair(mineta, maxeta);
    }

    std::pair<std::pair<double, double>, std::pair<double, double>> getCompatiblePhiRange(unsigned int detId,
                                                                                          double ptmin,
                                                                                          double ptmax) const {
      double minr = getMinR(detId);
      double maxr = getMaxR(detId);
      double minphi = getMinPhi(detId);
      double maxphi = getMaxPhi(detId);
      double A = k2Rinv1GeVf * kB / 2.;
      double pos_q_phi_lo_bound = phi_mpi_pi(A * minr / ptmax + minphi);
      double pos_q_phi_hi_bound = phi_mpi_pi(A * maxr / ptmin + maxphi);
      double neg_q_phi_lo_bound = phi_mpi_pi(-A * maxr / ptmin + minphi);
      double neg_q_phi_hi_bound = phi_mpi_pi(-A * minr / ptmax + maxphi);
      return std::make_pair(std::make_pair(pos_q_phi_lo_bound, pos_q_phi_hi_bound),
                            std::make_pair(neg_q_phi_lo_bound, neg_q_phi_hi_bound));
    }
  };
}  // namespace lstgeometry

#endif