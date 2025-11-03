#ifndef RecoTracker_LSTCore_interface_LSTGeometry_DetectorGeometry_h
#define RecoTracker_LSTCore_interface_LSTGeometry_DetectorGeometry_h

#include <algorithm>
#include <vector>
#include <functional>

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Module.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/LSTMath.h"

namespace lstgeometry {

  using LayerEtaBinPhiBinKey = std::tuple<unsigned int, unsigned int, unsigned int>;

  // We split modules into overlapping eta-phi bins so that it's easier to construct module maps
  // These values are just guesses and can be optimized later
  constexpr unsigned int kNEtaBins = 4;
  constexpr double kEtaBinRad = std::numbers::pi_v<double> / kNEtaBins;
  constexpr unsigned int kNPhiBins = 6;
  constexpr double kPhiBinWidth = 2 * std::numbers::pi_v<double> / kNPhiBins;

  bool isInEtaPhiBin(double eta, double phi, unsigned int eta_bin, unsigned int phi_bin) {
    double theta = 2. * std::atan(std::exp(-eta));

    if (eta_bin == 0) {
      if (theta > 3. * kEtaBinRad / 2.)
        return false;
    } else if (eta_bin == kNPhiBins - 1) {
      if (theta < (2 * (kNPhiBins - 1) - 1) * kEtaBinRad / 2.)
        return false;
    } else if (theta < (2 * eta_bin - 1) * kEtaBinRad / 2. || theta > (2 * (eta_bin + 1) + 1) * kEtaBinRad / 2.) {
      return false;
    }

    double pi = std::numbers::pi_v<double>;
    if (phi_bin == 0) {
      if (phi > -pi + kPhiBinWidth && phi < pi - kPhiBinWidth)
        return false;
    } else {
      if (phi < -pi + (phi_bin - 1) * kPhiBinWidth || phi > -pi + (phi_bin + 1) * kPhiBinWidth)
        return false;
    }

    return true;
  }

  std::pair<unsigned int, unsigned int> getEtaPhiBins(double eta, double phi) {
    double theta = 2. * std::atan(std::exp(-eta));

    unsigned int eta_bin = 0;
    if (theta <= kEtaBinRad) {
      eta_bin = 0;
    } else if (theta >= (kNEtaBins - 1) * kEtaBinRad) {
      eta_bin = kNEtaBins - 1;
    } else {
      for (unsigned int i = 1; i < kNEtaBins - 1; i++) {
        if (theta >= i * kEtaBinRad && theta <= (i + 1) * kEtaBinRad) {
          eta_bin = i;
          break;
        }
      }
    }

    unsigned int phi_bin = 0;
    double pi = std::numbers::pi_v<double>;

    if (phi <= -pi + kPhiBinWidth / 2. || phi >= pi - kPhiBinWidth / 2.) {
      phi_bin = 0;
    } else {
      for (unsigned int i = 1; i < kNPhiBins; i++) {
        if (phi >= -pi + ((2 * i - 1) * kPhiBinWidth) / 2. && phi <= -pi + ((2 * i + 1) * kPhiBinWidth) / 2.) {
          phi_bin = i;
          break;
        }
      }
    }

    return std::make_pair(eta_bin, phi_bin);
  }

  class DetectorGeometry {
  private:
    std::unordered_map<unsigned int, MatrixD4x3> corners_;
    std::vector<double> avg_radii_;
    std::vector<double> avg_z_;
    std::unordered_map<LayerEtaBinPhiBinKey, std::vector<unsigned int>, boost::hash<LayerEtaBinPhiBinKey>>
        barrel_lower_det_ids_;
    std::unordered_map<LayerEtaBinPhiBinKey, std::vector<unsigned int>, boost::hash<LayerEtaBinPhiBinKey>>
        endcap_lower_det_ids_;

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

      // Initialize all vectors
      for (unsigned int etabin = 0; etabin < kNEtaBins; etabin++) {
        for (unsigned int phibin = 0; phibin < kNPhiBins; phibin++) {
          for (unsigned int layer = 1; layer < 7; layer++) {
            barrel_lower_det_ids_[{layer, etabin, phibin}] = {};
          }
          for (unsigned int layer = 1; layer < 6; layer++) {
            endcap_lower_det_ids_[{layer, etabin, phibin}] = {};
          }
        }
      }

      for (unsigned int layer = 1; layer < 7; layer++) {
        auto detids = getDetIds([&layer](const auto& x) {
          Module m(x.first);
          return m.subdet() == 5 && m.layer() == layer && m.isLower() == 1;
        });
        for (auto detid : detids) {
          auto corners = getCorners(detid);
          RowVectorD3 center = corners.colwise().mean();
          center /= 4.;
          //double ref_phi = std::atan2(center(2), center(1));
          auto etaphi = getEtaPhi(center(1), center(2), center(0));
          for (unsigned int etabin = 0; etabin < kNEtaBins; etabin++) {
            for (unsigned int phibin = 0; phibin < kNPhiBins; phibin++) {
              if (isInEtaPhiBin(etaphi.first, etaphi.second, etabin, phibin)) {
                barrel_lower_det_ids_[{layer, etabin, phibin}].push_back(detid);
              }
            }
          }
        }
      }
      for (unsigned int layer = 1; layer < 6; layer++) {
        auto detids = getDetIds([&layer](const auto& x) {
          Module m(x.first);
          return m.subdet() == 4 && m.layer() == layer && m.isLower() == 1;
        });
        for (auto detid : detids) {
          auto corners = getCorners(detid);
          RowVectorD3 center = corners.colwise().mean();
          center /= 4.;
          //double ref_phi = std::atan2(center(2), center(1));
          auto etaphi = getEtaPhi(center(1), center(2), center(0));
          for (unsigned int etabin = 0; etabin < kNEtaBins; etabin++) {
            for (unsigned int phibin = 0; phibin < kNPhiBins; phibin++) {
              if (isInEtaPhiBin(etaphi.first, etaphi.second, etabin, phibin)) {
                endcap_lower_det_ids_[{layer, etabin, phibin}].push_back(detid);
              }
            }
          }
        }
      }
    }

    std::vector<unsigned int> const& getBarrelLayerDetIds(unsigned int layer,
                                                          unsigned int etabin,
                                                          unsigned int phibin) const {
      return barrel_lower_det_ids_.at({layer, etabin, phibin});
    }

    std::vector<unsigned int> const& getEndcapLayerDetIds(unsigned int layer,
                                                          unsigned int etabin,
                                                          unsigned int phibin) const {
      return endcap_lower_det_ids_.at({layer, etabin, phibin});
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

    double getMinZ(unsigned int detId) const {
      auto const& corners = corners_.at(detId);
      double minZ = std::numeric_limits<double>::max();
      for (int i = 0; i < corners.rows(); i++) {
        double z = corners(i, 0);
        minZ = std::min(minZ, z);
      }
      return minZ;
    }

    double getMaxZ(unsigned int detId) const {
      auto const& corners = corners_.at(detId);
      double maxZ = std::numeric_limits<double>::lowest();
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
      double maxPhi = std::numeric_limits<double>::lowest();
      double maxPosPhi = std::numeric_limits<double>::lowest();
      double maxNegPhi = std::numeric_limits<double>::lowest();
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
      double minz = getMinZ(detId);
      double maxz = getMaxZ(detId);
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